"""Empirical privacy evaluation (new in the revision; answers R1-C1, R2-C2).

Threat model (Section 4.1 of the revised paper): an honest-but-curious server that observes
every model it receives, and honest-but-curious peers that, in sequential methods, observe the
model handed over by their predecessor.  The adversary knows the architecture, the training
hyper-parameters and the global model of the round, and owns auxiliary data from the same
distribution (half of the test split, disjoint from the evaluation half).

1. Label-distribution inference (LDI) - property inference on the label histogram of a target
   client from an observed update.  A learning-based attacker (shadow updates -> histogram) in
   the spirit of Ganju et al. (2018) / Wainakh et al. (2022) uses output-layer update features.
   Reported: Jensen-Shannon divergence between the inferred and the true histogram of each target
   client (lower = more leakage) and the gain over the non-informative prior (the population mean
   histogram), plus top-1 (majority-class) recovery rate.
2. Membership inference (MIA) - loss-threshold attack (Yeom et al., 2018) on (a) the final global
   model and (b) the last-round release unit (client model for parallel methods, cluster model for
   sequential ones).  Reported: ROC-AUC and TPR at 1% FPR.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import build_model
from .train import local_train, per_sample_loss


def js_div(p, q, eps=1e-12):
    p = np.asarray(p, float) + eps
    q = np.asarray(q, float) + eps
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def _features(delta_out: torch.Tensor, C: int):
    W = delta_out[:-C].view(C, -1)
    b = delta_out[-C:]
    f = torch.cat([W.sum(1), W.norm(dim=1), b])
    return (f / (f.norm() + 1e-12)).numpy()


def _random_hist(rng, C, n):
    kind = rng.integers(4)
    if kind == 0:
        k = rng.integers(1, max(2, C // 2) + 1)
        p = np.zeros(C); cls = rng.choice(C, k, replace=False); p[cls] = rng.dirichlet(np.ones(k))
    elif kind == 1:
        p = rng.dirichlet(np.full(C, rng.choice([0.1, 0.3, 1.0])))
    elif kind == 2:
        p = np.ones(C) / C
    else:
        p = rng.dirichlet(np.full(C, 5.0))
    return rng.multinomial(n, p)


def ldi_attack(ds, cap, part, train_cfg, device, rng, n_shadow=240, max_chain=6, aux_frac=0.5):
    """Train a shadow meta-model at this round's global model and attack every captured unit."""
    C = ds.num_classes
    gstate = {k: v.to(device) for k, v in cap["global_state"].items()}
    gout = cap["global_out"]
    base = build_model(ds.name, C).to(device)
    base.load_state_dict(gstate)
    n_aux = int(ds.y_test.numel() * aux_frac)
    aux_y = ds.y_test[:n_aux].cpu().numpy()
    by_cls = [np.where(aux_y == c)[0] for c in range(C)]
    sizes = part.histograms.sum(1)

    # temporarily expose aux data through the train tensors of a light-weight view
    class _AuxView:
        pass
    view = _AuxView()
    view.x_train, view.y_train = ds.x_test[:n_aux], ds.y_test[:n_aux]
    view.prep, view.num_classes, view.name = ds.prep, C, ds.name

    X, Y = [], []
    for s in range(n_shadow):
        chain = 1 if s < n_shadow * 2 // 3 else int(rng.integers(2, max_chain + 1))
        model = copy.deepcopy(base)
        tot = np.zeros(C)
        for _ in range(chain):
            n = int(min(rng.choice(sizes), 1500))
            h = _random_hist(rng, C, n)
            ix = np.concatenate([rng.choice(by_cls[c], h[c], replace=True) for c in range(C) if h[c] > 0])
            local_train(model, view, torch.as_tensor(ix, device=device), train_cfg)
            tot += h
        d = torch.cat([model.out.weight.detach().reshape(-1), model.out.bias.detach()]).float().cpu() - gout
        X.append(_features(d, C)); Y.append(tot / tot.sum())
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    Y = torch.tensor(np.stack(Y), dtype=torch.float32)
    meta = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, C))
    opt = torch.optim.Adam(meta.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(600):
        opt.zero_grad()
        loss = F.kl_div(F.log_softmax(meta(X), 1), Y, reduction="batchmean")
        loss.backward(); opt.step()

    prior = part.histograms.sum(0) / part.histograms.sum()
    hist = part.histograms
    out = []
    with torch.no_grad():
        for u in cap["units"]:
            f = torch.tensor(_features(u["out"] - gout, C), dtype=torch.float32)[None]
            p_hat = F.softmax(meta(f), 1)[0].numpy()
            for c in u["clients"]:
                p_true = hist[c] / hist[c].sum()
                out.append({"view": u["view"], "unit_size": len(u["clients"]), "client": int(c),
                            "type": part.client_type[c],
                            "js_attack": js_div(p_hat, p_true), "js_prior": js_div(prior, p_true),
                            "top1_hit": float(p_hat.argmax() == p_true.argmax()),
                            "js_unit_true": js_div(hist[u["clients"]].sum(0), p_true)})
    return out


def mia(model_or_state, ds, member_idx: np.ndarray, device, rng, n=500, n_aux_frac=0.5):
    from sklearn.metrics import roc_auc_score, roc_curve

    if isinstance(model_or_state, dict):
        m = build_model(ds.name, ds.num_classes).to(device)
        m.load_state_dict({k: v.to(device) for k, v in model_or_state.items()})
    else:
        m = model_or_state
    k = min(n, len(member_idx))
    mem = torch.as_tensor(rng.choice(member_idx, k, replace=False), device=device)
    start = int(ds.y_test.numel() * n_aux_frac)
    non = torch.as_tensor(rng.choice(np.arange(start, ds.y_test.numel()), k, replace=False), device=device)
    lm = per_sample_loss(m, ds, ds.x_train[mem], ds.y_train[mem]).cpu().numpy()
    ln = per_sample_loss(m, ds, ds.x_test[non], ds.y_test[non]).cpu().numpy()
    y = np.r_[np.ones(k), np.zeros(k)]
    s = -np.r_[lm, ln]
    fpr, tpr, _ = roc_curve(y, s)
    return {"auc": float(roc_auc_score(y, s)), "tpr@1%fpr": float(np.interp(0.01, fpr, tpr)),
            "gap_loss": float(ln.mean() - lm.mean())}
