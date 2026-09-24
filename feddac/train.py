"""Local training / evaluation primitives shared by all algorithms."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_flat(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def set_flat(model, flat):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(flat[i:i + n].view_as(p))
        i += n


def state_clone(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def weighted_average(states, weights):
    w = torch.tensor(weights, dtype=torch.float64)
    w = (w / w.sum()).tolist()
    out = {}
    for k in states[0]:
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for s, wi in zip(states, w):
            acc += wi * s[k].float()
        out[k] = acc.to(states[0][k].dtype)
    return out


def num_params(model):
    return sum(p.numel() for p in model.parameters())


def local_train(model: nn.Module, ds, idx: torch.Tensor, cfg: dict, prox_ref=None, dp=None,
                amp: bool = False):
    """Train `model` in place on client indices `idx` (1-D LongTensor on device).

    cfg: lr, batch_size, local_epochs, mu (FedProx), optimizer ('adam'|'sgd').
    A fresh optimizer is created for every local update (stateless clients, identical for all
    methods).  Returns wall-clock seconds spent.
    """
    t0 = time.perf_counter()
    model.train()
    lr = cfg.get("lr", 1e-3)
    opt = (torch.optim.Adam if cfg.get("optimizer", "adam") == "adam" else torch.optim.SGD)(
        model.parameters(), lr=lr)
    mu = cfg.get("mu", 0.0)
    ref = [p.detach().clone() for p in prox_ref.parameters()] if (mu > 0 and prox_ref is not None) else None
    if dp is not None:
        _dp_train(model, ds, idx, opt, dp)
    else:
        B = cfg.get("batch_size", 10)
        n = idx.numel()
        for _ in range(cfg.get("local_epochs", 1)):
            perm = idx[torch.randperm(n, device=idx.device)]
            for s in range(0, n, B):
                b = perm[s:s + B]
                x = ds.prep(ds.x_train[b], train=True)
                y = ds.y_train[b]
                with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=amp):
                    loss = F.cross_entropy(model(x), y)
                if ref is not None:
                    loss = loss + 0.5 * mu * sum(((p - r) ** 2).sum() for p, r in zip(model.parameters(), ref))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
    if idx.is_cuda:
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def _dp_train(model, ds, idx, opt, dp):
    """DP-SGD with Poisson sampling and per-example clipping (torch.func)."""
    from torch.func import functional_call, grad, vmap

    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v for k, v in model.named_buffers()}

    def loss_one(p, b, x, y):
        out = functional_call(model, (p, b), (x.unsqueeze(0),))
        return F.cross_entropy(out, y.unsqueeze(0))

    per_ex = vmap(grad(loss_one), in_dims=(None, None, 0, 0))
    n = idx.numel()
    names = [k for k, _ in model.named_parameters()]
    plist = [p for _, p in model.named_parameters()]
    for _ in range(dp.steps_per_round):
        mask = torch.rand(n, device=idx.device) < dp.q
        b = idx[mask]
        summed = [torch.zeros_like(p) for p in plist]
        if b.numel() > 0:
            params = {k: v.detach() for k, v in model.named_parameters()}
            for s in range(0, b.numel(), 256):  # chunk to bound memory
                bb = b[s:s + 256]
                x = ds.prep(ds.x_train[bb], train=True)
                y = ds.y_train[bb]
                g = per_ex(params, buffers, x, y)
                norms = torch.sqrt(sum((g[k].reshape(g[k].shape[0], -1) ** 2).sum(1) for k in names))
                factor = (dp.clip / (norms + 1e-6)).clamp(max=1.0)
                for j, k in enumerate(names):
                    summed[j] += (g[k] * factor.view(-1, *([1] * (g[k].dim() - 1)))).sum(0)
        for j, p in enumerate(plist):
            noise = torch.normal(0.0, dp.sigma * dp.clip, size=p.shape, device=p.device)
            p.grad = (summed[j] + noise) / dp.batch_size
        opt.step()
        opt.zero_grad(set_to_none=True)


@torch.no_grad()
def evaluate(model, ds, batch_size=2000, return_preds=False):
    from sklearn.metrics import f1_score

    model.eval()
    preds, loss = [], 0.0
    for s in range(0, ds.y_test.numel(), batch_size):
        x = ds.prep(ds.x_test[s:s + batch_size])
        out = model(x).float()
        loss += F.cross_entropy(out, ds.y_test[s:s + batch_size], reduction="sum").item()
        preds.append(out.argmax(1))
    p = torch.cat(preds)
    y = ds.y_test
    acc = (p == y).float().mean().item() * 100
    f1 = f1_score(y.cpu().numpy(), p.cpu().numpy(), average="macro", zero_division=0) * 100
    res = {"acc": acc, "f1": f1, "loss": loss / y.numel()}
    if return_preds:
        res["preds"] = p.cpu().numpy()
    return res


@torch.no_grad()
def per_sample_loss(model, ds, x_uint8, y):
    model.eval()
    out = []
    for s in range(0, y.numel(), 2000):
        o = model(ds.prep(x_uint8[s:s + 2000])).float()
        out.append(F.cross_entropy(o, y[s:s + 2000], reduction="none"))
    return torch.cat(out)
