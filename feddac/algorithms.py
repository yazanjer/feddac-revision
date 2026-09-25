"""Federated training loop for FedDAC, baselines and ablations.

Every method is expressed as   (grouping, intra-group schedule, inter-group weighting):

  method            grouping                  intra     inter-group weights   label stats used
  ----------------  ------------------------  --------  --------------------  ----------------
  fedavg            singletons                parallel  n_i (samples)         no
  fedprox           singletons (+prox term)   parallel  n_i                   no
  ppcfl             fuzzy c-means on RR       parallel  membership, mean      RR presence bits
  fedse             label-balanced (KL)       sequential n_g                  histograms
  fedseq            FedSeq greedy superclient sequential n_g                  histograms
  feddac            DAC                       sequential |C_l| (Eq. 12)       histograms
  -- ablations --
  rand_seq          random, L groups          sequential |C_l|                no
  dac_par           DAC                       parallel  |C_l|                 histograms
  dac_uniform       DAC                       sequential 1/L                  histograms
  dac_samplew       DAC                       sequential n_l                  histograms
  dac_noisyhist     DAC on Laplace(eps_h=1)   sequential |C_l|                noisy histograms
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch

from . import clustering as CL
from .dp import ClientDP
from .models import build_model
from .train import (evaluate, local_train, num_params, state_clone, weighted_average)

USES_HIST = {"feddac", "fedse", "fedseq", "ppcfl", "dac_par", "dac_uniform", "dac_samplew",
             "dac_noisyhist"}
SEQUENTIAL = {"feddac", "fedse", "fedseq", "rand_seq", "dac_uniform", "dac_samplew", "dac_noisyhist"}
METHODS = ["fedavg", "fedprox", "ppcfl", "fedse", "fedseq", "feddac", "rand_seq", "dac_par",
           "dac_uniform", "dac_samplew", "dac_noisyhist"]


def make_groups(method, hist_true, L, rng, cfg):
    """Returns (groups, hist_seen_by_server, extra_info)."""
    N = len(hist_true)
    info = {}
    hist = hist_true
    if cfg.get("dp_eps") and method in USES_HIST and method != "ppcfl":
        hist = CL.laplace_histograms(hist_true, cfg["eps_hist"], rng)
    if method == "dac_noisyhist":
        hist = CL.laplace_histograms(hist_true, cfg.get("noisyhist_eps", 1.0), rng)
    if method in ("fedavg", "fedprox"):
        groups = [[i] for i in range(N)]
    elif method in ("feddac", "dac_par", "dac_uniform", "dac_samplew", "dac_noisyhist"):
        groups = CL.dac_cluster(hist, L)
    elif method == "rand_seq":
        groups = CL.random_groups(N, L, rng)
    elif method == "fedse":
        groups = CL.label_balanced(hist, L)
    elif method == "fedseq":
        groups = CL.fedseq_superclients(hist, rng, k_max=cfg.get("fedseq_kmax", 11),
                                        min_samples=cfg.get("fedseq_min_samples", 800))
    elif method == "ppcfl":
        rr_eps = cfg["eps_hist"] if cfg.get("dp_eps") else cfg.get("ppcfl_rr_eps", 0.8)
        B = CL.randomized_response_presence(hist_true, rr_eps, rng)
        S = CL.jaccard_matrix(B)
        U = CL.fuzzy_cmeans(S, L, rng)
        info["membership"] = U
        groups = [np.where(U.argmax(1) == k)[0].tolist() for k in range(L)]
        groups = [g for g in groups if g]
    else:
        raise ValueError(method)
    info["cluster_stats"] = CL.cluster_stats(groups, hist_true)
    return groups, info


def run_federated(ds, part, cfg, device, log=print, capture_rounds=(), save_units=False):
    """Main loop.  Returns a result dict (JSON-serialisable) and the final global model."""
    method = cfg["method"]
    seed = cfg["seed"]
    rng = np.random.default_rng(seed + 1000)
    torch.manual_seed(seed)
    N = len(part.client_indices)
    L = CL.num_clusters(N)
    C = ds.num_classes
    hist = part.histograms
    n_i = hist.sum(1).astype(float)
    idx = [torch.as_tensor(ix, device=device) for ix in part.client_indices]

    groups, ginfo = make_groups(method, hist, L, rng, cfg)
    G = len(groups)
    group_sizes = [len(g) for g in groups]
    group_samples = [float(n_i[g].sum()) for g in groups]
    seq = method in SEQUENTIAL
    if method == "feddac" or method in ("rand_seq", "dac_par", "dac_noisyhist"):
        inter_w = group_sizes
    elif method == "dac_uniform":
        inter_w = [1.0] * G
    else:  # fedavg/fedprox/fedse/fedseq/dac_samplew
        inter_w = group_samples

    global_model = build_model(ds.name, C).to(device)
    P = num_params(global_model)
    amp = ds.name in ("cifar100", "tinyimagenet") and device.type == "cuda"
    train_cfg = {k: cfg[k] for k in ("lr", "batch_size", "local_epochs", "optimizer", "momentum") if k in cfg}
    if method == "fedprox":
        train_cfg["mu"] = cfg.get("mu", 0.01)

    # ---- differential privacy set-up (record-level DP-SGD) ----
    dps, dp_report = None, None
    if cfg.get("dp_eps"):
        eps_sgd = cfg["dp_eps"] - (cfg["eps_hist"] if method in USES_HIST else 0.0)
        dps = [ClientDP(int(n_i[i]), cfg["dp_batch_size"], cfg["rounds"], cfg.get("local_epochs", 1),
                        eps_sgd, cfg["dp_delta"], cfg.get("dp_clip", 1.0)) for i in range(N)]
        reps = [d.report() for d in dps]
        dp_report = {"eps_total_target": cfg["dp_eps"], "eps_sgd": eps_sgd,
                     "eps_hist": cfg["eps_hist"] if method in USES_HIST else 0.0,
                     "delta": cfg["dp_delta"], "clip": cfg.get("dp_clip", 1.0),
                     "sigma_min": min(r["sigma"] for r in reps), "sigma_max": max(r["sigma"] for r in reps),
                     "eps_sgd_spent_max": max(r["eps"] for r in reps)}

    history, captures = [], []
    comm_transfers = 0
    t_start = time.perf_counter()
    sim_latency = 0.0
    client_time_total = 0.0
    eval_every = cfg.get("eval_every", 1)
    for r in range(cfg["rounds"]):
        gstate = state_clone(global_model)
        group_states, group_times = [], []
        capture = r + 1 in capture_rounds
        cap = {"round": r + 1, "units": []} if capture else None
        if capture:
            cap["global_out"] = _out_params(global_model)
            cap["global_state"] = {k: v.cpu() for k, v in gstate.items()}
        want_full = cfg.get("mia") and r + 1 == cfg["rounds"]
        full_units = []
        for g in groups:
            if seq:
                model = copy.deepcopy(global_model)
                order = list(rng.permutation(g)) if cfg.get("seq_order", "random") == "random" else list(g)
                t_g = 0.0
                prefix = []
                for c in order:
                    t_g += local_train(model, ds, idx[c], train_cfg, dp=dps[c] if dps else None, amp=amp)
                    if capture:
                        prefix.append({"clients": [int(x) for x in prefix[-1]["clients"] + [int(c)]] if prefix else [int(c)],
                                       "out": _out_params(model)})
                group_states.append(state_clone(model))
                if want_full:
                    full_units.append({"clients": [int(x) for x in g], "state": {k: v.cpu() for k, v in group_states[-1].items()}})
                group_times.append(t_g)
                client_time_total += t_g
                comm_transfers += len(g) + 1  # server->c1, hand-overs, c_last->server
                if capture:
                    # server view: final cluster model; peer views: prefixes (seen by next client)
                    cap["units"].append({"view": "server", "clients": [int(x) for x in order],
                                         "out": prefix[-1]["out"]})
                    for p in prefix[:-1]:
                        cap["units"].append({"view": "peer", **p})
            else:
                states, ts, ws = [], [], []
                for c in g:
                    model = copy.deepcopy(global_model)
                    ts.append(local_train(model, ds, idx[c], train_cfg, prox_ref=global_model,
                                          dp=dps[c] if dps else None, amp=amp))
                    states.append(state_clone(model))
                    ws.append(n_i[c])
                    if want_full:
                        full_units.append({"clients": [int(c)], "state": {k: v.cpu() for k, v in states[-1].items()}})
                    if capture:
                        cap["units"].append({"view": "server", "clients": [int(c)], "out": _out_params(model)})
                client_time_total += sum(ts)
                comm_transfers += 2 * len(g)
                if method == "ppcfl":
                    group_states.extend(states)  # aggregated below with memberships
                else:
                    group_states.append(weighted_average(states, ws))
                group_times.append(max(ts))
        # ---- aggregation ----
        if method == "ppcfl":
            members = [c for g in groups for c in g]
            U = ginfo["membership"][members]
            cluster_models = []
            for k in range(U.shape[1]):
                if U[:, k].sum() > 0:
                    cluster_models.append(weighted_average(group_states, U[:, k].tolist()))
            new_state = weighted_average(cluster_models, [1.0] * len(cluster_models))
        else:
            new_state = weighted_average(group_states, inter_w)
        global_model.load_state_dict(new_state)
        if want_full:
            captures.append({"round": r + 1, "final_units": full_units})
        sim_latency += max(group_times)
        if capture:
            captures.append(cap)
        if (r + 1) % eval_every == 0 or r + 1 == cfg["rounds"]:
            ev = evaluate(global_model, ds)
            ev["round"] = r + 1
            history.append(ev)
            if (r + 1) % max(1, cfg["rounds"] // 10) == 0 or r == 0:
                log(f"[{method}] round {r + 1}/{cfg['rounds']} acc={ev['acc']:.2f} f1={ev['f1']:.2f}")
    wall = time.perf_counter() - t_start
    final = history[-1]
    last_k = [h for h in history if h["round"] > cfg["rounds"] - 5]
    res = {
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "final_acc": final["acc"], "final_f1": final["f1"], "final_loss": final["loss"],
        "last5_acc": float(np.mean([h["acc"] for h in last_k])),
        "last5_f1": float(np.mean([h["f1"] for h in last_k])),
        "best_acc": max(h["acc"] for h in history),
        "history": [{k: h[k] for k in ("round", "acc", "f1", "loss")} for h in history],
        "num_groups": G, "L": L, "group_sizes": group_sizes, "group_samples": group_samples,
        "cluster_stats": ginfo["cluster_stats"],
        "client_types": part.client_type, "histograms": part.histograms.tolist(),
        "reused_samples": part.reused_samples,
        "cost": {"wall_clock_s": wall, "client_compute_s": client_time_total,
                 "simulated_latency_s": sim_latency, "model_transfers": comm_transfers,
                 "params": P, "comm_MB": comm_transfers * P * 4 / 2 ** 20,
                 "sequential": seq},
        "dp": dp_report,
        "groups": groups,
    }
    return res, global_model, captures


def _out_params(model):
    o = model.out
    return torch.cat([o.weight.detach().reshape(-1), o.bias.detach()]).float().cpu()
