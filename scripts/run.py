#!/usr/bin/env python
"""Run a single federated experiment and write results/<exp>/<run_id>.json (+ optional model).

Example:
  python scripts/run.py --dataset mnist --method feddac --clients 20 --alpha 0.2 \
      --ratios 0.1 0.3 0.6 --seed 0 --exp smoke --rounds 5
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feddac import attacks as A  # noqa: E402
from feddac.algorithms import run_federated  # noqa: E402
from feddac.data import load_dataset, partition  # noqa: E402

DEFAULTS = {  # Table 1 of the manuscript (unchanged unless stated)
    "mnist": dict(rounds=50, batch_size=10, lr=1e-3),
    "cifar10": dict(rounds=100, batch_size=10, lr=1e-3),
    "cifar100": dict(rounds=100, batch_size=32, lr=1e-3, eval_every=5),
    "tinyimagenet": dict(rounds=60, batch_size=32, lr=1e-3, eval_every=5),
    "synthetic": dict(rounds=5, batch_size=10, lr=1e-3),
}


def run_id(cfg):
    keys = ["dataset", "method", "clients", "alpha", "ratios", "seed", "dp_eps", "rounds", "tag"]
    s = json.dumps({k: cfg.get(k) for k in keys}, sort_keys=True)
    h = hashlib.sha1(s.encode()).hexdigest()[:10]
    r = "-".join(f"{x:g}" for x in cfg["ratios"])
    dp = f"_dp{cfg['dp_eps']:g}" if cfg.get("dp_eps") else ""
    return f"{cfg['dataset']}_{cfg['method']}_N{cfg['clients']}_a{cfg['alpha']:g}_r{r}{dp}_s{cfg['seed']}_{h}"


def parse(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--method", default="feddac")
    p.add_argument("--clients", type=int, default=20)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--ratios", type=float, nargs=3, default=[0.1, 0.3, 0.6])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--local_epochs", type=int, default=1)
    p.add_argument("--optimizer", default="adam", choices=["adam", "sgd"])
    p.add_argument("--momentum", type=float, default=0.0)
    p.add_argument("--mu", type=float, default=0.01)
    p.add_argument("--eval_every", type=int)
    p.add_argument("--dp_eps", type=float, default=None, help="total record-level epsilon")
    p.add_argument("--dp_delta", type=float, default=1e-5)
    p.add_argument("--dp_clip", type=float, default=1.0)
    p.add_argument("--dp_batch_size", type=int, default=64)
    p.add_argument("--eps_hist", type=float, default=0.5)
    p.add_argument("--ldi_rounds", type=int, nargs="*", default=[])
    p.add_argument("--mia", action="store_true")
    p.add_argument("--save_model", action="store_true")
    p.add_argument("--exp", default="default")
    p.add_argument("--tag", default="")
    p.add_argument("--out", default=os.environ.get("FEDDAC_RESULTS", "results"))
    return p.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    cfg = dict(DEFAULTS.get(a.dataset, {}))
    for k, v in vars(a).items():
        if v is not None and k not in ("out",):
            cfg[k] = v
    cfg["ratios"] = list(cfg["ratios"])
    rid = run_id(cfg)
    out_dir = os.path.join(a.out, a.exp)
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, rid + ".json")
    if os.path.exists(out_json):
        print("exists, skipping", out_json)
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    ds = load_dataset(a.dataset, device)
    part = partition(ds.y_train.cpu().numpy(), a.clients, tuple(a.ratios), a.alpha, ds.num_classes,
                     seed=a.seed)
    t0 = time.time()
    res, model, caps = run_federated(ds, part, cfg, device, capture_rounds=set(a.ldi_rounds))
    rng = np.random.default_rng(a.seed + 7)
    if a.ldi_rounds:
        tcfg = {k: cfg[k] for k in ("lr", "batch_size", "local_epochs") if k in cfg}
        res["ldi"] = []
        for cap in caps:
            if "units" in cap:
                recs = A.ldi_attack(ds, cap, part, tcfg, device, rng)
                for r_ in recs:
                    r_["round"] = cap["round"]
                res["ldi"].extend(recs)
    if a.mia:
        res["mia"] = {"global": [], "unit": []}
        final_units = next((c["final_units"] for c in caps if "final_units" in c), [])
        for c in range(len(part.client_indices)):
            m = A.mia(model, ds, part.client_indices[c], device, rng)
            m.update(client=c, type=part.client_type[c])
            res["mia"]["global"].append(m)
        for u in final_units:
            member = np.concatenate([part.client_indices[c] for c in u["clients"]])
            for c in u["clients"]:
                m = A.mia(u["state"], ds, part.client_indices[c], device, rng)
                m.update(client=c, unit_size=len(u["clients"]), type=part.client_type[c])
                res["mia"]["unit"].append(m)
    res["run_id"] = rid
    res["total_time_s"] = time.time() - t0
    res["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    res["torch"] = torch.__version__
    if a.save_model:
        mdir = os.path.join(a.out, "models", a.exp)
        os.makedirs(mdir, exist_ok=True)
        sd = {k: (v.half() if ds.name in ("cifar100", "tinyimagenet") else v).cpu()
              for k, v in model.state_dict().items()}
        torch.save(sd, os.path.join(mdir, rid + ".pt"))
        res["model_file"] = f"models/{a.exp}/{rid}.pt"
    tmp = out_json + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f)
    os.replace(tmp, out_json)
    print(f"done {rid}: acc={res['final_acc']:.2f} f1={res['final_f1']:.2f} ({res['total_time_s']:.0f}s)")


if __name__ == "__main__":
    main()
