#!/usr/bin/env python
"""Generate the full experiment grid of the revised manuscript as jobs.jsonl.

E1 main      MNIST / CIFAR-10: client-count, Dirichlet-alpha and client-composition sweeps
             (replaces Tables 2-7), 6 methods x 5 seeds, no DP.
E2 noniid    CIFAR-10, all clients Dirichlet-heterogeneous, alpha x N grid (Fig. 3), 2 seeds.
E3 ablation  component ablation (R2-C5) on three representative settings, 5 seeds.
E4 dp        formal record-level DP-SGD, eps in {1, 4, 8}, delta = 1e-5 (R2-C1, R2-C3), 5 seeds.
E5 scale     CIFAR-100 (N=20, 50) and Tiny-ImageNet (N=50) with ResNet-18-GN (R1-C3), 5 seeds.
E6 attack    label-distribution inference + membership inference (R1-C1, R2-C2), 3 seeds.
"""
import json
import sys

BASE = ["fedavg", "fedprox", "ppcfl", "fedse", "fedseq", "feddac"]
ABL = ["rand_seq", "dac_par", "dac_uniform", "dac_samplew", "dac_noisyhist"]
DEF = dict(clients=20, alpha=0.2, ratios=[0.1, 0.3, 0.6])
ROUNDS = {"mnist": 50, "cifar10": 100}
# CIFAR-10 non-private runs use plain SGD, lr 0.02: selected on the default setting (exp 'lrsel') as the
# best rate for every method; with Adam (lr 1e-3) the sequential methods drift towards the last client.
SGD_C10 = ["--optimizer", "sgd", "--lr", "0.02", "--tag", "sgd0.02"]


def job(exp, prio, **kw):
    j = dict(DEF)
    j.update(kw)
    j["exp"] = exp
    j["prio"] = prio
    return j


def grid():
    jobs = []
    # E6 attacks (small, run first so the privacy section can be written early)
    for d in ("mnist", "cifar10"):
        R = ROUNDS[d]
        for m in BASE + ["rand_seq"]:
            for s in range(3):
                jobs.append(job("attack", 0, dataset=d, method=m, seed=s, ldi_rounds=[1, R // 2, R], mia=True))
        for m in ("fedavg", "feddac", "fedse"):
            for s in range(3):
                jobs.append(job("attack", 0, dataset=d, method=m, seed=s, dp_eps=4.0,
                                ldi_rounds=[1, R // 2, R], mia=True))
    # E4 DP
    for d in ("mnist", "cifar10"):
        for eps in (1.0, 4.0, 8.0):
            for m in BASE:
                for s in range(5):
                    jobs.append(job("dp", 1, dataset=d, method=m, seed=s, dp_eps=eps, save_model=(s == 0)))
    # E1 main
    cfgs = []
    for N in (20, 30, 40, 50, 60, 70):
        cfgs.append(dict(clients=N))
    for a in (0.1, 0.3, 0.5, 0.7, 1.0):
        cfgs.append(dict(alpha=a))
    for h in range(10):
        if h == 3:
            continue  # = default composition
        cfgs.append(dict(ratios=[0.1, round(h / 10, 1), round(0.9 - h / 10, 1)]))
    for d in ("mnist", "cifar10"):
        for ci, c in enumerate(cfgs):
            for m in BASE:
                for s in range(5):
                    prio = 2 if ci == 0 else 4
                    j = job("main", prio, dataset=d, method=m, seed=s, save_model=(s == 0 and ci == 0), **c)
                    if d == "cifar10":
                        j["extra"] = list(SGD_C10)
                    jobs.append(j)
    # CIFAR-10 with the original optimiser (Adam, lr 1e-3): default setting only (sensitivity analysis)
    for m in BASE:
        for s in range(5):
            jobs.append(job("main", 2, dataset="cifar10", method=m, seed=s, save_model=(s == 0)))
    # E3 ablations
    for d in ("mnist", "cifar10"):
        for c in (dict(), dict(clients=50), dict(ratios=[0.1, 0.0, 0.9])):
            for m in ABL:
                for s in range(5):
                    j = job("ablation", 3, dataset=d, method=m, seed=s, **c)
                    if d == "cifar10":
                        j["extra"] = list(SGD_C10)
                    jobs.append(j)
    # E2 non-IID characterisation
    for a in (0.1, 0.5, 0.9):
        for N in (20, 40, 80):
            for m in BASE:
                for s in range(2):
                    jobs.append(job("noniid", 5, dataset="cifar10", method=m, seed=s, clients=N,
                                    alpha=a, ratios=[0.0, 1.0, 0.0], extra=list(SGD_C10)))
    jobs.sort(key=lambda j: j["prio"])
    return jobs


def scale_grid(extra=None):
    """E5 (run as a separate phase): CIFAR-100 (N=20, 50) and Tiny-ImageNet (N=50), ResNet-18-GN."""
    jobs = []
    for d, Ns in (("cifar100", (20, 50)), ("tinyimagenet", (50,))):
        for N in Ns:
            for m in BASE:
                for s in range(5):
                    j = job("scale", 2, dataset=d, method=m, seed=s, clients=N, save_model=(s == 0))
                    if extra:
                        j["extra"] = list(extra)
                    jobs.append(j)
    return jobs


def diag():
    """Short learning-rate / optimiser check for the ResNet-18-GN runs (not part of the paper)."""
    P = []
    for opt in (["--lr", "1e-3", "--tag", "adam1e-3"], ["--lr", "3e-4", "--tag", "adam3e-4"],
                ["--optimizer", "sgd", "--lr", "0.05", "--tag", "sgd0.05"]):
        for m in ("fedavg", "feddac"):
            j = job("diag", 0, dataset="cifar100", method=m, seed=0, rounds=10)
            j["extra"] = opt + ["--eval_every", "1"]
            P.append(j)
    return P


def mia_grid():
    """Membership inference with class-matched non-members (supersedes the MIA part of E6)."""
    jobs = []
    for d in ("mnist", "cifar10"):
        for m in BASE + ["rand_seq"]:
            for s in range(3):
                jobs.append(job("mia", 0, dataset=d, method=m, seed=s, mia=True))
        for m in ("fedavg", "feddac", "fedse"):
            for s in range(3):
                jobs.append(job("mia", 0, dataset=d, method=m, seed=s, dp_eps=4.0, mia=True))
    return jobs


def lrsel_grid():
    """CIFAR-10 learning-rate selection for plain SGD (per method), default setting, 2 seeds."""
    jobs = []
    for lr in (0.005, 0.01, 0.02, 0.05, 0.1):
        for m in BASE:
            for s in range(2):
                j = job("lrsel", 0, dataset="cifar10", method=m, seed=s)
                j["extra"] = ["--optimizer", "sgd", "--lr", str(lr), "--tag", f"sgd{lr}"]
                jobs.append(j)
    return jobs


def to_argv(j):
    a = ["--dataset", j["dataset"], "--method", j["method"], "--clients", str(j["clients"]),
         "--alpha", str(j["alpha"]), "--ratios", *map(str, j["ratios"]), "--seed", str(j["seed"]),
         "--exp", j["exp"]]
    if j.get("dp_eps"):
        a += ["--dp_eps", str(j["dp_eps"])]
    if j.get("ldi_rounds"):
        a += ["--ldi_rounds", *map(str, j["ldi_rounds"])]
    if j.get("mia"):
        a += ["--mia"]
    if j.get("save_model"):
        a += ["--save_model"]
    if j.get("rounds"):
        a += ["--rounds", str(j["rounds"])]
    a += list(j.get("extra", []))
    return a


def pilot():
    """Short timing / correctness pilot run on one pod before launching the full grid."""
    P = []
    for d, R in (("mnist", 3), ("cifar10", 3), ("cifar100", 2), ("tinyimagenet", 1)):
        for m in ("feddac", "fedavg", "ppcfl", "fedseq"):
            P.append(job("pilot", 0, dataset=d, method=m, seed=0, rounds=R))
    P.append(job("pilot", 0, dataset="mnist", method="feddac", seed=0, rounds=2, dp_eps=4.0))
    P.append(job("pilot", 0, dataset="cifar10", method="fedavg", seed=0, rounds=2, dp_eps=4.0))
    P.append(job("pilot", 0, dataset="mnist", method="feddac", seed=0, rounds=2, ldi_rounds=[1, 2], mia=True))
    P.append(job("pilot", 0, dataset="mnist", method="fedavg", seed=0, rounds=2, ldi_rounds=[1, 2], mia=True))
    return P


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ("pilot", "diag", "scale", "mia", "lrsel"):
        fn = {"pilot": pilot, "diag": diag, "mia": mia_grid, "lrsel": lrsel_grid,
              "scale": lambda: scale_grid(["--optimizer", "sgd", "--lr", "0.05"])}[sys.argv[1]]
        with open(f"configs/{sys.argv[1]}.jsonl", "w") as f:
            for j in fn():
                f.write(json.dumps(j) + "\n")
        sys.exit(0)
    jobs = grid()
    out = sys.argv[1] if len(sys.argv) > 1 else "configs/jobs.jsonl"
    with open(out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    from collections import Counter
    print(len(jobs), "jobs", Counter(j["exp"] for j in jobs))
