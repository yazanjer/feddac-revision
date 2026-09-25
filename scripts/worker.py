#!/usr/bin/env python
"""Pod-side job runner.

Assigns jobs of configs/jobs.jsonl to shards with a cost-balanced greedy rule (deterministic for a
given number of shards), runs this shard's jobs with `--procs` concurrent processes on the local
GPU, and periodically commits results/ and logs/ to the git branch results/pod-<shard>.
Already-finished runs (result JSON present) are skipped, so a pod can be restarted safely.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(__file__))
from make_grid import to_argv  # noqa: E402

COST = {"synthetic": 0.1, "mnist": 1.0, "cifar10": 2.5, "cifar100": 8.0, "tinyimagenet": 12.0}


def cost(j):
    c = COST[j["dataset"]]
    if j.get("dp_eps"):
        c *= 3
    if j.get("ldi_rounds"):
        c += 1.5
    return c


def shard_jobs(jobs, k, K):
    load = [0.0] * K
    mine = []
    for j in jobs:
        s = min(range(K), key=lambda i: load[i])
        load[s] += cost(j)
        if s == k:
            mine.append(j)
    return mine, load


def git_sync(branch, msg, lock):
    with lock:
        cmds = [["git", "add", "-A", "results", "logs"],
                ["git", "commit", "-q", "-m", msg],
                ["git", "push", "-q", "origin", f"HEAD:{branch}"]]
        for c in cmds:
            try:
                subprocess.run(c, capture_output=True, timeout=180)
            except subprocess.TimeoutExpired:
                print("git step timed out:", c[1], flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--num_shards", type=int, required=True)
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--jobs", default="configs/jobs.jsonl")
    p.add_argument("--sync_every", type=int, default=600)
    p.add_argument("--no_git", action="store_true")
    p.add_argument("--branch", default=None)
    a = p.parse_args()
    jobs = [json.loads(l) for l in open(a.jobs)]
    mine, load = shard_jobs(jobs, a.shard, a.num_shards)
    print(f"shard {a.shard}/{a.num_shards}: {len(mine)} jobs, est. cost {load[a.shard]:.0f}", flush=True)
    branch = a.branch or f"results/pod-{a.shard}"
    os.makedirs("logs", exist_ok=True)
    lock = threading.Lock()
    queue = list(mine)
    qlock = threading.Lock()
    done = [0]

    def worker(wid):
        while True:
            with qlock:
                if not queue:
                    return
                j = queue.pop(0)
            argv = [sys.executable, "scripts/run.py"] + to_argv(j)
            name = (f"{j['exp']}_{j['dataset']}_{j['method']}_N{j['clients']}_a{j['alpha']}_r"
                    + "-".join(str(x) for x in j["ratios"]) + f"_s{j['seed']}"
                    + (f"_dp{j['dp_eps']}" if j.get("dp_eps") else "") + ("_atk" if j.get("ldi_rounds") else ""))
            with open(f"logs/{name}.log", "w") as lf:
                t0 = time.time()
                env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"),
                           MKL_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
                r = subprocess.run(argv, stdout=lf, stderr=subprocess.STDOUT, env=env)
                lf.write(f"\nEXIT {r.returncode} in {time.time() - t0:.0f}s\n")
            with qlock:
                done[0] += 1
                print(f"[{done[0]}/{len(mine)}] rc={r.returncode} {name}", flush=True)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(a.procs)]
    for t in threads:
        t.start()
    last = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(15)
        if not a.no_git and time.time() - last > a.sync_every:
            git_sync(branch, f"pod {a.shard}: {done[0]}/{len(mine)} runs", lock)
            last = time.time()
    if not a.no_git:
        git_sync(branch, f"pod {a.shard}: finished {done[0]}/{len(mine)}", lock)
    open("SHARD_DONE", "w").write(str(done[0]))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
