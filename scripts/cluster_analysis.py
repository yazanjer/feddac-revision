#!/usr/bin/env python
"""Training-free analyses of the grouping step (CPU only, seconds to minutes).

(1) coverage / balance of DAC vs. random vs. label-balanced (FedSe) vs. FedSeq groups for
    10, 100 and 200 classes  -> answers R2-C2 (coverage != balance) and R1-C3 (scalability of DAC).
(2) cluster-size side channel (R1-C2): how much an observer who learns the aggregation weights
    |C_l|/N learns about (a) a client's heterogeneity type given the size of its own cluster and
    (b) the federation composition (number of extremely heterogeneous clients) given the sorted
    size vector.  Plug-in mutual-information estimates over many simulated federations, compared
    with the entropy of the target.
Writes results/analysis/cluster_analysis.json
"""
import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feddac import clustering as CL  # noqa: E402
from feddac.data import partition  # noqa: E402


def mi_discrete(x, y):
    n = len(x)
    cx, cy, cxy = Counter(x), Counter(y), Counter(zip(x, y))
    mi = sum(c / n * np.log2((c / n) / ((cx[a] / n) * (cy[b] / n))) for (a, b), c in cxy.items())
    hy = -sum(c / n * np.log2(c / n) for c in cy.values())
    return float(mi), float(hy)


def fake_labels(C, per_class):
    return np.repeat(np.arange(C), per_class)


def coverage_study(n_fed=100):
    out = {}
    for C, per_class in ((10, 5000), (100, 500), (200, 500)):
        labels = fake_labels(C, per_class)
        for N in (20, 50):
            rec = {m: [] for m in ("dac", "random", "fedse", "fedseq")}
            for s in range(n_fed):
                part = partition(labels, N, (0.1, 0.3, 0.6), 0.2, C, seed=s)
                L = CL.num_clusters(N)
                rng = np.random.default_rng(s)
                G = {"dac": CL.dac_cluster(part.histograms, L), "random": CL.random_groups(N, L, rng),
                     "fedse": CL.label_balanced(part.histograms, L),
                     "fedseq": CL.fedseq_superclients(part.histograms, rng)}
                for m, g in G.items():
                    st = CL.cluster_stats(g, part.histograms)
                    rec[m].append({"full_cov_frac": float(np.mean([x["coverage"] == 1.0 for x in st])),
                                   "mean_cov": float(np.mean([x["coverage"] for x in st])),
                                   "kl_uniform": float(np.mean([x["kl_uniform"] for x in st])),
                                   "cv": float(np.mean([x["cv"] for x in st])),
                                   "n_groups": len(g)})
            out[f"C{C}_N{N}"] = {m: {k: [float(np.mean([r[k] for r in v])), float(np.std([r[k] for r in v]))]
                                     for k in v[0]} for m, v in rec.items()}
            print(C, N, {m: round(out[f'C{C}_N{N}'][m]['full_cov_frac'][0], 3) for m in rec})
    return out


def size_channel(n_fed=2000):
    labels = fake_labels(10, 5000)
    own_size, own_type, size_vec, n_ext = [], [], [], []
    rng = np.random.default_rng(0)
    for s in range(n_fed):
        N = int(rng.choice([20, 30, 40, 50]))
        e = float(rng.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
        part = partition(labels, N, (0.1, round(0.9 - e, 1), e), 0.2, 10, seed=s)
        g = CL.dac_cluster(part.histograms, CL.num_clusters(N))
        sizes = [len(x) for x in g]
        ext, _, _ = CL.classify_clients(part.histograms)
        size_vec.append((N, tuple(sorted(sizes))))
        n_ext.append((N, len(ext)))
        for grp in g:
            for c in grp:
                own_size.append((N, len(grp)))
                own_type.append((N, "extreme" if c in ext else "other"))
    mi1, h1 = mi_discrete(own_size, own_type)
    mi_base, _ = mi_discrete([x[0] for x in own_size], own_type)  # N alone
    mi2, h2 = mi_discrete(size_vec, n_ext)
    mi2_base, _ = mi_discrete([x[0] for x in size_vec], n_ext)
    # Bayes accuracy of predicting a client's type from (N, own cluster size)
    table = Counter(zip(own_size, own_type))
    best = {}
    for (sz, t), c in table.items():
        if c > best.get(sz, (None, 0))[1]:
            best[sz] = (t, c)
    acc = sum(c for _, c in best.values()) / len(own_type)
    prior_tab = Counter(zip([x[0] for x in own_size], own_type))
    bestp = {}
    for (n, t), c in prior_tab.items():
        if c > bestp.get(n, (None, 0))[1]:
            bestp[n] = (t, c)
    acc_prior = sum(c for _, c in bestp.values()) / len(own_type)
    res = {"client_type_given_own_cluster_size": {"mi_bits": mi1 - mi_base, "H_target_given_N_bits": h1 - mi_base,
                                                  "bayes_acc": acc, "prior_acc": acc_prior},
           "composition_given_size_vector": {"mi_bits": mi2 - mi2_base, "H_target_given_N_bits": h2 - mi2_base},
           "n_federations": n_fed}
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    os.makedirs("results/analysis", exist_ok=True)
    res = {"coverage": coverage_study(), "size_channel": size_channel()}
    json.dump(res, open("results/analysis/cluster_analysis.json", "w"), indent=1)
