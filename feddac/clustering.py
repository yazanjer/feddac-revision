"""Client grouping strategies.

* dac_cluster        - Distribution Anonymization through Clustering (Algorithm 1 of the paper):
                       stage 1 classification, stage 2 greedy coverage maximisation for extremely
                       heterogeneous clients, stage 3 deviation minimisation for heterogeneous
                       clients, stage 4 round-robin placement of homogeneous clients.
* random_groups      - ablation: random, equal-size groups with the same number of groups L.
* label_balanced     - FedSe-style grouping (Qiao et al., ICASSP 2025): greedy assignment that
                       minimises the KL divergence between each group's label distribution and
                       the uniform distribution (our re-implementation from the published
                       description; no official code is available).
* fedseq_superclients- FedSeq grouping (Zaccone et al., ICPR 2022 / Silvi et al., IEEE Access 2024):
                       greedy construction of superclients maximising the dissimilarity to the
                       superclient centroid, with K_S,max clients and a minimum number of samples.
* fuzzy_cmeans       - used by PP-CFL (Luo et al., 2024) on the Jaccard similarity of randomised-
                       response label-presence vectors.
All functions take label histograms (N, C) and return a list of lists of client ids.
"""
from __future__ import annotations

import math

import numpy as np


def num_clusters(num_clients: int) -> int:
    """L = sqrt(N / 2) (Eq. 5), rounded, at least 2."""
    return max(2, int(round(math.sqrt(num_clients / 2))))


# --------------------------------------------------------------------------------------------
def classify_clients(hist: np.ndarray, zero_threshold: int | None = None):
    """Stage 1. Returns arrays of client ids for (extreme, hetero, homo)."""
    C = hist.shape[1]
    if zero_threshold is None:
        zero_threshold = C // 2  # = 5 for 10-class datasets (T_z in Algorithm 1)
    ext, het, homo = [], [], []
    for i, h in enumerate(hist):
        zeros = int((h == 0).sum())
        if zeros >= zero_threshold:
            ext.append(i)
        elif h.std() > 0.5 * h.mean():
            het.append(i)
        else:
            homo.append(i)
    return ext, het, homo


def _deviation(dist: np.ndarray) -> float:
    return float(np.abs(dist.max() - dist).sum())


def dac_cluster(hist: np.ndarray, L: int, zero_threshold: int | None = None, rng=None):
    hist = np.asarray(hist, dtype=np.float64)
    C = hist.shape[1]
    ext, het, homo = classify_clients(hist, zero_threshold)
    clusters = [[] for _ in range(L)]
    coverage = [set() for _ in range(L)]
    totals = np.zeros((L, C))

    # Stage 2: extreme clients sorted by heterogeneity (most zero classes first, then CV)
    def hetero_key(i):
        h = hist[i]
        return (-(h == 0).sum(), -(h.std() / (h.mean() + 1e-12)))

    for i in sorted(ext, key=hetero_key):
        cls = set(np.nonzero(hist[i])[0].tolist())
        best, best_gain = -1, -1
        for k in range(L):
            gain = len(cls - coverage[k])
            if gain > best_gain or (gain == best_gain and len(clusters[k]) < len(clusters[best])):
                best, best_gain = k, gain
        if best == -1 or best_gain == 0:
            best = int(np.argmin([len(c) for c in clusters]))
        clusters[best].append(i)
        coverage[best] |= cls
        totals[best] += hist[i]

    # Stage 3: heterogeneous clients -> cluster minimising the maximum-deviation criterion
    for i in het:
        devs = [_deviation(totals[k] + hist[i]) for k in range(L)]
        best = int(np.argmin(devs))
        clusters[best].append(i)
        coverage[best] |= set(np.nonzero(hist[i])[0].tolist())
        totals[best] += hist[i]

    # Stage 4: homogeneous clients -> smallest cluster (load balancing)
    for i in homo:
        best = int(np.argmin([len(c) for c in clusters]))
        clusters[best].append(i)
        coverage[best] |= set(np.nonzero(hist[i])[0].tolist())
        totals[best] += hist[i]

    return [c for c in clusters if c]


def random_groups(n: int, L: int, rng: np.random.Generator):
    perm = rng.permutation(n)
    return [perm[k::L].tolist() for k in range(L)]


def _kl_to_uniform(counts: np.ndarray) -> float:
    p = counts / max(counts.sum(), 1)
    C = len(p)
    nz = p > 0
    return float((p[nz] * np.log(p[nz] * C)).sum())


def label_balanced(hist: np.ndarray, L: int, rng=None):
    """FedSe-style label-balanced grouping (greedy KL-to-uniform minimisation)."""
    hist = np.asarray(hist, dtype=np.float64)
    order = np.argsort(-hist.sum(1), kind="stable")
    groups = [[] for _ in range(L)]
    totals = np.zeros((L, hist.shape[1]))
    cap = math.ceil(len(hist) / L)
    for i in order:
        best, best_v = None, np.inf
        for k in range(L):
            if len(groups[k]) >= cap:
                continue
            v = _kl_to_uniform(totals[k] + hist[i])
            if v < best_v:
                best, best_v = k, v
        groups[best].append(int(i))
        totals[best] += hist[i]
    return [g for g in groups if g]


def fedseq_superclients(hist: np.ndarray, rng: np.random.Generator, k_max: int = 11,
                        min_samples: int = 800):
    """FedSeq phi_greedy with KL-divergence on label distributions."""
    hist = np.asarray(hist, dtype=np.float64)
    P = (hist + 1e-6) / (hist + 1e-6).sum(1, keepdims=True)
    remaining = list(rng.permutation(len(hist)))
    groups = []
    while remaining:
        g = [remaining.pop(0)]
        n = hist[g].sum()
        while remaining and len(g) < k_max:
            centroid = P[g].mean(0)
            if n >= min_samples and _kl_to_uniform(hist[g].sum(0)) < 0.05:
                break
            kl = [float((P[j] * np.log(P[j] / centroid)).sum()) for j in remaining]
            j = remaining.pop(int(np.argmax(kl)))
            g.append(j)
            n += hist[j].sum()
        groups.append([int(x) for x in g])
    # merge a trailing group that violates the minimum-size constraint
    if len(groups) > 1 and hist[groups[-1]].sum() < min_samples:
        last = groups.pop()
        groups[-1].extend(last)
    return groups


# --------------------------------------------------------------------------------------------
def laplace_histograms(hist: np.ndarray, epsilon: float, rng: np.random.Generator):
    """Record-level epsilon-DP release of label histograms (L1 sensitivity 1).

    Used by the privacy-hardened variant: the server only ever sees noisy histograms."""
    if epsilon is None or epsilon <= 0 or np.isinf(epsilon):
        return np.asarray(hist, dtype=np.float64)
    noisy = hist + rng.laplace(0.0, 1.0 / epsilon, size=hist.shape)
    noisy = np.clip(np.round(noisy), 0, None)
    return noisy


def randomized_response_presence(hist: np.ndarray, epsilon: float, rng: np.random.Generator):
    """PP-CFL: per-client binary label-presence vector perturbed with randomized response."""
    b = (hist > 0).astype(int)
    p_flip = 1.0 / (math.exp(epsilon) + 1.0)
    flip = rng.random(b.shape) < p_flip
    return np.where(flip, 1 - b, b)


def jaccard_matrix(B: np.ndarray):
    inter = B @ B.T
    s = B.sum(1)
    union = s[:, None] + s[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        J = np.where(union > 0, inter / union, 1.0)
    return J


def fuzzy_cmeans(X: np.ndarray, c: int, rng: np.random.Generator, m: float = 2.0, iters: int = 300,
                 tol: float = 5e-3):
    """Plain fuzzy c-means (equivalent to skfuzzy.cmeans used in the PP-CFL reference code).

    Returns membership matrix U of shape (n, c)."""
    n = X.shape[0]
    U = rng.random((n, c))
    U /= U.sum(1, keepdims=True)
    for _ in range(iters):
        Um = U ** m
        centers = (Um.T @ X) / (Um.sum(0)[:, None] + 1e-12)
        d = np.linalg.norm(X[:, None, :] - centers[None], axis=2) + 1e-12
        inv = d ** (-2.0 / (m - 1))
        U_new = inv / inv.sum(1, keepdims=True)
        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new
    return U


# --------------------------------------------------------------------------------------------
def cluster_stats(groups, hist: np.ndarray):
    """Coverage and balance diagnostics for a grouping (reported in the paper)."""
    hist = np.asarray(hist, dtype=np.float64)
    C = hist.shape[1]
    out = []
    for g in groups:
        tot = hist[g].sum(0)
        p = tot / max(tot.sum(), 1)
        out.append({
            "size": len(g),
            "coverage": float((tot > 0).mean()),
            "kl_uniform": _kl_to_uniform(tot),
            "cv": float(tot.std() / (tot.mean() + 1e-12)),
            "min_class_frac": float(p.min() * C),
        })
    return out
