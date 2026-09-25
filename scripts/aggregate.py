#!/usr/bin/env python
"""Aggregate results/*/*.json into CSV summaries, LaTeX tables and PDF figures for the paper.

Usage: python scripts/aggregate.py --results results --out paper_assets
Every number that appears in the revised manuscript is produced by this script (mean +- std over
seeds; Welch t-test of FedDAC vs. the best baseline in each setting).
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

NAMES = {"feddac": "FedDAC", "fedavg": "FedAvg", "fedprox": "FedProx", "ppcfl": "PP-CFL",
         "fedse": "FedSe$^\\dagger$", "fedseq": "FedSeq$^\\dagger$", "rand_seq": "Random groups + Seq.",
         "dac_par": "DAC groups + Parallel", "dac_uniform": "FedDAC, uniform weights",
         "dac_samplew": "FedDAC, sample weights", "dac_noisyhist": "FedDAC, Laplace histograms ($\\epsilon_h{=}1$)"}
ORDER = ["feddac", "fedavg", "fedprox", "ppcfl", "fedse", "fedseq"]


def load(results):
    rows = []
    for f in glob.glob(os.path.join(results, "*", "*.json")):
        if "/analysis/" in f:
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        c = r["config"]
        exp = c["exp"]
        # CIFAR-10 non-private experiments use SGD (tag sgd0.02); earlier Adam runs are kept as a sensitivity study
        if c["dataset"] == "cifar10" and exp in ("main", "ablation", "noniid") and c.get("tag") != "sgd0.02":
            exp = exp + "_adam"
        # non-private CIFAR-10 attack runs: the SGD re-run (exp attack, tag sgd0.02, with class-matched MIA)
        # supersedes the Adam runs for both the LDI and the MIA table
        if c["dataset"] == "cifar10" and exp in ("attack", "mia") and not c.get("dp_eps") and c.get("tag") != "sgd0.02":
            exp = exp + "_adam"
        # DP-FedProx before the proximal term was added to the DP-SGD path (identical to DP-FedAvg) is superseded
        if exp in ("dp", "attack") and c["method"] == "fedprox" and c.get("dp_eps") and c.get("tag") != "proxfix":
            exp = exp + "_superseded"
        rows.append(dict(exp=exp, tag=c.get("tag", ""), lr=c.get("lr"), opt=c.get("optimizer", "adam"), dataset=c["dataset"], method=c["method"], N=c["clients"],
                         alpha=c["alpha"], ratios=tuple(c["ratios"]), seed=c["seed"],
                         dp=c.get("dp_eps"), acc=r["final_acc"], f1=r["final_f1"],
                         last5=r["last5_acc"], hist=r["history"], cost=r["cost"], L=r["L"],
                         G=r["num_groups"], cs=r["cluster_stats"], ldi=r.get("ldi"), mia=r.get("mia"),
                         dpr=r.get("dp"), hists=r.get("histograms"), types=r.get("client_types")))
        if exp == "attack" and c.get("tag") == "sgd0.02" and r.get("mia"):
            rows.append(dict(rows[-1], exp="mia"))
    return rows


def fit(tex):
    """Scale wide tables down to the text width (adjustbox 'max width' never enlarges)."""
    return tex.replace("\\begin{tabular}", "\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}").replace(
        "\\end{tabular}", "\\end{tabular}\n\\end{adjustbox}")


def ms(v):
    v = np.asarray(v, float)
    return (v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v))


def fmt(m, s, bold=False):
    t = f"{m:.2f}$\\pm${s:.2f}"
    return f"\\textbf{{{t}}}" if bold else t


def pval(a, b):
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def group(rows, keyf):
    g = defaultdict(list)
    for r in rows:
        g[keyf(r)].append(r)
    return g


def comparison_table(rows, dataset, exp, var, values, label, caption, methods=ORDER, fixed=None):
    sel = [r for r in rows if r["dataset"] == dataset and r["exp"] in exp and not r["dp"]]
    if fixed:
        sel = [r for r in sel if fixed(r)]
    head = {"N": "$N$", "alpha": "$\\alpha$", "ratios": "Hetero / Extreme"}[var]
    lines = ["\\begin{table}[htbp]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             "\\footnotesize", "\\begin{tabular}{llccc}", "\\toprule",
             f"{head} & Method & Accuracy (\\%) & Macro-F1 (\\%) & $p$ (vs. best other) \\\\", "\\midrule"]
    csv = []
    for v in values:
        cell = [r for r in sel if r[var] == v]
        if not cell:
            continue
        stats = {m: (ms([r["acc"] for r in cell if r["method"] == m]), ms([r["f1"] for r in cell if r["method"] == m]),
                     [r["acc"] for r in cell if r["method"] == m]) for m in methods}
        stats = {m: s for m, s in stats.items() if s[0][2] > 0}
        best = max(stats, key=lambda m: stats[m][0][0])
        vv = f"{int(v[1] * 100)}\\% / {int(v[2] * 100)}\\%" if var == "ratios" else f"{v:g}"
        first = True
        for m in methods:
            if m not in stats:
                continue
            (am, asd, n), (fm, fsd, _), accs = stats[m]
            others = [x for x in stats if x != m]
            bo = max(others, key=lambda x: stats[x][0][0]) if others else None
            p = pval(accs, stats[bo][2]) if (m == "feddac" and bo) else None
            ptxt = ("--" if p is not None and p != p else ("$<$0.001" if p is not None and p < 1e-3 else (f"{p:.3f}" if p is not None else "")))
            lines.append(f"{vv if first else ''} & {NAMES[m]} & {fmt(am, asd, m == best)} & {fmt(fm, fsd, m == best)} & {ptxt} \\\\")
            csv.append([dataset, str(v), m, am, asd, fm, fsd, n, p])
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\\\[2pt]{\\scriptsize Mean$\\pm$std over independent seeds; best mean in bold. "
              "$^\\dagger$Re-implemented from the published description. $p$: Welch $t$-test of FedDAC vs. the best other method.}",
              "\\end{table}"]
    return "\n".join(lines), csv


def dp_table(rows):
    sel = [r for r in rows if r["exp"] == "dp"]
    # non-private reference with the same optimiser as the DP runs (Adam)
    base = [r for r in rows if (r["exp"] == "main_adam" or (r["exp"] == "main" and r["dataset"] == "mnist"))
            and r["N"] == 20 and r["alpha"] == 0.2 and r["ratios"] == (0.1, 0.3, 0.6)]
    tuned = [r for r in rows if r["exp"] == "main" and r["N"] == 20 and r["alpha"] == 0.2 and r["ratios"] == (0.1, 0.3, 0.6)]
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Accuracy (\\%) under formal record-level $(\\varepsilon,\\delta)$-DP (DP-SGD, RDP accountant, "
             "$\\delta=10^{-5}$, $C=1$, lot size 64). $\\varepsilon$ is the total budget per client, including "
             "the label-histogram release for methods that use label statistics. Default setting ($N=20$, $\\alpha=0.2$, 10/30/60\\%). "
             "DP runs use Adam ($10^{-3}$); the last two columns give the non-private accuracy with the same optimiser and "
             "with the tuned optimiser of Table~\\ref{tab:hyperparams} (identical for MNIST).}",
             "\\label{tab:dp}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}", "\\begin{tabular}{llccccc}", "\\toprule",
             " & & & & & \\multicolumn{2}{c}{Non-private} \\\\ \\cmidrule(lr){6-7}",
             "Dataset & Method & $\\varepsilon=1$ & $\\varepsilon=4$ & $\\varepsilon=8$ & Adam & Tuned \\\\", "\\midrule"]
    for d in ("mnist", "cifar10"):
        first = True
        best = {}
        src = lambda eps: tuned if eps == "t" else (sel if eps else base)
        for eps in (1.0, 4.0, 8.0, None, "t"):
            vals = {m: [r["acc"] for r in src(eps) if r["dataset"] == d and r["method"] == m and (r["dp"] == eps if eps not in (None, "t") else True)] for m in ORDER}
            vals = {m: v for m, v in vals.items() if v}
            if vals:
                best[eps] = max(vals, key=lambda m: np.mean(vals[m]))
        for m in ORDER:
            cells = []
            for eps in (1.0, 4.0, 8.0, None, "t"):
                v = [r["acc"] for r in src(eps) if r["dataset"] == d and r["method"] == m and (r["dp"] == eps if eps not in (None, "t") else True)]
                cells.append(fmt(*ms(v)[:2], best.get(eps) == m) if v else "--")
            lines.append(f"{('MNIST' if d == 'mnist' else 'CIFAR-10') if first else ''} & {NAMES[m]} & " + " & ".join(cells) + " \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def scale_table(rows):
    sel = [r for r in rows if r["exp"] == "scale"]
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Scalability to larger label spaces (ResNet-18-GN). Final test accuracy and macro-F1 (\\%); SGD, learning rate 0.05, "
             "$\\alpha=0.2$, 10/30/60\\% homogeneous/heterogeneous/extreme clients.}", "\\label{tab:scale}", "\\footnotesize",
             "\\begin{tabular}{lllcc}", "\\toprule", "Dataset & $N$ & Method & Accuracy & Macro-F1 \\\\", "\\midrule"]
    for d, N in (("cifar100", 20), ("cifar100", 50), ("tinyimagenet", 50)):
        cell = [r for r in sel if r["dataset"] == d and r["N"] == N]
        if not cell:
            continue
        means = {m: np.mean([r["acc"] for r in cell if r["method"] == m]) for m in ORDER if any(r["method"] == m for r in cell)}
        best = max(means, key=means.get)
        first = True
        for m in ORDER:
            a = [r["acc"] for r in cell if r["method"] == m]
            f = [r["f1"] for r in cell if r["method"] == m]
            if not a:
                continue
            dn = {"cifar100": "CIFAR-100", "tinyimagenet": "Tiny-ImageNet"}[d]
            lines.append(f"{dn if first else ''} & {N if first else ''} & {NAMES[m]} & {fmt(*ms(a)[:2], m == best)} & {fmt(*ms(f)[:2], m == best)} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def ablation_table(rows):
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Component ablation (final accuracy, \\%). Rows isolate the grouping rule, the intra-group schedule and the inter-group weighting.}",
             "\\label{tab:ablation}", "\\footnotesize", "\\begin{tabular}{lcccccc}", "\\toprule",
             " & \\multicolumn{3}{c}{MNIST} & \\multicolumn{3}{c}{CIFAR-10} \\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
             "Variant & Default & $N{=}50$ & 90\\% extreme & Default & $N{=}50$ & 90\\% extreme \\\\", "\\midrule"]
    settings = [lambda r: r["N"] == 20 and r["ratios"] == (0.1, 0.3, 0.6),
                lambda r: r["N"] == 50 and r["ratios"] == (0.1, 0.3, 0.6),
                lambda r: r["N"] == 20 and r["ratios"] == (0.1, 0.0, 0.9)]
    for m in ["feddac", "rand_seq", "dac_par", "dac_uniform", "dac_samplew", "dac_noisyhist", "fedavg"]:
        cells = []
        for d in ("mnist", "cifar10"):
            for s in settings:
                v = [r["acc"] for r in rows if r["dataset"] == d and r["method"] == m and not r["dp"]
                     and r["alpha"] == 0.2 and r["exp"] in ("main", "ablation") and s(r)]
                cells.append(fmt(*ms(v)[:2]) if v else "--")
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def cost_table(rows):
    sel = [r for r in rows if r["exp"] == "main" and r["N"] in (20, 50, 70) and r["alpha"] == 0.2 and r["ratios"] == (0.1, 0.3, 0.6)]
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Computation and communication per training run (CIFAR-10). Transfers = model transmissions per run; "
             "latency = simulated critical-path compute time (parallel clients run concurrently; clients inside a sequential group wait for their predecessor).}",
             "\\label{tab:cost}", "\\footnotesize", "\\begin{tabular}{llccc}", "\\toprule",
             "$N$ & Method & Transfers & Comm. (MB) & Sim. latency (s) \\\\", "\\midrule"]
    for N in (20, 50, 70):
        first = True
        for m in ORDER:
            c = [r["cost"] for r in sel if r["dataset"] == "cifar10" and r["N"] == N and r["method"] == m]
            if not c:
                continue
            lines.append(f"{N if first else ''} & {NAMES[m]} & {int(np.mean([x['model_transfers'] for x in c]))} & "
                         f"{np.mean([x['comm_MB'] for x in c]):.0f} & {np.mean([x['simulated_latency_s'] for x in c]):.0f} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def attack_tables(rows):
    sel = [r for r in rows if r["exp"] == "attack"]
    out = {}
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Label-distribution inference (LDI) by an honest-but-curious server and by the next peer in a sequential chain. "
             "JS: Jensen--Shannon divergence (bits) between inferred and true client label distribution (lower = more leakage); "
             "prior: JS of the population-mean guess; Adv.: inference advantage (Eq.~\\ref{eq:advantage}); Gap: $\\mathrm{JS}(p_{\\text{unit}}\\|p_i)$, the error of an observer that recovers the released unit's label distribution exactly (Proposition~\\ref{prop:coverage}); Top-1: majority-class recovery rate. "
             "Peer: the model handed from the first client of a chain to the second, which reflects that client's update alone. Averaged over rounds $\\{1, R/2, R\\}$, clients and seeds.}",
             "\\label{tab:ldi}", "\\scriptsize", "\\begin{tabular}{lllccccc}", "\\toprule",
             "Dataset & Method & Observer & JS attack & JS prior & Adv. & Gap & Top-1 (\\%) \\\\", "\\midrule"]
    for d in ("mnist", "cifar10"):
        first = True
        for m in ORDER + ["rand_seq"]:
            for dp in (None, 4.0):
                recs = [x for r in sel if r["dataset"] == d and r["method"] == m and r["dp"] == dp and r["ldi"] for x in r["ldi"]]
                for view, sel_f in (("server", lambda x: x["view"] == "server"),
                                    ("peer (all)", lambda x: x["view"] == "peer"),
                                    ("peer", lambda x: x["view"] == "peer" and x["unit_size"] == 1)):
                    v = [x for x in recs if sel_f(x)]
                    if not v:
                        continue
                    name = NAMES[m] + (" (DP $\\varepsilon{=}4$)" if dp else "")
                    ja, jp = np.mean([x['js_attack'] for x in v]), np.mean([x['js_prior'] for x in v])
                    gap = np.mean([x['js_unit_true'] for x in v])
                    if view != "peer (all)":  # all hand-overs: kept in attack_summary.json only
                        lines.append(f"{('MNIST' if d == 'mnist' else 'CIFAR-10') if first else ''} & {name} & {view} & "
                                     f"{ja:.3f} & {jp:.3f} & {jp - ja:.3f} & {gap:.3f} & "
                                     f"{100 * np.mean([x['top1_hit'] for x in v]):.1f} \\\\")
                        first = False
                    out[f"{d}/{m}/{dp}/{view}"] = {"js_attack": float(np.mean([x['js_attack'] for x in v])),
                                                   "js_prior": float(np.mean([x['js_prior'] for x in v])),
                                                   "top1": float(np.mean([x['top1_hit'] for x in v])), "n": len(v),
                                                   "gap": float(gap), "adv": float(jp - ja)}
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    msel = [r for r in rows if r["exp"] == "mia"] or sel
    mia_lines = ["\\begin{table}[htbp]", "\\centering",
                 "\\caption{Loss-threshold membership inference against each client's training records, with non-members drawn from held-out data with the same label distribution as the members: AUC and TPR at 1\\% FPR, "
                 "on the final global model and on the last-round release unit (client model for parallel methods, cluster model for sequential ones).}",
                 "\\label{tab:mia}", "\\footnotesize", "\\begin{tabular}{llcccc}", "\\toprule",
                 " & & \\multicolumn{2}{c}{Global model} & \\multicolumn{2}{c}{Release unit} \\\\",
                 "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
                 "Dataset & Method & AUC & TPR@1\\% & AUC & TPR@1\\% \\\\", "\\midrule"]
    for d in ("mnist", "cifar10"):
        first = True
        for m in ORDER + ["rand_seq"]:
            for dp in (None, 4.0):
                rs = [r for r in msel if r["dataset"] == d and r["method"] == m and r["dp"] == dp and r["mia"]]
                if not rs:
                    continue
                g = [x for r in rs for x in r["mia"]["global"]]
                u = [x for r in rs for x in r["mia"]["unit"]]
                name = NAMES[m] + (" (DP $\\varepsilon{=}4$)" if dp else "")
                mia_lines.append(f"{('MNIST' if d == 'mnist' else 'CIFAR-10') if first else ''} & {name} & "
                                 f"{np.mean([x['auc'] for x in g]):.3f} & {100 * np.mean([x['tpr@1%fpr'] for x in g]):.1f} & "
                                 f"{np.mean([x['auc'] for x in u]):.3f} & {100 * np.mean([x['tpr@1%fpr'] for x in u]):.1f} \\\\")
                out[f"mia/{d}/{m}/{dp}"] = {"global_auc": float(np.mean([x['auc'] for x in g])),
                                            "unit_auc": float(np.mean([x['auc'] for x in u]))}
                first = False
        mia_lines.append("\\midrule")
    mia_lines[-1] = "\\bottomrule"
    mia_lines += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(lines), "\n".join(mia_lines), out


def optim_table(rows):
    """CIFAR-10 default setting: learning-rate selection with SGD and the original Adam configuration."""
    lrs = sorted({r["lr"] for r in rows if r["exp"] == "lrsel"})
    if not lrs:
        return ""
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Optimiser sensitivity on CIFAR-10 (default setting). Final accuracy (\\%) with plain SGD for several learning rates "
             "(2 seeds; used to select $\\eta=0.02$, the best value for every method) and with the Adam configuration of the original "
             "submission (lr $10^{-3}$, 5 seeds), under which the sequential methods drift towards the classes of the last client of each chain.}",
             "\\label{tab:optim}", "\\footnotesize", "\\begin{tabular}{l" + "c" * (len(lrs) + 1) + "}", "\\toprule",
             "Method & " + " & ".join(f"SGD {lr:g}" for lr in lrs) + " & Adam $10^{-3}$ \\\\", "\\midrule"]
    for m in ORDER:
        cells = []
        for lr in lrs:
            v = [r["acc"] for r in rows if r["exp"] == "lrsel" and r["method"] == m and r["lr"] == lr]
            cells.append(f"{np.mean(v):.1f}" if v else "--")
        v = [r["acc"] for r in rows if r["exp"] == "main_adam" and r["dataset"] == "cifar10" and r["method"] == m
             and r["N"] == 20 and r["alpha"] == 0.2 and r["ratios"] == (0.1, 0.3, 0.6)]
        cells.append(fmt(*ms(v)[:2]) if v else "--")
        lines.append(f"{NAMES[m]} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def coverage_table(results):
    f = os.path.join(results, "analysis", "cluster_analysis.json")
    if not os.path.exists(f):
        return ""
    cov = json.load(open(f))["coverage"]
    nm = {"dac": "DAC (FedDAC)", "random": "Random", "fedse": "Label-balanced (FedSe)", "fedseq": "FedSeq superclients"}
    lines = ["\\begin{table}[htbp]", "\\centering",
             "\\caption{Grouping quality over 100 simulated federations per setting (10/30/60\\% homogeneous/heterogeneous/extreme, "
             "$\\alpha=0.2$). Full cov.: fraction of groups containing every class; KL: divergence of the group label distribution "
             "from uniform (nats); mean over groups and federations. DAC, random and label-balanced grouping use the same number of groups $L$.}",
             "\\label{tab:coverage}", "\\footnotesize", "\\begin{tabular}{llcccc}", "\\toprule",
             "Classes, $N$ & Grouping & Groups & Full cov. & Mean cov. & KL \\\\", "\\midrule"]
    for key in ("C10_N20", "C10_N50", "C100_N20", "C100_N50", "C200_N20", "C200_N50"):
        c, n = key[1:].split("_N")
        first = True
        for m in ("dac", "random", "fedse", "fedseq"):
            v = cov[key][m]
            lines.append(f"{(c + ', ' + n) if first else ''} & {nm[m]} & {v['n_groups'][0]:.1f} & {v['full_cov_frac'][0]:.3f} & {v['mean_cov'][0]:.3f} & {v['kl_uniform'][0]:.3f} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def figures(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"feddac": "#1f77b4", "fedavg": "#7f7f7f", "fedprox": "#2ca02c", "ppcfl": "#9467bd",
           "fedse": "#ff7f0e", "fedseq": "#d62728"}
    # convergence curves, default setting
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, d in zip(axs, ("mnist", "cifar10")):
        for m in ORDER:
            H = [r["hist"] for r in rows if r["exp"] == "main" and r["dataset"] == d and r["method"] == m
                 and r["N"] == 20 and r["alpha"] == 0.2 and r["ratios"] == (0.1, 0.3, 0.6)]
            if not H:
                continue
            A = np.array([[h["acc"] for h in hh] for hh in H])
            x = [h["round"] for h in H[0]]
            ax.plot(x, A.mean(0), label=NAMES[m].replace("$^\\dagger$", "*"), color=col[m])
            ax.fill_between(x, A.mean(0) - A.std(0), A.mean(0) + A.std(0), color=col[m], alpha=0.15)
        ax.set_title({"mnist": "MNIST", "cifar10": "CIFAR-10"}[d]); ax.set_xlabel("Round"); ax.set_ylabel("Test accuracy (%)")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "convergence.pdf")); plt.close(fig)
    # client label distributions (default setting, seed 0)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, d in zip(axs, ("mnist", "cifar10")):
        H = [r for r in rows if r["exp"] == "main" and r["dataset"] == d and r["method"] == "feddac" and r["seed"] == 0
             and r["N"] == 20 and r["alpha"] == 0.2 and r["ratios"] == (0.1, 0.3, 0.6)]
        if not H:
            continue
        M = np.array(H[0]["hists"])
        im = ax.imshow(M, aspect="auto", cmap="Blues")
        ax.set_xlabel("Class"); ax.set_ylabel("Client")
        ax.set_title({"mnist": "MNIST", "cifar10": "CIFAR-10"}[d] + " ($N=20$, $\\alpha=0.2$, 10/30/60%)")
        ax.set_xticks(range(M.shape[1])); ax.set_yticks(range(M.shape[0]))
        ax.set_yticklabels([f"{i} ({t[0].upper()})" for i, t in enumerate(H[0]["types"])], fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(os.path.join(out, "data_heatmap.pdf")); plt.close(fig)
    # privacy-utility
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, d in zip(axs, ("mnist", "cifar10")):
        for m in ORDER:
            xs, ys, es = [], [], []
            for eps in (1.0, 4.0, 8.0):
                v = [r["acc"] for r in rows if r["exp"] == "dp" and r["dataset"] == d and r["method"] == m and r["dp"] == eps]
                if v:
                    xs.append(eps); ys.append(np.mean(v)); es.append(np.std(v))
            if xs:
                ax.errorbar(xs, ys, yerr=es, marker="o", label=NAMES[m].replace("$^\\dagger$", "*"), color=col[m], capsize=3)
        ax.set_xscale("log", base=2); ax.set_xlabel("$\\varepsilon$ (total, $\\delta=10^{-5}$)"); ax.set_ylabel("Test accuracy (%)")
        ax.set_title({"mnist": "MNIST", "cifar10": "CIFAR-10"}[d]); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "privacy_utility.pdf")); plt.close(fig)
    # non-IID characterisation heatmaps (CIFAR-10, all heterogeneous)
    sel = [r for r in rows if r["exp"] == "noniid"]
    if sel:
        alphas = sorted({r["alpha"] for r in sel}); Ns = sorted({r["N"] for r in sel})
        fig, axs = plt.subplots(1, len(ORDER), figsize=(3 * len(ORDER), 3))
        for ax, m in zip(axs, ORDER):
            M = np.full((len(Ns), len(alphas)), np.nan)
            for i, N in enumerate(Ns):
                for j, a in enumerate(alphas):
                    v = [r["acc"] for r in sel if r["method"] == m and r["N"] == N and r["alpha"] == a]
                    if v:
                        M[i, j] = np.mean(v)
            im = ax.imshow(M, vmin=np.nanmin([r["acc"] for r in sel]), vmax=np.nanmax([r["acc"] for r in sel]), cmap="viridis", origin="lower")
            for i in range(len(Ns)):
                for j in range(len(alphas)):
                    if not np.isnan(M[i, j]):
                        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, color="w")
            ax.set_xticks(range(len(alphas)), [f"{a:g}" for a in alphas]); ax.set_yticks(range(len(Ns)), Ns)
            ax.set_xlabel("$\\alpha$"); ax.set_title(NAMES[m].replace("$^\\dagger$", "*"))
        axs[0].set_ylabel("$N$")
        fig.tight_layout(); fig.savefig(os.path.join(out, "noniid_heatmaps.pdf")); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="paper_assets")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rows = load(a.results)
    print(len(rows), "runs loaded")
    allcsv = []
    specs = []
    for d, dn in (("mnist", "MNIST"), ("cifar10", "CIFAR-10")):
        specs += [
            (d, "N", [20, 30, 40, 50, 60, 70], f"tab:{d}_clients", f"{dn}: effect of the number of clients ($\\alpha=0.2$, 10/30/60\\% homogeneous/heterogeneous/extreme).",
             lambda r: r["alpha"] == 0.2 and r["ratios"] == (0.1, 0.3, 0.6)),
            (d, "alpha", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], f"tab:{d}_alpha", f"{dn}: effect of the Dirichlet concentration $\\alpha$ ($N=20$).",
             lambda r: r["N"] == 20 and r["ratios"] == (0.1, 0.3, 0.6)),
            (d, "ratios", [(0.1, h / 10, round(0.9 - h / 10, 1)) for h in range(10)], f"tab:{d}_ratio",
             f"{dn}: effect of client composition (10\\% homogeneous; heterogeneous/extreme shares vary; $N=20$, $\\alpha=0.2$).",
             lambda r: r["N"] == 20 and r["alpha"] == 0.2),
        ]
    for d, var, vals, label, cap, fx in specs:
        vals = [tuple(round(x, 1) for x in v) if isinstance(v, tuple) else v for v in vals]
        for r in rows:
            r["ratios"] = tuple(round(x, 1) for x in r["ratios"])
        tex, csv = comparison_table(rows, d, ("main",), var, vals, label, cap, fixed=fx)
        open(os.path.join(a.out, label.replace("tab:", "") + ".tex"), "w").write(fit(tex))
        allcsv += csv
    open(os.path.join(a.out, "dp.tex"), "w").write(fit(dp_table(rows)))
    open(os.path.join(a.out, "scale.tex"), "w").write(fit(scale_table(rows)))
    open(os.path.join(a.out, "ablation.tex"), "w").write(fit(ablation_table(rows)))
    open(os.path.join(a.out, "cost.tex"), "w").write(fit(cost_table(rows)))
    open(os.path.join(a.out, "coverage.tex"), "w").write(fit(coverage_table(a.results)))
    open(os.path.join(a.out, "optim.tex"), "w").write(fit(optim_table(rows)))
    ldi, mia, att = attack_tables(rows)
    open(os.path.join(a.out, "ldi.tex"), "w").write(fit(ldi))
    open(os.path.join(a.out, "mia.tex"), "w").write(fit(mia))
    json.dump(att, open(os.path.join(a.out, "attack_summary.json"), "w"), indent=1)
    with open(os.path.join(a.out, "summary.csv"), "w") as f:
        f.write("dataset,setting,method,acc_mean,acc_std,f1_mean,f1_std,n_seeds,p_vs_best_other\n")
        for c in allcsv:
            f.write(",".join(str(x) for x in c) + "\n")
    # DP accounting summary
    dpr = [r["dpr"] | {"dataset": r["dataset"], "method": r["method"]} for r in rows if r["dpr"]]
    json.dump(dpr[:50], open(os.path.join(a.out, "dp_accounting_sample.json"), "w"), indent=1)
    figures(rows, a.out)
    print("written to", a.out)


if __name__ == "__main__":
    main()
