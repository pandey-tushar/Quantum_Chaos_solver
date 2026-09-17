"""Paired comparisons of two models over (data seed x reservoir seed) runs.

Runs that share a data seed are not independent, so every statistic is reported twice:
at run level and at cluster (data-seed) level, where each data seed contributes its mean
difference. Lower scores are better (NRMSE), so ``wins_a`` counts runs where a < b.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def paired_comparison(a, b, clusters, n_boot: int = 20000, seed: int = 0, tie_tol: float = 5e-4) -> dict:
    a, b, clusters = np.asarray(a, float), np.asarray(b, float), np.asarray(clusters)
    d = a - b
    labels = np.unique(clusters)
    dc = np.array([d[clusters == c].mean() for c in labels])
    rng = np.random.default_rng(seed)
    boot_run = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    # cluster bootstrap: resample data seeds, keep all their runs
    idx = rng.integers(0, len(labels), (n_boot, len(labels)))
    sums = np.array([d[clusters == c].sum() for c in labels])
    counts = np.array([np.sum(clusters == c) for c in labels])
    boot_cluster = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)

    def ttest(x):
        return float(stats.ttest_1samp(x, 0.0).pvalue) if len(x) > 1 and np.std(x) > 0 else float("nan")

    def wilcoxon(x):
        return float(stats.wilcoxon(x).pvalue) if np.any(x != 0) and len(x) > 1 else float("nan")

    return {"n_runs": int(len(d)), "n_clusters": int(len(labels)),
            "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
            "ci95_run": [float(np.percentile(boot_run, 2.5)), float(np.percentile(boot_run, 97.5))],
            "ci95_cluster": [float(np.percentile(boot_cluster, 2.5)), float(np.percentile(boot_cluster, 97.5))],
            "wins_a": int(np.sum(d < -tie_tol)), "wins_b": int(np.sum(d > tie_tol)),
            "ties": int(np.sum(np.abs(d) <= tie_tol)),
            "p_t_run": ttest(d), "p_wilcoxon_run": wilcoxon(d),
            "p_t_cluster": ttest(dc), "cluster_wins_a": int(np.sum(dc < 0)),
            "tie_tol": tie_tol, "n_boot": n_boot}


def holm(pvalues) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    p = np.asarray(pvalues, float)
    if p.size == 0:
        raise ValueError("no p-values")
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj, 1.0)
    return out.tolist()
