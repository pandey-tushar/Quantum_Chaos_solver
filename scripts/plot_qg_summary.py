#!/usr/bin/env python3
"""
plot_qg_summary.py - Lead-time vs RMSE plot for the QG hybrid 8-seed run.
Reads results/qg_hybrid/qg_hybrid_results.json and writes
results/qg_hybrid/rmse_vs_leadtime.png alongside it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent


def main():
    src = ROOT / "results" / "qg_hybrid" / "qg_hybrid_results.json"
    with open(src) as f:
        d = json.load(f)
    feature_dim = d["config"]["feature_dim"]
    dt_save_h = d["data"]["dt_save_s"] / 3600.0

    def stack(method):
        arrs = []
        for r in d["per_seed"]:
            if method not in r:
                continue
            arrs.append(np.array(r[method]["closed_loop_per_lead_mse"]))
        if not arrs:
            return None
        max_lead = max(len(a) for a in arrs)
        padded = np.array([np.pad(a, (0, max_lead - len(a)),
                                    constant_values=np.nan) for a in arrs])
        return padded

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    series = [
        ("Physics only (linearised QG)",          "physics", "#888"),
        ("Physics + QRC residual correction",     "qrc",     "#0066cc"),
        ("Physics + ESN correction (N=32)",       "esn",     "#cc4400"),
    ]
    for label, key, color in series:
        m = stack(key)
        if m is None:
            continue
        leads = np.arange(1, m.shape[1] + 1)
        mean = np.nanmean(m, axis=0)
        std  = np.nanstd(m, axis=0)
        rmse_mean = np.sqrt(mean)
        ax.plot(leads, rmse_mean, label=label, color=color, lw=2)
        ax.fill_between(leads,
                         np.sqrt(np.maximum(mean - std, 0.0)),
                         np.sqrt(mean + std),
                         color=color, alpha=0.2)

    ax.set_xlabel(f"Lead time (steps; 1 step = {dt_save_h:.2f} h)")
    ax.set_ylabel("RMSE in POD coefficient space")
    n_seeds = len(d["per_seed"])
    p_phys = d.get("wilcoxon_qrc_vs_phys_lead1", float("nan"))
    p_esn  = d.get("wilcoxon_qrc_vs_esn_lead1",  float("nan"))
    ax.set_title(f"QG hybrid forecast skill (5 qubits, 5 modes, {n_seeds} seeds)\n"
                  f"lead-1 Wilcoxon: QRC vs phys p={p_phys:.4f}, "
                  f"QRC vs ESN p={p_esn:.4f}")
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out = ROOT / "results" / "qg_hybrid" / "rmse_vs_leadtime.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
