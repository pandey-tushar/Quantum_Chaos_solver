#!/usr/bin/env python3
"""
plot_kq_scaling_summary.py - Single-figure summary of the K=q matched-complexity
scaling experiments for the SWE+QRC POC.

Shows how each paired-Wilcoxon p-value evolves as we scale problem complexity
(K = n_qubits = 5, 6, 7, 8) for both Block A (autonomous QRC) and Block B
(physics + QRC residual correction). Reads the per-config JSON files and writes
results/swe_qrc/kq_scaling_summary.png.

Deliberately does NOT plot absolute MSE numbers -- this figure is intended for
a teaser slide where the recipe stays unpublished. Only p-values + significance
thresholds are shown.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent


def _block_a_p(q: int, K: int) -> float:
    p = ROOT / f"results/swe_qrc/scaling_q{q}_K{K}/swe_qrc_results.json"
    with open(p) as f:
        d = json.load(f)
    return float(d["statistics"]["wilcoxon_qrc_vs_esn_teacher"]["p_value"])


def _block_b_pvalues(q: int, K: int):
    if q == 5 and K == 5:
        # The original Block B run lives at the un-suffixed path.
        p = ROOT / "results/swe_hybrid/swe_hybrid_results.json"
    else:
        p = ROOT / f"results/swe_hybrid/scaling_q{q}_K{K}/swe_hybrid_results.json"
    with open(p) as f:
        d = json.load(f)
    phys = np.array([r["physics"]["closed_loop_per_lead_mse"][0] for r in d["per_seed"]])
    qrc  = np.array([r["qrc"]["closed_loop_per_lead_mse"][0]     for r in d["per_seed"]])
    esn  = np.array([r["esn"]["closed_loop_per_lead_mse"][0]     for r in d["per_seed"]])
    phys_m = np.array([np.mean(r["physics"]["closed_loop_per_lead_mse"]) for r in d["per_seed"]])
    qrc_m  = np.array([np.mean(r["qrc"]["closed_loop_per_lead_mse"])     for r in d["per_seed"]])
    esn_m  = np.array([np.mean(r["esn"]["closed_loop_per_lead_mse"])     for r in d["per_seed"]])
    return {
        "lead1_qrc_vs_phys": float(stats.wilcoxon(qrc,   phys  ).pvalue),
        "lead1_qrc_vs_esn":  float(stats.wilcoxon(qrc,   esn   ).pvalue),
        "mean_qrc_vs_phys":  float(stats.wilcoxon(qrc_m, phys_m).pvalue),
        "mean_qrc_vs_esn":   float(stats.wilcoxon(qrc_m, esn_m ).pvalue),
    }


def main():
    qs = [5, 6, 7, 8]
    feat_dims = [2 ** q for q in qs]

    a_p = np.array([_block_a_p(q, q) for q in qs])

    b_p_lead1_phys = []
    b_p_lead1_esn  = []
    b_p_mean_phys  = []
    b_p_mean_esn   = []
    for q in qs:
        b = _block_b_pvalues(q, q)
        b_p_lead1_phys.append(b["lead1_qrc_vs_phys"])
        b_p_lead1_esn.append(b["lead1_qrc_vs_esn"])
        b_p_mean_phys.append(b["mean_qrc_vs_phys"])
        b_p_mean_esn.append(b["mean_qrc_vs_esn"])

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    x = np.arange(len(qs))

    # Significant region (p < 0.05) shaded in green; very-significant (p < 0.01)
    # shaded slightly darker. Drawn first so points sit on top.
    ax.axhspan(0.001, 0.01, color="#2ca02c", alpha=0.10, zorder=0)
    ax.axhspan(0.01,  0.05, color="#2ca02c", alpha=0.05, zorder=0)
    ax.axhline(0.05, color="#444", lw=1.0, ls=":", alpha=0.8, zorder=0)
    ax.axhline(0.01, color="#444", lw=1.0, ls=":", alpha=0.5, zorder=0)
    ax.text(len(qs) - 1.0 + 0.08, 0.0185, "significant  (p < 0.05)",
             fontsize=9, color="#225522", va="center", ha="left", style="italic")
    ax.text(len(qs) - 1.0 + 0.08, 0.0048, "highly significant  (p < 0.01)",
             fontsize=9, color="#114411", va="center", ha="left", style="italic")

    # Tiny x-jitter so overlapping points (multiple lines hitting p=0.008) remain visible
    series = [
        ("Block A: QRC vs ESN (teacher MSE)",          a_p,            "#1f77b4", "o", "-",  -0.08),
        ("Block B lead-1: phys+QRC vs physics-only",   b_p_lead1_phys, "#2ca02c", "s", "-",  -0.04),
        ("Block B lead-1: phys+QRC vs phys+ESN",       b_p_lead1_esn,  "#9467bd", "^", "-",  +0.00),
        ("Block B 91-step mean: phys+QRC vs physics",  b_p_mean_phys,  "#d62728", "D", "--", +0.04),
        ("Block B 91-step mean: phys+QRC vs phys+ESN", b_p_mean_esn,   "#ff7f0e", "v", "--", +0.08),
    ]
    for label, vals, color, marker, ls, dx in series:
        ax.plot(x + dx, vals, ls, marker=marker, color=color, lw=2.2, ms=10,
                 markeredgecolor="white", markeredgewidth=1.0, label=label, zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(0.005, 1.0)
    ax.set_xlim(-0.4, len(qs) - 0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"K = q = {q}\n(feat dim {2**q})" for q in qs])
    ax.set_xlabel("matched problem complexity   (POD modes K = qubit count q)")
    ax.set_ylabel("paired Wilcoxon p-value (8 seeds)\nlower = more significant")
    ax.set_title("Quantum advantage emerges with problem complexity\n"
                  "p-values for QRC against classical baselines, K=q scaling")

    ax.legend(loc="upper right", fontsize=9, frameon=True, ncol=1)
    ax.grid(True, which="major", axis="y", alpha=0.3, zorder=0)
    ax.grid(True, which="minor", axis="y", alpha=0.12, zorder=0)

    fig.tight_layout()
    out = ROOT / "results" / "swe_qrc" / "kq_scaling_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")

    # Also dump the underlying data so the user can drop the numbers into a slide
    summary = {
        "qs": qs,
        "feature_dims": feat_dims,
        "block_a_qrc_vs_esn": a_p.tolist(),
        "block_b_lead1_qrc_vs_phys":  b_p_lead1_phys,
        "block_b_lead1_qrc_vs_esn":   b_p_lead1_esn,
        "block_b_mean_qrc_vs_phys":   b_p_mean_phys,
        "block_b_mean_qrc_vs_esn":    b_p_mean_esn,
    }
    out_json = ROOT / "results" / "swe_qrc" / "kq_scaling_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_json}")


if __name__ == "__main__":
    main()
