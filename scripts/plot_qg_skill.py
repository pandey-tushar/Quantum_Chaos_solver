#!/usr/bin/env python3
"""
plot_qg_skill.py - Readable atmospheric-style skill plot for QG hybrid.

Replaces the technical "log-RMSE in POD coefficient space" plot with two
panels using interpretable metrics:

    Top    : Skill score relative to physics-only as a function of lead
             time, mean +/- std across 8 seeds.
                 skill(t) = 1 - MSE_method(t) / MSE_physics(t)
             > 0 = correction helps; 0 = same as physics; < 0 = correction
             actively harms the forecast. The horizontal zero line is the
             do-nothing baseline. This is the standard atmospheric-
             community framing.

    Bottom : Number of seeds (out of 8) where each method beats physics-
             only at that lead time. A direct, statistic-free read of
             how often the correction helps.

Drops the catastrophically-diverging MLP curve from the top panel (it
saturates the y-axis); MLP is shown only in the bottom panel where it
fits naturally.

Reads results/qg_hybrid/baselines/qg_baselines_results.json and writes
results/qg_hybrid/baselines/skill_vs_leadtime.png .
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
    src = ROOT / "results/qg_hybrid/baselines/qg_baselines_results.json"
    with open(src) as f:
        d = json.load(f)
    per_seed = d["per_seed"]
    n_seeds = len(per_seed)
    methods = ["qrc", "esn", "rff", "linreg", "mlp", "persistence"]
    pretty = {
        "qrc":         ("QRC (5 qubits)",                "#0066cc", "-"),
        "esn":         ("Classical ESN (N=32)",          "#cc4400", "--"),
        "rff":         ("Random Fourier Features (N=32)", "#aa00aa", "--"),
        "linreg":      ("Linear regression",              "#bbbb00", "--"),
        "mlp":         ("Trainable MLP (~800 params)",   "#000000", "--"),
        "persistence": ("Persistence",                    "#666666", ":"),
    }

    # Stack arrays of MSE per (method, seed, lead)
    n_lead = min(len(r[m]["closed_loop_per_lead_mse"])
                  for r in per_seed for m in methods + ["physics"])
    leads = np.arange(1, n_lead + 1)
    arrs = {m: np.array([r[m]["closed_loop_per_lead_mse"][:n_lead] for r in per_seed])
             for m in methods + ["physics"]}

    eps = 1e-30
    skill = {m: 1.0 - arrs[m] / (arrs["physics"] + eps) for m in methods}
    # Per-seed wins vs physics-only at each lead
    wins = {m: np.sum(arrs[m] < arrs["physics"], axis=0) for m in methods}

    # ---- Plot ----
    dt_h = 3.15  # hours per snapshot in this dataset (constant from generation)
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8.5, 7.0),
                                            gridspec_kw={"height_ratios": [3, 1.2]},
                                            sharex=True)

    # Top: skill score (drop MLP because it explodes off-scale)
    plot_top = [m for m in methods if m != "mlp"]
    for m in plot_top:
        label, color, ls = pretty[m]
        mean = np.mean(skill[m], axis=0)
        std  = np.std(skill[m], axis=0)
        ax_top.plot(leads, mean, label=label, color=color, lw=2.0 if m == "qrc" else 1.5, ls=ls)
        ax_top.fill_between(leads, mean - std, mean + std, color=color, alpha=0.15)
    ax_top.axhline(0.0, color="#444", lw=1.0, ls="-", alpha=0.6)
    ax_top.text(leads[-1], 0.0, "  physics-only baseline",
                 fontsize=9, color="#444", va="center", ha="left")
    ax_top.set_ylabel("Skill score vs physics-only\n(higher = better;  0 = no improvement)")
    ax_top.set_ylim(-1.5, 1.0)
    ax_top.legend(loc="lower left", fontsize=9, frameon=True)
    ax_top.grid(True, alpha=0.3)
    ax_top.set_title(f"QG hybrid forecast skill ({n_seeds} seeds)\n"
                      "MLP omitted from top panel (catastrophically diverges); see bottom panel")

    # Bottom: per-lead-time win count vs physics, including MLP
    for m in methods:
        label, color, ls = pretty[m]
        ax_bot.plot(leads, wins[m], color=color, lw=1.7, ls=ls, marker="o", ms=3)
    ax_bot.axhline(n_seeds / 2, color="#888", lw=0.8, ls=":", alpha=0.6)
    ax_bot.text(leads[-1], n_seeds / 2, "  4/8 (chance)",
                 fontsize=8, color="#666", va="center", ha="left")
    ax_bot.set_ylim(-0.3, n_seeds + 0.3)
    ax_bot.set_yticks([0, 4, 8])
    ax_bot.set_xlabel(f"Lead time (steps; 1 step = {dt_h:.2f} h)")
    ax_bot.set_ylabel(f"Seeds where method\nbeats physics (of {n_seeds})")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    out = ROOT / "results/qg_hybrid/baselines/skill_vs_leadtime.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
