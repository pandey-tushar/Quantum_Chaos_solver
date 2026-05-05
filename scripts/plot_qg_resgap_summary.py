#!/usr/bin/env python3
"""
plot_qg_resgap_summary.py - Two-panel readable summary of the QG hybrid
resolution-gap K=q scaling experiment with the *fixed* (NeuralGCM-style
additive-correction) rollout.

Top panel    -- LEAD-1 mean MSE per (q, K). Log scale on y because
                 MLP and persistence are 1-3 orders of magnitude above
                 the rest. Plotted with markers per (method, K=q).

Bottom panel -- ROLLOUT-MEAN mean MSE per (q, K). Same scale convention.
                 This is the metric that matters for atmospheric closed-
                 loop forecasting and is where QRC's stability advantage
                 becomes visible: among learned correctors only QRC
                 stays statistically tied with physics-only.

Reads results/qg_resgap/scaling_v2/scaling_summary.json and per-cell
JSONs; writes results/qg_resgap/scaling_v2/summary_lead1_and_rollout.png.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "results" / "qg_resgap" / "scaling_v2"


def main():
    qs = [5, 6, 7, 8]
    methods = ["physics", "qrc", "rff", "esn", "linreg", "mlp", "persistence"]
    pretty = {
        "physics":     ("Physics only",                "#444",     "o",  "-"),
        "qrc":         ("QRC (q qubits)",              "#0066cc",  "s",  "-"),
        "rff":         ("RFF reservoir (N=2^q)",       "#aa00aa",  "^",  "--"),
        "esn":         ("Classical ESN (N=2^q)",       "#cc4400",  "v",  "--"),
        "linreg":      ("Linear regression",           "#bbbb00",  "D",  "--"),
        "mlp":         ("Trainable MLP (~params)",     "#000000",  "P",  ":"),
        "persistence": ("Persistence",                  "#888",    "X",  ":"),
    }

    # Aggregate summary
    with open(SRC_DIR / "scaling_summary.json") as f:
        summary = json.load(f)

    lead1 = {m: [summary[str(q)]["lead1_mean"][m] for q in qs] for m in methods}
    rollm = {m: [summary[str(q)]["rollout_mean_mean"][m] for q in qs] for m in methods}

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8.5, 7.5),
                                            gridspec_kw={"height_ratios": [1, 1]})

    x = np.arange(len(qs))
    for m in methods:
        label, color, marker, ls = pretty[m]
        ax_top.plot(x, lead1[m], color=color, marker=marker, ms=8, lw=1.8, ls=ls,
                     label=label, markeredgecolor="white", markeredgewidth=0.7)
        ax_bot.plot(x, rollm[m], color=color, marker=marker, ms=8, lw=1.8, ls=ls,
                     markeredgecolor="white", markeredgewidth=0.7)
    for ax in (ax_top, ax_bot):
        ax.set_xticks(x)
        ax.set_xticklabels([f"K = q = {q}\n(feat dim {2**q})" for q in qs])
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.3)
    ax_top.set_ylabel("Lead-1 mean MSE\n(lower is better)")
    ax_top.set_title("QG hybrid forecast on resolution-gap data, 8 seeds, fixed rollout")
    ax_top.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                   fontsize=9, frameon=True)

    ax_bot.set_ylabel("Rollout-mean MSE (28 steps, ~3.5 days)\n(lower is better)")
    ax_bot.set_xlabel("Matched problem complexity   (POD modes K = qubit count q)")
    fig.tight_layout()

    out = SRC_DIR / "summary_lead1_and_rollout.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")

    # Compact textual summary
    print()
    print("LEAD-1 MEAN MSE (best at each K=q in bold via underscore):")
    print(f"{'K=q':>5} | " + " | ".join(f"{m:>11}" for m in methods))
    print("-" * 100)
    for i, q in enumerate(qs):
        vals = {m: lead1[m][i] for m in methods}
        best = min(vals, key=lambda m: vals[m])
        line = f"{q:>5} | " + " | ".join(
            (f"_{vals[m]:>9.3e}_" if m == best else f" {vals[m]:>9.3e} ")
            for m in methods
        )
        print(line)

    print()
    print("ROLLOUT-MEAN MSE (best per row underscored):")
    print(f"{'K=q':>5} | " + " | ".join(f"{m:>11}" for m in methods))
    print("-" * 100)
    for i, q in enumerate(qs):
        vals = {m: rollm[m][i] for m in methods}
        best = min(vals, key=lambda m: vals[m])
        line = f"{q:>5} | " + " | ".join(
            (f"_{vals[m]:>9.3e}_" if m == best else f" {vals[m]:>9.3e} ")
            for m in methods
        )
        print(line)


if __name__ == "__main__":
    main()
