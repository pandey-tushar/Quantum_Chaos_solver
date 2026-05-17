#!/usr/bin/env python3
"""
aggregate_paper6_targetseeds.py - Combine paper-6 runs across multiple
target-Hamiltonian seeds.  Reports the QRC long-horizon edge with error
bars taken across both reservoir seeds AND target seeds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD_COLORS = {
    "QRC_state_injection":    "#0F4C5C",
    "Classical_shadows_RFF":  "#B8864A",
    "Classical_shadows_ESN":  "#9A4836",
    "Cheating_classical_RFF": "#5E7C6B",
    "Cheating_classical_ESN": "#1E7B8E",
}
METHOD_MARKERS = {
    "QRC_state_injection":    "o",
    "Classical_shadows_RFF":  "s",
    "Classical_shadows_ESN":  "^",
    "Cheating_classical_RFF": "D",
    "Cheating_classical_ESN": "v",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True,
                     help="paper-6 result dirs (one per target seed)")
    ap.add_argument("--out", default="results/paper6_longhz_aggregate.png")
    args = ap.parse_args()

    runs = []
    for d in args.dirs:
        with open(Path(d) / "summary.json") as f:
            s = json.load(f)
        runs.append(s)
    horizons = runs[0]["args"]["horizons"]
    methods = list(runs[0]["per_method_means"].keys())

    # Aggregate: for each (method, horizon), collect mean values across
    # target seeds AND each run's reservoir seeds (using k{h}_mean already
    # averaged over reservoir seeds in the summary).
    print(f"\n=== Aggregate over {len(runs)} target seeds "
            f"({len(runs) * runs[0]['args']['n_seeds']} reservoir runs) ===")
    print(f"Target seeds: {[r['args']['target_seed'] for r in runs]}")
    print()
    print(f"{'method':<26}  " + "  ".join([f"k={h:<5}" for h in horizons]))
    agg = {m: {h: [] for h in horizons} for m in methods}
    for s in runs:
        for m in methods:
            for h in horizons:
                agg[m][h].append(s["per_method_means"][m][f"k{h}_mean"])
    for m in methods:
        means = [np.mean(agg[m][h]) for h in horizons]
        stds  = [np.std(agg[m][h])  for h in horizons]
        row = f"{m:<26}  " + "  ".join(
            [f"{mu:.3f}+/-{sd:.3f}" for mu, sd in zip(means, stds)])
        print(row)

    # Identify the "best classical" per horizon and report the QRC-vs-best
    # gap (negative = QRC wins)
    print(f"\n=== QRC vs best classical (negative = QRC wins) ===")
    print(f"{'horizon':<8}  {'QRC':>14}  {'best classical':>28}  {'gap':>8}")
    for h in horizons:
        qrc_mean = float(np.mean(agg["QRC_state_injection"][h]))
        qrc_std  = float(np.std(agg["QRC_state_injection"][h]))
        classical = {m: float(np.mean(agg[m][h]))
                       for m in methods if m != "QRC_state_injection"}
        best_m = min(classical, key=classical.get)
        best_v = classical[best_m]
        gap = qrc_mean - best_v
        marker = "  <- QRC WINS" if gap < 0 else ""
        print(f"k={h:<5}  {qrc_mean:.3f}+/-{qrc_std:.3f}  "
                f"{best_m+':':<14} {best_v:.3f}  {gap:+.3f}{marker}")

    # Plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for m in methods:
        means = [np.mean(agg[m][h]) for h in horizons]
        stds  = [np.std(agg[m][h])  for h in horizons]
        ax.errorbar(horizons, means, yerr=stds,
                     marker=METHOD_MARKERS.get(m, "o"),
                     lw=2, ms=9, capsize=4,
                     label=m.replace("_", " "),
                     color=METHOD_COLORS.get(m, "#888"),
                     markeredgecolor="white", markeredgewidth=0.7)
    ax.set_xlabel("Prediction horizon k (timesteps)")
    ax.set_ylabel("Test NRMSE (mean +/- std across target+reservoir seeds)")
    ax_args = runs[0]["args"]
    title = (f"Paper 6 robustness: {len(runs)} target seeds x "
              f"{ax_args['n_seeds']} reservoir seeds\n"
              f"n_input={ax_args['n_input']}, q={ax_args['n_qubits']}, "
              f"reservoir={ax_args['reservoir']} tau={ax_args['tau']}, "
              f"target={ax_args['target_type']}")
    ax.set_title(title, fontsize=10)
    ax.axhline(1.0, color="#888", lw=0.7, ls=":",
                label="trivial-mean predictor")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
