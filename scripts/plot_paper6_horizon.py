#!/usr/bin/env python3
"""
plot_paper6_horizon.py - Standalone NRMSE-vs-horizon plotter for any
paper-6 summary.json.  Lets us re-plot a completed run without re-running
the whole experiment.

Usage:
    python scripts/plot_paper6_horizon.py results/paper6_longhz_n9q11
"""
from __future__ import annotations

import argparse
import json
import sys
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
    ap.add_argument("run_dir", help="results/<run_dir> containing summary.json")
    ap.add_argument("--out", default=None,
                     help="Output PNG path (default: <run_dir>/nrmse_vs_horizon.png)")
    args = ap.parse_args()
    rd = Path(args.run_dir)
    sj = rd / "summary.json"
    if not sj.exists():
        print(f"No summary.json at {sj}")
        sys.exit(1)
    with open(sj) as f:
        s = json.load(f)

    horizons = s["args"]["horizons"]
    out_fig = Path(args.out) if args.out else rd / "nrmse_vs_horizon.png"

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9.0, 5.5))

    print(f"\n=== {rd.name} ===")
    print(f"{'method':<26}  " + "  ".join([f"k={h:<4}" for h in horizons]))
    for method, m in s["per_method_means"].items():
        means = [m[f"k{h}_mean"] for h in horizons]
        stds  = [m[f"k{h}_std"]  for h in horizons]
        row = f"{method:<26}  " + "  ".join(
            [f"{mu:.3f}+/-{sd:.3f}" for mu, sd in zip(means, stds)])
        print(row)
        ax.errorbar(horizons, means, yerr=stds,
                     marker=METHOD_MARKERS.get(method, "o"),
                     lw=2, ms=9, capsize=4,
                     label=method.replace("_", " "),
                     color=METHOD_COLORS.get(method, "#888"),
                     markeredgecolor="white", markeredgewidth=0.7)

    ax.set_xlabel("Prediction horizon k (timesteps)")
    ax.set_ylabel("Test NRMSE (mean over targets, +/- std over seeds)")
    ax_args = s["args"]
    title = (f"Paper 6: Hilbert-space input QRC vs classical reservoirs\n"
              f"n_input={ax_args['n_input']}, q={ax_args['n_qubits']}, "
              f"reservoir={ax_args.get('reservoir','tfim')} "
              f"tau={ax_args.get('tau',1.0)}, "
              f"target={ax_args.get('target_type','pauli')}, "
              f"{ax_args['n_seeds']} seeds")
    ax.set_title(title, fontsize=10)
    ax.axhline(1.0, color="#888", lw=0.7, ls=":",
                label="trivial-mean predictor")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    print(f"\nSaved {out_fig}")


if __name__ == "__main__":
    main()
