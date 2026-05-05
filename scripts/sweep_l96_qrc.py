#!/usr/bin/env python3
"""
sweep_l96_qrc.py - Hyperparameter sensitivity sweep for the L96 K=q
matched-complexity scaling result.

At a fixed problem dimension N (default 8, matching arXiv 2604.23743),
sweep the QRC hyperparameter grid:

    n_layers   in {1, 2, 3}
    window     in {3, 5, 7}
    ridge_alpha in {0.1, 1.0, 10.0}

For each cell, run the autonomous L96 pipeline with QRC and matched-
feature-dim ESN (n_reservoir = 2^N), 8 seeds, paired Wilcoxon. Saves
results/l96_sensitivity/sweep.json and a heatmap.

Reuses the per-cell experiment in scripts/run_l96_kq_scaling.py via
import.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from run_l96_kq_scaling import run_cell   # type: ignore


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--n-layers", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--windows", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--ridge-alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "l96_sensitivity"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    grid = [(L, w, a) for L in args.n_layers for w in args.windows for a in args.ridge_alphas]
    print(f"L96 sensitivity sweep at N=q={args.N}, {len(grid)} cells, {args.n_seeds} seeds each\n")

    for idx, (L, w, a) in enumerate(grid, start=1):
        t0 = time.time()
        cell = run_cell(N=args.N, n_seeds=args.n_seeds, n_layers=L,
                          window=w, ridge_alpha=a)
        cell["n_layers"] = L; cell["window"] = w; cell["ridge_alpha"] = a
        cell["wall_time_s"] = time.time() - t0
        rows.append(cell)
        winner = "QRC" if cell["qrc_mean"] < cell["esn_mean"] else "ESN"
        print(f"[{idx:>2}/{len(grid)}] L={L} w={w} a={a:>5} | "
              f"QRC={cell['qrc_mean']:>7.3f}+/-{cell['qrc_std']:>5.3f}  "
              f"ESN={cell['esn_mean']:>7.3f}+/-{cell['esn_std']:>5.3f}  "
              f"QRC<ESN={cell['qrc_wins_vs_esn']}/{args.n_seeds}  "
              f"p={cell['wilcoxon_qrc_vs_esn']:.4f}  "
              f"lower-mean: {winner}  ({cell['wall_time_s']:.1f}s)")

    summary = {"N": args.N, "n_seeds": args.n_seeds, "rows": rows}
    out_json = out_dir / "sweep.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    n_significant = sum(1 for r in rows
                          if r["wilcoxon_qrc_vs_esn"] <= 0.05
                          and r["qrc_mean"] < r["esn_mean"])
    n_8of8 = sum(1 for r in rows if r["qrc_wins_vs_esn"] == args.n_seeds)
    print(f"\n{n_significant}/{len(grid)} cells: QRC < ESN with p <= 0.05")
    print(f"{n_8of8}/{len(grid)} cells: 8/8 seed wins for QRC")

    # Heatmap of QRC mean MSE vs (window, ridge_alpha) for each n_layers
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Ls = sorted(set(r["n_layers"] for r in rows))
        ws = sorted(set(r["window"] for r in rows))
        ass_ = sorted(set(r["ridge_alpha"] for r in rows))

        fig, axes = plt.subplots(1, len(Ls), figsize=(4.0 * len(Ls), 4),
                                    sharey=True)
        if len(Ls) == 1:
            axes = [axes]
        vmin = min(r["qrc_mean"] for r in rows)
        vmax = max(r["qrc_mean"] for r in rows)
        for ax, L in zip(axes, Ls):
            mat = np.zeros((len(ws), len(ass_)))
            for i, w in enumerate(ws):
                for j, a in enumerate(ass_):
                    cell = next((r for r in rows
                                 if r["n_layers"] == L and r["window"] == w
                                 and r["ridge_alpha"] == a), None)
                    mat[i, j] = cell["qrc_mean"] if cell else np.nan
            im = ax.imshow(mat, aspect="auto", cmap="viridis",
                            vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks(range(len(ass_)), [f"{a:g}" for a in ass_])
            ax.set_yticks(range(len(ws)), ws)
            ax.set_xlabel("ridge alpha")
            ax.set_title(f"{L} reservoir layers")
            for i in range(len(ws)):
                for j in range(len(ass_)):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="white" if mat[i, j] < (vmin + vmax) / 2 else "black",
                            fontsize=9)
            if ax is axes[0]:
                ax.set_ylabel("window")
        fig.colorbar(im, ax=axes, label="QRC test teacher MSE (8-seed mean)")
        fig.suptitle(f"L96 N=q={args.N}: QRC test MSE vs hyperparameters")
        out_fig = out_dir / "qrc_mse_heatmap.png"
        fig.savefig(out_fig, dpi=140, bbox_inches="tight")
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
