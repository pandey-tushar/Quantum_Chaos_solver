#!/usr/bin/env python3
"""
sweep_swe_qrc.py - Sensitivity sweep for the Block A SWE+QRC pipeline.

Grid: n_qubits in {5, 6, 7} x window in {3, 5, 7} x ridge_alpha in {0.1, 1.0, 10.0}.
Each cell runs the full 8-seed pipeline and reports:
  - Mean +/- std of teacher MSE for QRC, ESN, persistence
  - Paired Wilcoxon p-value for QRC vs ESN at matched feature dim
  - QRC training time per seed

Reuses the canonical evaluators in run_swe_qrc.py to keep the methodology
identical to the headline 8-seed result.

Run:
    python scripts/sweep_swe_qrc.py --data data/swe/swe_data.npz \\
        --out-dir results/swe_qrc/sensitivity
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pod_reduction import PODReducer
from run_swe_qrc import evaluate_qrc, evaluate_esn, evaluate_persistence  # type: ignore


def main():
    p = argparse.ArgumentParser(description="Sensitivity sweep for SWE+QRC Block A")
    p.add_argument("--data", type=str, default=str(ROOT / "data" / "swe" / "swe_data.npz"))
    p.add_argument("--n-modes", type=int, default=5)
    p.add_argument("--n-qubits", type=int, nargs="+", default=[5, 6, 7])
    p.add_argument("--windows", type=int, nargs="+", default=[3, 5, 7])
    p.add_argument("--ridge-alphas", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--encoding", type=str, default="global", choices=["global", "per_state"])
    p.add_argument("--no-delta-target", action="store_true")
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results" / "swe_qrc" / "sensitivity"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    delta = not args.no_delta_target

    print(f"Loading {args.data}")
    data = np.load(args.data, allow_pickle=False)
    ref_snaps = data["ref_snaps"]
    seed_snaps = data["seed_snaps"]
    seeds_in_file = data["seeds"].tolist()
    dt_save = float(data["dt_save"])

    pod = PODReducer(n_modes=args.n_modes).fit(ref_snaps)
    print(f"POD: {args.n_modes} modes")

    seeds_to_run = args.seeds if args.seeds is not None else seeds_in_file
    print(f"Seeds: {seeds_to_run}\n")

    # Project trajectories once per seed (POD is shared across grid cells)
    coeffs_per_seed = {}
    for s in seeds_to_run:
        if s not in seeds_in_file:
            continue
        coeffs_per_seed[s] = pod.transform(seed_snaps[seeds_in_file.index(s)])

    grid = []
    for nq in args.n_qubits:
        for w in args.windows:
            for a in args.ridge_alphas:
                grid.append((nq, w, a))
    print(f"Grid: {len(grid)} cells "
          f"(n_qubits={args.n_qubits}, window={args.windows}, ridge_alpha={args.ridge_alphas})\n")

    rows = []
    from scipy import stats as _stats
    for cell_idx, (nq, w, a) in enumerate(grid):
        feature_dim = 2 ** nq
        cell_t0 = time.time()
        per_seed_qrc, per_seed_esn, per_seed_pers = [], [], []
        per_seed_qrc_train_t = []
        for s, coeffs in coeffs_per_seed.items():
            T = len(coeffs)
            train_end = int(args.train_frac * T)
            qrc = evaluate_qrc(coeffs, train_end,
                                n_qubits=nq, n_layers=args.n_layers,
                                window=w, seed=s, ridge_alpha=a,
                                encoding=args.encoding, delta_target=delta)
            esn = evaluate_esn(coeffs, train_end,
                                n_reservoir=feature_dim, window=w,
                                seed=s, ridge_alpha=a, delta_target=delta)
            pers = evaluate_persistence(coeffs, train_end)
            per_seed_qrc.append(qrc["teacher_mse"])
            per_seed_esn.append(esn["teacher_mse"])
            per_seed_pers.append(pers["closed_loop_per_lead_mse"][0])
            per_seed_qrc_train_t.append(qrc["train_time_s"])

        qrc_v = np.asarray(per_seed_qrc)
        esn_v = np.asarray(per_seed_esn)
        pers_v = np.asarray(per_seed_pers)
        try:
            wstat, pval = _stats.wilcoxon(qrc_v, esn_v)
        except Exception:
            wstat, pval = float("nan"), float("nan")
        cell_dt = time.time() - cell_t0

        wins = int(np.sum(qrc_v < esn_v))
        row = dict(
            n_qubits=nq, window=w, ridge_alpha=a,
            feature_dim=feature_dim, n_seeds=len(qrc_v),
            qrc_mean=float(qrc_v.mean()), qrc_std=float(qrc_v.std()),
            esn_mean=float(esn_v.mean()), esn_std=float(esn_v.std()),
            persistence_mean=float(pers_v.mean()),
            qrc_wins_vs_esn=wins,
            wilcoxon_p=float(pval),
            qrc_train_t_mean=float(np.mean(per_seed_qrc_train_t)),
            cell_runtime_s=cell_dt,
        )
        rows.append(row)
        print(f"[{cell_idx+1:2d}/{len(grid)}] q={nq} w={w} a={a:>5} | "
              f"QRC {qrc_v.mean():>7.3f}+/-{qrc_v.std():>6.3f}  "
              f"ESN {esn_v.mean():>7.3f}+/-{esn_v.std():>6.3f}  "
              f"pers {pers_v.mean():>6.3f}  "
              f"QRC<ESN={wins}/{len(qrc_v)} p={pval:.4f}  "
              f"({cell_dt:.1f}s, {per_seed_qrc_train_t[0]:.2f}s/seed)")

    summary = {
        "config": {
            "n_modes": args.n_modes, "n_layers": args.n_layers,
            "encoding": args.encoding, "delta_target": delta,
            "train_frac": args.train_frac,
            "seeds": list(coeffs_per_seed.keys()),
            "grid": [{"n_qubits": nq, "window": w, "ridge_alpha": a}
                     for (nq, w, a) in grid],
        },
        "rows": rows,
    }
    out_json = out_dir / "sweep_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    # --- Quick visualisation: heatmap of QRC mean teacher MSE ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        nqs = sorted(set(r["n_qubits"] for r in rows))
        ws = sorted(set(r["window"] for r in rows))
        ass_ = sorted(set(r["ridge_alpha"] for r in rows))

        fig, axes = plt.subplots(1, len(nqs), figsize=(4.5 * len(nqs), 4), sharey=True)
        if len(nqs) == 1:
            axes = [axes]
        vmin = min(r["qrc_mean"] for r in rows)
        vmax = max(r["qrc_mean"] for r in rows)
        for ax, nq in zip(axes, nqs):
            mat = np.zeros((len(ws), len(ass_)))
            for i, w in enumerate(ws):
                for j, a in enumerate(ass_):
                    cell = next((r for r in rows
                                 if r["n_qubits"] == nq and r["window"] == w
                                 and r["ridge_alpha"] == a), None)
                    mat[i, j] = cell["qrc_mean"] if cell else np.nan
            im = ax.imshow(mat, aspect="auto", cmap="viridis",
                           vmin=vmin, vmax=vmax, origin="lower")
            ax.set_xticks(range(len(ass_)), [f"{a:g}" for a in ass_])
            ax.set_yticks(range(len(ws)), ws)
            ax.set_xlabel("ridge alpha")
            ax.set_title(f"{nq} qubits  (feature dim {2**nq})")
            for i in range(len(ws)):
                for j in range(len(ass_)):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            color="white" if mat[i, j] < (vmin + vmax) / 2 else "black",
                            fontsize=9)
            if ax is axes[0]:
                ax.set_ylabel("window")
        fig.colorbar(im, ax=axes, label="QRC teacher MSE (mean across seeds)")
        fig.suptitle("Block A sensitivity: QRC teacher MSE vs (qubits, window, ridge alpha)")
        out_fig = out_dir / "qrc_teacher_mse_heatmap.png"
        fig.savefig(out_fig, dpi=140, bbox_inches="tight")
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
