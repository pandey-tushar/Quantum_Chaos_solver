#!/usr/bin/env python3
"""
run_swe_hybrid.py - Block B: SWE forecast hybrid (physics step + QRC residual correction).

Story:
    state(t+dt) = physics_step(state(t)) + correction(t+dt)

The physics step uses the over-damped solver (high nu); the correction targets
the residual between coarse-physics and the resolved truth. We do all
correction-learning in the leading-K POD subspace (K small enough to fit a
n_qubits-qubit reservoir).

Training (teacher-forced one-step residuals, no compounding error):
    For each training time t, take truth(t) (in grid space).
    Run a single physics RK4 step to get phys_one_step(t+1).
    Project both truth(t+1) and phys_one_step(t+1) onto POD -> coeffs.
    Target = truth_coeffs(t+1) - phys_coeffs(t+1).
    QRC features = window of past truth-state QRC features.
    Ridge-fit features -> target.

Closed-loop test:
    Initialize from truth(train_end - 1).
    For each test step:
      1. Apply physics RK4 step in grid space -> phys_grid.
      2. Project phys_grid -> phys_coeffs.
      3. QRC predicts correction_coeffs from window of past *predicted* QRC features.
      4. new_coeffs = phys_coeffs + correction_coeffs.
      5. new_grid = pod.inverse_transform(new_coeffs[None, :])[0].
      6. Step the grid forward via new_grid (this is what the next physics step ingests).

Compare three forecasts:
    A) physics-only:   pure RK4 with high-nu solver.
    B) physics+QRC:    correction from QRC.
    C) physics+ESN:    correction from a classical ESN at matched feature dim.

Outputs:
    results/swe_hybrid/swe_hybrid_results.json
    results/swe_hybrid/rmse_vs_leadtime.png
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

from pod_reduction import PODReducer
from swe_solver import rk4_step

# Reuse exactly the QRC extractor + ESN from Block A so the comparison is
# apples-to-apples and we don't fork conventions.
from run_swe_qrc import (   # type: ignore
    make_qrc_extractor,
    _ESN,
    _build_windowed,
    _ridge_solve,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_dict_from_npz(data) -> dict:
    return {
        "Nx": int(data["Nx"]), "Ny": int(data["Ny"]),
        "Lx": float(data["Lx"]), "Ly": float(data["Ly"]),
        "dx": float(data["dx"]), "dy": float(data["dy"]),
    }


def _physics_step_block(state_grid: np.ndarray, n_phys_substeps: int,
                          dt: float, grid: dict, rhs_kwargs: dict) -> np.ndarray:
    """Apply n_phys_substeps RK4 steps with given dt; return new grid state."""
    s = state_grid.copy()
    for _ in range(n_phys_substeps):
        s = rk4_step(s, dt, grid, **rhs_kwargs)
    return s


def _save_snapshot_step_pair(truth_snaps_grid, dt_save_substeps, dt, grid, rhs_kwargs):
    """For each truth snapshot t, compute one *snapshot-step* of physics from truth.
    A snapshot-step is dt_save_substeps RK4 substeps with timestep dt.
    Returns array shape (n_t-1, 3, Nx, Ny) of physics_one_snapshot_step(truth(t)).
    """
    n_t = truth_snaps_grid.shape[0]
    out = np.empty_like(truth_snaps_grid[:n_t - 1])
    for t in range(n_t - 1):
        out[t] = _physics_step_block(truth_snaps_grid[t], dt_save_substeps,
                                     dt, grid, rhs_kwargs)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run SWE hybrid (physics + QRC correction).")
    p.add_argument("--data", type=str, default=str(ROOT / "data" / "swe" / "swe_hybrid_data.npz"))
    p.add_argument("--n-modes", type=int, default=5)
    p.add_argument("--n-qubits", type=int, default=5)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--no-esn", action="store_true")
    p.add_argument("--encoding", type=str, default="global", choices=["global", "per_state"])
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results" / "swe_hybrid"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data}")
    data = np.load(args.data, allow_pickle=False)
    ref_truth_snaps = data["ref_truth_snaps"]
    seed_truth_snaps = data["seed_truth_snaps"]
    seed_phys_snaps  = data["seed_phys_snaps"]
    seeds_in_file = data["seeds"].tolist()
    dt = float(data["dt"]); dt_save = float(data["dt_save"])
    n_phys_substeps = max(1, int(round(dt_save / dt)))
    grid = _grid_dict_from_npz(data)
    linear_phys = bool(int(data["linear_phys"])) if "linear_phys" in data.files else False
    rhs_phys = dict(g=float(data["g"]), f0=float(data["f0"]),
                    H=float(data["H"]), nu=float(data["nu_phys"]),
                    linear=linear_phys)
    print(f"  linear_phys = {linear_phys}")

    print(f"  ref_truth: {ref_truth_snaps.shape}, "
          f"seed_truth: {seed_truth_snaps.shape}, seed_phys: {seed_phys_snaps.shape}")
    print(f"  dt={dt:.2f}s, dt_save={dt_save:.1f}s ({dt_save/3600:.2f}h), "
          f"physics substeps per snapshot = {n_phys_substeps}")
    print(f"  nu_truth={float(data['nu_truth']):.1e}, nu_phys={float(data['nu_phys']):.1e}")

    # ---- POD on resolved truth reference ----
    pod = PODReducer(n_modes=args.n_modes).fit(ref_truth_snaps)
    print(f"\nPOD: {args.n_modes} modes, "
          f"singular values {[f'{s:.1f}' for s in pod.singular_values_]}")

    seeds_to_run = args.seeds if args.seeds is not None else seeds_in_file
    print(f"Seeds: {seeds_to_run}")
    feature_dim = 2 ** args.n_qubits
    print(f"QRC: {args.n_qubits}q x {args.n_layers}L, window={args.window}, "
          f"feature_dim={feature_dim}, ridge_alpha={args.ridge_alpha}")

    per_seed = []
    for s in seeds_to_run:
        if s not in seeds_in_file:
            print(f"  seed {s}: missing, skipping")
            continue
        idx = seeds_in_file.index(s)
        truth_grid = seed_truth_snaps[idx]                  # (n_t, 3, Nx, Ny)
        n_t = truth_grid.shape[0]
        train_end = int(args.train_frac * n_t)
        print(f"\n--- Seed {s}: T={n_t}, train={train_end}, test={n_t - train_end} ---")

        # Truth in POD coefficient space (these are also the QRC inputs)
        truth_coeffs = pod.transform(truth_grid)            # (n_t, K)

        # One-step-physics-from-truth used for residual targets in TRAINING.
        t0 = time.time()
        phys_one_grid = _save_snapshot_step_pair(
            truth_grid[:train_end + 1], n_phys_substeps, dt, grid, rhs_phys)
        # phys_one_grid[t] = physics(truth_grid[t]); shape (train_end, 3, Nx, Ny)
        phys_one_coeffs = pod.transform(phys_one_grid)      # (train_end, K)
        print(f"  computed {len(phys_one_grid)} one-snapshot physics steps "
              f"in {time.time()-t0:.1f}s")

        # Residual training targets: truth(t+1) - phys_one(t)
        # Indexing: phys_one_coeffs[t] corresponds to truth_coeffs[t+1] target.
        # So target_residual[t] = truth_coeffs[t+1] - phys_one_coeffs[t], for t in [0, train_end-1]
        target_residual = truth_coeffs[1:train_end + 1] - phys_one_coeffs

        # ---- Train QRC: window of past truth features -> next residual ----
        # Setup encoder bounds from training truth coeffs only
        train_coeffs = truth_coeffs[:train_end]
        scale_lo = train_coeffs.min(axis=0)
        scale_hi = train_coeffs.max(axis=0)
        margin = 0.1 * (scale_hi - scale_lo + 1e-12)
        scale_lo, scale_hi = scale_lo - margin, scale_hi + margin
        if args.encoding == "global":
            qrc_extractor = make_qrc_extractor(args.n_qubits, args.n_layers, s,
                                                scale_lo=scale_lo, scale_hi=scale_hi,
                                                encoding="global")
        else:
            qrc_extractor = make_qrc_extractor(args.n_qubits, args.n_layers, s,
                                                encoding="per_state")

        # Features for *all* truth snapshots (used at training and rollout init)
        all_qrc_feats = np.array([qrc_extractor(c) for c in truth_coeffs])  # (n_t, 2^q)

        # Build (window, residual) training pairs.
        # For pair i (0 <= i <= train_end - window - 1):
        #   features_window = all_qrc_feats[i : i + window]   (window features ending at i+window-1)
        #   target          = target_residual[i + window - 1]
        #                   = truth_coeffs[i + window] - phys_one_coeffs[i + window - 1]
        # i.e. given features for truth at times [i .. i+window-1], predict the
        # correction needed to advance from truth(i+window-1) to truth(i+window)
        # over a single physics step.
        F_qrc, S_res = [], []
        for i in range(train_end - args.window):
            F_qrc.append(all_qrc_feats[i:i + args.window].flatten())
            S_res.append(target_residual[i + args.window - 1])
        F_qrc = np.asarray(F_qrc)
        S_res = np.asarray(S_res)

        t0 = time.time()
        W_qrc = _ridge_solve(F_qrc, S_res, args.ridge_alpha)
        qrc_train_time = time.time() - t0

        train_pred = F_qrc @ W_qrc
        qrc_train_mse_residual = float(np.mean((train_pred - S_res) ** 2))
        # Train MSE on absolute next-state too (apples-to-apples with physics-only)
        # absolute_pred[i] = phys_one_coeffs[i+window-1] + train_pred[i]
        abs_targets = truth_coeffs[args.window:train_end]   # truth(window..train_end-1)
        abs_phys = phys_one_coeffs[args.window - 1:train_end - 1]
        abs_pred = abs_phys + train_pred
        qrc_train_mse_abs = float(np.mean((abs_pred - abs_targets) ** 2))

        print(f"  QRC training: feature_dim={F_qrc.shape[1]}, samples={F_qrc.shape[0]}, "
              f"resid_mse={qrc_train_mse_residual:.3f}, "
              f"abs_mse={qrc_train_mse_abs:.3f}, t={qrc_train_time:.2f}s")

        # ---- Train ESN baseline at matched feature dim ----
        esn_W, esn = None, None
        if not args.no_esn:
            esn = _ESN(n_input=args.n_modes, n_reservoir=feature_dim, n_output=args.n_modes,
                       ridge_alpha=args.ridge_alpha, seed=s)
            full_acts_esn = esn._activations(truth_coeffs)   # (n_t, feature_dim)
            F_esn, S_e = [], []
            for i in range(train_end - args.window):
                F_esn.append(full_acts_esn[i:i + args.window].flatten())
                S_e.append(target_residual[i + args.window - 1])
            F_esn = np.asarray(F_esn); S_e = np.asarray(S_e)
            t0 = time.time()
            esn_W = _ridge_solve(F_esn, S_e, args.ridge_alpha)
            esn_train_time = time.time() - t0
            esn_train_mse_residual = float(np.mean((F_esn @ esn_W - S_e) ** 2))
            print(f"  ESN training (N={feature_dim}): resid_mse={esn_train_mse_residual:.3f}, "
                  f"t={esn_train_time:.2f}s")

        # ---- Closed-loop forecasts on test window ----
        # The original rollout replaced state_grid with the K-mode POD
        # reconstruction at every step. Under linear physics this didn't
        # blow up but constrained the trajectory to the K-mode subspace,
        # distorting rollout-mean comparisons. Fixed (NeuralGCM-style):
        # keep full-grid state_grid; apply the correction additively in
        # grid space via inv-no-mean transform of the K-dim correction.
        n_test_steps = n_t - train_end
        truth_test_coeffs = truth_coeffs[train_end:train_end + n_test_steps]

        def _correction_to_grid(correction_pod):
            """K-dim POD correction -> grid-space delta (no mean term)."""
            corr_flat = (correction_pod @ pod.modes_.T) * pod.scales_
            return corr_flat.reshape(pod.field_shape_)

        def rollout(corrector):
            """corrector(coeff_window_features) -> correction_coeffs (or zeros)."""
            state_grid = truth_grid[train_end - 1].copy()
            qrc_feat_buf = list(all_qrc_feats[train_end - args.window:train_end])
            esn_r = (full_acts_esn[train_end - 1].copy()
                     if (corrector == "esn" and not args.no_esn) else None)
            esn_act_buf = (list(full_acts_esn[train_end - args.window:train_end])
                           if (corrector == "esn" and not args.no_esn) else None)
            preds = []
            for _ in range(n_test_steps):
                # 1. Physics step (one snapshot worth of substeps)
                phys_grid = _physics_step_block(state_grid, n_phys_substeps,
                                                dt, grid, rhs_phys)
                phys_coeffs = pod.transform(phys_grid[None, ...])[0]

                # 2. Compute correction
                if corrector == "physics":
                    correction = np.zeros(args.n_modes)
                elif corrector == "qrc":
                    flat = np.concatenate(qrc_feat_buf[-args.window:])
                    correction = flat @ W_qrc
                elif corrector == "esn":
                    flat = np.concatenate(esn_act_buf[-args.window:])
                    correction = flat @ esn_W
                else:
                    raise ValueError(corrector)

                # 3. Apply correction additively in grid space; do NOT
                # replace state_grid by the K-mode POD reconstruction.
                if corrector == "physics":
                    state_grid = phys_grid
                else:
                    state_grid = phys_grid + _correction_to_grid(correction)

                # 4. Record the OBSERVED POD coefficients of the actual
                # full-grid state.
                observed_coeffs = pod.transform(state_grid[None, ...])[0]
                preds.append(observed_coeffs)

                # 5. Update feature buffers (use observed, not predicted)
                qrc_feat_buf.append(qrc_extractor(observed_coeffs))
                if corrector == "esn":
                    esn_r = ((1 - esn.alpha) * esn_r
                             + esn.alpha * np.tanh(esn.W_res @ esn_r + esn.W_in @ observed_coeffs))
                    esn_act_buf.append(esn_r.copy())
            return np.asarray(preds)

        cl_phys = rollout("physics")
        cl_qrc  = rollout("qrc")
        cl_esn  = rollout("esn") if (esn_W is not None) else None

        def per_lead_mse(preds):
            n = min(len(preds), len(truth_test_coeffs))
            return np.mean((preds[:n] - truth_test_coeffs[:n]) ** 2, axis=1)

        phys_lead = per_lead_mse(cl_phys)
        qrc_lead  = per_lead_mse(cl_qrc)
        esn_lead  = per_lead_mse(cl_esn) if cl_esn is not None else None

        result = {
            "seed": int(s), "T": int(n_t), "train_end": int(train_end),
            "qrc": {
                "train_residual_mse": qrc_train_mse_residual,
                "train_abs_mse": qrc_train_mse_abs,
                "closed_loop_per_lead_mse": qrc_lead.tolist(),
                "train_time_s": qrc_train_time,
                "feature_dim": int(feature_dim),
            },
            "physics": {"closed_loop_per_lead_mse": phys_lead.tolist()},
        }
        if esn_lead is not None:
            result["esn"] = {
                "train_residual_mse": esn_train_mse_residual,
                "closed_loop_per_lead_mse": esn_lead.tolist(),
                "train_time_s": esn_train_time,
                "feature_dim": int(feature_dim),
            }
        print(f"  CL lead-1 MSE: physics={phys_lead[0]:.3f}, "
              f"phys+QRC={qrc_lead[0]:.3f}"
              + (f", phys+ESN={esn_lead[0]:.3f}" if esn_lead is not None else ""))
        print(f"  CL mean MSE  : physics={phys_lead.mean():.3f}, "
              f"phys+QRC={qrc_lead.mean():.3f}"
              + (f", phys+ESN={esn_lead.mean():.3f}" if esn_lead is not None else ""))
        per_seed.append(result)

    # ---- Aggregate ----
    summary = {
        "config": {
            "n_modes": args.n_modes, "n_qubits": args.n_qubits, "n_layers": args.n_layers,
            "window": args.window, "ridge_alpha": args.ridge_alpha,
            "train_frac": args.train_frac, "feature_dim": feature_dim,
            "encoding": args.encoding,
        },
        "data": {
            "ref_truth_snaps_shape": list(ref_truth_snaps.shape),
            "seed_truth_snaps_shape": list(seed_truth_snaps.shape),
            "dt_save_s": dt_save,
            "nu_truth": float(data["nu_truth"]),
            "nu_phys": float(data["nu_phys"]),
        },
        "pod_singular_values": pod.singular_values_.tolist(),
        "per_seed": per_seed,
    }

    if len(per_seed) >= 2:
        from scipy import stats as _stats
        print("\n" + "=" * 78)
        print(f"PER-SEED CL-MEAN MSE (n_seeds = {len(per_seed)})")
        print("=" * 78)
        header = f"{'seed':>4}  {'physics':>10}  {'phys+QRC':>10}"
        if any("esn" in r for r in per_seed):
            header += f"  {'phys+ESN':>10}"
        print(header)
        print("-" * 78)
        phys_v, qrc_v, esn_v = [], [], []
        for r in per_seed:
            phys_m = float(np.mean(r["physics"]["closed_loop_per_lead_mse"]))
            qrc_m  = float(np.mean(r["qrc"]["closed_loop_per_lead_mse"]))
            phys_v.append(phys_m); qrc_v.append(qrc_m)
            row = f"{r['seed']:>4}  {phys_m:>10.3f}  {qrc_m:>10.3f}"
            if "esn" in r:
                esn_m = float(np.mean(r["esn"]["closed_loop_per_lead_mse"]))
                esn_v.append(esn_m)
                row += f"  {esn_m:>10.3f}"
            print(row)
        phys_v, qrc_v = np.asarray(phys_v), np.asarray(qrc_v)
        print("-" * 78)
        print(f"{'mean':>4}  {phys_v.mean():>10.3f}  {qrc_v.mean():>10.3f}"
              + (f"  {np.mean(esn_v):>10.3f}" if esn_v else ""))
        print(f"{'std':>4}  {phys_v.std():>10.3f}  {qrc_v.std():>10.3f}"
              + (f"  {np.std(esn_v):>10.3f}" if esn_v else ""))

        def _wilcoxon(a, b, label):
            try:
                stat, pval = _stats.wilcoxon(a, b)
                print(f"\nWilcoxon ({label}, n={len(a)}): stat={stat:.2f}, p={pval:.4f}")
                return {"stat": float(stat), "p_value": float(pval),
                         "a_mean": float(a.mean()), "b_mean": float(b.mean())}
            except Exception as e:
                print(f"\nWilcoxon ({label}) skipped: {e}")
                return None

        stats_block = {
            "wilcoxon_qrc_vs_physics_mean_mse": _wilcoxon(qrc_v, phys_v, "phys+QRC vs physics-only (91-step mean)"),
        }
        if esn_v:
            esn_v = np.asarray(esn_v)
            stats_block["wilcoxon_qrc_vs_esn_mean_mse"] = _wilcoxon(
                qrc_v, esn_v, "phys+QRC vs phys+ESN (91-step mean)")

        # Lead-1 (and short-horizon mean) -- the lead times where the residual
        # learner is supposed to add value before error compounding takes over.
        phys_l1 = np.array([r["physics"]["closed_loop_per_lead_mse"][0] for r in per_seed])
        qrc_l1  = np.array([r["qrc"]["closed_loop_per_lead_mse"][0]     for r in per_seed])
        print("\n--- Lead-1 (1-step closed loop) ---")
        print(f"  phys mean = {phys_l1.mean():.3f}, QRC mean = {qrc_l1.mean():.3f}")
        stats_block["wilcoxon_qrc_vs_physics_lead1"] = _wilcoxon(
            qrc_l1, phys_l1, "phys+QRC vs physics-only (lead-1)")
        if esn_v.size if isinstance(esn_v, np.ndarray) else False:
            esn_l1 = np.array([r["esn"]["closed_loop_per_lead_mse"][0] for r in per_seed])
            print(f"  ESN mean  = {esn_l1.mean():.3f}")
            stats_block["wilcoxon_qrc_vs_esn_lead1"] = _wilcoxon(
                qrc_l1, esn_l1, "phys+QRC vs phys+ESN (lead-1)")

        summary["statistics"] = stats_block

    out_json = out_dir / "swe_hybrid_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        max_lead = max(len(r["physics"]["closed_loop_per_lead_mse"]) for r in per_seed)
        leads = np.arange(1, max_lead + 1)

        def stack(method):
            arrs = []
            for r in per_seed:
                if method not in r:
                    continue
                v = np.array(r[method]["closed_loop_per_lead_mse"])
                if len(v) < max_lead:
                    v = np.pad(v, (0, max_lead - len(v)), constant_values=np.nan)
                arrs.append(v)
            return np.array(arrs) if arrs else None

        fig, ax = plt.subplots(figsize=(7, 4.5))
        for method, label, color in [
            ("physics", "Physics only (high nu)", "#888"),
            ("qrc",     "Physics + QRC correction", "#0066cc"),
            ("esn",     f"Physics + ESN correction (N={feature_dim})", "#cc4400"),
        ]:
            data_s = stack(method)
            if data_s is None:
                continue
            mean = np.nanmean(data_s, axis=0)
            std = np.nanstd(data_s, axis=0)
            ax.plot(leads, np.sqrt(mean), label=label, color=color, lw=2)
            ax.fill_between(leads, np.sqrt(np.maximum(mean - std, 0.0)),
                             np.sqrt(mean + std), color=color, alpha=0.2)
        ax.set_xlabel(f"Lead time (steps; 1 step = {dt_save/3600:.2f} h)")
        ax.set_ylabel("RMSE in POD coefficient space (vs truth)")
        ax.set_title(f"Hybrid SWE forecast (mean +/- std over {len(per_seed)} seeds)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / "rmse_vs_leadtime.png"
        fig.savefig(fig_path, dpi=140)
        print(f"Saved {fig_path}")
    except Exception as e:
        print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
