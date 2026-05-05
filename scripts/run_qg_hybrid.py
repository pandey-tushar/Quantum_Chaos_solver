#!/usr/bin/env python3
"""
run_qg_hybrid.py - QG analogue of `run_swe_hybrid.py`.

Single-layer barotropic QG truth (full nonlinear) + linearised QG physics
core; QRC learns the residual between truth and physics in POD space.
The pipeline structure is identical to the SWE hybrid pipeline; only the
physics step and the data loader change.

This is a thin wrapper that:
  1. Loads paired (truth, physics) QG data produced by
     `scripts/generate_qg_hybrid_data.py` -- snapshots are 2-channel
     (vorticity, streamfunction).
  2. Fits POD on the reference truth trajectory.
  3. For each seed: trains QRC + ESN on the residual between
     truth_coeffs[t+1] and one-step-physics(truth_grid[t]) projected to
     POD; closed-loop forecasts on the test window with
     physics-only / physics+QRC / physics+ESN; reports paired Wilcoxon
     at lead-1 and the rollout mean.

Run:
    python scripts/run_qg_hybrid.py --n-modes 5 --n-qubits 5 \\
        --window 5 --ridge-alpha 1.0 --encoding global

Status (2026-05-04, v2.0.0): the framework is wired but expects
generate_qg_hybrid_data.py to have been executed at least once.
The truth-vs-linear-phys gap may need re-tuning for the QG setting
(initial smoke test showed 100%+ divergence over 30 days, which is
larger than the SWE setup; for an atmospheric-style application the
linear physics core is likely too crude and a resolution-difference
truncation may be preferable. The only file that needs to change for
that is generate_qg_hybrid_data.py).
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
from qg_solver import qg_rk4_step

# Reuse the SWE Block-B machinery wholesale; only the physics step
# (qg_rk4_step) and the data loader are QG-specific.
from run_swe_qrc import (   # type: ignore
    make_qrc_extractor,
    _ESN,
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


def _spectral_grid(grid: dict) -> dict:
    """Re-derive the spectral wavenumbers + dealias mask the QG solver needs."""
    Nx, Ny = grid["Nx"], grid["Ny"]
    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=grid["dx"])
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=grid["dy"])
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    kx_max = (2.0 / 3.0) * np.max(np.abs(kx))
    ky_max = (2.0 / 3.0) * np.max(np.abs(ky))
    dealias = (np.abs(KX) <= kx_max) & (np.abs(KY) <= ky_max)
    return {**grid, "KX": KX, "KY": KY, "K2": K2, "dealias": dealias}


def _physics_step_block(q_grid: np.ndarray, n_phys_substeps: int,
                          dt: float, grid: dict, rhs_kwargs: dict) -> np.ndarray:
    """Apply n_phys_substeps QG RK4 steps; q_grid shape is (Nx, Ny)."""
    q = q_grid.copy()
    for _ in range(n_phys_substeps):
        q = qg_rk4_step(q, dt, grid, **rhs_kwargs)
    return q


def _vorticity_to_psi(q: np.ndarray, grid: dict) -> np.ndarray:
    K2 = grid["K2"]
    psi_hat = -np.fft.fft2(q) / K2
    psi_hat[0, 0] = 0.0
    return np.real(np.fft.ifft2(psi_hat))


def _stack_q_psi(q: np.ndarray, grid: dict) -> np.ndarray:
    """For a single-snapshot q (Nx, Ny) -> (2, Nx, Ny) with channel 1 = psi."""
    psi = _vorticity_to_psi(q, grid)
    return np.stack([q, psi], axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run QG hybrid (physics + QRC correction).")
    p.add_argument("--data", type=str, default=str(ROOT / "data" / "qg" / "qg_hybrid_data.npz"))
    p.add_argument("--n-modes", type=int, default=5)
    p.add_argument("--n-qubits", type=int, default=5)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--no-esn", action="store_true")
    p.add_argument("--encoding", type=str, default="global", choices=["global", "per_state"])
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results" / "qg_hybrid"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data}")
    data = np.load(args.data, allow_pickle=False)
    ref_truth_snaps = data["ref_truth_snaps"]                    # (n_t, 2, Nx, Ny)
    seed_truth_snaps = data["seed_truth_snaps"]                  # (n_seeds, n_t, 2, Nx, Ny)
    seed_phys_snaps  = data["seed_phys_snaps"]
    seeds_in_file = data["seeds"].tolist()
    dt = float(data["dt"]); dt_save = float(data["dt_save"])
    n_phys_substeps = max(1, int(round(dt_save / dt)))
    grid_meta = _grid_dict_from_npz(data)
    grid = _spectral_grid(grid_meta)
    rhs_phys = dict(beta=float(data["beta"]),
                     r=float(data["r_drag"]),
                     nu=float(data["nu"]),
                     p_hyper=int(data["p_hyper"]),
                     linear=bool(int(data["linear_phys"])))

    print(f"  ref_truth: {ref_truth_snaps.shape}, "
          f"seed_truth: {seed_truth_snaps.shape}, seed_phys: {seed_phys_snaps.shape}")
    print(f"  dt={dt:.0f}s, dt_save={dt_save:.0f}s ({dt_save/3600:.2f}h), "
          f"physics substeps per snapshot = {n_phys_substeps}")

    # ---- POD on resolved truth reference (channels = vorticity, streamfunction) ----
    pod = PODReducer(n_modes=args.n_modes).fit(ref_truth_snaps)
    print(f"\nPOD: {args.n_modes} modes, "
          f"singular values {[f'{s:.3e}' for s in pod.singular_values_]}")

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
        truth_snaps = seed_truth_snaps[idx]                      # (n_t, 2, Nx, Ny)
        n_t = truth_snaps.shape[0]
        train_end = int(args.train_frac * n_t)
        print(f"\n--- Seed {s}: T={n_t}, train={train_end}, test={n_t - train_end} ---")

        # Truth in POD coefficient space
        truth_coeffs = pod.transform(truth_snaps)                # (n_t, K)

        # One-step-physics-from-truth used for residual targets in TRAINING.
        t0 = time.time()
        phys_one_pairs = []
        for t in range(train_end):
            # Take vorticity (channel 0) and run one snapshot-step of physics
            q_t = truth_snaps[t, 0]
            q_next = _physics_step_block(q_t, n_phys_substeps, dt, grid, rhs_phys)
            phys_one_pairs.append(_stack_q_psi(q_next, grid))
        phys_one_grid = np.stack(phys_one_pairs, axis=0)
        phys_one_coeffs = pod.transform(phys_one_grid)
        print(f"  computed {len(phys_one_grid)} one-snapshot physics steps "
              f"in {time.time()-t0:.1f}s")

        # Residual training targets: truth(t+1) - phys_one(t)
        target_residual = truth_coeffs[1:train_end + 1] - phys_one_coeffs

        # ---- Train QRC on (windowed truth coeffs) -> (next residual) ----
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
        all_qrc_feats = np.array([qrc_extractor(c) for c in truth_coeffs])

        F_qrc, S_res = [], []
        for i in range(train_end - args.window):
            F_qrc.append(all_qrc_feats[i:i + args.window].flatten())
            S_res.append(target_residual[i + args.window - 1])
        F_qrc = np.asarray(F_qrc); S_res = np.asarray(S_res)
        t0 = time.time()
        W_qrc = _ridge_solve(F_qrc, S_res, args.ridge_alpha)
        qrc_train_time = time.time() - t0
        train_pred = F_qrc @ W_qrc
        qrc_train_mse = float(np.mean((train_pred - S_res) ** 2))
        print(f"  QRC training: feature_dim={F_qrc.shape[1]}, samples={F_qrc.shape[0]}, "
              f"residual_mse={qrc_train_mse:.3e}, t={qrc_train_time:.2f}s")

        # ESN baseline
        esn_W = None
        if not args.no_esn:
            esn = _ESN(n_input=args.n_modes, n_reservoir=feature_dim, n_output=args.n_modes,
                       ridge_alpha=args.ridge_alpha, seed=s)
            full_acts = esn._activations(truth_coeffs)
            F_esn, S_e = [], []
            for i in range(train_end - args.window):
                F_esn.append(full_acts[i:i + args.window].flatten())
                S_e.append(target_residual[i + args.window - 1])
            F_esn = np.asarray(F_esn); S_e = np.asarray(S_e)
            t0 = time.time()
            esn_W = _ridge_solve(F_esn, S_e, args.ridge_alpha)
            esn_train_time = time.time() - t0
            esn_train_mse = float(np.mean((F_esn @ esn_W - S_e) ** 2))
            print(f"  ESN training (N={feature_dim}): "
                  f"residual_mse={esn_train_mse:.3e}, t={esn_train_time:.2f}s")

        # ---- Closed-loop forecasts on test window ----
        n_test_steps = n_t - train_end

        def rollout(corrector):
            q_state = truth_snaps[train_end - 1, 0].copy()       # vorticity only
            qrc_feat_buf = list(all_qrc_feats[train_end - args.window:train_end])
            esn_r = (full_acts[train_end - 1].copy()
                     if (corrector == "esn" and esn_W is not None) else None)
            esn_act_buf = (list(full_acts[train_end - args.window:train_end])
                           if (corrector == "esn" and esn_W is not None) else None)
            preds = []
            for _ in range(n_test_steps):
                # 1. Physics step in vorticity space
                q_next = _physics_step_block(q_state, n_phys_substeps, dt, grid, rhs_phys)
                phys_pair = _stack_q_psi(q_next, grid)             # (2, Nx, Ny)
                phys_coeffs = pod.transform(phys_pair[None, ...])[0]

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

                new_coeffs = phys_coeffs + correction
                preds.append(new_coeffs)

                # 3. Reconstruct vorticity for next physics step
                new_pair = pod.inverse_transform(new_coeffs[None, :])[0]
                q_state = new_pair[0]                              # discard psi channel

                # 4. Update buffers
                qrc_feat_buf.append(qrc_extractor(new_coeffs))
                if corrector == "esn":
                    esn_r = ((1 - esn.alpha) * esn_r
                             + esn.alpha * np.tanh(esn.W_res @ esn_r + esn.W_in @ new_coeffs))
                    esn_act_buf.append(esn_r.copy())
            return np.asarray(preds)

        truth_test_coeffs = truth_coeffs[train_end:train_end + n_test_steps]

        cl_phys = rollout("physics")
        cl_qrc  = rollout("qrc")
        cl_esn  = rollout("esn") if esn_W is not None else None

        def per_lead_mse(preds):
            n = min(len(preds), len(truth_test_coeffs))
            return np.mean((preds[:n] - truth_test_coeffs[:n]) ** 2, axis=1)

        result = {
            "seed": int(s), "T": int(n_t), "train_end": int(train_end),
            "qrc": {
                "train_residual_mse": qrc_train_mse,
                "closed_loop_per_lead_mse": per_lead_mse(cl_qrc).tolist(),
                "feature_dim": int(feature_dim),
            },
            "physics": {"closed_loop_per_lead_mse": per_lead_mse(cl_phys).tolist()},
        }
        if cl_esn is not None:
            result["esn"] = {
                "train_residual_mse": esn_train_mse,
                "closed_loop_per_lead_mse": per_lead_mse(cl_esn).tolist(),
                "feature_dim": int(feature_dim),
            }

        phys_lead = np.array(result["physics"]["closed_loop_per_lead_mse"])
        qrc_lead  = np.array(result["qrc"]["closed_loop_per_lead_mse"])
        print(f"  CL lead-1 MSE: physics={phys_lead[0]:.3e}, "
              f"phys+QRC={qrc_lead[0]:.3e}"
              + (f", phys+ESN={result['esn']['closed_loop_per_lead_mse'][0]:.3e}"
                  if cl_esn is not None else ""))
        per_seed.append(result)

    # ---- Save + Wilcoxon ----
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
            "beta": float(data["beta"]),
            "nu": float(data["nu"]),
        },
        "pod_singular_values": pod.singular_values_.tolist(),
        "per_seed": per_seed,
    }

    if len(per_seed) >= 2:
        from scipy import stats as _stats
        phys_l1 = np.array([r["physics"]["closed_loop_per_lead_mse"][0] for r in per_seed])
        qrc_l1  = np.array([r["qrc"]["closed_loop_per_lead_mse"][0] for r in per_seed])
        try:
            p_phys = _stats.wilcoxon(qrc_l1, phys_l1).pvalue
            print(f"\nWilcoxon (phys+QRC vs physics-only, lead-1, n={len(qrc_l1)}): "
                  f"p={p_phys:.4f}")
            summary["wilcoxon_qrc_vs_phys_lead1"] = float(p_phys)
        except Exception as e:
            print(f"Wilcoxon skipped: {e}")
        if any("esn" in r for r in per_seed):
            esn_l1 = np.array([r["esn"]["closed_loop_per_lead_mse"][0] for r in per_seed])
            try:
                p_esn = _stats.wilcoxon(qrc_l1, esn_l1).pvalue
                print(f"Wilcoxon (phys+QRC vs phys+ESN, lead-1, n={len(qrc_l1)}): "
                      f"p={p_esn:.4f}")
                summary["wilcoxon_qrc_vs_esn_lead1"] = float(p_esn)
            except Exception as e:
                print(f"Wilcoxon skipped: {e}")

    out_json = out_dir / "qg_hybrid_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")


if __name__ == "__main__":
    main()
