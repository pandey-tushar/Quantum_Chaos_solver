#!/usr/bin/env python3
"""
run_paper4_finance_synthetic.py - Paper 4 v0 experiment on a SYNTHETIC
multi-asset financial dataset.  Tests whether the paper-6 long-horizon
QRC edge transfers from quantum-state inputs to angle-encoded classical
financial returns.

Data: synthetic regime-switching multi-factor returns
  - 9 assets, 2000+ timesteps
  - Hidden Markov regime in {bull, bear, crisis}, persistent (~3% switch)
  - Per-regime drifts mu_i(s_t) and volatilities sigma_i(s_t)
  - AR(1) stochastic volatility multiplier (long memory)
  - Static cross-asset correlation matrix (factor structure)
  - Output: log-returns r_i(t) in roughly [-0.1, 0.1] range

Encoding: |psi_t> = U_ent * (tensor_i RY(pi * r_i(t)) |0>)
  - U_ent is a fixed entangling layer (one timestep of TFIM evolution
    with light coupling) -- gives a non-trivial 9-qubit state from the
    classical return vector

Methods (all at matched windowed feature dim D_eff):
  - QRC_state_injection: encode r_t into |psi_t>, inject into 11-qubit
    scram reservoir, sample M = q*1024 shots, extract local Paulis.
  - Classical_direct_RFF: RFF on the windowed raw return vector r_t.
    Classical has free access to r_t -- no shadow tomography needed.
  - Classical_direct_ESN: ESN on windowed r_t.  Same idea.
  - Classical_linear: linear ridge on windowed r_t.  Sanity baseline.

Targets: r(t+k) for each asset i, horizons k in {1, 5, 10, 20, 40, 80}.
Predicted simultaneously (multi-target) via shared feature map.

Outputs: results/paper4_synthetic/summary.json + nrmse_vs_horizon.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import linalg as sla

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_quantum_input_experiment import (
    pauli_string, single_site, two_site,
    tfim_hamiltonian, scrambling_hamiltonian,
    _apply_single_qubit_gate,
    make_esn, esn_run, rff_features,
    ridge_solve, windowed_features, feature_gram_condition,
)
from run_quantum_state_input import eval_multitarget, make_state_injection_qrc


def build_target(R: np.ndarray, k: int, target_type: str) -> np.ndarray:
    """Build the prediction target at horizon k.  Returns (T, n_assets)
    with NaN at the trailing k entries (where the future window runs off
    the end).
    """
    T, A = R.shape
    out = np.full((T, A), np.nan)
    if target_type == "raw_return":
        out[: T - k] = R[k:]
    elif target_type == "cumulative_return":
        # sum_{s=1..k} r_i(t+s) -- k-step forward log-return
        # vectorize via cumulative-sum trick
        cs = np.concatenate([np.zeros((1, A)), np.cumsum(R, axis=0)], axis=0)
        # forward sum over (t, t+k]: cs[t+k+1] - cs[t+1]
        for t in range(T - k):
            out[t] = cs[t + k + 1] - cs[t + 1]
    elif target_type == "realized_vol":
        sq = R ** 2
        cs = np.concatenate([np.zeros((1, A)), np.cumsum(sq, axis=0)], axis=0)
        for t in range(T - k):
            out[t] = cs[t + k + 1] - cs[t + 1]
    else:
        raise ValueError(target_type)
    return out


def eval_multitarget_per_horizon(F: np.ndarray, y_per_h: dict,
                                    horizons: list[int], train_frac: float,
                                    alpha: float = 1.0) -> dict:
    """Like eval_multitarget but takes a separately-built target per horizon
    (since cumulative/RV targets depend on k).
    y_per_h[h] is (T_feat, A) target matrix (rows aligned with F)."""
    T = len(F)
    n_train = int(round(train_frac * T))
    out = {"n_features": F.shape[1], "n_train": n_train,
            "kappa": feature_gram_condition(F[:n_train], alpha=alpha)}
    for h in horizons:
        y = y_per_h[h]
        # Find valid rows (not NaN)
        valid = ~np.isnan(y).any(axis=1)
        idx = np.where(valid)[0]
        if len(idx) < n_train + 10:
            out[f"k{h}_mean_nrmse"] = float("nan"); continue
        # split: use first n_train (within valid) for training
        tr_mask = idx < n_train
        te_mask = idx >= n_train
        idx_tr = idx[tr_mask]; idx_te = idx[te_mask]
        F_tr, y_tr = F[idx_tr], y[idx_tr]
        F_te, y_te = F[idx_te], y[idx_te]
        if len(F_tr) < 10 or len(F_te) < 10:
            out[f"k{h}_mean_nrmse"] = float("nan"); continue
        W = ridge_solve(F_tr, y_tr, alpha=alpha)
        y_pred = F_te @ W
        rmse_per = np.sqrt(np.mean((y_pred - y_te) ** 2, axis=0))
        std_per = np.std(y_te, axis=0)
        nrmse_per = rmse_per / (std_per + 1e-12)
        out[f"k{h}_per_target_nrmse"] = nrmse_per.tolist()
        out[f"k{h}_mean_nrmse"] = float(np.mean(nrmse_per))
    return out


# ---------------------------------------------------------------------------
# Synthetic financial-data generator
# ---------------------------------------------------------------------------

def generate_synthetic_returns(n_assets: int, n_steps: int, seed: int,
                                 regime_switch_prob: float = 0.03,
                                 vol_persistence: float = 0.95
                                 ) -> tuple[np.ndarray, np.ndarray]:
    """Multi-asset regime-switching returns with stochastic volatility.

    Returns (R, regime) where R is (n_steps, n_assets) log-returns and
    regime is (n_steps,) hidden regime label in {0,1,2}.

    Regime characteristics (typical of equity markets):
      0 bull:    mu = +0.0005, sigma_base = 0.008, low correlations
      1 bear:    mu = -0.0008, sigma_base = 0.018, mid correlations
      2 crisis:  mu = -0.0020, sigma_base = 0.035, HIGH correlations
    """
    rng = np.random.default_rng(seed)
    # Per-regime drifts (n_regimes x n_assets) and base vols (per asset)
    mu_per_regime = np.array([
        +0.0005 + 0.0002 * rng.standard_normal(n_assets),
        -0.0008 + 0.0003 * rng.standard_normal(n_assets),
        -0.0020 + 0.0005 * rng.standard_normal(n_assets),
    ])
    sigma_base_per_regime = np.array([
        0.008 + 0.001 * rng.uniform(0, 1, n_assets),
        0.018 + 0.002 * rng.uniform(0, 1, n_assets),
        0.035 + 0.005 * rng.uniform(0, 1, n_assets),
    ])
    # Per-regime correlation matrices (low in bull, high in crisis)
    def build_corr(rho):
        C = rho * np.ones((n_assets, n_assets)) + (1 - rho) * np.eye(n_assets)
        return C
    corr_per_regime = [build_corr(0.20), build_corr(0.45), build_corr(0.75)]
    chol_per_regime = [np.linalg.cholesky(C) for C in corr_per_regime]
    # Regime-switching Markov chain (transition matrix)
    P = np.array([
        [1.0 - 2 * regime_switch_prob, regime_switch_prob, regime_switch_prob],
        [regime_switch_prob, 1.0 - 2 * regime_switch_prob, regime_switch_prob],
        [regime_switch_prob, regime_switch_prob, 1.0 - 2 * regime_switch_prob],
    ])
    regime = np.zeros(n_steps, dtype=np.int32)
    regime[0] = 0
    for t in range(1, n_steps):
        regime[t] = rng.choice(3, p=P[regime[t-1]])
    # AR(1) volatility multiplier (per asset, long-memory-ish)
    vol_mult = np.ones((n_steps, n_assets))
    for t in range(1, n_steps):
        innov = rng.standard_normal(n_assets) * 0.15
        vol_mult[t] = vol_persistence * vol_mult[t-1] + (1 - vol_persistence) * 1.0 + innov
    vol_mult = np.clip(vol_mult, 0.3, 4.0)
    # Generate correlated returns
    R = np.zeros((n_steps, n_assets))
    for t in range(n_steps):
        s = int(regime[t])
        eps = chol_per_regime[s] @ rng.standard_normal(n_assets)
        sigma = sigma_base_per_regime[s] * vol_mult[t]
        R[t] = mu_per_regime[s] + sigma * eps
    return R, regime


# ---------------------------------------------------------------------------
# Quantum-state encoding of return vector
# ---------------------------------------------------------------------------

def make_angle_encoder(n_qubits: int, ent_strength: float = 0.5):
    """Build a fixed encoder that maps a real n_qubits-vector r in [-1,1]
    to a quantum state via angle encoding + entangling layer.

      |psi(r)> = U_ent * (tensor_i RY(pi * r_i) |0>)

    U_ent = exp(-i * ent_strength * H_TFIM(n_qubits)) is a single time
    step of TFIM evolution -- creates non-trivial entanglement without
    being maximally scrambling (otherwise input info would be destroyed
    immediately).
    """
    H_ent = tfim_hamiltonian(n_qubits, J=1.0, g=1.0)
    U_ent = sla.expm(-1j * H_ent * ent_strength)
    dim = 2 ** n_qubits

    def encode(r: np.ndarray) -> np.ndarray:
        # Clip r to [-1, 1] (return values typically much smaller; scale up first)
        r_clip = np.clip(r, -1.0, 1.0)
        psi = np.zeros(dim, dtype=complex); psi[0] = 1.0
        for q in range(n_qubits):
            theta = np.pi * r_clip[q]
            c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
            psi = _apply_single_qubit_gate(psi, n_qubits, q,
                                              np.array([[c, -s], [s, c]],
                                                       dtype=complex))
        psi = U_ent @ psi
        return psi

    return encode


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-assets", type=int, default=9)
    ap.add_argument("--n-qubits", type=int, default=11,
                     help="QRC reservoir size (must be >= n-assets)")
    ap.add_argument("--n-steps", type=int, default=2000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--shots-per-qubit", type=int, default=1024)
    ap.add_argument("--reservoir", choices=["tfim", "scram"], default="scram")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--ent-strength", type=float, default=0.5)
    ap.add_argument("--horizons", type=int, nargs="+",
                     default=[1, 5, 10, 20, 40, 80])
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--return-scale", type=float, default=10.0,
                     help="Scale factor applied to raw returns before "
                          "angle encoding (returns are typically ~0.01-0.03; "
                          "scale to ~0.1-0.3 to get usable encoding angles)")
    ap.add_argument("--data-seed", type=int, default=42,
                     help="Seed for synthetic-data generation")
    ap.add_argument("--target-type",
                     choices=["raw_return", "cumulative_return",
                                "realized_vol"],
                     default="cumulative_return",
                     help="raw_return: r_i(t+k) -- mostly noise at all k. "
                          "cumulative_return: sum_{s=1..k} r_i(t+s) -- "
                          "averages noise, exposes regime drift. "
                          "realized_vol: sum_{s=1..k} r_i(t+s)^2 -- the "
                          "volatility forecasting problem, which is "
                          "notoriously easier to predict than direction.")
    ap.add_argument("--out-dir", type=str,
                     default=str(ROOT / "results" / "paper4_synthetic"))
    ap.add_argument("--noise-std", type=float, default=0.01,
                     help="Gaussian noise std added to classical features in "
                          "the *_noisy baselines.  Default 0.01 matches the "
                          "per-feature QRC shot noise at M=q*1024 shots "
                          "(approximate, ~ 1/sqrt(M) ~ 0.009 at q=11).")
    ap.add_argument("--ridge-alpha-strong", type=float, default=100.0,
                     help="Strong ridge regularization for the "
                          "Classical_*_strongreg baselines (default 100x).")
    args = ap.parse_args()

    assert args.n_assets <= args.n_qubits
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Generate synthetic financial data ----
    print(f"=== Paper 4 v0: synthetic multi-asset returns ===")
    print(f"  n_assets={args.n_assets}, n_steps={args.n_steps}, "
            f"data_seed={args.data_seed}")
    R, regime = generate_synthetic_returns(args.n_assets, args.n_steps,
                                              args.data_seed)
    print(f"  returns: mean={R.mean():.5f}, std={R.std():.5f}, "
            f"range [{R.min():.3f}, {R.max():.3f}]")
    print(f"  regime occupancy: " + " ".join(
        [f"{r}={(regime==r).mean():.2f}" for r in range(3)]))

    # Build per-horizon targets aligned to (windowed) feature rows
    print(f"  target_type = {args.target_type}")
    y_per_horizon_full = {h: build_target(R, h, args.target_type)
                            for h in args.horizons}
    for h in args.horizons:
        valid = ~np.isnan(y_per_horizon_full[h]).any(axis=1)
        yv = y_per_horizon_full[h][valid]
        print(f"    h={h:>3}: {valid.sum()} valid rows, "
                f"target std={yv.std():.5f}, "
                f"range [{yv.min():.4f}, {yv.max():.4f}]")

    # ---- Encode returns into quantum states (for QRC) ----
    print(f"\n  Encoding returns into {args.n_assets}-qubit states "
            f"(scale={args.return_scale}, ent_strength={args.ent_strength})...")
    encoder = make_angle_encoder(args.n_assets, ent_strength=args.ent_strength)
    t0 = time.time()
    psi_t = np.array([encoder(args.return_scale * R[t])
                        for t in range(args.n_steps)])
    print(f"  encoded {args.n_steps} states in {time.time()-t0:.1f}s")

    # ---- Per-seed evaluation ----
    method_results = {
        "QRC_state_injection": [],
        "Classical_RFF_direct": [],
        "Classical_ESN_direct": [],
        "Classical_linear_direct": [],
        # Mechanism isolation: is QRC's long-horizon edge from a different
        # feature subspace, or just from shot-noise regularization?
        "Classical_RFF_noisy": [],          # RFF + iid Gaussian noise matched to QRC
        "Classical_ESN_noisy": [],          # ESN + same noise
        "Classical_RFF_strongreg": [],      # RFF with 100x ridge alpha
        "Classical_ESN_strongreg": [],      # ESN with 100x ridge alpha
    }
    feat_dims = {}
    for seed in range(args.n_seeds):
        print(f"\n=== seed {seed} ===")
        rng_qrc = np.random.default_rng(1000 + seed)
        if args.reservoir == "tfim":
            H_res = tfim_hamiltonian(args.n_qubits, J=1.0, g=1.0)
        else:
            H_res = scrambling_hamiltonian(args.n_qubits, seed=seed)

        # QRC
        t0 = time.time()
        qrc_step, qrc_D = make_state_injection_qrc(
            args.n_qubits, args.n_assets, H_res, tau=args.tau,
            shots_per_qubit=args.shots_per_qubit, rng=rng_qrc)
        feats_qrc = np.array([qrc_step(psi_t[t]) for t in range(args.n_steps)])
        feat_dims["QRC_state_injection"] = qrc_D
        Fw_qrc = windowed_features(feats_qrc, args.window)
        y_aligned = {h: y_per_horizon_full[h][args.window:]
                       for h in args.horizons}
        r = eval_multitarget_per_horizon(Fw_qrc, y_aligned, args.horizons,
                                            args.train_frac)
        print(f"  QRC ({time.time()-t0:.1f}s, D={qrc_D}): " + "  ".join(
            [f"k{h}={r[f'k{h}_mean_nrmse']:.3f}" for h in args.horizons]))
        method_results["QRC_state_injection"].append(r)

        D_eff_qrc = args.window * qrc_D

        # Classical direct: windowed raw returns -> RFF
        t0 = time.time()
        Fw_R = windowed_features(R, args.window)
        rff_feats = rff_features(Fw_R, D_eff_qrc, seed)
        r = eval_multitarget_per_horizon(rff_feats, y_aligned, args.horizons,
                                            args.train_frac)
        print(f"  ClassicalRFF ({time.time()-t0:.1f}s, D={D_eff_qrc}): "
                + "  ".join([f"k{h}={r[f'k{h}_mean_nrmse']:.3f}"
                              for h in args.horizons]))
        method_results["Classical_RFF_direct"].append(r)

        # Classical direct: ESN on raw returns
        t0 = time.time()
        W_in, W_res_esn, leak = make_esn(args.n_assets, D_eff_qrc, seed)
        acts = esn_run(R, W_in, W_res_esn, leak)
        Fw_esn = windowed_features(acts, args.window)
        r = eval_multitarget_per_horizon(Fw_esn, y_aligned, args.horizons,
                                            args.train_frac)
        print(f"  ClassicalESN ({time.time()-t0:.1f}s, N={D_eff_qrc}): "
                + "  ".join([f"k{h}={r[f'k{h}_mean_nrmse']:.3f}"
                              for h in args.horizons]))
        method_results["Classical_ESN_direct"].append(r)

        # Classical baseline: linear ridge on windowed raw returns
        t0 = time.time()
        r = eval_multitarget_per_horizon(Fw_R, y_aligned, args.horizons,
                                            args.train_frac)
        print(f"  ClassicalLin ({time.time()-t0:.1f}s, D={Fw_R.shape[1]}): "
                + "  ".join([f"k{h}={r[f'k{h}_mean_nrmse']:.3f}"
                              for h in args.horizons]))
        method_results["Classical_linear_direct"].append(r)

        # === Mechanism isolation baselines ===
        rng_noise = np.random.default_rng(3000 + seed)
        # 1. RFF + matched Gaussian noise (tests: is QRC's edge just shot-noise reg?)
        t0 = time.time()
        rff_noisy = rff_feats + rng_noise.standard_normal(rff_feats.shape) * args.noise_std
        r = eval_multitarget_per_horizon(rff_noisy, y_aligned, args.horizons,
                                            args.train_frac)
        print(f"  RFF+noise   ({time.time()-t0:.1f}s, sigma={args.noise_std}): "
                + "  ".join([f"k{h}={r[f'k{h}_mean_nrmse']:.3f}"
                              for h in args.horizons]))
        method_results["Classical_RFF_noisy"].append(r)

        # 2. ESN + matched Gaussian noise on features
        t0 = time.time()
        esn_noisy = Fw_esn + rng_noise.standard_normal(Fw_esn.shape) * args.noise_std
        r = eval_multitarget_per_horizon(esn_noisy, y_aligned, args.horizons,
                                            args.train_frac)
        print(f"  ESN+noise   ({time.time()-t0:.1f}s, sigma={args.noise_std}): "
                + "  ".join([f"k{h}={r[f'k{h}_mean_nrmse']:.3f}"
                              for h in args.horizons]))
        method_results["Classical_ESN_noisy"].append(r)

        # 3. RFF + strong ridge regularization (tests: is QRC's edge just reg?)
        t0 = time.time()
        r = eval_multitarget_per_horizon(rff_feats, y_aligned, args.horizons,
                                            args.train_frac,
                                            alpha=args.ridge_alpha_strong)
        print(f"  RFF strongreg ({time.time()-t0:.1f}s, alpha={args.ridge_alpha_strong}): "
                + "  ".join([f"k{h}={r[f'k{h}_mean_nrmse']:.3f}"
                              for h in args.horizons]))
        method_results["Classical_RFF_strongreg"].append(r)

        # 4. ESN + strong ridge regularization
        t0 = time.time()
        r = eval_multitarget_per_horizon(Fw_esn, y_aligned, args.horizons,
                                            args.train_frac,
                                            alpha=args.ridge_alpha_strong)
        print(f"  ESN strongreg ({time.time()-t0:.1f}s, alpha={args.ridge_alpha_strong}): "
                + "  ".join([f"k{h}={r[f'k{h}_mean_nrmse']:.3f}"
                              for h in args.horizons]))
        method_results["Classical_ESN_strongreg"].append(r)

    # ---- Summary ----
    summary = {"args": vars(args), "feature_dims": feat_dims,
                "methods": method_results, "per_method_means": {}}
    for method, runs in method_results.items():
        means = {}
        for h in args.horizons:
            vals = [r[f"k{h}_mean_nrmse"] for r in runs]
            means[f"k{h}_mean"] = float(np.nanmean(vals))
            means[f"k{h}_std"] = float(np.nanstd(vals))
        means["kappa_mean"] = float(np.nanmean([r["kappa"] for r in runs]))
        summary["per_method_means"][method] = means
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_dir / 'summary.json'}")

    print(f"\n=== Mean test NRMSE across {args.n_seeds} seeds "
            f"(averaged over {args.n_assets} asset targets) ===")
    print(f"{'method':<26}  " + "  ".join([f"k={h:<5}" for h in args.horizons]))
    for method, m in summary["per_method_means"].items():
        row = f"{method:<26}  " + "  ".join(
            [f"{m[f'k{h}_mean']:.3f}+/-{m[f'k{h}_std']:.3f}"
              for h in args.horizons])
        print(row)


if __name__ == "__main__":
    main()
