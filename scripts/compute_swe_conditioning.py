#!/usr/bin/env python3
"""
compute_swe_conditioning.py - Direct measurement of the feature-Gram
condition number kappa(F^T F + alpha I) at each K=q cell of the SWE
Block A pipeline, for both QRC and ESN reservoirs.  Parallels
compute_kq_conditioning.py (which does the same for L96) so the
paper-2 conditioning argument is supported by direct measurement on
both test systems.

Loads SWE data from data/swe/swe_data.npz, POD-reduces the reference
trajectory to K modes, and builds windowed feature matrices for each
reservoir + each seed.  Reports kappa(F^T F + alpha I) on the
training-set features used in the ridge solve.

Outputs:
    results/swe_qrc/conditioning.json
    results/swe_qrc/conditioning_vs_K.png
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


# ---------------------------------------------------------------------------
# QRC and ESN  (mirrors scripts/run_swe_qrc.py global-encoding pipeline)
# ---------------------------------------------------------------------------

def make_qrc_extractor(n_qubits, n_layers, seed, scale_lo, scale_hi):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(seed)
    rot_params = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))

    def _pad(a):
        a = np.asarray(a, dtype=float)
        if len(a) >= n_qubits:
            return a[:n_qubits]
        return np.concatenate([a, np.zeros(n_qubits - len(a))])

    lo = _pad(scale_lo); hi = _pad(scale_hi)
    span = np.where(hi - lo > 1e-12, hi - lo, 1.0)

    def extract(state):
        if len(state) >= n_qubits:
            x = state[:n_qubits].astype(float)
        else:
            x = np.concatenate([state.astype(float),
                                 np.zeros(n_qubits - len(state))])
        x = np.clip((x - lo) / span, 0.0, 1.0) * 2.0 * np.pi
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.ry(float(x[q]), q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qc.rx(float(rot_params[layer, q, 0]), q)
                qc.ry(float(rot_params[layer, q, 1]), q)
                qc.rz(float(rot_params[layer, q, 2]), q)
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)
        sv = Statevector.from_instruction(qc)
        return np.abs(sv.data) ** 2

    return extract


def _esn_run(states, W_in, W_res, alpha):
    T = len(states); n_res = W_res.shape[0]
    r = np.zeros(n_res); out = np.zeros((T, n_res))
    for t, u in enumerate(states):
        r = (1 - alpha) * r + alpha * np.tanh(W_res @ r + W_in @ u)
        out[t] = r
    return out


def make_esn(n_input, n_reservoir, seed,
              spectral_radius=0.9, input_scaling=0.1, leaking_rate=0.3):
    rng = np.random.default_rng(seed)
    W_in = rng.uniform(-input_scaling, input_scaling, (n_reservoir, n_input))
    density = 0.1
    W = rng.standard_normal((n_reservoir, n_reservoir))
    W *= (rng.uniform(0, 1, (n_reservoir, n_reservoir)) < density)
    eigs = np.linalg.eigvals(W)
    sr = float(np.max(np.abs(eigs)))
    if sr > 1e-8:
        W *= spectral_radius / sr
    return W_in, W, leaking_rate


def feature_gram_condition(F: np.ndarray, alpha: float = 1.0) -> dict:
    """Compute kappa(F^T F + alpha I) using SVD of F."""
    svals = np.linalg.svd(F, compute_uv=False)
    n_rows, n_cols = F.shape
    eigs = svals ** 2 + alpha
    if n_cols > n_rows:
        eigs = np.concatenate([eigs, np.full(n_cols - n_rows, alpha)])
    return {
        "kappa": float(eigs.max() / eigs.min()),
        "eig_max": float(eigs.max()),
        "eig_min": float(eigs.min()),
        "n_rows": int(n_rows),
        "n_cols": int(n_cols),
    }


# ---------------------------------------------------------------------------
# Per-cell measurement
# ---------------------------------------------------------------------------

def run_cell(K: int, ref_coeffs: np.ndarray, scale_lo: np.ndarray,
              scale_hi: np.ndarray, n_seeds: int, n_layers: int = 2,
              window: int = 5, ridge_alpha: float = 1.0,
              train_frac: float = 0.7):
    feature_dim = 2 ** K
    n_t = len(ref_coeffs)
    n_train = int(round(train_frac * n_t))
    train_coeffs = ref_coeffs[:n_train]

    qrc_kappas, esn_kappas = [], []
    for seed in range(n_seeds):
        # ---- QRC feature matrix ----
        extract = make_qrc_extractor(K, n_layers, seed,
                                       scale_lo=scale_lo, scale_hi=scale_hi)
        feats_train = np.array([extract(s) for s in train_coeffs])
        F_qrc = []
        for i in range(window, len(train_coeffs)):
            F_qrc.append(feats_train[i - window:i].flatten())
        F_qrc = np.asarray(F_qrc)
        qrc_kappas.append(feature_gram_condition(F_qrc, ridge_alpha))

        # ---- ESN feature matrix ----
        W_in, W_res, alpha = make_esn(K, feature_dim, seed)
        train_acts = _esn_run(train_coeffs, W_in, W_res, alpha)
        F_esn = []
        for i in range(window, len(train_coeffs)):
            F_esn.append(train_acts[i - window:i].flatten())
        F_esn = np.asarray(F_esn)
        esn_kappas.append(feature_gram_condition(F_esn, ridge_alpha))

    qrc_v = np.array([r["kappa"] for r in qrc_kappas])
    esn_v = np.array([r["kappa"] for r in esn_kappas])
    return {
        "K": K, "feature_dim": feature_dim,
        "n_train_samples": int(n_train),
        "qrc_kappa_mean": float(qrc_v.mean()),
        "qrc_kappa_std":  float(qrc_v.std()),
        "qrc_kappa_per_seed": qrc_v.tolist(),
        "esn_kappa_mean": float(esn_v.mean()),
        "esn_kappa_std":  float(esn_v.std()),
        "esn_kappa_per_seed": esn_v.tolist(),
        "qrc_details": qrc_kappas,
        "esn_details": esn_kappas,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ks", type=int, nargs="+", default=[5, 6, 7, 8])
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--data", type=str, default=str(ROOT / "data" / "swe" / "swe_data.npz"))
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "swe_qrc"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "conditioning.json"

    # Load SWE data and fit POD on the reference trajectory
    d = np.load(args.data)
    ref_snaps = d["ref_snaps"]      # (n_t, 3, Nx, Ny)
    n_t = ref_snaps.shape[0]
    print(f"Loaded {n_t} reference snapshots; ref_snaps.shape = {ref_snaps.shape}")

    # POD fit on reference; retain max requested K modes for projection
    K_max = max(args.Ks)
    pod = PODReducer(n_modes=K_max).fit(ref_snaps)
    ref_coeffs = pod.transform(ref_snaps)    # (n_t, K_max)
    print(f"POD fit done; ref coeffs shape = {ref_coeffs.shape}")

    # Global encoding bounds (training portion only, per-component min/max)
    train_frac = 0.7
    n_train_global = int(round(train_frac * n_t))
    scale_lo = ref_coeffs[:n_train_global, :K_max].min(axis=0)
    scale_hi = ref_coeffs[:n_train_global, :K_max].max(axis=0)

    rows = []
    for K in args.Ks:
        t0 = time.time()
        # Slice coeffs to first K modes for this cell
        cell_coeffs = ref_coeffs[:, :K]
        cell_lo = scale_lo[:K]
        cell_hi = scale_hi[:K]
        cell = run_cell(K=K, ref_coeffs=cell_coeffs,
                          scale_lo=cell_lo, scale_hi=cell_hi,
                          n_seeds=args.n_seeds)
        cell["wall_time_s"] = time.time() - t0
        rows.append(cell)
        ratio = cell["esn_kappa_mean"] / max(cell["qrc_kappa_mean"], 1.0)
        print(f"K={K:>2}  feat={cell['feature_dim']:>4} | "
              f"kappa_QRC={cell['qrc_kappa_mean']:.3e}+/-{cell['qrc_kappa_std']:.3e}  "
              f"kappa_ESN={cell['esn_kappa_mean']:.3e}+/-{cell['esn_kappa_std']:.3e}  "
              f"ratio={ratio:.2e}  ({cell['wall_time_s']:.1f}s)")
        with open(out_json, "w") as f:
            json.dump({"n_seeds": args.n_seeds, "rows": rows}, f, indent=2)
    print(f"\nSaved {out_json}")

    # ----- Plot kappa vs K -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        Ks = [r["K"] for r in rows]
        ax.errorbar(Ks, [r["qrc_kappa_mean"] for r in rows],
                     yerr=[r["qrc_kappa_std"] for r in rows],
                     marker="o", lw=2, ms=8, capsize=4,
                     label=r"QRC features  $\kappa(F^TF + \alpha I)$",
                     color="#0F4C5C", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.errorbar(Ks, [r["esn_kappa_mean"] for r in rows],
                     yerr=[r["esn_kappa_std"] for r in rows],
                     marker="s", lw=2, ms=8, capsize=4,
                     label=r"ESN features  $\kappa(F^TF + \alpha I)$",
                     color="#9A4836", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.set_xlabel(r"SWE POD modes  $K = q$  (matched complexity)")
        ax.set_ylabel(r"Feature-Gram condition number  $\kappa$")
        ax.set_title("Feature-Gram conditioning vs K (SWE Block A)")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out_fig = out_dir / "conditioning_vs_K.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
