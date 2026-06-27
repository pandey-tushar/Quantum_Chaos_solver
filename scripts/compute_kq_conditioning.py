#!/usr/bin/env python3
"""
compute_kq_conditioning.py - Direct measurement of the feature-Gram
condition number kappa(F^T F + alpha I) at each (N, seed) cell of the
L96 K=q sweep, for both QRC and ESN reservoirs. Upgrades the
conditioning argument of paper 2 from conjecture to evidence.

Outputs:
    results/l96_kq/conditioning.json
    results/l96_kq/conditioning_vs_N.png
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


# ---------------------------------------------------------------------------
# Lorenz-96
# ---------------------------------------------------------------------------

def l96_rhs(x, F=8.0):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def integrate_l96(x0, t_end, dt=0.01, F=8.0):
    t, x = 0.0, np.array(x0, dtype=float)
    traj = [x.copy()]
    while t < t_end - 1e-10:
        k1 = l96_rhs(x, F); k2 = l96_rhs(x + 0.5 * dt * k1, F)
        k3 = l96_rhs(x + 0.5 * dt * k2, F); k4 = l96_rhs(x + dt * k3, F)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        traj.append(x.copy())
    return np.array(traj)


# ---------------------------------------------------------------------------
# QRC and ESN  (mirror scripts/run_l96_kq_scaling.py exactly)
# ---------------------------------------------------------------------------

def make_qrc_extractor(n_qubits, n_layers, seed):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(seed)
    rot_params = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))

    def extract(state):
        x = state[:n_qubits] if len(state) >= n_qubits else np.pad(
            state, (0, n_qubits - len(state)))
        x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2 * np.pi
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
    """Compute kappa(F^T F + alpha I) along with the singular spectrum."""
    # Use SVD on F directly: sigma_i(F^T F + alpha I) = sigma_i(F)^2 + alpha
    svals = np.linalg.svd(F, compute_uv=False)
    # Eigenvalues of F^T F + alpha I are sigma^2 + alpha; if F has more cols than
    # rows, the remaining eigenvalues are exactly alpha (zero singular values).
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
        "top_sigmas": [float(s) for s in svals[:5].tolist()],
    }


# ---------------------------------------------------------------------------
# Per-cell measurement
# ---------------------------------------------------------------------------

def run_cell(N: int, n_seeds: int, n_layers: int = 2, window: int = 5,
              ridge_alpha: float = 1.0, dt: float = 0.01,
              t_train_end: float = 3.0, t_test_end: float = 4.0,
              n_train: int = 50, F_l96: float = 8.0):
    feature_dim = 2 ** N
    x0 = np.zeros(N); x0[0] = 0.01
    full_traj = integrate_l96(x0, t_test_end, dt=dt, F=F_l96)
    train_idx = np.linspace(0, int(t_train_end / dt) - 1, n_train, dtype=int)
    train_states = full_traj[train_idx, :N]

    qrc_kappas, esn_kappas = [], []
    for seed in range(n_seeds):
        # ---- QRC feature matrix ----
        extract = make_qrc_extractor(N, n_layers, seed)
        feats_train = np.array([extract(s) for s in train_states])
        F_qrc = []
        for i in range(window, len(train_states)):
            F_qrc.append(feats_train[i - window:i].flatten())
        F_qrc = np.asarray(F_qrc)
        qrc_kappas.append(feature_gram_condition(F_qrc, ridge_alpha))

        # ---- ESN feature matrix ----
        W_in, W_res, alpha = make_esn(N, feature_dim, seed)
        train_acts = _esn_run(train_states, W_in, W_res, alpha)
        F_esn = []
        for i in range(window, len(train_states)):
            F_esn.append(train_acts[i - window:i].flatten())
        F_esn = np.asarray(F_esn)
        esn_kappas.append(feature_gram_condition(F_esn, ridge_alpha))

    qrc_v = np.array([r["kappa"] for r in qrc_kappas])
    esn_v = np.array([r["kappa"] for r in esn_kappas])
    return {
        "N": N, "feature_dim": feature_dim,
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
    p.add_argument("--Ns", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10, 11])
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "l96_kq"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "conditioning.json"

    rows = []
    for N in args.Ns:
        t0 = time.time()
        cell = run_cell(N=N, n_seeds=args.n_seeds)
        cell["wall_time_s"] = time.time() - t0
        rows.append(cell)
        ratio = cell["esn_kappa_mean"] / max(cell["qrc_kappa_mean"], 1.0)
        print(f"N={N:>2}  feat={cell['feature_dim']:>4} | "
              f"kappa_QRC={cell['qrc_kappa_mean']:.3e}+/-{cell['qrc_kappa_std']:.3e}  "
              f"kappa_ESN={cell['esn_kappa_mean']:.3e}+/-{cell['esn_kappa_std']:.3e}  "
              f"ratio={ratio:.2e}  ({cell['wall_time_s']:.1f}s)")
        with open(out_json, "w") as f:
            json.dump({"n_seeds": args.n_seeds, "rows": rows}, f, indent=2)
    print(f"\nSaved {out_json}")

    # ----- Plot kappa vs N -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        Ns = [r["N"] for r in rows]
        ax.errorbar(Ns, [r["qrc_kappa_mean"] for r in rows],
                     yerr=[r["qrc_kappa_std"] for r in rows],
                     marker="o", lw=2, ms=8, capsize=4,
                     label=r"QRC features  $\kappa(F^TF + \alpha I)$",
                     color="#0F4C5C", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.errorbar(Ns, [r["esn_kappa_mean"] for r in rows],
                     yerr=[r["esn_kappa_std"] for r in rows],
                     marker="s", lw=2, ms=8, capsize=4,
                     label=r"ESN features  $\kappa(F^TF + \alpha I)$",
                     color="#9A4836", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.set_xlabel(r"L96 dimension  $N = q$  (matched complexity)")
        ax.set_ylabel(r"Feature-Gram condition number  $\kappa$")
        ax.set_title("Feature-Gram conditioning vs system size (L96 K=q sweep)")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out_fig = out_dir / "conditioning_vs_N.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
