#!/usr/bin/env python3
"""
run_scaling_crossover.py - QRC at q in {5..10} vs ESN at fixed N=500
on the 3-variable Lorenz system. Locates the empirical crossover qubit
count at which QRC's exponential feature dimension overtakes the
classical ESN baseline (paper/qst_submission/main.tex, section 5.3).

Outputs:
    results/scaling_crossover/summary.json
    results/scaling_crossover/qrc_vs_esn500.png
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
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Lorenz reference dynamics
# ---------------------------------------------------------------------------

def lorenz_rhs(state, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = state
    return np.array([sigma * (y - x),
                     x * (rho - z) - y,
                     x * y - beta * z])


def integrate(rhs, x0, t_end, dt=0.01):
    t, x = 0.0, np.array(x0, dtype=float)
    traj = [x.copy()]
    while t < t_end - 1e-10:
        k1 = rhs(x); k2 = rhs(x + 0.5 * dt * k1)
        k3 = rhs(x + 0.5 * dt * k2); k4 = rhs(x + dt * k3)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        traj.append(x.copy())
    return np.array(traj)


# ---------------------------------------------------------------------------
# QRC  (mirrors scripts/run_seeds.py encoding exactly)
# ---------------------------------------------------------------------------

def make_qrc_extractor(n_qubits: int, n_layers: int = 2, seed: int = 0):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(seed)
    rot_params = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))

    def extract(state: np.ndarray) -> np.ndarray:
        if len(state) >= n_qubits:
            norm_state = state[:n_qubits].astype(float)
        else:
            norm_state = np.pad(state.astype(float), (0, n_qubits - len(state)))
        denom = norm_state.max() - norm_state.min() + 1e-8
        angles = (norm_state - norm_state.min()) / denom * 2 * np.pi
        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.ry(float(angles[q]), q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qc.rx(float(rot_params[layer, q, 0]), q)
                qc.ry(float(rot_params[layer, q, 1]), q)
                qc.rz(float(rot_params[layer, q, 2]), q)
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)   # linear chain, no ring closure
        sv = Statevector.from_instruction(qc)
        return np.abs(sv.data) ** 2

    return extract


# ---------------------------------------------------------------------------
# ESN at fixed N=500 (mirrors scripts/run_esn_baseline.py pattern)
# ---------------------------------------------------------------------------

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


def _ridge_solve(F, S, alpha):
    A = F.T @ F + alpha * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ S)


# ---------------------------------------------------------------------------
# Per-cell experiment
# ---------------------------------------------------------------------------

def run_qrc_cell(q: int, n_seeds: int, train_states, test_states,
                  window: int = 5, ridge_alpha: float = 1.0):
    mses = []
    for seed in range(n_seeds):
        extract = make_qrc_extractor(q, n_layers=2, seed=seed)
        feats_train = np.array([extract(s) for s in train_states])
        F_tr, S_tr = [], []
        for i in range(window, len(train_states)):
            F_tr.append(feats_train[i - window:i].flatten())
            S_tr.append(train_states[i])
        F_tr = np.asarray(F_tr); S_tr = np.asarray(S_tr)
        W = _ridge_solve(F_tr, S_tr, ridge_alpha)
        feats_test = np.array([extract(s) for s in test_states])
        buf = list(feats_train[-window:])
        preds = []
        for feat in feats_test:
            window_flat = np.concatenate(buf[-window:])
            preds.append(window_flat @ W)
            buf.append(feat)
        mses.append(float(np.mean((np.array(preds) - test_states) ** 2)))
    return np.array(mses)


def run_esn500_cell(n_seeds: int, train_states, test_states,
                     N_res: int = 500, window: int = 5,
                     ridge_alpha: float = 1.0):
    mses = []
    for seed in range(n_seeds):
        W_in, W_res, alpha = make_esn(3, N_res, seed)
        train_acts = _esn_run(train_states, W_in, W_res, alpha)
        F_tr, S_tr = [], []
        for i in range(window, len(train_states)):
            F_tr.append(train_acts[i - window:i].flatten())
            S_tr.append(train_states[i])
        F_tr = np.asarray(F_tr); S_tr = np.asarray(S_tr)
        W_e = _ridge_solve(F_tr, S_tr, ridge_alpha)
        full_states = np.concatenate([train_states, test_states])
        full_acts = _esn_run(full_states, W_in, W_res, alpha)
        preds = []
        for i in range(len(train_states), len(full_states)):
            window_flat = full_acts[i - window:i].flatten()
            preds.append(window_flat @ W_e)
        mses.append(float(np.mean((np.array(preds) - test_states) ** 2)))
    return np.array(mses)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qs", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "scaling_crossover"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"

    # Reference trajectory (same protocol as run_seeds.py)
    dt, t_train_end, t_test_end = 0.01, 3.0, 4.0
    n_train, n_test = 50, 20
    x0 = np.array([1.0, 1.0, 1.0])
    full_traj = integrate(lorenz_rhs, x0, t_test_end, dt=dt)
    train_idx = np.linspace(0, int(t_train_end / dt) - 1, n_train, dtype=int)
    test_idx  = np.linspace(int(t_train_end / dt),
                              int(t_test_end / dt) - 1, n_test, dtype=int)
    train_states = full_traj[train_idx]
    test_states  = full_traj[test_idx]

    # ESN at fixed N=500 (q-independent, run once)
    print("=== ESN baseline (N=500) ===")
    t0 = time.time()
    esn_mses = run_esn500_cell(args.n_seeds, train_states, test_states)
    print(f"  test MSE: {esn_mses.mean():.3e} +/- {esn_mses.std():.3e}  "
           f"({time.time() - t0:.1f}s)")
    esn_mean, esn_std = float(esn_mses.mean()), float(esn_mses.std())

    # QRC across qubit counts
    rows = []
    for q in args.qs:
        t0 = time.time()
        qrc_mses = run_qrc_cell(q, args.n_seeds, train_states, test_states)
        elapsed = time.time() - t0
        wins = int(np.sum(qrc_mses < esn_mses))
        p_value = float(stats.wilcoxon(qrc_mses, esn_mses).pvalue) \
                    if not np.allclose(qrc_mses, esn_mses) else float("nan")
        winner = "QRC" if qrc_mses.mean() < esn_mean else "ESN"
        row = {
            "q": q, "feature_dim": 2 ** q,
            "qrc_mean": float(qrc_mses.mean()),
            "qrc_std": float(qrc_mses.std()),
            "qrc_per_seed": qrc_mses.tolist(),
            "esn_wins_below_qrc": args.n_seeds - wins,
            "qrc_wins_below_esn": wins,
            "wilcoxon_qrc_vs_esn": p_value,
            "wall_time_s": elapsed,
        }
        rows.append(row)
        print(f"q={q:>2}  feat={2**q:>5}  QRC={row['qrc_mean']:.3e}+/-{row['qrc_std']:.3e}  "
              f"vs ESN={esn_mean:.3e}  QRC<ESN={wins}/{args.n_seeds}  "
              f"p={p_value:.4f}  lower-mean: {winner}  ({elapsed:.1f}s)")
        with open(out_json, "w") as f:
            json.dump({
                "n_seeds": args.n_seeds,
                "esn_N": 500,
                "esn_mean": esn_mean,
                "esn_std": esn_std,
                "esn_per_seed": esn_mses.tolist(),
                "rows": rows,
            }, f, indent=2)
    print(f"\nSaved {out_json}")

    # ----- Plot -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        qs    = [r["q"] for r in rows]
        means = [r["qrc_mean"] for r in rows]
        stds  = [r["qrc_std"]  for r in rows]
        ax.errorbar(qs, means, yerr=stds, marker="o", lw=2, ms=8, capsize=4,
                     label=r"QRC at $q$ qubits (feature dim $2^q$)",
                     color="#0F4C5C", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.axhline(esn_mean, color="#9A4836", lw=2, ls="--",
                    label=fr"ESN at $N=500$  (mean test MSE = {esn_mean:.2e})")
        ax.fill_between(qs, esn_mean - esn_std, esn_mean + esn_std,
                         color="#9A4836", alpha=0.15)
        for r in rows:
            ax.annotate(f"p={r['wilcoxon_qrc_vs_esn']:.3f}",
                         (r["q"], r["qrc_mean"]),
                         xytext=(0, 10), textcoords="offset points",
                         ha="center", fontsize=8, color="#444")
        ax.set_xlabel("Qubit count  q   (QRC feature dim = $2^q$)")
        ax.set_ylabel(f"Test teacher MSE  (mean +/- std over {args.n_seeds} seeds)")
        ax.set_title("QRC scaling vs ESN(N=500) on Lorenz: crossover qubit count")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out_fig = out_dir / "qrc_vs_esn500.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
