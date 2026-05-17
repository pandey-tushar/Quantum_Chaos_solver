#!/usr/bin/env python3
"""
run_chaoticity_sweep.py - Vary the Lorenz-96 forcing F to sweep the
system's dynamical regime from the stable fixed point at low forcing
through weakly chaotic to fully chaotic. For each cell, compute the
empirical maximum Lyapunov exponent (Benettin's algorithm) and run
the QRC and the matched-feature ESN baselines on the same data with
the same protocol as scripts/run_l96_kq_scaling.py at q = N = 8.

The headline output is test-MSE-vs-Lyapunov-exponent: it should show
QRC and ESN both performing well in the non-chaotic regime, with
the QRC advantage emerging as lambda_max grows.

Outputs:
    results/chaoticity_l96/sweep_summary.json
    results/chaoticity_l96/mse_vs_lyapunov.png
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
# Lorenz-96 system  (vectorised RHS for speed)
# ---------------------------------------------------------------------------

def l96_rhs(x: np.ndarray, F: float = 8.0) -> np.ndarray:
    """Vectorised Lorenz-96 right-hand side with periodic indexing."""
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def integrate_l96(x0: np.ndarray, t_end: float, dt: float,
                   F: float = 8.0) -> np.ndarray:
    t, x = 0.0, np.array(x0, dtype=float)
    traj = [x.copy()]
    while t < t_end - 1e-10:
        k1 = l96_rhs(x,            F)
        k2 = l96_rhs(x + 0.5*dt*k1, F)
        k3 = l96_rhs(x + 0.5*dt*k2, F)
        k4 = l96_rhs(x + dt*k3,    F)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        traj.append(x.copy())
    return np.array(traj)


# ---------------------------------------------------------------------------
# Maximum Lyapunov exponent  (Benettin's algorithm)
# ---------------------------------------------------------------------------

def lyapunov_max(F: float, N: int = 8, delta0: float = 1e-8,
                  tau: float = 0.5, n_steps: int = 1000,
                  t_spinup: float = 30.0, dt: float = 0.01,
                  x0: np.ndarray | None = None) -> float:
    """Estimate the maximum Lyapunov exponent of L96 at forcing F via
    Benettin's renormalisation algorithm."""
    if x0 is None:
        x0 = np.zeros(N); x0[0] = 0.01

    # Spin up onto the attractor
    spinup = integrate_l96(x0, t_spinup, dt, F)
    x = spinup[-1].copy()

    delta = np.zeros(N); delta[0] = delta0
    log_growth = []
    for _ in range(n_steps):
        ref  = integrate_l96(x,         tau, dt, F)[-1]
        pert = integrate_l96(x + delta, tau, dt, F)[-1]
        sep = pert - ref
        d = float(np.linalg.norm(sep))
        if d <= 0.0 or not np.isfinite(d):
            break
        log_growth.append(np.log(d / delta0))
        delta = (delta0 / d) * sep   # renormalise back to delta0
        x = ref

    if len(log_growth) == 0:
        return float("nan")
    return float(np.mean(log_growth) / tau)


# ---------------------------------------------------------------------------
# QRC and ESN  (mirrors scripts/run_l96_kq_scaling.py exactly so the
#               implementation is the same one validated by the K=q sweep)
# ---------------------------------------------------------------------------

def make_qrc_extractor(n_qubits: int, n_layers: int, seed: int):
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    rng = np.random.default_rng(seed)
    rot_params = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))

    def extract(state: np.ndarray) -> np.ndarray:
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
                qc.cx(q, q + 1)   # linear chain, no ring closure
        sv = Statevector.from_instruction(qc)
        return np.abs(sv.data) ** 2

    return extract


def _esn_run(states: np.ndarray, W_in: np.ndarray, W_res: np.ndarray,
             alpha: float) -> np.ndarray:
    T = len(states)
    n_res = W_res.shape[0]
    r = np.zeros(n_res)
    out = np.zeros((T, n_res))
    for t, u in enumerate(states):
        r = (1 - alpha) * r + alpha * np.tanh(W_res @ r + W_in @ u)
        out[t] = r
    return out


def make_esn(n_input: int, n_reservoir: int, seed: int,
              spectral_radius: float = 0.9, input_scaling: float = 0.1,
              leaking_rate: float = 0.3):
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


def _ridge_solve(F: np.ndarray, S: np.ndarray, alpha: float) -> np.ndarray:
    A = F.T @ F + alpha * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ S)


# ---------------------------------------------------------------------------
# Single-cell experiment
# ---------------------------------------------------------------------------

def run_cell(F: float, N: int = 8, n_seeds: int = 8, n_layers: int = 2,
              window: int = 5, ridge_alpha: float = 1.0, dt: float = 0.01,
              t_train_end: float = 3.0, t_test_end: float = 4.0,
              n_train: int = 50, n_test: int = 20,
              compute_lyap: bool = True):
    n_qubits = N
    feature_dim = 2 ** n_qubits

    # Reference trajectory (deterministic, shared across seeds)
    x0 = np.zeros(N); x0[0] = 0.01
    full_traj = integrate_l96(x0, t_test_end, dt, F)
    train_idx = np.linspace(0, int(t_train_end / dt) - 1, n_train, dtype=int)
    test_idx  = np.linspace(int(t_train_end / dt),
                              int(t_test_end / dt) - 1, n_test, dtype=int)
    train_states = full_traj[train_idx, :N]
    test_states  = full_traj[test_idx, :N]

    # Lyapunov estimate (single number per cell, expensive)
    lyap = float("nan")
    if compute_lyap:
        try:
            lyap = lyapunov_max(F, N=N)
        except Exception as e:
            print(f"  (Lyapunov failed for F={F}: {e})")

    qrc_mses, esn_mses, pers_mses = [], [], []
    for seed in range(n_seeds):
        # ---- QRC ----
        extract = make_qrc_extractor(n_qubits, n_layers, seed)
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
        qrc_mses.append(float(np.mean((np.array(preds) - test_states) ** 2)))

        # ---- ESN at matched feature dim ----
        W_in, W_res, alpha = make_esn(N, feature_dim, seed)
        train_acts = _esn_run(train_states, W_in, W_res, alpha)
        F_tr_e = []
        for i in range(window, len(train_states)):
            F_tr_e.append(train_acts[i - window:i].flatten())
        F_tr_e = np.asarray(F_tr_e)
        W_e = _ridge_solve(F_tr_e, S_tr, ridge_alpha)
        full_states = np.concatenate([train_states, test_states])
        full_acts = _esn_run(full_states, W_in, W_res, alpha)
        preds_e = []
        for i in range(len(train_states), len(full_states)):
            window_flat = full_acts[i - window:i].flatten()
            preds_e.append(window_flat @ W_e)
        esn_mses.append(float(np.mean((np.array(preds_e) - test_states) ** 2)))

        # ---- Persistence ----
        pers_preds = np.tile(train_states[-1], (len(test_states), 1))
        pers_mses.append(float(np.mean((pers_preds - test_states) ** 2)))

    qrc_v = np.array(qrc_mses); esn_v = np.array(esn_mses); pers_v = np.array(pers_mses)
    return {
        "F": F, "N": N, "feature_dim": feature_dim,
        "lyapunov_max": lyap,
        "qrc_mean": float(qrc_v.mean()), "qrc_std": float(qrc_v.std()),
        "esn_mean": float(esn_v.mean()), "esn_std": float(esn_v.std()),
        "pers_mean": float(pers_v.mean()), "pers_std": float(pers_v.std()),
        "qrc_per_seed": qrc_v.tolist(),
        "esn_per_seed": esn_v.tolist(),
        "qrc_wins_vs_esn": int(np.sum(qrc_v < esn_v)),
        "wilcoxon_qrc_vs_esn": float(stats.wilcoxon(qrc_v, esn_v).pvalue)
                                if not np.allclose(qrc_v, esn_v) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Fs", type=float, nargs="+",
                    default=[0.5, 1.0, 1.5, 2.0, 4.0, 8.0, 16.0],
                    help="L96 forcing values to sweep. Defaults span "
                         "stable fixed point through weak to full chaos.")
    p.add_argument("--N", type=int, default=8,
                    help="L96 system dimension (= qubit count, matched).")
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "chaoticity_l96"))
    p.add_argument("--skip-lyap", action="store_true",
                    help="Skip Lyapunov computation (faster smoke test)")
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "sweep_summary.json"
    rows = []
    for F in args.Fs:
        t0 = time.time()
        cell = run_cell(F=F, N=args.N, n_seeds=args.n_seeds,
                         compute_lyap=not args.skip_lyap)
        cell["wall_time_s"] = time.time() - t0
        rows.append(cell)
        winner = "QRC" if cell["qrc_mean"] < cell["esn_mean"] else "ESN"
        regime = ("chaotic" if cell["lyapunov_max"] > 0.05 else
                   ("marginal" if cell["lyapunov_max"] > -0.05 else "stable"))
        print(f"F={F:>5.1f}  lambda_max={cell['lyapunov_max']:>+6.3f}  "
              f"({regime:>8s})  | "
              f"QRC={cell['qrc_mean']:.3e}+/-{cell['qrc_std']:.3e}  "
              f"ESN={cell['esn_mean']:.3e}+/-{cell['esn_std']:.3e}  "
              f"pers={cell['pers_mean']:.3e}  "
              f"QRC<ESN={cell['qrc_wins_vs_esn']}/{args.n_seeds}  "
              f"p={cell['wilcoxon_qrc_vs_esn']:.4f}  "
              f"lower-mean: {winner}  ({cell['wall_time_s']:.1f}s)")
        with open(out_json, "w") as f:
            json.dump({"n_seeds": args.n_seeds, "N": args.N, "rows": rows}, f, indent=2)

    print(f"\nSaved {out_json}")

    # ----- Plot: MSE vs Lyapunov exponent -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rows_sorted = sorted(rows, key=lambda r: r["lyapunov_max"]
                              if not np.isnan(r["lyapunov_max"]) else 0.0)
        lyaps = np.array([r["lyapunov_max"] for r in rows_sorted])
        Fs    = [r["F"] for r in rows_sorted]

        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        ax.errorbar(lyaps,
                     [r["qrc_mean"] for r in rows_sorted],
                     yerr=[r["qrc_std"] for r in rows_sorted],
                     marker="o", lw=2, ms=8, capsize=4,
                     label=f"QRC ({args.N} qubits)",
                     color="#0F4C5C", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.errorbar(lyaps,
                     [r["esn_mean"] for r in rows_sorted],
                     yerr=[r["esn_std"] for r in rows_sorted],
                     marker="s", lw=2, ms=8, capsize=4,
                     label=f"ESN (N={2**args.N})",
                     color="#9A4836", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.plot(lyaps,
                 [r["pers_mean"] for r in rows_sorted],
                 marker="X", ms=8, lw=1.4, ls=":",
                 label="Persistence", color="#888")
        ax.axvline(0.0, color="#888", lw=0.7, ls="--", alpha=0.6)
        for lyap, F in zip(lyaps, Fs):
            ax.annotate(rf"$F={F:g}$", (lyap, 0.0),
                         xytext=(0, -18), textcoords="offset points",
                         ha="center", fontsize=8, color="#444")
        ax.set_xlabel(r"empirical max Lyapunov exponent  $\lambda_{\max}$")
        ax.set_ylabel(f"Test teacher MSE  (mean +/- std over {args.n_seeds} seeds)")
        ax.set_title(f"Lorenz-96 (N={args.N}) chaoticity sweep:  "
                       f"prediction error vs dynamical regime")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out_fig = out_dir / "mse_vs_lyapunov.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
