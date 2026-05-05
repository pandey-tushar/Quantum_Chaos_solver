#!/usr/bin/env python3
"""
run_l96_kq_scaling.py - K=q matched-complexity scaling on Lorenz-96.

Mirrors the SWE Block A K=q scaling experiment but on the Lorenz-96
chaotic system, where the system dimension N is itself the
problem-complexity knob -- each variable is a genuine degree of
freedom, no POD bookkeeping. We sweep q = K = N over {5, 6, 7, 8, 9, 10}
and compare:

    * QRC (q qubits, fixed Hamiltonian reservoir, one L96 variable
      encoded per qubit, ridge readout with temporal window)
    * Classical ESN at matched feature dim 2^q
    * Persistence baseline

Same QRC encoder, window, ridge alpha as scripts/run_seeds.py at q=8
to make this a clean methodological extension of arXiv 2604.23743's
L96 result. This script reuses none of that code directly to keep the
experiment self-contained and auditable, but the configuration matches.

Outputs:
    results/l96_kq/scaling_summary.json
    results/l96_kq/skill_lead1_vs_kq.png
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
# Lorenz-96 system
# ---------------------------------------------------------------------------

def lorenz96_rhs(state: np.ndarray, F: float = 8.0) -> np.ndarray:
    N = len(state)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = ((state[(i + 1) % N] - state[(i - 2) % N])
                   * state[(i - 1) % N] - state[i] + F)
    return dxdt


def integrate_l96(x0: np.ndarray, t_end: float, dt: float = 0.01,
                   F: float = 8.0) -> np.ndarray:
    t, x = 0.0, np.array(x0, dtype=float)
    traj = [x.copy()]
    while t < t_end - 1e-10:
        k1 = lorenz96_rhs(x, F)
        k2 = lorenz96_rhs(x + 0.5 * dt * k1, F)
        k3 = lorenz96_rhs(x + 0.5 * dt * k2, F)
        k4 = lorenz96_rhs(x + dt * k3, F)
        x = x + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        t += dt
        traj.append(x.copy())
    return np.array(traj)


# ---------------------------------------------------------------------------
# QRC and ESN (vendored to keep the script self-contained)
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
                qc.cx(q, q + 1)
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

def run_cell(N: int, n_seeds: int, n_layers: int = 2, window: int = 5,
              ridge_alpha: float = 1.0, dt: float = 0.01,
              t_train_end: float = 3.0, t_test_end: float = 4.0,
              n_train: int = 50, n_test: int = 20,
              F_l96: float = 8.0):
    feature_dim = 2 ** N
    # Build reference trajectory once (deterministic)
    x0 = np.zeros(N)
    x0[0] = 0.01
    full_traj = integrate_l96(x0, t_test_end, dt=dt, F=F_l96)

    train_idx = np.linspace(0, int(t_train_end / dt) - 1, n_train, dtype=int)
    test_idx  = np.linspace(int(t_train_end / dt),
                              int(t_test_end / dt) - 1, n_test, dtype=int)
    train_states = full_traj[train_idx, :N]
    test_states  = full_traj[test_idx, :N]

    qrc_mses, esn_mses, pers_mses = [], [], []
    for seed in range(n_seeds):
        # ---- QRC ----
        extract = make_qrc_extractor(N, n_layers, seed)
        feats_train = np.array([extract(s) for s in train_states])
        F_tr = []
        S_tr = []
        for i in range(window, len(train_states)):
            F_tr.append(feats_train[i - window:i].flatten())
            S_tr.append(train_states[i])
        F_tr = np.asarray(F_tr); S_tr = np.asarray(S_tr)
        W = _ridge_solve(F_tr, S_tr, ridge_alpha)

        # Test (teacher forcing -- mirrors run_seeds.py)
        feats_test = np.array([extract(s) for s in test_states])
        buf = list(feats_train[-window:])
        preds = []
        for feat in feats_test:
            window_flat = np.concatenate(buf[-window:])
            preds.append(window_flat @ W)
            buf.append(feat)
        qrc_mse = float(np.mean((np.array(preds) - test_states) ** 2))
        qrc_mses.append(qrc_mse)

        # ---- ESN at matched feature dim ----
        W_in, W_res, alpha = make_esn(N, feature_dim, seed)
        train_acts = _esn_run(train_states, W_in, W_res, alpha)
        F_tr_e = []
        for i in range(window, len(train_states)):
            F_tr_e.append(train_acts[i - window:i].flatten())
        F_tr_e = np.asarray(F_tr_e)
        W_e = _ridge_solve(F_tr_e, S_tr, ridge_alpha)

        # Test teacher-forced: drive ESN with ground-truth states
        full_states = np.concatenate([train_states, test_states])
        full_acts = _esn_run(full_states, W_in, W_res, alpha)
        preds_e = []
        for i in range(len(train_states), len(full_states)):
            window_flat = full_acts[i - window:i].flatten()
            preds_e.append(window_flat @ W_e)
        esn_mse = float(np.mean((np.array(preds_e) - test_states) ** 2))
        esn_mses.append(esn_mse)

        # ---- Persistence ----
        pers_preds = np.tile(train_states[-1], (len(test_states), 1))
        pers_mses.append(float(np.mean((pers_preds - test_states) ** 2)))

    qrc_v = np.array(qrc_mses); esn_v = np.array(esn_mses); pers_v = np.array(pers_mses)
    return {
        "N": N, "feature_dim": feature_dim,
        "qrc_mean": float(qrc_v.mean()), "qrc_std": float(qrc_v.std()),
        "esn_mean": float(esn_v.mean()), "esn_std": float(esn_v.std()),
        "pers_mean": float(pers_v.mean()),
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
    p.add_argument("--Ns", type=int, nargs="+", default=[5, 6, 7, 8, 9, 10])
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "l96_kq"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "scaling_summary.json"
    rows = []
    for N in args.Ns:
        t0 = time.time()
        cell = run_cell(N=N, n_seeds=args.n_seeds)
        cell["wall_time_s"] = time.time() - t0
        rows.append(cell)
        winner = "QRC" if cell["qrc_mean"] < cell["esn_mean"] else "ESN"
        print(f"N=q=K={N}  feat={cell['feature_dim']:>4} | "
              f"QRC={cell['qrc_mean']:.3e}+/-{cell['qrc_std']:.3e}  "
              f"ESN={cell['esn_mean']:.3e}+/-{cell['esn_std']:.3e}  "
              f"pers={cell['pers_mean']:.3e}  "
              f"QRC<ESN={cell['qrc_wins_vs_esn']}/{args.n_seeds}  "
              f"p={cell['wilcoxon_qrc_vs_esn']:.4f}  "
              f"lower-mean: {winner}  ({cell['wall_time_s']:.1f}s)")
        # Incremental save so partial runs persist if the process is killed
        with open(out_json, "w") as f:
            json.dump({"n_seeds": args.n_seeds, "rows": rows}, f, indent=2)

    print(f"\nSaved {out_json}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        Ns = [r["N"] for r in rows]
        ax.errorbar(Ns, [r["qrc_mean"] for r in rows],
                     yerr=[r["qrc_std"] for r in rows],
                     marker="o", lw=2, ms=8, capsize=4, label="QRC (q qubits)",
                     color="#0066cc", markeredgecolor="white", markeredgewidth=0.7)
        ax.errorbar(Ns, [r["esn_mean"] for r in rows],
                     yerr=[r["esn_std"] for r in rows],
                     marker="s", lw=2, ms=8, capsize=4, label="ESN (N=2^q)",
                     color="#cc4400", markeredgecolor="white", markeredgewidth=0.7)
        ax.plot(Ns, [r["pers_mean"] for r in rows], marker="X", ms=8, lw=1.4,
                 ls=":", label="Persistence", color="#888")
        ax.set_xlabel("Lorenz-96 dimension  N = q  (qubit count = system size)")
        ax.set_ylabel("Test teacher MSE (mean +/- std over 8 seeds)")
        ax.set_title("L96 K=q matched-complexity scaling")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        for i, r in enumerate(rows):
            ax.annotate(f"p={r['wilcoxon_qrc_vs_esn']:.3f}",
                         (r["N"], min(r["qrc_mean"], r["esn_mean"]) * 0.6),
                         ha="center", fontsize=8, color="#444")
        fig.tight_layout()
        out_fig = out_dir / "skill_lead1_vs_kq.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
