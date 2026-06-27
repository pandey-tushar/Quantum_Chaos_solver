#!/usr/bin/env python3
"""
run_l96_kq_baselines.py - Broader baseline panel for the L96 K=q
matched-complexity sweep.  Extends the QRC + ESN + persistence
comparison in run_l96_kq_scaling.py with three additional classical
baselines so the paper-2 discussion section can address the
``ESN is just badly tuned'' reviewer objection directly:

  * linreg - ridge regression on windowed past states (no
    nonlinearity).  Tests ``does any nonlinearity help?''
  * RFF    - Random Fourier Features at matched feature dim 2^q,
    then ridge readout.  Tests ``does the specific quantum
    nonlinearity matter, or would any fixed nonlinear projection
    give the same scaling?''
  * MLP    - one-hidden-layer ReLU network, hidden size chosen so
    total trainable params match the QRC readout matrix.  Tests
    ``why not just train a small NN?''

The script also re-computes QRC and ESN at the same cells so the JSON
output is internally consistent and a single panel of methods can be
plotted side-by-side.

Outputs:
    results/l96_kq/baselines_summary.json
    results/l96_kq/baselines_panel.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

# Suppress sklearn convergence warnings (200 iters is intentional; we
# want to match the QPINN budget, not converge).
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Convergence.*")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Lorenz-96
# ---------------------------------------------------------------------------

def l96_rhs(x: np.ndarray, F: float = 8.0) -> np.ndarray:
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


def integrate_l96(x0: np.ndarray, t_end: float, dt: float = 0.01,
                   F: float = 8.0) -> np.ndarray:
    t, x = 0.0, np.array(x0, dtype=float)
    traj = [x.copy()]
    while t < t_end - 1e-10:
        k1 = l96_rhs(x, F); k2 = l96_rhs(x + 0.5*dt*k1, F)
        k3 = l96_rhs(x + 0.5*dt*k2, F); k4 = l96_rhs(x + dt*k3, F)
        x = x + dt / 6.0 * (k1 + 2.0*k2 + 2.0*k3 + k4)
        t += dt
        traj.append(x.copy())
    return np.array(traj)


# ---------------------------------------------------------------------------
# QRC and ESN  (mirrors run_l96_kq_scaling.py exactly)
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
# Linear regression on windowed past states
# ---------------------------------------------------------------------------

def _windowed_features(states: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Flatten the past `window` states into the regression input."""
    F_rows, S_rows = [], []
    for i in range(window, len(states)):
        F_rows.append(states[i - window:i].flatten())
        S_rows.append(states[i])
    return np.asarray(F_rows), np.asarray(S_rows)


def linreg_eval(train_states, test_states, window, ridge_alpha):
    F_tr, S_tr = _windowed_features(train_states, window)
    W = _ridge_solve(F_tr, S_tr, ridge_alpha)
    full = np.concatenate([train_states, test_states])
    preds, targets = [], []
    for i in range(len(train_states), len(full)):
        window_flat = full[i - window:i].flatten()
        preds.append(window_flat @ W)
        targets.append(full[i])
    return float(np.mean((np.array(preds) - np.array(targets)) ** 2))


# ---------------------------------------------------------------------------
# Random Fourier Features at matched feature dim 2^q
# ---------------------------------------------------------------------------

def make_rff_extractor(n_input_w: int, n_features: int, seed: int,
                        gamma: float | None = None):
    """RFF feature map.  `n_input_w` = window * N (the dim of the windowed
    past-state input).  `gamma` (inverse lengthscale^2) defaults to the
    median heuristic computed on the training set."""
    rng = np.random.default_rng(seed)
    state = {"W": None, "b": None, "gamma": gamma}

    def fit(F_tr: np.ndarray):
        if state["gamma"] is None:
            # Median heuristic on pairwise sq-distances
            n = min(F_tr.shape[0], 50)
            idx = rng.choice(F_tr.shape[0], n, replace=False)
            sample = F_tr[idx]
            d2 = np.sum((sample[:, None] - sample[None, :]) ** 2, axis=-1)
            med = float(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
            state["gamma"] = 1.0 / max(med, 1e-8)
        # Sample frequencies and biases for cos(W x + b) features.
        # Standard RBF kernel approximation: 2*cos(W x + b) / sqrt(D).
        D = n_features
        state["W"] = rng.standard_normal((D, n_input_w)) * np.sqrt(2.0 * state["gamma"])
        state["b"] = rng.uniform(0, 2 * np.pi, D)

    def transform(F: np.ndarray) -> np.ndarray:
        return np.sqrt(2.0 / state["W"].shape[0]) * np.cos(F @ state["W"].T + state["b"])

    return fit, transform


def rff_eval(train_states, test_states, n_features, window, ridge_alpha, seed):
    F_tr, S_tr = _windowed_features(train_states, window)
    fit, transform = make_rff_extractor(F_tr.shape[1], n_features, seed)
    fit(F_tr)
    Z_tr = transform(F_tr)
    W = _ridge_solve(Z_tr, S_tr, ridge_alpha)
    full = np.concatenate([train_states, test_states])
    F_te, S_te = _windowed_features(full, window)
    # Predict only on the test portion (indices >= n_train)
    n_train_pairs = len(train_states) - window
    F_te_only = F_te[n_train_pairs:]
    S_te_only = S_te[n_train_pairs:]
    Z_te = transform(F_te_only)
    preds = Z_te @ W
    return float(np.mean((preds - S_te_only) ** 2))


# ---------------------------------------------------------------------------
# MLP at matched parameter count
# ---------------------------------------------------------------------------

def _matched_hidden_size(q: int, window: int, N: int) -> int:
    """Choose hidden size H so an MLP with one hidden layer on a
    windowed input (w*N -> H -> N) has roughly the same parameter
    count as the QRC readout matrix at K=q (w * 2^q * N params)."""
    target = window * (2 ** q) * N
    denom = window * N + N + 1     # input_w*H + H + H*N + N -> ~H * (wN + N + 1)
    return max(8, int(round(target / denom)))


def mlp_eval(train_states, test_states, q, N, window, seed):
    from sklearn.neural_network import MLPRegressor
    F_tr, S_tr = _windowed_features(train_states, window)
    H = _matched_hidden_size(q, window, N)
    model = MLPRegressor(hidden_layer_sizes=(H,), activation="relu",
                          solver="adam", learning_rate_init=1e-3,
                          max_iter=200, random_state=seed,
                          early_stopping=False, alpha=1e-4)
    model.fit(F_tr, S_tr)
    full = np.concatenate([train_states, test_states])
    F_te, S_te = _windowed_features(full, window)
    n_train_pairs = len(train_states) - window
    preds = model.predict(F_te[n_train_pairs:])
    return float(np.mean((preds - S_te[n_train_pairs:]) ** 2)), H


# ---------------------------------------------------------------------------
# Per-cell experiment
# ---------------------------------------------------------------------------

def run_cell(N: int, n_seeds: int, n_layers: int = 2, window: int = 5,
              ridge_alpha: float = 1.0, dt: float = 0.01,
              t_train_end: float = 3.0, t_test_end: float = 4.0,
              n_train: int = 50, n_test: int = 20,
              F_l96: float = 8.0):
    feature_dim = 2 ** N
    x0 = np.zeros(N); x0[0] = 0.01
    full_traj = integrate_l96(x0, t_test_end, dt=dt, F=F_l96)
    train_idx = np.linspace(0, int(t_train_end / dt) - 1, n_train, dtype=int)
    test_idx  = np.linspace(int(t_train_end / dt),
                              int(t_test_end / dt) - 1, n_test, dtype=int)
    train_states = full_traj[train_idx, :N]
    test_states  = full_traj[test_idx, :N]

    # Persistence floor (deterministic)
    pers_preds = np.tile(train_states[-1], (len(test_states), 1))
    pers_mse = float(np.mean((pers_preds - test_states) ** 2))

    # Linreg (deterministic, no seed dependence)
    linreg_mse = linreg_eval(train_states, test_states, window, ridge_alpha)

    qrc_mses, esn_mses, rff_mses, mlp_mses = [], [], [], []
    mlp_hidden = None
    for seed in range(n_seeds):
        # ---- QRC ----
        extract = make_qrc_extractor(N, n_layers, seed)
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

        # ---- RFF at matched feature dim ----
        rff_mses.append(rff_eval(train_states, test_states,
                                    n_features=feature_dim, window=window,
                                    ridge_alpha=ridge_alpha, seed=seed))

        # ---- MLP at matched param count ----
        mlp_mse, H = mlp_eval(train_states, test_states, q=N, N=N,
                                window=window, seed=seed)
        mlp_mses.append(mlp_mse)
        mlp_hidden = H

    qrc_v = np.array(qrc_mses); esn_v = np.array(esn_mses)
    rff_v = np.array(rff_mses); mlp_v = np.array(mlp_mses)
    return {
        "N": N, "feature_dim": feature_dim,
        "mlp_hidden_size": mlp_hidden,
        "persistence_mse": pers_mse,
        "linreg_mse": linreg_mse,   # deterministic, no std
        "qrc_mean": float(qrc_v.mean()), "qrc_std": float(qrc_v.std()),
        "esn_mean": float(esn_v.mean()), "esn_std": float(esn_v.std()),
        "rff_mean": float(rff_v.mean()), "rff_std": float(rff_v.std()),
        "mlp_mean": float(mlp_v.mean()), "mlp_std": float(mlp_v.std()),
        "qrc_per_seed": qrc_v.tolist(),
        "esn_per_seed": esn_v.tolist(),
        "rff_per_seed": rff_v.tolist(),
        "mlp_per_seed": mlp_v.tolist(),
        "wilcoxon_qrc_vs_rff": float(stats.wilcoxon(qrc_v, rff_v).pvalue)
                                  if not np.allclose(qrc_v, rff_v) else float("nan"),
        "wilcoxon_qrc_vs_mlp": float(stats.wilcoxon(qrc_v, mlp_v).pvalue)
                                  if not np.allclose(qrc_v, mlp_v) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--Ns", type=int, nargs="+",
                    default=[5, 6, 7, 8, 9, 10, 11])
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "l96_kq"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "baselines_summary.json"
    rows = []
    for N in args.Ns:
        t0 = time.time()
        cell = run_cell(N=N, n_seeds=args.n_seeds)
        cell["wall_time_s"] = time.time() - t0
        rows.append(cell)
        print(f"N={N:>2}  feat={cell['feature_dim']:>4}  H_mlp={cell['mlp_hidden_size']:>4} | "
              f"pers={cell['persistence_mse']:.3e}  "
              f"linreg={cell['linreg_mse']:.3e}  "
              f"QRC={cell['qrc_mean']:.3e}  "
              f"ESN={cell['esn_mean']:.3e}  "
              f"RFF={cell['rff_mean']:.3e}  "
              f"MLP={cell['mlp_mean']:.3e}  "
              f"({cell['wall_time_s']:.1f}s)")
        with open(out_json, "w") as f:
            json.dump({"n_seeds": args.n_seeds, "rows": rows}, f, indent=2)
    print(f"\nSaved {out_json}")

    # ----- Plot -----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9.0, 5.5))
        Ns = [r["N"] for r in rows]
        ax.errorbar(Ns, [r["qrc_mean"] for r in rows],
                     yerr=[r["qrc_std"] for r in rows],
                     marker="o", lw=2, ms=8, capsize=4,
                     label="QRC (q qubits)",
                     color="#0F4C5C", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.errorbar(Ns, [r["esn_mean"] for r in rows],
                     yerr=[r["esn_std"] for r in rows],
                     marker="s", lw=2, ms=8, capsize=4,
                     label="ESN (N=2^q)",
                     color="#9A4836", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.errorbar(Ns, [r["rff_mean"] for r in rows],
                     yerr=[r["rff_std"] for r in rows],
                     marker="^", lw=2, ms=8, capsize=4,
                     label="RFF (N=2^q)",
                     color="#5E7C6B", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.errorbar(Ns, [r["mlp_mean"] for r in rows],
                     yerr=[r["mlp_std"] for r in rows],
                     marker="D", lw=2, ms=8, capsize=4,
                     label="MLP (matched params)",
                     color="#B8864A", markeredgecolor="white",
                     markeredgewidth=0.7)
        ax.plot(Ns, [r["linreg_mse"] for r in rows],
                 marker="v", ms=8, lw=1.4, ls=":",
                 label="Linear ridge", color="#666")
        ax.plot(Ns, [r["persistence_mse"] for r in rows],
                 marker="X", ms=8, lw=1.4, ls=":",
                 label="Persistence", color="#aaa")
        ax.set_xlabel(r"L96 dimension  $N = q$  (matched complexity)")
        ax.set_ylabel(f"Test teacher MSE  (mean +/- std over {args.n_seeds} seeds)")
        ax.set_title("L96 K=q broader baseline panel")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=9, ncol=2)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out_fig = out_dir / "baselines_panel.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
