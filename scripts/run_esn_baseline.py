#!/usr/bin/env python3
"""
run_esn_baseline.py — Classical Echo State Network (ESN) baseline for Lorenz system.

Implements a classical ESN with:
  - N=500 reservoir neurons
  - Spectral radius 0.9
  - Input scaling 0.1
  - Leaking rate 0.3
  - Ridge regression readout (same as QRC)
  - Temporal window w=5 (same as QRC)
  - Same train/test split as QRC: t∈[0,3] train, t∈[3,4] test
  - 5 seeds (0–4)

Saves results to results/esn_baseline.json
"""

import json
import time
import numpy as np
from pathlib import Path

# ── Config (matching QRC benchmark) ─────────────────────────────────────────
N_RESERVOIR  = 500
SPECTRAL_RAD = 0.9
INPUT_SCALE  = 0.1
LEAKING_RATE = 0.3
RIDGE_ALPHA  = 1.0       # same regularisation as QRC
WINDOW       = 5         # same temporal window as QRC
DT           = 0.01
T_TRAIN_END  = 3.0
T_TEST_END   = 4.0
N_TRAIN      = 50
N_TEST       = 20
SEEDS        = list(range(5))

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Lorenz system ─────────────────────────────────────────────────────────────

def lorenz_rhs(state, sigma=10.0, rho=28.0, beta=8.0/3.0):
    x, y, z = state
    return np.array([sigma*(y - x), x*(rho - z) - y, x*y - beta*z])


def integrate_lorenz(x0, t_end, dt=DT):
    t, x = 0.0, np.array(x0, dtype=float)
    trajectory, times = [x.copy()], [t]
    while t < t_end - 1e-10:
        k1 = lorenz_rhs(x)
        k2 = lorenz_rhs(x + dt/2 * k1)
        k3 = lorenz_rhs(x + dt/2 * k2)
        k4 = lorenz_rhs(x + dt * k3)
        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        trajectory.append(x.copy())
        times.append(t)
    return np.array(times), np.array(trajectory)


# ── ESN implementation ────────────────────────────────────────────────────────

class EchoStateNetwork:
    """
    Classical Echo State Network with leaky integration and ridge regression readout.

    Reservoir update:
        r(t+1) = (1 - alpha) * r(t) + alpha * tanh(W_res @ r(t) + W_in @ u(t))

    Readout:
        y_hat(t) = W_out @ r_window(t)   (ridge regression)
    """

    def __init__(self, n_input, n_reservoir=500, n_output=3,
                 spectral_radius=0.9, input_scaling=0.1,
                 leaking_rate=0.3, ridge_alpha=1.0, seed=0):
        self.n_input      = n_input
        self.n_reservoir  = n_reservoir
        self.n_output     = n_output
        self.alpha        = leaking_rate
        self.ridge_alpha  = ridge_alpha
        self.seed         = seed

        rng = np.random.default_rng(seed)

        # Input weight matrix — dense, random ±1 scaled
        self.W_in = rng.uniform(-input_scaling, input_scaling,
                                (n_reservoir, n_input))

        # Reservoir weight matrix — sparse Erdős–Rényi, then rescaled to target SR
        density = 0.1
        W = rng.standard_normal((n_reservoir, n_reservoir))
        mask = rng.uniform(0, 1, (n_reservoir, n_reservoir)) < density
        W = W * mask
        # Rescale spectral radius
        eigenvalues = np.linalg.eigvals(W)
        sr = np.max(np.abs(eigenvalues))
        if sr > 1e-8:
            W = W * (spectral_radius / sr)
        self.W_res = W

        self.W_out = None  # learned via ridge regression

    def _run_reservoir(self, states):
        """Run reservoir on a sequence of input states, return reservoir activations."""
        T = len(states)
        r = np.zeros(self.n_reservoir)
        activations = np.zeros((T, self.n_reservoir))
        for t, u in enumerate(states):
            r = ((1 - self.alpha) * r
                 + self.alpha * np.tanh(self.W_res @ r + self.W_in @ u))
            activations[t] = r
        return activations

    def fit(self, train_states, window=5):
        """Fit readout weights using ridge regression with temporal windowing."""
        activations = self._run_reservoir(train_states)
        # Build windowed feature matrix (same window logic as QRC)
        feat_list, target_list = [], []
        for i in range(window, len(train_states)):
            window_feats = activations[i - window:i].flatten()
            feat_list.append(window_feats)
            target_list.append(train_states[i])
        F = np.array(feat_list)    # (T - window, n_reservoir * window)
        S = np.array(target_list)  # (T - window, n_output)

        # Ridge regression closed form
        A = F.T @ F + self.ridge_alpha * np.eye(F.shape[1])
        self.W_out = np.linalg.solve(A, F.T @ S)

        self._train_activations = activations  # keep for test seeding
        self._window = window
        return F, S

    def predict_train(self, train_states, window=5):
        """Compute training predictions."""
        activations = self._run_reservoir(train_states)
        preds = []
        for i in range(window, len(train_states)):
            wf = activations[i - window:i].flatten()
            preds.append(wf @ self.W_out)
        return np.array(preds)

    def predict_test(self, train_states, test_states, window=5):
        """Predict test states, seeded with last `window` training activations."""
        # Re-run reservoir on training to get ending state
        train_acts = self._run_reservoir(train_states)
        # Seed buffer with last `window` training activations
        act_buffer = list(train_acts[-window:])
        r = train_acts[-1].copy()  # reservoir state after training

        preds = []
        for u in test_states:
            r = ((1 - self.alpha) * r
                 + self.alpha * np.tanh(self.W_res @ r + self.W_in @ u))
            act_buffer.append(r.copy())
            wf = np.array(act_buffer[-window:]).flatten()
            pred = wf @ self.W_out
            preds.append(pred)
        return np.array(preds)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Classical ESN Baseline — Lorenz System")
    print(f"  N_reservoir = {N_RESERVOIR}, spectral_radius = {SPECTRAL_RAD}")
    print(f"  input_scaling = {INPUT_SCALE}, leaking_rate = {LEAKING_RATE}")
    print(f"  ridge_alpha = {RIDGE_ALPHA}, window = {WINDOW}")
    print(f"  Seeds: {SEEDS}")
    print("=" * 60)

    # Generate Lorenz trajectory
    x0 = np.array([1.0, 1.0, 1.0])
    times, trajectory = integrate_lorenz(x0, T_TEST_END)

    # Extract train / test splits matching QRC benchmark
    # Same logic: uniform sampling of N_TRAIN points from t∈[0,3] and N_TEST from t∈[3,4]
    train_mask = times <= T_TRAIN_END
    test_mask  = (times > T_TRAIN_END) & (times <= T_TEST_END)

    train_times = times[train_mask]
    train_traj  = trajectory[train_mask]
    test_times  = times[test_mask]
    test_traj   = trajectory[test_mask]

    # Sub-sample to match QRC's 50 train + 20 test points
    train_idx = np.round(np.linspace(0, len(train_traj) - 1, N_TRAIN)).astype(int)
    test_idx  = np.round(np.linspace(0, len(test_traj)  - 1, N_TEST )).astype(int)
    train_states = train_traj[train_idx]
    test_states  = test_traj[test_idx]

    per_seed = []

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")
        t0 = time.time()

        esn = EchoStateNetwork(
            n_input=3,
            n_reservoir=N_RESERVOIR,
            n_output=3,
            spectral_radius=SPECTRAL_RAD,
            input_scaling=INPUT_SCALE,
            leaking_rate=LEAKING_RATE,
            ridge_alpha=RIDGE_ALPHA,
            seed=seed
        )

        F_mat, S_mat = esn.fit(train_states, window=WINDOW)
        train_time = time.time() - t0

        # Train MSE
        train_pred = esn.predict_train(train_states, window=WINDOW)
        train_mse = float(np.mean((train_pred - S_mat) ** 2))

        # Test MSE
        test_pred = esn.predict_test(train_states, test_states, window=WINDOW)
        test_mse  = float(np.mean((test_pred - test_states) ** 2))

        print(f"  Train MSE: {train_mse:.4f}")
        print(f"  Test  MSE: {test_mse:.4f}")
        print(f"  Time:      {train_time*1000:.2f} ms")

        per_seed.append({
            "seed":       seed,
            "train_mse":  train_mse,
            "test_mse":   test_mse,
            "train_time": train_time
        })

    # Aggregate
    train_mses = np.array([s["train_mse"] for s in per_seed])
    test_mses  = np.array([s["test_mse"]  for s in per_seed])
    train_times = np.array([s["train_time"] for s in per_seed])

    summary = {
        "method": "Classical ESN (N=500)",
        "config": {
            "n_reservoir":    N_RESERVOIR,
            "spectral_radius": SPECTRAL_RAD,
            "input_scaling":  INPUT_SCALE,
            "leaking_rate":   LEAKING_RATE,
            "ridge_alpha":    RIDGE_ALPHA,
            "window":         WINDOW,
            "n_train":        N_TRAIN,
            "n_test":         N_TEST
        },
        "per_seed": per_seed,
        "train_mse_mean": float(train_mses.mean()),
        "train_mse_std":  float(train_mses.std()),
        "test_mse_mean":  float(test_mses.mean()),
        "test_mse_std":   float(test_mses.std()),
        "train_time_mean_s": float(train_times.mean()),
        "train_time_std_s":  float(train_times.std())
    }

    out_path = RESULTS_DIR / "esn_baseline.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Train MSE: {summary['train_mse_mean']:.2f} ± {summary['train_mse_std']:.2f}")
    print(f"Test  MSE: {summary['test_mse_mean']:.2f}  ± {summary['test_mse_std']:.2f}")
    print(f"Time:      {summary['train_time_mean_s']*1000:.2f} ms per seed")
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
