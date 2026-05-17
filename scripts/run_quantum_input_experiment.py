#!/usr/bin/env python3
"""
run_quantum_input_experiment.py - QRC vs classical reservoirs on a
quantum-input prediction task.  Bowles-style trig-polynomial
dequantization arguments do NOT apply when the reservoir takes a
quantum state's partial information as input rather than a scalar /
classical vector, so this is the cleanest theoretically-defensible
zone where a QRC advantage might still exist at small qubit count.

Task:
  - Target system: 4-qubit Heisenberg chain with disorder, evolved
    from |+>^4 for T timesteps.
  - 6 target observables: a mix of single-site and two-body Paulis.
  - 3 input observables: a sparser subset (partial info).
  - At each t the reservoir sees a 3-dim input vector x[t] = <O_input>(t)
    and must predict the 6-dim output y[t+k] = <O_target>(t+k) for
    k in {1, 5, 20}.

Reservoirs compared (all start with the same input x[t]):
  - TFIM QRC at n_qubits (nearest-neighbour Ising, transverse field).
  - SCRAM QRC at n_qubits (random all-to-all 2-body Ising + random
    transverse/longitudinal fields; SYK-style scrambling).
  - ESN at matched feature dim 2^n_qubits.
  - RFF at matched feature dim 2^n_qubits.
  - RFF at matched parameter count (proxy for "any random projection").

For each: windowed ridge readout, w=5.  5 seeds.  Reports test NRMSE,
feature-Gram conditioning, and a per-horizon comparison.

Outputs:
    results/quantum_input/summary.json
    results/quantum_input/nrmse_vs_method.png
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
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))


# ---------------------------------------------------------------------------
# Pauli operators on n-qubit Hilbert space (Kronecker products)
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def pauli_string(ops: str) -> np.ndarray:
    """Build the matrix for a Pauli string like 'IZXI' on len(ops) qubits."""
    out = PAULI[ops[0]]
    for c in ops[1:]:
        out = np.kron(out, PAULI[c])
    return out


def single_site(op: str, site: int, n: int) -> np.ndarray:
    """Pauli op on `site` (0-indexed) acting on n qubits, identity elsewhere."""
    s = ["I"] * n
    s[site] = op
    return pauli_string("".join(s))


def two_site(op1: str, site1: int, op2: str, site2: int, n: int) -> np.ndarray:
    s = ["I"] * n
    s[site1] = op1
    s[site2] = op2
    return pauli_string("".join(s))


# ---------------------------------------------------------------------------
# Target quantum system: disordered Heisenberg chain
# ---------------------------------------------------------------------------

def heisenberg_chain_hamiltonian(n: int, J: float = 1.0, disorder: float = 0.5,
                                   seed: int = 42) -> np.ndarray:
    """H = J sum_{i,i+1} (X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1}) + sum_i h_i Z_i"""
    rng = np.random.default_rng(seed)
    h = rng.uniform(-disorder, disorder, n)
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        H += J * two_site("X", i, "X", i + 1, n)
        H += J * two_site("Y", i, "Y", i + 1, n)
        H += J * two_site("Z", i, "Z", i + 1, n)
    for i in range(n):
        H += h[i] * single_site("Z", i, n)
    return H


def evolve_states(H: np.ndarray, psi0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Return |psi(t)> for each t in `times`.  Returns shape (n_t, dim)."""
    # Diagonalise once, then exp-trick
    w, V = np.linalg.eigh(H)
    psi0_in_basis = V.conj().T @ psi0
    out = np.zeros((len(times), len(psi0)), dtype=complex)
    for k, t in enumerate(times):
        phases = np.exp(-1j * w * t)
        out[k] = V @ (phases * psi0_in_basis)
    return out


def expectation_values(states: np.ndarray, ops: list[np.ndarray]) -> np.ndarray:
    """For each state row and each op, compute <psi|O|psi>.  Returns (n_t, len(ops))."""
    n_t = len(states)
    out = np.zeros((n_t, len(ops)), dtype=float)
    for k in range(n_t):
        psi = states[k]
        for j, O in enumerate(ops):
            out[k, j] = float(np.real(psi.conj() @ (O @ psi)))
    return out


# ---------------------------------------------------------------------------
# Reservoirs:  TFIM-QRC, SCRAM-QRC, ESN, RFF
# ---------------------------------------------------------------------------

def tfim_hamiltonian(n: int, J: float = 1.0, g: float = 1.0) -> np.ndarray:
    """Nearest-neighbour TFIM: H = -J sum Z_i Z_{i+1} - g sum X_i."""
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for i in range(n - 1):
        H += -J * two_site("Z", i, "Z", i + 1, n)
    for i in range(n):
        H += -g * single_site("X", i, n)
    return H


def scrambling_hamiltonian(n: int, seed: int = 0) -> np.ndarray:
    """SYK-style: random all-to-all 2-body Ising + random fields on
    every qubit.  Strong scrambling, no nearest-neighbour locality."""
    rng = np.random.default_rng(seed)
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    sigma_J = 1.0 / np.sqrt(n)
    for i in range(n):
        for j in range(i + 1, n):
            J_ij = rng.normal(0.0, sigma_J)
            H += J_ij * two_site("Z", i, "Z", j, n)
    for i in range(n):
        H += rng.normal(0.0, 1.0) * single_site("X", i, n)
        H += rng.normal(0.0, 1.0) * single_site("Y", i, n)
    return H


def make_qrc_extractor(n_qubits: int, n_input: int, H_reservoir: np.ndarray,
                        tau: float = 1.0):
    """Build a recurrent QRC feature extractor.  At each time step:
      1. Re-encode input via RY(pi * (x_i + 1)) on the first n_input qubits
         (input expectation values are in [-1, 1] -> angles in [0, 2*pi]).
      2. Evolve the full state under H_reservoir for time tau.
      3. Return the 2^n_qubits basis-state probability vector.
    The state is carried over between time steps (recurrence)."""
    dim = 2 ** n_qubits
    U_evolve = sla.expm(-1j * H_reservoir * tau)

    def init_state() -> np.ndarray:
        # Start in |0...0>
        psi = np.zeros(dim, dtype=complex)
        psi[0] = 1.0
        return psi

    def reencode_step(psi: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Apply RY(pi*(x_i+1)) on each of the first n_input qubits,
        then evolve."""
        # Build the full Kronecker-product local rotation
        # Each RY(theta) acts on one qubit; identity on the others.
        # Implementation: apply per qubit by reshaping the state into a tensor.
        for qubit in range(n_input):
            theta = np.pi * (x[qubit] + 1.0)
            c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
            psi = _apply_single_qubit_gate(psi, n_qubits, qubit,
                                              np.array([[c, -s], [s, c]],
                                                       dtype=complex))
        psi = U_evolve @ psi
        return psi

    def extract_features(psi: np.ndarray) -> np.ndarray:
        return np.abs(psi) ** 2

    return init_state, reencode_step, extract_features


def _apply_single_qubit_gate(psi: np.ndarray, n: int, qubit: int,
                                gate: np.ndarray) -> np.ndarray:
    """Apply a 2x2 gate to `qubit` (0-indexed from the left in Kron order)."""
    # Reshape to (2,)*n, apply gate on axis `qubit`, reshape back.
    arr = psi.reshape((2,) * n)
    arr = np.moveaxis(arr, qubit, 0)
    arr = np.tensordot(gate, arr, axes=([1], [0]))
    arr = np.moveaxis(arr, 0, qubit)
    return arr.reshape(-1)


# Classical baselines ----------------------------------------------------------

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


def esn_run(inputs, W_in, W_res, leak):
    T = len(inputs); n_res = W_res.shape[0]
    r = np.zeros(n_res); out = np.zeros((T, n_res))
    for t in range(T):
        r = (1 - leak) * r + leak * np.tanh(W_res @ r + W_in @ inputs[t])
        out[t] = r
    return out


def rff_features(inputs, n_features: int, seed: int, gamma: float | None = None):
    """Random Fourier Features.  Non-recurrent — applies to windowed past inputs."""
    rng = np.random.default_rng(seed)
    d = inputs.shape[1]
    if gamma is None:
        # Median heuristic
        sample_n = min(200, len(inputs))
        idx = rng.choice(len(inputs), sample_n, replace=False)
        d2 = np.sum((inputs[idx, None] - inputs[None, idx]) ** 2, axis=-1)
        med = float(np.median(d2[d2 > 0])) if (d2 > 0).any() else 1.0
        gamma = 1.0 / max(med, 1e-8)
    W = rng.standard_normal((n_features, d)) * np.sqrt(2.0 * gamma)
    b = rng.uniform(0, 2 * np.pi, n_features)
    feats = np.sqrt(2.0 / n_features) * np.cos(inputs @ W.T + b)
    return feats


# Common ridge solve ---------------------------------------------------------

def ridge_solve(F: np.ndarray, S: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    A = F.T @ F + alpha * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ S)


def windowed_features(F: np.ndarray, window: int) -> np.ndarray:
    """Concatenate the past `window` rows for each time step >= window."""
    return np.array([F[i - window:i].flatten()
                       for i in range(window, len(F))])


def feature_gram_condition(F: np.ndarray, alpha: float = 1.0) -> float:
    svals = np.linalg.svd(F, compute_uv=False)
    n_rows, n_cols = F.shape
    eigs = svals ** 2 + alpha
    if n_cols > n_rows:
        eigs = np.concatenate([eigs, np.full(n_cols - n_rows, alpha)])
    return float(eigs.max() / eigs.min())


# ---------------------------------------------------------------------------
# Per-method evaluation
# ---------------------------------------------------------------------------

def eval_qrc(x_input, y_target, n_qubits, H_reservoir, window, horizons,
              train_frac, seed):
    """Build QRC features and evaluate at each horizon."""
    init_state, reencode_step, extract_features = make_qrc_extractor(
        n_qubits, x_input.shape[1], H_reservoir)
    psi = init_state()
    feats = []
    for t in range(len(x_input)):
        psi = reencode_step(psi, x_input[t])
        feats.append(extract_features(psi))
    F_raw = np.array(feats)
    Fw = windowed_features(F_raw, window)
    return _eval_windowed(Fw, y_target, window, horizons, train_frac, seed)


def eval_esn(x_input, y_target, n_reservoir, window, horizons, train_frac, seed):
    W_in, W_res, leak = make_esn(x_input.shape[1], n_reservoir, seed)
    acts = esn_run(x_input, W_in, W_res, leak)
    Fw = windowed_features(acts, window)
    return _eval_windowed(Fw, y_target, window, horizons, train_frac, seed)


def eval_rff(x_input, y_target, n_features, window, horizons, train_frac, seed):
    Fw_inputs = windowed_features(x_input, window)
    feats = rff_features(Fw_inputs, n_features, seed)
    return _eval_windowed_no_window(feats, y_target[window:], horizons,
                                       train_frac, seed)


def _eval_windowed(Fw, y_target, window, horizons, train_frac, seed):
    return _eval_windowed_no_window(Fw, y_target[window:], horizons,
                                       train_frac, seed)


def _eval_windowed_no_window(F, y_aligned, horizons, train_frac, seed):
    """Given a feature matrix F (rows = time steps after windowing) and the
    aligned target y_aligned (same number of rows), train a ridge readout
    and evaluate test NRMSE at multiple prediction horizons."""
    T = len(F)
    n_train = int(round(train_frac * T))
    results = {}
    kappa = feature_gram_condition(F[:n_train], alpha=1.0)
    results["kappa"] = kappa
    results["n_features"] = F.shape[1]
    results["n_train"] = n_train
    for k in horizons:
        if T - k <= n_train + 5:
            results[f"k{k}_nrmse"] = float("nan")
            continue
        F_tr = F[:n_train - k]
        y_tr = y_aligned[k:n_train]
        F_te = F[n_train:T - k]
        y_te = y_aligned[n_train + k:T]
        W = ridge_solve(F_tr, y_tr, alpha=1.0)
        y_pred = F_te @ W
        rmse = float(np.sqrt(np.mean((y_pred - y_te) ** 2)))
        std = float(np.std(y_te))
        nrmse = rmse / (std + 1e-12)
        results[f"k{k}_nrmse"] = nrmse
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-target", type=int, default=4,
                    help="qubits in target system")
    p.add_argument("--n-qubits", type=int, default=7,
                    help="qubits in reservoir")
    p.add_argument("--n-input-obs", type=int, default=3)
    p.add_argument("--n-target-obs", type=int, default=6)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--t-max", type=float, default=200.0)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "quantum_input"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"

    # ---- Generate target trajectory ----
    print(f"=== Target system: {args.n_target}-qubit disordered Heisenberg ===")
    H_tgt = heisenberg_chain_hamiltonian(args.n_target, seed=42)
    # Start in |+>^n (uniform superposition)
    psi0 = np.ones(2 ** args.n_target, dtype=complex) / np.sqrt(2 ** args.n_target)
    times = np.linspace(0, args.t_max, args.n_steps)
    t0 = time.time()
    psi_t = evolve_states(H_tgt, psi0, times)
    print(f"  Evolved {args.n_steps} steps over T=[0,{args.t_max}] in {time.time()-t0:.1f}s")

    # ---- Target observables (6): single-site Z on each + 2 nearest-neighbour ZZ ----
    target_ops = []
    target_labels = []
    for i in range(args.n_target):  # 4 single-site Z
        target_ops.append(single_site("Z", i, args.n_target))
        target_labels.append(f"Z_{i}")
    target_ops.append(two_site("Z", 0, "Z", 1, args.n_target))
    target_labels.append("Z_0 Z_1")
    target_ops.append(two_site("X", 1, "X", 2, args.n_target))
    target_labels.append("X_1 X_2")
    assert len(target_ops) == args.n_target_obs

    # ---- Input observables (3): a sparser subset (single-site X on first 3 sites) ----
    input_ops = []
    input_labels = []
    for i in range(args.n_input_obs):
        input_ops.append(single_site("X", i, args.n_target))
        input_labels.append(f"X_{i}")

    print(f"  Computing {len(target_ops)} target + {len(input_ops)} input expectations...")
    t0 = time.time()
    y_target = expectation_values(psi_t, target_ops)
    x_input = expectation_values(psi_t, input_ops)
    print(f"  done in {time.time()-t0:.1f}s   |y_target| ranges {y_target.min():.3f} to {y_target.max():.3f}")

    feature_dim = 2 ** args.n_qubits

    # ---- Per-seed evaluation ----
    # RFF_matched_dim:    D = 2^q (matches per-timestep feature dim).  Unfair
    #                       to QRC because after windowing QRC has w*2^q
    #                       effective features.
    # RFF_matched_effdim: D = w * 2^q (matches QRC's effective feature dim
    #                       after windowing).  This is the fair comparison.
    effective_dim = args.window * feature_dim
    methods_results = {"TFIM_QRC": [], "SCRAM_QRC": [], "ESN_matched": [],
                       "RFF_matched_dim": [], "RFF_matched_effdim": []}
    for seed in range(args.n_seeds):
        print(f"\n=== seed {seed} ===")
        # Build reservoir Hamiltonians per seed
        H_tfim = tfim_hamiltonian(args.n_qubits, J=1.0, g=1.0)
        H_scram = scrambling_hamiltonian(args.n_qubits, seed=seed)

        t0 = time.time()
        r_tfim = eval_qrc(x_input, y_target, args.n_qubits, H_tfim,
                            args.window, args.horizons, args.train_frac, seed)
        print(f"  TFIM_QRC ({time.time()-t0:.1f}s): k1={r_tfim['k1_nrmse']:.3f}  k5={r_tfim['k5_nrmse']:.3f}  k20={r_tfim['k20_nrmse']:.3f}  kappa={r_tfim['kappa']:.2e}")
        methods_results["TFIM_QRC"].append(r_tfim)

        t0 = time.time()
        r_scram = eval_qrc(x_input, y_target, args.n_qubits, H_scram,
                             args.window, args.horizons, args.train_frac, seed)
        print(f"  SCRAM_QRC ({time.time()-t0:.1f}s): k1={r_scram['k1_nrmse']:.3f}  k5={r_scram['k5_nrmse']:.3f}  k20={r_scram['k20_nrmse']:.3f}  kappa={r_scram['kappa']:.2e}")
        methods_results["SCRAM_QRC"].append(r_scram)

        t0 = time.time()
        r_esn = eval_esn(x_input, y_target, feature_dim,
                          args.window, args.horizons, args.train_frac, seed)
        print(f"  ESN_matched ({time.time()-t0:.1f}s): k1={r_esn['k1_nrmse']:.3f}  k5={r_esn['k5_nrmse']:.3f}  k20={r_esn['k20_nrmse']:.3f}  kappa={r_esn['kappa']:.2e}")
        methods_results["ESN_matched"].append(r_esn)

        t0 = time.time()
        r_rff = eval_rff(x_input, y_target, feature_dim,
                          args.window, args.horizons, args.train_frac, seed)
        print(f"  RFF_matched_dim ({time.time()-t0:.1f}s, D={feature_dim}): k1={r_rff['k1_nrmse']:.3f}  k5={r_rff['k5_nrmse']:.3f}  k20={r_rff['k20_nrmse']:.3f}  kappa={r_rff['kappa']:.2e}")
        methods_results["RFF_matched_dim"].append(r_rff)

        t0 = time.time()
        r_rff_eff = eval_rff(x_input, y_target, effective_dim,
                              args.window, args.horizons, args.train_frac, seed)
        print(f"  RFF_matched_effdim ({time.time()-t0:.1f}s, D={effective_dim}): k1={r_rff_eff['k1_nrmse']:.3f}  k5={r_rff_eff['k5_nrmse']:.3f}  k20={r_rff_eff['k20_nrmse']:.3f}  kappa={r_rff_eff['kappa']:.2e}")
        methods_results["RFF_matched_effdim"].append(r_rff_eff)

    # ---- Summary ----
    summary = {"args": vars(args),
                "target_observables": target_labels,
                "input_observables": input_labels,
                "feature_dim_qrc": feature_dim,
                "methods": methods_results,
                "per_method_means": {}}
    for method, runs in methods_results.items():
        means = {}
        for k in args.horizons:
            vals = [r[f"k{k}_nrmse"] for r in runs]
            means[f"k{k}_mean"] = float(np.nanmean(vals))
            means[f"k{k}_std"] = float(np.nanstd(vals))
        means["kappa_mean"] = float(np.nanmean([r["kappa"] for r in runs]))
        summary["per_method_means"][method] = means

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    # ---- Print table ----
    print(f"\n=== Mean test NRMSE across {args.n_seeds} seeds ===")
    print(f"{'method':<18}  " + "  ".join([f"k={k:<3}" for k in args.horizons]) + "  kappa")
    for method, m in summary["per_method_means"].items():
        row = f"{method:<18}  " + "  ".join(
            [f"{m[f'k{k}_mean']:.3f}+/-{m[f'k{k}_std']:.3f}" for k in args.horizons])
        row += f"  {m['kappa_mean']:.2e}"
        print(row)

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        method_colors = {"TFIM_QRC": "#0F4C5C", "SCRAM_QRC": "#1E7B8E",
                          "ESN_matched": "#9A4836",
                          "RFF_matched_dim": "#B8864A",
                          "RFF_matched_effdim": "#5E7C6B"}
        for method, m in summary["per_method_means"].items():
            means = [m[f"k{k}_mean"] for k in args.horizons]
            stds  = [m[f"k{k}_std"]  for k in args.horizons]
            ax.errorbar(args.horizons, means, yerr=stds,
                         marker="o", lw=2, ms=8, capsize=4,
                         label=method, color=method_colors.get(method, "#888"),
                         markeredgecolor="white", markeredgewidth=0.7)
        ax.set_xlabel("Prediction horizon k (time steps)")
        ax.set_ylabel("Test NRMSE (mean +/- std)")
        ax.set_title(f"Quantum-input task: {args.n_target}-qubit Heisenberg target,\n"
                       f"{args.n_qubits}-qubit reservoir, {args.n_seeds} seeds")
        ax.set_yscale("log")
        ax.legend(loc="best", fontsize=10)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out_fig = out_dir / "nrmse_vs_method.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
