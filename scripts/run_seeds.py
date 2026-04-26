"""
run_seeds.py — Multi-seed statistical replication for QRC vs QPINN comparison
==============================================================================
Runs both QRC and QPINN across N_SEEDS random seeds and reports:
  - Mean ± std of Train MSE, Test MSE, Training Time for each method
  - p-value (Wilcoxon signed-rank test) on train MSE difference
  - Per-seed results saved to results/seeds_results.json

Also optionally runs the Rössler attractor benchmark (--system rossler)
and Lorenz-96 with N=8 variables (--system lorenz96).

Usage:
  python scripts/run_seeds.py                        # Lorenz, 10 seeds
  python scripts/run_seeds.py --seeds 5              # Lorenz, 5 seeds
  python scripts/run_seeds.py --system rossler       # Rössler, 10 seeds
  python scripts/run_seeds.py --system lorenz96      # Lorenz-96 N=8, 10 seeds
  python scripts/run_seeds.py --dry-run              # Print config only

DO NOT RUN YET — requires multi-hour wall time for QPINN seeds.
Estimated time: ~3h per QPINN seed × N_SEEDS seeds + QRC (negligible).
Run on a machine with >8GB RAM; prefer overnight batch execution.
"""

import argparse
import json
import time
import numpy as np
from pathlib import Path
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────────
DEFAULT_SEEDS   = 10
QPINN_ITERS     = 200
DT              = 0.01
T_TRAIN_END     = 3.0
T_TEST_END      = 4.0
N_TRAIN         = 50
N_TEST          = 20

# QRC config
QRC_QUBITS      = 5
QRC_LAYERS      = 2
QRC_WINDOW      = 5
QRC_SHOTS       = 1024
QRC_ALPHA       = 1.0   # Ridge regularisation

# QPINN config
QPINN_QUBITS    = 4
QPINN_LAYERS    = 3
QPINN_LR        = 1e-3

RESULTS_DIR     = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── System definitions ───────────────────────────────────────────────────────

def lorenz_rhs(state, sigma=10.0, rho=28.0, beta=8/3):
    x, y, z = state
    return np.array([sigma*(y - x), x*(rho - z) - y, x*y - beta*z])


def rossler_rhs(state, a=0.2, b=0.2, c=5.7):
    x, y, z = state
    return np.array([-y - z, x + a*y, b + z*(x - c)])


def lorenz96_rhs(state, F=8.0):
    """Lorenz-96 model with N variables."""
    N = len(state)
    dxdt = np.zeros(N)
    for i in range(N):
        dxdt[i] = (state[(i+1) % N] - state[(i-2) % N]) * state[(i-1) % N] - state[i] + F
    return dxdt


def integrate(rhs, x0, t_end, dt=DT):
    """RK4 integration."""
    t, x = 0.0, np.array(x0, dtype=float)
    trajectory = [x.copy()]
    while t < t_end - 1e-10:
        k1 = rhs(x)
        k2 = rhs(x + dt/2 * k1)
        k3 = rhs(x + dt/2 * k2)
        k4 = rhs(x + dt * k3)
        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        t += dt
        trajectory.append(x.copy())
    return np.array(trajectory)


def get_system(name):
    if name == "lorenz":
        x0 = np.array([1.0, 1.0, 1.0])
        return lorenz_rhs, x0, 3
    elif name == "rossler":
        x0 = np.array([1.0, 1.0, 1.0])
        return rossler_rhs, x0, 3
    elif name == "lorenz96":
        N = 8
        x0 = np.zeros(N); x0[0] = 0.01   # small perturbation
        return lorenz96_rhs, x0, N
    else:
        raise ValueError(f"Unknown system: {name}")


# ── QRC ─────────────────────────────────────────────────────────────────────

def run_qrc(train_states, test_states, seed, dim):
    """Run QRC with fixed random reservoir, return (train_mse, test_mse, time_s)."""
    try:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError:
        raise ImportError("qiskit required. pip install qiskit")

    rng = np.random.default_rng(seed)
    n_qubits = QRC_QUBITS

    # Generate fixed random reservoir parameters
    rot_params = rng.uniform(0, 2*np.pi, (QRC_LAYERS, n_qubits, 3))  # Rx, Ry, Rz per qubit

    def extract_features(state):
        """Encode state into QRC and extract exact statevector probabilities."""
        norm_state = state[:n_qubits] if len(state) >= n_qubits else np.pad(state, (0, n_qubits - len(state)))
        angles = (norm_state - norm_state.min()) / (norm_state.max() - norm_state.min() + 1e-8) * 2 * np.pi

        qc = QuantumCircuit(n_qubits)
        for i in range(min(n_qubits, len(angles))):
            qc.ry(float(angles[i]), i)

        for layer in range(QRC_LAYERS):
            for i in range(n_qubits):
                qc.rx(float(rot_params[layer, i, 0]), i)
                qc.ry(float(rot_params[layer, i, 1]), i)
                qc.rz(float(rot_params[layer, i, 2]), i)
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)

        sv = Statevector.from_instruction(qc)
        return np.abs(sv.data) ** 2   # exact probabilities, size 2^n_qubits

    t0 = time.time()

    # Build temporal feature matrix for training
    all_feats = [extract_features(s) for s in train_states]
    feature_dim = len(all_feats[0])
    window_feats, targets = [], []
    for i in range(QRC_WINDOW, len(train_states)):
        window = np.concatenate(all_feats[i - QRC_WINDOW:i])
        window_feats.append(window)
        targets.append(train_states[i])
    F_mat = np.array(window_feats)
    S_mat = np.array(targets)

    # Ridge regression (closed form)
    A = F_mat.T @ F_mat + QRC_ALPHA * np.eye(F_mat.shape[1])
    W = np.linalg.solve(A, F_mat.T @ S_mat)

    train_time = time.time() - t0

    # Train MSE
    train_pred = F_mat @ W
    train_mse = float(np.mean((train_pred - S_mat)**2))

    # Test MSE — seed with last QRC_WINDOW training features, roll forward
    test_feats = [extract_features(s) for s in test_states]
    feat_buffer = list(all_feats[-QRC_WINDOW:])   # exactly QRC_WINDOW entries
    test_preds, test_targets = [], []
    for i, feat in enumerate(test_feats):
        window = np.concatenate(feat_buffer[-QRC_WINDOW:])   # always 160-D
        pred = window @ W
        test_preds.append(pred)
        test_targets.append(test_states[i])
        feat_buffer.append(feat)

    test_mse = float(np.mean((np.array(test_preds) - np.array(test_targets))**2))
    return train_mse, test_mse, train_time


# ── QPINN ────────────────────────────────────────────────────────────────────

def run_qpinn(train_times, train_states, test_times, test_states, seed, dim):
    """Run QPINN (Lorenz only) and return (train_mse, test_mse, time_s)."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    try:
        from quantum_circuit import LorenzQuantumCircuit
        from physics_loss import LorenzPhysicsLoss
        from training import train_lorenz_circuit
    except ImportError as e:
        print(f"  [QPINN seed={seed}] ImportError: {e}")
        return float('nan'), float('nan'), float('nan')

    try:
        t0 = time.time()

        # Build circuit and physics-informed loss
        circuit_obj = LorenzQuantumCircuit(
            n_qubits=QPINN_QUBITS,
            n_layers=QPINN_LAYERS
        )
        loss_fn = LorenzPhysicsLoss(
            circuit_obj,
            t_points=train_times,
            initial_condition=train_states[0]   # [x0, y0, z0]
        )

        # Train — suppress per-iteration checkpoints, print every 50 iters
        seed_results_dir = RESULTS_DIR / f"qpinn_seed{seed}"
        best_theta, history = train_lorenz_circuit(
            circuit_obj, loss_fn,
            n_iterations=QPINN_ITERS,
            learning_rate=QPINN_LR,
            print_every=50,
            save_every=QPINN_ITERS,   # checkpoint only at the end
            results_dir=seed_results_dir,
            random_seed=seed
        )
        train_time = time.time() - t0

        # Compute MSE against true trajectory (not physics residual)
        train_pred = loss_fn.evaluate_circuit_at_times(best_theta, train_times)
        train_mse  = float(np.mean((train_pred - train_states) ** 2))

        test_pred  = loss_fn.evaluate_circuit_at_times(best_theta, test_times)
        test_mse   = float(np.mean((test_pred  - test_states)  ** 2))

        return train_mse, test_mse, train_time

    except Exception as e:
        import traceback
        print(f"  [QPINN seed={seed}] Error: {e}")
        traceback.print_exc()
        return float('nan'), float('nan'), float('nan')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Multi-seed QRC vs QPINN benchmark")
    parser.add_argument("--seeds",      type=int, default=DEFAULT_SEEDS,
                        help="Total number of seeds (exclusive end of range)")
    parser.add_argument("--start-seed", type=int, default=0,
                        help="First seed index to run (default 0); use to resume a partial run")
    parser.add_argument("--system",  type=str,   default="lorenz",
                        choices=["lorenz", "rossler", "lorenz96"])
    parser.add_argument("--dry-run", action="store_true", help="Print config and exit")
    parser.add_argument("--qrc-only", action="store_true",
                        help="Skip QPINN (fast, for testing QRC multi-seed only)")
    args = parser.parse_args()

    if args.start_seed >= args.seeds:
        print(f"Error: --start-seed ({args.start_seed}) must be < --seeds ({args.seeds})")
        return

    n_to_run = args.seeds - args.start_seed
    seed_range_str = (f"seeds {args.start_seed}-{args.seeds-1}"
                      if args.start_seed > 0 else f"{args.seeds} seeds")

    print(f"\n{'='*60}")
    print(f"  Multi-seed benchmark: {args.system.upper()}, {seed_range_str}")
    print(f"  QRC:   {QRC_QUBITS}q, {QRC_LAYERS}L, window={QRC_WINDOW}")
    print(f"  QPINN: {QPINN_QUBITS}q, {QPINN_LAYERS}L, {QPINN_ITERS} iters")
    print(f"  Estimated wall time: ~{n_to_run * 3}h (QPINN) + <5min (QRC)")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Dry run — exiting.")
        return

    rhs, x0_base, dim = get_system(args.system)

    # Generate trajectory once (deterministic; seeds only affect quantum circuits)
    full_traj = integrate(rhs, x0_base, T_TEST_END, DT)
    n_total   = int(T_TRAIN_END / DT)
    n_test_end = int(T_TEST_END / DT)
    train_idx  = np.linspace(0, n_total - 1, N_TRAIN, dtype=int)
    test_idx   = np.linspace(n_total, n_test_end - 1, N_TEST, dtype=int)

    train_times  = train_idx * DT
    train_states = full_traj[train_idx, :dim]
    test_times   = test_idx * DT
    test_states  = full_traj[test_idx, :dim]

    seeds = list(range(args.start_seed, args.seeds))
    qrc_results, qpinn_results = [], []

    for i, seed in enumerate(seeds):
        print(f"\n-- Seed {args.start_seed + i + 1}/{args.seeds} (seed={seed}) --")

        print(f"  QRC...", end=" ", flush=True)
        tr_mse, te_mse, t = run_qrc(train_states, test_states, seed, dim)
        qrc_results.append({"seed": seed, "train_mse": tr_mse, "test_mse": te_mse, "time_s": t})
        print(f"train_mse={tr_mse:.3f}, test_mse={te_mse:.3f}, t={t:.1f}s")

        if not args.qrc_only:
            print(f"  QPINN...", end=" ", flush=True)
            tr_mse, te_mse, t = run_qpinn(train_times, train_states,
                                            test_times, test_states, seed, dim)
            qpinn_results.append({"seed": seed, "train_mse": tr_mse, "test_mse": te_mse, "time_s": t})
            print(f"train_mse={tr_mse:.3f}, test_mse={te_mse:.3f}, t={t:.1f}s")

    # ── Summary statistics ───────────────────────────────────────────────────
    def summarise(results, label):
        vals = np.array([r["train_mse"] for r in results if not np.isnan(r["train_mse"])])
        test = np.array([r["test_mse"]  for r in results if not np.isnan(r["test_mse"])])
        times = np.array([r["time_s"]   for r in results if not np.isnan(r["time_s"])])
        print(f"\n{label}:")
        print(f"  Train MSE : {vals.mean():.3f} ± {vals.std():.3f}  (n={len(vals)})")
        print(f"  Test  MSE : {test.mean():.3f} ± {test.std():.3f}  (n={len(test)})")
        print(f"  Time (s)  : {times.mean():.1f} ± {times.std():.1f}")
        return vals

    print(f"\n{'='*60}")
    qrc_train  = summarise(qrc_results,   "QRC")
    qpinn_train = summarise(qpinn_results, "QPINN") if qpinn_results else None

    if qpinn_train is not None and len(qrc_train) == len(qpinn_train):
        stat, pval = stats.wilcoxon(qrc_train, qpinn_train)
        print(f"\nWilcoxon signed-rank test (train MSE): p={pval:.4f}")
        improvement = (qpinn_train.mean() - qrc_train.mean()) / qpinn_train.mean() * 100
        print(f"Mean MSE reduction: {improvement:.1f}%")

    # ── Save results ─────────────────────────────────────────────────────────
    out = {
        "system":       args.system,
        "n_seeds":      args.seeds,
        "start_seed":   args.start_seed,
        "qrc_config":   {"qubits": QRC_QUBITS, "layers": QRC_LAYERS, "window": QRC_WINDOW},
        "qpinn_config": {"qubits": QPINN_QUBITS, "layers": QPINN_LAYERS, "iters": QPINN_ITERS},
        "qrc_results":  qrc_results,
        "qpinn_results": qpinn_results,
    }
    if args.start_seed > 0:
        out_path = RESULTS_DIR / f"seeds_{args.system}_s{args.start_seed}to{args.seeds-1}.json"
    else:
        out_path = RESULTS_DIR / f"seeds_{args.system}_{args.seeds}seeds.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
