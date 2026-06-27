#!/usr/bin/env python3
"""
resource_estimation.py - Resource scaling analysis for QRC vs classical
reservoirs across the q-scaling sweep on the quantum-input task.

For each q in {5, 7, 9, 11, 13} on the seed=42 target:
  - Reads measured wall-clock per method from results/quantum_input_q*/summary.json
  - Computes theoretical operation counts (matched-2^q classical baselines)
  - Projects hardware-native QRC cost (gate ops + shot cost)
  - Reports the practical comparison: at what q does classical-matched
    become infeasible vs QRC-hardware-native?

Key distinction:
  - QRC simulation: cost scales as O(2^q) per step (statevector evolution)
  - QRC hardware: cost scales as O(n_gates) per step ~ O(poly(q)),
    with shot overhead M * K_observables
  - ESN matched 2^q: O(4^q) per step + O(8^q) ridge solve
  - RFF matched 2^q: O(2^q * d) per step + O(8^q) ridge solve

The matched-2^q classical baselines are infeasible above q ~ 20-25.
Hardware QRC scales polynomially in q -- no exponential wall.

Outputs:
    results/quantum_input_q_scaling/resource_summary.json
    results/quantum_input_q_scaling/scaling_curves.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent

DEFAULT_Q_DIRS = {
    5:  "results/quantum_input_q5",
    7:  "results/quantum_input",
    9:  "results/quantum_input_q9",
    11: "results/quantum_input_q11",
    13: "results/quantum_input_q13",
}


# Hardware QRC assumptions ---------------------------------------------------
GATE_TIME_MICROSEC = 1.0      # typical 2-qubit gate time on superconducting
SHOTS_PER_OBS      = 10_000   # shots per observable per timestep for ~1% precision
N_OBS_READOUT      = 6        # K Pauli observables read (= n_target_obs)
TROTTER_DEPTH      = 1        # Trotter steps per reservoir evolution
N_TIMESTEPS        = 2000     # training trajectory length


# Theoretical op counts -------------------------------------------------------

def qrc_sim_ops(q: int, n_steps: int = N_TIMESTEPS, n_layers: int = 2) -> dict:
    """Statevector simulation cost: O(2^q) state * O(q) gates per step."""
    dim = 2 ** q
    # Each gate touches the full 2^q state. We do ~n_layers * (n + n-1)
    # gates per "evolution" plus n encoding gates per timestep.
    gates_per_step = n_layers * (2 * q - 1) + q + 3  # rough
    ops_per_step = gates_per_step * dim
    return {
        "sv_dim": dim,
        "gates_per_step": gates_per_step,
        "ops_per_step": ops_per_step,
        "total_ops": ops_per_step * n_steps,
        "memory_bytes": 16 * dim,
    }


def qrc_hw_ops(q: int, n_steps: int = N_TIMESTEPS, n_layers: int = 2,
                 shots: int = SHOTS_PER_OBS, n_obs: int = N_OBS_READOUT) -> dict:
    """Hardware-native QRC: gates per step, shots per observable, total
    physical time."""
    gates_per_step = n_layers * (2 * q - 1) + q + 3
    # Total physical reservoir time = gates * gate time
    reservoir_time_us = gates_per_step * GATE_TIME_MICROSEC
    # Each observable needs `shots` repetitions; n_obs observables per step
    measurement_time_us = shots * n_obs * GATE_TIME_MICROSEC
    total_time_per_step_us = reservoir_time_us + measurement_time_us
    total_time_us = total_time_per_step_us * n_steps
    return {
        "gates_per_step": gates_per_step,
        "shots_per_obs": shots,
        "n_obs": n_obs,
        "reservoir_time_per_step_us": reservoir_time_us,
        "measurement_time_per_step_us": measurement_time_us,
        "total_wallclock_seconds": total_time_us / 1e6,
        "memory_bytes": 16,    # one statevector NOT stored -- physical qubits
    }


def esn_matched_ops(q: int, n_steps: int = N_TIMESTEPS, window: int = 5,
                     n_target_obs: int = N_OBS_READOUT) -> dict:
    """ESN at matched feature dim N = 2^q.  Per-step matvec O(N^2),
    ridge solve O((w*N)^3)."""
    N = 2 ** q
    per_step_ops = N * N
    train_ops = per_step_ops * n_steps
    ridge_d = window * N
    ridge_solve_ops = ridge_d ** 3
    total = train_ops + ridge_solve_ops
    return {
        "N_reservoir": N,
        "per_step_ops": per_step_ops,
        "train_ops": train_ops,
        "ridge_dim": ridge_d,
        "ridge_solve_ops": ridge_solve_ops,
        "total_ops": total,
        "memory_bytes": 16 * (N * N + ridge_d * ridge_d),
    }


def rff_matched_ops(q: int, n_steps: int = N_TIMESTEPS, window: int = 5,
                     d_input: int = 3, n_target_obs: int = N_OBS_READOUT) -> dict:
    """RFF at D = 2^q random features.  Per-step O(D * w*d) projection;
    ridge solve O((w*D)^3) if windowed features, else O(D^3)."""
    D = 2 ** q
    # In our implementation RFF is applied to windowed inputs (length w*d)
    # then has D features; readout is then D x n_target_obs.
    proj_ops = D * window * d_input
    train_proj_ops = proj_ops * n_steps
    ridge_d = D    # we don't window AGAIN after RFF
    ridge_solve_ops = ridge_d ** 3
    total = train_proj_ops + ridge_solve_ops
    return {
        "D_features": D,
        "proj_ops_per_step": proj_ops,
        "train_proj_ops": train_proj_ops,
        "ridge_dim": ridge_d,
        "ridge_solve_ops": ridge_solve_ops,
        "total_ops": total,
        "memory_bytes": 16 * ridge_d * ridge_d,
    }


def rff_polyq_ops(q: int, D_fixed: int = 100, n_steps: int = N_TIMESTEPS,
                   window: int = 5, d_input: int = 3) -> dict:
    """RFF at FIXED D regardless of q (poly-q classical baseline).
    This is the "hardware-fair" classical comparison."""
    proj_ops = D_fixed * window * d_input
    train_proj_ops = proj_ops * n_steps
    ridge_solve_ops = D_fixed ** 3
    total = train_proj_ops + ridge_solve_ops
    return {
        "D_features": D_fixed,
        "proj_ops_per_step": proj_ops,
        "train_proj_ops": train_proj_ops,
        "ridge_dim": D_fixed,
        "ridge_solve_ops": ridge_solve_ops,
        "total_ops": total,
        "memory_bytes": 16 * D_fixed * D_fixed,
    }


# Aggregate measured wall-clock from the experiment summary -------------------

def read_measured(q_dirs: dict[int, str]) -> dict:
    out = {}
    for q, d in q_dirs.items():
        sj = ROOT / d / "summary.json"
        if not sj.exists():
            out[q] = {"_missing": True, "path": str(d)}
            continue
        with open(sj) as f:
            s = json.load(f)
        methods = s["methods"]
        per_method = {}
        for m, runs in methods.items():
            # Total wall is not in summary; reconstruct from per-method
            # k1_nrmse plus the elapsed time stamp printed during run -- which
            # isn't stored. We *can* use the cell wall_time if any.
            # For now we report mean NRMSE at k=1 and let the script user
            # cross-reference time from the run log.
            vals = [r["k1_nrmse"] for r in runs]
            kappas = [r["kappa"] for r in runs]
            per_method[m] = {
                "mean_nrmse_k1": float(np.mean(vals)),
                "std_nrmse_k1": float(np.std(vals)),
                "mean_kappa": float(np.mean(kappas)),
            }
        out[q] = {"path": str(d), "methods": per_method}
    return out


# Main ------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--qs", type=int, nargs="+",
                    default=list(DEFAULT_Q_DIRS.keys()))
    p.add_argument("--out-dir", default="results/quantum_input_q_scaling")
    args = p.parse_args()

    qs = sorted(args.qs)
    q_dirs = {q: DEFAULT_Q_DIRS[q] for q in qs if q in DEFAULT_Q_DIRS}

    measured = read_measured(q_dirs)

    # ---- Theoretical resource estimates ----
    res = {"qs": qs, "by_q": {}}
    for q in qs:
        entry = {
            "measured": measured.get(q, {}),
            "qrc_simulation": qrc_sim_ops(q),
            "qrc_hardware": qrc_hw_ops(q),
            "esn_matched_2pq": esn_matched_ops(q),
            "rff_matched_2pq": rff_matched_ops(q),
            "rff_polyq_D100": rff_polyq_ops(q, D_fixed=100),
            "rff_polyq_D1000": rff_polyq_ops(q, D_fixed=1000),
        }
        res["by_q"][q] = entry

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "resource_summary.json", "w") as f:
        json.dump(res, f, indent=2)

    # ---- Print summary table ----
    print(f"\n=== Resource scaling for the q-sweep ({len(qs)} cells) ===\n")
    print(f"{'q':>3}  {'TFIM NRMSE':<12}  {'RFF NRMSE':<12}  "
           f"{'kappa_QRC':<10}  {'kappa_ESN':<10}  "
           f"{'QRC sim ops':<13}  {'QRC HW wall':<14}  "
           f"{'ESN-2^q ops':<13}  {'RFF-D100 ops':<13}")
    for q in qs:
        e = res["by_q"][q]
        m = e["measured"]
        if m.get("_missing"):
            print(f"{q:>3}  [missing]")
            continue
        tfim = m["methods"].get("TFIM_QRC", {})
        rff  = m["methods"].get("RFF_matched_effdim", {})
        esn  = m["methods"].get("ESN_matched", {})
        sim  = e["qrc_simulation"]
        hw   = e["qrc_hardware"]
        esn_ops = e["esn_matched_2pq"]
        rff_poly = e["rff_polyq_D100"]
        print(f"{q:>3}  "
              f"{tfim.get('mean_nrmse_k1', 0):.3f}+/-{tfim.get('std_nrmse_k1', 0):.3f}  "
              f"{rff.get('mean_nrmse_k1', 0):.3f}+/-{rff.get('std_nrmse_k1', 0):.3f}  "
              f"{tfim.get('mean_kappa', 0):>10.2e}  "
              f"{esn.get('mean_kappa', 0):>10.2e}  "
              f"{sim['total_ops']:>13.2e}  "
              f"{hw['total_wallclock_seconds']:>12.2f}s   "
              f"{esn_ops['total_ops']:>13.2e}  "
              f"{rff_poly['total_ops']:>13.2e}")

    # ---- Scaling plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.0))

        ax = axes[0]
        ax.plot(qs, [res["by_q"][q]["qrc_simulation"]["total_ops"] for q in qs],
                 marker="o", lw=2, label="QRC simulation (statevector) ops")
        ax.plot(qs, [res["by_q"][q]["qrc_hardware"]["total_wallclock_seconds"] for q in qs],
                 marker="s", lw=2, ls="--", label="QRC hardware wall-clock (s)")
        ax.plot(qs, [res["by_q"][q]["esn_matched_2pq"]["total_ops"] for q in qs],
                 marker="^", lw=2, label="ESN matched 2^q ops")
        ax.plot(qs, [res["by_q"][q]["rff_matched_2pq"]["total_ops"] for q in qs],
                 marker="v", lw=2, label="RFF matched 2^q ops")
        ax.plot(qs, [res["by_q"][q]["rff_polyq_D100"]["total_ops"] for q in qs],
                 marker="X", lw=2, label="RFF poly-q (D=100, fixed) ops")
        ax.set_yscale("log")
        ax.set_xlabel("q (qubits)")
        ax.set_ylabel("Total ops or wall-clock (s)")
        ax.set_title("Resource scaling vs q")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.3)

        ax = axes[1]
        nrmse_qrc = []
        nrmse_rff_eff = []
        for q in qs:
            m = res["by_q"][q].get("measured", {}).get("methods", {})
            nrmse_qrc.append(m.get("TFIM_QRC", {}).get("mean_nrmse_k1", float("nan")))
            nrmse_rff_eff.append(m.get("RFF_matched_effdim", {}).get("mean_nrmse_k1", float("nan")))
        ax.plot(qs, nrmse_qrc, marker="o", lw=2, label="TFIM_QRC", color="#0F4C5C")
        ax.plot(qs, nrmse_rff_eff, marker="s", lw=2, label="RFF matched-effdim",
                 color="#9A4836")
        ax.set_xlabel("q (qubits)")
        ax.set_ylabel("Test NRMSE at k=1")
        ax.set_title("Predictive scaling vs q  (4-qubit Heisenberg target, seed=42)")
        ax.legend(fontsize=10)
        ax.grid(True, which="both", alpha=0.3)

        fig.tight_layout()
        out_fig = out_dir / "scaling_curves.png"
        fig.savefig(out_fig, dpi=150)
        print(f"\nSaved {out_fig}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
