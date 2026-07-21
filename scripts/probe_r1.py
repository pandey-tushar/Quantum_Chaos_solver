#!/usr/bin/env python3
"""
probe_r1.py -- R1 revision PROBE: measure per-unit runtimes of the QPINN
code so we can budget referee-demanded experiments. PROBES ONLY.

Reuses the exact physics from scripts/run_qpinn_gradient_logging.py
(4-qubit / 3-layer / 45-param statevector QPINN, the code path that
produced the paper baseline of ~2.4 h / 200 FD iters).

Units measured (>=3 reps each, report mean):
  1. t_loss       one full physics+boundary loss eval, default 4q/3L, n_train collocation pts
  2. t_fd_grad    one forward finite-difference gradient (1 base + 45 = 46 loss evals)
  3. t_spsa_iter  one SPSA iteration (2 loss evals + update), minimal SPSA inline
  4. t_ps_grad    one parameter-shift gradient (2 evals/param = 90 loss evals)
                   -> derived 90*t_loss AND spot-checked with a 5-param partial PS
  5. t_jac        Jacobian d f_theta / d theta of the CIRCUIT OUTPUT (3 obs x 45 params)
                   via parameter-shift on the raw predict_state (2 evals/param = 90 state evals)
  6. t_loss_L     loss-eval time at L = 1, 2, 5, 8 layers

This file lives in a throwaway worktree and is never committed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

# Reuse the exact QPINN physics from the paper's entry point.
import run_qpinn_gradient_logging as qp  # noqa: E402
from qiskit import QuantumCircuit  # noqa: E402
from qiskit.quantum_info import Statevector  # noqa: E402


# ---------------------------------------------------------------------------
# Fixed problem setup, matching run_seed() defaults in the paper entry point.
# ---------------------------------------------------------------------------
SEED = 0
N_TRAIN = 50          # collocation points (run_seed default)
T_TRAIN_END = 3.0
X0 = np.array([1.0, 1.0, 1.0])
T_MAX = T_TRAIN_END
T_POINTS = np.linspace(0.0, T_TRAIN_END, N_TRAIN)

rng = np.random.default_rng(SEED)
THETA = rng.uniform(0.0, 2.0 * np.pi, qp.N_PARAMS)   # 45 params


def timeit(fn, reps):
    """Return (mean_s, std_s, [samples])."""
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.mean(samples)), float(np.std(samples)), samples


# ---------------------------------------------------------------------------
# Parametric circuit builder for the layer-scaling probe (t_loss_L).
# Mirrors run_qpinn_gradient_logging.build_qpinn_circuit gate-for-gate, but
# takes n_layers as a parameter. 15 params/layer (12 single-qubit + 3 RZ).
# ---------------------------------------------------------------------------
N_QUBITS = qp.N_QUBITS  # 4


def build_qpinn_circuit_L(t, theta, t_max, n_layers, n_qubits=N_QUBITS):
    qc = QuantumCircuit(n_qubits)
    theta_t = 2.0 * np.pi * t / t_max
    qc.ry(float(theta_t), 0)
    qc.rz(float(theta_t), 1)
    qc.rx(float(theta_t), 2)
    qc.ry(float(np.pi * t / t_max), 3)
    p = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.rx(float(theta[p + 0]), q)
            qc.ry(float(theta[p + 1]), q)
            qc.rz(float(theta[p + 2]), q)
            p += 3
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
            qc.rz(float(theta[p]), q + 1)
            p += 1
    return qc


def predict_state_L(t, theta, t_max, n_layers):
    qc = build_qpinn_circuit_L(t, theta, t_max, n_layers)
    z0 = qp.expectation_z(qc, 0)
    z1 = qp.expectation_z(qc, 1)
    z2 = qp.expectation_z(qc, 2)
    x = (z0 + 1.0) / 2.0 * 40.0 - 20.0
    y = (z1 + 1.0) / 2.0 * 60.0 - 30.0
    z = (z2 + 1.0) / 2.0 * 50.0 + 0.0
    return np.array([x, y, z])


def physics_loss_L(theta, t_points, x0, t_max, n_layers, lam=10.0):
    L_phys = 0.0
    eps = 1e-3
    for t in t_points:
        u = predict_state_L(t, theta, t_max, n_layers)
        u_dot = (predict_state_L(t + eps, theta, t_max, n_layers)
                 - predict_state_L(t - eps, theta, t_max, n_layers)) / (2.0 * eps)
        F_u = qp.lorenz_rhs(u)
        L_phys += float(np.sum((u_dot - F_u) ** 2))
    L_phys /= len(t_points)
    u0 = predict_state_L(0.0, theta, t_max, n_layers)
    L_bc = float(np.sum((u0 - x0) ** 2))
    return L_phys + lam * L_bc


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("R1 PROBE: QPINN unit-cost timing (4q / 3L / 45 params)")
    print(f"  n_train (collocation) = {N_TRAIN}, t_end = {T_TRAIN_END}")
    print("=" * 70)

    results = {}

    # -- 1. t_loss --------------------------------------------------------
    def one_loss():
        return qp.physics_loss(THETA, T_POINTS, X0, T_MAX)

    # warm up (JIT/import caches) once, not counted
    one_loss()
    t_loss_mean, t_loss_std, t_loss_samples = timeit(one_loss, 5)
    results["t_loss"] = t_loss_mean
    print(f"\n[1] t_loss       = {t_loss_mean*1000:8.2f} ms  "
          f"(std {t_loss_std*1000:.2f}, n=5)  samples(ms)="
          f"{[round(s*1000,1) for s in t_loss_samples]}")

    # -- 2. t_fd_grad -----------------------------------------------------
    def one_fd_grad():
        return qp.finite_diff_per_parameter(THETA, T_POINTS, X0, T_MAX)

    t_fd_mean, t_fd_std, t_fd_samples = timeit(one_fd_grad, 3)
    results["t_fd_grad"] = t_fd_mean
    print(f"[2] t_fd_grad    = {t_fd_mean:8.3f} s   "
          f"(std {t_fd_std:.3f}, n=3)  "
          f"derived 46*t_loss = {46*t_loss_mean:.3f} s")

    # -- 3. t_spsa_iter (minimal SPSA inline) -----------------------------
    def one_spsa_iter():
        # Bernoulli +/-1 perturbation
        c = 0.1
        delta = rng.choice([-1.0, 1.0], size=qp.N_PARAMS)
        lp = qp.physics_loss(THETA + c * delta, T_POINTS, X0, T_MAX)
        lm = qp.physics_loss(THETA - c * delta, T_POINTS, X0, T_MAX)
        ghat = (lp - lm) / (2.0 * c) / delta   # SPSA gradient estimate
        # a fake update step (cost is negligible; included for completeness)
        _ = THETA - 1e-3 * ghat
        return ghat

    t_spsa_mean, t_spsa_std, t_spsa_samples = timeit(one_spsa_iter, 3)
    results["t_spsa_iter"] = t_spsa_mean
    print(f"[3] t_spsa_iter  = {t_spsa_mean:8.3f} s   "
          f"(std {t_spsa_std:.3f}, n=3)  "
          f"derived 2*t_loss = {2*t_loss_mean:.3f} s")

    # -- 4. t_ps_grad (parameter-shift): partial 5-param spot check -------
    #    Full PS = 2 evals/param * 45 params = 90 loss evals.
    #    Spot-check by measuring 5 params (10 evals) and extrapolate x9.
    def ps_grad_5params():
        shift = np.pi / 2.0
        g = np.zeros(5)
        for i in range(5):
            tp = THETA.copy(); tp[i] += shift
            tm = THETA.copy(); tm[i] -= shift
            lp = qp.physics_loss(tp, T_POINTS, X0, T_MAX)
            lm = qp.physics_loss(tm, T_POINTS, X0, T_MAX)
            g[i] = (lp - lm) / 2.0
        return g

    t_ps5_mean, t_ps5_std, _ = timeit(ps_grad_5params, 3)
    t_ps_full_extrap = t_ps5_mean * 9.0   # 5 params -> 45 params
    results["t_ps_grad"] = t_ps_full_extrap
    print(f"[4] t_ps_grad(5) = {t_ps5_mean:8.3f} s (10 evals)  "
          f"-> full 45p x9 = {t_ps_full_extrap:.3f} s   "
          f"derived 90*t_loss = {90*t_loss_mean:.3f} s")

    # -- 5. t_jac: Jacobian of circuit OUTPUT (3 obs x 45 params) ---------
    #    Parameter-shift on predict_state: 2 state-evals/param * 45 = 90 evals,
    #    each eval returns all 3 observables at once. Spot-check 5 params x9.
    def jac_5params():
        shift = np.pi / 2.0
        J = np.zeros((3, 5))
        # single reference time point (Jacobian is per-time; use t=T_TRAIN_END/2)
        tref = T_TRAIN_END / 2.0
        for i in range(5):
            tp = THETA.copy(); tp[i] += shift
            tm = THETA.copy(); tm[i] -= shift
            fp = qp.predict_state(tref, tp, T_MAX)
            fm = qp.predict_state(tref, tm, T_MAX)
            J[:, i] = (fp - fm) / 2.0
        return J

    t_jac5_mean, t_jac5_std, _ = timeit(jac_5params, 3)
    t_jac_extrap = t_jac5_mean * 9.0
    results["t_jac"] = t_jac_extrap
    print(f"[5] t_jac(5p)    = {t_jac5_mean*1000:8.2f} ms (10 state-evals) "
          f"-> full 45p x9 = {t_jac_extrap:.3f} s "
          f"(single time pt; x N_train for full-traj Jacobian)")

    # -- 6. t_loss_L for L = 1, 2, 5, 8 -----------------------------------
    print(f"\n[6] t_loss_L (loss eval at varying layer count):")
    t_loss_L = {}
    for L in (1, 2, 5, 8):
        nparams = 15 * L
        th = rng.uniform(0.0, 2.0 * np.pi, nparams)

        def one_loss_L(th=th, L=L):
            return physics_loss_L(th, T_POINTS, X0, T_MAX, L)

        one_loss_L()  # warm
        m, s, _ = timeit(one_loss_L, 3)
        t_loss_L[L] = m
        print(f"      L={L}  ({nparams:3d} params)  t_loss = {m*1000:8.2f} ms "
              f"(std {s*1000:.2f})")
    # sanity: L=3 via the parametric builder should match qp.physics_loss t_loss
    def loss_L3():
        return physics_loss_L(THETA, T_POINTS, X0, T_MAX, 3)
    loss_L3()
    m3, _, _ = timeit(loss_L3, 3)
    t_loss_L[3] = m3
    print(f"      L=3  ( 45 params)  t_loss = {m3*1000:8.2f} ms "
          f"(cross-check vs [1] t_loss={t_loss_mean*1000:.2f} ms)")
    results["t_loss_L"] = t_loss_L

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MACHINE-READABLE SUMMARY")
    print("=" * 70)
    import json
    print(json.dumps({k: v for k, v in results.items()}, indent=2))


if __name__ == "__main__":
    main()
