#!/usr/bin/env python3
"""
diag_highbody_target.py - paper 6, the structural-advantage experiment.

Thesis: QRC reads O_i = U_dag Z_i U of the injected input as a LOCAL Z
measurement on the reservoir, so its shot variance is ~1/M regardless of
how high-body O_i is in the input frame. A classical method estimating a
k-body target observable directly from classical shadows pays variance
3^k / M. Therefore QRC's advantage over shadow-based prediction of a
k-body target should GROW with the body count k.

Setup (n_in=9, q=11, scrambling reservoir, multi-basis local readout):
  Target: <X_0 X_1 ... X_{k-1}> of the input state at horizon t+h, sweeping
          body count k in {2,4,6,8}.
  QRC          : windowed multi-basis local Pauli readout of the reservoir,
                 features computed ONCE (target-independent), ridge per target.
  Shadows_direct: classical shadows estimate the k-body target lag
                 (variance 3^k/M) + 1- and 2-body context, ridge.
  Cheat        : exact k-body lag + exact 1-,2-body context, ridge (upper bound,
                 isolates the shadow-noise cost from feature informativeness).

All methods share M = shots_per_qubit * q shots per timestep.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT / "scripts"))
from run_quantum_input_experiment import (
    build_target_hamiltonian, evolve_states, scrambling_hamiltonian,
    single_site, two_site, pauli_string, ridge_solve, windowed_features)
from diag_multibasis_readout import make_qrc

N_IN, Q, NSTEPS, TMAX, W = 9, 11, 1000, 60.0, 5
TAU = 0.6
BODIES = [2, 4, 6, 8]
HORIZONS = [1, 5, 20]
SEEDS = [0, 1, 2]
M = 1024 * Q


def nrmse_single(F, y, h, train_frac=0.7, alpha=1.0):
    T = len(F); ntr = int(round(train_frac * T))
    if T - h <= ntr + 5: return float("nan")
    W_ = ridge_solve(F[:ntr - h], y[h:ntr, None], alpha=alpha)
    yp = (F[ntr:T - h] @ W_).ravel(); yte = y[ntr + h:T]
    return float(np.sqrt(np.mean((yp - yte) ** 2)) / (np.std(yte) + 1e-12))


def main():
    print(f"=== high-body target sweep: n_in={N_IN}, q={Q}, scram, tau={TAU}, "
          f"M={M}/step, bodies={BODIES} ===")
    H_tgt = build_target_hamiltonian("heisenberg", N_IN, 42, disorder=0.5)
    psi0 = np.ones(2 ** N_IN, dtype=complex) / np.sqrt(2 ** N_IN)
    psi_t = evolve_states(H_tgt, psi0, np.linspace(0, TMAX, NSTEPS))

    # --- exact context Paulis (1- and 2-body, seed-independent), computed once ---
    t0 = time.time()
    ctx_ops, ctx_w = [], []
    for i in range(N_IN):
        for pl in ("X", "Y", "Z"):
            ctx_ops.append(single_site(pl, i, N_IN)); ctx_w.append(1)
    for i in range(N_IN):
        for j in range(i + 1, N_IN):
            for pl in ("X", "Y", "Z"):
                ctx_ops.append(two_site(pl, i, pl, j, N_IN)); ctx_w.append(2)
    ctx_w = np.array(ctx_w)
    ctx_exact = np.array([[float(np.real(p.conj() @ (O @ p))) for O in ctx_ops]
                          for p in psi_t])
    # --- exact k-body targets, once each ---
    tgt_exact = {}
    for k in BODIES:
        s = ["I"] * N_IN
        for i in range(k): s[i] = "X"
        P = pauli_string("".join(s))
        tgt_exact[k] = np.array([float(np.real(p.conj() @ (P @ p))) for p in psi_t])
    print(f"  exact features+targets ({time.time()-t0:.0f}s); "
          f"target stds: " + ", ".join(f"k{k}={tgt_exact[k].std():.3f}" for k in BODIES))

    # --- per-seed: QRC features + shadow noise ---
    results = {m: {k: {h: [] for h in HORIZONS} for k in BODIES}
               for m in ["QRC", "Shadows", "Cheat"]}
    for seed in SEEDS:
        H_res = scrambling_hamiltonian(Q, seed=seed)
        rng_q = np.random.default_rng(1000 + seed)
        step, D = make_qrc(Q, N_IN, H_res, TAU, 1024, rng_q, bases=("Z", "X", "Y"))
        Fq = windowed_features(np.array([step(p) for p in psi_t]), W)

        rng_s = np.random.default_rng(2000 + seed)
        # shadow context: exact + Gaussian(sqrt(3^|P|/M))
        ctx_sigma = np.sqrt(3.0 ** ctx_w / M)
        ctx_shadow = ctx_exact + rng_s.standard_normal(ctx_exact.shape) * ctx_sigma

        for k in BODIES:
            y = tgt_exact[k]
            yw = y[W:]
            # QRC: target-independent reservoir features
            for h in HORIZONS:
                results["QRC"][k][h].append(nrmse_single(Fq, yw, h))
            # shadow target-lag: variance 3^k/M
            tgt_sig = np.sqrt(3.0 ** k / M)
            tgt_shadow = y + rng_s.standard_normal(len(y)) * tgt_sig
            F_sh = windowed_features(
                np.column_stack([tgt_shadow[:, None], ctx_shadow]), W)
            F_ch = windowed_features(
                np.column_stack([y[:, None], ctx_exact]), W)
            for h in HORIZONS:
                results["Shadows"][k][h].append(nrmse_single(F_sh, yw, h))
                results["Cheat"][k][h].append(nrmse_single(F_ch, yw, h))
        print(f"  seed {seed} done")

    # --- report ---
    print(f"\n{'body k':>7} {'horizon':>8} | {'QRC':>12} {'Shadows':>12} "
          f"{'Cheat':>12} | {'QRC<Shadows?':>13}")
    print("-" * 74)
    for k in BODIES:
        for h in HORIZONS:
            q_ = np.mean(results["QRC"][k][h])
            s_ = np.mean(results["Shadows"][k][h])
            c_ = np.mean(results["Cheat"][k][h])
            qs = np.std(results["QRC"][k][h]); ss = np.std(results["Shadows"][k][h])
            verdict = "QRC wins" if q_ < s_ else "shadows"
            print(f"{k:>7} {h:>8} | {q_:>6.3f}+/-{qs:<4.2f} {s_:>6.3f}+/-{ss:<4.2f} "
                  f"{c_:>11.3f} | {verdict:>13}")
        print()
    print("NRMSE, lower=better, <1.0=predictive. Hypothesis: QRC-vs-Shadows gap "
          "widens as body count k grows (3^k/M shadow cost).")


if __name__ == "__main__":
    main()
