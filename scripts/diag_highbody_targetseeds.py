#!/usr/bin/env python3
"""
diag_highbody_targetseeds.py - paper 6 HEADLINE experiment, 3x3 version.

Extends diag_highbody_target.py to the standing-rule 3 target seeds x
3 reservoir seeds, and adds paired bootstrap CIs on the QRC-vs-Shadows
gap at the headline cells (k in {4,6}, horizon=20).

Thesis: QRC reads O_i = U_dag Z_i U of the injected input as a LOCAL Z
measurement on the reservoir, so its shot variance is ~1/M regardless of
the body count of O_i in the input frame. A classical method estimating a
k-body target from classical shadows pays variance 3^k/M. So QRC's edge
over shadow-based long-horizon prediction of a k-body target should be
largest at intermediate body count.

Setup: n_in=9, q=11, scrambling reservoir, multi-basis (ZXY) local readout,
tau=0.6, M=1024*q shots/step shared by all methods.

Methods (all share M shots/step):
  QRC          : windowed multi-basis local Pauli readout of the reservoir
                 (features computed ONCE, target-independent), ridge per target.
  Shadows      : classical shadows estimate the k-body target lag
                 (variance 3^k/M) + exact-ish 1-/2-body context (3^|P|/M), ridge.
  Cheat        : exact k-body lag + exact 1-,2-body context, ridge (upper bound).

Outputs: results/paper6_highbody_3x3/summary.json + console table.
"""
from __future__ import annotations
import sys, time, json
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
RES_SEEDS = [0, 1, 2]
TARGET_SEEDS = [42, 7, 123]
M = 1024 * Q
TRAIN_FRAC = 0.7
N_BOOT = 2000
HEADLINE_CELLS = [(4, 20), (6, 20)]   # (body, horizon) cells for bootstrap CI
OUT = ROOT / "results" / "paper6_highbody_3x3"


def split_idx(T, h, train_frac=TRAIN_FRAC):
    ntr = int(round(train_frac * T))
    return ntr


def fit_predict(F, y, h, train_frac=TRAIN_FRAC, alpha=1.0):
    """Return (y_pred, y_true) on the test horizon, or (None, None)."""
    T = len(F); ntr = int(round(train_frac * T))
    if T - h <= ntr + 5:
        return None, None
    W_ = ridge_solve(F[:ntr - h], y[h:ntr, None], alpha=alpha)
    yp = (F[ntr:T - h] @ W_).ravel()
    yte = y[ntr + h:T]
    return yp, yte


def nrmse_from_pred(yp, yte):
    return float(np.sqrt(np.mean((yp - yte) ** 2)) / (np.std(yte) + 1e-12))


def main():
    print(f"=== paper6 high-body 3x3: n_in={N_IN}, q={Q}, scram, tau={TAU}, "
          f"M={M}/step ===")
    print(f"    target seeds {TARGET_SEEDS} x reservoir seeds {RES_SEEDS}, "
          f"bodies {BODIES}, horizons {HORIZONS}")
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # results[method][k][h] = list of NRMSE over (target_seed, res_seed) pairs
    results = {m: {k: {h: [] for h in HORIZONS} for k in BODIES}
               for m in ["QRC", "Shadows", "Cheat"]}
    # store per-(headline cell) residuals for paired bootstrap, per pair
    #   boot_store[(k,h)] = list of (yp_qrc, yp_sh, yte) tuples (test-set arrays)
    boot_store = {cell: [] for cell in HEADLINE_CELLS}

    for tseed in TARGET_SEEDS:
        H_tgt = build_target_hamiltonian("heisenberg", N_IN, tseed, disorder=0.5)
        psi0 = np.ones(2 ** N_IN, dtype=complex) / np.sqrt(2 ** N_IN)
        psi_t = evolve_states(H_tgt, psi0, np.linspace(0, TMAX, NSTEPS))

        # exact context Paulis (1- and 2-body), computed once per target
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
        # exact k-body targets
        tgt_exact = {}
        for k in BODIES:
            s = ["I"] * N_IN
            for i in range(k): s[i] = "X"
            P = pauli_string("".join(s))
            tgt_exact[k] = np.array([float(np.real(p.conj() @ (P @ p)))
                                      for p in psi_t])

        for seed in RES_SEEDS:
            H_res = scrambling_hamiltonian(Q, seed=seed)
            rng_q = np.random.default_rng(1000 + seed)
            step, D = make_qrc(Q, N_IN, H_res, TAU, 1024, rng_q,
                                bases=("Z", "X", "Y"))
            Fq = windowed_features(np.array([step(p) for p in psi_t]), W)

            rng_s = np.random.default_rng(2000 + 100 * tseed + seed)
            ctx_sigma = np.sqrt(3.0 ** ctx_w / M)
            ctx_shadow = ctx_exact + rng_s.standard_normal(ctx_exact.shape) * ctx_sigma

            for k in BODIES:
                y = tgt_exact[k]; yw = y[W:]
                tgt_sig = np.sqrt(3.0 ** k / M)
                tgt_shadow = y + rng_s.standard_normal(len(y)) * tgt_sig
                F_sh = windowed_features(
                    np.column_stack([tgt_shadow[:, None], ctx_shadow]), W)
                F_ch = windowed_features(
                    np.column_stack([y[:, None], ctx_exact]), W)
                for h in HORIZONS:
                    yp_q, yte_q = fit_predict(Fq, yw, h)
                    yp_s, yte_s = fit_predict(F_sh, yw, h)
                    yp_c, yte_c = fit_predict(F_ch, yw, h)
                    if yp_q is not None:
                        results["QRC"][k][h].append(nrmse_from_pred(yp_q, yte_q))
                        results["Shadows"][k][h].append(nrmse_from_pred(yp_s, yte_s))
                        results["Cheat"][k][h].append(nrmse_from_pred(yp_c, yte_c))
                        if (k, h) in HEADLINE_CELLS:
                            boot_store[(k, h)].append((yp_q, yp_s, yte_q))
            print(f"  tseed={tseed} rseed={seed} done "
                  f"({time.time()-t_start:.0f}s)")

    # ---- aggregate report ----
    n_runs = len(TARGET_SEEDS) * len(RES_SEEDS)
    print(f"\n=== Mean +/- std NRMSE over {n_runs} runs "
          f"({len(TARGET_SEEDS)} target x {len(RES_SEEDS)} reservoir seeds) ===")
    print(f"{'body k':>7} {'horizon':>8} | {'QRC':>14} {'Shadows':>14} "
          f"{'Cheat':>14} | verdict")
    print("-" * 78)
    summary = {"config": {"n_in": N_IN, "q": Q, "tau": TAU, "M": M,
                            "nsteps": NSTEPS, "window": W,
                            "target_seeds": TARGET_SEEDS,
                            "res_seeds": RES_SEEDS, "bodies": BODIES,
                            "horizons": HORIZONS, "train_frac": TRAIN_FRAC},
                "cells": {}}
    for k in BODIES:
        for h in HORIZONS:
            q_ = np.array(results["QRC"][k][h])
            s_ = np.array(results["Shadows"][k][h])
            c_ = np.array(results["Cheat"][k][h])
            verdict = "QRC wins" if q_.mean() < s_.mean() else "shadows"
            qpred = q_.mean() < 1.0
            tag = f"k{k}_h{h}"
            summary["cells"][tag] = {
                "body": k, "horizon": h,
                "QRC_mean": float(q_.mean()), "QRC_std": float(q_.std()),
                "Shadows_mean": float(s_.mean()), "Shadows_std": float(s_.std()),
                "Cheat_mean": float(c_.mean()), "Cheat_std": float(c_.std()),
                "QRC_predictive": bool(qpred),
                "verdict": verdict,
            }
            star = " *" if (q_.mean() < 1.0 and s_.mean() > 1.0) else ""
            print(f"{k:>7} {h:>8} | {q_.mean():>6.3f}+/-{q_.std():<5.2f} "
                  f"{s_.mean():>6.3f}+/-{s_.std():<5.2f} "
                  f"{c_.mean():>6.3f}+/-{c_.std():<5.2f} | {verdict}{star}")
        print()

    # ---- paired bootstrap CIs on the headline cells ----
    print(f"=== Paired bootstrap CIs (N={N_BOOT}) on headline cells ===")
    print("gap = NRMSE(Shadows) - NRMSE(QRC); positive => QRC better.")
    rng_b = np.random.default_rng(99)
    summary["bootstrap"] = {}
    for cell in HEADLINE_CELLS:
        k, h = cell
        # pool test residuals across runs by concatenation (paired by index)
        yp_q = np.concatenate([t[0] for t in boot_store[cell]])
        yp_s = np.concatenate([t[1] for t in boot_store[cell]])
        yte = np.concatenate([t[2] for t in boot_store[cell]])
        n = len(yte)
        std_y = np.std(yte) + 1e-12
        gaps = np.empty(N_BOOT)
        for b in range(N_BOOT):
            idx = rng_b.integers(0, n, n)
            yt = yte[idx]; sy = np.std(yt) + 1e-12
            nq = np.sqrt(np.mean((yp_q[idx] - yt) ** 2)) / sy
            ns = np.sqrt(np.mean((yp_s[idx] - yt) ** 2)) / sy
            gaps[b] = ns - nq
        lo, med, hi = np.percentile(gaps, [2.5, 50, 97.5])
        frac_pos = float((gaps > 0).mean())
        summary["bootstrap"][f"k{k}_h{h}"] = {
            "gap_median": float(med), "gap_lo95": float(lo),
            "gap_hi95": float(hi), "frac_QRC_better": frac_pos}
        sig = "SIGNIFICANT (CI excludes 0)" if lo > 0 else "not significant"
        print(f"  k={k} h={h}: gap = {med:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(QRC better)={frac_pos:.3f}  -> {sig}")

    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {OUT / 'summary.json'}  (total {time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
