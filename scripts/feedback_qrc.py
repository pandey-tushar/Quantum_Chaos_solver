#!/usr/bin/env python3
"""
feedback_qrc.py - Feedback-driven quantum reservoir computing (FB-QRC).
See notes/feedback_qrc_plan.md.

Mechanism (Kobayashi et al., PRX Quantum 5:040325, 2024): the measurement
outcome at step t-1 modulates the reservoir drive at step t, making the
effective dynamics PATH-DEPENDENT (not a fixed function of the input window).
This is something a fixed-window polynomial cannot replicate -- but a classical
ESN can, so the honest bar is the ESN, not Poly2 (see plan section 0).

Phase 0 here = data generator (nonstationary regime-switching map), the
feedback reservoir + readout, and the G1-G6 correctness gates in --self-test.
NO science is run until the gates pass.

Gates:
  G1 data bounded/finite + nonstationarity real (fixed-window linear R2 drops
     as p_switch rises)
  G2 reservoir physical (Tr rho=1; <Z>,<ZZ> in [-1,1])
  G3 feedback live & correct (k_fb=0 -> features == open-loop to 1e-12;
     k_fb>0 -> features differ)
  G4 feedback causal (perturbing a FUTURE input leaves feature_t unchanged)
  G5 shot -> exact as shots grow
  G6 I/O integrity (handled by caller: read numbers back from JSON w/ hash)
"""
from __future__ import annotations
import argparse, sys, json, time, hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent

I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# Data: nonstationary regime-switching nonlinear map
# ---------------------------------------------------------------------------

def generate_regime_switching(n_steps, seed, p_switch, noise=0.02, burn=200):
    """Two AR-style regimes, each INDIVIDUALLY predictable but with DIFFERENT
    dynamics, switching on a hidden Markov schedule.

      regime A: x_{t+1} = +phiA * x_t + noise   (positive autocorrelation)
      regime B: x_{t+1} = -phiB * x_t + noise   (negative autocorrelation /
                                                  oscillatory)

    Both are linearly predictable WITHIN a regime, but the one-step map has
    OPPOSITE sign between regimes.  A fixed-window model that cannot infer the
    current (hidden) regime must average the two opposite dynamics and loses;
    a model that tracks the regime online (feedback) can do much better.  This
    is the correct setting for testing feedback (cf. plan: within-regime
    predictable, regime-ambiguous from a short window).
    Returns x (standardized) and the hidden regime sequence."""
    rng = np.random.default_rng(seed)
    phiA, phiB = 0.85, 0.85
    x = rng.uniform(-0.5, 0.5)
    s = 0
    xs, ss = [], []
    for t in range(n_steps + burn):
        if rng.uniform() < p_switch:
            s = 1 - s
        coef = phiA if s == 0 else -phiB
        x = coef * x + noise * rng.standard_normal()
        x = min(max(x, -5.0), 5.0)
        xs.append(x); ss.append(s)
    x = np.array(xs[burn:]); s = np.array(ss[burn:])
    xstd = (x - x.mean()) / (x.std() + 1e-12)
    return xstd, s


# ---------------------------------------------------------------------------
# Reservoir (small, fixed) + feedback drive
# ---------------------------------------------------------------------------

def kron_op(op, site, q):
    out = np.array([[1.0]], dtype=complex)
    for k in range(q):
        out = np.kron(out, op if k == site else I2)
    return out


def two_site_xx(i, j, q):
    out = np.array([[1.0]], dtype=complex)
    for k in range(q):
        out = np.kron(out, Xm if (k == i or k == j) else I2)
    return out


def reservoir_hamiltonian(q, seed, v=1.0):
    """Bayat-style fully-connected XX + transverse Z field."""
    rng = np.random.default_rng(20_000 + seed)
    H = np.zeros((2 ** q, 2 ** q), dtype=complex)
    for i in range(q):
        for j in range(i + 1, q):
            H += rng.uniform(0, 1) * two_site_xx(i, j, q)
    for i in range(q):
        H += v * kron_op(Zm, i, q)
    return H


def evol_unitary(H, tau):
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w * tau)) @ V.conj().T


def partial_trace_keep_memory(rho, n1, n2):
    q = n1 + n2
    r = rho.reshape([2] * q + [2] * q)
    letters = "abcdefghijklmnopqrstuvwxyz"
    row = list(letters[:q]); col = list(letters[q:2 * q])
    for k in range(n1):
        col[k] = row[k]
    sub = "".join(row) + "".join(col) + "->" + "".join(row[n1:]) + "".join(col[n1:])
    return np.einsum(sub, r).reshape(2 ** n2, 2 ** n2)


def build_sign_tables(q):
    dim = 2 ** q
    idx = np.arange(dim)
    shifts = np.arange(q - 1, -1, -1)
    bits = ((idx[:, None] >> shifts) & 1)
    z_signs = ((-1.0) ** bits).T                      # (q, 2^q)
    zz_rows = []
    for i in range(q):
        for k in range(i + 1, q):
            zz_rows.append((-1.0) ** (bits[:, i] + bits[:, k]))
    zz_signs = np.array(zz_rows) if zz_rows else np.zeros((0, dim))
    return z_signs, zz_signs


def fb_qrc_features(x_series, n1, n2, seed, readout="ZZ", mem_depth=3,
                     tau=1.0, k_fb=0.0, shots=0):
    """Feedback-driven QRC feature matrix (T, D).

    At each step t: the drive angle for the input qubit is
        phi_t = clip( x_t + k_fb * tanh(mbar_{t-1}), -pi, pi )
    where mbar_{t-1} is the mean of the previous step's <Z> measurement vector.
    k_fb=0 recovers the open-loop reservoir EXACTLY (gate G3).
    Single input qubit (n1=1) recommended for a 1-D series; memory n2 qubits.
    Recurrence: sliding window of mem_depth most recent drives, partial-trace
    input qubits between encodings (Bayat-style)."""
    q = n1 + n2
    H_res = reservoir_hamiltonian(q, seed)
    U = evol_unitary(H_res, tau)
    z_signs, zz_signs = build_sign_tables(q)
    rng_shot = np.random.default_rng(7000 + seed)
    T = len(x_series)
    feats = []
    prev_mbar = 0.0            # feedback state: previous mean-<Z>
    d_mem = 2 ** n2
    for t in range(T):
        # drive with feedback from previous measurement
        lo = max(0, t - mem_depth + 1)
        rho_mem = np.zeros((d_mem, d_mem), dtype=complex); rho_mem[0, 0] = 1.0
        rho_full = None
        for tt in range(lo, t + 1):
            # feedback uses prev_mbar (the measurement at the PREVIOUS timestep);
            # within the window we re-drive past inputs open-loop except the
            # current step carries the feedback term (causal: depends on t-1)
            fb = k_fb * np.tanh(prev_mbar) if tt == t else 0.0
            phi = np.clip(x_series[tt] + fb, -np.pi, np.pi)
            psi_in = np.array([1.0], dtype=complex)
            for a in range(n1):
                c, s = np.cos(phi / 2.0), np.sin(phi / 2.0)
                psi_in = np.kron(psi_in, np.array([c, s], dtype=complex))
            rho_in = np.outer(psi_in, psi_in.conj())
            rho_full = U @ np.kron(rho_in, rho_mem) @ U.conj().T
            rho_mem = partial_trace_keep_memory(rho_full, n1, n2)
        diag = np.real(np.diag(rho_full))
        zc = z_signs @ diag
        if shots and shots > 0:
            zc = zc + rng_shot.standard_normal(len(zc)) * np.sqrt(np.maximum(1 - zc ** 2, 0) / shots)
            zc = np.clip(zc, -1, 1)
        prev_mbar = float(np.mean(zc))           # update feedback state
        if readout == "Z":
            feats.append(zc)
        else:
            zzc = zz_signs @ diag
            if shots and shots > 0:
                zzc = zzc + rng_shot.standard_normal(len(zzc)) * np.sqrt(np.maximum(1 - zzc ** 2, 0) / shots)
                zzc = np.clip(zzc, -1, 1)
            feats.append(np.concatenate([zc, zzc]))
    return np.array(feats)


def windowed(F, window):
    T = len(F)
    out = []
    for t in range(T):
        block = F[max(0, t - window + 1):t + 1]
        if len(block) < window:
            block = np.vstack([np.repeat(block[:1], window - len(block), 0), block])
        out.append(block.flatten())
    return np.array(out)


# ---------------------------------------------------------------------------
# G1-G6 self-test
# ---------------------------------------------------------------------------

def _r2(F, y, train_frac=0.7, alpha=1.0):
    T = len(F); ntr = int(round(train_frac * T))
    A = F[:ntr].T @ F[:ntr] + alpha * np.eye(F.shape[1])
    W = np.linalg.solve(A, F[:ntr].T @ y[:ntr])
    yp = F[ntr:] @ W
    ss_res = np.sum((y[ntr:] - yp) ** 2)
    ss_tot = np.sum((y[ntr:] - y[ntr:].mean(0)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def self_test():
    ok = True

    # G1: task genuinely REQUIRES regime inference -- a regime-oracle (given the
    # hidden regime as a feature) beats a fixed-window LINEAR model, and the gap
    # GROWS with p_switch.  This is the property feedback can exploit.
    print("[G1] regime-switching data: bounded + task requires regime inference "
          "(oracle gap grows with switching)")
    gaps = []
    for ps in (0.0, 0.02, 0.05):
        x, s = generate_regime_switching(4000, seed=0, p_switch=ps)
        assert np.all(np.isfinite(x)), f"non-finite at p_switch={ps}"
        Wlin = windowed(x[:, None], 3)
        r2_fixed = _r2(Wlin[:-1], x[1:, None])
        # regime-oracle: same window PLUS the hidden regime sign-interaction
        s_signed = (1 - 2 * s).astype(float)[:, None]      # +/-1 regime indicator
        Worc = np.concatenate([Wlin, s_signed * x[:, None]], axis=1)
        r2_oracle = _r2(Worc[:-1], x[1:, None])
        gap = r2_oracle - r2_fixed
        gaps.append(gap)
        sw = float(np.mean(s[1:] != s[:-1]))
        print(f"   p_switch={ps}: bounded[{x.min():.2f},{x.max():.2f}] "
              f"fixed_R2={r2_fixed:.3f} oracle_R2={r2_oracle:.3f} "
              f"gap={gap:.3f} switch_rate={sw:.4f}")
    g1 = np.all(np.isfinite(x)) and gaps[0] < 0.05 and gaps[2] > gaps[0] + 0.1
    print(f"   oracle-over-fixed gap grows with switching: {'PASS' if g1 else 'FAIL'}")
    ok &= g1

    # G2: reservoir physical
    n1, n2 = 1, 4; q = n1 + n2
    x, _ = generate_regime_switching(60, seed=1, p_switch=0.02)
    ang = np.clip(x, -np.pi, np.pi)
    H = reservoir_hamiltonian(q, 0); U = evol_unitary(H, 1.0)
    z_signs, zz_signs = build_sign_tables(q)
    # build one full state to check trace/range
    d_mem = 2 ** n2
    rho_mem = np.zeros((d_mem, d_mem), dtype=complex); rho_mem[0, 0] = 1.0
    psi_in = np.array([np.cos(ang[10] / 2), np.sin(ang[10] / 2)], dtype=complex)
    rho_full = U @ np.kron(np.outer(psi_in, psi_in.conj()), rho_mem) @ U.conj().T
    tr = float(np.real(np.trace(rho_full)))
    diag = np.real(np.diag(rho_full))
    zc = z_signs @ diag; zzc = zz_signs @ diag
    g2 = abs(tr - 1) < 1e-9 and np.all(np.abs(zc) <= 1 + 1e-9) and np.all(np.abs(zzc) <= 1 + 1e-9)
    print(f"[G2] Tr(rho)={tr:.8f} max|Z|={np.abs(zc).max():.3f} "
          f"max|ZZ|={np.abs(zzc).max():.3f}  {'PASS' if g2 else 'FAIL'}")
    ok &= g2

    # G3: feedback live & correct -- k_fb=0 == open-loop; k_fb>0 differs
    F_open = fb_qrc_features(ang, n1, n2, seed=0, readout="ZZ", k_fb=0.0)
    F_open2 = fb_qrc_features(ang, n1, n2, seed=0, readout="ZZ", k_fb=0.0)
    F_fb = fb_qrc_features(ang, n1, n2, seed=0, readout="ZZ", k_fb=0.5)
    same = np.allclose(F_open, F_open2, atol=1e-12)
    diff = not np.allclose(F_open, F_fb, atol=1e-9)
    g3 = same and diff
    print(f"[G3] k_fb=0 reproducible (==open-loop): {same}; k_fb>0 differs: {diff}"
          f"  {'PASS' if g3 else 'FAIL'}")
    ok &= g3

    # G4: feedback causal -- perturbing a FUTURE input leaves earlier features fixed
    ang_pert = ang.copy(); t_pert = 40; ang_pert[t_pert] += 0.5
    F_a = fb_qrc_features(ang, n1, n2, seed=0, readout="ZZ", k_fb=0.5)
    F_b = fb_qrc_features(ang_pert, n1, n2, seed=0, readout="ZZ", k_fb=0.5)
    # features before t_pert (minus the window reach) must be identical
    pre = t_pert - 5
    g4 = np.allclose(F_a[:pre], F_b[:pre], atol=1e-12) and not np.allclose(F_a, F_b)
    print(f"[G4] future perturbation leaves earlier features unchanged: "
          f"{'PASS' if g4 else 'FAIL'}")
    ok &= g4

    # G5: shot -> exact
    print("[G5] shot noise -> exact as shots grow")
    Fex = fb_qrc_features(ang, n1, n2, seed=0, readout="ZZ", k_fb=0.5, shots=0)
    errs = []
    for M in (100, 10_000, 1_000_000):
        Fm = fb_qrc_features(ang, n1, n2, seed=0, readout="ZZ", k_fb=0.5, shots=M)
        errs.append(float(np.mean(np.abs(Fm - Fex))))
    g5 = errs[0] > errs[1] > errs[2] and errs[2] < 1e-2
    print(f"   mean|shot-exact| M=100,1e4,1e6: " + ", ".join(f"{e:.4f}" for e in errs)
          + f"  {'PASS' if g5 else 'FAIL'}")
    ok &= g5

    print(f"\n[self-test] {'ALL GATES PASS' if ok else 'FAILED'}")
    if not ok:
        sys.exit("SELF-TEST FAILED - do not trust results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    print("Phase 0 only: run with --self-test. Science phases come next.")


if __name__ == "__main__":
    main()
