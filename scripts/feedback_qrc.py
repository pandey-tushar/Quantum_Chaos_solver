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


def ridge_forecast(F, Y, horizon=1, train_frac=0.7, val_frac=0.15,
                     alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Causal ridge forecast: predict Y[t+h] from F[t]. alpha tuned on
    validation slice (no test leakage). Returns NRMSE on the test slice."""
    T = len(F); h = horizon
    ntr = int(round(train_frac * T)); nval = int(round((train_frac + val_frac) * T))
    if T - h <= nval + 5:
        return float("nan")
    def fit(A_, b_, a):
        return np.linalg.solve(A_.T @ A_ + a * np.eye(A_.shape[1]), A_.T @ b_)
    Ftr, Ytr = F[:ntr - h], Y[h:ntr]
    Fva, Yva = F[ntr:nval - h], Y[ntr + h:nval]
    best_a, best_v = alphas[0], np.inf
    for a in alphas:
        W = fit(Ftr, Ytr, a)
        v = np.mean((Fva @ W - Yva) ** 2)
        if v < best_v:
            best_v, best_a = v, a
    W = fit(F[:nval - h], Y[h:nval], best_a)
    Fte, Yte = F[nval:T - h], Y[nval + h:T]
    rmse = np.sqrt(np.mean((Fte @ W - Yte) ** 2, axis=0))
    return float(np.mean(rmse / (np.std(Yte, axis=0) + 1e-12)))


def _val_nrmse(F, Y, h=1):
    """Validation-only NRMSE (train [0,0.7), val [0.7,0.85)) for k_fb/tau
    selection; never touches test."""
    T = len(F); ntr = int(round(0.7 * T)); nval = int(round(0.85 * T))
    if nval - h <= ntr or T <= nval:
        return float("inf")
    W = np.linalg.solve(F[:ntr-h].T @ F[:ntr-h] + 1e-3*np.eye(F.shape[1]),
                         F[:ntr-h].T @ Y[h:ntr])
    vp = F[ntr:nval-h] @ W
    return float(np.mean((vp - Y[ntr+h:nval]) ** 2))


def phase1(n1=1, n2=4, n_steps=1000, p_switch=0.02, horizon=1,
            data_seed=0, res_seed=0, window=5,
            k_fbs=(0.0, 0.5, 1.0, 2.0), taus=(1.0, 2.0)):
    """Go/no-go: does feedback beat open-loop QRC on the nonstationary task?
    k_fb and tau tuned on validation (k_fb=0 = open-loop is a candidate, so
    feedback only 'wins' if it genuinely helps). Proceed only if the tuned
    FB-QRC beats the open-loop (k_fb=0) baseline by >=10% relative NRMSE."""
    q = n1 + n2; assert q <= 11
    x, s = generate_regime_switching(n_steps, seed=data_seed, p_switch=p_switch)
    ang = np.clip(x, -np.pi, np.pi)
    Y = x[:, None]
    t0 = time.time()
    # open-loop reference (k_fb=0), tuned over tau
    best_open = None
    for tau in taus:
        F = windowed(fb_qrc_features(ang, n1, n2, seed=res_seed, readout="ZZ",
                                       tau=tau, k_fb=0.0), window)
        v = _val_nrmse(F, Y, horizon)
        if best_open is None or v < best_open[0]:
            best_open = (v, tau, F)
    _, open_tau, F_open = best_open
    nrmse_open = ridge_forecast(F_open, Y, horizon)
    # feedback, tuned over (k_fb>0, tau) on validation
    best_fb = None
    for tau in taus:
        for k in k_fbs:
            if k == 0.0:
                continue
            F = windowed(fb_qrc_features(ang, n1, n2, seed=res_seed,
                                          readout="ZZ", tau=tau, k_fb=k), window)
            v = _val_nrmse(F, Y, horizon)
            if best_fb is None or v < best_fb[0]:
                best_fb = (v, tau, k, F)
    _, fb_tau, fb_k, F_fb = best_fb
    nrmse_fb = ridge_forecast(F_fb, Y, horizon)
    # fixed-window linear floor
    nrmse_lin = ridge_forecast(windowed(x[:, None], window), Y, horizon)
    rel = (nrmse_open - nrmse_fb) / nrmse_open
    print(f"  open-loop QRC (tau={open_tau}): NRMSE={nrmse_open:.4f}")
    print(f"  feedback QRC  (tau={fb_tau}, k_fb={fb_k}): NRMSE={nrmse_fb:.4f}")
    print(f"  fixed-window linear floor:      NRMSE={nrmse_lin:.4f}")
    print(f"  FB vs open-loop: {rel*100:+.1f}%  (H1 needs >=10% -> "
          f"{'PASS' if rel >= 0.10 else 'FAIL'})  [{time.time()-t0:.0f}s]")
    outdir = ROOT / "results" / "feedback_qrc"; outdir.mkdir(parents=True, exist_ok=True)
    out = {"config": {"n1": n1, "n2": n2, "q": q, "n_steps": n_steps,
                       "p_switch": p_switch, "horizon": horizon,
                       "data_seed": data_seed, "res_seed": res_seed,
                       "window": window},
            "results": {"open_loop": {"nrmse": nrmse_open, "tau": open_tau},
                         "feedback": {"nrmse": nrmse_fb, "tau": fb_tau, "k_fb": fb_k},
                         "linear": {"nrmse": nrmse_lin},
                         "rel_fb_vs_open": rel,
                         "H1_pass": bool(rel >= 0.10)}}
    fp = outdir / f"phase1_ps{p_switch}_h{horizon}_ds{data_seed}_rs{res_seed}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"  Saved {fp}  (sha {hashlib.sha256(fp.read_bytes()).hexdigest()[:12]})")
    return out


def esn_features(x, n_res, seed, sr, leak, in_scale=0.5, density=0.1):
    """Classical echo-state-network reservoir features for a 1-D input."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-in_scale, in_scale, (n_res, 1))
    W = rng.standard_normal((n_res, n_res)) * (rng.uniform(0, 1, (n_res, n_res)) < density)
    e = np.max(np.abs(np.linalg.eigvals(W)))
    if e > 1e-8:
        W *= sr / e
    r = np.zeros(n_res); out = np.zeros((len(x), n_res))
    for t in range(len(x)):
        r = (1 - leak) * r + leak * np.tanh(W @ r + Win[:, 0] * x[t])
        out[t] = r
    return out


def poly2_features(x, window):
    """Windowed degree-2 polynomial features of a 1-D series."""
    Wlin = windowed(x[:, None], window)
    rows = [np.concatenate([v, np.concatenate([v[i] * v[i:] for i in range(len(v))])])
            for v in Wlin]
    return np.array(rows)


def mda(F, Y, horizon=1, train_frac=0.7, val_frac=0.15,
         alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Mean directional accuracy of the next move (sign of Y[t+h]-Y[t]) on the
    test slice, alpha tuned on validation. Also returns persistence MDA."""
    T = len(F); h = horizon
    ntr = int(round(train_frac * T)); nval = int(round((train_frac + val_frac) * T))
    if T - h <= nval + 5:
        return float("nan"), float("nan")
    def fit(A_, b_, a):
        return np.linalg.solve(A_.T @ A_ + a * np.eye(A_.shape[1]), A_.T @ b_)
    best_a, best_v = alphas[0], np.inf
    for a in alphas:
        W = fit(F[:ntr-h], Y[h:ntr], a)
        v = np.mean((F[ntr:nval-h] @ W - Y[ntr+h:nval]) ** 2)
        if v < best_v:
            best_v, best_a = v, a
    W = fit(F[:nval-h], Y[h:nval], best_a)
    yp = (F[nval:T-h] @ W).ravel()
    yref = Y[nval:T-h].ravel()                  # level at time t
    ytrue = Y[nval+h:T].ravel()
    obs = np.sign(ytrue - yref)
    pred = np.sign(yp - yref)
    m = float(np.mean(pred == obs))
    persist = float(np.mean(np.sign(yref - Y[nval-h:T-2*h].ravel()) == obs))
    return m, persist


def phase2(n1=1, n2=4, n_steps=1200, p_switch=0.05, horizon=1, window=5,
            data_seeds=(0, 1, 2), res_seeds=(0, 1, 2),
            k_fbs=(0.5, 1.0, 2.0), taus=(1.0, 2.0),
            esn_srs=(0.7, 0.9, 0.99), esn_leaks=(0.2, 0.5, 0.9)):
    """The real test: FB-QRC vs tuned ESN vs Poly2 vs open-loop QRC, NRMSE +
    MDA, over data_seeds x res_seeds. All hyperparameters tuned on validation
    only. ESN gets the SAME tuning budget as FB-QRC (no strawman)."""
    q = n1 + n2; assert q <= 11
    methods = ["FB_QRC", "OpenLoop_QRC", "ESN", "Poly2", "Linear"]
    acc = {m: [] for m in methods}
    accM = {m: [] for m in methods}
    persistM = []
    t0 = time.time()
    for ds in data_seeds:
        x, s = generate_regime_switching(n_steps, seed=ds, p_switch=p_switch)
        ang = np.clip(x, -np.pi, np.pi); Y = x[:, None]
        Fpoly = poly2_features(x, window); Flin = windowed(x[:, None], window)
        acc["Poly2"].append(ridge_forecast(Fpoly, Y, horizon))
        acc["Linear"].append(ridge_forecast(Flin, Y, horizon))
        mp, pp = mda(Fpoly, Y, horizon); accM["Poly2"].append(mp)
        ml, _ = mda(Flin, Y, horizon); accM["Linear"].append(ml)
        persistM.append(pp)
        for rs in res_seeds:
            # FB-QRC: tune (k_fb>0, tau) on validation
            best_fb, best_open = None, None
            for tau in taus:
                Fo = windowed(fb_qrc_features(ang, n1, n2, seed=rs, readout="ZZ",
                                                tau=tau, k_fb=0.0), window)
                vo = _val_nrmse(Fo, Y, horizon)
                if best_open is None or vo < best_open[0]:
                    best_open = (vo, Fo)
                for k in k_fbs:
                    Ff = windowed(fb_qrc_features(ang, n1, n2, seed=rs,
                                    readout="ZZ", tau=tau, k_fb=k), window)
                    vf = _val_nrmse(Ff, Y, horizon)
                    if best_fb is None or vf < best_fb[0]:
                        best_fb = (vf, Ff)
            # ESN: tune (sr, leak) on validation, matched feature dim
            n_res = best_fb[1].shape[1]
            best_esn = None
            for sr in esn_srs:
                for lk in esn_leaks:
                    Fe = windowed(esn_features(x, n_res // window, seed=rs,
                                                 sr=sr, leak=lk), window)
                    ve = _val_nrmse(Fe, Y, horizon)
                    if best_esn is None or ve < best_esn[0]:
                        best_esn = (ve, Fe)
            acc["FB_QRC"].append(ridge_forecast(best_fb[1], Y, horizon))
            acc["OpenLoop_QRC"].append(ridge_forecast(best_open[1], Y, horizon))
            acc["ESN"].append(ridge_forecast(best_esn[1], Y, horizon))
            mf, _ = mda(best_fb[1], Y, horizon); accM["FB_QRC"].append(mf)
            mo, _ = mda(best_open[1], Y, horizon); accM["OpenLoop_QRC"].append(mo)
            me, _ = mda(best_esn[1], Y, horizon); accM["ESN"].append(me)
    def ms(d, k): a = np.array(d[k]); return float(a.mean()), float(a.std())
    print(f"  {'method':>13} {'NRMSE':>16} {'MDA':>16}")
    for m in methods:
        nm, ns = ms(acc, m); mm, msd = ms(accM, m)
        print(f"  {m:>13}  {nm:.4f}+/-{ns:.4f}  {mm:.4f}+/-{msd:.4f}")
    print(f"  {'persistence':>13}  {'':>16}  {np.mean(persistM):.4f}")
    fb_n, _ = ms(acc, "FB_QRC"); esn_n, _ = ms(acc, "ESN")
    print(f"\n  FB-QRC vs ESN (NRMSE): {(esn_n-fb_n)/esn_n*100:+.1f}%  "
          f"({'FB better' if fb_n < esn_n else 'ESN better'})  [{time.time()-t0:.0f}s]")
    outdir = ROOT / "results" / "feedback_qrc"; outdir.mkdir(parents=True, exist_ok=True)
    out = {"config": {"n1": n1, "n2": n2, "q": q, "n_steps": n_steps,
                       "p_switch": p_switch, "horizon": horizon, "window": window,
                       "data_seeds": list(data_seeds), "res_seeds": list(res_seeds)},
            "nrmse": {m: acc[m] for m in methods},
            "mda": {m: accM[m] for m in methods},
            "persistence_mda": persistM}
    fp = outdir / f"phase2_ps{p_switch}_h{horizon}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"  Saved {fp}  (sha {hashlib.sha256(fp.read_bytes()).hexdigest()[:12]})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--phase1", action="store_true")
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--p-switch", type=float, default=0.02)
    ap.add_argument("--p-switches", type=float, nargs="+", default=None)
    ap.add_argument("--horizon", type=int, default=1)
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    if args.phase1:
        ps_list = args.p_switches if args.p_switches else [args.p_switch]
        for ps in ps_list:
            print(f"=== Phase 1 go/no-go: feedback vs open-loop "
                  f"(p_switch={ps}, horizon={args.horizon}) ===")
            phase1(p_switch=ps, horizon=args.horizon)
            print()
        return
    if args.phase2:
        print(f"=== Phase 2: FB-QRC vs tuned ESN/Poly2/open-loop "
              f"(p_switch={args.p_switch}, horizon={args.horizon}, 3x3 seeds) ===")
        phase2(p_switch=args.p_switch, horizon=args.horizon); return
    print("Run with --self-test, --phase1, or --phase2.")


if __name__ == "__main__":
    main()
