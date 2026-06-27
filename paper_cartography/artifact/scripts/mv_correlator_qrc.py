#!/usr/bin/env python3
"""
mv_correlator_qrc.py - 2-point-correlator QRC for multivariate volatility with
linear + BILINEAR cross-asset spillover.  See notes/mv_correlator_plan.md.

Phase 0 here = data generator (linear gamma + bilinear beta), Bayat-faithful
recurrent TFIM reservoir, three readout tiers (Z / ZZ / ZZ_QR2), and the
G1-G6 correctness gates in --self-test.  No science is run until the gates pass.

Readout tiers:
  Z      : <Z_i>                         (q features)
  ZZ     : <Z_i> + <Z_i Z_k>             (q + q(q-1)/2 features)
  ZZ_QR2 : ZZ computed at tau AND tau/2, concatenated   (Bayat QR2 ensemble)

Gates:
  G1 data finite & nonlinear (linear-R2 of target drops with beta; full R2 high)
  G2 reservoir physical (Tr rho=1; <Z>,<ZZ> in [-1,1])
  G3 2-point correctness (<Z_iZ_k> from rho == independent statevector calc, 1e-10)
  G4 shot->exact (mean|feat_shot-feat_exact| monotone ->0 as M grows)
  G5 QR2 distinctness (dim 2x QR1; tau/2 block != tau block)
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
# Data: linear (gamma) + bilinear (beta) nonlinear spillover, bounded
# ---------------------------------------------------------------------------

def generate_spillover(n_assets, n_steps, seed, gamma, beta,
                        phi=0.6, noise=0.5, n_bilinear=None):
    """Zero-mean log-vol, mean-reverting, with bounded linear + bilinear
    cross-asset spillover.

      h_{i,t} = phi h_{i,t-1}
                + gamma * sum_j S_ij g(h_{j,t-1})              # linear (VAR-decodable)
                + beta  * sum_{(j,k) in pairs_i} g(h_j) g(h_k) # bilinear (needs quadratic)
                + noise * eps
      g = tanh (bounded -> cannot diverge).

    S: row-stochastic linear routing (zero diag).
    pairs_i: a fixed random set of off-diagonal (j,k) products feeding asset i.
    """
    rng = np.random.default_rng(seed)
    S = rng.uniform(0, 1, (n_assets, n_assets)); np.fill_diagonal(S, 0.0)
    S /= (S.sum(1, keepdims=True) + 1e-12)
    if n_bilinear is None:
        n_bilinear = n_assets                      # ~N quadratic terms per asset
    # fixed bilinear index sets: for each asset i, n_bilinear (j,k) pairs, j!=k!=i
    pairs = []
    for i in range(n_assets):
        others = [a for a in range(n_assets) if a != i]
        ps = []
        for _ in range(n_bilinear):
            j, k = rng.choice(others, size=2, replace=False)
            ps.append((int(j), int(k)))
        pairs.append(ps)
    H = np.zeros((n_steps, n_assets))
    H[0] = noise * rng.standard_normal(n_assets)
    for t in range(1, n_steps):
        g = np.tanh(H[t - 1])
        lin = gamma * (S @ g)
        bil = np.array([beta * sum(g[j] * g[k] for (j, k) in pairs[i])
                          for i in range(n_assets)])
        H[t] = phi * H[t - 1] + lin + bil + noise * rng.standard_normal(n_assets)
    return H


# ---------------------------------------------------------------------------
# Bayat-faithful recurrent TFIM reservoir
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
    rng = np.random.default_rng(10_000 + seed)
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
    """Trace out first n1 (input) qubits, keep last n2 (memory)."""
    q = n1 + n2
    r = rho.reshape([2] * q + [2] * q)
    letters = "abcdefghijklmnopqrstuvwxyz"
    row = list(letters[:q]); col = list(letters[q:2 * q])
    for k in range(n1):
        col[k] = row[k]
    sub = "".join(row) + "".join(col) + "->" + "".join(row[n1:]) + "".join(col[n1:])
    return np.einsum(sub, r).reshape(2 ** n2, 2 ** n2)


def _z_ops(q):
    return [kron_op(Zm, i, q) for i in range(q)]


def _zz_ops(q):
    ops = []
    for i in range(q):
        for k in range(i + 1, q):
            zi = np.array([[1.0]], dtype=complex)
            for s in range(q):
                zi = np.kron(zi, Zm if (s == i or s == k) else I2)
            ops.append(zi)
    return ops


def reservoir_state_at_t(H_series, t, n1, n2, U, mem_depth):
    """Return the full post-evolution density matrix at time t (q qubits),
    using the recurrent partial-trace memory over the last mem_depth inputs."""
    q = n1 + n2
    d_mem = 2 ** n2
    rho_mem = np.zeros((d_mem, d_mem), dtype=complex); rho_mem[0, 0] = 1.0
    lo = max(0, t - mem_depth + 1)
    rho_full = None
    for tt in range(lo, t + 1):
        x = H_series[tt]
        psi_in = np.array([1.0], dtype=complex)
        for a in range(n1):
            c, s = np.cos(x[a] / 2.0), np.sin(x[a] / 2.0)
            psi_in = np.kron(psi_in, np.array([c, s], dtype=complex))
        rho_in = np.outer(psi_in, psi_in.conj())
        rho_full = U @ np.kron(rho_in, rho_mem) @ U.conj().T
        rho_mem = partial_trace_keep_memory(rho_full, n1, n2)
    return rho_full          # last full state (q qubits), pre-trace


def build_sign_tables(q):
    """Precompute parity sign tables for diagonal Z and ZZ observables.
    Z_i and Z_iZ_k are diagonal in the computational basis, so
    <O> = diag(rho) . signs.  Returns (z_signs (q,2^q), zz_signs (P,2^q)).
    Verified equivalent to trace(rho@O) by gate G3 (1e-16)."""
    dim = 2 ** q
    idx = np.arange(dim)
    shifts = np.arange(q - 1, -1, -1)
    bits = ((idx[:, None] >> shifts) & 1)              # (2^q, q)
    z_signs = ((-1.0) ** bits).T                        # (q, 2^q): <Z_i>
    zz_rows = []
    for i in range(q):
        for k in range(i + 1, q):
            zz_rows.append((-1.0) ** (bits[:, i] + bits[:, k]))
    zz_signs = np.array(zz_rows)                         # (q(q-1)/2, 2^q)
    return z_signs, zz_signs


def features_from_rho(rho, readout, z_signs, zz_signs, rng_shot=None, shots=0):
    """Readout features from a q-qubit density matrix via DIAGONAL observables
    (no per-operator matmul): <Z_i>,<Z_iZ_k> = diag(rho) . signs."""
    diag = np.real(np.diag(rho))                         # (2^q,)
    zc = z_signs @ diag                                  # (q,)
    if shots and shots > 0:
        zc = zc + rng_shot.standard_normal(len(zc)) * np.sqrt(np.maximum(1 - zc ** 2, 0) / shots)
        zc = np.clip(zc, -1, 1)
    if readout == "Z":
        return zc
    zzc = zz_signs @ diag                                # (q(q-1)/2,)
    if shots and shots > 0:
        zzc = zzc + rng_shot.standard_normal(len(zzc)) * np.sqrt(np.maximum(1 - zzc ** 2, 0) / shots)
        zzc = np.clip(zzc, -1, 1)
    return np.concatenate([zc, zzc])


def qrc_feature_matrix(H_series, n1, n2, seed, readout, mem_depth=3,
                         tau=1.0, shots=0):
    """Full QRC feature matrix (T, D) for the chosen readout tier.
    ZZ_QR2 concatenates features computed at tau and tau/2."""
    q = n1 + n2
    H_res = reservoir_hamiltonian(q, seed)
    z_signs, zz_signs = build_sign_tables(q)
    rng_shot = np.random.default_rng(7000 + seed)
    taus = [tau, tau / 2.0] if readout == "ZZ_QR2" else [tau]
    base_readout = "ZZ" if readout == "ZZ_QR2" else readout
    blocks = []
    for tt in taus:
        U = evol_unitary(H_res, tt)
        feats = np.array([
            features_from_rho(reservoir_state_at_t(H_series, t, n1, n2, U, mem_depth),
                                base_readout, z_signs, zz_signs, rng_shot, shots)
            for t in range(len(H_series))])
        blocks.append(feats)
    return np.concatenate(blocks, axis=1)


# ---------------------------------------------------------------------------
# G1-G6 self-test
# ---------------------------------------------------------------------------

def _r2(F, y, train_frac=0.7, alpha=1.0):
    """In-sample-fit R2 on a held-out split (no horizon, t->t map) for a
    quick linear-decodability probe."""
    T = len(F); ntr = int(round(train_frac * T))
    A = F[:ntr].T @ F[:ntr] + alpha * np.eye(F.shape[1])
    W = np.linalg.solve(A, F[:ntr].T @ y[:ntr])
    yp = F[ntr:] @ W
    ss_res = np.sum((y[ntr:] - yp) ** 2)
    ss_tot = np.sum((y[ntr:] - y[ntr:].mean(0)) ** 2)
    return 1 - ss_res / (ss_tot + 1e-12)


def self_test():
    ok = True

    # ---- G1: data finite, bounded & STRONGLY quadratically-decodable ----
    # Coupled Henon: linear predictors fail (low R2), quadratic succeed (high
    # R2), with a large clean gain -- the decisive structure for QRC-ZZ vs Poly2.
    print("[G1] coupled-Henon data: bounded + strong quad-over-linear gain")
    N = 5
    g1 = True
    for c in (0.0, 0.1, 0.2):
        X = generate_coupled_henon(N, 4000, seed=0, coupling=c)
        fin = np.all(np.isfinite(X)) and X.std() < 10
        Xin, y = X[:-1], X[1:]
        lr = _r2(Xin, y); qr = _r2(_quad_features(Xin), y)
        gain = qr - lr
        good = fin and (gain > 0.15)
        g1 &= good
        print(f"   coupling={c}: Xstd={X.std():.2f} linR2={lr:.3f} "
              f"quadR2={qr:.3f} gain={gain:.3f}  {'ok' if good else 'WEAK'}")
    print(f"   {'PASS' if g1 else 'FAIL'}")
    ok &= g1

    # ---- G2 / G3 / G5: reservoir physical + 2-point correctness + QR2 ----
    n1, n2 = 4, 3; q = n1 + n2
    H = generate_spillover(n1, 50, seed=1, gamma=0.5, beta=1.0)
    ang = np.tanh(H)  # bounded angles
    H_res = reservoir_hamiltonian(q, 0)
    U = evol_unitary(H_res, 1.0)
    z_ops, zz_ops = _z_ops(q), _zz_ops(q)
    rho = reservoir_state_at_t(ang, 10, n1, n2, U, mem_depth=3)
    tr = float(np.real(np.trace(rho)))
    zc = np.array([np.real(np.trace(rho @ Z)) for Z in z_ops])
    zzc = np.array([np.real(np.trace(rho @ O)) for O in zz_ops])
    g2 = abs(tr - 1) < 1e-9 and np.all(np.abs(zc) <= 1 + 1e-9) and np.all(np.abs(zzc) <= 1 + 1e-9)
    print(f"[G2] Tr(rho)={tr:.8f}, max|Z|={np.abs(zc).max():.4f}, "
          f"max|ZZ|={np.abs(zzc).max():.4f}  {'PASS' if g2 else 'FAIL'}")
    ok &= g2

    # G3: <Z_i Z_k> from rho vs independent eigen-decomposition of the SAME state.
    # rho here is mixed (post partial trace memory), so verify against the
    # operator definition computed a different way: <ZZ> = sum_s p_s parity,
    # using the diagonal of rho in the computational basis (Z-basis), since
    # Z_iZ_k is diagonal.  parity_s = (-1)^{bit_i + bit_k}.
    dim = 2 ** q
    diag = np.real(np.diag(rho))
    idx = np.arange(dim)
    shifts = np.arange(q - 1, -1, -1)
    bits = ((idx[:, None] >> shifts) & 1)
    max_err = 0.0
    p = 0
    for i in range(q):
        for k in range(i + 1, q):
            parity = (-1.0) ** (bits[:, i] + bits[:, k])
            zz_indep = float(diag @ parity)
            max_err = max(max_err, abs(zz_indep - zzc[p])); p += 1
    g3 = max_err < 1e-10
    print(f"[G3] max|<ZZ>_rho - <ZZ>_indep| = {max_err:.2e}  {'PASS' if g3 else 'FAIL'}")
    ok &= g3

    # G3b: optimized diagonal readout == explicit trace(rho@O) path (the speedup)
    z_signs, zz_signs = build_sign_tables(q)
    feat_fast = features_from_rho(rho, "ZZ", z_signs, zz_signs)
    feat_ref = np.concatenate([zc, zzc])      # from explicit trace above
    err_fast = float(np.max(np.abs(feat_fast - feat_ref)))
    g3b = err_fast < 1e-12
    print(f"[G3b] max|diag_readout - trace_readout| = {err_fast:.2e}  "
          f"{'PASS' if g3b else 'FAIL'}")
    ok &= g3b

    # G5: QR2 distinctness
    Fq1 = qrc_feature_matrix(ang, n1, n2, seed=0, readout="ZZ", mem_depth=3)
    Fq2 = qrc_feature_matrix(ang, n1, n2, seed=0, readout="ZZ_QR2", mem_depth=3)
    half = Fq2.shape[1] // 2
    g5 = (Fq2.shape[1] == 2 * Fq1.shape[1]
          and np.allclose(Fq2[:, :half], Fq1)
          and not np.allclose(Fq2[:, half:], Fq1))
    print(f"[G5] QR1 dim={Fq1.shape[1]}, QR2 dim={Fq2.shape[1]} (2x & tau/2 block "
          f"differs)  {'PASS' if g5 else 'FAIL'}")
    ok &= g5

    # ---- G4: shot -> exact ----
    print("[G4] shot noise -> exact as shots grow")
    Fex = qrc_feature_matrix(ang, n1, n2, seed=0, readout="ZZ", mem_depth=3, shots=0)
    errs = []
    for M in (100, 10_000, 1_000_000):
        Fm = qrc_feature_matrix(ang, n1, n2, seed=0, readout="ZZ", mem_depth=3, shots=M)
        errs.append(float(np.mean(np.abs(Fm - Fex))))
    g4 = errs[0] > errs[1] > errs[2] and errs[2] < 1e-2
    print(f"   mean|shot-exact| M=100,1e4,1e6: " + ", ".join(f"{e:.4f}" for e in errs)
          + f"  {'PASS' if g4 else 'FAIL'}")
    ok &= g4

    # ---- G1': C6a second quadratic testbed (coupled logistic) is bounded +
    #          quadratically decodable, like the Henon map in Case I ----
    print("[G1'] coupled-logistic data: bounded + quad-over-linear gain")
    g1c = True
    for c in (0.0, 0.1, 0.2):
        X = generate_coupled_logistic(5, 4000, seed=0, coupling=c)
        fin = np.all(np.isfinite(X)) and X.std() < 10
        Xin, y = X[:-1], X[1:]
        rl = _r2(Xin, y); rq = _r2(_quad_features(Xin), y)
        gain = rq - rl
        good = fin and (gain > 0.05)
        g1c &= good
        print(f"   coupling={c}: Xstd={X.std():.2f} lin_R2={rl:.3f} "
              f"quad_R2={rq:.3f} gain={gain:.3f}  {'ok' if good else 'WEAK'}")
    print(f"   {'PASS' if g1c else 'FAIL'}")
    ok &= g1c

    print(f"\n[self-test] {'ALL GATES PASS' if ok else 'FAILED'}")
    if not ok:
        sys.exit("SELF-TEST FAILED - do not trust results")


def generate_coupled_henon(n_assets, n_steps, seed, coupling,
                             a=1.4, b=0.3, obs_noise=0.0, burn=200):
    """Coupled Henon maps: bounded, deterministic, GENUINELY quadratic
    multivariate dynamics (the x^2 term).  Linear predictors do poorly;
    quadratic do well -- a clean, well-conditioned testbed.

      x_{i,t+1} = 1 - a*xc_i^2 + b*y_{i,t},   xc_i = (1-c)x_i + c*mean_neighbors
      y_{i,t+1} = x_{i,t}
    Returns standardized x-series (n_steps, n_assets).  `coupling` c in [0,1)
    sets cross-asset structure; c=0 -> independent Henon maps.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, n_assets)
    y = rng.uniform(-0.5, 0.5, n_assets)
    out = np.zeros((n_steps + burn, n_assets))
    for t in range(n_steps + burn):
        xbar = x.mean()
        xc = (1 - coupling) * x + coupling * xbar
        xn = 1 - a * xc ** 2 + b * y
        y = x
        x = np.clip(xn, -2.0, 2.0)
        out[t] = x
    X = out[burn:]
    if obs_noise > 0:
        X = X + obs_noise * rng.standard_normal(X.shape)
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def _quad_features(X):
    """[X, all products X_i X_j for i<=j]."""
    N = X.shape[1]
    cols = [X]
    for i in range(N):
        cols.append(X[:, [i]] * X[:, i:])
    return np.concatenate(cols, axis=1)


def generate_coupled_logistic(n_assets, n_steps, seed, coupling,
                                r=3.8, obs_noise=0.0, burn=300):
    """Coupled logistic maps: a SECOND bounded quadratic multivariate testbed,
    structurally different from the Henon map (Case I), used for the C6a
    robustness check.  The quadratic x(1-x) nonlinearity makes Poly2 the
    natural capacity-matched control here too.

      x_{i,t+1} = r * xc_i * (1 - xc_i),  xc_i = (1-c)*x_i + c*mean_neighbors
    c sets cross-asset coupling.  Returns standardized x-series."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.2, 0.8, n_assets)
    out = np.zeros((n_steps + burn, n_assets))
    for t in range(n_steps + burn):
        xbar = x.mean()
        xc = (1 - coupling) * x + coupling * xbar
        x = np.clip(r * xc * (1 - xc), 0.0, 1.0)
        out[t] = x
    X = out[burn:]
    if obs_noise > 0:
        X = X + obs_noise * rng.standard_normal(X.shape)
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def ridge_forecast(F, Y, horizon, train_frac=0.7, val_frac=0.15,
                     alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
                     X_for_dir=None):
    """Predict Y[t+h] from F[t].  Ridge alpha chosen on validation (no test
    leakage).  Returns dict with 'nrmse' (per-asset, mean over assets) and
    'mda' (mean directional accuracy: fraction of test points where the
    predicted CHANGE direction sign(Y[t+h]-X_ref[t]) matches observed).
    X_for_dir is the reference level at time t (defaults to Y[t], i.e. the
    last observed value -> direction of the next move).  MDA is reported
    alongside a persistence baseline (predict 'no change' / last sign)."""
    T = len(F); h = horizon
    ntr = int(round(train_frac * T))
    nval = int(round((train_frac + val_frac) * T))
    if T - h <= nval + 5:
        return {"nrmse": float("nan"), "mda": float("nan"),
                "mda_persist": float("nan")}

    def fit(Ftr, Ytr, a):
        A = Ftr.T @ Ftr + a * np.eye(Ftr.shape[1])
        return np.linalg.solve(A, Ftr.T @ Ytr)

    Ftr, Ytr = F[:ntr - h], Y[h:ntr]
    Fva, Yva = F[ntr:nval - h], Y[ntr + h:nval]
    best_a, best_v = alphas[0], np.inf
    for a in alphas:
        W = fit(Ftr, Ytr, a)
        v = np.mean((Fva @ W - Yva) ** 2)
        if v < best_v:
            best_v, best_a = v, a
    Ftv, Ytv = F[:nval - h], Y[h:nval]
    W = fit(Ftv, Ytv, best_a)
    Fte, Yte = F[nval:T - h], Y[nval + h:T]
    Yp = Fte @ W
    rmse = np.sqrt(np.mean((Yp - Yte) ** 2, axis=0))
    std = np.std(Yte, axis=0) + 1e-12
    nrmse = float(np.mean(rmse / std))

    # MDA: direction of the move from the reference level X_ref[t] to Y[t+h].
    Xref = (X_for_dir if X_for_dir is not None else Y)
    ref_te = Xref[nval:T - h]                      # level at time t (causal)
    obs_dir = np.sign(Yte - ref_te)
    pred_dir = np.sign(Yp - ref_te)
    mda = float(np.mean(pred_dir == obs_dir))
    # persistence baseline: predict the move continues in last observed dir
    last_move = np.sign(ref_te - Xref[nval - h:T - 2 * h]) if nval - h >= 0 else obs_dir * 0
    mda_persist = float(np.mean(last_move == obs_dir))
    return {"nrmse": nrmse, "mda": mda, "mda_persist": mda_persist}


def ridge_forecast_nrmse(F, Y, horizon, **kw):
    return ridge_forecast(F, Y, horizon, **kw)["nrmse"]


def phase1(N=5, n2=3, n_steps=1500, coupling=0.1, horizon=1,
            data_seed=0, res_seed=0, shots=0):
    """Readout ablation: Z vs ZZ vs ZZ_QR2 at a single seed.  Go/no-go:
    proceed only if ZZ beats Z by >=15% relative NRMSE (H1)."""
    q = N + n2
    assert q <= 11
    X = generate_coupled_henon(N, n_steps, seed=data_seed, coupling=coupling)
    ang = np.clip(X * (np.pi / 3), -np.pi, np.pi)   # scale to RY angle range
    Y = X                                            # forecast the series itself
    res = {}
    for readout in ("Z", "ZZ", "ZZ_QR2"):
        t0 = time.time()
        F = qrc_feature_matrix(ang, N, n2, seed=res_seed, readout=readout,
                                mem_depth=3, shots=shots)
        nrmse = ridge_forecast_nrmse(F, Y, horizon)
        res[readout] = {"nrmse": nrmse, "D": F.shape[1]}
        print(f"  {readout:>7}: D={F.shape[1]:>3}  NRMSE={nrmse:.4f}  "
              f"({time.time()-t0:.0f}s)")
    z, zz, qr2 = res["Z"]["nrmse"], res["ZZ"]["nrmse"], res["ZZ_QR2"]["nrmse"]
    rel_zz = (z - zz) / z
    rel_qr2 = (zz - qr2) / zz
    res["_summary"] = {"rel_drop_ZZ_vs_Z": rel_zz,
                        "rel_drop_QR2_vs_ZZ": rel_qr2,
                        "H1_pass": bool(rel_zz >= 0.15)}
    print(f"\n  ZZ vs Z:  {rel_zz*100:+.1f}% NRMSE  (H1 needs >=15% drop -> "
          f"{'PASS' if rel_zz >= 0.15 else 'FAIL'})")
    print(f"  QR2 vs ZZ:{rel_qr2*100:+.1f}% NRMSE  (H2, expect small +)")
    outdir = ROOT / "results" / "mv_correlator"; outdir.mkdir(parents=True, exist_ok=True)
    cfg = {"N": N, "n2": n2, "q": q, "n_steps": n_steps, "coupling": coupling,
           "horizon": horizon, "data_seed": data_seed, "res_seed": res_seed,
           "shots": shots}
    out = {"config": cfg, "results": res}
    fp = outdir / f"phase1_c{coupling}_h{horizon}.json"
    fp.write_text(json.dumps(out, indent=2))
    h = hashlib.sha256(fp.read_bytes()).hexdigest()[:12]
    print(f"\n  Saved {fp}  (sha {h})")
    return out


# ---------------------------------------------------------------------------
# Classical baselines (same forecast harness)
# ---------------------------------------------------------------------------

def poly2_features(X, window=3):
    """Windowed degree-2 polynomial features: per time t, stack the last
    `window` rows, then [linear, all pairwise products incl squares]."""
    T, N = X.shape
    rows = []
    for t in range(T):
        block = X[max(0, t - window + 1):t + 1]
        if len(block) < window:
            block = np.vstack([np.repeat(block[:1], window - len(block), 0), block])
        v = block.flatten()
        quad = np.concatenate([v[i] * v[i:] for i in range(len(v))])
        rows.append(np.concatenate([v, quad]))
    return np.array(rows)


def window_features(X, window=3):
    T, N = X.shape
    rows = []
    for t in range(T):
        block = X[max(0, t - window + 1):t + 1]
        if len(block) < window:
            block = np.vstack([np.repeat(block[:1], window - len(block), 0), block])
        rows.append(block.flatten())
    return np.array(rows)


def esn_features(X, n_res, seed, sr=0.9, in_scale=0.5, leak=0.3, density=0.1):
    rng = np.random.default_rng(seed)
    N = X.shape[1]
    Win = rng.uniform(-in_scale, in_scale, (n_res, N))
    W = rng.standard_normal((n_res, n_res)) * (rng.uniform(0, 1, (n_res, n_res)) < density)
    e = np.max(np.abs(np.linalg.eigvals(W)))
    if e > 1e-8:
        W *= sr / e
    r = np.zeros(n_res); out = np.zeros((len(X), n_res))
    for t in range(len(X)):
        r = (1 - leak) * r + leak * np.tanh(W @ r + Win @ X[t])
        out[t] = r
    return out


def _val_mse(F, Y, h, alpha=1e-3):
    """Validation-only MSE (train [0,0.7), val [0.7,0.85)); no test peek."""
    T = len(F)
    ntr = int(round(0.7 * T)); nval = int(round(0.85 * T))
    A = F[:ntr-h].T @ F[:ntr-h] + alpha * np.eye(F.shape[1])
    W = np.linalg.solve(A, F[:ntr-h].T @ Y[h:ntr])
    vp = F[ntr:nval-h] @ W
    return float(np.mean((vp - Y[ntr+h:nval]) ** 2))


def phase2(N=5, n2=3, n_steps=800, coupling=0.1, horizons=(1,),
            data_seed=0, res_seed=0, shots=0, obs_noise=0.0,
            angle_scales=(np.pi/4, np.pi/2),
            taus=(1.0, 2.0)):
    """QRC (tuned over readout/angle-scale/tau on validation) vs classical
    baselines.  QRC feature matrices are built ONCE per (readout,scale,tau)
    and cached, then reused across all horizons (features are horizon-
    independent; only the ridge target shifts).  obs_noise>0 makes the map
    non-trivial (no method hits NRMSE 0 via the exact polynomial generator)."""
    q = N + n2; assert q <= 11
    X = generate_coupled_henon(N, n_steps, seed=data_seed, coupling=coupling,
                                 obs_noise=obs_noise)
    Y = X

    # --- build & cache every QRC feature matrix once (horizon-independent) ---
    cache = {}   # (readout, scale, tau) -> F
    t0 = time.time()
    for readout in ("ZZ", "ZZ_QR2"):
        for sc in angle_scales:
            ang = np.clip(X * sc, -np.pi, np.pi)
            for tau in taus:
                cache[(readout, round(sc, 4), tau)] = qrc_feature_matrix(
                    ang, N, n2, seed=res_seed, readout=readout,
                    mem_depth=3, tau=tau, shots=shots)
    print(f"  built {len(cache)} QRC feature matrices in {time.time()-t0:.0f}s")

    # classical features (horizon-independent) built once
    Fpoly = poly2_features(X); Fwin = window_features(X)
    Fe = esn_features(X, n_res=2 * next(iter(cache.values())).shape[1], seed=res_seed)

    out_per_h = {}
    for horizon in horizons:
        res = {}
        # QRC: pick (readout,scale,tau) by validation MSE at THIS horizon
        best = None
        for key, F in cache.items():
            v = _val_mse(F, Y, horizon)
            if best is None or v < best[0]:
                best = (v, key, F)
        _, (bro, bsc, btau), Fq = best
        res["QRC_tuned"] = {"nrmse": ridge_forecast_nrmse(Fq, Y, horizon),
                              "D": Fq.shape[1], "readout": bro,
                              "angle_scale": bsc, "tau": btau}
        res["Poly2"] = {"nrmse": ridge_forecast_nrmse(Fpoly, Y, horizon),
                          "D": Fpoly.shape[1]}
        res["VAR"] = {"nrmse": ridge_forecast_nrmse(Fwin, Y, horizon),
                       "D": Fwin.shape[1]}
        res["ESN"] = {"nrmse": ridge_forecast_nrmse(Fe, Y, horizon),
                       "D": Fe.shape[1]}
        res["Linear"] = {"nrmse": ridge_forecast_nrmse(X, Y, horizon),
                          "D": X.shape[1]}
        qn, pn = res["QRC_tuned"]["nrmse"], res["Poly2"]["nrmse"]
        print(f"  h={horizon}: " + "  ".join(
            f"{k}={res[k]['nrmse']:.4f}" for k in
            ("QRC_tuned", "Poly2", "ESN", "VAR", "Linear"))
            + f"  [QRC {bro} sc={bsc:.3f} tau={btau}; "
            + f"QRC vs Poly2 {(pn-qn)/pn*100:+.1f}%]")
        out_per_h[str(horizon)] = res

    outdir = ROOT / "results" / "mv_correlator"; outdir.mkdir(parents=True, exist_ok=True)
    out = {"config": {"N": N, "n2": n2, "q": q, "n_steps": n_steps,
                       "coupling": coupling, "horizons": list(horizons),
                       "data_seed": data_seed, "res_seed": res_seed,
                       "shots": shots, "obs_noise": obs_noise},
            "results_per_horizon": out_per_h}
    fp = outdir / (f"phase2_c{coupling}_n{obs_noise}"
                    f"_ds{data_seed}_rs{res_seed}.json")
    fp.write_text(json.dumps(out, indent=2))
    print(f"  Saved {fp}  (sha {hashlib.sha256(fp.read_bytes()).hexdigest()[:12]})")
    return out


def phase3(N=5, n2=3, n_steps=800, coupling=0.1, horizons=(1, 5),
            data_seed=0, res_seed=0, shots=0, obs_noise=0.1,
            angle_scales=(np.pi/4, np.pi/2), taus=(1.0, 2.0),
            dataset="henon"):
    """Concat ablation (Poly2 vs QRC vs Poly2+QRC) with MDA, per horizon.
    Decisive: if Poly2+QRC materially beats Poly2, the quantum features carry
    information the polynomial basis lacks; if concat == Poly2, QRC is
    redundant given a classical quadratic model.
    dataset='henon' (Case I) or 'logistic' (C6a second quadratic testbed)."""
    q = N + n2; assert q <= 11
    if dataset == "henon":
        X = generate_coupled_henon(N, n_steps, seed=data_seed, coupling=coupling,
                                     obs_noise=obs_noise)
    elif dataset == "logistic":
        X = generate_coupled_logistic(N, n_steps, seed=data_seed,
                                        coupling=coupling, obs_noise=obs_noise)
    else:
        raise ValueError(f"unknown dataset {dataset}")
    Y = X

    # cache QRC feature matrices once
    cache = {}
    t0 = time.time()
    for readout in ("ZZ", "ZZ_QR2"):
        for sc in angle_scales:
            ang = np.clip(X * sc, -np.pi, np.pi)
            for tau in taus:
                cache[(readout, round(sc, 4), tau)] = qrc_feature_matrix(
                    ang, N, n2, seed=res_seed, readout=readout,
                    mem_depth=3, tau=tau, shots=shots)
    Fpoly = poly2_features(X)
    print(f"  built {len(cache)} QRC caches + Poly2 in {time.time()-t0:.0f}s")

    out_per_h = {}
    for h in horizons:
        # pick best QRC config by validation MSE at this horizon
        best = None
        for key, F in cache.items():
            v = _val_mse(F, Y, h)
            if best is None or v < best[0]:
                best = (v, key, F)
        _, qkey, Fq = best
        m_poly = ridge_forecast(Fpoly, Y, h, X_for_dir=Y)
        m_qrc = ridge_forecast(Fq, Y, h, X_for_dir=Y)
        Fcat = np.concatenate([Fpoly, Fq], axis=1)
        m_cat = ridge_forecast(Fcat, Y, h, X_for_dir=Y)
        res = {"Poly2": m_poly, "QRC": m_qrc, "Poly2+QRC": m_cat,
               "qrc_cfg": {"readout": qkey[0], "scale": qkey[1], "tau": qkey[2]}}
        out_per_h[str(h)] = res
        dn = (m_poly["nrmse"] - m_cat["nrmse"]) / m_poly["nrmse"] * 100
        print(f"  h={h}: NRMSE Poly2={m_poly['nrmse']:.3f} QRC={m_qrc['nrmse']:.3f} "
              f"cat={m_cat['nrmse']:.3f} ({dn:+.1f}% vs Poly2) | "
              f"MDA Poly2={m_poly['mda']:.3f} QRC={m_qrc['mda']:.3f} "
              f"cat={m_cat['mda']:.3f} persist={m_poly['mda_persist']:.3f}")

    outdir = ROOT / "results" / "mv_correlator"; outdir.mkdir(parents=True, exist_ok=True)
    out = {"config": {"N": N, "n2": n2, "q": q, "n_steps": n_steps,
                       "coupling": coupling, "horizons": list(horizons),
                       "data_seed": data_seed, "res_seed": res_seed,
                       "shots": shots, "obs_noise": obs_noise,
                       "dataset": dataset},
            "results_per_horizon": out_per_h}
    tag = "" if dataset == "henon" else f"_{dataset}"
    fp = outdir / f"phase3{tag}_c{coupling}_n{obs_noise}_ds{data_seed}_rs{res_seed}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"  Saved {fp}  (sha {hashlib.sha256(fp.read_bytes()).hexdigest()[:12]})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--phase1", action="store_true")
    ap.add_argument("--phase2", action="store_true")
    ap.add_argument("--phase3", action="store_true")
    ap.add_argument("--coupling", type=float, default=0.1)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--horizons", type=int, nargs="+", default=None)
    ap.add_argument("--obs-noise", type=float, default=0.0)
    ap.add_argument("--dataset", choices=["henon", "logistic"], default="henon")
    args = ap.parse_args()
    if args.phase2:
        couplings = [args.coupling] if args.coupling >= 0 else [0.0, 0.1, 0.2]
        horizons = tuple(args.horizons) if args.horizons else (args.horizon,)
        for c in couplings:
            print(f"=== Phase 2 QRC(tuned) vs classical (coupling={c}, "
                  f"horizons={horizons}, obs_noise={args.obs_noise}) ===")
            phase2(coupling=c, horizons=horizons, obs_noise=args.obs_noise)
            print()
        return
    if args.phase3:
        couplings = [args.coupling] if args.coupling >= 0 else [0.0, 0.1, 0.2]
        horizons = tuple(args.horizons) if args.horizons else (1, 5)
        for c in couplings:
            print(f"=== Phase 3 concat ablation + MDA (dataset={args.dataset}, "
                  f"coupling={c}, horizons={horizons}, "
                  f"obs_noise={args.obs_noise}) ===")
            phase3(coupling=c, horizons=horizons, obs_noise=args.obs_noise,
                    dataset=args.dataset)
            print()
        return
    if args.self_test:
        self_test(); return
    if args.phase1:
        print(f"=== Phase 1 readout ablation (coupling={args.coupling}, "
              f"horizon={args.horizon}) ===")
        phase1(coupling=args.coupling, horizon=args.horizon); return
    print("Run with --self-test, --phase1, --phase2, or --phase3.")


if __name__ == "__main__":
    main()
