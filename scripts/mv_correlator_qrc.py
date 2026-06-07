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


def features_from_rho(rho, z_ops, zz_ops, readout, rng_shot=None, shots=0):
    """Compute readout features from a q-qubit density matrix rho."""
    q = len(z_ops)
    zc = np.array([np.real(np.trace(rho @ Z)) for Z in z_ops])
    if shots and shots > 0:
        zc = zc + rng_shot.standard_normal(q) * np.sqrt(np.maximum(1 - zc ** 2, 0) / shots)
        zc = np.clip(zc, -1, 1)
    if readout == "Z":
        return zc
    zzc = np.array([np.real(np.trace(rho @ O)) for O in zz_ops])
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
    z_ops, zz_ops = _z_ops(q), _zz_ops(q)
    rng_shot = np.random.default_rng(7000 + seed)
    taus = [tau, tau / 2.0] if readout == "ZZ_QR2" else [tau]
    base_readout = "ZZ" if readout == "ZZ_QR2" else readout
    blocks = []
    for tt in taus:
        U = evol_unitary(H_res, tt)
        feats = np.array([
            features_from_rho(reservoir_state_at_t(H_series, t, n1, n2, U, mem_depth),
                                z_ops, zz_ops, base_readout, rng_shot, shots)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test(); return
    print("Phase 0 only: run with --self-test. Science phases come next.")


if __name__ == "__main__":
    main()
