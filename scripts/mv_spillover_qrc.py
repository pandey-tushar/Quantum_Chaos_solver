#!/usr/bin/env python3
"""
mv_spillover_qrc.py - Multivariate cross-asset volatility-spillover forecasting:
Bayat-style quantum reservoir vs strong classical baselines.

THESIS (the theoretical-advantage axis): a fully-connected quantum reservoir's
all-to-all entangling dynamics natively encode JOINT, NONLINEAR cross-asset
volatility spillover.  Bayat et al. (arXiv:2505.13933, PRR 2026) tested only a
SINGLE asset.  We extend to N assets with a tunable NONLINEAR spillover knob
gamma and ask: does the QRC's joint encoding beat strong classical baselines,
and does any edge GROW with spillover strength?

Why nonlinear spillover: linear spillover is optimally captured by a VAR
baseline, so QRC could not win there.  We use THRESHOLD spillover (asset i's
log-vol gets a kick from asset j only when asset j's vol exceeds a threshold) --
the regime/crisis nonlinearity that is well documented in markets and that
linear HAR/VAR miss.  The real competitors are nonlinear: ESN and (optionally)
LSTM.

DATA (synthetic, controllable):
  Latent log-vol VAR(1) with nonlinear threshold spillover:
    h_{i,t} = omega_i + phi * h_{i,t-1}
              + gamma * sum_{j!=i} S_ij * relu(h_{j,t-1} - thr) + eps
  RV target = h (log realized vol).  gamma = spillover strength (advantage axis).

QRC (Bayat-faithful):
  H = sum_{i<j} J_ij X_i X_j + v sum_i Z_i,  J_ij ~ U[0,1], v=1, fixed per seed.
  RY angle encoding of the N current log-vols on n1=N input qubits.
  Recurrent memory depth k: encode the k most recent input vectors sequentially;
  between steps, partial-trace the input qubits, keep n2 memory qubits (density
  matrix evolution).  Measure <Z_j> on all q=n1+n2 qubits (optionally shot-noisy:
  var (1-<Z>^2)/M).  Linear ridge readout to all-N next-step (and multi-horizon)
  log-vols.

BASELINES (all predict all N assets, matched train/test):
  - HAR_per_asset : classical HAR (lags 1, avg-3, avg-12) PER asset, no spillover.
  - VAR_joint     : linear VAR on the windowed joint vector (captures LINEAR
                    spillover; should beat QRC if spillover were linear).
  - ESN_joint     : echo-state network on the joint vector (nonlinear competitor).
  - Ridge_window  : ridge on the windowed joint vector (linear control).

METRIC: per-asset NRMSE on log-vol (normalized by per-asset target std), mean
over assets, at horizons {1, 5, 20}.  Swept over gamma.

CORRECTNESS GATE (--self-test): (a) data generator: cross-asset vol correlation
increases with gamma; (b) reservoir: Tr(rho)=1 and <Z> in [-1,1] after evolution;
(c) shot noise -> exact as M->inf.  Run FIRST; if it fails nothing is trusted.
"""
from __future__ import annotations
import argparse, sys, json, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).parent.parent

# --- single-qubit ops ---
I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Ym = np.array([[0, -1j], [1j, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)


# ---------------------------------------------------------------------------
# Multi-asset nonlinear-spillover data
# ---------------------------------------------------------------------------

def generate_spillover_logvol(n_assets, n_steps, seed, gamma,
                                phi=0.75, thr=1.5, noise=0.5):
    """Latent (zero-mean) log-vol with mean-reverting AR(1) persistence +
    BOUNDED nonlinear threshold spillover.  Returns H (n_steps, n_assets).

    h_{i,t} = phi * h_{i,t-1}
              + gamma * sum_{j!=i} S_ij * tanh(relu(h_{j,t-1} - thr))
              + noise * eps
    The spillover term is bounded (tanh) so the recursion cannot diverge; the
    threshold makes it NONLINEAR (only high-vol neighbors spill), which is the
    regime structure linear VAR/HAR miss.  h is zero-mean (omega absorbed),
    representing standardized log-vol; nonnegativity of RV is not needed for the
    forecasting task and would only add a constant the readout learns.
    """
    rng = np.random.default_rng(seed)
    S = rng.uniform(0.0, 1.0, (n_assets, n_assets))
    np.fill_diagonal(S, 0.0)
    S /= (S.sum(axis=1, keepdims=True) + 1e-12)     # row-stochastic
    H = np.zeros((n_steps, n_assets))
    H[0] = noise * rng.standard_normal(n_assets)
    for t in range(1, n_steps):
        kick = np.tanh(np.maximum(H[t - 1] - thr, 0.0))   # bounded in [0,1)
        spill = gamma * (S @ kick)
        H[t] = phi * H[t - 1] + spill + noise * rng.standard_normal(n_assets)
    return H


# ---------------------------------------------------------------------------
# Bayat-style recurrent quantum reservoir
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


def bayat_reservoir_hamiltonian(q, seed, v=1.0):
    """H = sum_{i<j} J_ij X_i X_j + v sum_i Z_i, J_ij ~ U[0,1]."""
    rng = np.random.default_rng(10_000 + seed)
    H = np.zeros((2 ** q, 2 ** q), dtype=complex)
    for i in range(q):
        for j in range(i + 1, q):
            H += rng.uniform(0.0, 1.0) * two_site_xx(i, j, q)
    for i in range(q):
        H += v * kron_op(Zm, i, q)
    return H


def _z_ops(q):
    return [kron_op(Zm, i, q) for i in range(q)]


def partial_trace_keep_memory(rho, n1, n2):
    """Trace out the first n1 (input) qubits, keep last n2 (memory).
    rho is (2^q, 2^q), q=n1+n2.  Returns (2^n2, 2^n2)."""
    q = n1 + n2
    r = rho.reshape([2] * q + [2] * q)
    # axes: 0..q-1 = bra-ish (row) qubits, q..2q-1 = ket (col) qubits
    # trace input qubits = first n1 row indices with first n1 col indices
    for k in range(n1):
        # after each contraction indices shift; contract row-axis 0 with its
        # col partner. Recompute via einsum on remaining tensor each time is
        # messy; do it in one einsum below instead.
        break
    # one-shot einsum: label row qubits r0.., col qubits c0..; sum r_k=c_k for
    # k in input block; keep memory block open.
    letters = "abcdefghijklmnopqrstuvwxyz"
    row = list(letters[:q])
    col = list(letters[q:2 * q])
    for k in range(n1):                # tie input row==col (trace)
        col[k] = row[k]
    out_row = row[n1:]                 # memory rows
    out_col = col[n1:]                 # memory cols
    sub = "".join(row) + "".join(col) + "->" + "".join(out_row) + "".join(out_col)
    red = np.einsum(sub, r)
    d2 = 2 ** n2
    return red.reshape(d2, d2)


def qrc_features(H_series, n1, n2, U, z_ops, mem_depth, rng_shot, shots=0):
    """Run the recurrent reservoir over the log-vol series.
    H_series: (T, n1) inputs (already scaled to angles).
    Returns feature matrix (T, q) of <Z_j>, j over all q qubits.
    Recurrence: at step t, start from memory rho_mem (n2 qubits) tensor a fresh
    input register prepared by RY-encoding x_t; evolve U; measure; partial-trace
    to update memory.  mem_depth re-injects the last mem_depth inputs each step
    (Bayat 'k layers')."""
    q = n1 + n2
    d_mem = 2 ** n2
    # memory starts in |0><0|
    rho_mem = np.zeros((d_mem, d_mem), dtype=complex); rho_mem[0, 0] = 1.0
    T = len(H_series)
    feats = np.zeros((T, q))
    for t in range(T):
        # build input pure state by RY encoding x_t on n1 qubits
        lo = max(0, t - mem_depth + 1)
        rho = rho_mem
        for tt in range(lo, t + 1):
            x = H_series[tt]
            # input register density matrix = |psi_in><psi_in|
            psi_in = np.array([1.0], dtype=complex)
            for a in range(n1):
                c, s = np.cos(x[a] / 2.0), np.sin(x[a] / 2.0)
                psi_in = np.kron(psi_in, np.array([c, s], dtype=complex))
            rho_in = np.outer(psi_in, psi_in.conj())
            rho_full = np.kron(rho_in, rho)            # input (x) memory
            rho_full = U @ rho_full @ U.conj().T
            rho = partial_trace_keep_memory(rho_full, n1, n2)
        rho_mem = rho
        # to measure all q qubits we need the full post-evolution state of the
        # LAST layer; recompute it (cheap) with input+memory before the trace
        x = H_series[t]
        psi_in = np.array([1.0], dtype=complex)
        for a in range(n1):
            c, s = np.cos(x[a] / 2.0), np.sin(x[a] / 2.0)
            psi_in = np.kron(psi_in, np.array([c, s], dtype=complex))
        rho_in = np.outer(psi_in, psi_in.conj())
        # memory used for this measurement = memory BEFORE this step's trace,
        # i.e. rho at second-to-last; for simplicity use current rho_mem
        rho_full = np.kron(rho_in, rho_mem)
        rho_full = U @ rho_full @ U.conj().T
        zexp = np.array([np.real(np.trace(rho_full @ Z)) for Z in z_ops])
        if shots and shots > 0:
            var = np.maximum(1.0 - zexp ** 2, 0.0) / shots
            zexp = zexp + rng_shot.standard_normal(q) * np.sqrt(var)
            zexp = np.clip(zexp, -1.0, 1.0)
        feats[t] = zexp
    return feats


# ---------------------------------------------------------------------------
# Classical baselines
# ---------------------------------------------------------------------------

def har_features(H, window_short=3, window_long=12):
    """Per-asset HAR design: [h_{t}, mean(h_{t-2:t}), mean(h_{t-11:t})] per asset,
    concatenated.  Returns (T, 3N).  No cross-asset terms beyond concat."""
    T, N = H.shape
    F = np.zeros((T, 3 * N))
    for t in range(T):
        s = H[max(0, t - window_short + 1):t + 1].mean(axis=0)
        l = H[max(0, t - window_long + 1):t + 1].mean(axis=0)
        F[t] = np.concatenate([H[t], s, l])
    return F


def window_feats(H, w):
    T, N = H.shape
    F = np.zeros((T, w * N))
    for t in range(T):
        block = H[max(0, t - w + 1):t + 1]
        if len(block) < w:
            block = np.vstack([np.repeat(block[:1], w - len(block), axis=0), block])
        F[t] = block.flatten()
    return F


def esn_run(inputs, n_res, seed, sr=0.9, in_scale=0.5, leak=0.3, density=0.1):
    rng = np.random.default_rng(seed)
    N = inputs.shape[1]
    Win = rng.uniform(-in_scale, in_scale, (n_res, N))
    W = rng.standard_normal((n_res, n_res)) * (rng.uniform(0, 1, (n_res, n_res)) < density)
    eig = np.max(np.abs(np.linalg.eigvals(W)))
    if eig > 1e-8:
        W *= sr / eig
    r = np.zeros(n_res); out = np.zeros((len(inputs), n_res))
    for t in range(len(inputs)):
        r = (1 - leak) * r + leak * np.tanh(W @ r + Win @ inputs[t])
        out[t] = r
    return out


def ridge_fit_predict(F, Y, h, train_frac=0.7, alpha=1.0):
    """Predict Y[t+h] from F[t].  Returns per-asset NRMSE (mean over assets)."""
    T = len(F); ntr = int(round(train_frac * T))
    if T - h <= ntr + 5:
        return float("nan")
    Ftr, Ytr = F[:ntr - h], Y[h:ntr]
    Fte, Yte = F[ntr:T - h], Y[ntr + h:T]
    A = Ftr.T @ Ftr + alpha * np.eye(Ftr.shape[1])
    W = np.linalg.solve(A, Ftr.T @ Ytr)
    Yp = Fte @ W
    rmse = np.sqrt(np.mean((Yp - Yte) ** 2, axis=0))
    std = np.std(Yte, axis=0) + 1e-12
    return float(np.mean(rmse / std))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    print("[self-test] data: cross-asset vol correlation should rise with gamma")
    cors = []
    for g in (0.0, 0.5, 1.0, 2.0):
        H = generate_spillover_logvol(5, 3000, seed=0, gamma=g)
        assert np.all(np.isfinite(H)), f"data diverged at gamma={g}"
        C = np.corrcoef(H.T)
        off = C[~np.eye(5, dtype=bool)].mean()
        cors.append(off)
        print(f"  gamma={g}: mean off-diag corr = {off:.3f}")
    ok_data = all(cors[i] < cors[i + 1] for i in range(len(cors) - 1))
    print(f"  monotone increasing: {'PASS' if ok_data else 'FAIL'}")

    print("[self-test] reservoir: Tr(rho)=1 and <Z> in [-1,1]")
    n1, n2 = 4, 3; q = n1 + n2
    H_res = bayat_reservoir_hamiltonian(q, seed=0)
    w, V = np.linalg.eigh(H_res)
    U = (V * np.exp(-1j * w * 1.0)) @ V.conj().T
    z_ops = _z_ops(q)
    Hs = np.tanh(generate_spillover_logvol(n1, 60, seed=1, gamma=0.5))  # angles
    rng_s = np.random.default_rng(0)
    F = qrc_features(Hs, n1, n2, U, z_ops, mem_depth=3, rng_shot=rng_s, shots=0)
    ok_range = bool(np.all(np.abs(F) <= 1.0 + 1e-9))
    # trace check on one explicit evolution
    d_mem = 2 ** n2
    rho_mem = np.zeros((d_mem, d_mem), dtype=complex); rho_mem[0, 0] = 1.0
    psi_in = np.zeros(2 ** n1, dtype=complex); psi_in[0] = 1.0
    rho_full = np.kron(np.outer(psi_in, psi_in.conj()), rho_mem)
    rho_full = U @ rho_full @ U.conj().T
    tr = float(np.real(np.trace(rho_full)))
    red = partial_trace_keep_memory(rho_full, n1, n2)
    tr_red = float(np.real(np.trace(red)))
    print(f"  Tr(rho_full)={tr:.6f}, Tr(partial_trace)={tr_red:.6f}, "
          f"|Z|<=1: {ok_range}")
    ok_trace = abs(tr - 1) < 1e-9 and abs(tr_red - 1) < 1e-9

    print("[self-test] shot noise -> exact as shots grow")
    F_exact = qrc_features(Hs, n1, n2, U, z_ops, 3, np.random.default_rng(0), shots=0)
    errs = []
    for M in (100, 10_000, 1_000_000):
        Fm = qrc_features(Hs, n1, n2, U, z_ops, 3, np.random.default_rng(1), shots=M)
        errs.append(float(np.mean(np.abs(Fm - F_exact))))
    print(f"  mean|F_shot - F_exact| at M=100,1e4,1e6: "
          + ", ".join(f"{e:.4f}" for e in errs))
    ok_shot = errs[0] > errs[1] > errs[2] and errs[2] < 1e-2

    allok = ok_data and ok_range and ok_trace and ok_shot
    print(f"[self-test] {'PASS' if allok else 'FAIL'}")
    if not allok:
        sys.exit("SELF-TEST FAILED - do not trust results")


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--n-assets", type=int, default=5)
    ap.add_argument("--n2-memory", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=800)
    ap.add_argument("--mem-depth", type=int, default=3)
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 1.5])
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--res-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--data-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--shots", type=int, default=0,
                     help="0 = exact statevector (Bayat); >0 = shot-noisy")
    ap.add_argument("--esn-size", type=int, default=200)
    ap.add_argument("--out-dir", type=str,
                     default=str(ROOT / "results" / "mv_spillover"))
    args = ap.parse_args()

    if args.self_test:
        self_test(); return

    N, n2 = args.n_assets, args.n2_memory
    n1 = N
    q = n1 + n2
    assert q <= 11, f"q={q} exceeds standing rule q<=11"
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== MV spillover QRC: N={N} assets, q={q} ({n1}+{n2}), "
          f"mem_depth={args.mem_depth}, shots={args.shots or 'exact'} ===")
    print(f"    gammas={args.gammas}, horizons={args.horizons}, "
          f"res_seeds={args.res_seeds}, data_seeds={args.data_seeds}")

    methods = ["QRC", "HAR_per_asset", "VAR_joint", "ESN_joint", "Ridge_window"]
    # results[gamma][method][horizon] = list over (data_seed,res_seed)
    results = {g: {m: {h: [] for h in args.horizons} for m in methods}
               for g in args.gammas}

    t0 = time.time()
    for g in args.gammas:
        for ds in args.data_seeds:
            H = generate_spillover_logvol(N, args.n_steps, seed=ds, gamma=g)
            Y = H                                  # predict log-vol of all assets
            ang = np.tanh(H)                       # scale to (-1,1) for RY angles
            # classical features (seed-independent)
            F_har = har_features(H)
            F_win = window_feats(H, args.mem_depth)
            for h in args.horizons:
                results[g]["HAR_per_asset"][h].append(ridge_fit_predict(F_har, Y, h))
                results[g]["VAR_joint"][h].append(ridge_fit_predict(F_win, Y, h))
                results[g]["Ridge_window"][h].append(ridge_fit_predict(F_win, Y, h, alpha=10.0))
            for rs in args.res_seeds:
                # ESN (classical, seed rs)
                esn = esn_run(H, args.esn_size, seed=rs)
                # QRC (reservoir seed rs)
                H_res = bayat_reservoir_hamiltonian(q, seed=rs)
                wv, Vv = np.linalg.eigh(H_res)
                U = (Vv * np.exp(-1j * wv * 1.0)) @ Vv.conj().T
                z_ops = _z_ops(q)
                rng_s = np.random.default_rng(7000 + rs)
                Fq = qrc_features(ang, n1, n2, U, z_ops, args.mem_depth,
                                   rng_s, shots=args.shots)
                for h in args.horizons:
                    results[g]["ESN_joint"][h].append(ridge_fit_predict(esn, Y, h))
                    results[g]["QRC"][h].append(ridge_fit_predict(Fq, Y, h))
            print(f"  gamma={g} data_seed={ds} done ({time.time()-t0:.0f}s)")

    # ---- report ----
    summary = {"config": vars(args), "results": {}}
    print(f"\n{'gamma':>6} {'horizon':>7} | " +
          " ".join(f"{m:>13}" for m in methods) + " | QRC best?")
    print("-" * 96)
    for g in args.gammas:
        gd = {}
        for h in args.horizons:
            cells = {}
            for m in methods:
                vals = [v for v in results[g][m][h] if not np.isnan(v)]
                cells[m] = (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)
            gd[h] = cells
            qrc_mean = cells["QRC"][0]
            best_classical = min(cells[m][0] for m in methods if m != "QRC")
            win = qrc_mean < best_classical
            row = f"{g:>6} {h:>7} | " + " ".join(
                f"{cells[m][0]:>6.3f}+-{cells[m][1]:<4.2f}" for m in methods)
            print(row + f" | {'YES' if win else ''}")
        summary["results"][str(g)] = gd
        print()

    fpath = out_dir / f"mv_N{N}_q{q}_shots{args.shots}.json"
    with open(fpath, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {fpath}")
    print("NRMSE per asset (mean over assets & seeds). <1 = predictive. "
          "Thesis: QRC edge over best classical GROWS with gamma (spillover).")


if __name__ == "__main__":
    main()
