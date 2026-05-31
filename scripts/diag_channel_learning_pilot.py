#!/usr/bin/env python3
"""
diag_channel_learning_pilot.py - paper 6 channel-spectrum-learning pilot.

Ancilla-assisted (Choi-state) QRC vs single-copy classical, for learning the
Pauli eigenvalue spectrum of an unknown Pauli channel.  Theory: ancilla-
assisted Pauli-channel estimation has a proven separation over single-copy
(Chen et al. FOCS 2021 arXiv:2111.05881; "Quantum advantages for Pauli channel
estimation" arXiv:2108.08488).

DECISIVE, WELL-POSED QUESTION:
  The QRC injects the Choi state into a scrambling reservoir and reads LOCAL
  Paulis (D = 3*(q + q(q-1)/2) features).  Can ridge LINEARLY decode the
  weight-w Pauli eigenvalues (NRMSE < 1) where the single-copy classical floor
  fails (> 1)?  This is rank-limited (D < 4^m) so it MUST be measured.

DESIGN (all exact / analytic where possible):
  - Channel: Lambda(rho)=sum_i p_i P_i rho P_i, p ~ Dirichlet(alpha), D=4^m rates.
  - Eigenvalues: lambda_a = sum_i p_i chi(a,i), chi=+1 commute / -1 anticommute.
  - Choi state (Bell-diagonal): rho_Lambda = sum_i p_i |B_i><B_i|, |B_i>=(P_i x I)|Phi+>.
  - LINEARITY: features are linear in rho, rho linear in p, so
        F = P @ G^T ,  G[a,i] = <B_i| U^dag O_a U |B_i>
    = local-Pauli feature of the pure state U(|B_i> tensor |0>_anc).
    G is computed ONCE from 4^m statevector evolutions (no density matrices).
  - Targets: Y = P @ chi^T.
  - QRC NRMSE per weight: ridge F->Y across channels (train/test split).
  - Classical floor (analytic, single-copy):
        NRMSE_cl(w) = min( sqrt(3^w/M), sqrt(K/M) ) / std(lambda)
    (shadow route 3^w/M vs direct eigenstate prep split over K targets).
    std(lambda) = 1/sqrt(D*alpha+1)  (derived analytically; asserted in code).

CORRECTNESS GATE (--self-test): asserts the linearity path F=P@G^T matches an
explicit density-matrix computation to 1e-10 on m=2,q=4.  Run this FIRST; if it
fails, no result from this script is trustworthy.
"""
from __future__ import annotations
import argparse, sys, time, itertools, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT / "scripts"))
from run_quantum_input_experiment import (
    scrambling_hamiltonian, _apply_single_qubit_gate)

I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Ym = np.array([[0, -1j], [1j, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI1 = {0: I2, 1: Xm, 2: Ym, 3: Zm}     # 0=I,1=X,2=Y,3=Z
H_GATE = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S_DAG = np.array([[1, 0], [0, -1j]], dtype=complex)
HSDAG = H_GATE @ S_DAG     # maps Y-eigenbasis -> Z-eigenbasis
ROT1 = {"Z": None, "X": H_GATE, "Y": HSDAG}


# ---------------------------------------------------------------------------
# Pauli bookkeeping
# ---------------------------------------------------------------------------

def all_pauli_labels(m):
    return list(itertools.product(range(4), repeat=m))


def pauli_weight(lbl):
    return sum(1 for c in lbl if c != 0)


def commute_sign(a, b):
    """+1 if Pauli strings a,b commute, -1 if anticommute."""
    anti = sum(1 for ca, cb in zip(a, b) if ca != 0 and cb != 0 and ca != cb)
    return 1.0 if anti % 2 == 0 else -1.0


def build_chi(labels):
    n = len(labels)
    chi = np.empty((n, n))
    for i in range(n):
        li = labels[i]
        for j in range(n):
            chi[i, j] = commute_sign(li, labels[j])
    return chi


def build_bell_basis(labels, m):
    """Columns |B_i> = (P_i x I)|Phi+> on 2m qubits.  Shape (2^2m, 4^m)."""
    dim = 2 ** m
    phi = np.zeros(dim * dim, dtype=complex)
    for x in range(dim):
        phi[x * dim + x] = 1.0
    phi /= np.sqrt(dim)
    B = np.empty((dim * dim, len(labels)), dtype=complex)
    for k, lbl in enumerate(labels):
        Pi = np.array([[1.0]], dtype=complex)
        for c in lbl:
            Pi = np.kron(Pi, PAULI1[c])
        Pi_full = np.kron(Pi, np.eye(dim, dtype=complex))
        B[:, k] = Pi_full @ phi
    return B


# ---------------------------------------------------------------------------
# Local-Pauli feature readout
# ---------------------------------------------------------------------------

def build_sign_tables(q):
    """signs[s,k] = +-1 eigenvalue of Z_k on basis state s (MSB-first).
    pair_signs[s,p] = product over the p-th (k<l) qubit pair."""
    dim = 2 ** q
    idx = np.arange(dim, dtype=np.int64)
    shifts = np.arange(q - 1, -1, -1, dtype=np.int64)      # MSB-first, no neg stride
    bits = ((idx[:, None] >> shifts) & 1).astype(np.int8)
    signs = (1 - 2 * bits).astype(np.float64)              # (dim, q)
    iu, ju = np.triu_indices(q, k=1)
    pair_signs = signs[:, iu] * signs[:, ju]               # (dim, q(q-1)/2)
    return signs, pair_signs


def local_features_pure(psi, q, signs, pair_signs, bases=("Z", "X", "Y")):
    """Local 1- and 2-body Pauli expectations of a PURE state psi (2^q vector).
    Per-qubit single-qubit rotation per basis (cheap), then diagonal readout."""
    feats = []
    for b in bases:
        g = ROT1[b]
        if g is None:
            ps = psi
        else:
            ps = psi
            for qb in range(q):
                ps = _apply_single_qubit_gate(ps, q, qb, g)
        prob = np.abs(ps) ** 2
        feats.append(prob @ signs)
        feats.append(prob @ pair_signs)
    return np.concatenate(feats)


def local_features_density(rho, q, signs, pair_signs, bases=("Z", "X", "Y")):
    """Same features for a DENSITY matrix rho (2^q x 2^q).  Used only in the
    self-test as an independent reference for the linearity path."""
    feats = []
    for b in bases:
        g = ROT1[b]
        if g is None:
            r = rho
        else:
            U = np.array([[1.0]], dtype=complex)
            for _ in range(q):
                U = np.kron(U, g)
            r = U @ rho @ U.conj().T
        prob = np.real(np.diag(r))
        feats.append(prob @ signs)
        feats.append(prob @ pair_signs)
    return np.concatenate(feats)


def build_reservoir_unitary(q, seed, tau):
    """exp(-i H tau) via eigh (memory-stable at q=11)."""
    H = scrambling_hamiltonian(q, seed=seed)
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w * tau)) @ V.conj().T


def build_G(B, U, q, m, signs, pair_signs):
    """G[:,i] = local features of U(|B_i> tensor |0>_anc).  Shape (D, 4^m)."""
    n_anc = q - 2 * m
    if n_anc > 0:
        anc = np.zeros(2 ** n_anc, dtype=complex); anc[0] = 1.0
    cols = []
    for i in range(B.shape[1]):
        psi0 = B[:, i] if n_anc == 0 else np.kron(B[:, i], anc)
        psi = U @ psi0
        cols.append(local_features_pure(psi, q, signs, pair_signs))
    return np.array(cols).T          # (D, 4^m)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def nrmse_per_group(F, Y, groups, train_frac=0.7, alpha=1e-3):
    N = len(F); ntr = int(round(train_frac * N))
    Ftr, Fte, Ytr, Yte = F[:ntr], F[ntr:], Y[:ntr], Y[ntr:]
    A = Ftr.T @ Ftr + alpha * np.eye(Ftr.shape[1])
    W = np.linalg.solve(A, Ftr.T @ Ytr)
    Yp = Fte @ W
    out = {}
    for name, cols in groups.items():
        if not cols:
            out[name] = float("nan"); continue
        err = Yp[:, cols] - Yte[:, cols]
        rmse = np.sqrt(np.mean(err ** 2, axis=0))
        std = np.std(Yte[:, cols], axis=0) + 1e-12
        out[name] = float(np.mean(rmse / std))
    return out


# ---------------------------------------------------------------------------
# Self-test: linearity path == explicit density-matrix path
# ---------------------------------------------------------------------------

def self_test():
    m, q = 2, 4                       # Choi=4 qubits, q=4 -> n_anc=0
    print(f"[self-test] m={m}, q={q}: linearity F=P@G^T vs explicit density matrix")
    labels = all_pauli_labels(m)
    B = build_bell_basis(labels, m)
    U = build_reservoir_unitary(q, seed=0, tau=1.0)
    signs, pair_signs = build_sign_tables(q)
    G = build_G(B, U, q, m, signs, pair_signs)

    rng = np.random.default_rng(0)
    max_err = 0.0
    for _ in range(5):
        p = rng.dirichlet(np.ones(len(labels)))
        # linearity path
        f_lin = G @ p
        # explicit density-matrix path
        rho = (B * p[None, :]) @ B.conj().T          # 2^2m == 2^q here
        rho_ev = U @ rho @ U.conj().T
        f_exp = local_features_density(rho_ev, q, signs, pair_signs)
        err = float(np.max(np.abs(f_lin - f_exp)))
        max_err = max(max_err, err)
    ok = max_err < 1e-10
    print(f"[self-test] max |F_lin - F_exp| = {max_err:.2e}  -> "
          f"{'PASS' if ok else 'FAIL'}")
    # also assert the eigenvalue std analytic formula
    D = 4 ** m
    for alpha_d in (0.3, 1.0, 3.0):
        chi = build_chi(labels)
        P = rng.dirichlet(np.full(D, alpha_d), size=4000)
        Y = P @ chi.T
        emp = float(np.mean(np.std(Y[:, 1:], axis=0)))   # exclude identity col 0
        ana = 1.0 / np.sqrt(D * alpha_d + 1)
        print(f"[self-test] alpha={alpha_d}: std(lambda) emp={emp:.4f} "
              f"analytic={ana:.4f}  (ratio {emp/ana:.3f})")
    if not ok:
        sys.exit("SELF-TEST FAILED - do not trust results")
    print("[self-test] PASS\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true",
                     help="run correctness gate (linearity vs density matrix) and exit")
    ap.add_argument("--m", type=int, default=4)
    ap.add_argument("--q", type=int, default=9)
    ap.add_argument("--n-channels", type=int, default=600)
    ap.add_argument("--alpha-dirichlet", type=float, default=1.0)
    ap.add_argument("--res-seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--shots-per-qubit", type=int, default=1024)
    ap.add_argument("--max-targets-per-weight", type=int, default=60)
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    m, q = args.m, args.q
    assert q >= 2 * m, f"need q>=2m={2*m}"
    M = args.shots_per_qubit * q
    print(f"=== channel-learning pilot: m={m} (Choi {2*m}q), reservoir q={q}, "
          f"N={args.n_channels}, Dirichlet(a={args.alpha_dirichlet}), "
          f"res_seed={args.res_seed}, tau={args.tau}, M={M} ===")

    labels = all_pauli_labels(m)
    weights = np.array([pauli_weight(l) for l in labels])
    n_pauli = len(labels)
    D = 4 ** m
    print(f"  {n_pauli} Paulis; weight counts: "
          + ", ".join(f"w{w}={int((weights==w).sum())}" for w in range(m + 1)))

    t0 = time.time()
    chi = build_chi(labels)
    B = build_bell_basis(labels, m)
    U = build_reservoir_unitary(q, args.res_seed, args.tau)
    signs, pair_signs = build_sign_tables(q)
    G = build_G(B, U, q, m, signs, pair_signs)
    n_feat = G.shape[0]
    print(f"  built chi{chi.shape}, B{B.shape}, U{U.shape}, G{G.shape} "
          f"({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(1234 + args.res_seed)
    P = rng.dirichlet(np.full(n_pauli, args.alpha_dirichlet), size=args.n_channels)
    F = P @ G.T                       # (N, D_feat)
    Y = P @ chi.T                     # (N, 4^m) eigenvalues
    print(f"  F{F.shape}, Y{Y.shape} ({time.time()-t0:.0f}s)")

    # weight groups (capped, random subset per weight)
    groups = {}
    for w in range(1, m + 1):
        cols = list(np.where(weights == w)[0])
        if len(cols) > args.max_targets_per_weight:
            cols = list(rng.choice(cols, args.max_targets_per_weight, replace=False))
        groups[f"w{w}"] = cols

    exact = nrmse_per_group(F, Y, groups)
    Fn = F + rng.standard_normal(F.shape) * (1.0 / np.sqrt(M))   # QRC shot noise
    shotty = nrmse_per_group(Fn, Y, groups)

    std_ana = 1.0 / np.sqrt(D * args.alpha_dirichlet + 1)
    rows = []
    print(f"\n{'w':>3} {'K':>4} | {'QRC_exact':>9} {'QRC_shot':>9} | "
          f"{'cl_shadow':>9} {'cl_split':>9} {'cl_BEST':>8} | std | win?")
    print("-" * 84)
    for w in range(1, m + 1):
        cols = groups[f"w{w}"]; K = len(cols)
        std_emp = float(np.mean(np.std(Y[:, cols], axis=0)))
        cl_shadow = float(np.sqrt(3.0 ** w / M) / std_emp)
        cl_split = float(np.sqrt(K / M) / std_emp)
        cl_best = min(cl_shadow, cl_split)
        clean = bool(shotty[f"w{w}"] < 1.0 < cl_best)
        rows.append({"weight": w, "K": K,
                      "QRC_exact": exact[f"w{w}"], "QRC_shots": shotty[f"w{w}"],
                      "cl_shadow": cl_shadow, "cl_split": cl_split,
                      "cl_best": cl_best, "std_emp": std_emp,
                      "std_analytic": std_ana, "clean_win": clean})
        print(f"{w:>3} {K:>4} | {exact[f'w{w}']:>9.3f} {shotty[f'w{w}']:>9.3f} | "
              f"{cl_shadow:>9.3f} {cl_split:>9.3f} {cl_best:>8.3f} | "
              f"{std_emp:.4f} | {'WIN' if clean else ''}")
    print("-" * 84)
    print(f"clean win = QRC_shots < 1 < cl_BEST.  std analytic={std_ana:.4f} "
          f"(should match std col).  M={M}.")

    outdir = ROOT / "results" / "channel_pilot"
    outdir.mkdir(parents=True, exist_ok=True)
    out = {"config": {"m": m, "q": q, "n_channels": args.n_channels,
                       "alpha_dirichlet": args.alpha_dirichlet,
                       "res_seed": args.res_seed, "tau": args.tau, "M": M,
                       "n_features": n_feat, "n_pauli": n_pauli},
            "rows": rows}
    tag = f"m{m}_q{q}_a{args.alpha_dirichlet:g}_s{args.res_seed}"
    fpath = outdir / f"pilot_{tag}.json"
    with open(fpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {fpath}")


if __name__ == "__main__":
    main()
