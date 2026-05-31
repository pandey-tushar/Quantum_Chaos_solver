#!/usr/bin/env python3
"""
diag_channel_learning_pilot.py - paper 6 NEW DIRECTION pilot.

Channel-learning QRC vs single-copy classical, the 2-copy advantage zone
(Chen-Cotler-Huang-Li FOCS 2021 / Huang et al. Science 2022).

DECISIVE QUESTION (this pilot): can a scrambling reservoir's LOCAL Pauli
readout of the 2-copy Choi state LINEARLY DECODE high-weight Pauli
eigenvalues of an unknown Pauli channel?  If YES (exact NRMSE << 1 at
high weight), the reservoir advantage is real and we add shot noise.
If NO (exact NRMSE ~ 1), generic scrambling cannot decode -> the
reservoir framing is a dead end (would need a structured Bell
measurement, which is not a reservoir).

Setup:
  - Unknown m-qubit Pauli channel Lambda(rho) = sum_i p_i P_i rho P_i,
    rates p ~ Dirichlet(alpha) over the 4^m Paulis.
  - Pauli eigenvalues lambda_i = sum_j p_j chi(P_i,P_j), chi = +1 if
    [P_i,P_j]=0 else -1 (Walsh-Hadamard transform of the rates).
  - Choi state (Bell-diagonal): rho_Lambda = sum_i p_i |B_i><B_i| on 2m
    qubits.  We build it as a density matrix.
  - QRC features: inject Choi state (+ ancillas) into q-qubit scrambling
    reservoir, evolve rho -> U rho U^dag, take EXACT local Pauli
    expectations (1- and 2-body, Z/X/Y bases).  [exact = no shots here]
  - Decode: ridge from features -> eigenvalue targets, across N channels.
  - Report EXACT (noiseless) NRMSE per Pauli weight.

Also reports the analytic classical single-copy shadow NRMSE per weight
( sqrt(3^w / M) / std(lambda_w) ) for context.

Pilot defaults: m=4 (Choi 8 qubits), q=9 reservoir, fast.
"""
from __future__ import annotations
import argparse, sys, time, itertools
from pathlib import Path
import numpy as np
from scipy import linalg as sla
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT / "scripts"))
from run_quantum_input_experiment import (
    scrambling_hamiltonian, _apply_single_qubit_gate)

I2 = np.eye(2, dtype=complex)
Xm = np.array([[0, 1], [1, 0]], dtype=complex)
Ym = np.array([[0, -1j], [1j, 0]], dtype=complex)
Zm = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI1 = {0: I2, 1: Xm, 2: Ym, 3: Zm}     # 0=I,1=X,2=Y,3=Z
H_GATE = (1/np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S_DAG = np.array([[1, 0], [0, -1j]], dtype=complex)
HSDAG = H_GATE @ S_DAG


def all_pauli_labels(m):
    """All 4^m Pauli labels as tuples in {0,1,2,3}^m (0=I)."""
    return list(itertools.product(range(4), repeat=m))


def pauli_weight(lbl):
    return sum(1 for c in lbl if c != 0)


def commute_sign(a, b):
    """+1 if Pauli strings a,b commute, -1 if anticommute.
    Two single-qubit Paulis anticommute iff both non-identity and different."""
    anti = 0
    for ca, cb in zip(a, b):
        if ca != 0 and cb != 0 and ca != cb:
            anti += 1
    return 1.0 if anti % 2 == 0 else -1.0


def eigenvalues_from_rates(p, labels):
    """lambda_i = sum_j p_j chi(P_i, P_j).  Returns array over labels."""
    n = len(labels)
    chi = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            chi[i, j] = commute_sign(labels[i], labels[j])
    return chi @ p


def build_bell_basis(labels, m):
    """Return B: columns are the Bell-state vectors |B_i> = (P_i x I)|Phi+>
    on 2m qubits.  Shape (2^2m, 4^m).  Precompute once, reuse per channel."""
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


def bell_diagonal_choi(p, B):
    """rho = sum_i p_i |B_i><B_i| = (B * p) @ B^dag.  Fast (B precomputed)."""
    return (B * p[None, :]) @ B.conj().T


def build_feature_helpers(q, bases=("Z", "X", "Y")):
    """Precompute sign tables and per-basis full rotation unitaries once."""
    dim = 2 ** q
    idx = np.arange(dim, dtype=np.uint64)
    bits = ((idx[:, None] >> np.arange(q)[::-1]) & 1).astype(np.int8)
    signs = (1 - 2 * bits).astype(np.float64)
    iu, ju = np.triu_indices(q, k=1)
    pair_signs = signs[:, iu] * signs[:, ju]
    rot1 = {"Z": None, "X": H_GATE, "Y": HSDAG}
    Us = {}
    for b in bases:
        if rot1[b] is None:
            Us[b] = None
        else:
            U = np.array([[1.0]], dtype=complex)
            for _ in range(q):
                U = np.kron(U, rot1[b])
            Us[b] = U
    return signs, pair_signs, Us, bases


def local_pauli_features_exact(rho, helpers):
    """Exact local 1- and 2-body Pauli expectations of rho, using precomputed
    helpers = (signs, pair_signs, Us, bases)."""
    signs, pair_signs, Us, bases = helpers
    feats = []
    for b in bases:
        U = Us[b]
        r = rho if U is None else U @ rho @ U.conj().T
        diag = np.real(np.diag(r))
        feats.append(diag @ signs)
        feats.append(diag @ pair_signs)
    return np.concatenate(feats)


def nrmse_per_group(F, Y, groups, train_frac=0.7, alpha=1e-3):
    """Ridge F->Y across samples; return dict group->mean NRMSE over targets
    in that group.  groups: dict name->list of column indices."""
    N = len(F); ntr = int(round(train_frac * N))
    Ftr, Fte = F[:ntr], F[ntr:]
    Ytr, Yte = Y[:ntr], Y[ntr:]
    # ridge (primal, F is wide-ish but N modest)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4, help="channel qubits (Choi=2m)")
    ap.add_argument("--q", type=int, default=9, help="reservoir qubits (>=2m)")
    ap.add_argument("--n-channels", type=int, default=400)
    ap.add_argument("--alpha-dirichlet", type=float, default=1.0)
    ap.add_argument("--res-seed", type=int, default=0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--shots-per-qubit", type=int, default=1024)
    ap.add_argument("--max-targets-per-weight", type=int, default=60,
                     help="cap targets per weight group for speed/identifiability")
    args = ap.parse_args()
    m, q = args.m, args.q
    assert q >= 2 * m, f"need q>=2m={2*m}"
    M = args.shots_per_qubit * q
    print(f"=== channel-learning pilot: m={m} (Choi {2*m}q), reservoir q={q}, "
          f"N={args.n_channels} channels, Dirichlet(a={args.alpha_dirichlet}), "
          f"M={M} ===")

    labels = all_pauli_labels(m)
    weights = np.array([pauli_weight(l) for l in labels])
    n_pauli = len(labels)
    wc = {int(w): int((weights == w).sum()) for w in range(m + 1)}
    print(f"  {n_pauli} Paulis (expect 4^{m}={4**m}); weight counts: "
          + ", ".join(f"w{w}={wc[w]}" for w in range(m + 1))
          + f"  [expected C({m},w)*3^w]")

    # reservoir unitary (q qubits)
    H_res = scrambling_hamiltonian(q, seed=args.res_seed)
    U = sla.expm(-1j * H_res * args.tau)
    dim_q = 2 ** q
    dim_choi = 2 ** (2 * m)
    n_anc = q - 2 * m
    anc = np.zeros((2 ** n_anc, 2 ** n_anc), dtype=complex) if n_anc > 0 else None
    if anc is not None:
        anc[0, 0] = 1.0

    # precompute eigenvalue (Walsh-Hadamard) transform matrix chi once
    t_chi = time.time()
    chi = np.empty((n_pauli, n_pauli))
    for i in range(n_pauli):
        for j in range(n_pauli):
            chi[i, j] = commute_sign(labels[i], labels[j])
    print(f"  built chi {chi.shape} ({time.time()-t_chi:.0f}s)")
    B = build_bell_basis(labels, m)
    print(f"  built Bell basis {B.shape}")

    helpers = build_feature_helpers(q)
    rng = np.random.default_rng(1234)
    t0 = time.time()
    feats, eigs = [], []
    for c in range(args.n_channels):
        p = rng.dirichlet(np.full(n_pauli, args.alpha_dirichlet))
        lam = chi @ p
        rho_choi = bell_diagonal_choi(p, B)                    # 2^2m
        rho_q = rho_choi if anc is None else np.kron(rho_choi, anc)
        rho_ev = U @ rho_q @ U.conj().T
        f = local_pauli_features_exact(rho_ev, helpers)
        feats.append(f); eigs.append(lam)
        if (c + 1) % 100 == 0:
            print(f"  {c+1}/{args.n_channels} channels ({time.time()-t0:.0f}s)")
    F = np.array(feats); Y = np.array(eigs)
    print(f"  features D={F.shape[1]}, targets K={Y.shape[1]} "
          f"({time.time()-t0:.0f}s)")

    # build weight groups, capped
    groups = {}
    for w in range(1, m + 1):
        cols = list(np.where(weights == w)[0])
        if len(cols) > args.max_targets_per_weight:
            cols = list(rng.choice(cols, args.max_targets_per_weight, replace=False))
        groups[f"w{w}"] = cols

    # EXACT (noiseless) QRC decoding NRMSE per weight
    exact = nrmse_per_group(F, Y, groups)

    # QRC + shot noise (add 1/sqrt(M) Gaussian to features)
    Fn = F + rng.standard_normal(F.shape) * (1.0 / np.sqrt(M))
    shotty = nrmse_per_group(Fn, Y, groups)

    # analytic classical single-copy shadow NRMSE per weight:
    #   estimate of lambda_i = lambda_i + N(0, 3^w/M); regression can't
    #   denoise (eigenvalues uncorrelated across channels), so NRMSE =
    #   sqrt(3^w/M)/std(lambda_w).
    import json
    rows = []
    print(f"\n{'weight':>6} {'#tgt':>5} | {'QRC_exact':>10} {'QRC_shots':>10} "
          f"{'Classical':>10} | std(lambda)")
    print("-" * 64)
    for w in range(1, m + 1):
        cols = groups[f"w{w}"]
        std_w = float(np.mean(np.std(Y[:, cols], axis=0)))
        cl = float(np.sqrt(3.0 ** w / M) / (std_w + 1e-12))
        clean = bool(exact[f"w{w}"] < 1 < cl)
        tag = ("QRC wins" if clean else "")
        rows.append({"weight": w, "n_tgt": len(cols),
                      "QRC_exact": exact[f"w{w}"], "QRC_shots": shotty[f"w{w}"],
                      "classical": cl, "std_lambda": std_w, "clean_win": clean})
        print(f"{w:>6} {len(cols):>5} | {exact[f'w{w}']:>10.3f} "
              f"{shotty[f'w{w}']:>10.3f} {cl:>10.3f} | {std_w:.4f}  {tag}")
    print("-" * 64)
    print("EXACT NRMSE is the decisive number: <1 => scrambled local readout "
          "CAN linearly decode that weight; ~1 => cannot (dead end).")
    print(f"(M={M} shots; classical = sqrt(3^w/M)/std, single-copy shadow floor.)")

    outdir = ROOT / "results" / "channel_pilot"
    outdir.mkdir(parents=True, exist_ok=True)
    out = {"config": {"m": m, "q": q, "n_channels": args.n_channels,
                       "alpha_dirichlet": args.alpha_dirichlet,
                       "res_seed": args.res_seed, "tau": args.tau, "M": M,
                       "n_features": int(F.shape[1]), "n_pauli": n_pauli,
                       "weight_counts": wc},
            "rows": rows}
    tag = f"m{m}_q{q}_a{args.alpha_dirichlet:g}_s{args.res_seed}"
    with open(outdir / f"pilot_{tag}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {outdir / f'pilot_{tag}.json'}")


if __name__ == "__main__":
    main()
