#!/usr/bin/env python3
"""
diag_multibasis_readout.py - diagnostic for paper 6.

Hypothesis: the headline "QRC worse than mean" is an artifact of a Z-only
readout being scored against X/Y/Z targets. QRC reads only <Z_i>,<Z_iZ_j>
of the reservoir, so it is blind to X/Y target observables.

Fix under test: a multi-basis QRC readout that measures the reservoir in
the Z, X, and Y bases (splitting the SAME shot budget three ways), giving
<X_i>,<Y_i>,<Z_i> and same-letter 2-body moments -- matching the feature
set the classical shadow/cheating baselines already get.

Compares, on the multipauli target:
  - QRC_Zonly   : original Z-only readout (D = q(q+1)/2)
  - QRC_XYZ     : multi-basis readout  (D = 3*q(q+1)/2, M/3 shots per basis)
  - Cheat_RFF   : exact local X/Y/Z 1-2 body Paulis of the input + RFF
  - Shadows_ESN : shot-noisy shadow estimates + ESN

Reports overall and per-target-basis (X/Y/Z) NRMSE so we can see whether
the multi-basis readout lifts QRC's X/Y target performance up to its
Z-target performance.
"""
from __future__ import annotations
import argparse, time, sys
from pathlib import Path
import numpy as np
from scipy import linalg as sla

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from run_quantum_input_experiment import (
    I2, X, Y, Z, single_site, two_site,
    build_target_hamiltonian, evolve_states,
    tfim_hamiltonian, scrambling_hamiltonian,
    _apply_single_qubit_gate,
    make_esn, esn_run, rff_features,
    ridge_solve, windowed_features, feature_gram_condition,
)
from run_quantum_state_input import (
    make_shadow_extractor, make_cheating_classical_extractor,
)

H_GATE = (1.0 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)
S_DAG = np.array([[1, 0], [0, -1j]], dtype=complex)
HSDAG = H_GATE @ S_DAG     # rotate Y-eigenbasis into Z


def make_qrc(n_qubits, n_in, H_res, tau, shots_per_qubit, rng, bases=("Z",)):
    """State-injection QRC. `bases` lists measurement bases; the shot budget
    M = shots_per_qubit*q is split evenly across them."""
    q = n_qubits
    dim = 2 ** q
    M = shots_per_qubit * q
    M_per = max(1, M // len(bases))
    U = sla.expm(-1j * H_res * tau)
    idx = np.arange(dim, dtype=np.uint32)
    bits = ((idx[:, None] >> np.arange(q)[::-1]) & 1).astype(np.int8)
    signs = (1 - 2 * bits).astype(np.float64)
    iu, ju = np.triu_indices(q, k=1)
    pair_signs = signs[:, iu] * signs[:, ju]
    dim_rest = 2 ** (q - n_in)
    rest = np.zeros(dim_rest, dtype=complex); rest[0] = 1.0
    rot = {"Z": None, "X": H_GATE, "Y": HSDAG}

    def step(psi_in):
        psi0 = U @ np.kron(psi_in, rest)
        feats = []
        for b in bases:
            psi = psi0
            if rot[b] is not None:
                for qb in range(q):
                    psi = _apply_single_qubit_gate(psi, q, qb, rot[b])
            p = np.abs(psi) ** 2
            sp = p.sum();  p = p / sp if sp > 0 else p
            w = rng.multinomial(M_per, p).astype(np.float64) / M_per
            feats.append(w @ signs)
            feats.append(w @ pair_signs)
        return np.concatenate(feats)

    D = len(bases) * (q + q * (q - 1) // 2)
    return step, D


def multipauli_targets(psi_t, n):
    ops, letters = [], []
    for i in range(n):
        for pl in ("X", "Y", "Z"):
            ops.append(single_site(pl, i, n)); letters.append(pl)
    for i in range(n):
        for j in range(i + 1, n):
            for pl in ("X", "Y", "Z"):
                ops.append(two_site(pl, i, pl, j, n)); letters.append(pl)
    y = np.zeros((len(psi_t), len(ops)))
    for ti, psi in enumerate(psi_t):
        for ki, O in enumerate(ops):
            y[ti, ki] = float(np.real(psi.conj() @ (O @ psi)))
    return y, np.array(letters)


def eval_mt(F, y, horizons, train_frac=0.7, alpha=1.0):
    T, K = y.shape
    ntr = int(round(train_frac * T))
    out = {}
    for h in horizons:
        if T - h <= ntr + 5:
            out[h] = None; continue
        W = ridge_solve(F[:ntr - h], y[h:ntr], alpha=alpha)
        yp = F[ntr:T - h] @ W
        yte = y[ntr + h:T]
        nrmse = np.sqrt(np.mean((yp - yte) ** 2, axis=0)) / (np.std(yte, axis=0) + 1e-12)
        out[h] = nrmse
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-input", type=int, default=9)
    ap.add_argument("--n-qubits", type=int, default=11)
    ap.add_argument("--n-steps", type=int, default=1000)
    ap.add_argument("--t-max", type=float, default=60.0)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--reservoir", choices=["tfim", "scram"], default="scram")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots-per-qubit", type=int, default=1024)
    ap.add_argument("--target-seed", type=int, default=42)
    args = ap.parse_args()

    q, n_in = args.n_qubits, args.n_input
    print(f"=== multibasis diagnostic: n_in={n_in}, q={q}, reservoir={args.reservoir}, "
          f"tau={args.tau}, target_seed={args.target_seed} ===")

    H_tgt = build_target_hamiltonian("heisenberg", n_in, args.target_seed, disorder=0.5)
    psi0 = np.ones(2 ** n_in, dtype=complex) / np.sqrt(2 ** n_in)
    psi_t = evolve_states(H_tgt, psi0, np.linspace(0, args.t_max, args.n_steps))
    y, letters = multipauli_targets(psi_t, n_in)
    print(f"  target: K={y.shape[1]} multipauli observables, std={y.std():.3f}")

    rng_q = np.random.default_rng(1000 + args.seed)
    rng_s = np.random.default_rng(2000 + args.seed)
    H_res = (tfim_hamiltonian(q, J=1.0, g=1.0) if args.reservoir == "tfim"
             else scrambling_hamiltonian(q, seed=args.seed))

    feats = {}
    t0 = time.time()
    step_z, Dz = make_qrc(q, n_in, H_res, args.tau, args.shots_per_qubit, rng_q, bases=("Z",))
    feats["QRC_Zonly"] = (np.array([step_z(p) for p in psi_t]), Dz)
    print(f"  QRC_Zonly D={Dz} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    step_xyz, Dxyz = make_qrc(q, n_in, H_res, args.tau, args.shots_per_qubit, rng_q,
                              bases=("Z", "X", "Y"))
    feats["QRC_XYZ"] = (np.array([step_xyz(p) for p in psi_t]), Dxyz)
    print(f"  QRC_XYZ  D={Dxyz} ({time.time()-t0:.1f}s)")

    # Classical baselines
    t0 = time.time()
    cheat_step, cD = make_cheating_classical_extractor(n_in, include_2body=True)
    fc = np.array([cheat_step(p) for p in psi_t])
    feats["Cheat_RFF"] = (fc, cD, "rff")
    sh_step, sD = make_shadow_extractor(n_in, args.shots_per_qubit, q, rng_s, include_2body=True)
    fs = np.array([sh_step(p) for p in psi_t])
    feats["Shadows_ESN"] = (fs, sD, "esn")
    print(f"  classical features ({time.time()-t0:.1f}s)")

    W = args.window
    yw = y[W:]
    print(f"\n{'method':<14}{'D':>5} | " +
          "  ".join(f"k={h} (all/X/Y/Z)" for h in args.horizons))
    print("-" * 92)

    def report(name, F, D):
        res = eval_mt(F, yw, args.horizons)
        cells = []
        for h in args.horizons:
            nr = res[h]
            if nr is None:
                cells.append("  na"); continue
            a = nr.mean()
            xb = nr[letters == "X"].mean(); yb = nr[letters == "Y"].mean(); zb = nr[letters == "Z"].mean()
            cells.append(f"{a:.2f}/{xb:.2f}/{yb:.2f}/{zb:.2f}")
        print(f"{name:<14}{D:>5} | " + "   ".join(cells))

    for name in ["QRC_Zonly", "QRC_XYZ"]:
        F, D = feats[name]
        report(name, windowed_features(F, W), D)
    # classical
    Fc, cD, _ = feats["Cheat_RFF"]
    Fcw = windowed_features(Fc, W)
    report("Cheat_RFF", rff_features(Fcw, W * feats["QRC_XYZ"][1], args.seed), cD)
    Fs, sD, _ = feats["Shadows_ESN"]
    Win, Wr, lk = make_esn(sD, W * feats["QRC_XYZ"][1], args.seed)
    report("Shadows_ESN", windowed_features(esn_run(Fs, Win, Wr, lk), W), sD)
    print("-" * 92)
    print("cells = mean NRMSE all-targets / X-targets / Y-targets / Z-targets. <1.0 = predictive.")


if __name__ == "__main__":
    main()
