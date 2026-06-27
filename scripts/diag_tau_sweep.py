#!/usr/bin/env python3
"""tau/reservoir sweep reusing one classical-feature computation.
Tests whether QRC evolution time or reservoir type unlocks an advantage
over the fixed classical baselines at n_in=9, q=11 (multi-basis readout)."""
import sys, time
from pathlib import Path
import numpy as np
from scipy import linalg as sla
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT/"scripts"))
from run_quantum_input_experiment import (build_target_hamiltonian, evolve_states,
    tfim_hamiltonian, scrambling_hamiltonian, _apply_single_qubit_gate,
    make_esn, esn_run, rff_features, ridge_solve, windowed_features)
from run_quantum_state_input import make_shadow_extractor, make_cheating_classical_extractor
from diag_multibasis_readout import make_qrc, multipauli_targets, eval_mt

n_in, q, nsteps, W = 9, 11, 1000, 5
hz = [1, 20, 80]
H_tgt = build_target_hamiltonian("heisenberg", n_in, 42, disorder=0.5)
psi0 = np.ones(2**n_in, dtype=complex)/np.sqrt(2**n_in)
psi_t = evolve_states(H_tgt, psi0, np.linspace(0, 60.0, nsteps))
y, letters = multipauli_targets(psi_t, n_in); yw = y[W:]
print(f"n_in={n_in} q={q} K={y.shape[1]} horizons={hz}")

# classical features ONCE
t0=time.time()
rng_s = np.random.default_rng(2000)
cstep,cD = make_cheating_classical_extractor(n_in, True)
fc = np.array([cstep(p) for p in psi_t])
sstep,sD = make_shadow_extractor(n_in, 1024, q, rng_s, include_2body=True)
fs = np.array([sstep(p) for p in psi_t])
Drff = W*198
cheat = eval_mt(rff_features(windowed_features(fc,W), Drff, 0), yw, hz)
Win,Wr,lk = make_esn(sD, Drff, 0)
shesn = eval_mt(windowed_features(esn_run(fs,Win,Wr,lk), W), yw, hz)
print(f"classical baselines ({time.time()-t0:.0f}s):")
print(f"  {'Cheat_RFF':<16} " + "  ".join(f"k{h}={cheat[h].mean():.2f}" for h in hz))
print(f"  {'Shadows_ESN':<16} " + "  ".join(f"k{h}={shesn[h].mean():.2f}" for h in hz))

print(f"\n{'reservoir':<8}{'tau':>5} | " + "  ".join(f"k{h} all/Z" for h in hz))
print("-"*60)
Hs = {"tfim": tfim_hamiltonian(q, J=1.0, g=1.0),
      "scram": scrambling_hamiltonian(q, seed=0)}
for res in ["tfim","scram"]:
    for tau in [0.3, 0.6, 1.0, 2.0, 4.0]:
        rng_q = np.random.default_rng(1000)
        step,D = make_qrc(q, n_in, Hs[res], tau, 1024, rng_q, bases=("Z","X","Y"))
        F = np.array([step(p) for p in psi_t])
        r = eval_mt(windowed_features(F,W), yw, hz)
        cells = "  ".join(f"{r[h].mean():.2f}/{r[h][letters=='Z'].mean():.2f}" for h in hz)
        print(f"{res:<8}{tau:>5.1f} | {cells}")
print("-"*60); print("cells = all-target / Z-target mean NRMSE. <1.0 predictive.")
