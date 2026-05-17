# Shot-Realistic QRC vs Classical: Honest Verdict

**Date:** 2026-05-17
**Branch:** paper-5-quantum-input
**Script:** `scripts/run_quantum_input_experiment.py` (now supports `--shot-mode`)

## The Question

At a **hardware-realistic shot budget** (M = q × 1024 shots per timestep) and
**hardware-realistic features** (local Pauli moments ⟨Z_i⟩ + ⟨Z_i Z_j⟩,
D = q(q+1)/2), can QRC beat a fixed-D classical baseline (RFF / ESN at the
same D) on the quantum-input task?

The earlier "QRC wins at q=5,7,9" results compared QRC's exponential 2^q
statevector probabilities against RFF at the same 2^q feature dim. Neither
side is hardware-realistic — you cannot read out a 2^q-dim feature vector
from poly(q) shots with any precision.

## What I Changed

`make_qrc_extractor` now supports two orthogonal modes:

- `--shot-mode {shots, exact}` — multinomial sampling vs exact `|⟨s|ψ⟩|²`
- `--shot-features {local_paulis, full_hist}` — D = q(q+1)/2 moments vs the
  full 2^q histogram

Default: `shot-mode=shots`, `shot-features=local_paulis`, `shots-per-qubit=1024`.
At q=7: M=7168 shots/step, D=28 per step, D_eff (windowed)=140.

## Smoke Result at q=7, 2 seeds, 500 timesteps

| Method                     | k=1 NRMSE | k=5 NRMSE | k=20 NRMSE |
| -------------------------- | --------- | --------- | ---------- |
| TFIM_QRC (shots, Paulis)   | 1.345     | 1.378     | 1.285      |
| SCRAM_QRC (shots, Paulis)  | 1.183     | 1.223     | 1.155      |
| ESN_matched (N=28)         | 1.085     | 1.084     | 1.082      |
| **RFF_matched_dim (D=28)** | **0.278** | **0.333** | **0.295**  |
| RFF_matched_effdim (D=140) | 0.285     | 0.310     | 0.285      |

**Under hardware-realistic features and shots, RFF beats QRC by ~5× NRMSE.**

## Isolation: Is It Shot Noise or the Feature Switch?

Run with `--shot-mode exact --shot-features local_paulis` (Pauli moments,
zero shot noise):

| Method (exact, local Paulis) | k=1 NRMSE |
| ---------------------------- | --------- |
| TFIM_QRC                     | 1.342     |
| SCRAM_QRC                    | 1.176     |
| RFF_matched_dim (D=28)       | 0.278     |

Numbers are **identical to within seed variation**. Conclusion: the QRC
collapse is from the **feature switch** (2^q histogram → local Pauli
moments), not from shot noise. The shot-noise floor σ ≈ 1/√M ≈ 0.012 at
q=7 is small relative to the signal, so on this task it adds nothing
to a verdict that was already determined by the feature-space restriction.

## Why This Is the Right Verdict

Three independent arguments from parallel agents converged:

1. **Input bottleneck (Schuld 2021):** x[t] is 3 classical scalars. The
   Fourier-series expressivity in x is bounded by (encoded-qubit count) ×
   (depth/τ), not by q. Growing q from 5 to 20 buys mixing capacity but
   no new frequencies. Windowed RFF on 15 scalars is already lossless.

2. **Bowles 2025 dequantization:** the RY-encoded trig-polynomial class
   is matchable by fixed-D RFF noiselessly. QRC pays shot noise on top of
   the same expressivity class.

3. **Sannia 2025 concentration:** global observables decay as 2^(-q/2)
   and drown in the shot floor at q≥15. Local Paulis survive, but they
   are classically computable from low-order Taylor expansions of the
   input → output map.

## What to Do Next

**Reposition paper 5.** The current "QRC wins on quantum-input task"
framing does not survive a hardware-realistic comparison. Honest
options:

1. **Honest negative-result paper:** "Under hardware-realistic shot budgets
   and local Pauli features, QRC does not outperform fixed-D classical
   reservoirs on classical-input prediction tasks, including a
   quantum-information-style task with partial-info inputs. The apparent
   advantage in prior small-q studies disappears once the comparison is
   matched at the hardware feature dim D = q(q+1)/2 rather than the
   unmeasurable 2^q histogram."

2. **Pivot to genuinely quantum inputs.** Design a task where the
   **input itself lives in Hilbert space** (D_input scales with 2^q):
   - Quantum state classification / functional regression: input = an
     unknown |ψ⟩ on q qubits, target = some f(|ψ⟩) (entanglement, fidelity).
   - Quantum-channel learning from shadow-tomography snapshots.
   - Hamiltonian-parameter inference from short-time dynamics.
   For these, classical methods cannot encode the input without
   exponential overhead — the QRC advantage argument becomes defensible.

3. **Add a recurrence-stress task.** ESN at matched N. Long-memory
   targets (NARMA-30+). The recurrence is the only QRC feature classical
   stateless RFF cannot match, so isolate it.

Recommended order: (1) write up the negative result honestly as a strong
paper 5 (deflates a community myth); (2) start paper 6 on a Hilbert-space
input task. Both are publishable.

## Open Run

A q-sweep `q ∈ {5, 7, 9, 11, 13, 15, 20}` with the new shot-realistic
defaults would tell us if QRC ever catches up at large q (it shouldn't,
per the Schuld bound, but worth verifying). Estimated time: ~30 min
total at 2000 timesteps × 3 seeds.
