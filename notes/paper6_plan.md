# Paper 6: Hardware-Realistic Quantum Reservoir Advantage on Hilbert-Space Inputs

**Status:** Experiments COMPLETE; ready for paper write-up (2026-05-17)
**Branch:** `paper-6-hilbert-input`
**Absorbs:** all paper 5 work — classical-scalar-input results become
Appendix A ("hardware-realistic dequantization on scalar inputs").

## Confirmed result

3 target seeds × {5,3,3} reservoir seeds = 15 runs at n_in=9, q=11,
scram τ=1.0, multipauli K=135 targets, horizons {1, 5, 10, 20, 40, 80}:

**QRC wins by 12% at k=80, ties at k=40, loses at k≤20.**
ESN catastrophically collapses at long horizons (k=80 NRMSE ~1.6).
QRC has lowest variance (±0.14) → most stable predictor.

Headline plot: `results/paper6_longhz_aggregate.png`.
See `notes/paper_roadmap.md` for the full table and outstanding work.

## One-Line Thesis

QRC has a defensible, hardware-realistic advantage over classical reservoirs
**only when the input itself is a quantum state** (not a classical scalar
vector). On classical inputs, dequantization arguments (Schuld 2021, Bowles
2025) cap QRC expressivity below fixed-D RFF; on quantum-state inputs, the
classical method must first burn its shot budget on shadow tomography just
to *encode* the input, while QRC integrates the input into its dynamics
natively.

## Structural Argument (Why a Separation Should Exist)

| Resource           | QRC (quantum input)              | Classical + shadows (quantum input) |
| ------------------ | -------------------------------- | ----------------------------------- |
| Input access       | M copies of |ψ_t⟩ per timestep   | M copies of |ψ_t⟩ per timestep      |
| Per-step shot cost | M = q · 1024                     | M shadow snapshots                  |
| Feature extraction | reservoir evolution + Paulis     | random Clifford → poly(q) features  |
| Per-step dim       | q(q+1)/2 local Paulis            | poly(q) shadow estimators           |
| Recurrence         | quantum state carries between t  | classical features only             |

**The asymmetry:** the classical method has no way to *avoid* shadow
tomography — it cannot encode a 2^q-dim quantum state into a feature vector
without measuring it. The QRC method *uses the same M shots* to do both
input encoding and feature extraction simultaneously via the reservoir's
unitary mixing. This is the structural separator paper 5 was missing.

## Proposed Experiment

### Task: Quantum-state trajectory functional prediction

- **Input generator:** disordered Heisenberg chain on n_in = 5 qubits,
  evolved from |+⟩^5 under fixed H_T. Produces a trajectory |ψ_t⟩, t = 0..T.
- **Target:** y[t+k] = f(|ψ_{t+k}⟩) where f is a non-local functional that
  cannot be read off from a few single-site Paulis:
   - (a) half-chain second Rényi entropy S_2 (requires the full reduced
         density matrix)
   - (b) ⟨X_0 X_1 X_2 X_3⟩ (4-body observable, sample-cost exponential in
         body count via shadows)
   - (c) fidelity ⟨ψ_{t+k}|ψ_target⟩ to a fixed reference state
- **Horizons:** k ∈ {1, 5, 20}

### Method A: QRC with state injection (the proposed method)

1. At each t, prepare |ψ_t⟩ directly on the first n_in reservoir qubits
   (state-injection — physically: a SWAP gate from the input register).
2. Remaining (q − n_in) reservoir qubits initialized to |0⟩.
3. Evolve full reservoir under H_TFIM(q) for time τ.
4. Sample M = q · 1024 shots; extract local Paulis (D = q(q+1)/2).
5. Windowed ridge readout (w = 5).
6. **Crucially, recurrence:** the post-evolution reservoir state is NOT
   carried over (no quantum memory across timesteps in a NISQ device).
   The state-injection step re-prepares fresh input.
   The "memory" comes from windowed features only — same constraint as
   classical baseline. Fair.

### Method B: Classical shadows + RFF (the main baseline)

1. At each t, use M shots to perform classical shadow tomography on |ψ_t⟩:
   pick M random single-qubit Clifford bases, measure, store snapshots.
2. Estimate K = q(q+1)/2 local Paulis ⟨Z_i⟩, ⟨Z_iZ_j⟩, ⟨X_i⟩, ⟨X_iX_j⟩,
   ⟨Y_i⟩, ⟨Y_iY_j⟩ from snapshots via standard shadow estimators.
3. Windowed RFF (or linear ridge) on these features.

**Matched shot budget:** both methods get M = q · 1024 shots per
timestep. The classical method spends all M on tomography; QRC spends them
on reservoir-then-measure.

### Method C: Cheating classical (optional upper bound)

- Skip tomography: classical model gets EXACT Pauli expectations of |ψ_t⟩
  (i.e., the full no-noise statistical descriptor). RFF/ESN on these.
- If QRC beats this, advantage is structurally compelling.
- If QRC ties this, advantage comes only from the shadow shot cost.

## Predicted Outcome

- **Method C (cheating)** likely ties or beats QRC. This bounds the
  "pure dynamics" contribution.
- **Method B (shadows)** should lose to QRC because shadow shot noise
  on multi-body Paulis grows as 3^k / M for k-body observables. QRC's
  reservoir mixing keeps shot noise on its Pauli moments fixed at
  1/sqrt(M) independent of target body count.
- **Q-scaling:** the gap (QRC – Method B) should *grow* with q on
  many-body targets, because shadow cost for k-body observables scales
  as 3^k while QRC's effective features stay shot-noise-limited at
  sigma ~ 1/sqrt(M).

This is the clean q-scaling story paper 5 lacked.

## Why This Survives Dequantization

- **Schuld 2021 Fourier bound:** does not apply — input is not a scalar
  vector, no Fourier-series-in-x argument. Input lives in a 2^q-dim
  Hilbert space.
- **Bowles 2025 trig-poly:** does not apply — same reason; their proof
  assumes scalar-encoded inputs.
- **Sannia 2025 concentration:** still bites for global observables, but
  we use local Paulis as features. Concentration only matters for the
  *target* observable if it's global, and we use local k-body targets
  for that reason.

## Code Plan

- `scripts/run_quantum_state_input.py` (new): state-injection QRC,
  shadow-tomography classical, cheating classical.
- Reuse `make_qrc_extractor` with a new flag `--input-mode {classical, state_injection}`.
- Add a shadow-tomography module (~50 LOC).
- Q-sweep q ∈ {7, 9, 11, 13} (n_in = 5 fixed for now).

## Paper Structure

1. Intro: dequantization debate; need for honest separation
2. Setup: hardware shot model; local Paulis; shadows
3. Negative result on classical inputs (paper 5 material, condensed)
4. Hilbert-space input task: design and theoretical motivation
5. Main results: QRC vs Method B (shadows) at matched shot budget
6. Scaling with q and with target body count
7. Discussion: relation to Sannia, Bowles, Schuld
8. **Appendix A:** paper 5 full negative result + isolation tests

## Open Questions Before Coding

1. State-injection mechanism: physical (SWAP) or oracle? For simulation,
   it's just initialization of the input register, but we should be clear
   about the QPU primitive being modeled.
2. n_in vs q: do we fix n_in = 5 and grow q, or grow both? Growing both
   keeps input "matched" to reservoir but conflates two scaling axes.
   First experiment: fix n_in = 5, sweep q ∈ {7, 9, 11, 13}.
3. Target choice: pick ONE clean target for v0 (4-body Pauli ⟨X_0 X_1 X_2 X_3⟩
   is cleanest because shadow cost scales obviously with body count).
