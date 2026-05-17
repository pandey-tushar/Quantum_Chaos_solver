# Paper Roadmap — Status & Branch Map

**Updated:** 2026-05-17

| #     | Topic                                              | Branch                 | Status                              |
| ----- | -------------------------------------------------- | ---------------------- | ----------------------------------- |
| 1     | Capacity-vs-BP for QRC vs QPINN                    | `paper-1-qst`          | **Submitted to QST** (ScholarOne)   |
| 2     | Conditioning argument, fixed-vs-variational        | `paper-2-methods`      | Draft + κ data; awaiting arXiv push |
| 3     | Chaoticity sweep on L96                            | `paper-3-chaoticity`   | **Parked** (low priority)           |
| 4     | QRC for multi-asset finance                        | (not started)          | Lit review done; design TBD         |
| ~~5~~ | ~~QRC win on quantum-input classical-scalar task~~ | `paper-5-quantum-input` | **Folded into paper 6 as appendix** |
| 6     | Hilbert-space-input QRC advantage                  | **`paper-6-hilbert-input`** ← active | Script v0 written, smoke test next  |

## Why paper 5 became an appendix

Paper 5's original thesis ("QRC beats classical reservoirs on a quantum-input
task") did not survive the hardware-realistic comparison:

- Original setup: input x[t] = 3 classical scalars (Pauli expectations of a
  4-qubit Heisenberg target). QRC features = exact 2^q statevector
  probabilities. RFF at matched D = 2^q.
- That comparison was **not hardware-realistic** — neither side's features
  are extractable from a poly(q)-shot budget on a real device.
- Under realistic features (local Pauli moments, D = q(q+1)/2 from
  M = q·1024 shots), RFF beats QRC by ~3× NRMSE at every q ∈ {5,7,9,11}.
  Isolation test confirms: it's the feature switch (2^q → local Paulis),
  not shot noise, that kills the apparent QRC advantage.
- Root cause: classical-scalar input bottleneck (Schuld 2021 Fourier bound,
  Bowles 2025 trig-poly dequantization). Adding reservoir qubits doesn't add
  new frequencies in x.

The negative result is publishable but not by itself — it works as an
appendix to paper 6, which presents the regime where QRC *does* win and
explains why.

## Paper 6 — Active

**Thesis:** QRC has a defensible, hardware-realistic advantage over
classical reservoirs **only when the input itself is a quantum state**
(D_input ∝ 2^q). On scalar inputs (paper 5 appendix), QRC ≤ fixed-D RFF.
On Hilbert-space inputs, the classical baseline must spend its shot budget
on shadow tomography just to encode the input, while QRC integrates input
encoding and reservoir mixing simultaneously.

**Design (see `notes/paper6_plan.md` for full detail):**
- Task: predict a k-body Pauli expectation ⟨X_0…X_{k−1}⟩(t+horizon) on a
  trajectory |ψ_t⟩ from a disordered Heisenberg input system.
- Method A (QRC): state-injection of |ψ_t⟩ onto the first n_in reservoir
  qubits; TFIM evolution; M = q·1024 shots; local Pauli features.
- Method B (classical shadows + RFF): Pauli-shadow tomography on |ψ_t⟩ at
  matched shot budget; estimate ⟨X⟩, ⟨Y⟩, ⟨Z⟩ and 2-body terms; RFF readout.
- Method C (cheating classical): exact local Paulis on |ψ_t⟩ + RFF — upper
  bound for any classical method that uses local features.
- Sweep q ∈ {7, 9, 11, 13}; n_in fixed = 5.

**Expected separation:** shadow shot cost for k-body observables grows as
3^k / M, while QRC's local Pauli features stay at σ ≈ 1/√M independent of
target body count. The gap (QRC − shadows) should grow with both q and k.

**Script:** `scripts/run_quantum_state_input.py` (v0 written; shadow
tomography now batched by basis pattern for ~80× speedup on n_in=4).

**Status:** about to run smoke test (n_in=4, q=7, 400 steps, 2 seeds, 4-body target).

## Paper 4 — Backlog

Multi-asset QRC for finance. Lit review in `notes/paper4_finance_literature_review.md`.
Concrete design depends on paper 6 outcome — same hardware-realistic
framework should carry over once we know what works.
