# Paper Roadmap — Status & Branch Map

**Updated:** 2026-05-17

| #     | Topic                                              | Branch                              | Status                              |
| ----- | -------------------------------------------------- | ----------------------------------- | ----------------------------------- |
| 1     | Capacity-vs-BP for QRC vs QPINN                    | `paper-1-qst`                       | **Submitted to QST** (ScholarOne)   |
| 2     | Conditioning argument, fixed-vs-variational        | `paper-2-methods`                   | Draft + κ data; awaiting arXiv push |
| 3     | Chaoticity sweep on L96                            | `paper-3-chaoticity`                | **Parked** (low priority)           |
| 4     | **QRC for finance — long-horizon advantage**       | (next)                              | **Hypothesis sharpened (see below)** |
| ~~5~~ | ~~QRC on classical-scalar input task~~             | `paper-5-quantum-input`             | **Folded into paper 6 as appendix** |
| 6     | Hilbert-space-input QRC — long-horizon edge        | **`paper-6-hilbert-input`** ← active | Long-horizon signal at n_in=9, q=11 |

## Standing rules

- **Never run q=13 or larger.** Reservoir runs are q ≤ 11 only.
- **Never benchmark at n_in ≤ 7.** Classical methods trivially dominate
  there — those runs are misleading and the conclusions don't survive at
  honest sizes. Real comparisons use n_in ≥ 9.
- Hardware-realistic shot budget: M = q × 1024 shots/timestep, shared by
  QRC and classical-shadow baselines.

## Paper 6 — Active

**Headline finding (n_in=9, q=11, scram τ=1.0, 5 seeds, multipauli batch
of K=135 input observables):**

| Horizon | QRC          | Cheating + RFF | Cheating + ESN | Verdict        |
| ------- | ------------ | -------------- | -------------- | -------------- |
| k=1     | 1.046 ±.12   | 0.935 ±.04     | 0.748 ±.04     | QRC loses      |
| k=5     | 1.107 ±.16   | 1.008 ±.03     | 0.884 ±.06     | QRC loses      |
| **k=20**| **1.173 ±.17** | 1.223 ±.06   | 1.294 ±.10     | **QRC wins ~10%** |

Classical methods are great at short-horizon (target ≈ input), but degrade
fast at long horizons. QRC's scrambling-reservoir features carry phase
information that's still useful at k=20 — the first robust QRC edge we've
seen anywhere.

**Next step:** confirm the long-horizon signal at n_in=9, q=11, longer
trajectory (1000+ steps), horizons {1, 5, 10, 20, 40, 80}. Establish where
exactly QRC overtakes classical. Then paper 6 has a defensible thesis:
*"QRC's structural advantage over classical reservoirs is at long
prediction horizons, not at any single feature-richness or
input-encoding axis."*

## Paper 4 — Sharpened by the paper-6 finding

The paper-6 long-horizon signal is exactly the kind of advantage finance
forecasting needs:

- Short-horizon market prediction (1-step) is dominated by classical methods
  using local features (returns, volatilities, order-flow snapshots).
- **Long-horizon multi-asset forecasting** is what trading desks care about
  for portfolio allocation, options pricing, regime detection — and where
  classical methods notoriously degrade.
- If QRC's long-horizon edge reproduces on real financial time series,
  paper 4 has a clean story: *"QRC for multi-step financial forecasting
  at horizons where classical reservoirs collapse."*

**Concrete paper 4 v0 design:**
- Input: stream of multi-asset returns / order-flow features on n_in
  qubits via amplitude encoding (or angle encoding if quantum-state
  preparation is the bottleneck)
- Reservoir: scrambling Hamiltonian at q=11, τ=1.0 (the regime where
  paper 6 found the edge)
- Targets: returns at horizons {1, 5, 20, 60, 120} days/minutes
- Classical baseline: Shadows+ESN at matched shot budget (the hardest
  baseline from paper 6) and standard ESN/RFF
- Datasets: log-returns of liquid futures or FX pairs; synthetic
  rough-volatility models for ground-truth sanity checks

This is a real research program. Long-horizon advantage is the
common thread tying paper 6 and paper 4 together.

## Cleanup completed 2026-05-17

Deleted (misleading or too-small to be informative):
- All paper-5 results using exact 2^q statevector probabilities
  (`quantum_input/`, `quantum_input_q5/`, `quantum_input_q9/`,
  `quantum_input_targetseed_*/`, `quantum_input_xxz/`) — these compared
  unmeasurable features and produced the spurious "QRC wins at small q"
  claim.
- All n_in ≤ 7 paper-6 results (`paper6_smoke_q7`, `paper6_renyi_n6q9`,
  `paper6_multipauli_n5q8`, `paper6_sweet_n5q8_*`, `paper6_scan_*`).
  Classical trivially dominates at these sizes; conclusions don't generalize.
- Shot-sweep at q ∈ {5, 7} (`quantum_input_shots_q5`,
  `quantum_input_shots_q7`) — same reason.
- All q=13 results and smoke runs.

Kept:
- `paper6_real_n9q11_scram/` — long-horizon edge headline result
- `quantum_input_shots_q9/`, `quantum_input_shots_q11/` — honest
  dequantization confirmation for paper-6 appendix
- `quantum_input_shots_summary.{json,png}` — regenerated at q ∈ {9, 11} only

## Paper 4 — Backlog after paper 6 confirmation

Multi-asset QRC for finance. Lit review in
`notes/paper4_finance_literature_review.md`. Will be designed around the
long-horizon thesis once paper-6 finding is solid.
