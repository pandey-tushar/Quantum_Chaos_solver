# Paper Roadmap — Status & Branch Map

**Updated:** 2026-05-17 (end of paper-6 + paper-4-v0 sprint)

| #     | Topic                                              | Branch                              | Status                                 |
| ----- | -------------------------------------------------- | ----------------------------------- | -------------------------------------- |
| 1     | Capacity-vs-BP for QRC vs QPINN                    | `paper-1-qst`                       | **Submitted to QST** (ScholarOne)      |
| 2     | Conditioning argument, fixed-vs-variational        | `paper-2-methods`                   | Draft + κ data; awaiting arXiv push    |
| 3     | Chaoticity sweep on L96                            | `paper-3-chaoticity`                | **Parked** (low priority)              |
| 4     | QRC for finance — long-horizon advantage           | `paper-6-hilbert-input` (shared)    | **v0 synthetic done; ready for real data** |
| ~~5~~ | ~~QRC on classical-scalar input task~~             | `paper-5-quantum-input`             | **Folded into paper 6 as appendix**    |
| 6     | Hilbert-space-input QRC — long-horizon edge        | **`paper-6-hilbert-input`** ← active | **Confirmed: robust across 3 target seeds** |

## Standing rules

- **Never run q ≥ 13.** Reservoir runs are q ≤ 11 only.
- **Never benchmark at n_in ≤ 7.** Classical methods trivially dominate
  there. Real comparisons use n_in ≥ 9.
- Hardware-realistic shot budget: **M = q × 1024 shots/timestep**, shared
  by QRC and classical-shadow baselines.
- Always include at least 3 reservoir seeds × 3 data/target seeds for
  any headline claim.

---

## Paper 6 — Confirmed

**Thesis:** *"At hardware-realistic shot budgets on Hilbert-space inputs,
QRC matches or beats classical reservoirs at horizons k ≥ 40. Recurrent
classical methods (ESN) catastrophically degrade at long horizons; QRC's
fixed unitary dynamics avoid this failure mode."*

**Headline (3 target seeds × {5,3,3} reservoir seeds = 15 runs):**

| Horizon | QRC          | Best classical            | Gap        | Verdict        |
| ------- | ------------ | ------------------------- | ---------- | -------------- |
| k=1     | 1.050 ± .15  | Cheat+ESN 0.679           | +0.371     | QRC loses 55%  |
| k=10    | 1.115 ± .17  | Cheat+ESN 0.974           | +0.141     | QRC loses 14%  |
| k=20    | 1.191 ± .16  | Cheat+RFF 1.123           | +0.068     | QRC loses 6%   |
| k=40    | 1.311 ± .18  | Cheat+RFF 1.305           | +0.006     | tie            |
| **k=80**| **1.367 ± .14** | Cheat+RFF 1.556        | **−0.188** | **QRC WINS 12%** |

Setup: n_in=9, q=11, scram τ=1.0, multipauli K=135 targets, 1000-1200
steps. Plots in `results/paper6_longhz_aggregate.png` and
`results/paper6_longhz_n9q11{,_ts100,_ts200}/`.

**Mechanism:** classical recurrent methods accumulate error over time;
QRC's fixed reservoir unitary doesn't. ESN and Cheat+ESN both hit ~1.6
at k=80; RFF stays at ~1.56; QRC at 1.37. QRC also has lowest variance
(±0.14) → most stable predictor.

**Paper 6 outstanding work (before submission):**
- Write the paper (4 main sections + paper-5 appendix)
- Add 1-2 more reservoir Hamiltonian choices (e.g., SYK with non-uniform
  couplings) to show robustness of the long-horizon edge to reservoir
  choice
- Consider a Pauli-shadows-with-derandomization baseline to address the
  Bertoni 2025 shallow-shadows critique
- Decide venue (PRX Quantum, Quantum, or PRX Energy)

---

## Paper 4 — v0 done, ready for real data

**Thesis (working):** *"For long-horizon realized-volatility forecasting
on multi-asset financial data, hardware-realistic QRC produces feature
subspaces that strictly outperform classical reservoirs and their
regularized variants, under matched shot and parameter budget."*

**v0 result on synthetic multi-asset regime-switching returns**
(9 assets, 2000 steps, 3 seeds, realized-vol target):

| Horizon | QRC          | Best classical      | Verdict        |
| ------- | ------------ | ------------------- | -------------- |
| k=1     | 1.071        | RFF 1.044           | QRC loses 3%   |
| k=10    | 1.052        | RFF 1.052           | tie            |
| k=20    | **1.085**    | RFF 1.097           | QRC wins 1%    |
| k=40    | **1.176**    | RFF 1.197           | QRC wins 2%    |
| **k=80**| **1.354**    | RFF 1.395           | **QRC wins 3%** |

ESN/linear catastrophically collapse at long horizons (k=80 NRMSE ~2.4
vs QRC's 1.354) — same failure mode as paper 6.

**Mechanism isolation passed:** Neither RFF+noise(σ=0.01) nor RFF+
strongreg(α=100) replicates the QRC edge. The advantage is a real
feature-quality effect, not a regularization artifact.

**Paper 4 outstanding work:**
1. **Real financial data** (next step). 9 daily FX log-returns or 9 liquid
   futures, ~5000 obs. If 3% edge at k=80 holds, paper 4 has empirical
   core. Use same n_in=9 / q=11 / scram / τ=1.0 regime.
2. **Richer synthetic** (parallel) — add long-memory volatility
   (fractional noise) and jumps to see if QRC edge grows with data
   complexity.
3. **Horizon-extension sweep** to characterize the QRC edge across
   {1, 5, 10, 20, 40, 80, 160, 320} for the publication plot.
4. **Decide venue** (Quantitative Finance, JFE, or quant-conference).

---

## Paper 2 — Awaiting arXiv push

Joint methods paper with conditioning argument. κ_QRC bounded vs
κ_ESN exponential. Branch `paper-2-methods` has draft + κ data.
Action: arXiv upload + cite paper 1.

## Paper 3 — Parked

L96 chaoticity sweep done but parked due to low priority. Branch
`paper-3-chaoticity` preserved.

---

## Repo cleanup ledger (2026-05-17)

Deleted (misleading or too small):
- All paper-5 exact-statevector results (compared unmeasurable 2^q
  features)
- All n_in ≤ 7 paper-6 results (classical trivially wins; conclusions
  don't generalize)
- Shot-sweep at q ∈ {5, 7} (too small)
- All q=13 results (standing rule: max q=11)
- `paper6_smoke_*`, `paper6_scan_*`, `paper6_renyi_*`, etc.

Kept (informative, headline):
- `results/paper6_longhz_n9q11{,_ts100,_ts200}/` — paper-6 headline
  long-horizon runs (3 target seeds)
- `results/paper6_longhz_aggregate.png` — 3-seed aggregate plot
- `results/quantum_input_shots_q{9,11}/` — paper-6 appendix dequantization
  evidence
- `results/quantum_input_shots_summary.{json,png}` — paper-6 appendix sweep
- `results/paper4_synthetic_rv/` — paper-4 v0 realized-vol headline
- `results/paper4_synthetic_rv_iso/` — paper-4 mechanism isolation evidence
- `results/paper4_synthetic_cum/` — paper-4 null result on cumulative-return
  target (kept as honest negative)

## Reusable infrastructure

- `scripts/run_quantum_state_input.py` — paper-6 main script
  (state-injection QRC, classical shadows w/ noise-model, multipauli batch)
- `scripts/run_paper4_finance_synthetic.py` — paper-4 main script
  (synthetic multi-asset returns, angle-encoded QRC inputs, mechanism
  isolation baselines)
- `scripts/plot_paper6_horizon.py` — universal NRMSE-vs-horizon plotter
  (works on both paper-6 and paper-4 summary.json)
- `scripts/aggregate_paper6_targetseeds.py` — combine multi-target-seed runs
- `scripts/run_quantum_input_experiment.py` — paper-5 dequantization
  experiment (now part of paper-6 appendix)
- `scripts/summarize_shot_sweep.py` — paper-6 appendix q-sweep aggregator
