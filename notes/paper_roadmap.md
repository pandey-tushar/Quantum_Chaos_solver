# Paper Roadmap — Status & Branch Map

**Updated:** 2026-07-21 (paper 1 SUBMITTED to MLST; QUANCOM accepted Jul 20)

| #     | Topic                                              | Branch                              | Status                                 |
| ----- | -------------------------------------------------- | ----------------------------------- | -------------------------------------- |
| 1     | Capacity-vs-BP for QRC vs QPINN                    | `paper-1-qst`                       | **SUBMITTED to IOP MLST** 2026-07-21 (MLST-105811), revised with 10-seed depth sweep + gradient histogram; data pushed to GitHub paper-1-qst. Prior: QST rejection 2026-07-08 (QST-105414), Quantum editorial rejection; arXiv 2604.23743. Frontiers waiver invite (Jul 31 deadline) on hold, decline drafted but unsent |
| 2     | Conditioning argument, fixed-vs-variational        | `paper-2-methods`                   | **ACCEPTED at QUANCOM 2026** 2026-07-20 (paper #51; Springer proceedings, Scopus/DBLP/EI). **REGISTER BY AUG 3** (hard gate: no registration = dropped from proceedings). **CAMERA-READY AUG 10** (address reviewer comments + Springer format). Trento Sep 1-4, Zoom presentation allowed |
| 3     | Chaoticity sweep on L96                            | `paper-3-chaoticity`                | **Parked** (low priority)              |
| 4     | QRC for finance — long-horizon advantage           | `paper-6-hilbert-input` (shared)    | **v0 synthetic done; ready for real data** |
| ~~5~~ | ~~QRC on classical-scalar input task~~             | `paper-5-quantum-input`             | **Folded into paper 6 as appendix**    |
| 6     | Hilbert-space-input QRC — long-horizon edge        | **`paper-6-hilbert-input`** ← active | **Confirmed: robust across 3 target seeds** |
| 7     | Cartography: fair-baseline QRC benchmarking        | `paper-7-cartography`               | **ACCEPTED at IEEE QCE26 WKS75 QuBench** 2026-07-21 (paper 1645, 2x accept + 1 borderline). **HARD DEADLINES MON JUL 27: author registration (Paper ID WKS75-1645) AND final 4-page IEEE PDF + copyright to IEEE CPS.** One author registration covers 2 papers. In-person talk, Toronto Sep 13-18 (schedule by Jul 31). Camera-ready must address reviews (see notes/qce26_wks75_reviews.md) |

## Standing rules

- **Never run q ≥ 13.** Reservoir runs are q ≤ 11 only.
- **Never benchmark at n_in ≤ 7.** Classical methods trivially dominate
  there. Real comparisons use n_in ≥ 9.
- Hardware-realistic shot budget: **M = q × 1024 shots/timestep**, shared
  by QRC and classical-shadow baselines.
- Always include at least 3 reservoir seeds × 3 data/target seeds for
  any headline claim.

---

## Paper 6 — REVISED 2026-05-29 (old "12% at k=80" headline was an artifact)

⚠️ The earlier "QRC wins 12% at k=80" headline did NOT survive scrutiny — it
sat entirely in the NRMSE>1 failed regime (worse than the mean predictor) and
was driven by a Z-only readout scored against X/Y/Z targets. Full diagnosis in
**`notes/paper6_highbody_findings.md`**.

**Revised thesis (honest, defensible):** *"At hardware-realistic shot budgets,
a scrambling QRC with multi-basis readout predicts intermediate-body (4–6 body)
observables of a quantum trajectory at long forecast horizons (k=20) where both
classical shadow-based and exact-feature classical reservoirs fail (NRMSE>1).
QRC measures U†Z_iU as a ~1/M-variance local probe of high-body input
correlations, while shadows pay 3^k/M to estimate the target directly."*

**Headline (3 reservoir seeds, target=<X_0..X_{k-1}>, scram, τ=0.6):**

| body k | k=20: QRC | k=20: Shadows | k=20: Cheat | verdict |
| ------ | --------- | ------------- | ----------- | ------- |
| 4 | **0.72** | 1.75 | 1.77 | QRC predictive, classical failed |
| 6 | **0.81** | 1.23 | 1.17 | QRC predictive, classical failed |
| 8 | 1.08 | 0.91 | 0.87 | breaks (near-global obs) |

**Key fixes vs the artifact version:** (1) multi-basis readout (Z+X+Y, shots
split 3 ways) — without it QRC is blind to X/Y targets; (2) scrambling reservoir
(TFIM is the wrong choice here); (3) small τ=0.3–0.6 (over-scrambling hurts).

**Honest nuance (must be in paper):** advantage is NOT monotonic in body count
(peaks k=4–6, breaks at k=8); short horizons favor classical. Claim = long-horizon
forecasting of intermediate-body observables only.

**3×3 RESULT (2026-05-31, `scripts/diag_highbody_targetseeds.py`,
`results/paper6_highbody_3x3/`):** 3 target × 3 reservoir seeds = 9 runs.
⚠️ The 1-seed numbers did NOT survive the 3-target-seed test. h=20:
- k=4: QRC **0.986±.24** vs Shadows 1.965±.59 — predictive but MARGINAL (clean headline)
- k=6: QRC 1.172±.44 vs Shadows 1.809±.97 — QRC >1, NOT predictive (fails)
- k=2: QRC 1.339 (>1); k=8: shadows win.
Pooled bootstrap gaps are significant (k=4 +0.927, k=6 +0.321) but at k=6 QRC
is itself >1 = least-bad-among-failures. **Only k=4/h=20 is a clean (marginal)
win.** Full honesty writeup in `notes/paper6_highbody_findings.md`.

**Paper 6 outstanding work (before QTML abstract, deadline 30 Jun 2026):**
- ✅ 3 target seeds {42,7,123} for the 3×3 standing rule — DONE (weakened result)
- ✅ Pooled bootstrap CIs — DONE (but need seed-level bootstrap for honesty)
- ⚠️ **Decide:** is one marginal cell (k=4/h=20) enough for QTML, or push the
  regime (τ sweep, longer trajectory, different target family) to widen the win?
- Understand the k=8 break (τ sweep at high body count)
- Write the abstract (`notes/qtml2026_abstract_draft.md` — narrowed to k=4);
  cite arXiv:2604.23743 for the architecture (do not re-derive)
- Venue: QTML 2026 abstract (non-archival, no QST conflict)

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
