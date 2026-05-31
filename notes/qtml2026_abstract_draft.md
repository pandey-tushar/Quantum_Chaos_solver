# QTML 2026 Abstract — Draft

**Venue:** QTML 2026, Stellenbosch, South Africa, Dec 6–11.
**Deadline:** 30 June 2026. Abstract-based, non-archival (no QST conflict).
**Architecture citation:** arXiv:2604.23743 (cite, do not re-derive).
**Status:** ⚠️ NEEDS REVISION. The 3×3 run (`results/paper6_highbody_3x3/`)
weakened the story: only k=4/h=20 survives as a (marginal) headline; k=6 does
NOT (QRC>1.0 across seeds 7,123). Numbers below are the REAL 3×3 values. The
abstract prose must be narrowed to the single k=4 cell or the regime pushed
further before submission. See `notes/paper6_highbody_findings.md`.

---

## Title (candidates)

1. *Long-horizon forecasting of intermediate-body quantum observables: a
   hardware-realistic quantum-reservoir advantage*
2. *Where quantum reservoirs help: shot-budget-matched prediction of
   intermediate-body trajectory observables*

## Abstract (target ~200–250 words)

Quantum reservoir computing (QRC) is often claimed to offer an advantage over
classical learning, but many comparisons are not hardware-realistic: they
compare exponentially-sized statevector features against finite-dimensional
classical models, or use under-tuned classical baselines. We ask a sharper,
shot-budget-matched question: given a fixed measurement budget of M shots per
timestep, is there a regime where a QRC strictly outperforms strong classical
baselines that share the same budget?

We identify one such regime. The task is to forecast a k-body observable
⟨X_0…X_{k-1}⟩ of a quantum trajectory at long horizon. Our QRC injects the
input state into an (n=9 → q=11)-qubit scrambling reservoir, evolves for a
short time τ, and reads local Pauli moments in three bases (Z, X, Y) under a
split shot budget. Crucially, the reservoir maps the high-body input
observable O = U†Z_iU onto a *local* Z probe, so its shot variance scales as
~1/M independent of body count, while a classical shadow estimator of the same
k-body target pays variance 3^k/M.

At horizon h=20 for a 4-body target (k=4), the QRC is predictive
(NRMSE = 0.99 ± 0.24, just under the trivial-mean threshold of 1) whereas both
classical shadow-based (1.97 ± 0.59) and exact-feature (1.96 ± 0.61) reservoirs
fail (NRMSE > 1). The advantage is specific: it is absent at short horizons
(at k=4, classical shadows win at h=1), it is marginal even where present, and
it does NOT extend to k=6 (QRC 1.17 > 1, also failed) or k=8 (shadows win)
across 3 target Hamiltonians. The defensible claim is therefore narrow:
*there exists a long-horizon, intermediate-body regime (here k=4, h=20) where
a shot-budget-matched QRC is predictive and classical reservoirs are not.*
[NOTE: a single marginal cell may be too thin for QTML — push the regime first.]

## Key numbers (REAL 3×3, mean ± std over 9 runs, h=20)

| body k | h=20 QRC | h=20 Shadows | h=20 Cheat | bootstrap CI (pooled, gap) | clean headline? |
| ------ | ------------ | ------------ | ------------ | -------------------------- | --------------- |
| 2      | 1.339 ± 0.20 | 1.987 ± 0.43 | 1.974 ± 0.42 | —                          | NO (QRC >1)     |
| 4      | **0.986 ± 0.24** | 1.965 ± 0.59 | 1.959 ± 0.61 | +0.927 [+0.889, +0.967]    | YES (marginal)  |
| 6      | 1.172 ± 0.44 | 1.809 ± 0.97 | 1.739 ± 0.94 | +0.321 [+0.290, +0.351]    | NO (QRC >1)     |
| 8      | 1.278 ± 0.26 | 1.054 ± 0.08 | 1.006 ± 0.14 | —                          | NO (shadows win)|

⚠️ Only k=4 has QRC < 1.0 (predictive) AND both classical > 1.0 (failed). At
k=6 the bootstrap gap is positive/significant but QRC itself is 1.17 > 1 —
least-bad-among-failures, not a real win. Bootstrap CIs pool test residuals
(within-sample noise); the across-seed std (±0.24, ±0.44) is the honest
Hamiltonian-level uncertainty. Full table incl. h=1,5 in
`results/paper6_highbody_3x3/summary.json`.

## Honesty caveats to keep in the abstract / talk

- NRMSE normalized by target std → NRMSE=1 is the trivial mean predictor.
  All "win" claims are sub-1.0 for QRC AND >1.0 for classical.
- Advantage is NOT monotonic in body count: peaks k=4–6, breaks at k=8.
- Short horizons: classical wins (esp. exact-feature). QRC value is
  purely long-horizon.
- Reservoir choice matters: scrambling >> TFIM; small τ (0.3–0.6) beats
  over-scrambling.

## Mechanism one-liner (for the talk)

QRC pre-mixes high-body input correlations into local, low-variance reservoir
probes (1/M), so it sidesteps the 3^k/M shadow cost of estimating a k-body
target directly — the edge is a measurement-variance argument, not an
expressivity one.

## Open items before submission

- [ ] Replace TBD numbers with 3×3 confirmed values + bootstrap CIs
- [ ] Decide title
- [ ] Trim to the QTML word limit (check exact limit on submission page)
- [ ] Optional: one figure (NRMSE vs body count at h=20, QRC vs Shadows vs Cheat)
