# QTML 2026 Abstract — Draft

**Venue:** QTML 2026, Stellenbosch, South Africa, Dec 6–11.
**Deadline:** 30 June 2026. Abstract-based, non-archival (no QST conflict).
**Architecture citation:** arXiv:2604.23743 (cite, do not re-derive).
**Status:** Numbers CONFIRMED at 3×3 (3 target × 3 reservoir seeds) +
bootstrap CIs. Source: `results/paper6_highbody_3x3/summary.json`.

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

At horizon h=20, the QRC predicts intermediate-body targets (k=4–6) with
NRMSE < 1 (predictive), whereas both classical shadow-based and exact-feature
reservoirs exceed NRMSE 1 (worse than the mean predictor). At k=6, h=20: QRC
NRMSE = 0.81 ± 0.07 vs shadows 1.23 ± 0.12; at k=4, h=20: QRC 0.73 ± 0.10 vs
shadows 1.71 ± 0.28. Paired bootstrap 95% CIs on the Shadows−QRC gap exclude
zero in both cells (k=4: [+0.73, +1.23]; k=6: [+0.31, +0.52]; P(QRC better)=1.0).
The advantage is specific: it is absent at short horizons and breaks at
near-global body count (k=8), so the claim is precisely long-horizon
forecasting of intermediate-body observables. We confirm robustness across 3
target Hamiltonians × 3 reservoir instances (9 runs).

## Key numbers (CONFIRMED 3×3, mean ± std over 9 runs)

| body k | h=20 QRC | h=20 Shadows | h=20 Cheat | bootstrap 95% CI on Shadows−QRC gap |
| ------ | ------------ | ------------ | ------------ | ----------------------------------- |
| 2      | 1.156 ± 0.10 | 2.211 ± 0.36 | 2.226 ± 0.36 | (QRC "wins" but NOT predictive, >1.0)|
| 4      | **0.730 ± 0.10** | 1.706 ± 0.28 | 1.703 ± 0.27 | +0.974 [+0.732, +1.232], P=1.000   |
| 6      | **0.812 ± 0.07** | 1.228 ± 0.12 | 1.156 ± 0.16 | +0.413 [+0.314, +0.522], P=1.000   |
| 8      | 1.063 ± 0.10 | 0.918 ± 0.08 | 0.872 ± 0.08 | (break — classical wins)            |

Headline cells = k=4 and k=6 at h=20: QRC predictive (<1.0), BOTH classical
baselines failed (>1.0), bootstrap CI on the gap excludes zero. Short horizons
(h=1,5) favor classical (esp. exact-feature Cheat). Full table with h=1,5 in
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
