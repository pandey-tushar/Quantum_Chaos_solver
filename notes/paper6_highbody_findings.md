# Paper 6 — High-Body-Target Findings (the real advantage zone)

**Date:** 2026-05-29
**Scripts:** `scripts/diag_multibasis_readout.py`, `scripts/diag_tau_sweep.py`,
`scripts/diag_highbody_target.py`

## How we got here (3 fast diagnostic rounds)

The original roadmap "Paper 6: QRC wins 12% at k=80" headline did **not** survive
scrutiny. Two artifacts:

1. **Metric artifact.** NRMSE is normalized by target std (line 87,
   `run_quantum_state_input.py`), so NRMSE=1 is the trivial mean predictor.
   The old headline had QRC at NRMSE 1.05–1.56 at every horizon — i.e. *worse
   than predicting the mean everywhere*. The "win at k=80" was QRC being the
   least-bad of several methods that had all failed (all >1.0).

2. **Readout-basis artifact.** QRC read only Z-basis Paulis (<Z_i>,<Z_iZ_j>)
   while the multipauli target spans X/Y/Z. Per-target breakdown:
   QRC on Z-targets = 0.77 (predictive, 84% <1.0); on X/Y-targets = 1.37–1.39
   (blind). The Z-only readout against X/Y/Z targets created the false negative.

## Fixes that matter

- **Multi-basis readout** (measure reservoir in Z, X, Y; split shot budget 3
  ways): removes the artifact. QRC becomes genuinely predictive across targets.
- **Scrambling (SYK-like) reservoir >> TFIM** on this task. TFIM is the wrong
  reservoir (NRMSE 1.25–1.84). Scram gives 0.82–1.0.
- **Small tau (0.3–0.6) beats large tau.** Over-scrambling (tau>=2) hurts.

After fixes, at n_in=9 q=11 on the 1-/2-body multipauli target: QRC is
*competitive and horizon-stable* but has no clean advantage (classical
Shadows-ESN wins short horizons). That alone = the "competitive + stable"
story, weaker than hoped.

## The real advantage zone: high-body targets at long horizon

`diag_highbody_target.py`, n_in=9, q=11, scram, tau=0.6, M=11264/step,
3 reservoir seeds, target = <X_0..X_{k-1}> at horizon t+h.

| body k | h=1 (QRC/Shadows/Cheat) | h=5 | **h=20** |
|--------|-------------------------|-----|----------|
| 2 | 1.09 / 0.56 / 0.51 | 1.19 / 1.01 / 0.98 | 1.23 / **2.45** / 2.48 |
| 4 | 0.55 / 0.41 / 0.26 | 0.62 / 0.61 / 0.55 | **0.72 / 1.75** / 1.77 |
| 6 | 0.74 / 0.83 / 0.28 | 0.77 / 0.80 / 0.49 | **0.81 / 1.23** / 1.17 |
| 8 | 1.10 / 0.92 / 0.17 | 1.12 / 0.81 / 0.32 | 1.08 / 0.91 / 0.87 |

(NRMSE, lower=better, <1.0=predictive. Mean over 3 seeds.)

### Headline (honest, defensible)

**At long forecast horizon (k=20), a scrambling QRC predicts intermediate-body
(4–6 body) observables of a quantum trajectory that BOTH classical baselines
fail at (NRMSE>1).** At k=6/h=20: QRC 0.81 (predictive) vs shadows 1.23 vs
exact-feature cheat 1.17 (both failed). QRC's error is markedly horizon-stable
(k=6: 0.74 -> 0.81 over h=1 -> 20) while classical degrades past the mean
predictor.

### Mechanism (matches the structural prediction)

QRC measures O_i = U_dag Z_i U of the injected input. This is a *local Z*
measurement on the reservoir, so its shot variance is ~1/M **independent of
how high-body O_i is in the input frame**. Classical shadows pay 3^k/M to
estimate a k-body target directly; exact-feature classical has clean features
but cannot extrapolate 20 steps. QRC's reservoir pre-mixes high-body input
correlations into local, low-variance probes.

### The honest nuance (must be in the paper)

- **Not monotonic in body count.** Edge peaks at k=4–6, **breaks at k=8**
  (near-global observable; tau=0.6 scrambling likely doesn't reach 8-body
  support, so shadows win). Claim must be "intermediate-body," not "all k."
- **Short horizons: classical wins** (esp. exact-feature cheat, 0.17–0.28).
  QRC's value is purely long-horizon.
- So the precise claim is: *long-horizon forecasting of intermediate-body
  quantum-trajectory observables.*

## 3×3 RESULT (2026-05-31) — `scripts/diag_highbody_targetseeds.py`

3 target seeds {42,7,123} × 3 reservoir seeds {0,1,2} = 9 runs.
Source: `results/paper6_highbody_3x3/summary.json` (real numbers).

⚠️ **The 1-seed numbers did NOT hold up at 3×3.** Seed 42 was favorable;
seeds 7 and 123 are much worse, and the across-seed std is large.

| body k | h=20 QRC | h=20 Shadows | h=20 Cheat | QRC predictive (<1)? |
|--------|----------|--------------|------------|----------------------|
| 2 | 1.339±.20 | 1.987±.43 | 1.974±.42 | NO (>1) |
| 4 | **0.986±.24** | 1.965±.59 | 1.959±.61 | yes, but MARGINAL |
| 6 | 1.172±.44 | 1.809±.97 | 1.739±.94 | NO (>1) |
| 8 | 1.278±.26 | 1.054±.08 | 1.006±.14 | shadows win |

For context, shorter horizons (k=4): h=1 QRC 0.991 vs Shadows 0.803 (shadows
win); h=5 QRC 1.002 vs Shadows 1.062 (~tie, both near 1).

**Paired bootstrap CIs (N=2000), gap = NRMSE(Shadows) − NRMSE(QRC):**
- k=4 h=20: +0.927, 95% CI [+0.889, +0.967], P(QRC better)=1.000
- k=6 h=20: +0.321, 95% CI [+0.290, +0.351], P(QRC better)=1.000

⚠️ **Bootstrap caveat:** these CIs pool test-point residuals across runs, so
they capture within-sample noise, NOT across-seed/Hamiltonian variance. The
across-seed std (±0.24 at k=4, ±0.44 at k=6) is the honest uncertainty for
"does this hold across Hamiltonians." A bootstrap that resamples whole seeds
would give much wider CIs.

### Honest verdict after 3×3

- **Only k=4 at h=20 is a clean (if marginal) headline:** QRC 0.986 is
  predictive (just under 1), BOTH classical baselines fail (1.96), and the
  gap is large. But QRC=0.986 is barely sub-1.0 — not a comfortable margin.
- **k=6 does NOT survive.** QRC=1.172 > 1.0 → worse than the trivial mean
  predictor. The "QRC beats shadows" bootstrap is real but it is the
  least-bad-among-failures trap (exactly what we warned about). k=6 looked
  good ONLY on seed 42.
- **k=2, k=8 are not headlines** (QRC >1 or shadows win).

This substantially weakens the QTML story relative to the 1-seed handoff. The
defensible claim now is narrow: *at h=20, k=4, a scrambling QRC is (marginally)
predictive of a 4-body trajectory observable where shot-budget-matched
classical shadow and exact-feature reservoirs both fail.* Whether this single
marginal cell is enough for a QTML abstract is a judgment call — see options
in the roadmap.

## Still outstanding before any QTML claim

1. **Decide if k=4-only is enough.** It is one marginal cell. Options:
   (a) push the regime to widen the win (tau sweep, longer trajectory for
   lower-variance estimates, different target observable family); (b) reframe
   honestly as "a narrow but real advantage zone exists"; (c) shelve the QTML
   abstract.
2. **Seed-level bootstrap** (resample whole seeds) for honest CIs.
3. **Tau sweep at k=4 and k=6** — does a different tau make k=6 genuinely
   predictive (<1) across all 3 seeds, or push k=4 to a safer margin?
4. **Understand the k=8 break** (likely tau=0.6 scrambling doesn't reach
   8-body support).

## QTML logistics

Stellenbosch, South Africa, Dec 6–11 2026. **Abstract deadline 30 June 2026.**
Abstract-based, non-archival (no conflict with QST paper 1). Cite the arXiv
preprint (2604.23743) for the QRC architecture rather than re-deriving.
