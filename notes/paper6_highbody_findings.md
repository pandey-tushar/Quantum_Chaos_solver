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

## CONFIRMED at 3×3 (2026-05-31) — `scripts/diag_highbody_targetseeds.py`

3 target seeds {42,7,123} × 3 reservoir seeds {0,1,2} = 9 runs.
Source: `results/paper6_highbody_3x3/summary.json`.

| body k | h=20 QRC | h=20 Shadows | h=20 Cheat | verdict |
|--------|----------|--------------|------------|---------|
| 2 | 1.156±.10 | 2.211±.36 | 2.226±.36 | QRC wins but NOT predictive (>1) |
| 4 | **0.730±.10** | 1.706±.28 | 1.703±.27 | QRC predictive, classical failed |
| 6 | **0.812±.07** | 1.228±.12 | 1.156±.16 | QRC predictive, classical failed |
| 8 | 1.063±.10 | 0.918±.08 | 0.872±.08 | breaks (classical wins) |

**Paired bootstrap CIs (N=2000) on the headline cells — both significant:**
- k=4 h=20: gap (Shadows−QRC) = +0.974, 95% CI [+0.732, +1.232], P(QRC better)=1.000
- k=6 h=20: gap = +0.413, 95% CI [+0.314, +0.522], P(QRC better)=1.000

The 3×3 means match the original 1-seed numbers (k4: 0.72→0.730, k6: 0.81→0.812)
→ robust. Standing-rule 3×3 requirement satisfied; bootstrap confirms the
QRC-vs-Shadows gap excludes zero at both headline cells.

## Still outstanding before the QTML abstract

1. **Understand / push the k=8 break.** Does larger tau extend the win to
   higher body count? (q capped at 11; tau sweep at k=8 is cheap.) — IN PROGRESS
2. **One more reservoir family** (different scram seed distribution) for
   robustness of the "scram works, TFIM doesn't" claim. (optional)
3. **Write + trim the QTML abstract** (`notes/qtml2026_abstract_draft.md`,
   numbers already filled). Cite arXiv:2604.23743.

## QTML logistics

Stellenbosch, South Africa, Dec 6–11 2026. **Abstract deadline 30 June 2026.**
Abstract-based, non-archival (no conflict with QST paper 1). Cite the arXiv
preprint (2604.23743) for the QRC architecture rather than re-deriving.
