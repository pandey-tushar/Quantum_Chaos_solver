# Paper outline: "Where Quantum Reservoir Computing Doesn't Help (Yet):
# A Fair-Comparison Cartography at NISQ Scale"

**Status:** outline / proposal. Working title; venue TBD.
**Branch:** paper-6-hilbert-input (shares the harness commits).
**Date:** current session.

## The one-sentence thesis

Across five structurally distinct mechanisms by which quantum reservoir
computing (QRC) is claimed to gain an advantage, a *properly tuned, matched-
budget classical baseline* captures what the quantum reservoir offers at NISQ
scale (q <= 11). The contribution is the fair-comparison methodology — and the
negative map it produces.

## Why this is publishable (not just "we failed")

1. The QRC-finance / QRC-time-series literature is full of "quantum beats
   classical" claims that rest on UNDERTUNED or MISMATCHED classical baselines
   (e.g. the baselines not on equal footing; efficiency reported as simulation
   not on equal footing, efficiency = simulation artifact). A disciplined null
   result with the controls those papers skip is a real service to the field.
2. We introduce two methodological controls almost nobody runs:
   - the **matched-quadratic-capacity control** (Poly2 / concat ablation), and
   - the **same-tuning-budget recurrent control** (ESN tuned exactly as hard
     as the QRC).
3. The result is not "QRC is bad" — it is a precise MAP of where the quantum
   part adds nothing once the comparison is fair, plus the one place a real
   (if mechanism-not-advantage) effect appears (feedback rescues open-loop).

## The five closed angles (the evidence base, all gated + hash-verified)

| # | Mechanism | Fair classical control | Verdict |
|---|-----------|------------------------|---------|
| 1 | High-body trajectory observables (long horizon) | shadows + matched readout | marginal, non-robust across seeds |
| 2 | Channel-spectrum learning (2-copy/Choi) | single-copy shadow floor | no clean win at q<=11 (4^m rank wall) |
| 3 | Cross-asset volatility spillover (multivariate) | linear VAR | QRC loses; coupling helps VAR not QRC |
| 4 | 2-point-correlator QRC on quadratic dynamics | Poly2 (matched quadratic) + concat | quantum part redundant; concat ~ Poly2 |
| 5 | Feedback-driven QRC on nonstationary data | tuned ESN (same budget) | FB live (beats open-loop) but loses ESN -14% |

Common thread: every "QRC helps" signal collapses against a matched-capacity
classical model. Where a quantum mechanism IS live (feedback, angle 5), it
reproduces classical recurrent memory rather than exceeding it.

## Proposed structure

1. **Intro.** The reproducibility/fair-baseline gap in QRC claims. State the
   thesis and the two controls.
2. **Methods — the fair-comparison harness.** Hardware-realistic shot model;
   matched feature dim; validation-only tuning for ALL methods incl. classical;
   correctness gates (physical reservoir, causal features, shot->exact,
   feedback-wiring); hash-verified I/O. This section is the reusable contribution.
3. **Five case studies** (the table above), each: mechanism, the matched
   control, result, why it closes.
4. **The one live mechanism.** Feedback rescues open-loop QRC (1.06 -> 0.76)
   but a same-budget ESN still wins — "quantum recurrence = classical recurrence."
5. **Discussion.** What would have to change for an advantage (larger q beyond
   sim reach; genuinely Hilbert-space inputs; magic/non-Clifford resources);
   why small-q sim studies systematically over-claim.
6. **Appendix.** The earlier dequantization material (paper-5) as the
   scalar-input special case.

## What's still needed before writing

- Decide scope: all five angles, or the 2-3 cleanest (4 & 5 are the tightest).
- Re-run each headline cell in ONE clean session (this session had tool-output
  corruption; everything committed is hash-verified from disk, but a clean
  re-verification pass before submission is prudent).
- Honest framing review: is this a methods/comment paper (e.g. to a venue that
  takes negative results / reproducibility studies) or a short letter?

## Caveat to keep central

This is a NISQ-scale (q<=11), simulation-based map. The honest claim is bounded:
"at these sizes, with fair baselines, no advantage" — NOT "QRC can never help."
That boundedness is a strength if stated plainly; it's what separates this from
the over-claiming it critiques.
