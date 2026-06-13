# Design Plan: Feedback-Driven Quantum Reservoir Computing (FB-QRC)

**Date:** current session
**Motivated by:** Kobayashi et al., "Feedback-driven quantum reservoir
computing for time-series analysis," PRX Quantum 5:040325 (2024).
**Status:** DESIGN ONLY. No code until approved. No commits until a
hash-verified, gate-passed result exists. Restore point if it fails:
`git checkout phase3-concat-negative`.

---

## 0. The honest framing (read this first)

Every QRC we have tested is **fixed, open-loop**: input -> encode -> evolve ->
measure -> readout. Its feature map is a fixed (if complex) function of the
input window, which is exactly why a degree-2 polynomial (Poly2) or an ESN on
the windowed input matched or beat it every time.

Feedback-driven QRC closes the loop: the measurement outcome at step t
modulates the reservoir drive at step t+1. This makes the effective dynamics
**path-dependent** — the reservoir's response to x_t depends nonlinearly on its
own output history, not just a fixed window of inputs.

**Critical caveat that sets the bar:** an ESN is ALSO a recurrent, closed-loop
system with unbounded nonlinear memory. So path-dependence alone does NOT
distinguish FB-QRC from a classical ESN — only from fixed-window models (Poly2,
windowed ridge). The genuine test is therefore:

> **Does FB-QRC beat a well-tuned ESN** (both recurrent), not merely Poly2?

Our own literature review found QRC does not beat a well-tuned ESN. So the
honest prior is that FB-QRC most likely TIES the ESN. The value of this
experiment is to test that fairly, and — if it ties — to characterize *where*
the quantum feedback loop differs from a classical one, if anywhere. We set the
bar at the ESN and pre-register the likely-tie outcome so we are not fooled by a
QRC-beats-Poly2 result that an ESN would also achieve.

## 1. Central question (falsifiable)

> At matched resources and with honest causal evaluation, does a feedback-driven
> quantum reservoir outperform a well-tuned classical ESN (and Poly2, and
> open-loop QRC) on a NONSTATIONARY / regime-switching forecasting task — the
> regime where Kobayashi et al. claim feedback specifically helps?

Decisive sub-questions:
- (a) FB-QRC vs **open-loop QRC** (same reservoir, feedback off): does closing
  the loop help at all? (If not, the mechanism is inert — stop.)
- (b) FB-QRC vs **ESN** (the real bar): does the *quantum* feedback loop beat a
  *classical* feedback loop at matched feature dim?
- (c) FB-QRC vs **Poly2 / windowed ridge** (fixed-window control): confirms
  whether recurrence matters at all on this task.

## 2. Hypotheses & pre-registered predictions

- **H1 (mechanism live):** FB-QRC differs from open-loop QRC — closed-loop
  features are not a fixed function of the input window. Prediction: ≥10%
  relative NRMSE improvement of FB over open-loop on the nonstationary task.
- **H2 (the real test):** FB-QRC vs ESN. Pre-registered prediction: **TIE**
  (within seed std). A clean win would be surprising and would need the gate
  checks to rule out artifacts.
- **H3 (nonstationarity is the axis):** any FB-QRC advantage over fixed-window
  models grows with regime-switching rate; on stationary data FB≈open-loop.
- **H4 (honest null is informative):** if FB-QRC ties ESN and both beat Poly2,
  the result is "quantum feedback reproduces classical recurrent memory, no
  quantum advantage" — publishable as part of the cartography.

## 3. Task & data: nonstationary, regime-switching dynamics

Feedback should help most when the *generating process itself switches* and the
model must track the switch online. Design:

- **Driven nonstationary map.** A 1-D (or low-D) nonlinear map whose parameter
  switches between regimes on a hidden Markov schedule:
    x_{t+1} = f_{s_t}(x_t) + noise,   s_t in {A, B} (persistent HMM)
  e.g. two logistic/Henon parameter sets (chaotic vs periodic), switching with
  probability p_switch. The model never sees s_t — it must infer regime from
  the observable history. This is exactly where an adaptive feedback loop can
  pay off over a fixed-window map.
- **Stationary control:** same map with p_switch=0 (single regime). H3 predicts
  FB advantage vanishes here.
- Knob: p_switch in {0, 0.005, 0.02, 0.05} (advantage axis = switching rate).
- Bounded, standardized, train-only normalization (no leakage — the S&P lesson
  from the confidential note applies to US too).

## 4. Methods (all matched: same data, same readout, causal split)

### 4.1 FB-QRC (the proposed method)
- Reservoir: small (q=4-6) fixed Hamiltonian (TFIM or the Bayat XX+Z), tau tuned
  on validation.
- Open-loop step: encode x_t via RY, evolve U, measure <Z_i> (and optionally
  <Z_iZ_k>).
- **Feedback:** the previous measurement vector m_{t-1} modulates the current
  drive — concretely an extra input-qubit rotation angle
    phi_t = x_t + k_fb * g(m_{t-1})
  where g is a fixed readout-to-angle map and k_fb is the feedback gain (tuned
  on validation; k_fb=0 recovers open-loop). This is the discrete analog of
  Kobayashi's measurement feedback.
- Readout: windowed ridge on the measurement features, alpha tuned on validation.

### 4.2 Baselines (matched feature dim & window)
- **open-loop QRC** (k_fb=0): isolates the feedback contribution (H1).
- **ESN** (the real bar, H2): classical recurrent reservoir, tuned spectral
  radius / leak / input scaling on validation, matched feature dim.
- **Poly2** and **windowed ridge** (fixed-window controls, H3/c).
- **Linear** (floor).

### 4.3 Fairness rules (preempt the failure modes we already hit)
- All hyperparameters (tau, k_fb, alpha, ESN sr/leak) tuned on a validation
  slice, NEVER test. ESN gets the SAME tuning budget as FB-QRC (no strawman ESN
  — that is the mistake the a hybrid QRC architecture paper makes and we will not repeat it).
- Matched effective feature dim between FB-QRC and ESN.
- Causal, train-only normalization. Walk-forward or fixed chronological split.
- 3 data seeds x 3 reservoir seeds for any headline.

## 5. Correctness gates (--self-test; run FIRST, nothing trusted until pass)

- **G1 data:** regime-switching series is bounded, finite; regime is genuinely
  hidden (single-window linear R2 drops as p_switch rises -> nonstationarity is
  real and fixed-window models should struggle).
- **G2 reservoir physical:** Tr(rho)=1; <Z>, <ZZ> in [-1,1] across the run.
- **G3 feedback is live & correct:** with k_fb=0 the FB-QRC feature matrix
  EXACTLY equals the open-loop feature matrix (to 1e-12); with k_fb>0 it
  differs. (Proves the feedback path is wired correctly and is a genuine
  modification, not a no-op or a bug.)
- **G4 feedback is causal:** feature at time t depends only on inputs/measurements
  up to t (shifting a future input leaves m_t unchanged) — rules out leakage
  through the loop.
- **G5 shot->exact:** measurement features converge to exact as shots grow
  (carried over from the existing harness).
- **G6 I/O integrity:** every reported number read back from on-disk JSON with a
  SHA check (no trusting stdout — session history demands this).

## 6. Metrics & statistics
- NRMSE (amplitude) AND MDA with a persistence baseline (direction) — both, per
  the confidential-note lesson that MDA without persistence is meaningless.
- Horizons {1, 5}. Seeds 3x3. Mean +/- across-seed std; paired bootstrap (pooled
  AND seed-level) for the FB-vs-ESN and FB-vs-open-loop gaps.

## 7. Phased execution (each gated by the previous; PROBE before any sweep)

- **Phase 0 — gates.** Implement data + FB reservoir + readout; pass G1-G6.
  STOP if any fail. Commit as restore point (like phase0-gates-pass).
- **Phase 1 — mechanism check (cheap, 1 seed):** FB-QRC vs open-loop QRC on the
  nonstationary task. Go/no-go: proceed only if FB beats open-loop by >=10%
  (H1). If feedback is inert, the mechanism is dead — stop and report.
- **Phase 2 — the real test (3x3 seeds):** FB-QRC vs ESN vs Poly2 vs open-loop,
  NRMSE + MDA, at the nonstationary regime.
- **Phase 3 — advantage axis:** sweep p_switch {0, 0.005, 0.02, 0.05}; does any
  FB-vs-fixed-window edge grow with switching rate (H3)? Is FB-vs-ESN flat (H2)?
- **Always:** probe per-cell time before each sweep; ask before any long run.

## 8. Pre-registered decision criteria

| Outcome | Condition | Verdict |
|---|---|---|
| Mechanism dead | FB ~ open-loop (H1 fails) | feedback inert here; stop, report |
| Honest tie (likely) | FB ~ ESN, both > Poly2, both predictive | quantum feedback = classical recurrent memory; no advantage |
| Real edge (surprising) | FB < ESN across seeds, seed-level CI excludes 0, grows with p_switch | genuine FB-QRC advantage; scale-up plan + re-verify clean session |
| Negative | FB >= ESN or FB not predictive | honest negative |

## 9. Risks & mitigations
- **R1 ESN strawman:** give ESN the same validation tuning budget as FB-QRC.
  This is the single most important fairness control — most "QRC wins" papers
  fail here.
- **R2 feedback instability:** large k_fb can blow up the loop; bound g() (use a
  saturating map) and tune k_fb on validation only. Gate G2 catches divergence.
- **R3 nonstationarity too weak/too strong:** calibrate p_switch so fixed-window
  R2 drops meaningfully but the task stays learnable (like the Henon gain
  calibration we did for Phase 0 of the last study).
- **R4 session I/O corruption:** G6 hash-gate on every number; re-verify in a
  clean session before any commit/writeup.
- **R5 multiple comparisons:** pre-register the headline cell (nonstationary,
  h=1, FB-QRC vs ESN) before looking; rest is secondary.

## 10. Deliverables
- `scripts/feedback_qrc.py` with `--self-test` (G1-G6), `--phase1`, `--phase2`,
  `--phase3`, hash-logged JSON in `results/feedback_qrc/`.
- Honest findings note stating which decision-criteria outcome occurred.

---

### Why this is worth one more shot (and why I'm not over-promising)

It is the ONLY untried mechanism with a specific, published claim
(Kobayashi PRX Quantum 2024) that the *dynamics class* differs from fixed
reservoirs — and path-dependence is genuinely something Poly2 cannot fake. But
the ESN can, so I expect a tie with ESN. If it ties, that is a clean, honest
contribution to the "where QRC does/doesn't help" cartography, not a wasted run.
If it wins, it is the first real signal we've found and we verify it hard.
