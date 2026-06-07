# Scientific Plan: 2-Point-Correlator QRC for Multivariate Nonlinear Volatility Spillover

**Date:** 2026 (current session)
**Script (to extend):** `scripts/mv_spillover_qrc.py`
**Builds on:** Bayat et al. 2505.13933 (recurrent TFIM QRC, QR2 ensemble) +
Vercellino et al. 2606.04686 (1- and 2-point ⟨Z_iZ_k⟩ correlator readout).
**Status:** PLAN ONLY — no code until plan is approved. No commits until a
hash-verified result exists.

---

## 0. Motivation & what went wrong before

Two prior negatives, each with an identified cause:
1. **Feature-starved readout.** Earlier MV-spillover QRC read only ⟨Z_i⟩
   (q features). Both reference papers independently report ⟨Z_i⟩-only is
   insufficient and that adding ⟨Z_iZ_k⟩ two-point correlators is what gives
   expressivity. → FIX: add 2-point correlators (Vercellino Eq. 3).
2. **Linearly-predictable data.** The threshold-ReLU spillover, once
   triggered, drove assets into ~lockstep (corr 0.96) — a regime that LINEAR
   VAR forecasts optimally, so no nonlinear method could win. → FIX: data with
   a tunable *bilinear* (genuinely non-linearly-decodable) spillover term.

## 1. Central question (well-posed, falsifiable)

> At matched resources, does a quantum reservoir's measured **1- AND 2-point
> Pauli-Z correlators** forecast multivariate volatility with **nonlinear
> cross-asset spillover** better than — or merely comparably to — strong
> classical baselines, *including a classical model with the same quadratic
> feature capacity*?

Decisive sub-question (this is what makes the result interpretable):
> Does QRC beat **degree-2 polynomial ridge** on the same windowed inputs?
> Poly-2 ridge has explicit access to all pairwise products xᵢxⱼ — the
> classical analog of ⟨Z_iZ_k⟩. If QRC only ties poly-2, the honest verdict is
> "quantum reservoir = classical quadratic features, no advantage." If QRC
> beats poly-2 at matched/disfavored feature budget, that is a real (if small)
> signal worth chasing.

## 2. Hypotheses & pre-registered predictions

- **H1 (readout).** Adding ⟨Z_iZ_k⟩ to ⟨Z_i⟩ lowers QRC NRMSE materially
  (replicates Vercellino). Prediction: ≥15% relative NRMSE drop at h=1.
- **H2 (ensemble).** QR2 (τ and τ/2 concatenated) beats QR1 (replicates Bayat).
  Prediction: small but consistent improvement.
- **H3 (advantage axis).** Any QRC edge over the *best LINEAR* baseline grows
  with bilinear-spillover strength β (not with linear strength γ).
- **H4 (the hard test).** QRC vs degree-2 poly ridge. Three possible outcomes,
  all reportable:
  - (a) QRC < poly-2 across seeds, bootstrap CI on gap excludes 0, edge grows
        with β → genuine (small) quantum-reservoir advantage.
  - (b) QRC ≈ poly-2 (within seed std) → honest tie; quantum reservoir
        reproduces classical quadratic features, no advantage, mechanism
        validated.
  - (c) QRC > poly-2 → honest negative.
- **H5 (shot robustness).** Under M = q·1024 shots, any QR1/QR2 conclusion at
  h=1 survives (NRMSE shift < 1 across-seed std).

## 3. Data: tunable linear + bilinear spillover (bounded)

Latent zero-mean log-vol, mean-reverting, with TWO spillover knobs:

    h_{i,t} = φ h_{i,t-1}
              + γ  Σ_j  S_ij · g(h_{j,t-1})              # LINEAR spillover (VAR captures)
              + β  Σ_{(j,k)} T_ijk · g(h_j) g(h_k)        # BILINEAR (only quadratic+ captures)
              + noise · ε

  g = tanh (bounded → no divergence; lesson from prior overflow bug).
  S row-stochastic; T a sparse, fixed, zero-diagonal 3-tensor of bilinear
  couplings. φ<1 for stationarity. RV target = h (standardized log-vol).

  Knobs: γ (linear), β (bilinear = the advantage axis). Sweep β ∈ {0, 0.5, 1,
  2} at fixed modest γ. The point: VAR's achievable NRMSE should be ~flat in β
  (it can't use the bilinear term), while quadratic-capable methods (poly-2,
  QRC-with-ZZ) improve as β rises.

## 4. Methods

### 4.1 QRC (extend existing, Bayat-faithful)
- Reservoir H = Σ_ij J_ij X_iX_j + v Σ_i Z_i, J~U[0,1], v=1, τ=1, fixed/seed.
- N input + n2 memory qubits, q = N+n2 ≤ 11 (use N=5, n2=3 → q=8).
- Recurrence: memory depth k=3, partial-trace input qubits each step (already
  implemented & self-test-passing).
- **NEW readout tiers (ablation):**
  - `Z`     : ⟨Z_i⟩  (q features) — the old, starved readout.
  - `ZZ`    : ⟨Z_i⟩ + ⟨Z_iZ_k⟩  (q + q(q-1)/2 features; q=8 → 36).
  - `ZZ+QR2`: above, computed for τ AND τ/2, concatenated (Bayat QR2; 72 feat).
- Optional shot noise: ⟨Z_i⟩ var (1-⟨Z⟩²)/M; ⟨Z_iZ_k⟩ var (1-⟨Z_iZ_k⟩²)/M,
  M=q·1024. (Exact statevector default, matching Bayat.)
- Readout: ridge (α tuned on a validation slice, not test).

### 4.2 Classical baselines (tiered by capacity — the scientific spine)
LINEAR tier (β-blind, the floor to beat):
- `HAR_per_asset` — classical HAR lags per asset, no cross terms.
- `VAR_joint` — ridge on windowed joint vector (captures all LINEAR spillover).
NONLINEAR tier (the real competitors):
- `Poly2_ridge` — **degree-2 polynomial features on the windowed joint input**
  (explicit xᵢxⱼ) + ridge. THE decisive control (matched quadratic capacity).
- `ESN_joint` — echo-state network on the joint vector.
- `RFF_joint` — random Fourier features at matched dim.

Fair-budget note: QRC-ZZ has 36 feats (q=8); QR2 72. Poly-2 on window=3×N=5
=15 inputs → 15 + C(15,2)+15 ≈ 135 feats. So the classical quadratic control
has MORE features than QRC → if QRC ties/wins it is NOT a feature-count
artifact. State this explicitly in any writeup.

## 5. Correctness gates (RUN FIRST; nothing trusted until all pass)

Given this session's history of fabricated outputs, every result is gated:
- **G1 data finite & nonlinear:** no NaN/overflow at all β; linear-R² of target
  on linear features DROPS as β rises while full (nonlinear) R² stays high
  (confirms β injects genuinely nonlinear structure).
- **G2 reservoir physical:** Tr(ρ)=1 after each step; ⟨Z_i⟩, ⟨Z_iZ_k⟩ ∈ [-1,1].
- **G3 2-point correctness:** ⟨Z_iZ_k⟩ from the density matrix == exact value
  from the same statevector via independent computation, to 1e-10.
- **G4 shot→exact:** mean|feature_shot − feature_exact| → 0 as M→∞ (monotone).
- **G5 QR2 distinctness:** QR2 feature dim = 2× QR1 and τ/2 block differs from τ.
- **G6 I/O integrity:** every reported number read back from on-disk JSON with a
  SHA check; never quote a number only seen in stdout.

## 6. Metrics & statistics

- Per-asset NRMSE (normalized by per-asset target std), mean over assets.
  NRMSE=1 ⇒ trivial mean predictor. **Every "win" must be sub-1.0.**
- Horizons h ∈ {1, 5, 20} (one-step is the cleanest; long-horizon secondary).
- Seeds: 3 data × 3 reservoir = 9 runs (standing rule). Report mean ± across-
  seed std (the honest uncertainty).
- **Paired bootstrap** over the pooled test set for the QRC−best-classical gap
  AND a **seed-level** bootstrap (resample whole seeds) — report both; the
  seed-level CI is the conservative one.
- Ridge α chosen on a validation split, never on test (no leakage).

## 7. Experimental matrix (phased, each phase gated by the previous)

- **Phase 0 — gates.** Implement & pass G1–G6. STOP if any fail.
- **Phase 1 — readout ablation** (1 data×1 res seed, β=1, h=1):
  Z vs ZZ vs ZZ+QR2. Confirms H1, H2. Fast (~minutes). Decision: proceed only
  if ZZ materially beats Z (else the whole premise is dead).
- **Phase 2 — tiered baselines** (β=1, 3×3 seeds, h∈{1,5,20}):
  QRC-ZZ+QR2 vs all 5 classical baselines. The headline comparison, esp. vs
  Poly2_ridge (H4).
- **Phase 3 — advantage axis** (β ∈ {0,0.5,1,2}, 3×3 seeds, h=1):
  Does QRC−VAR gap grow with β (H3)? Does QRC−Poly2 gap behave?
- **Phase 4 — shot robustness** (best β, M=q·1024, 3×3 seeds, h=1): H5.
- **Phase 5 — synthesis:** NRMSE-vs-β figure (QRC, Poly2, VAR, ESN), gap table
  with seed-level bootstrap CIs, honest verdict per H4 (a/b/c).

## 8. Pre-registered decision criteria

| Outcome | Condition | Action |
|---|---|---|
| Strong | QRC<1 & QRC<Poly2 across seeds, seed-level CI on gap excludes 0, gap grows with β | Write it up as a real edge; scale-up plan |
| Tie (likely) | QRC ≈ Poly2 within seed std, both beat VAR & sub-1 | Honest "QRC ≡ classical quadratic features; mechanism validated, no advantage" |
| Negative | QRC ≥ best classical, or QRC>1 where classical<1 | Honest negative; QRC not competitive here |

All three are publishable as an honest NISQ-scale benchmark — both reference
papers themselves report QRC NOT cleanly beating strong classical baselines, so
a rigorous fair comparison (with the Poly2 control nobody else runs) is the
contribution regardless of which way it lands.

## 9. Risks & mitigations

- **R1 tanh saturation in encoding** (prior bug): standardize per-asset on TRAIN
  stats, tanh(scale·z), scale tuned so feature std is healthy. Gate G2-adjacent
  check on feature variance > threshold.
- **R2 QRC just replicates Poly2:** that's not a failure, it's H4(b) — a clean,
  honest result. Foregrounded, not hidden.
- **R3 q=8 density-matrix cost** with 2-point readout: 256×256, trivial; QR2
  doubles evolutions, still seconds/seed. q≤11 hard cap respected.
- **R4 multiple-comparisons** across β×h×method: pre-register the headline cell
  (β=1, h=1, QRC-ZZ+QR2 vs Poly2) before looking; treat the rest as secondary.
- **R5 session I/O corruption:** G6 hash-gate on every number; re-verify in a
  clean session before any commit or writeup.

## 10. Deliverables

- Extended `scripts/mv_spillover_qrc.py` with `--readout {Z,ZZ,ZZ_QR2}`,
  `--beta`, `Poly2_ridge` baseline, and a `--self-test` covering G1–G6.
- `results/mv_correlator/` JSONs (hash-logged).
- One figure (NRMSE vs β, methods overlaid) + one gap table with seed-level CIs.
- A short honest findings note stating which of H4(a/b/c) occurred.
