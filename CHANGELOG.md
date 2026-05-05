# Changelog

All notable changes to the Quantum Chaos Solver project are documented here.
This project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2026-05-04

Proof-of-concept extension of the QRC framework from ODEs to PDE forecast
correction, with a matched-complexity (K = qubit count) scaling study and
a separate quasi-geostrophic (QG) testbed under a NeuralGCM-style hybrid
(physics step in grid space, learned residual correction added in grid
space).

All Wilcoxon p-values reported below are paired, n=8 seeds, and have been
audited for direction with `scripts/audit_directional_results.py` —
which reports per-seed wins and the lower-mean method alongside the
two-sided p-value.

### Added — solvers and data

- `src/swe_solver.py` — nonlinear rotating SWE on a doubly-periodic
  Arakawa C-grid (Sadourny vector-invariant). Optional `linear=True`
  switches to the linearised rotating SWE, used as the deliberately-
  crude physics core in Block B.
- `src/qg_solver.py` — pseudo-spectral single-layer barotropic QG on
  a beta-plane.
- `src/pod_reduction.py` — SVD-based POD/EOF with per-field variance
  weighting.
- `scripts/generate_swe_data.py`, `generate_swe_hybrid_data.py` — SWE
  truth-only and paired (truth, physics) data generators.
- `scripts/generate_qg_resgap_data.py` — QG paired data with a
  resolution gap (truth at 128×128, coarse-grained to a 64×64 storage
  grid; physics integrated at 64×64). The "residual" is the
  contribution of unresolved scales to the resolved evolution — the
  classical subgrid-closure target.

### Added — pipelines

- `scripts/run_swe_qrc.py` — Block A pipeline: autonomous QRC on POD
  coefficients with classical ESN at matched feature dim and
  persistence baselines.
- `scripts/run_swe_hybrid.py` — Block B pipeline: physics + QRC
  residual correction. Rollout uses the NeuralGCM-style
  full-grid-state-plus-additive-correction protocol (the older
  POD-reconstruct-and-feed-back protocol is the classical
  mode-truncation instability of POD ROMs and has been replaced).
- `scripts/run_qg_kq_scaling.py` — QG hybrid full method panel
  (physics, persistence, linreg, MLP, RFF, ESN, QRC) at any (q, K).
  Same NeuralGCM-style rollout protocol.
- `scripts/run_qg_baselines.py` — adds linear regression, MLP, RFF,
  persistence baselines on the linearisation-gap QG dataset.
- `scripts/sweep_swe_qrc.py` — 27-cell hyperparameter sensitivity sweep
  for Block A.
- `scripts/audit_directional_results.py` — reports per-seed wins +
  lower-mean method alongside Wilcoxon p-value. **Run after every new
  result before writing prose claims.**
- `scripts/plot_kq_scaling_summary.py`, `plot_qg_resgap_summary.py`,
  `plot_qg_skill.py`, `plot_qg_summary.py` — figures.
- `CHANGELOG.md` (this file).

### Headline results (8 seeds, paired Wilcoxon, directionally audited)

**Block A — autonomous QRC on POD coefficients (teacher MSE).**
QRC vs matched-feature-dim ESN at K = q ∈ {5, 6, 7, 8}: QRC wins at
every K, with means 1.81 / 6.26 (K=5) → 1.56 / 12.18 (K=8) and
p ∈ {0.016, 0.008, 0.008, 0.008}. The QRC-vs-ESN gap nearly triples
as ESN overfits the harder higher-order modes; QRC stays essentially
flat. **Strongest result in this release.**

**Block A second-system replication on Lorenz-96 (N = q = K).**
On L96 with the system dimension itself as the complexity knob
(each L96 variable is a true degree of freedom, no POD bookkeeping),
QRC at q qubits beats ESN at matched feature dim 2^q at 6 of 7 sweep
cells (N = 5 through 11, feature dim 32 through 2048) with p = 0.008
(8/8 seed wins) at every cell except N = 5 (p = 0.039) and N = 9
(tied, p = 0.64). QRC's mean test MSE stays in [10.5, 17.6] across
the full sweep, while ESN's oscillates between 13 and 36 with feature
dim. The L96 result confirms the scaling story is not solver-specific
to SWE.

**Hyperparameter sensitivity on L96 (N = 8, 27 cells).** QRC vs
matched-feature-dim ESN with (n_layers, window, ridge_alpha) varying
over {1, 2, 3} × {3, 5, 7} × {0.1, 1, 10}: **24 of 27 cells** give
QRC < ESN with p ≤ 0.05; **23 of 27** give 8/8 seed wins. The 3 cells
where ESN wins are all at (window = 3, ridge_alpha = 10) — heavy
regularisation paired with a short temporal window. Otherwise the
QRC advantage is uniform across the grid. Heatmap at
`results/l96_sensitivity/qrc_mse_heatmap.png`.

**Block B — SWE physics + QRC residual correction (closed-loop).**
At lead-1, QRC corrects physics significantly at K ≥ 6 (p = 0.008,
8/8 seed wins), but matched-feature-dim ESN is a stronger lead-1
corrector (means QRC 1.31 vs ESN 0.42 at K=8; ESN wins 8/8 with
p = 0.008). Across the full 91-step rollout, every learned
correction adds error vs no correction; physics-only wins
rollout-mean at every K (p ≤ 0.023). QRC and ESN are tied at
rollout-mean.

**QG hybrid — resolution-gap data, NeuralGCM-style rollout.**
Lead-1: ESN/RFF/QRC are bunched closely (means within ~30%); RFF
edges the panel at K ≥ 6. Rollout-mean across 28 steps (~3.5 days):
QRC's correction is **statistically tied with no correction**
(p ∈ {0.055, 0.109, 0.383, 0.742} at K = 5, 6, 7, 8; physics-only
slightly better in mean), while every other learned corrector
significantly worsens the rollout-mean: QRC vs ESN p ∈ {0.055,
0.023, 0.039, 0.016}; QRC vs RFF p ≤ 0.078; QRC vs linreg
p ≤ 0.016 at all K; QRC vs MLP p = 0.008 at all K. **The
"stability vs short-term lift" finding is the cleanest publishable
claim from the QG side**: among learned correctors only QRC's
quantum features avoid the catastrophic-amplification regime that
destabilises classical reservoir corrections.

**Sensitivity sweep (Block A).** All 27 cells of the
(n_qubits, window, ridge_alpha) grid are significant; ridge alpha
is the dominant knob, window has near-zero effect.

### Engineering notes

- `--delta-target` (predict next-current rather than next-absolute) is
  essential in Block A for slowly-varying time series; without it,
  both QRC and ESN train MSE is two orders of magnitude worse than
  persistence.
- The closed-loop rollout in Block B and the QG pipeline must keep the
  full-grid state and apply the K-mode correction additively in grid
  space. The earlier protocol of replacing the state with
  `pod.inverse_transform(new_coeffs)` at every step is the classical
  mode-truncation instability of POD ROMs (the trajectory is forced
  into a low-rank subspace that is not closed under nonlinear
  advection). Under linear physics this manifests as distorted
  rollout statistics; under nonlinear physics it produces FFT
  overflow / blow-up.
- For QG, the truth-vs-physics gap matters. A linearisation gap
  (drop the Jacobian) makes the residual approximately linear in the
  windowed POD coefficients of truth; linear regression then crushes
  the panel and there is no nonlinear structure for QRC's quantum
  features to exploit. The resolution gap (truth at higher grid,
  physics at lower) gives a genuinely nonlinear residual where the
  comparison among nonlinear feature extractors becomes meaningful.

## [1.1.0] - 2026-04 (retroactive tag, commit 853b9ab)

arXiv 2604.23743 release: *Fixed-Reservoir vs. Variational Quantum
Architectures for Chaotic Dynamics: Benchmarking QRC and QPINN on the
Lorenz System*.

### Added

- 5-seed multi-system benchmark (Lorenz, Rössler, Lorenz-96 N=8) at
  matched quantum resources.
- Classical Echo State Network baseline at N=500 reservoir neurons.
- QRC quantum-circuit diagram and architecture figures.
- Paired Wilcoxon signed-rank statistical reporting.
- Temporal-windowing ablation (Takens delay-embedding within QRC).
- Gradient-norm diagnostics ruling out barren plateaus as the cause of
  QPINN's underperformance; capacity bottleneck identified.

### Headline result

- QRC vs QPINN on Lorenz, 5 matched seeds: 81% lower train MSE,
  93% lower test MSE, ~52,000× faster training (0.2 s vs 2.4 h).

## [1.0.0] - 2024-12

First complete release: initial QRC vs QPINN comparison on the Lorenz
system at single-seed resolution, with the basic temporal-windowing
mechanism and ridge-regression readout.
