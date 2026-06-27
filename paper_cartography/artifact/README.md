# Artifact: "Do Fair Baselines Erase the Quantum Edge?"

Reproduction artifact for the two case studies in the paper. Every number in
both tables is produced by the two scripts below and stored as a result JSON in
`results/`. All runs are deterministic (fixed seeds) and reproduce bit-for-bit.

## Contents

```
scripts/
  mv_correlator_qrc.py     # Case I: 2-point-correlator QRC vs Poly2 (+concat)
  mv_correlator_seeds.py   # Case I: multi-seed driver for Table 1
  feedback_qrc.py          # Case II: feedback QRC vs tuning-matched ESN
results/
  mv_correlator/           # Case I result JSONs (one per coupling x seed pair)
    phase3_c{c}_n0.1_ds{d}_rs{r}.json
    caseI_multiseed_summary.json     # aggregated mean +/- std (Table 1)
  feedback_qrc/            # Case II result JSONs
    phase2_ps0.05_h1.json            # 10x10 = 100-seed run (Table 2 lower, Fig 2)
    phase1_ps{0.02,0.05,0.1}_h1_ds0_rs0.json   # single-seed sweep (Table 2 upper)
```

## Environment

Python 3 with `numpy`, `scipy`, `scikit-learn`. No quantum-hardware or
special dependencies; everything is statevector simulation at `q <= 11`.

## Reproducing the tables

All commands are run from the directory that contains `scripts/` and `results/`.
Each script writes its JSON under `results/` and prints a SHA-256 of the file.

### Case study I (Table 1, Fig. 1) — correlator QRC vs. Poly2

The correctness gates (G1-G6: physical reservoir, exact 2-point readout,
shot->exact convergence, QR2 distinctness, quadratically-decodable data) must
pass before any science is run:

```
python scripts/mv_correlator_qrc.py --self-test
```

Single-seed concat ablation at one coupling (data_seed = res_seed = 0):

```
python scripts/mv_correlator_qrc.py --phase3 --coupling 0.1 --obs-noise 0.1
```

Full multi-seed Table 1 (five data-seed x reservoir-seed pairs per coupling
c in {0.0, 0.1, 0.2}, obs_noise 0.1), with aggregated mean +/- std and the
cat-vs-Poly2 delta written to `caseI_multiseed_summary.json`:

```
python scripts/mv_correlator_seeds.py
```

The reported Table 1 values are the `QRC_mean/std`, `Poly2_mean/std`,
`cat_mean/std`, and `cat_vs_Poly2_pct_mean/std` fields of
`results/mv_correlator/caseI_multiseed_summary.json`, per coupling.

### Case study II (Table 2, Fig. 2) — feedback QRC vs. tuning-matched ESN

Correctness gates (including the feedback-wiring gate: gain 0 reproduces
open-loop to machine precision, and strict causality):

```
python scripts/feedback_qrc.py --self-test
```

Table 2 *upper* block (single-seed feedback-vs-open-loop sweep over switching
rates), one file per p_switch:

```
python scripts/feedback_qrc.py --phase1 --p-switches 0.02 0.05 0.1
```

Table 2 *lower* block and Fig. 2 (the fair 10x10 = 100-seed comparison at
p_switch = 0.05):

```
python scripts/feedback_qrc.py --phase2 --p-switch 0.05 --n-seeds 10
```

This writes `results/feedback_qrc/phase2_ps0.05_h1.json`, which stores the
per-run NRMSE and MDA arrays for FB_QRC, OpenLoop_QRC, ESN, Poly2, and Linear.
The Table 2 lower means/std are `mean`/`std` over these arrays; the paired
statistics in the text (paired t-test, Wilcoxon, and the bootstrap 95% CI
[0.057, 0.078]) are computed from the 100 per-run ESN-minus-FB_QRC NRMSE
differences in that file.
