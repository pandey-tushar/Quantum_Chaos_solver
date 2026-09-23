# qrc_bench

Fair benchmarking of quantum reservoir computing (QRC) against classical models that get the
same tuning budget, the same inputs and the same number of features.

## Papers

Each paper's code, data and manuscript live on their own branch.

| Branch | Paper | Link |
|---|---|---|
| [`paper-1-qst`](../../tree/paper-1-qst) | Fixed-Reservoir vs Variational Quantum Architectures for Chaotic Dynamics: Benchmarking QRC and QPINN on the Lorenz System | [arXiv:2604.23743](https://arxiv.org/abs/2604.23743) |
| [`paper-2-methods`](../../tree/paper-2-methods) | A Quantum Reservoir Architecture for Chaotic Forecasting and a Test of Whether Its High Dimension Helps (QUANCOM 2026) | [arXiv:2607.07978](https://arxiv.org/abs/2607.07978) |
| [`paper-7-cartography`](../../tree/paper-7-cartography) | When Classical Baselines Are Tuned as Carefully as the Quantum Model, Does Quantum Reservoir Computing Still Win? (IEEE QCE 2026, QuBench workshop) | [arXiv:2607.09905](https://arxiv.org/abs/2607.09905) |

Exploratory branches that are not papers: `paper-3-chaoticity`, `paper-5-quantum-input`.

## What the pipeline does

1. **Tasks:** `henon` (coupled Hénon maps), `lorenz96`, `switching` / `switching_multi`
   (AR(1) with a hidden sign regime), `drift`. Any number of series.
2. **Quantum reservoir:** angle encoding (`per_series`, or `dense_rxrz` with two series per
   qubit), a fixed random Hamiltonian (`ising_xx`, `xxz`, `xxz_uniform`), Z / ZZ readout, or ZX
   with a second measurement basis.
   Memory is `reset` (the last *L* inputs re-driven at every step, a fixed-window feature map)
   or `recurrent` (the memory state carried forward), with optional measurement feedback.
3. **Classical models:** ESN, random features, Poly2, linear; persistence as a floor.
4. **Matched comparisons:**
   - `window`: reset-memory QRC vs random features, Poly2 and linear, all on the same input
     window, with QRC and random features at the same feature width.
   - `recurrent`: recurrent-memory QRC vs an ESN of the same width.
5. **Same tuning for every model:** Optuna TPE with the same sampler seed, held-out tuning seeds,
   rolling-origin folds and a standardised ridge readout. The budget is matched by *search
   density*, not by trial count: each model gets `--trials-per-dim` trials per knob it tunes and
   keeps going while it is still improving, because an equal count would search a one-knob model
   exhaustively and a five-knob model sparsely. Convergence curves are recorded. Each model's best
   setting is frozen and evaluated once on fresh data and reservoir seeds.
6. **Statistics:** paired comparisons at run and data-seed level, bootstrap confidence
   intervals, Wilcoxon, Holm correction. Comparisons against a model without a reservoir seed are
   flagged, as only their data-seed-level statistics are independent. Every result records trial
   times, package versions, GPU and git commit.

The simulation is exact (no truncation). It runs batched on the CPU (numpy) or GPU (cupy);
`auto` uses the GPU from 10 qubits, in single precision.

**Finite shots.** Hardware reports counts, not expectation values, so every QRC readout is measured
from a finite number of Z-basis shots per step: `--shots auto` (the default) uses 1024 x q shots,
all in the single Z basis, from which the commuting Z and ZZ features and the feedback signal are
all read, so their shot noise is correlated the way it is on hardware. This is the restarting
protocol -- each step's measurement is a fresh preparation of the same circuit, so the memory state
carried into the next step stays exact and is never collapsed. Pass `--shots N` for an explicit
budget, or `--shots 0` for exact expectation values (which no hardware can measure; the run logs a
warning and records `"shots": null`).

**Measurement bases.** By default everything is read from the single Z basis, where Z and ZZ
commute, so one measurement per step gives all 78 features at q = 12. `--readout ZX` measures the
same one- and two-point block a second time in the non-commuting X basis, doubling the width
(156 at q = 12). The *total* shot budget is unchanged and split 50/50 across the two bases, so the
comparison is "more bases" against "more shots per basis" at a fixed cost; classical widths follow
the QRC width as always, so the comparison stays matched. The feedback signal and the memory state
stay on the Z basis. `ZX` runs on `batched` only -- the reference simulators know only the Z
diagonal -- and holds a second unitary per reservoir on the device (134 MB more at q = 12 in
single precision).

## Install

```bash
pip install -e ".[test]"        # add ".[gpu]" for cupy
```

## Use

```bash
python -m qrc_bench list
python -m qrc_bench layout --n-series 9 --encoding dense_rxrz --n-mem 3
python -m qrc_bench experiment --task lorenz96 --task-arg n_series=10 --kind window \
    --input-window 3 --n-mem 2 --trials 100 --out results/l96_window
python -m qrc_bench bench --layouts 5:3 6:6 --steps 200
```

`experiment` runs at `q >= 10` (`--q-min` to change), resumes an interrupted run from the Optuna
studies under `--out`, and refuses to resume when the comparison, protocol or simulation settings
have changed.

## Test

```bash
python -m pytest        # includes regressions against the cartography paper's published numbers
```

## License

Apache 2.0 (see `LICENSE`).
