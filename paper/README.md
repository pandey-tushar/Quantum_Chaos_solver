# Paper: Quantum Reservoir Computing for Chaotic Dynamics

## Paper Details

**Title:** Quantum Reservoir Computing Outperforms Quantum Physics-Informed Neural Networks for Chaotic Dynamics: A Comparative Study on the Lorenz System

**Author:** Tushar Pandey (Texas A&M University)

**Target:** arXiv preprint

**Estimated Length:** 4-5 pages

## Files

- `main.tex` - Main LaTeX source file (ready for arXiv submission)

## Compilation

### Option 1: Local LaTeX
```bash
pdflatex main.tex
pdflatex main.tex  # Run twice for references
```

### Option 2: Overleaf
1. Upload `main.tex` to [Overleaf](https://www.overleaf.com/)
2. Compile

### Option 3: arXiv
Upload `main.tex` directly to arXiv - it will compile automatically.

## Abstract

> We present a systematic comparison of two quantum machine learning approaches for representing chaotic dynamical systems: Quantum Physics-Informed Neural Networks (QPINN) and Quantum Reservoir Computing (QRC). Using the Lorenz system as a benchmark, we demonstrate that QRC significantly outperforms QPINN, achieving 40.8% lower mean squared error while training approximately 14,000× faster.

## Key Results

| Method | Train MSE | Test MSE | Training Time |
|--------|-----------|----------|---------------|
| QPINN | 37.77 | --- | 3.1 hours |
| **QRC** | **22.37** | **3.16** | **0.8 sec** |

## Dependencies (for figures)

To regenerate figures from the results:
```bash
cd ../
python scripts/create_comparison_plots.py
```

## Citation

```bibtex
@article{pandey2026quantum,
  title={Quantum Reservoir Computing Outperforms Quantum Physics-Informed Neural Networks for Chaotic Dynamics},
  author={Pandey, Tushar},
  journal={arXiv preprint},
  year={2026}
}
```

