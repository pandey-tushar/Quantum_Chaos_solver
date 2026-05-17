# Paper

Two manuscript trees in this directory:

## 1. Original arXiv version — `main.tex` (this directory)

**Status:** Published as arXiv:2604.23743 (January 2026).

**Title:** *Fixed-Reservoir vs. Variational Quantum Architectures for Chaotic
Dynamics: Benchmarking QRC and QPINN on the Lorenz System*

**Author:** Tushar Pandey (Texas A&M University)

**Framing:** Benchmarking-paper — leads with the QRC-vs-QPINN comparison
(81% lower MSE, ~52,000× faster training).

**Files:** `main.tex`, `fig1_trajectory.png`, `fig2_circuit.png`,
`fig2_training.png`, `fig3_ablation.png`, `fig4_comparison.png`.

## 2. QST submission version — `qst_submission/` subdirectory

**Status (2026-05-16):** Compiled, anonymized, uploaded to ScholarOne,
PDF proof viewed. **Pending final "Submit" click in the active browser
session.**

**Title:** *Capacity, Not Barren Plateaus: Diagnosing Variational Quantum
Training Failure on Chaotic Dynamics, and a Fixed-Reservoir Resolution*

**Target venue:** Quantum Science and Technology (IOP), gold open access
fee waived by the Texas A&M institutional transformative agreement.

**Strategic shift from the arXiv version:** the diagnosis (capacity-limited
QPINN failure, ruled out by direct gradient-norm measurement against the
McClean barren-plateau threshold) is moved to the headline; the comparison
becomes the supporting evidence. New WVQC ablation, per-layer gradient-norm
figure, and empirical scaling sweep have all been added.

See `qst_submission/PLAN.md` for the complete checklist, submission session
log, and remaining steps.

## Compilation (both versions)

```bash
# Original arXiv version
cd paper/
pdflatex main.tex && pdflatex main.tex

# QST anonymized submission version
cd paper/qst_submission/
pdflatex main_anon.tex && pdflatex main_anon.tex
pdflatex cover_letter.tex
```
