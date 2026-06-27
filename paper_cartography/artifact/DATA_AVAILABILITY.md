# Data Availability statement (draft to paste into the paper)

Replace `<REPO-URL>` with the public repository URL once created.

---

**Data Availability.** The result files and the two analysis scripts that
generate every table in this paper are available at `<REPO-URL>`. The repository
includes the per-seed result JSONs for Case I and Case II, the scripts
(`mv_correlator_qrc.py`, `feedback_qrc.py`), and a README mapping each table to
the command that reproduces it. All results are deterministic and reproduce
bit-for-bit from the provided random seeds.

---

LaTeX version (already inserted in `main.tex` and `main_ieee.tex` before the
bibliography):

```latex
\section*{Data Availability}
The result files and the two analysis scripts that generate every table in this
paper are available at \url{<REPO-URL>}. The repository includes the per-seed
result JSONs for Case~I and Case~II, the scripts (\texttt{mv\_correlator\_qrc.py},
\texttt{feedback\_qrc.py}), and a README mapping each table to the command that
reproduces it. All results are deterministic and reproduce bit-for-bit from the
provided random seeds.
```
