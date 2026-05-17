# QST Submission Plan — Fixed-Reservoir vs Variational

**Status as of 2026-05-16:** Submission is **ready in ScholarOne, pending final
"Submit" click**. All 7 ScholarOne steps green-checked; PDF proof viewed;
Submit button is active. Awaiting author go-ahead before the irreversible
final click.

**Target venue:** Quantum Science and Technology (IOP). Subscription
publication free. Texas A&M institutional agreement covers the $4,090 gold
open access fee — confirmed eligible (see "Open access" section below).
QST is selective — accepts only a small proportion of submissions; needs
lasting scientific impact + essential reading for the sub-field.

**Honest acceptance probability with the present manuscript:** 40–55%.
Transfer-to-other-IOP-journals is offered if rejected.

## Strategic shift from the published arXiv version

The published arXiv (2604.23743) leads with the QRC-vs-QPINN comparison
("81% lower MSE, 52000× faster"). For QST, the comparison is the *evidence*,
not the contribution. The QST-worthy contribution is the **diagnosis**:
> "The variational quantum failure on chaotic dynamics at NISQ scale is
> capacity-limited, not barren-plateau-driven, and a fixed-reservoir
> architecture bypasses both modes by construction."

This reframing was achieved by reordering — the diagnostic data was
already in the arXiv version but buried in Discussion.

## Pre-submission checklist (all done)

| # | Item | Status |
|---|---|---|
| 1 | WVQC ablation (5 seeds) | **DONE.** Train MSE 61.7 ± 12.4 / test 39.8 ± 14.6; recovers ~32% of QRC-QPINN gap. Result wired into §5.4. Compute: ~3.2h. |
| 2 | Per-layer gradient-norm distribution figure | **DONE.** 3 seeds × 200 iters; measured `‖∇_layer‖` in [3.6e3, 4.8e4] — 4-5 decades above McClean threshold ~0.23. Figure wired into §4.1. Compute: ~5.3h. |
| 3 | Theoretical capacity argument (McClean bound) | **DONE in §4.1 of main.tex.** |
| 4 | Scaling-to-crossover **measurement** (not just heuristic) | **DONE.** Empirical sweep q ∈ {5..10} vs ESN(N=500) shows QRC degrades monotonically with q at fixed target dim (signal-dilution). §5.3 reframed from "heuristic prediction" to "empirical scaling shows the right axis is matched-complexity." |
| 5 | Reframe abstract + title around capacity-vs-BP diagnosis | **DONE.** |
| 6 | Reorder Results: diagnosis BEFORE comparison | **DONE.** §4.1 = diagnosis, §4.2 = QRC vs QPINN, §4.3 = windowing, §4.4 = multi-system. |
| 7 | Move ESN baseline out of main results table → Discussion | **DONE.** Table 1 has only QPINN and QRC; ESN is in §5.2. |
| 8 | ~~Convert to `iopart.cls`~~ | **NOT NEEDED.** IOP author guide confirms: "You can format your paper in the way that you choose" — they typeset. The article-class `main.tex` compiles to a submission-ready PDF directly. `main_iopart.tex` is retained as a fallback but unused. |
| 9 | Significance statement (~120 words plain language) | **DONE.** `significance_statement.txt` (122 words). |
| 10 | Cover letter emphasising lasting impact | **DONE.** `cover_letter.tex` → `cover_letter.pdf` (2 pp, 110 KB). |

## Double-anonymous adaptation (not in original plan)

Discovered at the ScholarOne welcome page: **QST now uses double-anonymous
peer review** (was single-anonymous in the older author guide). Required an
anonymized manuscript:

- `main_anon.tex` is `main.tex` with the author block replaced by
  *"Author name(s) and affiliation(s) removed for double-anonymous peer review"*
  and the GitHub URL in §Code Availability replaced by a placeholder
  ("will be provided upon acceptance").
- `main_anon.pdf` is the compile target uploaded to ScholarOne (12 pp, 699 KB).
- The non-anonymised `main.pdf` (12 pp, 769 KB) is retained for reference
  and for the post-acceptance camera-ready version.

## Open access eligibility (confirmed free)

Verified via the Texas A&M Libraries scholarly-communications page and the
IOP Publishing Support transformative-agreements hub:

- Texas A&M University – College Station is in the **TAMU System** entry
  on IOP's US-institutions list (no per-author article cap).
- Quantum Science and Technology is in **List A** (journals included in all
  transformative agreements).
- ScholarOne OA selection: **Yes: Standard rate**. The system will
  auto-detect the institutional agreement and waive the $4,090 APC at
  acceptance. Per the ScholarOne note: "*Whichever choice you select, we
  will investigate your eligibility for funding under an institutional
  open access agreement.*"

## ScholarOne submission session log (2026-05-16)

| Step | Status | Notes |
|---|---|---|
| 1: Article Information | ✓ | Type=Paper. Title pre-filled (132/300 chars). Abstract trimmed to 263/300 words via in-form edit. Social Media Abstract set to 93/100 chars. Data availability = Yes (public). |
| 2: File Upload | ✓ | `main_anon.pdf` uploaded as "Complete Document for Review (PDF Only)". |
| 3: Keywords | ✓ | 6 keywords: quantum reservoir computing, variational quantum circuits, barren plateaus, chaotic dynamics, NISQ, quantum machine learning. |
| 4: Author Information | ✓ | Author manually updated from stale "PhD student" title in ScholarOne profile. |
| 5: Referees | ✓ | Filled by author. |
| 6: Policies & Information | ✓ | `cover_letter.pdf` attached. Funding=No. OA=Yes Standard rate. Reproduced material=No. Policy checkboxes + ethics statement + data availability (Option 1, GitHub URL) confirmed by author. WoS Peer Review Transparency=Agree. |
| 7: Review & Submit | **PENDING FINAL CLICK** | All sections show ✓. PDF proof viewed. Submit button active. Awaiting explicit author authorization. |

## Files in this directory

- `main.tex` — restructured paper (non-anonymized). For arXiv v2 and
  camera-ready submission after acceptance.
- `main.pdf` — compiled non-anonymized PDF (12 pp, 769 KB).
- `main_anon.tex` — anonymized version of main.tex. Author block and
  GitHub URL removed. **This is the version uploaded to ScholarOne.**
- `main_anon.pdf` — compiled anonymized PDF (12 pp, 699 KB).
- `main_iopart.tex` — IOP-class fallback version (unused; IOP accepts any
  format). Keep for reference; do not delete.
- `cover_letter.tex` / `cover_letter.pdf` — formal cover letter to the
  QST editor.
- `significance_statement.txt` — 122-word plain-language statement
  (pasted into the ScholarOne form).
- `PLAN.md` — this file.
- `per_layer_gradient_norms.png`, `fig1_trajectory.png`, `fig2_training.png`
  — local copies of figures referenced by `main.tex` and `main_anon.tex`.

## What's left

1. **The final "Submit" click in ScholarOne.** Irreversible. Awaiting
   author authorization in the active browser session.
2. **Post-submission:** track via ScholarOne dashboard
   (`mc04.manuscriptcentral.com/qst-iop`). Typical median time to first
   decision before peer review = 7 days; after peer review = 69 days.
3. **If accepted:** OA verification + camera-ready upload of the
   non-anonymized `main.pdf`; arXiv v2 update with the restructured
   manuscript.

## Lessons learned (for future submissions)

- **IOP accepts any LaTeX format.** Don't waste a day on `iopart.cls`
  conversion. Compile your normal article-class .tex and upload the PDF.
- **Check the peer review model on the ScholarOne welcome page**, not
  just the author guide. Journals change models; the author guide lags.
  If double-anonymous, anonymize the manuscript LaTeX (author block + any
  identifying URLs/acks) and recompile.
- **Use ScholarOne pre-fill.** Drop the manuscript PDF on the welcome
  modal; auto-populates title and abstract. Saves several minutes of
  retyping and reduces transcription error.
- **Watch the abstract word limit on the form**, not just the
  manuscript. The form's `DOCUMENT_ABSTRACT` field has a hard 300-word
  limit even if the manuscript PDF abstract is longer.
- **Texas A&M is on the IOP transformative-agreements list.** No need
  to apply or pay; the agreement auto-applies at acceptance. Verify
  before submission by checking the Texas A&M Libraries OA page or the
  IOP US-institutions list.
