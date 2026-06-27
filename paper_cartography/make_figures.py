#!/usr/bin/env python3
"""Generate the two figures for the cartography paper from the verified JSONs.
Reads results/mv_correlator/ (Case I) and results/feedback_qrc/ (Case II).
Outputs fig_caseI.pdf and fig_caseII.pdf in this directory."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent.parent
MVC = ROOT / "results" / "mv_correlator"
FBQ = ROOT / "results" / "feedback_qrc"
HERE = Path(__file__).parent

plt.rcParams.update({"font.size": 11})

# Okabe-Ito colorblind-safe palette
C_CLASSICAL = "#0072B2"  # blue
C_QUANTUM   = "#D55E00"  # vermillion
C_CAT       = "#009E73"  # bluish green
C_OPENLOOP  = "#999999"  # grey

# ---------------------------------------------------------------------------
# Figure 1 (Case I): NRMSE vs coupling at h=1 -- QRC vs Poly2 vs concat
# Multi-seed means +/- s.d. (5 seed pairs per coupling), matching Table 1.
# ---------------------------------------------------------------------------
summ = json.load(open(MVC / "caseI_multiseed_summary.json"))["per_coupling"]
couplings = [0.0, 0.1, 0.2]
qrc   = [summ[f"{c}"]["QRC_mean"]   for c in couplings]
qrc_s = [summ[f"{c}"]["QRC_std"]    for c in couplings]
poly2 = [summ[f"{c}"]["Poly2_mean"] for c in couplings]
ply_s = [summ[f"{c}"]["Poly2_std"]  for c in couplings]
cat   = [summ[f"{c}"]["cat_mean"]   for c in couplings]
cat_s = [summ[f"{c}"]["cat_std"]    for c in couplings]

fig, ax = plt.subplots(figsize=(5.2, 3.7))
ax.errorbar(couplings, poly2, yerr=ply_s, fmt="o-", color=C_CLASSICAL, capsize=3,
            label="Poly2 (capacity-matched)", zorder=3)
ax.errorbar(couplings, qrc, yerr=qrc_s, fmt="s--", color=C_QUANTUM, capsize=3,
            label="correlator QRC", zorder=3)
ax.errorbar(couplings, cat, yerr=cat_s, fmt="^:", color=C_CAT, capsize=3,
            label=r"Poly2 $\oplus$ QRC", zorder=3)
ax.axhline(1.0, color="black", lw=1.0, ls=":", zorder=5)
ax.text(0.99, 1.01, "trivial-mean predictor", fontsize=8, color="black",
         ha="right", va="bottom", zorder=6, transform=ax.get_yaxis_transform())
ax.set_xlabel("cross-asset coupling $c$")
ax.set_ylabel("test NRMSE (one-step)")
ax.set_title("Case I: correlator QRC vs.\npolynomial control", fontsize=11)
ax.set_xticks(couplings)
ax.legend(frameon=False, fontsize=9, loc="center left")
ax.grid(alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(HERE / "fig_caseI.pdf")
print("wrote fig_caseI.pdf  | QRC", [round(x,3) for x in qrc],
      "Poly2", [round(x,3) for x in poly2], "cat", [round(x,3) for x in cat])

# ---------------------------------------------------------------------------
# Figure 2 (Case II): NRMSE bar chart, 3x3-seed mean +/- std, p_switch=0.05
# ---------------------------------------------------------------------------
d = json.load(open(FBQ / "phase2_ps0.05_h1.json"))
order = ["ESN", "Linear", "Poly2", "FB_QRC", "OpenLoop_QRC"]
labels = ["ESN\n(tuned)", "Linear", "Poly2", "Feedback\nQRC", "Open-loop\nQRC"]
colors = [C_CLASSICAL, C_CLASSICAL, C_CLASSICAL, C_QUANTUM, C_OPENLOOP]
means = [np.mean(d["nrmse"][m]) for m in order]
stds = [np.std(d["nrmse"][m]) for m in order]

fig, ax = plt.subplots(figsize=(5.2, 3.7))
bars = ax.bar(range(len(order)), means, yerr=stds, capsize=4,
        color=colors, edgecolor="black", linewidth=0.6, zorder=2)
ax.axhline(1.0, color="black", lw=1.0, ls=":", zorder=5)
ax.text(0.98, 1.005, "trivial-mean predictor", fontsize=8, color="black",
         ha="right", va="bottom", zorder=6,
         transform=ax.get_yaxis_transform())
ax.set_ylim(0, max(means) + max(stds) + 0.12)
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("test NRMSE (one-step)")
ax.set_title("Case II: feedback QRC vs. tuning-matched baselines", fontsize=10.5)
# Legend distinguishing classical / quantum / open-loop
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=C_CLASSICAL, edgecolor="black", label="classical"),
                   Patch(facecolor=C_QUANTUM, edgecolor="black", label="feedback QRC"),
                   Patch(facecolor=C_OPENLOOP, edgecolor="black", label="open-loop QRC")],
          frameon=False, fontsize=8.5, loc="upper left")
ax.grid(axis="y", alpha=0.3, zorder=0)
fig.tight_layout()
fig.savefig(HERE / "fig_caseII.pdf")
print("wrote fig_caseII.pdf |", {m: round(np.mean(d["nrmse"][m]),3) for m in order})
