#!/usr/bin/env python3
"""
make_fig3_variance.py -- generate paper/mlst_submission/fig3_variance.pdf, the
corrected variance figure the manuscript's Figure-1 TODO calls for.

The plot (log-y, vs parameter count {15, 45, 75}) shows:
  (i)   the POOLED mean McClean-comparable variance (the paper's headline
        number, which is flat within ~2x) -- dilution artifact;
  (ii)  the LIVE-parameter mean variance (structurally-dead parameters removed),
        which decays ~9x from L=1 to L=5;
  (iii) a horizontal reference at the Haar/2-design saturation scale
        2^{-2n} = 2^{-8} = 3.906e-3 (local single-qubit-Pauli cost, n=4);
  (iv)  a dashed exponential-collapse reference anchored at the L=1 live value:
        what a genuine barren plateau (exponential decay in circuit size) would
        look like -- for visual contrast only.

Reads results/r1/r1_corrections_analysis.json (produced by
analyze_r1_corrections.py) so the numbers are exactly the audited ones.

Agg backend, colorblind-safe palette, no title, single-column (~5x3.5 in).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "r1"
OUT = ROOT / "paper" / "mlst_submission" / "fig3_variance.pdf"

# Okabe-Ito colorblind-safe palette.
C_POOLED = "#0072B2"   # blue
C_LIVE = "#D55E00"     # vermillion
C_HAAR = "#009E73"     # green
C_BP = "#555555"       # grey (barren-plateau reference)


def main():
    ana = json.loads((RESULTS / "r1_corrections_analysis.json").read_text())
    ld = ana["e1_live_dead"]

    Ls = (1, 3, 5)
    params = [15, 45, 75]
    pooled = [ld["per_L"][str(L)]["pooled_mean_var"] for L in Ls]
    live = [ld["per_L"][str(L)]["live_mean_var"] for L in Ls]
    haar = ld["haar_saturation_scale"]["value"]   # 2^-8 = 3.906e-3

    # Barren-plateau contrast reference: exponential decay in circuit size.
    # Anchor at the L=1 live value and decay by a constant factor per added
    # layer block (illustrative slope; a true BP would collapse far faster in
    # the number of QUBITS, but here we contrast in the paper's depth axis).
    # Use a decade-per-30-params slope so the curve visibly plunges below 2^-8.
    bp_anchor = live[0]
    bp = [bp_anchor * 10.0 ** (-(p - params[0]) / 30.0) for p in params]

    plt.rcParams.update({"font.size": 11, "axes.linewidth": 0.8,
                         "xtick.direction": "out", "ytick.direction": "out"})
    fig, ax = plt.subplots(figsize=(5.0, 3.5))

    ax.plot(params, pooled, "o-", color=C_POOLED, lw=1.8, ms=6,
            label="pooled mean (all params)")
    ax.plot(params, live, "s-", color=C_LIVE, lw=1.8, ms=6,
            label="live params only")
    ax.plot(params, bp, "^--", color=C_BP, lw=1.5, ms=5,
            label="barren-plateau (exp.) reference")
    ax.axhline(haar, color=C_HAAR, ls=":", lw=1.6,
               label=r"Haar floor $2^{-2n}=2^{-8}$")

    ax.set_yscale("log")
    ax.set_xlabel("parameter count")
    ax.set_ylabel(r"gradient variance $\mathrm{Var}_{\mathrm{est}}$")
    ax.set_xticks(params)
    ax.set_xticklabels([f"{p}\n(L={L})" for p, L in zip(params, Ls)])
    ax.set_xlim(params[0] - 8, params[-1] + 8)
    ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.35)
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.9)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[saved] {OUT}")
    print(f"  pooled = {[f'{v:.3e}' for v in pooled]}")
    print(f"  live   = {[f'{v:.3e}' for v in live]}")
    print(f"  Haar 2^-8 = {haar:.3e}")
    print(f"  live decay L1/L5 = {live[0]/live[2]:.2f}x")


if __name__ == "__main__":
    main()
