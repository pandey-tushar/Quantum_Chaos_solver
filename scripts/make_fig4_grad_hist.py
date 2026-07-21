#!/usr/bin/env python3
"""
make_fig4_grad_hist.py -- generate paper/mlst_submission/fig4_grad_hist.pdf, the
gradient-variance histogram panel requested by referee point R2.6 ("add
gradient-variance distributions/histograms, not just norms").

For each depth L in {1, 3, 5} the figure histograms the per-parameter gradient
variance: for every structurally live parameter-time cell (k, t) it takes the
variance, across the 100 random initialisations, of the exact parameter-shift,
fixed-circuit, fixed-t gradient dC_t/dtheta_k of the bounded local cost. A cell
is live if that across-init variance exceeds 1e-28, the same live/dead split
used in Table 2. Each histogram therefore has one entry per live cell (48, 432
and 1128 cells at L = 1, 3, 5), and its mean is, by construction, the live
variance reported in Table 2 and Figure 3 (6.6e-2, 1.6e-2, 7.3e-3). The vertical
reference marks the local-cost Haar/2-design floor 2^{-2n} = 2^{-8} = 3.906e-3
at n=4. The whole distribution, not merely its mean, migrates toward that floor
as depth grows and stacks up against it rather than crossing below, which is the
decay-then-saturate signature of a fixed-n circuit approaching its 2-design
floor and not a barren plateau.

Reads results/r1/variance_b1_e1_L{1,3,5}.json (the per_init_gradients arrays);
no new experiments are run.

Agg backend, Okabe-Ito colourblind-safe palette, no title, all fonts >= 12,
savefig dpi = 600.
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
OUT = ROOT / "paper" / "mlst_submission" / "fig4_grad_hist.pdf"

# Okabe-Ito colourblind-safe palette (matched to fig3 conventions).
C_L = {1: "#0072B2", 3: "#D55E00", 5: "#009E73"}   # blue, vermillion, green
C_FLOOR = "#000000"                                 # black reference line

DEAD_VAR_THRESHOLD = 1e-28          # same as r1_corrections_analysis.json
HAAR_FLOOR = 2.0 ** (-8)            # 2^{-2n}, n=4 -> 3.90625e-3


def sci(x: float) -> str:
    """Format x as a LaTeX 'm.m\\times10^{e}' string (matches Table 2 style)."""
    e = int(np.floor(np.log10(x)))
    m = x / 10.0 ** e
    return r"%.1f\times10^{%d}" % (m, e)


def live_cell_variances(path: Path):
    """Return (per-live-cell across-init gradient variances, their mean)."""
    d = json.loads(path.read_text())
    pig = d["per_init_gradients"]
    # mcclean_bounded is [n_inits][n_times][n_params]; exact PS gradient of the
    # bounded local cost C_t at each fixed collocation time t.
    arr = np.array([g["mcclean_bounded"] for g in pig])   # (inits, times, params)
    var_cells = arr.var(axis=0, ddof=0)                   # (times, params)
    v = var_cells[var_cells > DEAD_VAR_THRESHOLD].ravel()  # live cells only
    return v, float(v.mean())


def main():
    Ls = (1, 3, 5)
    data = {L: live_cell_variances(RESULTS / f"variance_b1_e1_L{L}.json") for L in Ls}

    plt.rcParams.update({
        "font.size": 13,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
    })
    fig, ax = plt.subplots(figsize=(6.8, 4.6))

    # Common log-spaced bins across all three depths.
    all_v = np.concatenate([data[L][0] for L in Ls])
    lo, hi = all_v.min(), all_v.max()
    bins = np.logspace(np.log10(lo), np.log10(hi), 26)

    ymax = 0.0
    for L in Ls:
        v, mean_v = data[L]
        # Fractional histogram (each depth integrates to 1 over its own cells);
        # avoids the log-bin-width inflation of a per-unit density.
        w = np.ones_like(v) / v.size
        counts, _ = np.histogram(v, bins=bins, weights=w)
        ymax = max(ymax, counts.max())
        ax.hist(v, bins=bins, weights=w, histtype="stepfilled",
                color=C_L[L], alpha=0.32, edgecolor=C_L[L], linewidth=1.7,
                label=(r"$L=%d$ (mean $%s$)" % (L, sci(mean_v))))
        # thin dashed marker at this distribution's mean (= Table 2 live variance)
        ax.axvline(mean_v, color=C_L[L], ls="--", lw=1.4, alpha=0.9)

    ax.axvline(HAAR_FLOOR, color=C_FLOOR, ls=":", lw=2.4,
               label=r"$2$-design floor $2^{-2n}=3.9\times10^{-3}$")

    ax.set_xscale("log")
    ax.set_xlabel(r"per-parameter gradient variance "
                  r"$\mathrm{Var}_{\mathrm{init}}(\partial C_t/\partial\theta_k)$")
    ax.set_ylabel("fraction of live parameters")
    ax.set_ylim(0.0, ymax * 1.08)
    ax.set_xlim(lo * 0.6, hi * 1.9)
    ax.grid(True, which="both", ls="-", lw=0.3, alpha=0.30)
    # Legend placed fully ABOVE the axes so it touches no bar, line, or tick.
    ax.legend(fontsize=12, loc="lower left", bbox_to_anchor=(0.0, 1.02, 1.0, 0.16),
              mode="expand", ncol=2, framealpha=1.0, borderaxespad=0.0,
              handlelength=1.6, columnspacing=1.4)

    fig.tight_layout(pad=0.5)
    fig.savefig(OUT, format="pdf", bbox_inches="tight", dpi=600)
    # Companion PNG for on-screen visual verification of the layout rule.
    fig.savefig(OUT.with_suffix(".png"), format="png", bbox_inches="tight", dpi=600)
    plt.close(fig)

    print(f"[saved] {OUT}")
    for L in Ls:
        v, mean_v = data[L]
        print(f"  L={L}: n_live_cells={v.size}  live_mean_var={mean_v:.4e}")
    print(f"  Haar floor 2^-8 = {HAAR_FLOOR:.4e}")


if __name__ == "__main__":
    main()
