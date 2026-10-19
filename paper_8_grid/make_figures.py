"""Figures of the paper from the pipeline's JSON records (grayscale-safe, single-column width).

Usage: python make_figures.py [--results DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from make_tables import CELLS, EXACT, LABEL, REFINE, RESULTS, load, mean_sd  # noqa: E402

HERE = Path(__file__).resolve().parent
MODELS = ["qrc", "random_features", "poly2", "linear", "esn"]
STYLE = {  # (face gray, hatch)
    "qrc": ("0.15", ""), "random_features": ("1.0", "///"), "poly2": ("0.6", ""),
    "linear": ("1.0", "..."), "esn": ("0.85", "xxx"),
}
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "hatch.linewidth": 0.6, "pdf.fonttype": 42})


def bar(ax, x, h, err, model, width, label=True):
    face, hatch = STYLE[model]
    ax.bar(x, h, width, yerr=err, color=face, hatch=hatch, edgecolor="black", linewidth=0.6,
           error_kw={"elinewidth": 0.6, "capsize": 1.5}, label=LABEL[model] if label else None)


def fig_cells(results: Path):
    recs = [load(results, c[3]) for c in CELLS]
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    w = 0.16
    for j, m in enumerate(MODELS):
        for i, rec in enumerate(recs):
            if m in rec["models"]:
                mu, sd = mean_sd(rec, m)
                bar(ax, i + (j - 2) * w, mu, sd, m, w, label=(i == 7 or (m != "esn" and i == 0)))
    names = {"xxz": "XXZ", "ising": "Ising"}
    ax.set_xticks(range(len(CELLS)))
    ax.set_xticklabels([f"{names[r]}\n{'Z/ZZ' if ro == 'ZZ' else '+X/XX'}" for _, r, ro, _ in CELLS])
    ax.axvline(3.5, color="black", linewidth=0.6)
    ax.text(1.5, 0.83, "window (reset memory)", ha="center")
    ax.text(5.5, 0.83, "recurrent memory", ha="center")
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("test NRMSE (one step)")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(LABEL[m]) for m in MODELS]
    ax.legend([handles[k] for k in order], [labels[k] for k in order], ncol=5, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), frameon=False)
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig_cells.pdf")


def fig_convergence(results: Path):
    rec = load(results, REFINE)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for m, ls in (("qrc", "-"), ("random_features", "--")):
        t = rec["models"][m]["tuning"]
        y = t["best_so_far"]
        start = t["n_trials"] - t["budget"]["refine_evals"]
        ax.plot(np.arange(1, len(y) + 1), y, ls, color="black", linewidth=1.0, label=LABEL[m])
        ax.axvline(start + 0.5, color="0.5", linestyle=ls, linewidth=0.8)
        ax.text(start - 3, 0.43 if m == "qrc" else 0.41, f"SPSA\n({'QRC' if m == 'qrc' else 'RF'})",
                ha="right", fontsize=7)
    ax.set_xlabel("trial")
    ax.set_ylabel("best validation NRMSE so far")
    ax.legend(frameon=False, loc="upper center")
    fig.tight_layout()
    fig.savefig(HERE / "figures" / "fig_convergence.pdf")


def fig_shots(results: Path):
    by = {(k, r, ro): c for k, r, ro, c in CELLS}
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.4), sharey=True)
    for ax, kind in zip(axes, ("window", "recurrent")):
        ex, sh = load(results, EXACT[kind]), load(results, by[(kind, "xxz", "ZZ")])
        models = [m for m in MODELS if m in ex["models"]]
        w = 0.38
        for i, m in enumerate(models):
            for k, (rec, face, hatch, lab) in enumerate(((ex, "1.0", "", "exact expectations"),
                                                         (sh, "0.45", "///", "12288 shots"))):
                mu, sd = mean_sd(rec, m)
                ax.bar(i + (k - 0.5) * w, mu, w, yerr=sd, color=face, hatch=hatch, edgecolor="black",
                       linewidth=0.6, error_kw={"elinewidth": 0.6, "capsize": 1.5}, label=lab if i == 0 else None)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([LABEL[m].replace("Random features", "Random\nfeatures") for m in models])
        ax.set_title(f"{kind}, XXZ, Z/ZZ", fontsize=9)
    axes[0].set_ylabel("test NRMSE (one step)")
    axes[0].set_ylim(0, 0.85)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(HERE / "figures" / "fig_shots.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    results = ap.parse_args().results
    (HERE / "figures").mkdir(exist_ok=True)
    fig_cells(results)
    fig_convergence(results)
    fig_shots(results)
    print("wrote", sorted(p.name for p in (HERE / "figures").glob("*.pdf")))


if __name__ == "__main__":
    main()
