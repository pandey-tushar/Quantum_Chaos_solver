#!/usr/bin/env python3
"""
summarize_shot_sweep.py - Aggregate shot-realistic QRC vs classical
results across q in {5, 7, 9, 11, 13, 15} and produce a single
publication-quality plot + table.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parent.parent
RESULTS_ROOT = ROOT / "results"


def load(q: int) -> dict | None:
    f = RESULTS_ROOT / f"quantum_input_shots_q{q}" / "summary.json"
    if not f.exists():
        return None
    with open(f) as fh:
        return json.load(fh)


def main():
    qs = [9, 11]   # honest reservoir sizes (>= 9 qubits)
    rows = []
    for q in qs:
        s = load(q)
        if s is None:
            continue
        m = s["per_method_means"]
        D = q * (q + 1) // 2
        rows.append({
            "q": q, "D": D,
            "tfim_k1": m["TFIM_QRC"]["k1_mean"],
            "tfim_k1_std": m["TFIM_QRC"]["k1_std"],
            "scram_k1": m["SCRAM_QRC"]["k1_mean"],
            "scram_k1_std": m["SCRAM_QRC"]["k1_std"],
            "esn_k1": m["ESN_matched"]["k1_mean"],
            "rff_k1": m["RFF_matched_dim"]["k1_mean"],
            "rff_k1_std": m["RFF_matched_dim"]["k1_std"],
            "rffeff_k1": m["RFF_matched_effdim"]["k1_mean"],
        })

    print(f"\n=== Shot-realistic QRC vs classical (k=1 horizon, 3 seeds) ===")
    print(f"{'q':>3}  {'D':>4}  {'TFIM_QRC':>14}  {'SCRAM_QRC':>14}  "
            f"{'ESN':>8}  {'RFF(D)':>14}  {'RFF(D_eff)':>10}")
    for r in rows:
        print(f"{r['q']:>3}  {r['D']:>4}  "
                f"{r['tfim_k1']:.3f}+/-{r['tfim_k1_std']:.3f}  "
                f"{r['scram_k1']:.3f}+/-{r['scram_k1_std']:.3f}  "
                f"{r['esn_k1']:>8.3f}  "
                f"{r['rff_k1']:.3f}+/-{r['rff_k1_std']:.3f}  "
                f"{r['rffeff_k1']:>10.3f}")

    out_json = RESULTS_ROOT / "quantum_input_shots_summary.json"
    with open(out_json, "w") as fh:
        json.dump(rows, fh, indent=2)
    print(f"\nSaved {out_json}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        qs_x = [r["q"] for r in rows]
        ax.errorbar(qs_x, [r["tfim_k1"] for r in rows],
                     yerr=[r["tfim_k1_std"] for r in rows],
                     marker="o", lw=2, ms=9, capsize=4,
                     color="#0F4C5C", label="TFIM QRC (shots, local Paulis)",
                     markeredgecolor="white", markeredgewidth=0.7)
        ax.errorbar(qs_x, [r["scram_k1"] for r in rows],
                     yerr=[r["scram_k1_std"] for r in rows],
                     marker="s", lw=2, ms=9, capsize=4,
                     color="#1E7B8E", label="SCRAM QRC (shots, local Paulis)",
                     markeredgecolor="white", markeredgewidth=0.7)
        ax.plot(qs_x, [r["esn_k1"] for r in rows],
                 marker="^", lw=2, ms=9,
                 color="#9A4836", label="ESN (matched N)")
        ax.errorbar(qs_x, [r["rff_k1"] for r in rows],
                     yerr=[r["rff_k1_std"] for r in rows],
                     marker="D", lw=2, ms=9, capsize=4,
                     color="#5E7C6B", label="RFF (matched D = q(q+1)/2)",
                     markeredgecolor="white", markeredgewidth=0.7)
        ax.set_xlabel("Reservoir size q (qubits)")
        ax.set_ylabel("Test NRMSE @ k=1  (lower is better)")
        ax.set_title("Hardware-realistic QRC vs classical, 4-qubit Heisenberg target\n"
                       "M = q*1024 shots/step, features = <Z_i> + <Z_i Z_j>")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=10)
        ax.axhline(1.0, color="#888", lw=0.7, ls=":")
        fig.tight_layout()
        out_fig = RESULTS_ROOT / "quantum_input_shots_summary.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
