#!/usr/bin/env python3
"""
audit_directional_results.py - Directional audit of every Wilcoxon claim
in the SWE Block A, Block B, and QG hybrid result trees.

The Wilcoxon signed-rank test returns a two-sided p-value. A small
p-value tells you the paired difference is statistically distinguishable
from zero, but says nothing about which method is better. To know that,
you need to look at the per-seed direction: how many seeds favour A vs
B, and the means of the per-seed values.

This script reads every results JSON, recomputes the comparison
explicitly with direction, and prints a single table per result file.
It is intended as a check on what was claimed in earlier reports.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent


def _audit_pair(name_a: str, name_b: str, a: np.ndarray, b: np.ndarray) -> dict:
    """Return dict with direction, p-value, mean-of-each, and per-seed wins."""
    if len(a) != len(b):
        raise ValueError("paired arrays must have same length")
    n = len(a)
    a_wins = int(np.sum(a < b))
    try:
        _, pval = stats.wilcoxon(a, b)
    except ValueError:
        pval = float("nan")
    if a.mean() < b.mean():
        better = name_a
    elif b.mean() < a.mean():
        better = name_b
    else:
        better = "tied"
    return {
        "name_a": name_a, "name_b": name_b,
        "n": n,
        "a_mean": float(a.mean()), "b_mean": float(b.mean()),
        "a_wins": a_wins, "b_wins": n - a_wins,
        "lower_mean": better,
        "wilcoxon_p": float(pval),
    }


def _print_table(rows: list[dict], header: str):
    print("\n" + "=" * 110)
    print(header)
    print("=" * 110)
    print(f"{'A':<14}  {'B':<14}  {'A mean':>10}  {'B mean':>10}  "
          f"{'A wins':>6}  {'B wins':>6}  {'p':>8}  {'lower mean':>14}")
    print("-" * 110)
    for r in rows:
        print(f"{r['name_a']:<14}  {r['name_b']:<14}  "
              f"{r['a_mean']:>10.3e}  {r['b_mean']:>10.3e}  "
              f"{r['a_wins']:>6}  {r['b_wins']:>6}  "
              f"{r['wilcoxon_p']:>8.4f}  {r['lower_mean']:>14}")


def audit_swe_block_a():
    rows = []
    for q in [5, 6, 7, 8]:
        path = ROOT / f"results/swe_qrc/scaling_q{q}_K{q}/swe_qrc_results.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        qrc  = np.array([r["qrc"]["teacher_mse"]                            for r in d["per_seed"]])
        esn  = np.array([r["esn"]["teacher_mse"]                            for r in d["per_seed"]])
        pers = np.array([r["persistence"]["closed_loop_per_lead_mse"][0]    for r in d["per_seed"]])
        rows.append({"label": f"SWE_BA q=K={q} (feat dim {2**q}) - teacher MSE",
                       "pairs": [
                           _audit_pair("qrc",  "esn",  qrc,  esn),
                           _audit_pair("qrc",  "pers", qrc,  pers),
                       ]})
    return rows


def audit_swe_block_b():
    rows = []
    for q, path in [(5, "results/swe_hybrid/swe_hybrid_results.json"),
                       (6, "results/swe_hybrid/scaling_q6_K6/swe_hybrid_results.json"),
                       (7, "results/swe_hybrid/scaling_q7_K7/swe_hybrid_results.json"),
                       (8, "results/swe_hybrid/scaling_q8_K8/swe_hybrid_results.json")]:
        full = ROOT / path
        if not full.exists():
            continue
        with open(full) as f:
            d = json.load(f)
        per_seed = d["per_seed"]
        # Lead-1 closed-loop MSE
        phys_l1 = np.array([r["physics"]["closed_loop_per_lead_mse"][0] for r in per_seed])
        qrc_l1  = np.array([r["qrc"]["closed_loop_per_lead_mse"][0]     for r in per_seed])
        esn_l1  = np.array([r["esn"]["closed_loop_per_lead_mse"][0]     for r in per_seed])
        # Mean across full rollout
        phys_m = np.array([np.mean(r["physics"]["closed_loop_per_lead_mse"]) for r in per_seed])
        qrc_m  = np.array([np.mean(r["qrc"]["closed_loop_per_lead_mse"])     for r in per_seed])
        esn_m  = np.array([np.mean(r["esn"]["closed_loop_per_lead_mse"])     for r in per_seed])
        rows.append({"label": f"SWE_BB q=K={q} - lead-1 closed loop",
                       "pairs": [
                           _audit_pair("qrc",  "physics", qrc_l1, phys_l1),
                           _audit_pair("qrc",  "esn",     qrc_l1, esn_l1),
                       ]})
        rows.append({"label": f"SWE_BB q=K={q} - rollout-mean closed loop",
                       "pairs": [
                           _audit_pair("qrc",  "physics", qrc_m,  phys_m),
                           _audit_pair("qrc",  "esn",     qrc_m,  esn_m),
                       ]})
    return rows


def audit_qg_baselines():
    rows = []
    path = ROOT / "results/qg_hybrid/baselines/qg_baselines_results.json"
    if not path.exists():
        return rows
    with open(path) as f:
        d = json.load(f)
    methods = ["physics", "qrc", "esn", "mlp", "linreg", "rff", "persistence"]
    per_seed = d["per_seed"]
    # Lead-1
    arr = {m: np.array([r[m]["closed_loop_per_lead_mse"][0] for r in per_seed]) for m in methods}
    pairs = []
    qrc = arr["qrc"]
    for m in methods:
        if m == "qrc":
            continue
        pairs.append(_audit_pair("qrc", m, qrc, arr[m]))
    rows.append({"label": "QG baselines - lead-1 closed loop", "pairs": pairs})
    # Rollout mean
    arrm = {m: np.array([np.mean(r[m]["closed_loop_per_lead_mse"]) for r in per_seed]) for m in methods}
    pairs = []
    qrc_m = arrm["qrc"]
    for m in methods:
        if m == "qrc":
            continue
        pairs.append(_audit_pair("qrc", m, qrc_m, arrm[m]))
    rows.append({"label": "QG baselines - rollout-mean closed loop", "pairs": pairs})
    return rows


def main():
    print("DIRECTIONAL AUDIT: explicit per-seed wins, Wilcoxon p, and which method has lower mean.")
    print("'A wins' counts seeds where A's metric is strictly less than B's.")
    print("'lower mean' is the method whose mean across seeds is smaller (= better, since these are MSEs).")

    for row in audit_swe_block_a():
        _print_table(row["pairs"], row["label"])
    for row in audit_swe_block_b():
        _print_table(row["pairs"], row["label"])
    for row in audit_qg_baselines():
        _print_table(row["pairs"], row["label"])


if __name__ == "__main__":
    main()
