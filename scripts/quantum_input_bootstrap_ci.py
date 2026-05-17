#!/usr/bin/env python3
"""
quantum_input_bootstrap_ci.py - Bootstrap CIs and paired comparison
test for the paper-4 quantum-input experiment.  Reads existing
summary.json files (no new compute) and produces:

  * Per-method 95% bootstrap CI on test NRMSE (over the 5 seeds).
  * Paired bootstrap of (TFIM_QRC - RFF_matched_effdim) per seed,
    with 95% CI on the difference and a one-sided "QRC < RFF" p-value
    estimate.
  * Paired Wilcoxon (one-sided) when applicable.
  * Combined ranking table across all available result trees.

Handles the TFIM_QRC zero-variance case gracefully: if a method's
seed std is < 1e-9 the bootstrap collapses to a single point
estimate, which is reported as-is.

Outputs:
    results/quantum_input/bootstrap_ci.json (combined across trees)
    Stdout: ranked table.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent

DEFAULT_DIRS = [
    "results/quantum_input",
    "results/quantum_input_q5",
    "results/quantum_input_q9",
    "results/quantum_input_targetseed_100",
    "results/quantum_input_targetseed_200",
    "results/quantum_input_xxz",
]


def bootstrap_ci(values: np.ndarray, n_boot: int = 10_000,
                  alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    """Return (mean, lo, hi) of percentile bootstrap CI of the mean."""
    if np.std(values) < 1e-12:
        m = float(np.mean(values))
        return m, m, m
    rng = np.random.default_rng(seed)
    boots = np.array([np.mean(rng.choice(values, len(values), replace=True))
                       for _ in range(n_boot)])
    return (float(np.mean(values)),
            float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000,
                       alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap CI on the mean of (a - b) and one-sided p (fraction of
    bootstrap samples where mean(a - b) >= 0, i.e. test 'a < b')."""
    if len(a) != len(b):
        raise ValueError("paired bootstrap requires equal-length arrays")
    diff = a - b
    if np.std(diff) < 1e-12:
        m = float(np.mean(diff))
        return {"mean_diff": m, "ci_lo": m, "ci_hi": m,
                "p_one_sided_a_lt_b": 1.0 if m >= 0 else 0.0}
    rng = np.random.default_rng(seed)
    idx = np.arange(len(diff))
    boots = np.array([np.mean(diff[rng.choice(idx, len(idx), replace=True)])
                       for _ in range(n_boot)])
    return {
        "mean_diff": float(np.mean(diff)),
        "ci_lo": float(np.percentile(boots, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(boots, 100 * (1 - alpha / 2))),
        "p_one_sided_a_lt_b": float(np.mean(boots >= 0)),
    }


def analyze_dir(dir_path: Path) -> dict:
    sj = dir_path / "summary.json"
    if not sj.exists():
        return {"_missing": True, "path": str(dir_path)}
    with open(sj) as f:
        s = json.load(f)
    methods = s["methods"]
    horizons = s["args"]["horizons"]
    out = {"path": str(dir_path), "args": s["args"]}

    # Per-method CIs at each horizon
    per_method_ci = {}
    for method, runs in methods.items():
        per_method_ci[method] = {}
        for k in horizons:
            vals = np.array([r[f"k{k}_nrmse"] for r in runs])
            mean, lo, hi = bootstrap_ci(vals)
            per_method_ci[method][f"k{k}"] = {"mean": mean, "lo": lo, "hi": hi}
    out["per_method_ci"] = per_method_ci

    # Paired QRC vs RFF and ESN
    if "TFIM_QRC" in methods and "RFF_matched_effdim" in methods:
        qrc = np.array([r["k1_nrmse"] for r in methods["TFIM_QRC"]])
        rff = np.array([r["k1_nrmse"] for r in methods["RFF_matched_effdim"]])
        out["paired_TFIM_vs_RFFeff_k1"] = paired_bootstrap(qrc, rff)
        # Wilcoxon: skip if all diffs identical (zero-variance edge case)
        try:
            out["wilcoxon_TFIM_vs_RFFeff_k1_pvalue"] = float(
                stats.wilcoxon(qrc, rff, alternative="less").pvalue)
        except ValueError:
            out["wilcoxon_TFIM_vs_RFFeff_k1_pvalue"] = float("nan")

    if "SCRAM_QRC" in methods and "RFF_matched_effdim" in methods:
        scram = np.array([r["k1_nrmse"] for r in methods["SCRAM_QRC"]])
        rff = np.array([r["k1_nrmse"] for r in methods["RFF_matched_effdim"]])
        out["paired_SCRAM_vs_RFFeff_k1"] = paired_bootstrap(scram, rff)
        try:
            out["wilcoxon_SCRAM_vs_RFFeff_k1_pvalue"] = float(
                stats.wilcoxon(scram, rff, alternative="less").pvalue)
        except ValueError:
            out["wilcoxon_SCRAM_vs_RFFeff_k1_pvalue"] = float("nan")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=DEFAULT_DIRS)
    p.add_argument("--out", default="results/quantum_input/bootstrap_ci.json")
    args = p.parse_args()

    combined = {}
    for d in args.dirs:
        path = ROOT / d
        result = analyze_dir(path)
        combined[d] = result

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(combined, f, indent=2)

    # ----- Print ranked table -----
    print("\n=== Per-cell summary  ===")
    print(f"{'cell':<50}  {'TFIM_QRC':<22}  {'RFF_eff':<22}  diff_QRC_minus_RFF  one-sided_p")
    for d, r in combined.items():
        if r.get("_missing"):
            print(f"{d:<50}  [missing]")
            continue
        if "TFIM_QRC" in r["per_method_ci"] and "RFF_matched_effdim" in r["per_method_ci"]:
            q = r["per_method_ci"]["TFIM_QRC"]["k1"]
            f_ = r["per_method_ci"]["RFF_matched_effdim"]["k1"]
            paired = r.get("paired_TFIM_vs_RFFeff_k1", {})
            md = paired.get("mean_diff", float("nan"))
            pv = paired.get("p_one_sided_a_lt_b", float("nan"))
            print(f"{d:<50}  "
                  f"{q['mean']:.3f}[{q['lo']:.3f},{q['hi']:.3f}]  "
                  f"{f_['mean']:.3f}[{f_['lo']:.3f},{f_['hi']:.3f}]  "
                  f"{md:+.3f}             {pv:.4f}")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
