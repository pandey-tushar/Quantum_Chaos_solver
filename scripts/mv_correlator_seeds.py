#!/usr/bin/env python3
"""mv_correlator_seeds.py - multi-seed replication of Case I phase3 concat
ablation. Reuses phase3() from mv_correlator_qrc.py (same settings: obs_noise
0.1, henon, q=8) across a grid of data x reservoir seeds, for couplings
c in {0.0, 0.1, 0.2}. Aggregates h=1 NRMSE for Poly2, QRC, and the
Poly2(+)QRC concatenation, plus the per-run cat-vs-Poly2 relative delta.

Writes one phase3 JSON per (c, data_seed, res_seed) into results/mv_correlator/
(same naming as the existing single-seed files) and an aggregate summary
results/mv_correlator/caseI_multiseed_summary.json with mean/std across seeds.
All numbers are read back from the per-run config returned by phase3().
"""
from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import mv_correlator_qrc as mv

ROOT = Path(__file__).parent.parent
OUT = ROOT / "results" / "mv_correlator"

# Five (data_seed, reservoir_seed) combinations per coupling. We vary BOTH
# seeds to sample the data x reservoir space rather than holding one fixed.
SEED_PAIRS = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
COUPLINGS = [0.0, 0.1, 0.2]
HORIZON = 1


def run():
    summary = {"settings": {"obs_noise": 0.1, "dataset": "henon", "q": 8,
                            "n_steps": 800, "horizon": HORIZON,
                            "seed_pairs": SEED_PAIRS,
                            "n_combos_per_c": len(SEED_PAIRS)},
               "per_coupling": {}}
    for c in COUPLINGS:
        poly, qrc, cat, delta = [], [], [], []
        for ds, rs in SEED_PAIRS:
            if True:
                out = mv.phase3(coupling=c, horizons=(HORIZON,), obs_noise=0.1,
                                data_seed=ds, res_seed=rs, dataset="henon")
                r = out["results_per_horizon"][str(HORIZON)]
                p = r["Poly2"]["nrmse"]; q = r["QRC"]["nrmse"]; ct = r["Poly2+QRC"]["nrmse"]
                poly.append(p); qrc.append(q); cat.append(ct)
                delta.append((p - ct) / p * 100.0)  # +%: cat lower than Poly2
        poly, qrc, cat, delta = map(np.array, (poly, qrc, cat, delta))
        summary["per_coupling"][str(c)] = {
            "n": len(poly),
            "QRC_mean": float(poly.size and qrc.mean()), "QRC_std": float(qrc.std(ddof=1)),
            "Poly2_mean": float(poly.mean()), "Poly2_std": float(poly.std(ddof=1)),
            "cat_mean": float(cat.mean()), "cat_std": float(cat.std(ddof=1)),
            "cat_vs_Poly2_pct_mean": float(delta.mean()),
            "cat_vs_Poly2_pct_std": float(delta.std(ddof=1)),
            "QRC_all": qrc.tolist(), "Poly2_all": poly.tolist(),
            "cat_all": cat.tolist(), "delta_pct_all": delta.tolist(),
        }
        print(f"\n=== c={c} (n={len(poly)}) ===")
        print(f"  QRC   {qrc.mean():.3f} +/- {qrc.std(ddof=1):.3f}")
        print(f"  Poly2 {poly.mean():.3f} +/- {poly.std(ddof=1):.3f}")
        print(f"  cat   {cat.mean():.3f} +/- {cat.std(ddof=1):.3f}")
        print(f"  cat vs Poly2  {delta.mean():+.2f}% +/- {delta.std(ddof=1):.2f}%")

    fp = OUT / "caseI_multiseed_summary.json"
    fp.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved {fp}  (sha {hashlib.sha256(fp.read_bytes()).hexdigest()[:12]})")


if __name__ == "__main__":
    run()
