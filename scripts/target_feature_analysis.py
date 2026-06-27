#!/usr/bin/env python3
"""
target_feature_analysis.py - Compute structural features of target
Hamiltonians and correlate them with the QRC-vs-RFF NRMSE gap from
existing paper-5 results.  Goal: identify which target-system
property predicts when QRC beats classical reservoirs (option 2 from
the paper-5 robustness investigation).

Features per target Hamiltonian:
  - mean_r : mean level-spacing ratio <r> (Wigner-Dyson = 0.5295,
             Poisson = 0.3863).  Probe of quantum chaos.
  - mid_band_S : mean half-chain entanglement entropy of eigenstates
                 in the central energy band.  Probe of eigenstate
                 thermalization.
  - ipr_initial : energy-basis inverse participation ratio of the
                  initial state |+>^n.  High IPR = state lives in
                  few eigenstates (less rich dynamics).
  - bandwidth : E_max - E_min, the spectral bandwidth.
  - spectral_density_peakedness : kurtosis of the eigenvalue
                                   distribution.

Reads summary.json from each results/quantum_input*/ directory,
extracts the target config from `args`, rebuilds the Hamiltonian
to compute features, then reports a scatter table.

Outputs:
    results/quantum_input/target_features.json
    results/quantum_input/feature_vs_qrc_gap.png   (correlation scatter)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Reuse Hamiltonian builders
sys.path.insert(0, str(ROOT / "scripts"))
from run_quantum_input_experiment import build_target_hamiltonian


# ---------------------------------------------------------------------------
# Structural features
# ---------------------------------------------------------------------------

def level_spacing_ratio(eigs: np.ndarray) -> float:
    """<r> = mean of min(s_{n+1}, s_n) / max(s_{n+1}, s_n)."""
    eigs = np.sort(eigs)
    s = np.diff(eigs)
    if np.any(s <= 0):
        s = s[s > 1e-12]
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    return float(np.mean(r))


def half_chain_entanglement_entropy(state: np.ndarray, n: int) -> float:
    """von Neumann entropy of half-chain reduced density matrix."""
    nA = n // 2
    nB = n - nA
    psi = state.reshape((2 ** nA, 2 ** nB))
    # SVD = Schmidt decomposition; squared singular values = eigenvalues of rho_A
    svals = np.linalg.svd(psi, compute_uv=False)
    p = svals ** 2
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


def mid_band_entanglement(eigvecs: np.ndarray, n: int,
                            frac: float = 0.2) -> float:
    """Mean half-chain entropy over a central band of eigenstates."""
    dim = eigvecs.shape[1]
    n_keep = max(int(dim * frac), 4)
    lo = (dim - n_keep) // 2
    hi = lo + n_keep
    entropies = [half_chain_entanglement_entropy(eigvecs[:, i], n)
                  for i in range(lo, hi)]
    return float(np.mean(entropies))


def energy_basis_ipr(psi0: np.ndarray, eigvecs: np.ndarray) -> float:
    """Inverse participation ratio of psi0 in the energy eigenbasis."""
    coeffs = eigvecs.conj().T @ psi0
    probs = np.abs(coeffs) ** 2
    return float(np.sum(probs ** 2))


def compute_features(H: np.ndarray, n: int) -> dict:
    eigs, vecs = np.linalg.eigh(H)
    # |+>^n initial state
    psi0 = np.ones(2 ** n, dtype=complex) / np.sqrt(2 ** n)
    return {
        "mean_r": level_spacing_ratio(eigs),
        "mid_band_S": mid_band_entanglement(vecs, n, frac=0.2),
        "ipr_initial": energy_basis_ipr(psi0, vecs),
        "bandwidth": float(eigs.max() - eigs.min()),
        "kurtosis_eigs": float(stats.kurtosis(eigs)),
        "n_eigs": len(eigs),
    }


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_all(dirs: list[str]) -> list[dict]:
    rows = []
    for d in dirs:
        sj = ROOT / d / "summary.json"
        if not sj.exists():
            continue
        with open(sj) as f:
            s = json.load(f)
        args_d = s["args"]
        n_target = args_d.get("n_target", 4)
        target_system = args_d.get("target_system", "heisenberg")
        target_seed = args_d.get("target_seed", 42)
        xxz_delta = args_d.get("xxz_delta", 0.5)

        H = build_target_hamiltonian(target_system, n_target, target_seed,
                                       xxz_delta=xxz_delta)
        feats = compute_features(H, n_target)

        # QRC vs RFF gap at k=1, mean over seeds
        methods = s["methods"]
        qrc_vals = [r["k1_nrmse"] for r in methods["TFIM_QRC"]]
        rff_vals = [r["k1_nrmse"] for r in methods["RFF_matched_effdim"]]
        qrc_mean = float(np.mean(qrc_vals))
        rff_mean = float(np.mean(rff_vals))
        gap = qrc_mean - rff_mean   # negative = QRC wins

        rows.append({
            "dir": d,
            "target_system": target_system,
            "target_seed": target_seed,
            "xxz_delta": xxz_delta if target_system == "xxz" else None,
            "n_target": n_target,
            "qrc_k1": qrc_mean,
            "rff_k1": rff_mean,
            "gap_qrc_minus_rff": gap,
            "qrc_wins": gap < 0,
            "features": feats,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dirs", nargs="+", default=[
        "results/quantum_input",
        "results/quantum_input_q5",
        "results/quantum_input_q9",
        "results/quantum_input_targetseed_100",
        "results/quantum_input_targetseed_200",
        "results/quantum_input_xxz",
    ])
    p.add_argument("--out-json", default="results/quantum_input/target_features.json")
    p.add_argument("--out-fig",  default="results/quantum_input/feature_vs_qrc_gap.png")
    args = p.parse_args()

    rows = analyze_all(args.dirs)
    if not rows:
        print("No summary files found.")
        sys.exit(1)

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n=== Per-target features and QRC-RFF gap ({len(rows)} targets) ===")
    print(f"{'target':<45}  {'qrc_k1':>7}  {'rff_k1':>7}  {'gap':>7}  "
           f"{'<r>':>5}  {'S_mid':>6}  {'IPR_in':>6}  {'BW':>5}")
    for r in rows:
        f = r["features"]
        tag = f"{r['target_system']:>4} seed={r['target_seed']}"
        if r["target_system"] == "xxz":
            tag += f" d={r['xxz_delta']}"
        print(f"{r['dir']:<45}  {r['qrc_k1']:>7.3f}  {r['rff_k1']:>7.3f}  "
              f"{r['gap_qrc_minus_rff']:>+7.3f}  "
              f"{f['mean_r']:>5.3f}  {f['mid_band_S']:>6.3f}  "
              f"{f['ipr_initial']:>6.4f}  {f['bandwidth']:>5.2f}")

    # Pearson correlations (gap vs each feature, with sign of gap convention:
    # negative gap = QRC wins, so positive correlation means "feature increases
    # when QRC loses").
    feat_names = ["mean_r", "mid_band_S", "ipr_initial", "bandwidth",
                   "kurtosis_eigs"]
    gaps = np.array([r["gap_qrc_minus_rff"] for r in rows])
    print(f"\n=== Pearson r (feature, gap) — neg r = feature predicts QRC win ===")
    for fn in feat_names:
        vals = np.array([r["features"][fn] for r in rows])
        if np.std(vals) < 1e-8 or np.std(gaps) < 1e-8:
            print(f"  {fn:<15} : insufficient variance")
            continue
        rho = float(np.corrcoef(vals, gaps)[0, 1])
        print(f"  {fn:<15} : r = {rho:+.3f}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        for ax, fn in zip(axes.flat, feat_names):
            vals = [r["features"][fn] for r in rows]
            ax.scatter(vals, gaps, c=["#0F4C5C" if g < 0 else "#9A4836" for g in gaps],
                        s=80, edgecolor="white", linewidth=0.7)
            ax.axhline(0, color="#888", lw=0.7, ls="--")
            ax.set_xlabel(fn)
            ax.set_ylabel("QRC − RFF NRMSE  (negative = QRC wins)")
            ax.grid(True, alpha=0.3)
            ax.set_title(fn)
        axes.flat[-1].axis("off")
        fig.suptitle("Target-Hamiltonian features vs QRC-RFF gap")
        fig.tight_layout()
        fig.savefig(args.out_fig, dpi=150)
        print(f"\nSaved {args.out_fig}")
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
