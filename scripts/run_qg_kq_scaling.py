#!/usr/bin/env python3
"""
run_qg_kq_scaling.py - Full method panel on QG hybrid for any (n_qubits, K) pair.

Mirrors the SWE Block A K=q matched-complexity scaling experiment in the QG
residual-correction (Block B) setting. For each (q, K) the panel includes:

    * physics       - no correction baseline
    * persistence   - next state = current state
    * linreg        - ridge regression on windowed POD coefficients
    * mlp           - tiny ReLU MLP on the same features (matched param
                       count to QRC's readout, capped)
    * rff           - Random Fourier Features reservoir at matched
                       feature dim 2^q
    * esn           - Echo State Network at matched feature dim 2^q
    * qrc           - quantum reservoir at q qubits

All methods predict the per-step residual (truth(t+1) - phys_one_step(truth(t)))
in POD space and are evaluated identically (closed-loop rollout, lead-1 MSE,
rollout-mean MSE, win counts vs physics-only at each lead).

CLI:
    python scripts/run_qg_kq_scaling.py
        # runs (q,K) in {(5,5),(6,6),(7,7),(8,8)}, default
    python scripts/run_qg_kq_scaling.py --qs 5 6 7

Outputs (per cell):
    results/qg_hybrid/scaling/q{q}_K{K}/results.json
Aggregate:
    results/qg_hybrid/scaling/scaling_summary.json
    results/qg_hybrid/scaling/skill_at_lead1_vs_kq.png
    results/qg_hybrid/scaling/skill_curves_grid.png
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from pod_reduction import PODReducer
from run_qg_hybrid import (   # type: ignore
    _grid_dict_from_npz, _spectral_grid, _physics_step_block,
    _stack_q_psi,
)
from run_swe_qrc import (    # type: ignore
    make_qrc_extractor, _ESN, _ridge_solve,
)
from run_qg_baselines import _train_mlp, _rff_extractor   # type: ignore


# ---------------------------------------------------------------------------
# Single-cell experiment
# ---------------------------------------------------------------------------

def _run_cell(q: int, K: int, data, grid, dt, dt_save, n_phys_substeps,
               rhs_phys, ref_truth_snaps, seed_truth_snaps, seeds_in_file,
               window: int = 5, train_frac: float = 0.7, ridge_alpha: float = 1.0):
    feature_dim = 2 ** q
    pod = PODReducer(n_modes=K).fit(ref_truth_snaps)

    per_seed = []
    for s in seeds_in_file:
        idx = seeds_in_file.index(s)
        truth_snaps = seed_truth_snaps[idx]                  # (n_t, 2, Nx, Ny)
        n_t = truth_snaps.shape[0]
        train_end = int(train_frac * n_t)
        n_test = n_t - train_end
        truth_coeffs = pod.transform(truth_snaps)            # (n_t, K)

        # one-step physics from training truth snapshots
        phys_one_pairs = []
        for t in range(train_end):
            q_next = _physics_step_block(truth_snaps[t, 0], n_phys_substeps,
                                          dt, grid, rhs_phys)
            phys_one_pairs.append(_stack_q_psi(q_next, grid))
        phys_one_grid = np.stack(phys_one_pairs, axis=0)
        phys_one_coeffs = pod.transform(phys_one_grid)
        target_residual = truth_coeffs[1:train_end + 1] - phys_one_coeffs

        # windowed feature inputs (raw POD coeffs, used by linreg + MLP)
        F_lin, S_res = [], []
        for i in range(train_end - window):
            F_lin.append(truth_coeffs[i:i + window].flatten())
            S_res.append(target_residual[i + window - 1])
        F_lin = np.asarray(F_lin)
        S_res = np.asarray(S_res)
        in_dim = F_lin.shape[1]

        # QRC features (windowed quantum-statevector probabilities)
        train_coeffs = truth_coeffs[:train_end]
        scale_lo = train_coeffs.min(axis=0)
        scale_hi = train_coeffs.max(axis=0)
        margin = 0.1 * (scale_hi - scale_lo + 1e-12)
        scale_lo, scale_hi = scale_lo - margin, scale_hi + margin
        qrc_extract = make_qrc_extractor(q, 2, s,
                                          scale_lo=scale_lo, scale_hi=scale_hi,
                                          encoding="global")
        qrc_feats_all = np.array([qrc_extract(c) for c in truth_coeffs])
        F_qrc = []
        for i in range(train_end - window):
            F_qrc.append(qrc_feats_all[i:i + window].flatten())
        F_qrc = np.asarray(F_qrc)
        W_qrc = _ridge_solve(F_qrc, S_res, ridge_alpha)

        # ESN features
        esn = _ESN(n_input=K, n_reservoir=feature_dim, n_output=K,
                    ridge_alpha=ridge_alpha, seed=s)
        esn_acts_all = esn._activations(truth_coeffs)
        F_esn = []
        for i in range(train_end - window):
            F_esn.append(esn_acts_all[i:i + window].flatten())
        F_esn = np.asarray(F_esn)
        W_esn = _ridge_solve(F_esn, S_res, ridge_alpha)

        # RFF features (fixed nonlinear projection of raw POD coeff)
        rff_extract = _rff_extractor(in_dim=K, n_features=feature_dim,
                                       sigma=float(np.std(truth_coeffs)),
                                       seed=s)
        rff_feats_all = np.array([rff_extract(c) for c in truth_coeffs])
        F_rff = []
        for i in range(train_end - window):
            F_rff.append(rff_feats_all[i:i + window].flatten())
        F_rff = np.asarray(F_rff)
        W_rff = _ridge_solve(F_rff, S_res, ridge_alpha)

        # Linear regression on windowed POD coeffs
        W_lin = _ridge_solve(F_lin, S_res, ridge_alpha)

        # MLP at matched-param-count to QRC's readout (capped)
        # QRC W is (window * 2^q) x K; total = window * 2^q * K params.
        # MLP(hidden=h) total = (in_dim+1)*h + (K+1)*h ... wait simpler:
        #   in_dim*h + h + h*K + K = h*(in_dim + K + 1) + K
        # Solve for h s.t. MLP params ~= QRC params:
        target_params = window * feature_dim * K
        h = max(8, min(96, int((target_params - K) / (in_dim + K + 1))))
        mlp_predict, _, _ = _train_mlp(F_lin, S_res, hidden=h,
                                         weight_decay=1e-3, n_iter=2000, seed=s)

        # ----- Closed-loop rollouts -----
        # Rollout protocol (NeuralGCM-style):
        # The full-grid q_state evolves under the physics solver; the learned
        # correction is applied as an additive perturbation in grid space,
        # NOT as a re-projection of the state onto the K-mode POD subspace.
        # Replacing q_state with pod.inverse_transform(new_coeffs)[0] at each
        # step is the classical "mode-truncation instability" of POD ROMs --
        # the trajectory becomes constrained to a low-rank subspace that is
        # not closed under nonlinear advection, and small errors compound
        # exponentially. See the closure-modeling literature for context.
        # We additionally clip |q_state| to a generous safety bound (10x the
        # training-data std of vorticity) as a last-resort guard against
        # transient catastrophic blow-up; this should rarely trigger.
        truth_test_coeffs = truth_coeffs[train_end:train_end + n_test]
        q_train_std = float(np.std(truth_snaps[:train_end, 0]))
        q_clip = 10.0 * q_train_std

        def _correction_to_grid_q(correction_pod):
            """Map a K-dim POD-space correction to its grid-space delta on the
            q channel (no mean term added). Inverts the linear part of
            pod.inverse_transform: corr_grid_flat = (c @ modes.T) * scales."""
            corr_grid_flat = (correction_pod @ pod.modes_.T) * pod.scales_
            corr_field = corr_grid_flat.reshape(pod.field_shape_)
            return corr_field[0]   # q channel only

        def rollout(corrector):
            q_state = truth_snaps[train_end - 1, 0].copy()
            coeff_buf = list(truth_coeffs[train_end - window:train_end])
            qrc_buf = list(qrc_feats_all[train_end - window:train_end])
            esn_r = esn_acts_all[train_end - 1].copy()
            esn_buf = list(esn_acts_all[train_end - window:train_end])
            rff_buf = list(rff_feats_all[train_end - window:train_end])
            preds = []
            for _ in range(n_test):
                q_next = _physics_step_block(q_state, n_phys_substeps, dt, grid, rhs_phys)
                phys_pair = _stack_q_psi(q_next, grid)
                phys_coeffs = pod.transform(phys_pair[None, ...])[0]

                if corrector == "physics":
                    correction = np.zeros(K)
                elif corrector == "persistence":
                    # next = current  =>  correction = current - physics
                    correction = coeff_buf[-1] - phys_coeffs
                elif corrector == "linreg":
                    correction = np.concatenate(coeff_buf[-window:]) @ W_lin
                elif corrector == "mlp":
                    flat = np.concatenate(coeff_buf[-window:])[None, :]
                    correction = mlp_predict(flat)[0]
                elif corrector == "qrc":
                    correction = np.concatenate(qrc_buf[-window:]) @ W_qrc
                elif corrector == "esn":
                    correction = np.concatenate(esn_buf[-window:]) @ W_esn
                elif corrector == "rff":
                    correction = np.concatenate(rff_buf[-window:]) @ W_rff
                else:
                    raise ValueError(corrector)

                # Apply correction additively in grid space; do NOT replace
                # q_state by a low-rank reconstruction.
                if corrector == "physics":
                    q_state = q_next
                else:
                    q_state = q_next + _correction_to_grid_q(correction)
                # Safety clip: prevents transient catastrophic blow-up from
                # contaminating downstream FFTs. Should rarely if ever trigger.
                np.clip(q_state, -q_clip, q_clip, out=q_state)

                # Record the OBSERVED POD coefficients of the actual grid state.
                # (For the additive-correction path this equals phys_coeffs +
                # correction up to the field-shape projection; for physics it
                # equals phys_coeffs. We compute it via pod.transform to avoid
                # any subtle mismatch and to also mirror what an online observer
                # would measure.)
                snap_pair = _stack_q_psi(q_state, grid)
                observed_coeffs = pod.transform(snap_pair[None, ...])[0]
                preds.append(observed_coeffs)

                coeff_buf.append(observed_coeffs)
                qrc_buf.append(qrc_extract(observed_coeffs))
                esn_r = ((1 - esn.alpha) * esn_r
                         + esn.alpha * np.tanh(esn.W_res @ esn_r + esn.W_in @ observed_coeffs))
                esn_buf.append(esn_r.copy())
                rff_buf.append(rff_extract(observed_coeffs))
            return np.asarray(preds)

        results = {}
        for m in ["physics", "persistence", "linreg", "mlp", "qrc", "esn", "rff"]:
            preds = rollout(m)
            mse_per_lead = np.mean((preds - truth_test_coeffs) ** 2, axis=1)
            results[m] = mse_per_lead.tolist()

        per_seed.append({"seed": int(s), "mse_per_lead": results,
                           "mlp_hidden": h, "mlp_params":
                               h * (in_dim + K + 1) + K})

    # Aggregate
    methods = ["physics", "persistence", "linreg", "mlp", "qrc", "esn", "rff"]
    arrs = {m: np.array([r["mse_per_lead"][m] for r in per_seed]) for m in methods}

    # Lead-1 stats
    lead1 = {m: arrs[m][:, 0] for m in methods}
    summary = {"q": q, "K": K, "feature_dim": feature_dim,
                 "window": window, "ridge_alpha": ridge_alpha,
                 "n_seeds": len(per_seed),
                 "mlp_hidden_chosen": h, "mlp_params": h * (in_dim + K + 1) + K,
                 "qrc_readout_params": window * feature_dim * K,
                 "lead1_mean": {m: float(lead1[m].mean()) for m in methods},
                 "lead1_std":  {m: float(lead1[m].std())  for m in methods},
                 "lead1_qrc_wins_vs": {},
                 "lead1_wilcoxon_qrc_vs": {},
                 "rollout_mean_mean": {m: float(arrs[m].mean()) for m in methods},
                 }
    for m in methods:
        if m == "qrc":
            continue
        wins = int(np.sum(lead1["qrc"] < lead1[m]))
        try:
            _, p = stats.wilcoxon(lead1["qrc"], lead1[m])
        except ValueError:
            p = float("nan")
        summary["lead1_qrc_wins_vs"][m] = wins
        summary["lead1_wilcoxon_qrc_vs"][m] = float(p)

    return summary, per_seed, arrs


# ---------------------------------------------------------------------------
# Aggregate plot
# ---------------------------------------------------------------------------

def _plot_scaling(out_dir: Path, qs: list[int], summaries: dict,
                    arrs_by_q: dict, dt_h: float):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ["qrc", "esn", "rff", "linreg", "mlp", "persistence"]
    pretty = {
        "qrc":         ("QRC",                   "#0066cc", "-"),
        "esn":         ("ESN",                    "#cc4400", "--"),
        "rff":         ("RFF",                    "#aa00aa", "--"),
        "linreg":      ("Linear regression",      "#bbbb00", "--"),
        "mlp":         ("Trainable MLP",          "#000000", "--"),
        "persistence": ("Persistence",            "#666666", ":"),
    }

    # ---- Plot 1: lead-1 skill score vs (q,K) ----
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    x = np.arange(len(qs))
    for m in methods:
        y = []
        for q in qs:
            phys = summaries[q]["lead1_mean"]["physics"]
            mse  = summaries[q]["lead1_mean"][m]
            skill = 1.0 - mse / max(phys, 1e-30)
            y.append(skill)
        label, color, ls = pretty[m]
        ax.plot(x, y, color=color, lw=2.0 if m == "qrc" else 1.5,
                 marker="o", ms=8, ls=ls, label=label,
                 markeredgecolor="white", markeredgewidth=0.8)
    ax.axhline(0.0, color="#444", lw=1.0, ls="-", alpha=0.7)
    ax.text(x[-1], 0.0, "  physics-only baseline",
             fontsize=9, color="#444", va="center", ha="left")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K = q = {q}\n(feat dim {2**q})" for q in qs])
    ax.set_xlabel("Matched problem complexity   (POD modes K = qubit count q)")
    ax.set_ylabel("Skill score at lead-1 vs physics-only\n"
                   "(higher = better; 0 = same as physics)")
    ax.set_title("Lead-1 forecast skill vs problem complexity (8 seeds, QG hybrid)")
    ax.legend(loc="lower left", fontsize=9, frameon=True, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out1 = out_dir / "skill_at_lead1_vs_kq.png"
    fig.savefig(out1, dpi=150)
    print(f"Saved {out1}")

    # ---- Plot 2: 2x2 grid of skill-vs-leadtime curves ----
    n_q = len(qs)
    ncols = 2
    nrows = (n_q + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows),
                                sharex=True, sharey=True)
    axes = np.atleast_2d(axes).flatten()

    for i, q in enumerate(qs):
        ax = axes[i]
        arrs = arrs_by_q[q]
        n_lead = arrs["physics"].shape[1]
        leads = np.arange(1, n_lead + 1)
        eps = 1e-30
        for m in [mm for mm in methods if mm != "mlp"]:  # MLP often blows up the y-axis
            label, color, ls = pretty[m]
            skill = 1.0 - arrs[m] / (arrs["physics"] + eps)
            mean = np.mean(skill, axis=0)
            ax.plot(leads, mean, color=color, lw=2.0 if m == "qrc" else 1.4,
                    ls=ls, label=label)
        ax.axhline(0.0, color="#444", lw=0.8, ls="-", alpha=0.6)
        ax.set_ylim(-1.5, 1.0)
        ax.set_title(f"K = q = {q}  (feat dim {2**q})", fontsize=11)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="lower left", fontsize=8, frameon=True, ncol=1)

    # Hide unused panels
    for j in range(len(qs), len(axes)):
        axes[j].set_visible(False)

    for ax in axes[(nrows - 1) * ncols:]:
        ax.set_xlabel(f"Lead time (steps; 1 step = {dt_h:.2f} h)")
    for ax in axes[::ncols]:
        ax.set_ylabel("Skill score vs physics-only")
    fig.suptitle("Skill vs lead time, 8 seeds, by matched problem complexity\n"
                  "(MLP omitted: catastrophic divergence in closed-loop)", fontsize=11)
    fig.tight_layout()
    out2 = out_dir / "skill_curves_grid.png"
    fig.savefig(out2, dpi=150)
    print(f"Saved {out2}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str,
                    default=str(ROOT / "data" / "qg" / "qg_hybrid_data.npz"))
    p.add_argument("--qs", type=int, nargs="+", default=[5, 6, 7, 8])
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--out-dir", type=str,
                    default=str(ROOT / "results" / "qg_hybrid" / "scaling"))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data}")
    data = np.load(args.data, allow_pickle=False)
    seed_truth = data["seed_truth_snaps"]
    ref_truth  = data["ref_truth_snaps"]
    seeds_in_file = data["seeds"].tolist()
    dt = float(data["dt"]); dt_save = float(data["dt_save"])
    n_phys_substeps = max(1, int(round(dt_save / dt)))
    grid_meta = _grid_dict_from_npz(data)
    grid = _spectral_grid(grid_meta)
    rhs_phys = dict(beta=float(data["beta"]),
                     r=float(data["r_drag"]),
                     nu=float(data["nu"]),
                     p_hyper=int(data["p_hyper"]),
                     linear=bool(int(data["linear_phys"])))
    print(f"  seed_truth: {seed_truth.shape}, ref_truth: {ref_truth.shape}")

    summaries = {}
    arrs_by_q = {}
    for q in args.qs:
        K = q
        cell_dir = out_dir / f"q{q}_K{K}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n========== q = K = {q} (feat dim {2**q}) ==========")
        t0 = time.time()
        summary, per_seed, arrs = _run_cell(
            q=q, K=K, data=data, grid=grid, dt=dt, dt_save=dt_save,
            n_phys_substeps=n_phys_substeps, rhs_phys=rhs_phys,
            ref_truth_snaps=ref_truth, seed_truth_snaps=seed_truth,
            seeds_in_file=seeds_in_file,
            window=args.window, ridge_alpha=args.ridge_alpha)
        summary["wall_time_s"] = time.time() - t0
        summaries[q] = summary
        arrs_by_q[q] = arrs

        # Save per-cell
        with open(cell_dir / "results.json", "w") as f:
            json.dump({"summary": summary, "per_seed": per_seed}, f, indent=2)

        # Quick console table
        print(f"\nLead-1 mean MSE (lower is better):")
        for m in ["physics", "persistence", "linreg", "mlp", "qrc", "esn", "rff"]:
            print(f"  {m:>11}: {summary['lead1_mean'][m]:>10.3e}")
        print("\nQRC at lead-1 vs each baseline (8 seeds):")
        for m, p_val in summary["lead1_wilcoxon_qrc_vs"].items():
            wins = summary["lead1_qrc_wins_vs"][m]
            mean_qrc = summary["lead1_mean"]["qrc"]
            mean_other = summary["lead1_mean"][m]
            direction = "QRC < " + m if mean_qrc < mean_other else m + " < QRC"
            print(f"  {m:>11}: {wins}/8 seeds QRC<{m},  p = {p_val:.4f},  "
                  f"lower mean: {direction}")
        print(f"  cell wall time: {summary['wall_time_s']:.1f} s")

    # Aggregate save
    with open(out_dir / "scaling_summary.json", "w") as f:
        json.dump({str(q): summaries[q] for q in args.qs}, f, indent=2)
    print(f"\nSaved aggregate {out_dir / 'scaling_summary.json'}")

    # Plot
    _plot_scaling(out_dir, args.qs, summaries, arrs_by_q,
                   dt_h=float(data["dt_save"]) / 3600.0)


if __name__ == "__main__":
    main()
