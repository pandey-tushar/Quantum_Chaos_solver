#!/usr/bin/env python3
"""
run_qg_baselines.py - Additional baselines for the QG hybrid pipeline.

Runs four extra correctors on the same QG data the main pipeline used,
so that the atmospheric-extension paper's audience sees a fuller set of
comparisons:

    1. Persistence (next state = current state)
    2. Linear regression: windowed POD coeffs -> next-step residual
       (closed-form ridge with regularisation, no nonlinearity)
    3. Single-hidden-layer MLP at matched-parameter-count to the QRC
       readout, ReLU, trained via L-BFGS-B with weight decay
    4. Random Fourier Features (RFF) reservoir at matched feature dim
       (fixed nonlinear projection + ridge readout; classical analogue
        of "fixed reservoir")

The QRC and ESN closed-loop curves are loaded directly from the existing
results/qg_hybrid/qg_hybrid_results.json so the merged plot is fully
apples-to-apples.

Outputs:
    results/qg_hybrid/baselines/qg_baselines_results.json
    results/qg_hybrid/baselines/rmse_vs_leadtime_full.png
"""
from __future__ import annotations

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
from qg_solver import qg_rk4_step

from run_qg_hybrid import (   # type: ignore
    _grid_dict_from_npz, _spectral_grid, _physics_step_block,
    _vorticity_to_psi, _stack_q_psi,
)
from run_swe_qrc import _ridge_solve   # type: ignore


# ---------------------------------------------------------------------------
# Tiny MLP trained with L-BFGS (no torch dependency)
# ---------------------------------------------------------------------------

def _train_mlp(X: np.ndarray, Y: np.ndarray, hidden: int = 32,
                weight_decay: float = 1e-3, n_iter: int = 2000,
                seed: int = 0):
    """Train a single-hidden-layer ReLU MLP via batch L-BFGS-B."""
    in_dim = X.shape[1]
    out_dim = Y.shape[1]
    n_W1 = in_dim * hidden
    n_b1 = hidden
    n_W2 = hidden * out_dim
    n_b2 = out_dim
    n_total = n_W1 + n_b1 + n_W2 + n_b2

    rng = np.random.default_rng(seed)
    theta0 = np.empty(n_total)
    # He init for the two weight matrices; zeros for biases
    theta0[:n_W1]                          = rng.standard_normal(n_W1) * np.sqrt(2.0 / in_dim)
    theta0[n_W1:n_W1 + n_b1]               = 0.0
    theta0[n_W1 + n_b1:n_W1 + n_b1 + n_W2] = rng.standard_normal(n_W2) * np.sqrt(2.0 / hidden)
    theta0[n_W1 + n_b1 + n_W2:]            = 0.0

    def unpack(theta):
        i = 0
        W1 = theta[i:i + n_W1].reshape(in_dim, hidden); i += n_W1
        b1 = theta[i:i + n_b1];                          i += n_b1
        W2 = theta[i:i + n_W2].reshape(hidden, out_dim); i += n_W2
        b2 = theta[i:i + n_b2]
        return W1, b1, W2, b2

    N = X.shape[0]

    def loss_and_grad(theta):
        W1, b1, W2, b2 = unpack(theta)
        z = X @ W1 + b1                # (N, hidden)
        h = np.maximum(0.0, z)         # ReLU
        Yhat = h @ W2 + b2             # (N, out_dim)
        diff = Yhat - Y
        loss = 0.5 * np.mean(diff ** 2) \
               + 0.5 * weight_decay * (np.sum(W1 ** 2) + np.sum(W2 ** 2))

        dYhat = diff / N
        dW2 = h.T @ dYhat + weight_decay * W2
        db2 = dYhat.sum(axis=0)
        dh = dYhat @ W2.T
        dz = dh * (z > 0)
        dW1 = X.T @ dz + weight_decay * W1
        db1 = dz.sum(axis=0)

        grad = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])
        return float(loss), grad

    res = minimize(loss_and_grad, theta0, jac=True, method="L-BFGS-B",
                    options={"maxiter": n_iter, "ftol": 1e-10, "gtol": 1e-8})
    W1, b1, W2, b2 = unpack(res.x)

    def predict(X_new):
        z = X_new @ W1 + b1
        h = np.maximum(0.0, z)
        return h @ W2 + b2

    return predict, int(res.nit), float(res.fun)


# ---------------------------------------------------------------------------
# Random Fourier Features
# ---------------------------------------------------------------------------

def _rff_extractor(in_dim: int, n_features: int, sigma: float, seed: int):
    """Return (extract, n_features). extract(x_vec) -> (n_features,) cosine projection."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((in_dim, n_features)) / max(sigma, 1e-9)
    b = rng.uniform(0.0, 2.0 * np.pi, n_features)
    scale = np.sqrt(2.0 / n_features)

    def extract(x):
        return np.cos(x @ W + b) * scale

    return extract


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    data_path = ROOT / "data" / "qg" / "qg_hybrid_data.npz"
    qrc_results_path = ROOT / "results" / "qg_hybrid" / "qg_hybrid_results.json"
    out_dir = ROOT / "results" / "qg_hybrid" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load ----
    print(f"Loading {data_path}")
    data = np.load(data_path, allow_pickle=False)
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

    # Mirror the main pipeline's choices exactly
    n_modes  = 5
    window   = 5
    train_frac = 0.7
    feature_dim = 32   # matched to QRC's 2^5 in the main pipeline

    pod = PODReducer(n_modes=n_modes).fit(ref_truth)
    print(f"POD: {n_modes} modes")

    # Load existing QRC + ESN results so the final plot is on one axis
    with open(qrc_results_path) as f:
        qrc_run = json.load(f)
    qrc_per_seed = {r["seed"]: r for r in qrc_run["per_seed"]}

    per_seed = []
    for s in seeds_in_file:
        idx = seeds_in_file.index(s)
        truth_snaps = seed_truth[idx]                     # (n_t, 2, Nx, Ny)
        n_t = truth_snaps.shape[0]
        train_end = int(train_frac * n_t)
        n_test_steps = n_t - train_end
        print(f"\n--- Seed {s}: T={n_t}, train={train_end}, test={n_test_steps} ---")

        truth_coeffs = pod.transform(truth_snaps)         # (n_t, K)

        # One-step physics from each training truth snapshot
        t0 = time.time()
        phys_pairs = []
        for t in range(train_end):
            q_next = _physics_step_block(truth_snaps[t, 0], n_phys_substeps,
                                          dt, grid, rhs_phys)
            phys_pairs.append(_stack_q_psi(q_next, grid))
        phys_one_grid = np.stack(phys_pairs, axis=0)
        phys_one_coeffs = pod.transform(phys_one_grid)
        print(f"  physics-step pre-compute: {time.time()-t0:.1f}s")

        target_residual = truth_coeffs[1:train_end + 1] - phys_one_coeffs

        # Build windowed input on RAW POD coefficients (matched I/O for MLP / linreg)
        F_lin, S_res = [], []
        for i in range(train_end - window):
            F_lin.append(truth_coeffs[i:i + window].flatten())
            S_res.append(target_residual[i + window - 1])
        F_lin = np.asarray(F_lin)
        S_res = np.asarray(S_res)
        in_dim = F_lin.shape[1]
        print(f"  training pairs: {F_lin.shape[0]} samples, "
              f"{in_dim}-dim input -> {n_modes}-dim residual")

        # ----- 1. Persistence -----
        # predict next coeff = current coeff   <=> correction = next - current
        # We use it as a STAND-ALONE forecaster on the test window (i.e. closed-loop
        # next-state = current-state at every step), so the rollout is just the
        # truth at lead-1 shifted to compare against truth at lead-1, etc.
        # That is the conventional persistence-skill curve.
        truth_test_coeffs = truth_coeffs[train_end:train_end + n_test_steps]
        pers_preds = np.tile(truth_coeffs[train_end - 1], (n_test_steps, 1))
        pers_per_lead = np.mean((pers_preds - truth_test_coeffs) ** 2, axis=1)

        # ----- 2. Linear regression on windowed POD coeffs -> residual -----
        t0 = time.time()
        W_lin = _ridge_solve(F_lin, S_res, alpha=1.0)
        lin_train_t = time.time() - t0

        # ----- 3. MLP at matched param count (~hidden=8 keeps trainable params
        # in the same ballpark as QRC's 5*32*5 = 800 readout matrix) -----
        # in_dim = 25 (window=5, modes=5)
        # MLP params = 25*hidden + hidden + hidden*5 + 5 = 30*hidden + hidden + 5
        # For hidden=24:  30*24 + 24 + 5 = 749  -> close to 800
        hidden = 24
        t0 = time.time()
        mlp_predict, mlp_iters, mlp_loss = _train_mlp(
            F_lin, S_res, hidden=hidden, weight_decay=1e-3, n_iter=2000, seed=s)
        mlp_train_t = time.time() - t0
        mlp_train_pred = mlp_predict(F_lin)
        mlp_train_mse = float(np.mean((mlp_train_pred - S_res) ** 2))
        print(f"  MLP({hidden}): {mlp_iters} L-BFGS iters, "
              f"final loss {mlp_loss:.3e}, train_mse {mlp_train_mse:.3e}, "
              f"t={mlp_train_t:.2f}s")

        # ----- 4. RFF reservoir at matched feature_dim -----
        # raw POD coefficient vector -> RFF features (feature_dim=32)
        # then concat over window to get window*feature_dim flattened features,
        # ridge-regress to residual. Mirrors QRC's window-of-features structure.
        rff_extract = _rff_extractor(in_dim=n_modes, n_features=feature_dim,
                                       sigma=float(np.std(truth_coeffs)),
                                       seed=s)
        rff_feats_all = np.array([rff_extract(c) for c in truth_coeffs])
        F_rff, _ = [], []
        for i in range(train_end - window):
            F_rff.append(rff_feats_all[i:i + window].flatten())
        F_rff = np.asarray(F_rff)
        t0 = time.time()
        W_rff = _ridge_solve(F_rff, S_res, alpha=1.0)
        rff_train_t = time.time() - t0

        # ----- Closed-loop rollouts -----
        def rollout(corrector):
            q_state = truth_snaps[train_end - 1, 0].copy()
            coeff_buf = list(truth_coeffs[train_end - window:train_end])
            rff_buf   = list(rff_feats_all[train_end - window:train_end])
            preds = []
            for _ in range(n_test_steps):
                q_next = _physics_step_block(q_state, n_phys_substeps,
                                                dt, grid, rhs_phys)
                phys_pair = _stack_q_psi(q_next, grid)
                phys_coeffs = pod.transform(phys_pair[None, ...])[0]

                if corrector == "physics":
                    correction = np.zeros(n_modes)
                elif corrector == "linreg":
                    flat = np.concatenate(coeff_buf[-window:])
                    correction = flat @ W_lin
                elif corrector == "mlp":
                    flat = np.concatenate(coeff_buf[-window:])
                    correction = mlp_predict(flat[None, :])[0]
                elif corrector == "rff":
                    flat = np.concatenate(rff_buf[-window:])
                    correction = flat @ W_rff
                else:
                    raise ValueError(corrector)

                new_coeffs = phys_coeffs + correction
                preds.append(new_coeffs)
                new_pair = pod.inverse_transform(new_coeffs[None, :])[0]
                q_state = new_pair[0]
                coeff_buf.append(new_coeffs)
                if corrector == "rff":
                    rff_buf.append(rff_extract(new_coeffs))
            return np.asarray(preds)

        cl_phys_check = rollout("physics")          # sanity (already in qrc results)
        cl_lin = rollout("linreg")
        cl_mlp = rollout("mlp")
        cl_rff = rollout("rff")

        def per_lead_mse(preds):
            return np.mean((preds - truth_test_coeffs) ** 2, axis=1)

        # Pull the existing QRC + ESN + physics from the main run for this seed
        existing = qrc_per_seed[s]
        result = {
            "seed": int(s), "T": int(n_t), "train_end": int(train_end),
            "physics":      {"closed_loop_per_lead_mse":
                              existing["physics"]["closed_loop_per_lead_mse"]},
            "qrc":          {"closed_loop_per_lead_mse":
                              existing["qrc"]["closed_loop_per_lead_mse"]},
            "esn":          {"closed_loop_per_lead_mse":
                              existing["esn"]["closed_loop_per_lead_mse"]},
            "persistence":  {"closed_loop_per_lead_mse": pers_per_lead.tolist()},
            "linreg":       {"closed_loop_per_lead_mse": per_lead_mse(cl_lin).tolist()},
            "mlp":          {"closed_loop_per_lead_mse": per_lead_mse(cl_mlp).tolist(),
                              "hidden_units": hidden,
                              "n_params": 30 * hidden + hidden + 5,
                              "train_mse": mlp_train_mse,
                              "train_time_s": mlp_train_t,
                              "lbfgs_iters": mlp_iters},
            "rff":          {"closed_loop_per_lead_mse": per_lead_mse(cl_rff).tolist(),
                              "feature_dim": feature_dim,
                              "train_time_s": rff_train_t},
        }
        per_seed.append(result)

        print(f"  CL lead-1 MSE: phys={existing['physics']['closed_loop_per_lead_mse'][0]:.3e}, "
              f"qrc={existing['qrc']['closed_loop_per_lead_mse'][0]:.3e}, "
              f"esn={existing['esn']['closed_loop_per_lead_mse'][0]:.3e}, "
              f"mlp={result['mlp']['closed_loop_per_lead_mse'][0]:.3e}, "
              f"linreg={result['linreg']['closed_loop_per_lead_mse'][0]:.3e}, "
              f"rff={result['rff']['closed_loop_per_lead_mse'][0]:.3e}, "
              f"pers={result['persistence']['closed_loop_per_lead_mse'][0]:.3e}")

    # ---- Wilcoxon table at lead-1 ----
    methods = ["physics", "qrc", "esn", "mlp", "linreg", "rff", "persistence"]
    print("\n" + "=" * 96)
    print("PER-SEED LEAD-1 MSE")
    print("=" * 96)
    header = f"{'seed':>4}  " + "  ".join(f"{m:>10}" for m in methods)
    print(header)
    print("-" * len(header))
    seed_lead1 = {m: [] for m in methods}
    for r in per_seed:
        row = [f"{r['seed']:>4}"]
        for m in methods:
            v = r[m]["closed_loop_per_lead_mse"][0]
            seed_lead1[m].append(v)
            row.append(f"{v:>10.3e}")
        print("  ".join(row))
    print("-" * len(header))
    print(f"{'mean':>4}  " + "  ".join(f"{np.mean(seed_lead1[m]):>10.3e}" for m in methods))
    print(f"{'std':>4}   " + "  ".join(f"{np.std(seed_lead1[m]):>10.3e}" for m in methods))

    print("\nWilcoxon (paired, n=8) -- QRC vs each baseline at lead-1:")
    qrc_v = np.array(seed_lead1["qrc"])
    wilcoxon = {}
    for m in methods:
        if m == "qrc":
            continue
        v = np.array(seed_lead1[m])
        try:
            stat, pval = stats.wilcoxon(qrc_v, v)
            wins = int(np.sum(qrc_v < v))
            print(f"  qrc vs {m:>11}: p = {pval:.4f}  ({wins}/{len(qrc_v)} seeds QRC < {m})")
            wilcoxon[f"qrc_vs_{m}"] = {"p_value": float(pval), "wins": wins,
                                          "n": int(len(qrc_v))}
        except Exception as e:
            print(f"  qrc vs {m:>11}: skipped ({e})")

    summary = {
        "config": {"n_modes": n_modes, "window": window, "feature_dim": feature_dim,
                   "train_frac": train_frac, "mlp_hidden": 24,
                   "rff_sigma_strategy": "std(truth_coeffs)"},
        "per_seed": per_seed,
        "lead1_wilcoxon_qrc_vs": wilcoxon,
    }
    out_json = out_dir / "qg_baselines_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        max_lead = min(len(per_seed[0][m]["closed_loop_per_lead_mse"]) for m in methods)
        leads = np.arange(1, max_lead + 1)

        plot_methods = [
            ("physics",     "Physics only",                     "#888",    "-"),
            ("persistence", "Persistence",                      "#444",    ":"),
            ("linreg",      "Linear regression",                "#bbbb00", "--"),
            ("mlp",         "Trainable MLP (~800 params)",      "#000",    "--"),
            ("esn",         "Classical ESN (N=32)",             "#cc4400", "--"),
            ("rff",         "Random Fourier Features (N=32)",   "#aa00aa", "--"),
            ("qrc",         "QRC (5 qubits)",                   "#0066cc", "-"),
        ]
        for key, label, color, ls in plot_methods:
            arrs = np.array([r[key]["closed_loop_per_lead_mse"][:max_lead] for r in per_seed])
            mean = np.mean(arrs, axis=0)
            ax.plot(leads, np.sqrt(mean), label=label, color=color, lw=2.0 if key == "qrc" else 1.6,
                     ls=ls)

        dt_save_h = float(data["dt_save"]) / 3600.0
        ax.set_xlabel(f"Lead time (steps; 1 step = {dt_save_h:.2f} h)")
        ax.set_ylabel("RMSE in POD coefficient space (mean across 8 seeds)")
        ax.set_yscale("log")
        ax.set_title("QG hybrid forecast skill, full baseline matrix (8 seeds)")
        ax.legend(loc="lower right", fontsize=9, ncol=1)
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        out_fig = out_dir / "rmse_vs_leadtime_full.png"
        fig.savefig(out_fig, dpi=150)
        print(f"Saved {out_fig}")
    except Exception as e:
        print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
