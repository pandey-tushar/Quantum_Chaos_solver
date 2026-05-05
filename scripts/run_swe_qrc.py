#!/usr/bin/env python3
"""
run_swe_qrc.py - SWE forecasting POC: QRC vs ESN vs persistence.

Pipeline:
    1. Load SWE data produced by `generate_swe_data.py`.
    2. Fit POD on the reference trajectory (ref_snaps), retain K leading modes.
    3. For each seed: project the seed trajectory onto POD coeffs (n_t, K).
    4. Train QRC on the first 70% of coeffs (windowed ridge regression on quantum
       reservoir features) and an ESN at matched feature dim, then evaluate:
         - one-step-ahead teacher-forced MSE on the held-out 30%
         - closed-loop rollout RMSE vs lead time (forecast skill)
    5. Compare against a persistence baseline.

Outputs:
    results/swe_qrc/swe_qrc_results.json
    results/swe_qrc/rmse_vs_leadtime.png

Mirror of `scripts/run_seeds.py` patterns: same Statevector-based feature
extraction, same temporal windowing, same ridge readout. The only thing that
changes is the upstream data source (POD coeffs of SWE state instead of
Lorenz/Rossler/L96 state).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pod_reduction import PODReducer

# Lazy imports of qiskit / matplotlib happen inside functions that need them.


# ---------------------------------------------------------------------------
# QRC feature extraction (Statevector-based, dim-agnostic; mirrors run_seeds.py)
# ---------------------------------------------------------------------------

def make_qrc_extractor(n_qubits: int, n_layers: int, seed: int,
                        scale_lo: np.ndarray | None = None,
                        scale_hi: np.ndarray | None = None,
                        encoding: str = "global"):
    """
    Build a quantum reservoir feature extractor.

    Encoding modes:
      - "global"    : map each state component to [0, 2*pi] using fixed
                      (scale_lo, scale_hi) per-component bounds (from training
                      stats). Preserves absolute magnitude across time.
      - "per_state" : map each state to [0, 2*pi] using its own min/max.
                      Spreads encoded angles widely; loses absolute magnitude
                      but increases reservoir-feature entropy. This is what
                      run_seeds.py uses for Lorenz/Rossler/L96.

    Output is the 2^n_qubits probability vector of the reservoir's final
    statevector. n_layers fixed-random Rx/Ry/Rz + linear-chain CNOTs.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    rng = np.random.default_rng(seed)
    rot_params = rng.uniform(0, 2 * np.pi, (n_layers, n_qubits, 3))

    if encoding == "global":
        if scale_lo is None or scale_hi is None:
            raise ValueError("encoding='global' requires scale_lo and scale_hi")

        def _pad(a):
            a = np.asarray(a, dtype=float)
            if len(a) >= n_qubits:
                return a[:n_qubits]
            return np.concatenate([a, np.zeros(n_qubits - len(a))])

        lo = _pad(scale_lo)
        hi = _pad(scale_hi)
        span = np.where(hi - lo > 1e-12, hi - lo, 1.0)
    elif encoding != "per_state":
        raise ValueError(f"Unknown encoding: {encoding}")

    def extract(state: np.ndarray) -> np.ndarray:
        if len(state) >= n_qubits:
            x = state[:n_qubits].astype(float)
        else:
            x = np.concatenate([state.astype(float),
                                np.zeros(n_qubits - len(state))])

        if encoding == "global":
            x = np.clip((x - lo) / span, 0.0, 1.0) * 2.0 * np.pi
        else:  # per_state
            x = (x - x.min()) / (x.max() - x.min() + 1e-8) * 2.0 * np.pi

        qc = QuantumCircuit(n_qubits)
        for q in range(n_qubits):
            qc.ry(float(x[q]), q)
        for layer in range(n_layers):
            for q in range(n_qubits):
                qc.rx(float(rot_params[layer, q, 0]), q)
                qc.ry(float(rot_params[layer, q, 1]), q)
                qc.rz(float(rot_params[layer, q, 2]), q)
            for q in range(n_qubits - 1):
                qc.cx(q, q + 1)

        sv = Statevector.from_instruction(qc)
        return np.abs(sv.data) ** 2  # 2^n_qubits exact probabilities

    return extract


# ---------------------------------------------------------------------------
# Classical ESN at matched feature dim (no extra dependencies)
# ---------------------------------------------------------------------------

class _ESN:
    """
    Lightweight echo-state network mirroring the structure used in
    run_esn_baseline.py, but parameterised so feature dim matches QRC
    (n_reservoir == 2 ** n_qubits) for an apples-to-apples comparison.
    """

    def __init__(self, n_input: int, n_reservoir: int, n_output: int,
                 spectral_radius: float = 0.9, input_scaling: float = 0.1,
                 leaking_rate: float = 0.3, ridge_alpha: float = 1.0,
                 seed: int = 0):
        self.alpha = leaking_rate
        rng = np.random.default_rng(seed)
        self.W_in = rng.uniform(-input_scaling, input_scaling, (n_reservoir, n_input))
        density = 0.1
        W = rng.standard_normal((n_reservoir, n_reservoir))
        W *= (rng.uniform(0, 1, (n_reservoir, n_reservoir)) < density)
        eigs = np.linalg.eigvals(W)
        sr = float(np.max(np.abs(eigs)))
        if sr > 1e-8:
            W *= spectral_radius / sr
        self.W_res = W
        self.n_reservoir = n_reservoir
        self.ridge_alpha = ridge_alpha
        self.W_out = None

    def _activations(self, sequence: np.ndarray) -> np.ndarray:
        T = len(sequence)
        r = np.zeros(self.n_reservoir)
        out = np.zeros((T, self.n_reservoir))
        for t, u in enumerate(sequence):
            r = (1 - self.alpha) * r + self.alpha * np.tanh(self.W_res @ r + self.W_in @ u)
            out[t] = r
        return out


# ---------------------------------------------------------------------------
# Common train/eval helpers
# ---------------------------------------------------------------------------

def _build_windowed(features: np.ndarray, targets: np.ndarray, window: int):
    """
    Returns (F, S):
      F[i] = concat(features[i:i+window])    (i = 0 .. T - window - 1)
      S[i] = targets[i + window]
    """
    T = len(features)
    n_samples = T - window
    if n_samples <= 0:
        raise ValueError(f"Not enough samples: T={T}, window={window}")
    feat_dim = features.shape[1]
    F = np.zeros((n_samples, feat_dim * window))
    S = np.zeros((n_samples, targets.shape[1]))
    for i in range(n_samples):
        F[i] = features[i:i + window].flatten()
        S[i] = targets[i + window]
    return F, S


def _ridge_solve(F: np.ndarray, S: np.ndarray, alpha: float) -> np.ndarray:
    A = F.T @ F + alpha * np.eye(F.shape[1])
    return np.linalg.solve(A, F.T @ S)


def _closed_loop_rollout(extractor, last_train_features, W_out, n_test, window,
                          delta_target: bool = False,
                          start_state: np.ndarray | None = None):
    """
    Run a closed-loop forecast for `n_test` steps starting from the end of training.
    `last_train_features` is a list of `window` feature vectors from the end of training.
    If delta_target, the readout outputs (next - current); start_state must be the
    state at training end so we can accumulate.
    Returns predictions of shape (n_test, output_dim).
    """
    buf = list(last_train_features[-window:])
    preds = []
    cur = None if start_state is None else np.array(start_state, dtype=float).copy()
    for _ in range(n_test):
        flat = np.concatenate(buf[-window:])
        out = flat @ W_out
        if delta_target:
            cur = cur + out
            nxt = cur.copy()
        else:
            nxt = out
        preds.append(nxt)
        feat = extractor(nxt)
        buf.append(feat)
    return np.array(preds)


# ---------------------------------------------------------------------------
# QRC and ESN evaluation
# ---------------------------------------------------------------------------

def evaluate_qrc(coeffs: np.ndarray, train_end: int, n_qubits: int, n_layers: int,
                 window: int, seed: int, ridge_alpha: float,
                 encoding: str = "global", delta_target: bool = False):
    """
    coeffs: (T, K) trajectory in POD space
    train_end: index where train ends and test begins
    encoding: "global" (per-component, train-set bounds) or "per_state"
    delta_target: if True, fit ridge to predict (next - current) and add back at
                  inference. Helps when the trajectory varies slowly between
                  consecutive samples relative to the modal range.
    Returns dict with per-step predictions for both teacher-forced and closed-loop.
    """
    if encoding == "global":
        train_coeffs = coeffs[:train_end]
        scale_lo = train_coeffs.min(axis=0)
        scale_hi = train_coeffs.max(axis=0)
        margin = 0.1 * (scale_hi - scale_lo + 1e-12)
        scale_lo = scale_lo - margin
        scale_hi = scale_hi + margin
        extractor = make_qrc_extractor(n_qubits, n_layers, seed,
                                        scale_lo=scale_lo, scale_hi=scale_hi,
                                        encoding="global")
    else:
        extractor = make_qrc_extractor(n_qubits, n_layers, seed,
                                        encoding="per_state")

    t0 = time.time()
    all_feats = np.array([extractor(s) for s in coeffs])     # (T, 2**n_qubits)

    # Train: predict coeffs[t+1] (or its delta from coeffs[t]) from window of
    # features ending at t.
    train_feats = all_feats[:train_end]
    train_targets = coeffs[:train_end]
    F_tr, S_tr = _build_windowed(train_feats, train_targets, window)
    if delta_target:
        # current state is the LAST feature's source state, i.e. coeffs[i + window - 1]
        cur_train = np.array([coeffs[i + window - 1] for i in range(len(F_tr))])
        S_fit = S_tr - cur_train
    else:
        S_fit = S_tr
    W = _ridge_solve(F_tr, S_fit, ridge_alpha)
    train_time = time.time() - t0

    # Train MSE on absolute targets (apples-to-apples with persistence/ESN)
    if delta_target:
        train_pred = (F_tr @ W) + cur_train
    else:
        train_pred = F_tr @ W
    train_mse = float(np.mean((train_pred - S_tr) ** 2))

    # Teacher-forced one-step on test (use ground-truth features in the window)
    teacher_preds = []
    teacher_targets = []
    for i in range(train_end, len(coeffs) - 1):
        flat = all_feats[i - window + 1:i + 1].flatten()
        delta_or_next = flat @ W
        nxt = (coeffs[i] + delta_or_next) if delta_target else delta_or_next
        teacher_preds.append(nxt)
        teacher_targets.append(coeffs[i + 1])
    teacher_preds = np.array(teacher_preds)
    teacher_targets = np.array(teacher_targets)
    teacher_mse = float(np.mean((teacher_preds - teacher_targets) ** 2))

    # Closed-loop rollout
    n_test_steps = len(coeffs) - train_end
    last_feats = list(all_feats[train_end - window:train_end])
    cl_preds = _closed_loop_rollout(extractor, last_feats, W, n_test_steps,
                                     window, delta_target=delta_target,
                                     start_state=coeffs[train_end - 1])
    cl_targets = coeffs[train_end:train_end + n_test_steps]
    # Per-leadtime MSE (mean across coefficient dimensions)
    cl_per_lead = np.mean((cl_preds - cl_targets) ** 2, axis=1)

    return {
        "train_mse": train_mse,
        "teacher_mse": teacher_mse,
        "closed_loop_per_lead_mse": cl_per_lead.tolist(),
        "train_time_s": train_time,
        "feature_dim": int(all_feats.shape[1]),
    }


def evaluate_esn(coeffs: np.ndarray, train_end: int, n_reservoir: int,
                 window: int, seed: int, ridge_alpha: float,
                 spectral_radius: float = 0.9, input_scaling: float = 0.1,
                 leaking_rate: float = 0.3,
                 delta_target: bool = False):
    K = coeffs.shape[1]
    esn = _ESN(n_input=K, n_reservoir=n_reservoir, n_output=K,
               spectral_radius=spectral_radius, input_scaling=input_scaling,
               leaking_rate=leaking_rate, ridge_alpha=ridge_alpha, seed=seed)

    t0 = time.time()
    train_acts = esn._activations(coeffs[:train_end])
    F_tr, S_tr = _build_windowed(train_acts, coeffs[:train_end], window)
    if delta_target:
        cur_train = np.array([coeffs[i + window - 1] for i in range(len(F_tr))])
        S_fit = S_tr - cur_train
    else:
        S_fit = S_tr
    W = _ridge_solve(F_tr, S_fit, ridge_alpha)
    train_time = time.time() - t0

    if delta_target:
        train_pred = (F_tr @ W) + cur_train
    else:
        train_pred = F_tr @ W
    train_mse = float(np.mean((train_pred - S_tr) ** 2))

    # Teacher-forced one-step on test (drive reservoir with ground-truth coeffs)
    full_acts = esn._activations(coeffs)
    teacher_preds, teacher_targets = [], []
    for i in range(train_end, len(coeffs) - 1):
        flat = full_acts[i - window + 1:i + 1].flatten()
        out = flat @ W
        nxt = (coeffs[i] + out) if delta_target else out
        teacher_preds.append(nxt)
        teacher_targets.append(coeffs[i + 1])
    teacher_preds = np.array(teacher_preds)
    teacher_targets = np.array(teacher_targets)
    teacher_mse = float(np.mean((teacher_preds - teacher_targets) ** 2))

    # Closed-loop: feed ESN's own predictions back as input
    n_test_steps = len(coeffs) - train_end
    r = train_acts[-1].copy()
    act_buf = list(train_acts[-window:])
    cl_preds = []
    next_input = coeffs[train_end - 1]
    cur = coeffs[train_end - 1].copy()
    for _ in range(n_test_steps):
        r = (1 - esn.alpha) * r + esn.alpha * np.tanh(esn.W_res @ r + esn.W_in @ next_input)
        act_buf.append(r.copy())
        flat = np.concatenate(act_buf[-window:])
        out = flat @ W
        if delta_target:
            cur = cur + out
            nxt = cur.copy()
        else:
            nxt = out
        cl_preds.append(nxt)
        next_input = nxt
    cl_preds = np.array(cl_preds)
    cl_targets = coeffs[train_end:train_end + n_test_steps]
    cl_per_lead = np.mean((cl_preds - cl_targets) ** 2, axis=1)

    return {
        "train_mse": train_mse,
        "teacher_mse": teacher_mse,
        "closed_loop_per_lead_mse": cl_per_lead.tolist(),
        "train_time_s": train_time,
        "feature_dim": n_reservoir,
    }


def evaluate_persistence(coeffs: np.ndarray, train_end: int):
    """At each test step, predict next coeff = current coeff."""
    targets = coeffs[train_end:]
    preds = coeffs[train_end - 1:-1]
    n = min(len(targets), len(preds))
    cl_per_lead = np.mean((preds[:n] - targets[:n]) ** 2, axis=1)
    return {
        "closed_loop_per_lead_mse": cl_per_lead.tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Run SWE+QRC POC pipeline")
    p.add_argument("--data", type=str, default=str(ROOT / "data" / "swe" / "swe_data.npz"))
    p.add_argument("--n-modes", type=int, default=5)
    p.add_argument("--n-qubits", type=int, default=5)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--ridge-alpha", type=float, default=1.0)
    p.add_argument("--train-frac", type=float, default=0.7)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="Override seed list (default: use all from data file)")
    p.add_argument("--no-esn", action="store_true", help="Skip the ESN baseline")
    p.add_argument("--encoding", type=str, default="global",
                   choices=["global", "per_state"],
                   help="QRC angle-encoding scheme: global (default) or per_state")
    p.add_argument("--delta-target", action="store_true",
                   help="Predict coeff increments (next-current) instead of next absolute")
    p.add_argument("--out-dir", type=str, default=str(ROOT / "results" / "swe_qrc"))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data}")
    data = np.load(args.data, allow_pickle=False)
    ref_snaps = data["ref_snaps"]
    seed_snaps = data["seed_snaps"]
    seeds_in_file = data["seeds"].tolist()
    dt_save = float(data["dt_save"])
    print(f"  ref_snaps: {ref_snaps.shape}; seed_snaps: {seed_snaps.shape}; "
          f"dt_save={dt_save:.1f}s ({dt_save/3600:.2f}h)")

    # ---- POD on reference trajectory ----
    pod = PODReducer(n_modes=args.n_modes).fit(ref_snaps)
    print(f"\nPOD: {args.n_modes} modes, "
          f"singular values {[f'{s:.1f}' for s in pod.singular_values_]}")

    seeds_to_run = args.seeds if args.seeds is not None else seeds_in_file
    print(f"Seeds: {seeds_to_run}")
    print(f"QRC: {args.n_qubits}q x {args.n_layers}L, window={args.window}, "
          f"feature_dim=2^{args.n_qubits}={2**args.n_qubits}, ridge_alpha={args.ridge_alpha}")

    feature_dim = 2 ** args.n_qubits
    per_seed = []

    for s in seeds_to_run:
        if s not in seeds_in_file:
            print(f"  seed {s}: not in data file, skipping")
            continue
        idx = seeds_in_file.index(s)
        snaps_s = seed_snaps[idx]                            # (n_t, 3, Nx, Ny)
        coeffs = pod.transform(snaps_s)                       # (n_t, K)
        T = len(coeffs)
        train_end = int(args.train_frac * T)
        print(f"\n--- Seed {s}: T={T} steps, train={train_end}, test={T - train_end} ---")
        print(f"  POD coeff ranges: " + ", ".join(
            f"[{coeffs[:, k].min():+.1f},{coeffs[:, k].max():+.1f}]"
            for k in range(args.n_modes)))

        result = {"seed": int(s), "T": T, "train_end": train_end}

        # Persistence
        result["persistence"] = evaluate_persistence(coeffs, train_end)
        pers0 = result["persistence"]["closed_loop_per_lead_mse"][0]
        pers_mean = float(np.mean(result["persistence"]["closed_loop_per_lead_mse"]))
        print(f"  persistence: lead-1 MSE={pers0:.3f}, mean MSE={pers_mean:.3f}")

        # QRC
        qrc = evaluate_qrc(coeffs, train_end,
                           n_qubits=args.n_qubits, n_layers=args.n_layers,
                           window=args.window, seed=s,
                           ridge_alpha=args.ridge_alpha,
                           encoding=args.encoding,
                           delta_target=args.delta_target)
        result["qrc"] = qrc
        cl0 = qrc["closed_loop_per_lead_mse"][0]
        cl_mean = float(np.mean(qrc["closed_loop_per_lead_mse"]))
        print(f"  QRC:         train_mse={qrc['train_mse']:.3f}, "
              f"teacher={qrc['teacher_mse']:.3f}, "
              f"CL lead-1={cl0:.3f}, CL mean={cl_mean:.3f}, t={qrc['train_time_s']:.2f}s")

        # ESN at matched feature dim
        if not args.no_esn:
            esn = evaluate_esn(coeffs, train_end,
                               n_reservoir=feature_dim, window=args.window,
                               seed=s, ridge_alpha=args.ridge_alpha,
                               delta_target=args.delta_target)
            result["esn"] = esn
            cl0_e = esn["closed_loop_per_lead_mse"][0]
            cl_mean_e = float(np.mean(esn["closed_loop_per_lead_mse"]))
            print(f"  ESN ({feature_dim}r):  train_mse={esn['train_mse']:.3f}, "
                  f"teacher={esn['teacher_mse']:.3f}, "
                  f"CL lead-1={cl0_e:.3f}, CL mean={cl_mean_e:.3f}, t={esn['train_time_s']:.2f}s")

        per_seed.append(result)

    # ---- Aggregate & save ----
    summary = {
        "config": {
            "n_modes": args.n_modes,
            "n_qubits": args.n_qubits,
            "n_layers": args.n_layers,
            "window": args.window,
            "ridge_alpha": args.ridge_alpha,
            "train_frac": args.train_frac,
            "feature_dim": feature_dim,
        },
        "data": {
            "ref_snaps_shape": list(ref_snaps.shape),
            "seed_snaps_shape": list(seed_snaps.shape),
            "dt_save_s": dt_save,
        },
        "pod_singular_values": pod.singular_values_.tolist(),
        "per_seed": per_seed,
    }
    # ---- Per-seed table + statistical tests across seeds ----
    if len(per_seed) >= 2:
        from scipy import stats as _stats
        print("\n" + "=" * 78)
        print(f"PER-SEED TEACHER MSE  (n_seeds = {len(per_seed)})")
        print("=" * 78)
        header = f"{'seed':>4}  {'persistence':>12}  {'QRC':>12}"
        if any("esn" in r for r in per_seed):
            header += f"  {'ESN':>12}"
        print(header)
        print("-" * 78)
        qrc_v, esn_v, pers_v = [], [], []
        for r in per_seed:
            pers_lead1 = r["persistence"]["closed_loop_per_lead_mse"][0]
            qrc_t = r["qrc"]["teacher_mse"]
            row = f"{r['seed']:>4}  {pers_lead1:>12.3f}  {qrc_t:>12.3f}"
            pers_v.append(pers_lead1)
            qrc_v.append(qrc_t)
            if "esn" in r:
                esn_t = r["esn"]["teacher_mse"]
                esn_v.append(esn_t)
                row += f"  {esn_t:>12.3f}"
            print(row)
        print("-" * 78)
        qrc_v = np.array(qrc_v)
        pers_v = np.array(pers_v)
        agg = f"{'mean':>4}  {pers_v.mean():>12.3f}  {qrc_v.mean():>12.3f}"
        if esn_v:
            esn_v = np.array(esn_v)
            agg += f"  {esn_v.mean():>12.3f}"
        print(agg)
        agg = f"{'std':>4}  {pers_v.std():>12.3f}  {qrc_v.std():>12.3f}"
        if isinstance(esn_v, np.ndarray):
            agg += f"  {esn_v.std():>12.3f}"
        print(agg)

        def _wilcoxon(a, b, label):
            try:
                stat, pval = _stats.wilcoxon(a, b)
                better = "QRC<ESN" if a.mean() < b.mean() else "QRC>=ESN"
                print(f"\nWilcoxon (paired, n={len(a)}): {label} -> "
                      f"stat={stat:.2f}, p={pval:.4f}  ({better})")
                return {"stat": float(stat), "p_value": float(pval)}
            except Exception as e:
                print(f"\nWilcoxon ({label}) skipped: {e}")
                return None

        stats_block = {}
        if isinstance(esn_v, np.ndarray):
            stats_block["wilcoxon_qrc_vs_esn_teacher"] = _wilcoxon(
                qrc_v, esn_v, "QRC vs ESN teacher MSE")
        stats_block["wilcoxon_qrc_vs_persistence_teacher"] = _wilcoxon(
            qrc_v, pers_v, "QRC vs persistence (lead-1) MSE")
        summary["statistics"] = stats_block

    out_json = out_dir / "swe_qrc_results.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_json}")

    # ---- Plot RMSE vs lead time ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        max_lead = max(len(r["persistence"]["closed_loop_per_lead_mse"]) for r in per_seed)
        leads = np.arange(1, max_lead + 1)

        def stack(method):
            arrs = []
            for r in per_seed:
                if method not in r:
                    continue
                v = np.array(r[method]["closed_loop_per_lead_mse"])
                if len(v) < max_lead:
                    v = np.pad(v, (0, max_lead - len(v)), constant_values=np.nan)
                arrs.append(v)
            return np.array(arrs) if arrs else None

        for method, label, color in [
            ("persistence", "Persistence", "#888"),
            ("qrc", "QRC", "#0066cc"),
            ("esn", f"ESN (N={feature_dim})", "#cc4400"),
        ]:
            data_s = stack(method)
            if data_s is None:
                continue
            mean = np.nanmean(data_s, axis=0)
            std = np.nanstd(data_s, axis=0)
            ax.plot(leads, np.sqrt(mean), label=label, color=color, lw=2)
            ax.fill_between(leads, np.sqrt(np.maximum(mean - std, 0.0)),
                             np.sqrt(mean + std), color=color, alpha=0.2)

        ax.set_xlabel(f"Lead time (steps; 1 step = {dt_save/3600:.2f} h)")
        ax.set_ylabel("RMSE in POD coefficient space")
        ax.set_title(f"Closed-loop forecast skill (mean +/- std over {len(per_seed)} seeds)")
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig_path = out_dir / "rmse_vs_leadtime.png"
        fig.savefig(fig_path, dpi=140)
        print(f"Saved {fig_path}")
    except Exception as e:
        print(f"(Plot skipped: {e})")


if __name__ == "__main__":
    main()
