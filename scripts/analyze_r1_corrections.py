#!/usr/bin/env python3
"""
analyze_r1_corrections.py -- post-hoc corrections analysis for the MLST
resubmission, computed ENTIRELY from the existing results/r1/*.json artifacts
(no experiment is re-run).

It produces the quantitative support the corrected manuscript needs, in four
blocks that map to the four adjudicated review findings:

  (a) E1 live/dead parameter split  -- the "variance stays flat" headline is a
      pooling artifact of structurally-dead parameters whose fraction shrinks
      with depth; live-only variance decays ~9x from L=1 to L=5 toward a
      Haar-scale floor.
  (b) Capacity-curve dispersion      -- per-seed final losses, mean, sample std
      (ddof=1) and sem for each depth, so Table 5 can carry error bars.
  (c) Optimizer-comparison support   -- per-seed final losses for FD/PS/
      layerwise/SPSA, the FD-vs-PS relative agreement on shared seeds (they are
      numerically the same optimizer), and honest loss-EVALUATION counts per
      optimizer read off the harness's actual control flow.
  (d) Jacobian rank table            -- exact per-init ranks and their mean
      (32.67 at L=4, NOT rounded to 33) plus participation ratios.

READ ONLY. This script does not import r1_experiments.py and does not touch any
experiment file; it just reloads the deposited JSONs and re-derives statistics.

Output: results/r1/r1_corrections_analysis.json (+ printed SHA256).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "r1"

# Structural-zero threshold for the across-init variance of a (t, k) gradient
# component. A parameter that cannot influence <Z_0> at a given depth has
# IDENTICALLY zero gradient (the shift-rule difference is exact 0 up to float
# round-off), so its across-init variance is bounded by ~ (machine eps)^2.
# 1e-28 sits ~13 orders of magnitude below the live-parameter variances (1e-2)
# and ~5 orders above the observed dead-entry variances (~1e-33), so the split
# is unambiguous; we additionally verify the dead entries are exact zeros by
# checking max|gradient| ~ 1e-16 among them.
DEAD_VAR_THRESHOLD = 1e-28


def _load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


# ==========================================================================
# (a) E1 live/dead parameter split.
# ==========================================================================

def analyze_variance_live_dead() -> dict:
    """For each L in {1,3,5}: classify (t,k) gradient components as structurally
    dead (across-init variance < 1e-28) vs live, and report the pooled mean
    (must reproduce the harness's mcclean_stats value), the live-only mean, and
    the live-only L1/L5 decay ratio.

    per_init_gradients[i]["mcclean_bounded"] has shape (n_times, n_params); it
    is the EXACT per-t parameter-shift gradient of C_t = (1 - <Z_0>(t))/2. The
    McClean quantity is Var across inits at FIXED (t, k) -- a fixed circuit.
    """
    out = {"threshold_dead_var": DEAD_VAR_THRESHOLD, "per_L": {}}
    live_means = {}
    for L in (1, 3, 5):
        d = _load(f"variance_b1_e1_L{L}.json")
        # M shape: (n_inits, n_times, n_params)
        M = np.array([np.array(pi["mcclean_bounded"])
                      for pi in d["per_init_gradients"]])
        n_inits, n_t, npar = M.shape
        # Across-init variance at each (t, k): shape (n_times, n_params).
        var_kt = np.var(M, axis=0, ddof=0)
        dead_mask = var_kt < DEAD_VAR_THRESHOLD
        live_mask = ~dead_mask
        n_cells = var_kt.size
        n_dead = int(dead_mask.sum())
        dead_frac = n_dead / n_cells

        # Pooled mean over ALL (t, k) cells -- the harness's headline number.
        # The harness computes mean over components of var_kt, then mean over t
        # (mean of per-t means). For a rectangular (t, k) grid that equals the
        # flat mean of var_kt, which is what we compute here.
        pooled_mean = float(var_kt.mean())
        # Live-only mean (over the live cells only).
        live_mean = float(var_kt[live_mask].mean()) if live_mask.any() else 0.0
        live_means[L] = live_mean

        # Cross-check: the deposited harness value.
        harness_val = d["mcclean_stats"]["var_mean_over_components_and_t"]
        harness_median = d["mcclean_stats"]["var_median_over_components_and_t"]

        # Verify dead entries are EXACT structural zeros: the max |gradient|
        # over all inits at the dead (t, k) cells should be ~machine epsilon.
        # M is (inits, t, k); broadcast the (t,k) dead mask over the init axis.
        dead_grad_vals = M[:, dead_mask]  # (n_inits, n_dead)
        max_abs_dead_grad = (float(np.max(np.abs(dead_grad_vals)))
                             if dead_grad_vals.size else 0.0)
        max_abs_live_grad = float(np.max(np.abs(M[:, live_mask])))

        # Dilution identity: pooled_mean == live_mean * live_frac (dead ~ 0).
        live_frac = 1.0 - dead_frac
        dilution_check = live_mean * live_frac

        out["per_L"][L] = {
            "n_inits": int(n_inits),
            "n_times": int(n_t),
            "n_params": int(npar),
            "n_cells": int(n_cells),
            "n_dead": n_dead,
            "dead_fraction": dead_frac,
            "live_fraction": live_frac,
            "pooled_mean_var": pooled_mean,
            "live_mean_var": live_mean,
            "harness_mcclean_mean": harness_val,
            "harness_mcclean_median": harness_median,
            "pooled_vs_harness_abs_diff": abs(pooled_mean - harness_val),
            "max_abs_gradient_dead_cells": max_abs_dead_grad,
            "max_abs_gradient_live_cells": max_abs_live_grad,
            "dilution_identity_live_mean_x_live_frac": dilution_check,
            "dilution_identity_abs_diff": abs(dilution_check - pooled_mean),
        }

    # Live-only decay ratio L1 / L5 (the corrected "9x decay" number).
    out["live_decay_ratio_L1_over_L5"] = (
        live_means[1] / live_means[5] if live_means[5] else None)
    out["live_decay_ratio_L1_over_L3"] = (
        live_means[1] / live_means[3] if live_means[3] else None)
    out["pooled_ratio_L1_over_L5"] = (
        out["per_L"][1]["pooled_mean_var"] / out["per_L"][5]["pooled_mean_var"])

    # ------------------------------------------------------------------
    # Haar / 2-design saturation scale for this local single-qubit cost.
    # For an n-qubit circuit that forms an approximate 2-design, McClean's
    # bound gives Var[d<O>/dtheta] ~ 2^{-2n} for a LOCAL (single-qubit-Pauli)
    # cost operator O; the (1-<Z_0>)/2 cost here is exactly such a local cost.
    # At n=4, 2^{-2n} = 2^{-8} = 3.90625e-3. This is the floor the live-only
    # variance decays TOWARD (L=5 live mean 7.3e-3 ~ 1.9 * 2^-8), and the
    # reference line used in the figure.
    # ------------------------------------------------------------------
    n = 4
    haar_scale = 2.0 ** (-2 * n)
    out["haar_saturation_scale"] = {
        "formula": "2^{-2n} for a local single-qubit-Pauli cost at n qubits",
        "n_qubits": n,
        "value": haar_scale,
        "note": "McClean-2018 2-design variance for a LOCAL cost operator; "
                "the (1-<Z_0>)/2 cost is local, so its Haar/2-design floor is "
                "2^{-2n}=2^{-8}=3.906e-3 at n=4. Live-only variance decays "
                "from ~2^{-4} (L=1) toward this floor (L=5), the canonical "
                "decay-then-saturate barren-plateau depth profile.",
    }
    # Live-only variance in units of the Haar floor, per depth.
    out["live_mean_in_haar_units"] = {
        L: out["per_L"][L]["live_mean_var"] / haar_scale for L in (1, 3, 5)}
    return out


# ==========================================================================
# (b) Capacity-curve dispersion.
# ==========================================================================

def analyze_capacity_dispersion() -> dict:
    """Per-depth SPSA capacity-sweep dispersion: per-seed final losses, mean,
    sample std (ddof=1) and sem, so Table 5 can carry error bars."""
    out = {"per_L": {}}
    for L in (1, 2, 3, 5, 8):
        d = _load(f"train_spsa_b1_e3_cap_L{L}.json")
        seeds = [r["seed"] for r in d["runs"]]
        finals = np.array([r["final_loss"] for r in d["runs"]], dtype=float)
        n = len(finals)
        mean = float(finals.mean())
        std = float(finals.std(ddof=1)) if n > 1 else 0.0
        sem = std / np.sqrt(n) if n > 1 else 0.0
        out["per_L"][L] = {
            "n_seeds": n,
            "n_iters": d["config"]["n_iters"],
            "seeds": seeds,
            "per_seed_final_loss": finals.tolist(),
            "mean": mean,
            "std_ddof1": std,
            "sem": float(sem),
            "min": float(finals.min()),
            "max": float(finals.max()),
            "spread_max_over_min": float(finals.max() / finals.min()),
        }
    return out


# ==========================================================================
# (c) Optimizer-comparison support.
# ==========================================================================

def _spsa_loss_exact_near_200(run: dict):
    """Return (iter, loss_exact) for the loss_exact record nearest iteration
    200. The harness logs loss_exact at loop-index it (0-based) evaluated
    BEFORE the update, recorded as iter it+1, every K=10 -> logged iters
    1, 11, ..., 191, 201. The record nearest 200 is iter 201, which IS the
    exact loss after exactly 200 SPSA updates."""
    le = {it["iter"]: it["loss_exact"] for it in run["iters"]
          if it.get("loss_exact") is not None}
    if not le:
        return None, None
    target = 200
    nearest = min(le.keys(), key=lambda k: abs(k - target))
    return nearest, le[nearest]


def analyze_optimizer_comparison() -> dict:
    out = {}

    # --- FD (10 seeds) -----------------------------------------------------
    fd = _load("train_fd_b1_e4_fd_10seed.json")
    fd_seeds = [r["seed"] for r in fd["runs"]]
    fd_finals = {r["seed"]: float(r["final_loss"]) for r in fd["runs"]}
    out["fd"] = {
        "n_seeds": len(fd_seeds),
        "n_iters": fd["config"]["n_iters"],
        "seeds": fd_seeds,
        "per_seed_final_loss": [fd_finals[s] for s in fd_seeds],
        "mean": float(np.mean([fd_finals[s] for s in fd_seeds])),
    }

    # --- PS (5 seeds) ------------------------------------------------------
    ps = _load("train_ps_b1_e2b_ps.json")
    ps_seeds = [r["seed"] for r in ps["runs"]]
    ps_finals = {r["seed"]: float(r["final_loss"]) for r in ps["runs"]}
    out["ps"] = {
        "n_seeds": len(ps_seeds),
        "n_iters": ps["config"]["n_iters"],
        "seeds": ps_seeds,
        "per_seed_final_loss": [ps_finals[s] for s in ps_seeds],
        "mean": float(np.mean([ps_finals[s] for s in ps_seeds])),
    }

    # --- Layerwise (5 seeds, 3 stages x 200 = 600 Adam steps each) ---------
    lw = _load("train_layerwise_b1_e2d_layerwise.json")
    lw_seeds = [r["seed"] for r in lw["runs"]]
    lw_finals = {r["seed"]: float(r["final_loss"]) for r in lw["runs"]}
    lw_stages = lw["runs"][0]["stages"]
    lw_iters_recorded = len(lw["runs"][0]["iters"])
    out["layerwise"] = {
        "n_seeds": len(lw_seeds),
        "config_n_iters_per_stage": lw["config"]["n_iters"],
        "stages": lw_stages,
        "total_adam_steps_per_seed": lw_iters_recorded,
        "seeds": lw_seeds,
        "per_seed_final_loss": [lw_finals[s] for s in lw_seeds],
        "mean": float(np.mean([lw_finals[s] for s in lw_seeds])),
    }

    # --- SPSA (the e2a optimizer-comparison run: 5 seeds, 2000 iters) ------
    sp = _load("train_spsa_b1_e2a_spsa.json")
    sp_seeds = [r["seed"] for r in sp["runs"]]
    near = {r["seed"]: _spsa_loss_exact_near_200(r) for r in sp["runs"]}
    sp_iter_used = {s: near[s][0] for s in sp_seeds}
    sp_loss_at_200 = {s: near[s][1] for s in sp_seeds}
    sp_final_2000 = {r["seed"]: float(r["final_loss"]) for r in sp["runs"]}
    out["spsa"] = {
        "n_seeds": len(sp_seeds),
        "config_n_iters": sp["config"]["n_iters"],
        "seeds": sp_seeds,
        "loss_exact_iter_used": [sp_iter_used[s] for s in sp_seeds],
        "loss_exact_near_200_note": "record nearest iter 200 is iter 201; "
                                    "the exact loss after 200 SPSA updates "
                                    "(loss_exact is evaluated before each update "
                                    "and logged every 10 iters).",
        "per_seed_loss_exact_near_200": [sp_loss_at_200[s] for s in sp_seeds],
        "mean_loss_exact_near_200": float(np.mean(
            [sp_loss_at_200[s] for s in sp_seeds])),
        "per_seed_final_loss_at_2000": [sp_final_2000[s] for s in sp_seeds],
        "mean_final_loss_at_2000": float(np.mean(
            [sp_final_2000[s] for s in sp_seeds])),
    }

    # --- FD-vs-PS relative agreement on shared seeds 0..4 ------------------
    shared = [s for s in ps_seeds if s in fd_finals]
    rel_agree = {}
    for s in shared:
        a, b = fd_finals[s], ps_finals[s]
        rel_agree[s] = abs(a - b) / (abs(b) + 1e-300)
    fd_shared_mean = float(np.mean([fd_finals[s] for s in shared]))
    ps_shared_mean = float(np.mean([ps_finals[s] for s in shared]))
    out["fd_vs_ps_shared_seeds"] = {
        "shared_seeds": shared,
        "per_seed_rel_agreement": {str(s): rel_agree[s] for s in shared},
        "max_rel_agreement": float(max(rel_agree.values())),
        "fd_mean_shared": fd_shared_mean,
        "ps_mean_shared": ps_shared_mean,
        "note": "forward FD (eps=1e-4) into the identical Adam loop is "
                "numerically the same optimizer as exact parameter-shift; "
                "per-seed finals agree to ~1e-5 relative. FD and PS are a "
                "consistency check, not two independent optimizers. The FD "
                "row's higher mean is entirely the extra 5 seeds (5..9).",
    }

    # --- Honest loss-EVALUATION counts per optimizer -----------------------
    # Read from the harness control flow (r1_experiments.py) at L=3, 200 iters:
    #   FD  (cmd_train fd branch -> fd_gradient): per iter = 1 base +
    #       n_params forward evals = n_params + 1 = 46 loss evals/iter;
    #       x 200 = 9200 loss evals per seed.
    #   PS  (cmd_train ps branch -> ps_loss_gradient): per iter = 1 loss eval
    #       + a parameter-shift Jacobian evaluated at t, t+eps, t-eps for each
    #       collocation point. The harness's OWN wall-time estimator
    #       (estimate_train_wall_s, ps branch) charges 1 + 4*n_params = 181
    #       loss-eval-equivalents/iter -> 181 x 200 = 36200. Counting raw
    #       z_vector calls it is even higher (1 + 6*n_params per iter). Either
    #       way PS is ~4x more expensive per iteration than FD; it is NOT
    #       budget-matched to FD in loss evaluations -- iteration-matched only.
    #   layerwise: per iter = 1 base + 15 (active slice) = 16 loss evals; total
    #       iters = stages*n_iters = 3*200 = 600 -> 16 x 600 = 9600 loss evals.
    #   SPSA: per iter = 2 perturbed loss evals; + 1 loss_exact every K=10 and
    #       at the final iter; + 5 calibration probes once at theta_0. Read out
    #       at iter 200: 2*200 + (loss_exact at iters 0,10,...,190 -> 20) + 5
    #       calibration = 425 loss evals -- ~20x cheaper than FD. (The full
    #       2000-iter run is far more.)
    npar_L3 = 3 * 15  # 45
    fd_evals = (npar_L3 + 1) * 200                      # 46 * 200 = 9200
    ps_evals_harness_estimator = (1 + 4 * npar_L3) * 200  # 181 * 200 = 36200
    ps_evals_zvector_calls = (1 + 6 * npar_L3) * 200    # raw z_vector calls
    lw_evals = 16 * (lw_stages * lw["config"]["n_iters"])  # 16 * 600 = 9600
    spsa_evals_at_200 = 2 * 200 + (200 // 10) + 5  # +loss_exact every 10 +calib
    out["loss_evaluation_counts"] = {
        "L": 3,
        "n_params": npar_L3,
        "fd_forward_evals": fd_evals,
        "ps_evals_harness_estimator_unit": ps_evals_harness_estimator,
        "ps_zvector_calls_raw": ps_evals_zvector_calls,
        "layerwise_evals": lw_evals,
        "spsa_loss_evals_at_iter_200": spsa_evals_at_200,
        "note": "Iteration-matching (200) is NOT compute-matching. Only FD "
                "(9200) and layerwise (9600) are matched to ~1.04x in loss "
                "evaluations. PS costs ~4x more per iteration (36200 in the "
                "harness's own loss-eval unit) because it evaluates a "
                "parameter-shift Jacobian at a 3-point time stencil each "
                "iteration; it is iteration-matched but NOT budget-matched to "
                "FD. SPSA at iter 200 is only ~425 loss evals (~20x cheaper "
                "than FD) and must be disclosed as such. Layerwise ran 600 "
                "Adam steps (200/stage x 3 stages), 3x the stated 200-iteration "
                "budget. Honest framing: the comparison is ITERATION-matched, "
                "not compute-matched; report per-optimizer eval counts.",
    }
    return out


# ==========================================================================
# (d) Jacobian rank table exact values.
# ==========================================================================

def analyze_jacobian_ranks() -> dict:
    out = {"per_L": {}}
    for L in range(1, 9):
        d = _load(f"jacobian_b1_e3_jac_L{L}.json")
        ranks = [e["rank_1e-6"] for e in d["entries"]]
        prs = [e["participation_ratio"] for e in d["entries"]]
        shape = d["entries"][0]["jacobian_shape"]
        out["per_L"][L] = {
            "per_init_ranks": ranks,
            "mean_rank": float(np.mean(ranks)),
            "per_init_participation_ratio": prs,
            "mean_participation_ratio": float(np.mean(prs)),
            "jacobian_shape": shape,
            "n_inits": len(ranks),
        }
    # The corrected closure: saturated rank 33 = 3 observables x (1 + 2*5)
    # basis functions of the corrected band {0, .5, 1, 1.5, 2, 2.5}/t_max
    # (5 nonzero half-integer harmonics + DC). NOTE (honesty): this exact
    # 3*(1+2*m) identity holds only at SATURATION (L>=5); at low depth the
    # three observables have DIFFERENT reachable sets, so the intermediate
    # ranks 5/15/25 are NOT 3*(1+2*m) for a single m -- see the fourier
    # verification for the per-depth, per-observable reachable sets.
    out["saturation_closure"] = {
        "saturated_rank": 33,
        "decomposition": "3 observables x (1 + 2*5) = 3 x 11 = 33",
        "band": "{0, 0.5, 1, 1.5, 2, 2.5}/t_max (DC + 5 half-integer harmonics)",
        "caveat": "The 3*(1+2*m) identity is exact only at saturation (L>=5). "
                  "The intermediate ranks 5/15/25 at L=1/2/3 are NOT a single "
                  "3*(1+2*m): the reachable harmonic set grows with depth AND "
                  "differs across the three observables at low depth. See "
                  "fourier_ceiling_verification.json for per-depth sets.",
    }
    return out


# ==========================================================================
# Assemble + save.
# ==========================================================================

def main():
    payload = {
        "description": "Post-hoc corrections analysis for the MLST "
                       "resubmission, computed from results/r1/*.json only "
                       "(no experiment re-run).",
        "e1_live_dead": analyze_variance_live_dead(),
        "capacity_dispersion": analyze_capacity_dispersion(),
        "optimizer_comparison": analyze_optimizer_comparison(),
        "jacobian_ranks": analyze_jacobian_ranks(),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      default=float).encode("utf-8")
    digest = hashlib.sha256(blob).hexdigest()
    payload["sha256"] = digest
    out_path = RESULTS / "r1_corrections_analysis.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float))

    # --- printed summary ---------------------------------------------------
    print(f"[saved] {out_path}")
    print(f"[sha256] {digest}")
    print()
    ld = payload["e1_live_dead"]
    print("=== (a) E1 live/dead split ===")
    for L in (1, 3, 5):
        r = ld["per_L"][L]
        print(f"  L={L}: dead_frac={r['dead_fraction']*100:.1f}%  "
              f"pooled_mean={r['pooled_mean_var']:.3e}  "
              f"live_mean={r['live_mean_var']:.3e}  "
              f"(harness={r['harness_mcclean_mean']:.3e}, "
              f"diff={r['pooled_vs_harness_abs_diff']:.1e})  "
              f"max|dead grad|={r['max_abs_gradient_dead_cells']:.1e}")
    print(f"  live decay L1/L5 = {ld['live_decay_ratio_L1_over_L5']:.2f}x  "
          f"(pooled L1/L5 = {ld['pooled_ratio_L1_over_L5']:.2f}x)")
    print(f"  Haar floor 2^-8 = {ld['haar_saturation_scale']['value']:.3e}; "
          f"live in Haar units: "
          f"{ {L: round(v,2) for L,v in ld['live_mean_in_haar_units'].items()} }")
    print()
    print("=== (b) capacity dispersion ===")
    for L in (1, 2, 3, 5, 8):
        r = payload["capacity_dispersion"]["per_L"][L]
        print(f"  L={L}: mean={r['mean']:.1f}  sem={r['sem']:.1f}  "
              f"std={r['std_ddof1']:.1f}  seeds={r['per_seed_final_loss']}")
    print()
    print("=== (c) optimizer comparison ===")
    oc = payload["optimizer_comparison"]
    print(f"  FD    mean={oc['fd']['mean']:.1f} (n={oc['fd']['n_seeds']})")
    print(f"  PS    mean={oc['ps']['mean']:.1f} (n={oc['ps']['n_seeds']})")
    print(f"  LW    mean={oc['layerwise']['mean']:.1f} "
          f"({oc['layerwise']['total_adam_steps_per_seed']} Adam steps/seed)")
    print(f"  SPSA  loss_exact@~200 mean={oc['spsa']['mean_loss_exact_near_200']:.1f} "
          f"(iter {oc['spsa']['loss_exact_iter_used'][0]}); "
          f"@2000 mean={oc['spsa']['mean_final_loss_at_2000']:.1f}")
    print(f"  FD vs PS max rel agreement (shared seeds): "
          f"{oc['fd_vs_ps_shared_seeds']['max_rel_agreement']:.1e}")
    ec = oc["loss_evaluation_counts"]
    print(f"  eval counts: FD={ec['fd_forward_evals']} "
          f"PS(harness unit)={ec['ps_evals_harness_estimator_unit']} "
          f"LW={ec['layerwise_evals']} SPSA@200={ec['spsa_loss_evals_at_iter_200']}")
    print()
    print("=== (d) jacobian ranks ===")
    for L in range(1, 9):
        r = payload["jacobian_ranks"]["per_L"][L]
        print(f"  L={L}: ranks={r['per_init_ranks']} mean={r['mean_rank']:.4f} "
              f"PR={r['mean_participation_ratio']:.3f} shape={r['jacobian_shape']}")


if __name__ == "__main__":
    main()
