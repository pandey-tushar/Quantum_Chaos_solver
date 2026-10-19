"""Tune every model under one protocol, freeze its best setting, evaluate once on fresh seeds.

Shared by every model: TPE sampler and seed, tuning seeds, rolling-origin folds, washout, feature
standardisation, objective (mean validation NRMSE over tuning data seeds x tuning reservoir seeds x
folds) and ridge alpha range. Evaluation seeds never enter tuning.

The trial budget is per model, not shared: an equal count would search a one-knob model exhaustively
and a five-knob model sparsely, so each model gets ``trials_per_dim`` trials per knob it tunes and
keeps going while it is still finding new bests. ``best_so_far`` records the convergence curve, so
"tuned until tuning stopped helping" is evidence rather than an assumption. With ``refine="spsa"``
every model's best TPE trial is then refined by the same SPSA local search, run as ordinary trials.
"""
from __future__ import annotations

import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.distributions import distribution_to_json

from qrc_bench import registry
from qrc_bench.evaluate import fold_bounds, persistence_test, score_test, val_nrmse_folds
from qrc_bench.stats import holm, paired_comparison
from qrc_bench.tuning.spaces import ALPHA_RANGE, MODELS, Comparison, Protocol, Sim, build_trial, suggest_alpha


BUDGET_KEYS = ("n_trials", "trials_per_dim", "patience", "max_trials")


def _complete(study):
    """Completed TPE trials; the SPSA trials of the refine stage are not part of the TPE budget."""
    return [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and "refine" not in t.user_attrs]


def _best(trials):
    return min(trials, key=lambda t: t.value)


def _run_budget(study, objective, proto: Protocol) -> dict:
    """Trials for one model. A fixed ``n_trials`` gives every model the same budget, which
    favours the smallest search space; the default rule instead gives each model
    ``trials_per_dim`` trials per tuned knob and keeps going while it is still improving.
    """
    if proto.n_trials is not None:
        remaining = proto.n_trials - len([t for t in study.trials if t.state.is_finished()
                                          and "refine" not in t.user_attrs])
        if remaining > 0:
            study.optimize(objective, n_trials=remaining)
        n_dims = len(_complete(study)[0].distributions) if _complete(study) else 0
        return {"rule": "fixed", "n_dims": n_dims, "floor": proto.n_trials,
                "patience": None, "max_trials": proto.n_trials, "stopped": "fixed"}

    if not _complete(study):                       # one trial reveals the search space
        study.optimize(objective, n_trials=1)
    n_dims = len(_complete(study)[0].distributions)
    floor = min(proto.trials_per_dim * n_dims, proto.max_trials)
    state = {"stopped": "max_trials"}

    def stop_when_converged(st, trial):
        done = len(_complete(st))
        if done >= proto.max_trials:
            st.stop()
        elif done >= floor and trial.number - _best(_complete(st)).number >= proto.patience:
            state["stopped"] = "patience"
            st.stop()

    done = _complete(study)
    if len(done) >= proto.max_trials:
        pass                                       # a resumed study that already hit the cap
    elif len(done) >= floor and done[-1].number - _best(done).number >= proto.patience:
        state["stopped"] = "patience"              # a resumed study that had already converged
    else:
        remaining = proto.max_trials - len([t for t in study.trials if t.state.is_finished()
                                            and "refine" not in t.user_attrs])
        study.optimize(objective, n_trials=remaining, callbacks=[stop_when_converged])
    return {"rule": "per_dim", "n_dims": n_dims, "floor": floor, "trials_per_dim": proto.trials_per_dim,
            "patience": proto.patience, "max_trials": proto.max_trials, "stopped": state["stopped"]}


def _bounds(d):
    return (np.log10(d.low), np.log10(d.high)) if d.log else (d.low, d.high)


def _from_cube(u, d) -> float:
    """Unit-cube coordinate to a FloatDistribution value, uniform in log10 for a log knob."""
    lo, hi = _bounds(d)
    x = lo + u * (hi - lo)
    return float(np.clip(10 ** x if d.log else x, d.low, d.high))


def _to_cube(x, d) -> float:
    lo, hi = _bounds(d)
    return float(((np.log10(x) if d.log else x) - lo) / (hi - lo))


def _spsa(f, x0, budget: int, seed: int):
    """SPSA on the unit cube from ``x0``: two evaluations per iteration, ``budget // 2`` iterations.
    The step is normalised (a_k in cube units), so it does not depend on the objective's scale.
    A non-finite value counts as the worst finite value seen (1e3 before any). Returns the best
    evaluated point, its value and every evaluated point in order."""
    rng = np.random.default_rng(seed)
    theta = np.asarray(x0, dtype=float)
    best_x, best_v, points, seen = theta, np.inf, [], []
    for k in range(budget // 2):
        delta = rng.choice([-1.0, 1.0], size=theta.size)
        c = 0.05 / (k + 1) ** 0.101
        vals = []
        for x in (np.clip(theta + c * delta, 0, 1), np.clip(theta - c * delta, 0, 1)):
            v = f(x)
            points.append(x)
            if np.isfinite(v):
                seen.append(v)
                if v < best_v:
                    best_x, best_v = x, v
            else:
                v = max(seen, default=1e3)
            vals.append(v)
        g = (vals[0] - vals[1]) / (2 * c) * delta
        theta = np.clip(theta - 0.1 / (k + 1 + 10) ** 0.602 * g / (np.linalg.norm(g) + 1e-12), 0, 1)
    return best_x, best_v, points


def _refine(study, objective, proto: Protocol, floor: int) -> dict:
    """SPSA from the best TPE trial, every evaluation an ordinary trial marked ``refine``. On resume
    the stored refine trials are replayed in order (SPSA is deterministic) rather than rerun."""
    if proto.refine == "none":
        return {"refine_evals": 0, "refine_gain": 0.0}
    start = _best(_complete(study))
    dists = start.distributions
    refine = [t for t in study.trials if "refine" in t.user_attrs]
    stored = iter([t for t in refine if t.state == optuna.trial.TrialState.COMPLETE])
    # an interrupted run can leave its last refine trial enqueued but never run
    waiting = [t for t in refine if t.state == optuna.trial.TrialState.WAITING]

    def f(u):
        params = {k: _from_cube(x, d) for x, (k, d) in zip(u, dists.items())}
        t = next(stored, None)
        if t is None:
            if not waiting:
                study.enqueue_trial(params, user_attrs={"refine": True})
            elif waiting.pop().system_attrs["fixed_params"] != params:
                raise ValueError("a waiting refine trial does not match the SPSA point; use a new --out directory")
            study.optimize(objective, n_trials=1)
            t = study.trials[-1]
        elif t.params != params:
            raise ValueError("stored refine trials were started from a different TPE best; use a new --out directory")
        return t.value

    x0 = [_to_cube(start.params[k], d) for k, d in dists.items()]
    points = _spsa(f, x0, int(proto.refine_frac * floor), proto.sampler_seed)[2]
    gain = start.value - study.best_value if np.isfinite([start.value, study.best_value]).all() else 0.0
    return {"refine_evals": len(points), "refine_gain": gain}


_SERIES_CACHE: dict = {}


def _series(proto: Protocol, data_seed: int):
    """One array per (task, length, seed, kwargs): every model of an experiment tunes and evaluates
    on the same series, and generating one (e.g. Lorenz-96 at 2000 x 25 RK4 steps) costs seconds."""
    key = (proto.task, proto.n_steps, data_seed, json.dumps(proto.task_kwargs, sort_keys=True))
    if key not in _SERIES_CACHE:
        _SERIES_CACHE[key] = registry.get("task", proto.task)(proto.n_steps, data_seed, **proto.task_kwargs)
    return _SERIES_CACHE[key]


def _check_width(model, F, cmp):
    if model in ("qrc", "esn", "random_features") and F.shape[1] != cmp.width:
        raise AssertionError(f"{model} built {F.shape[1]} features, comparison width is {cmp.width}")


def tune_model(model: str, cmp: Comparison, proto: Protocol, sim: Sim, storage: str | None = None,
               log=print) -> dict:
    """One Optuna study; with ``storage`` (sqlite:///file.db) the study is named after the model and resumes.
    The record's ``n_trials``, ``trial_values`` and ``best_so_far`` include the refine trials."""
    cmp.check_models([model])
    spec = MODELS[model]
    data = {ds: _series(proto, ds) for ds in proto.tune_data_seeds}
    res_seeds = proto.tune_res_seeds if spec.uses_seed else (None,)
    folds = fold_bounds(proto.n_steps, proto.n_folds, proto.val_frac, proto.test_frac)
    fixed_cache = {}                                   # param-free models build once per data seed
    seeds = list(data)

    def objective(trial):
        t0 = time.perf_counter()
        p = spec.space(trial, cmp)
        alpha = suggest_alpha(trial)
        per_run = {}
        # every tuning series of every reservoir seed of this trial in one batched build
        todo = [rs for rs in res_seeds if p or rs not in fixed_cache]
        built = build_trial(model, p, [data[ds] for ds in seeds], todo, cmp, sim) if todo else {}
        for rs in res_seeds:
            if not p and rs in fixed_cache:
                Fs = fixed_cache[rs]
            else:
                Fs = [built[(rs, i)] for i in range(len(seeds))]
                for F in Fs:
                    _check_width(model, F, cmp)
                if not p:
                    fixed_cache[rs] = Fs
            for ds, F in zip(seeds, Fs):
                try:
                    v = val_nrmse_folds(F, data[ds], proto.horizon, alpha, folds, proto.standardize, proto.washout)
                except np.linalg.LinAlgError:
                    v = float("inf")
                per_run[f"ds{ds}_rs{rs}"] = v if np.isfinite(v) else float("inf")
        trial.set_user_attr("params", p)
        trial.set_user_attr("per_run", per_run)
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(np.mean(list(per_run.values())))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=proto.sampler_seed),
                                storage=storage, study_name=model if storage else None,
                                load_if_exists=storage is not None)
    fingerprint = json.dumps({"comparison": cmp.as_dict(), "sim": sim.as_dict(),
                              "protocol": {k: v for k, v in proto.as_dict().items() if k not in BUDGET_KEYS
                                           and not (proto.refine == "none" and k in ("refine", "refine_frac"))}},
                             sort_keys=True, default=list)
    stored = study.user_attrs.get("fingerprint")
    if stored is None:
        study.set_user_attr("fingerprint", fingerprint)
    elif stored != fingerprint:
        raise ValueError(f"stored study {model!r} was created with a different comparison, protocol or sim; "
                         "use a new --out directory")
    t0 = time.perf_counter()
    budget = _run_budget(study, objective, proto)
    budget |= _refine(study, objective, proto, budget["floor"])
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial
    space = {}
    for t in study.trials:
        space.update({k: json.loads(distribution_to_json(d)) for k, d in t.distributions.items()})
    rec = {"model": model, "n_trials": len(done), "sampler": "TPE+SPSA" if proto.refine == "spsa" else "TPE",
           "sampler_seed": proto.sampler_seed, "search_space": space, "budget": budget,
           "best_params": best.user_attrs["params"], "best_alpha": best.params["alpha"],
           "best_trial": best.number, "best_val": best.value, "best_val_per_run": best.user_attrs["per_run"],
           "trial_values": [t.value for t in done], "trial_seconds": [t.user_attrs["seconds"] for t in done],
           "best_so_far": np.minimum.accumulate([t.value for t in done]).tolist(),
           "study_seconds": round(time.perf_counter() - t0, 3)}
    log(f"  tuned {model:15s} val {best.value:.4f}  trials {rec['n_trials']} "
        f"({budget['n_dims']}d, floor {budget['floor']}, stopped on {budget['stopped']})  "
        f"[{sum(rec['trial_seconds']):.0f}s in trials]"
        + (f" refine spsa: {budget['refine_evals']} evals, gain {budget['refine_gain']:.4g}"
           if proto.refine == "spsa" else ""))
    return rec


def evaluate_frozen(model: str, params: dict, alpha: float, cmp: Comparison, proto: Protocol, sim: Sim) -> list[dict]:
    """Frozen setting on every evaluation (data seed, reservoir seed): fit before the test slice, score once."""
    spec = MODELS[model]
    data = {ds: _series(proto, ds) for ds in proto.eval_data_seeds}
    seeds = list(data)
    # every evaluation series of every evaluation reservoir seed in one batched build; a model whose
    # features do not depend on the seed is built once and reused across the reservoir seeds
    res_seeds = list(proto.eval_res_seeds) if spec.uses_seed else list(proto.eval_res_seeds[:1])
    t0 = time.perf_counter()
    built = build_trial(model, params, [data[ds] for ds in seeds], res_seeds, cmp, sim)
    for F in built.values():
        _check_width(model, F, cmp)
    seconds = (time.perf_counter() - t0) / (len(seeds) * len(proto.eval_res_seeds))
    rows = {}
    for rs in proto.eval_res_seeds:
        for i, ds in enumerate(seeds):
            F = built[(rs if spec.uses_seed else res_seeds[0], i)]
            rows[(ds, rs)] = {"data_seed": ds, "res_seed": rs, "width": int(F.shape[1]),
                              "test_nrmse": score_test(F, data[ds], proto.horizon, alpha, proto.test_frac,
                                                       proto.standardize, proto.washout),
                              "build_seconds": round(seconds, 4)}
    return [rows[(ds, rs)] for ds in seeds for rs in proto.eval_res_seeds]


def environment() -> dict:
    import numpy
    import scipy
    env = {"python": platform.python_version(), "platform": platform.platform(), "numpy": numpy.__version__,
           "scipy": scipy.__version__, "optuna": optuna.__version__, "cupy": None, "gpu": None, "git_commit": None}
    try:
        import cupy
        env["cupy"] = cupy.__version__
        env["gpu"] = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        pass
    try:
        env["git_commit"] = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
                                           cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return env


def run_comparison(cmp: Comparison, proto: Protocol, sim: Sim, models=None, out_dir=None, resume: bool = False,
                   log=print) -> dict:
    models = list(models or cmp.default_models())
    cmp.check_models(models)
    if "qrc" not in models:
        raise ValueError("a comparison needs the qrc model")
    out_dir = Path(out_dir) if out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{out_dir / f'studies_{cmp.kind}.db'}" if out_dir and resume else None
    log(f"=== {cmp.kind} comparison: q={cmp.q} ({cmp.n_in}+{cmp.n_mem}), width {cmp.width}, "
        f"window {cmp.input_window}, shots {cmp.shots if cmp.shots else 'exact'}, models {models}")
    t0 = time.perf_counter()
    result = {"comparison": cmp.as_dict(), "protocol": proto.as_dict() | {"alpha_range": list(ALPHA_RANGE)},
              "sim": sim.as_dict(), "environment": environment(), "models": {}}
    for m in models:
        tuning = tune_model(m, cmp, proto, sim, storage=storage, log=log)
        ev = evaluate_frozen(m, tuning["best_params"], tuning["best_alpha"], cmp, proto, sim)
        result["models"][m] = {"tuning": tuning, "eval": ev}
    result["persistence"] = {str(ds): persistence_test(_series(proto, ds), proto.horizon, proto.test_frac)
                             for ds in proto.eval_data_seeds}
    q_rows = result["models"]["qrc"]["eval"]
    key = [(r["data_seed"], r["res_seed"]) for r in q_rows]
    stats = {}
    for m in models:
        if m == "qrc":
            continue
        other = {(r["data_seed"], r["res_seed"]): r["test_nrmse"] for r in result["models"][m]["eval"]}
        s = paired_comparison([r["test_nrmse"] for r in q_rows], [other[k] for k in key], [k[0] for k in key])
        # a model without a reservoir seed repeats one value across the reservoir seeds, so its runs
        # are not independent: only the cluster-level statistics are valid for that comparison
        s["run_level_independent"] = MODELS[m].uses_seed and MODELS["qrc"].uses_seed
        stats[f"qrc_vs_{m}"] = s
    if stats:
        names = list(stats)
        for label, field_ in (("p_holm_cluster", "p_t_cluster"), ("p_holm_run", "p_t_run")):
            ps = [stats[n][field_] for n in names]
            finite = [i for i, p in enumerate(ps) if np.isfinite(p)]
            adj = holm([ps[i] for i in finite]) if finite else []
            for n in names:
                stats[n][label] = float("nan")
            for i, a in zip(finite, adj):
                stats[names[i]][label] = a
    result["stats"] = stats
    result["total_seconds"] = round(time.perf_counter() - t0, 3)
    if out_dir:
        (out_dir / f"comparison_{cmp.kind}.json").write_text(json.dumps(result, indent=2))
    for n, s in stats.items():
        note = "" if s["run_level_independent"] else "  [run-level stats not independent]"
        log(f"  {n:22s} mean {s['mean_a']:.4f} vs {s['mean_b']:.4f}  diff {s['mean_diff']:+.4f}  "
            f"seed wins {s['cluster_wins_a']}/{s['n_clusters']}  p_cluster(Holm) {s['p_holm_cluster']:.3g}{note}")
    return result
