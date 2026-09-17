"""Tune every model under one protocol, freeze its best setting, evaluate once on fresh seeds.

Shared by every model: trial budget, TPE sampler and seed, tuning seeds, rolling-origin folds,
washout, feature standardisation, objective (mean validation NRMSE over tuning data seeds x
tuning reservoir seeds x folds) and ridge alpha range. Evaluation seeds never enter tuning.
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
from qrc_bench.tuning.spaces import ALPHA_RANGE, MODELS, Comparison, Protocol, Sim, build_streams, suggest_alpha


def _series(proto: Protocol, data_seed: int):
    return registry.get("task", proto.task)(proto.n_steps, data_seed, **proto.task_kwargs)


def _check_width(model, F, cmp):
    if model in ("qrc", "esn", "random_features") and F.shape[1] != cmp.width:
        raise AssertionError(f"{model} built {F.shape[1]} features, comparison width is {cmp.width}")


def tune_model(model: str, cmp: Comparison, proto: Protocol, sim: Sim, storage: str | None = None,
               log=print) -> dict:
    """One Optuna study; with ``storage`` (sqlite:///file.db) the study is named after the model and resumes."""
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
        for rs in res_seeds:                          # all tuning series of one reservoir in one batch
            if not p and rs in fixed_cache:
                Fs = fixed_cache[rs]
            else:
                Fs = build_streams(model, p, [data[ds] for ds in seeds], rs, cmp, sim)
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
    t0 = time.perf_counter()
    remaining = proto.n_trials - len([t for t in study.trials if t.state.is_finished()])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = study.best_trial
    space = {}
    for t in study.trials:
        space.update({k: json.loads(distribution_to_json(d)) for k, d in t.distributions.items()})
    rec = {"model": model, "n_trials": len(done), "sampler": "TPE", "sampler_seed": proto.sampler_seed,
           "search_space": space, "best_params": best.user_attrs["params"], "best_alpha": best.params["alpha"],
           "best_trial": best.number, "best_val": best.value, "best_val_per_run": best.user_attrs["per_run"],
           "trial_values": [t.value for t in done], "trial_seconds": [t.user_attrs["seconds"] for t in done],
           "study_seconds": round(time.perf_counter() - t0, 3)}
    log(f"  tuned {model:15s} val {best.value:.4f}  trials {rec['n_trials']}  "
        f"[{sum(rec['trial_seconds']):.0f}s in trials]")
    return rec


def evaluate_frozen(model: str, params: dict, alpha: float, cmp: Comparison, proto: Protocol, sim: Sim) -> list[dict]:
    """Frozen setting on every evaluation (data seed, reservoir seed): fit before the test slice, score once."""
    spec = MODELS[model]
    data = {ds: _series(proto, ds) for ds in proto.eval_data_seeds}
    seeds = list(data)
    rows, fixed = {}, None
    for rs in proto.eval_res_seeds:                   # all evaluation series of one reservoir in one batch
        t0 = time.perf_counter()
        if spec.uses_seed or fixed is None:
            fixed = build_streams(model, params, [data[ds] for ds in seeds], rs, cmp, sim)
            for F in fixed:
                _check_width(model, F, cmp)
        seconds = (time.perf_counter() - t0) / len(seeds)
        for ds, F in zip(seeds, fixed):
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
        f"window {cmp.input_window}, models {models}")
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
        stats[f"qrc_vs_{m}"] = paired_comparison([r["test_nrmse"] for r in q_rows], [other[k] for k in key],
                                                 [k[0] for k in key])
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
        log(f"  {n:22s} mean {s['mean_a']:.4f} vs {s['mean_b']:.4f}  diff {s['mean_diff']:+.4f}  "
            f"wins {s['wins_a']}/{s['n_runs']}  p_cluster(Holm) {s['p_holm_cluster']:.3g}")
    return result
