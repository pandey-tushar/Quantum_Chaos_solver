"""Run one Optuna study per model under an identical protocol.

Shared by every model: trial budget, TPE sampler and its seed, chronological split,
objective (validation NRMSE of the ridge readout), and the ridge alpha range.
The test slice is scored once, on the best trial, after the study ends.
"""
from __future__ import annotations

import json
import time

import numpy as np
import optuna
from optuna.distributions import distribution_to_json

from qrc_bench.evaluate import test_nrmse, val_nrmse
from qrc_bench.tuning.spaces import ALPHA_RANGE, MODELS, Context, suggest_alpha


def tune_model(model: str, ctx: Context, n_trials: int = 100, sampler_seed: int = 0,
               enqueue_defaults: bool = False, storage: str | None = None, log=print) -> dict:
    """With ``storage`` (e.g. sqlite:///file.db) the study is named after the model and resumes."""
    spec = MODELS[model]
    last = {"key": None, "F": None}          # one-entry cache: Poly2/linear build once

    def features(p):
        key = json.dumps(p, sort_keys=True)
        if key != last["key"]:
            last["key"], last["F"] = key, spec.build(p, ctx)
        return last["F"]

    def objective(trial):
        p = spec.space(trial, ctx)
        alpha = suggest_alpha(trial)
        F = features(p)
        trial.set_user_attr("params", p)
        trial.set_user_attr("width", int(F.shape[1]))
        try:
            v = val_nrmse(F, ctx.X, ctx.horizon, alpha)
        except np.linalg.LinAlgError:
            return float("inf")
        return v if np.isfinite(v) else float("inf")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=sampler_seed),
                                storage=storage, study_name=model if storage else None,
                                load_if_exists=storage is not None)
    if enqueue_defaults and not study.trials:
        study.enqueue_trial(spec.defaults, skip_if_exists=True)
    finished = [t for t in study.trials if t.state.is_finished()]    # enqueued trials are WAITING
    remaining = n_trials - len(finished)
    t0 = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    best = study.best_trial
    F = spec.build(best.user_attrs["params"], ctx)
    test = test_nrmse(F, ctx.X, ctx.horizon, best.params["alpha"])
    space = {}
    for t in study.trials:
        space.update({k: json.loads(distribution_to_json(d)) for k, d in t.distributions.items()})
    record = {"model": model, "n_trials": len(study.trials), "sampler": "TPE", "sampler_seed": sampler_seed,
              "search_space": space, "best_params": best.user_attrs["params"],
              "best_alpha": best.params["alpha"], "best_trial": best.number, "width": best.user_attrs["width"],
              "val_nrmse": best.value, "test_nrmse": test, "wall_time_s": round(time.time() - t0, 2)}
    log(f"  {model:7s} val {best.value:.4f}  test {test:.4f}  width {record['width']:4d}  "
        f"trials {record['n_trials']}  [{record['wall_time_s']:.0f}s]")
    return record


def tune_all(models, ctx: Context, n_trials: int = 100, sampler_seed: int = 0, **kw) -> dict:
    """Tune each model with the same budget and sampler; the record makes that auditable."""
    records = {m: tune_model(m, ctx, n_trials=n_trials, sampler_seed=sampler_seed, **kw) for m in models}
    assert len({r["n_trials"] for r in records.values()}) == 1, "unequal trial budgets"
    return {"protocol": {"n_trials": n_trials, "sampler": "TPE", "sampler_seed": sampler_seed,
                         "objective": "validation NRMSE, ridge readout", "split": "70/15/15 chronological",
                         "alpha_range": list(ALPHA_RANGE), "horizon": ctx.horizon, "window": ctx.window,
                         "lag_window": ctx.lag_window, "reservoir": ctx.reservoir, "res_seed": ctx.res_seed,
                         "method": ctx.method, "backend": ctx.backend},
            "models": records}
