"""Search spaces and feature builders for each tunable model.

Every model draws its ridge penalty from the same ALPHA_RANGE; everything else is the
model's own knobs. Bounds live here only, and each study writes them to its output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from qrc_bench import registry
from qrc_bench.baselines import esn_features, linear_features, poly2_features, windowed
from qrc_bench.encoders import angle_ry
from qrc_bench.layout import Q_MAX
from qrc_bench.qrc import qrc_features

ALPHA_RANGE = (1e-6, 1e2)


@dataclass(frozen=True)
class Context:
    """What a study needs besides the trial: data and the fixed (not tuned) settings.

    ``window`` stacks the last steps of reservoir features (QRC and ESN);
    ``lag_window`` is the input window of Poly2 and linear.
    """
    X: np.ndarray
    res_seed: int
    horizon: int = 1
    window: int = 1
    lag_window: int = 1
    reservoir: str = "ising_xx"
    n_mem_range: tuple[int, int] = (1, 6)
    feedback: bool = True
    esn_units: int | None = None
    method: str = "branch"
    backend: str = "numpy"
    q_max: int = Q_MAX


@dataclass(frozen=True)
class Model:
    space: Callable        # (trial, ctx) -> params dict (ridge alpha excluded)
    build: Callable        # (params, ctx) -> features (T, D)
    defaults: dict         # partial trial to enqueue with --enqueue-defaults


def suggest_alpha(trial) -> float:
    return trial.suggest_float("alpha", *ALPHA_RANGE, log=True)


def _qrc_space(trial, ctx: Context) -> dict:
    n_in = ctx.X.shape[1]
    lo, hi = ctx.n_mem_range[0], min(ctx.n_mem_range[1], ctx.q_max - n_in)
    if hi < lo:
        raise ValueError(f"n_mem range {ctx.n_mem_range} is empty for n_in={n_in}, q_max={ctx.q_max}")
    return {"n_mem": trial.suggest_int("n_mem", lo, hi),
            "tau": trial.suggest_float("tau", 0.1, 10.0, log=True),
            "scale": trial.suggest_float("scale", 0.1, np.pi, log=True),
            "mem_depth": trial.suggest_int("mem_depth", 1, 5),
            "readout": trial.suggest_categorical("readout", ["Z", "ZZ"]),
            "two_taus": trial.suggest_categorical("two_taus", [False, True]),
            "k_fb": trial.suggest_float("k_fb", 0.0, 8.0) if ctx.feedback else 0.0,
            "reservoir": registry.get("reservoir", ctx.reservoir).suggest(trial)}


def _qrc_build(p: dict, ctx: Context) -> np.ndarray:
    n_in = ctx.X.shape[1]
    res = registry.get("reservoir", ctx.reservoir)(n_in + p["n_mem"], ctx.res_seed, **p["reservoir"])
    taus = (p["tau"], p["tau"] / 2.0) if p["two_taus"] else (p["tau"],)
    F = qrc_features(angle_ry(ctx.X, p["scale"]), n_in, p["n_mem"], res, taus=taus, readout=p["readout"],
                     mem_depth=p["mem_depth"], k_fb=p["k_fb"], method=ctx.method, backend=ctx.backend)
    return windowed(F, ctx.window)


def _esn_space(trial, ctx: Context) -> dict:
    return {"sr": trial.suggest_float("sr", 0.05, 1.5),
            "leak": trial.suggest_float("leak", 0.01, 1.0, log=True),
            "in_scale": trial.suggest_float("in_scale", 0.01, 5.0, log=True),
            "density": trial.suggest_float("density", 0.02, 1.0),
            "n_res": ctx.esn_units or trial.suggest_int("n_res", 10, 300, log=True)}


def _esn_build(p: dict, ctx: Context) -> np.ndarray:
    F = esn_features(ctx.X, p["n_res"], ctx.res_seed, sr=p["sr"], leak=p["leak"],
                     in_scale=p["in_scale"], density=p["density"])
    return windowed(F, ctx.window)


MODELS = {
    "qrc": Model(_qrc_space, _qrc_build,
                 {"tau": 1.0, "scale": 1.0, "mem_depth": 3, "readout": "ZZ", "two_taus": False, "alpha": 1e-3}),
    "esn": Model(_esn_space, _esn_build,
                 {"sr": 0.9, "leak": 0.3, "in_scale": 0.5, "density": 0.1, "alpha": 1e-3}),
    "poly2": Model(lambda trial, ctx: {}, lambda p, ctx: poly2_features(ctx.X, ctx.lag_window), {"alpha": 1e-3}),
    "linear": Model(lambda trial, ctx: {}, lambda p, ctx: linear_features(ctx.X, ctx.lag_window), {"alpha": 1e-3}),
}
