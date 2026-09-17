"""What is compared (Comparison), how it is tuned (Protocol), how it is simulated (Sim),
and each model's search space and feature builder.

A Comparison fixes the structure every model shares, so a comparison is never won by
seeing more input or having more features:
  window     reset-memory QRC (no feedback) vs random features, Poly2, linear:
             all see exactly the last ``input_window`` steps; QRC and random features have
             the same width. Poly2 and linear widths follow from the window (reported).
  recurrent  recurrent-memory QRC (feedback tunable) vs ESN, both see the whole history
             with the same width; window models are references at ``input_window``.
Every model draws its ridge penalty from ALPHA_RANGE; the rest are the model's own knobs.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

from qrc_bench import registry
from qrc_bench.baselines import esn_features, linear_features, poly2_features, random_features
from qrc_bench.encoders import encode, n_encoded_qubits
from qrc_bench.qrc import qrc_features
from qrc_bench.readouts import readout_width

ALPHA_RANGE = (1e-6, 1e2)
KINDS = ("window", "recurrent")
WINDOW_MODELS = ("qrc", "random_features", "poly2", "linear")
RECURRENT_ONLY = ("esn",)


@dataclass(frozen=True)
class Comparison:
    kind: str
    n_series: int
    input_window: int
    n_mem: int
    readout: str = "ZZ"
    n_taus: int = 1
    reservoir: str = "ising_xx"
    encoding: str = "per_series"
    feedback: bool = True          # recurrent kind only

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown comparison kind {self.kind!r}; expected one of {KINDS}")

    @property
    def n_in(self) -> int:
        return n_encoded_qubits(self.n_series, self.encoding)

    @property
    def q(self) -> int:
        return self.n_in + self.n_mem

    @property
    def width(self) -> int:
        return readout_width(self.readout, self.q) * self.n_taus

    def default_models(self) -> tuple[str, ...]:
        return WINDOW_MODELS if self.kind == "window" else ("qrc", "esn") + WINDOW_MODELS[1:]

    def check_models(self, models):
        unknown = [m for m in models if m not in MODELS]
        if unknown:
            raise ValueError(f"unknown models {unknown}; expected from {sorted(MODELS)}")
        if self.kind == "window":
            bad = [m for m in models if m in RECURRENT_ONLY]
            if bad:
                raise ValueError(f"{bad} see the whole history; not allowed in a window comparison")

    def as_dict(self):
        return asdict(self) | {"n_in": self.n_in, "q": self.q, "width": self.width}


@dataclass(frozen=True)
class Protocol:
    task: str
    task_kwargs: dict = field(default_factory=dict)
    n_steps: int = 1200
    horizon: int = 1
    tune_data_seeds: tuple = (100, 101, 102)
    tune_res_seeds: tuple = (100, 101, 102)
    eval_data_seeds: tuple = (0, 1, 2, 3, 4)
    eval_res_seeds: tuple = (0, 1, 2, 3, 4)
    n_trials: int = 100
    sampler_seed: int = 0
    n_folds: int = 3
    val_frac: float = 0.10
    test_frac: float = 0.15
    washout: int = 50
    standardize: bool = True

    def __post_init__(self):
        for a, b, what in ((self.tune_data_seeds, self.eval_data_seeds, "data"),
                           (self.tune_res_seeds, self.eval_res_seeds, "reservoir")):
            if set(a) & set(b):
                raise ValueError(f"tuning and evaluation {what} seeds overlap: {sorted(set(a) & set(b))}")

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Sim:
    method: str = "batched"
    backend: str = "auto"
    precision: str = "auto"

    def as_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class Model:
    space: Callable        # (trial, cmp) -> params (ridge alpha excluded)
    build: Callable        # (params, X, seed, cmp, sim) -> features (T, D)
    uses_seed: bool        # False: features do not depend on the reservoir / random seed


def suggest_alpha(trial) -> float:
    return trial.suggest_float("alpha", *ALPHA_RANGE, log=True)


def _qrc_space(trial, cmp: Comparison) -> dict:
    p = {"tau": trial.suggest_float("tau", 0.1, 10.0, log=True),
         "scale": trial.suggest_float("scale", 0.1, np.pi, log=True),
         "reservoir": registry.get("reservoir", cmp.reservoir).suggest(trial)}
    p["k_fb"] = trial.suggest_float("k_fb", 0.0, 8.0) if cmp.kind == "recurrent" and cmp.feedback else 0.0
    return p


def _qrc_build(p, X, seed, cmp: Comparison, sim: Sim):
    res = registry.get("reservoir", cmp.reservoir)(cmp.q, seed, **p["reservoir"])
    taus = tuple(p["tau"] / 2 ** k for k in range(cmp.n_taus))
    memory = "reset" if cmp.kind == "window" else "recurrent"
    return qrc_features(encode(X, cmp.encoding, p["scale"]), cmp.n_in, cmp.n_mem, res, taus=taus, readout=cmp.readout,
                        mem_depth=cmp.input_window, k_fb=p["k_fb"], memory=memory, method=sim.method,
                        backend=sim.backend, precision=sim.precision)


def _esn_space(trial, cmp):
    return {"sr": trial.suggest_float("sr", 0.05, 1.5),
            "leak": trial.suggest_float("leak", 0.01, 1.0, log=True),
            "in_scale": trial.suggest_float("in_scale", 0.01, 5.0, log=True),
            "density": trial.suggest_float("density", 0.02, 1.0)}


def _esn_build(p, X, seed, cmp, sim):
    return esn_features(X, cmp.width, seed, sr=p["sr"], leak=p["leak"], in_scale=p["in_scale"],
                        density=p["density"])


def _rf_space(trial, cmp):
    return {"scale": trial.suggest_float("scale", 0.05, 10.0, log=True),
            "bias_scale": trial.suggest_float("bias_scale", 0.0, 3.0)}


def _rf_build(p, X, seed, cmp, sim):
    return random_features(X, cmp.width, seed, window=cmp.input_window, scale=p["scale"],
                           bias_scale=p["bias_scale"])


MODELS = {
    "qrc": Model(_qrc_space, _qrc_build, uses_seed=True),
    "esn": Model(_esn_space, _esn_build, uses_seed=True),
    "random_features": Model(_rf_space, _rf_build, uses_seed=True),
    "poly2": Model(lambda trial, cmp: {}, lambda p, X, s, cmp, sim: poly2_features(X, cmp.input_window), False),
    "linear": Model(lambda trial, cmp: {}, lambda p, X, s, cmp, sim: linear_features(X, cmp.input_window), False),
}


def build_streams(model: str, params: dict, Xs, seed, cmp: Comparison, sim: Sim) -> list:
    """Features for several series with one seed. The QRC runs all series as one batch
    (same reservoir, equal lengths); other models are built one series at a time."""
    if model != "qrc" or len({len(X) for X in Xs}) != 1:
        return [MODELS[model].build(params, X, seed, cmp, sim) for X in Xs]
    res = registry.get("reservoir", cmp.reservoir)(cmp.q, seed, **params["reservoir"])
    taus = tuple(params["tau"] / 2 ** k for k in range(cmp.n_taus))
    memory = "reset" if cmp.kind == "window" else "recurrent"
    ang = np.stack([encode(X, cmp.encoding, params["scale"]) for X in Xs])
    F = qrc_features(ang, cmp.n_in, cmp.n_mem, res, taus=taus, readout=cmp.readout, mem_depth=cmp.input_window,
                     k_fb=params["k_fb"], memory=memory, method=sim.method, backend=sim.backend,
                     precision=sim.precision, streams=True)
    return list(F)
