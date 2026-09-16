"""Coupled Henon maps (cartography paper, Case I)."""
from __future__ import annotations

import numpy as np

from qrc_bench.registry import register


@register("task", "henon")
def coupled_henon(n_steps: int, seed: int, n_series: int = 5, coupling: float = 0.1,
                  a: float = 1.4, b: float = 0.3, obs_noise: float = 0.1, burn: int = 200):
    """x_{i,t+1} = 1 - a*xc_i^2 + b*x_{i,t-1},  xc_i = (1 - c)*x_i + c*mean(x).

    Observation noise is added after the burn-in; each series is standardised.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, n_series)
    y = rng.uniform(-0.5, 0.5, n_series)
    out = np.zeros((n_steps + burn, n_series))
    for t in range(n_steps + burn):
        xc = (1 - coupling) * x + coupling * x.mean()
        xn = 1 - a * xc ** 2 + b * y
        y = x
        x = np.clip(xn, -2.0, 2.0)
        out[t] = x
    X = out[burn:]
    if obs_noise > 0:
        X = X + obs_noise * rng.standard_normal(X.shape)
    return (X - X.mean(0)) / (X.std(0) + 1e-12)
