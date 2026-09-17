"""Lorenz-96: K coupled chaotic variables on a ring, a scalable multi-series task."""
from __future__ import annotations

import numpy as np

from qrc_bench.registry import register


def _l96_rhs(x, forcing):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + forcing


@register("task", "lorenz96")
def lorenz96(n_steps: int, seed: int, n_series: int = 10, forcing: float = 8.0, dt: float = 0.01,
             sample_every: int = 5, obs_noise: float = 0.0, burn: int = 1000):
    """dx_k/dt = (x_{k+1} - x_{k-2}) x_{k-1} - x_k + F, RK4 with step dt, one sample every
    ``sample_every`` steps after ``burn`` samples; optional observation noise; standardised."""
    rng = np.random.default_rng(seed)
    x = forcing + 0.01 * rng.standard_normal(n_series)
    out = np.zeros((n_steps + burn, n_series))
    for i in range(n_steps + burn):
        for _ in range(sample_every):
            k1 = _l96_rhs(x, forcing)
            k2 = _l96_rhs(x + 0.5 * dt * k1, forcing)
            k3 = _l96_rhs(x + 0.5 * dt * k2, forcing)
            k4 = _l96_rhs(x + dt * k3, forcing)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = x
    X = out[burn:]
    if obs_noise > 0:
        X = X + obs_noise * X.std(0) * rng.standard_normal(X.shape)
    return (X - X.mean(0)) / (X.std(0) + 1e-12)
