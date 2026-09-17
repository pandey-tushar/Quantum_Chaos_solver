"""Nonstationary AR(1) tasks (cartography paper, Case II and the drift task)."""
from __future__ import annotations

import numpy as np

from qrc_bench.registry import register


@register("task", "switching")
def regime_switching(n_steps: int, seed: int, p_switch: float = 0.05, phi: float = 0.85,
                     noise: float = 0.02, burn: int = 200):
    """x_{t+1} = s_t * phi * x_t + noise, hidden sign s_t = +-1 flips with prob p_switch."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5)
    s = 0
    xs = []
    for _ in range(n_steps + burn):
        if rng.uniform() < p_switch:
            s = 1 - s
        x = (phi if s == 0 else -phi) * x + noise * rng.standard_normal()
        x = min(max(x, -5.0), 5.0)
        xs.append(x)
    x = np.array(xs[burn:])
    return ((x - x.mean()) / (x.std() + 1e-12))[:, None]


@register("task", "switching_multi")
def regime_switching_multi(n_steps: int, seed: int, n_series: int = 9, p_switch: float = 0.05, phi: float = 0.85,
                           coupling: float = 0.1, noise: float = 0.02, burn: int = 200):
    """N series sharing one hidden sign s_t = +-1 (flips with prob p_switch):
    x_{i,t+1} = s_t * phi * x_{i,t} + coupling * (mean_j x_{j,t} - x_{i,t}) + noise_i."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, n_series)
    s = 1.0
    out = np.zeros((n_steps + burn, n_series))
    for t in range(n_steps + burn):
        if rng.uniform() < p_switch:
            s = -s
        x = s * phi * x + coupling * (x.mean() - x) + noise * rng.standard_normal(n_series)
        x = np.clip(x, -5.0, 5.0)
        out[t] = x
    X = out[burn:]
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


@register("task", "drift")
def ar_drift(n_steps: int, seed: int, period: int = 400, drift_amp: float = 0.85,
             phi_base: float = 0.0, noise: float = 0.02, burn: int = 200):
    """x_{t+1} = phi_t * x_t + noise, phi_t = phi_base + drift_amp * sin(2 pi t / period)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5)
    xs = []
    for t in range(n_steps + burn):
        phi = phi_base + drift_amp * np.sin(2 * np.pi * t / period)
        x = phi * x + noise * rng.standard_normal()
        x = min(max(x, -5.0), 5.0)
        xs.append(x)
    x = np.array(xs[burn:])
    return ((x - x.mean()) / (x.std() + 1e-12))[:, None]
