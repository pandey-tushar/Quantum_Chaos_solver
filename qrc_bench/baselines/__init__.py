"""Classical feature maps. Each takes X (T, n_series) and returns features (T, D)."""
from __future__ import annotations

import numpy as np

from qrc_bench.registry import register


def windowed(F: np.ndarray, window: int) -> np.ndarray:
    """Stack the last ``window`` rows (oldest first); the first rows repeat row 0."""
    out = []
    for t in range(len(F)):
        block = F[max(0, t - window + 1):t + 1]
        if len(block) < window:
            block = np.vstack([np.repeat(block[:1], window - len(block), 0), block])
        out.append(block.flatten())
    return np.array(out)


@register("baseline", "linear")
def linear_features(X: np.ndarray, window: int = 1) -> np.ndarray:
    return windowed(X, window)


@register("baseline", "poly2")
def poly2_features(X: np.ndarray, window: int = 3) -> np.ndarray:
    """Complete degree-2 basis on the window: n linear + n(n+1)/2 products (squares included)."""
    rows = []
    for v in windowed(X, window):
        rows.append(np.concatenate([v, np.concatenate([v[i] * v[i:] for i in range(len(v))])]))
    return np.array(rows)


@register("baseline", "random_features")
def random_features(X: np.ndarray, width: int, seed: int, window: int = 1, scale: float = 1.0,
                    bias_scale: float = 0.5) -> np.ndarray:
    """Fixed random nonlinear features of the input window: tanh(A z_t + b).

    z_t stacks the last ``window`` steps (n = n_series * window values); A ~ N(0, scale^2 / n),
    b ~ U(-bias_scale, bias_scale). The same-window, same-width control for a reset-memory QRC.
    """
    Z = windowed(X, window)
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((width, Z.shape[1])) * (scale / np.sqrt(Z.shape[1]))
    b = rng.uniform(-bias_scale, bias_scale, width)
    return np.tanh(Z @ A.T + b)


@register("baseline", "esn")
def esn_features(X: np.ndarray, n_res: int, seed: int, sr: float = 0.9, leak: float = 0.3,
                 in_scale: float = 0.5, density: float = 0.1) -> np.ndarray:
    """r_t = (1 - leak) r_{t-1} + leak * tanh(W r_{t-1} + W_in x_t), W rescaled to spectral radius sr."""
    rng = np.random.default_rng(seed)
    W_in = rng.uniform(-in_scale, in_scale, (n_res, X.shape[1]))
    W = rng.standard_normal((n_res, n_res)) * (rng.uniform(0, 1, (n_res, n_res)) < density)
    e = np.max(np.abs(np.linalg.eigvals(W)))
    if e > 1e-8:
        W *= sr / e
    r = np.zeros(n_res)
    out = np.zeros((len(X), n_res))
    for t in range(len(X)):
        r = (1 - leak) * r + leak * np.tanh(W @ r + W_in @ X[t])
        out[t] = r
    return out
