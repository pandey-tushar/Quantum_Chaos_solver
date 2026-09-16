"""Classical data -> qubit rotation angles."""
from __future__ import annotations

import numpy as np


def angle_ry(X: np.ndarray, scale: float) -> np.ndarray:
    """One R_Y angle per series per step: clip(scale * x, -pi, pi). Shape (T, n_series)."""
    return np.clip(X * scale, -np.pi, np.pi)
