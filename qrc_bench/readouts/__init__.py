"""Readouts computed from diag(rho): Z-basis one- and two-point expectations."""
from __future__ import annotations

import numpy as np

READOUTS = ("Z", "ZZ")


def readout_matrix(D: np.ndarray, kind: str, z_signs: np.ndarray, zz_signs: np.ndarray) -> np.ndarray:
    """D: (T, 2^q) diagonals. Z -> q features; ZZ -> q + q(q-1)/2 features."""
    zc = D @ z_signs.T
    if kind == "Z":
        return zc
    if kind == "ZZ":
        return np.concatenate([zc, D @ zz_signs.T], axis=1)
    raise ValueError(f"unknown readout {kind!r}; expected one of {READOUTS}")


def readout_width(kind: str, q: int) -> int:
    return q if kind == "Z" else q + q * (q - 1) // 2
