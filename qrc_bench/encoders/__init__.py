"""Classical data -> qubit rotation angles.

per_series  one qubit per series: R_Y(clip(scale * x)) |0>; angles (T, n_series)
dense_rxrz  two series per qubit: R_Z(b) R_Y(a) |0> with a, b the pair's scaled values
            (an odd series count is padded with 0); angles (T, ceil(n_series / 2), 2)
Measurement feedback shifts only the R_Y angle.
"""
from __future__ import annotations

import math

import numpy as np

ENCODINGS = ("per_series", "dense_rxrz")


def angle_ry(X: np.ndarray, scale: float) -> np.ndarray:
    """One R_Y angle per series per step: clip(scale * x, -pi, pi). Shape (T, n_series)."""
    return np.clip(X * scale, -np.pi, np.pi)


def dense_rxrz(X: np.ndarray, scale: float) -> np.ndarray:
    T, n = X.shape
    A = np.clip(X * scale, -np.pi, np.pi)
    if n % 2:
        A = np.hstack([A, np.zeros((T, 1))])
    return A.reshape(T, -1, 2)


def n_encoded_qubits(n_series: int, encoding: str) -> int:
    if encoding == "per_series":
        return n_series
    if encoding == "dense_rxrz":
        return math.ceil(n_series / 2)
    raise ValueError(f"unknown encoding {encoding!r}; expected one of {ENCODINGS}")


def encode(X: np.ndarray, encoding: str, scale: float) -> np.ndarray:
    n_encoded_qubits(X.shape[1], encoding)              # validates the name
    return angle_ry(X, scale) if encoding == "per_series" else dense_rxrz(X, scale)


def shift_ry(angles, shift):
    """Add ``shift`` to the R_Y angle of every qubit (feedback)."""
    if angles.ndim == 2:
        return angles + shift
    out = angles.copy()
    out[..., 0] = out[..., 0] + shift
    return out
