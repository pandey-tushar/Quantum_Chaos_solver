"""Readouts computed from measured outcome distributions: Z-basis one- and two-point
expectations, optionally beside the same block measured in the non-commuting X basis."""
from __future__ import annotations

import numpy as np

READOUTS = ("Z", "ZZ", "ZX")
# the measurement bases each readout needs, in feature-block order. Z and ZZ commute, so one
# Z-basis distribution carries the whole "ZZ" block; "ZX" needs a second, X-basis distribution.
BASES = {"Z": ("Z",), "ZZ": ("Z",), "ZX": ("Z", "X")}


def readout_bases(kind: str) -> tuple[str, ...]:
    if kind not in BASES:
        raise ValueError(f"unknown readout {kind!r}; expected one of {READOUTS}")
    return BASES[kind]


def readout_matrix(D, kind: str, z_signs, zz_signs, xp=np):
    """D: (T, 2^q) diagonals -- one array for the single-basis readouts, one per basis (P_Z, P_X)
    for 'ZX'. Z -> q features; ZZ -> q + q(q-1)/2; ZX -> the ZZ block of P_Z followed by the same
    block of P_X, which the same sign tables turn into <X_i> and <X_i X_j>."""
    if kind == "ZX":
        return xp.concatenate([_zz_block(P, z_signs, zz_signs, xp) for P in D], axis=1)
    zc = D @ z_signs.T
    if kind == "Z":
        return zc
    if kind == "ZZ":
        return xp.concatenate([zc, D @ zz_signs.T], axis=1)
    raise ValueError(f"unknown readout {kind!r}; expected one of {READOUTS}")


def _zz_block(P, z_signs, zz_signs, xp):
    return xp.concatenate([P @ z_signs.T, P @ zz_signs.T], axis=1)


def readout_width(kind: str, q: int) -> int:
    if kind == "Z":
        return q
    return (q + q * (q - 1) // 2) * len(readout_bases(kind))
