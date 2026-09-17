"""Heisenberg XXZ reservoirs: H = sum_<ij> J_ij (X_i X_j + Y_i Y_j + Delta Z_i Z_j) + sum_i h_i Z_i + hx sum_i X_i.

Two disorder modes, registered under separate names:
  xxz          uniform couplings J, random fields h_i ~ U(-W, W)   (disordered-field model)
  xxz_uniform  random couplings J_ij ~ U(0, J), uniform field h

With hx = 0 the Hamiltonian conserves total magnetisation sum_i Z_i (U(1)), so the
dynamics never mixes Hamming-weight sectors of the computational basis; hx > 0 breaks it.
H is built directly in the computational basis (qubit 0 = most significant bit):
the flip-flop term (XX + YY) maps |..0..1..> <-> |..1..0..> with amplitude 2 J_ij.
"""
from __future__ import annotations

import numpy as np

from qrc_bench.registry import register
from qrc_bench.reservoirs.base import Reservoir

TOPOLOGIES = ("all", "chain")


def edges(q: int, topology: str):
    if topology == "all":
        return [(i, j) for i in range(q) for j in range(i + 1, q)]
    if topology == "chain":
        return [(i, i + 1) for i in range(q - 1)]
    raise ValueError(f"unknown topology {topology!r}; expected one of {TOPOLOGIES}")


def xxz_hamiltonian(q: int, couplings: dict, delta: float, fields: np.ndarray, hx: float = 0.0) -> np.ndarray:
    """couplings: {(i, j): J_ij}; fields: (q,) longitudinal h_i; hx: uniform transverse field."""
    dim = 2 ** q
    idx = np.arange(dim)
    bits = (idx[:, None] >> np.arange(q - 1, -1, -1)) & 1          # (dim, q)
    z = 1.0 - 2.0 * bits                                            # Z eigenvalues
    H = np.zeros((dim, dim), dtype=complex)
    diag = z @ np.asarray(fields, dtype=float)
    for (i, j), J in couplings.items():
        diag += delta * J * z[:, i] * z[:, j]
        flip = bits[:, i] != bits[:, j]
        mask = (1 << (q - 1 - i)) | (1 << (q - 1 - j))
        src = idx[flip]
        H[src ^ mask, src] += 2.0 * J
    if hx:
        for i in range(q):
            H[idx ^ (1 << (q - 1 - i)), idx] += hx
    H[idx, idx] += diag
    return H


class _XXZBase(Reservoir):
    def __init__(self, q: int, seed: int, delta: float = 1.0, topology: str = "all", hx: float = 0.0,
                 seed_offset: int = 30_000):
        super().__init__(q, seed)
        self.delta, self.topology, self.hx, self.seed_offset = delta, topology, hx, seed_offset

    def _disorder(self, rng):
        raise NotImplementedError

    def hamiltonian(self) -> np.ndarray:
        rng = np.random.default_rng(self.seed_offset + self.seed)
        couplings, fields = self._disorder(rng)
        return xxz_hamiltonian(self.q, couplings, self.delta, fields, self.hx)


@register("reservoir", "xxz")
class XXZRandomField(_XXZBase):
    """Uniform couplings J, random fields h_i ~ U(-W, W)."""

    def __init__(self, q: int, seed: int, J: float = 1.0, W: float = 1.0, **kw):
        super().__init__(q, seed, **kw)
        self.J, self.W = J, W

    @staticmethod
    def suggest(trial) -> dict:
        return {"delta": trial.suggest_float("delta", -2.0, 2.0),
                "W": trial.suggest_float("W", 0.1, 10.0, log=True)}

    def _disorder(self, rng):
        couplings = {e: self.J for e in edges(self.q, self.topology)}
        return couplings, rng.uniform(-self.W, self.W, self.q)


@register("reservoir", "xxz_uniform")
class XXZRandomCoupling(_XXZBase):
    """Random couplings J_ij ~ U(0, J), uniform field h."""

    def __init__(self, q: int, seed: int, J: float = 1.0, h: float = 1.0, **kw):
        super().__init__(q, seed, **kw)
        self.J, self.h = J, h

    @staticmethod
    def suggest(trial) -> dict:
        return {"delta": trial.suggest_float("delta", -2.0, 2.0),
                "h": trial.suggest_float("h", 0.1, 10.0, log=True)}

    def _disorder(self, rng):
        couplings = {e: rng.uniform(0, self.J) for e in edges(self.q, self.topology)}
        return couplings, np.full(self.q, self.h)
