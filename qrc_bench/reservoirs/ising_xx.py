"""All-to-all XX couplings plus a uniform Z field (the cartography paper's reservoir)."""
from __future__ import annotations

import numpy as np

from qrc_bench.registry import register
from qrc_bench.reservoirs.base import Reservoir
from qrc_bench.simulate.ops import X, Z, pauli_string


@register("reservoir", "ising_xx")
class IsingXX(Reservoir):
    """H = sum_{i<j} J_ij X_i X_j + v sum_i Z_i,  J_ij ~ U(0, 1).

    The coupling RNG is seeded with ``seed_offset + seed``: the paper used 10000
    for Case I and 20000 for Case II.
    """

    def __init__(self, q: int, seed: int, v: float = 1.0, seed_offset: int = 10_000):
        super().__init__(q, seed)
        self.v, self.seed_offset = v, seed_offset

    def hamiltonian(self) -> np.ndarray:
        q = self.q
        rng = np.random.default_rng(self.seed_offset + self.seed)
        H = np.zeros((2 ** q, 2 ** q), dtype=complex)
        for i in range(q):
            for j in range(i + 1, q):
                H += rng.uniform(0, 1) * pauli_string({i: X, j: X}, q)
        for i in range(q):
            H += self.v * pauli_string({i: Z}, q)
        return H
