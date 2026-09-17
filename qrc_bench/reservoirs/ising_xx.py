"""All-to-all XX couplings plus a uniform Z field (the cartography paper's reservoir)."""
from __future__ import annotations

import numpy as np

from qrc_bench.registry import register
from qrc_bench.reservoirs.base import Reservoir


@register("reservoir", "ising_xx")
class IsingXX(Reservoir):
    """H = sum_{i<j} J_ij X_i X_j + v sum_i Z_i,  J_ij ~ U(0, 1).

    The coupling RNG is seeded with ``seed_offset + seed``: the paper used 10000
    for Case I and 20000 for Case II.
    """

    def __init__(self, q: int, seed: int, v: float = 1.0, seed_offset: int = 10_000):
        super().__init__(q, seed)
        self.v, self.seed_offset = v, seed_offset

    @staticmethod
    def suggest(trial) -> dict:
        return {"v": trial.suggest_float("v", 0.1, 10.0, log=True)}

    def hamiltonian(self) -> np.ndarray:
        """Built in the computational basis: X_i X_j flips bits i and j; Z_i is diagonal."""
        q = self.q
        rng = np.random.default_rng(self.seed_offset + self.seed)
        idx = np.arange(2 ** q)
        bits = (idx[:, None] >> np.arange(q - 1, -1, -1)) & 1
        H = np.zeros((2 ** q, 2 ** q))
        for i in range(q):
            for j in range(i + 1, q):
                H[idx ^ ((1 << (q - 1 - i)) | (1 << (q - 1 - j))), idx] += rng.uniform(0, 1)
        H[idx, idx] += self.v * (1.0 - 2.0 * bits).sum(axis=1)
        return H
