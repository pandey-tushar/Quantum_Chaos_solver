from __future__ import annotations

import numpy as np


class Reservoir:
    """Fixed random Hamiltonian H; U(tau) = exp(-i H tau).

    H is diagonalised once, so every further tau costs one matrix product.
    """

    def __init__(self, q: int, seed: int):
        self.q, self.seed = q, seed
        self._eig = None

    def hamiltonian(self) -> np.ndarray:
        raise NotImplementedError

    def unitary(self, tau: float) -> np.ndarray:
        if self._eig is None:
            self._eig = np.linalg.eigh(self.hamiltonian())
        w, V = self._eig
        return (V * np.exp(-1j * w * tau)) @ V.conj().T
