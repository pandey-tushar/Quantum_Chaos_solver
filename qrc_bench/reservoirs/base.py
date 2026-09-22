from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class Reservoir:
    """Fixed random Hamiltonian H; U(tau) = exp(-i H tau).

    H is diagonalised once on the CPU in double precision, block by block: the connected
    components of its sparsity graph are sectors H never mixes (both registered Hamiltonians are
    block-diagonal in the computational basis), and a dense H simply gives one block, so nothing
    is assumed about the symmetry. Every further tau costs one matrix product per block.
    """

    def __init__(self, q: int, seed: int):
        self.q, self.seed = q, seed
        self._eigenblocks = None
        self._u = {}

    @staticmethod
    def suggest(trial) -> dict:
        """Optuna search space for this reservoir's own parameters (constructor kwargs)."""
        return {}

    def hamiltonian(self) -> np.ndarray:
        raise NotImplementedError

    def eigenblocks(self) -> list:
        """[(idx, w, V), ...]: one double-precision eigh (real solver when H is real) per connected
        component of the sparsity graph of H, computed once and cached."""
        if self._eigenblocks is None:
            H = self.hamiltonian()
            n, labels = connected_components(csr_matrix(H != 0), directed=False)
            blocks = []
            for c in range(n):
                idx = np.flatnonzero(labels == c)
                w, V = np.linalg.eigh(H[np.ix_(idx, idx)])
                blocks.append((idx, w, V))
            self._eigenblocks = blocks
        return self._eigenblocks

    def blocks(self) -> list:
        """Index arrays of the connected components of H (a single entry when H is dense)."""
        return [idx for idx, _, _ in self.eigenblocks()]

    def unitary(self, tau: float, xp=np, dtype=np.complex128):
        """exp(-i H tau), assembled block-wise on the CPU in double precision from the cached
        eigendecomposition (U = V cos V^T - i V sin V^T per block when V is real), then cast to
        dtype and moved to the backend. Only the most recent U is cached per (backend, tau, dtype)
        key, because it is 2^q x 2^q on the device."""
        key = (xp.__name__, float(tau), np.dtype(dtype).str)
        if key in self._u:
            return self._u[key]
        eig = self.eigenblocks()
        dim = sum(len(idx) for idx, _, _ in eig)
        U = np.zeros((dim, dim), dtype=np.complex128)
        for idx, w, V in eig:
            if np.isrealobj(V):
                block = (V * np.cos(w * tau)) @ V.T - 1j * ((V * np.sin(w * tau)) @ V.T)
            else:
                block = (V * np.exp(-1j * w * tau)) @ V.conj().T
            U[np.ix_(idx, idx)] = block
        U = xp.asarray(U.astype(dtype, copy=False))
        self._u.clear()
        self._u[key] = U
        return U
