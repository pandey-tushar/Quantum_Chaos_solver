from __future__ import annotations

import numpy as np


class Reservoir:
    """Fixed random Hamiltonian H; U(tau) = exp(-i H tau).

    H is diagonalised once (on the requested backend), so every further tau costs one
    matrix product.
    """

    def __init__(self, q: int, seed: int):
        self.q, self.seed = q, seed
        self._eig = {}
        self._u = {}

    @staticmethod
    def suggest(trial) -> dict:
        """Optuna search space for this reservoir's own parameters (constructor kwargs)."""
        return {}

    def hamiltonian(self) -> np.ndarray:
        raise NotImplementedError

    def unitary(self, tau: float, xp=np, dtype=np.complex128):
        """exp(-i H tau), caching the eigendecomposition per (backend, precision) and only the most
        recent U (it is 2^q x 2^q on the device).

        CPU: double-precision eigensolver (real when H is real), U = V cos V^T - i V sin V^T.
        GPU: the eigensolver runs in the requested precision (a real single-precision solve is several
        times faster on consumer GPUs and gives ~1e-6 unitary error), U assembled in that precision."""
        single = np.dtype(dtype) == np.complex64
        key = (xp.__name__, float(tau), np.dtype(dtype).str)
        if key in self._u:
            return self._u[key]
        ekey = (xp.__name__, single and xp is not np)
        if ekey not in self._eig:
            H = self.hamiltonian()
            if xp is np:
                self._eig[ekey] = np.linalg.eigh(H)
            else:
                real = np.isrealobj(H)
                edtype = (np.float32 if real else np.complex64) if single else (np.float64 if real else np.complex128)
                self._eig[ekey] = xp.linalg.eigh(xp.asarray(H, dtype=edtype))
        w, V = self._eig[ekey]
        if single:
            w = w.astype(np.float32)
            V = V.astype(np.float32 if np.isrealobj(V) else np.complex64)
        if np.isrealobj(V):
            U = (V * xp.cos(w * tau)) @ V.T - 1j * ((V * xp.sin(w * tau)) @ V.T)
        else:
            U = (V * xp.exp(-1j * w * tau)) @ V.conj().T
        U = U.astype(dtype)
        self._u.clear()
        self._u[key] = U
        return U
