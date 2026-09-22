"""Block-wise CPU eigensolver behind Reservoir.unitary.

Both registered Hamiltonians are block-diagonal in the computational basis (xxz with hx = 0
conserves total magnetisation, ising_xx conserves Hamming-weight parity), so diagonalising the
connected components of the sparsity graph is much cheaper than one 2^q x 2^q eigh -- and a dense
H simply gives a single block.
"""
import math

import numpy as np
import pytest

from qrc_bench.reservoirs.base import Reservoir
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.reservoirs.xxz import XXZRandomField

try:
    import cupy
    HAS_CUPY = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUPY = False


class DenseReservoir(Reservoir):
    """Random symmetric H with no zeros: the sparsity graph is a single component."""

    def hamiltonian(self) -> np.ndarray:
        A = np.random.default_rng(self.seed).uniform(0.5, 1.5, (2 ** self.q, 2 ** self.q))
        return A + A.T


def reference_unitary(res, tau):
    """U(tau) from one FULL eigh of H."""
    w, V = np.linalg.eigh(res.hamiltonian())
    return (V * np.exp(-1j * w * tau)) @ V.conj().T


RESERVOIRS = {
    "xxz_hx0": (lambda: XXZRandomField(6, 3, delta=0.7, W=2.0), [math.comb(6, k) for k in range(7)]),
    "ising_xx": (lambda: IsingXX(6, 1, v=1.0), [32, 32]),
    "xxz_hx": (lambda: XXZRandomField(6, 3, delta=0.7, W=2.0, hx=0.3), [64]),
    "dense": (lambda: DenseReservoir(5, 0), [32]),
}


@pytest.mark.parametrize("name", list(RESERVOIRS))
def test_block_sizes(name):
    make, sizes = RESERVOIRS[name]
    blocks = make().blocks()
    assert len(blocks) == len(sizes)
    assert sorted(len(b) for b in blocks) == sorted(sizes)
    assert sorted(np.concatenate(blocks).tolist()) == list(range(sum(sizes)))


@pytest.mark.parametrize("name", list(RESERVOIRS))
def test_unitary_matches_full_eigh(name):
    res = RESERVOIRS[name][0]()
    U = res.unitary(0.7)
    assert U.dtype == np.complex128
    assert np.max(np.abs(U - reference_unitary(res, 0.7))) < 1e-12
    assert np.max(np.abs(U.conj().T @ U - np.eye(U.shape[0]))) < 1e-12


def test_tau_cache_and_single_diagonalisation(monkeypatch):
    res = XXZRandomField(6, 3, delta=0.7, W=2.0)
    calls = []
    real_eigh = np.linalg.eigh
    monkeypatch.setattr(np.linalg, "eigh", lambda a: (calls.append(a.shape), real_eigh(a))[1])

    U1 = res.unitary(0.7)
    assert res.unitary(0.7) is U1
    n_after_first = len(calls)
    assert n_after_first == 7                      # one eigh per magnetisation sector

    eig = res.eigenblocks()
    res.unitary(1.1)
    assert len(calls) == n_after_first             # a new tau reuses the cached blocks
    assert res.eigenblocks() is eig


@pytest.mark.skipif(not HAS_CUPY, reason="cupy device required")
def test_cupy_single_precision_matches_cpu():
    res = IsingXX(6, 1, v=1.0)
    U = res.unitary(0.9, xp=cupy, dtype=np.complex64)
    assert isinstance(U, cupy.ndarray)
    assert U.dtype == np.complex64
    assert np.max(np.abs(cupy.asnumpy(U) - res.unitary(0.9))) < 1e-5
