import numpy as np
import pytest

from qrc_bench import registry
from qrc_bench.qrc import qrc_features
from qrc_bench.reservoirs.xxz import XXZRandomCoupling, XXZRandomField, edges, xxz_hamiltonian
from qrc_bench.simulate.ops import X, Y, Z, pauli_string


def _explicit(q, couplings, delta, fields, hx):
    H = sum(J * (pauli_string({i: X, j: X}, q) + pauli_string({i: Y, j: Y}, q)
                 + delta * pauli_string({i: Z, j: Z}, q)) for (i, j), J in couplings.items())
    H = H + sum(h * pauli_string({i: Z}, q) for i, h in enumerate(fields))
    return H + sum(hx * pauli_string({i: X}, q) for i in range(q))


@pytest.mark.parametrize("topology", ["all", "chain"])
@pytest.mark.parametrize("hx", [0.0, 0.4])
def test_fast_construction_matches_pauli_strings(topology, hx):
    q = 4
    rng = np.random.default_rng(0)
    couplings = {e: rng.uniform(0, 1) for e in edges(q, topology)}
    fields = rng.uniform(-1, 1, q)
    assert np.allclose(xxz_hamiltonian(q, couplings, 0.7, fields, hx), _explicit(q, couplings, 0.7, fields, hx))


@pytest.mark.parametrize("cls", [XXZRandomField, XXZRandomCoupling])
def test_hermitian_and_unitary(cls):
    res = cls(5, seed=2, delta=0.5)
    H = res.hamiltonian()
    assert np.allclose(H, H.conj().T)
    U = res.unitary(0.8)
    assert np.allclose(U @ U.conj().T, np.eye(32), atol=1e-12)


def test_u1_conserved_without_transverse_field():
    q = 4
    Ztot = sum(pauli_string({i: Z}, q) for i in range(q))
    H = XXZRandomField(q, seed=1, delta=1.3, W=2.0).hamiltonian()
    assert np.allclose(H @ Ztot, Ztot @ H)
    H_broken = XXZRandomField(q, seed=1, delta=1.3, W=2.0, hx=0.5).hamiltonian()
    assert not np.allclose(H_broken @ Ztot, Ztot @ H_broken)


def test_delta_zero_no_field_is_xx_plus_yy():
    q = 3
    H = XXZRandomField(q, seed=0, delta=0.0, W=0.0, topology="chain").hamiltonian()
    ref = sum(pauli_string({i: X, i + 1: X}, q) + pauli_string({i: Y, i + 1: Y}, q) for i in range(q - 1))
    assert np.allclose(H, ref)


def test_disorder_modes_and_seeds():
    a = XXZRandomField(4, seed=0, W=3.0).hamiltonian()
    b = XXZRandomField(4, seed=1, W=3.0).hamiltonian()
    assert not np.allclose(a, b)
    field = np.real(np.diag(XXZRandomCoupling(3, seed=0, delta=0.0, h=0.5).hamiltonian()))
    assert np.allclose(field, [1.5, 0.5, 0.5, -0.5, 0.5, -0.5, -0.5, -1.5])   # uniform h * sum z


def test_registered_and_usable_in_features():
    assert {"xxz", "xxz_uniform"} <= set(registry.names("reservoir"))
    res = registry.get("reservoir", "xxz")(4, 0, delta=0.5, W=1.0)
    F = qrc_features(np.random.default_rng(3).uniform(-1, 1, (8, 2)), 2, 2, res, readout="ZZ")
    assert F.shape == (8, 10) and np.all(np.abs(F) <= 1 + 1e-12)
