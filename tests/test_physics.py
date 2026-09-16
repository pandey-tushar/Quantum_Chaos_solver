import numpy as np
import pytest

from qrc_bench.qrc import qrc_features
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.simulate import branch, dense
from qrc_bench.simulate.ops import Z, pauli_string, product_state, sign_tables


def test_sign_tables_match_explicit_operators():
    q = 4
    rng = np.random.default_rng(0)
    psi = rng.standard_normal(2 ** q) + 1j * rng.standard_normal(2 ** q)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    z_signs, zz_signs = sign_tables(q)
    diag = np.real(np.diag(rho))
    z_explicit = [np.real(np.trace(rho @ pauli_string({i: Z}, q))) for i in range(q)]
    zz_explicit = [np.real(np.trace(rho @ pauli_string({i: Z, k: Z}, q)))
                   for i in range(q) for k in range(i + 1, q)]
    assert np.allclose(z_signs @ diag, z_explicit, atol=1e-12)
    assert np.allclose(zz_signs @ diag, zz_explicit, atol=1e-12)


def test_ising_xx_is_hermitian_and_unitary():
    res = IsingXX(4, seed=3)
    H = res.hamiltonian()
    assert np.allclose(H, H.conj().T)
    U = res.unitary(1.3)
    assert np.allclose(U @ U.conj().T, np.eye(16), atol=1e-12)


def test_product_state_is_normalised_ry():
    psi = product_state(np.array([0.3, -1.1]))
    assert np.isclose(np.linalg.norm(psi), 1.0)
    assert np.allclose(psi, np.kron([np.cos(0.15), np.sin(0.15)], [np.cos(-0.55), np.sin(-0.55)]))


@pytest.mark.parametrize("n_in,n_mem", [(1, 4), (3, 2), (5, 3)])
def test_branch_matches_dense(n_in, n_mem):
    res = IsingXX(n_in + n_mem, seed=1)
    U = res.unitary(1.0)
    rng = np.random.default_rng(7)
    psis = [product_state(a) for a in rng.uniform(-np.pi / 2, np.pi / 2, (3, n_in))]
    d_dense = dense.final_diag(psis, U, n_in, n_mem, np)
    d_branch = branch.final_diag(psis, U, n_in, n_mem, np)
    assert np.isclose(d_dense.sum(), 1.0)
    assert np.max(np.abs(d_dense - d_branch)) < 1e-12


@pytest.mark.parametrize("k_fb", [0.0, 1.5])
def test_feature_methods_agree_with_feedback(k_fb):
    res = IsingXX(5, seed=2, seed_offset=20_000)
    ang = np.random.default_rng(4).uniform(-1, 1, (12, 1))
    Fd = qrc_features(ang, 1, 4, res, taus=(1.0,), k_fb=k_fb, method="dense")
    Fb = qrc_features(ang, 1, 4, res, taus=(1.0,), k_fb=k_fb, method="branch")
    assert np.max(np.abs(Fd - Fb)) < 1e-12
    assert np.all(np.abs(Fd) <= 1 + 1e-12)


def test_feedback_is_live_and_causal():
    res = IsingXX(5, seed=0, seed_offset=20_000)
    ang = np.random.default_rng(5).uniform(-1, 1, (10, 1))
    open_loop = qrc_features(ang, 1, 4, res, k_fb=0.0)
    fb = qrc_features(ang, 1, 4, res, k_fb=2.0)
    assert np.max(np.abs(open_loop - fb)) > 1e-3
    future = ang.copy()
    future[7:] += 0.5
    assert np.allclose(qrc_features(future, 1, 4, res, k_fb=2.0)[:7], fb[:7], atol=1e-14)
