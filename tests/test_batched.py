"""The batched simulator must reproduce the per-step reference (dense / branch) exactly."""
import numpy as np
import pytest

from qrc_bench.qrc import qrc_features
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.reservoirs.xxz import XXZRandomField

try:
    import cupy  # noqa: F401
    HAS_CUPY = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUPY = False


def _angles(T, n_in, seed=0):
    return np.random.default_rng(seed).uniform(-np.pi / 2, np.pi / 2, (T, n_in))


@pytest.mark.parametrize("n_in,n_mem,mem_depth", [(1, 4, 3), (3, 2, 1), (2, 3, 5), (5, 3, 3), (2, 0, 2), (1, 3, 5)])
@pytest.mark.parametrize("k_fb", [0.0, 1.5])
def test_batched_matches_dense_reference(n_in, n_mem, mem_depth, k_fb):
    res = IsingXX(n_in + n_mem, seed=1)
    ang = _angles(14, n_in)
    kw = dict(taus=(1.0, 0.5), readout="ZZ", mem_depth=mem_depth, k_fb=k_fb)
    ref = qrc_features(ang, n_in, n_mem, res, method="dense", **kw)
    out = qrc_features(ang, n_in, n_mem, res, method="batched", **kw)
    assert out.shape == ref.shape
    assert np.max(np.abs(out - ref)) < 1e-11


def test_chunking_does_not_change_results():
    res = XXZRandomField(5, seed=0, delta=0.7, W=2.0)
    ang = _angles(23, 2)
    a = qrc_features(ang, 2, 3, res, method="batched", chunk_size=4)
    b = qrc_features(ang, 2, 3, res, method="batched", chunk_size=1000)
    assert np.max(np.abs(a - b)) < 1e-13


def test_single_precision_is_close():
    res = IsingXX(6, seed=2)
    ang = _angles(20, 2)
    a = qrc_features(ang, 2, 4, res, method="batched")
    b = qrc_features(ang, 2, 4, res, method="batched", precision="single")
    assert np.max(np.abs(a - b)) < 1e-4


@pytest.mark.parametrize("k_fb", [0.0, 2.0])
def test_recurrent_memory_matches_sequential_dense(k_fb):
    """memory='recurrent' carries the memory state forward: one drive per step, no reset."""
    from qrc_bench.simulate.ops import partial_trace_first, product_state, sign_tables

    n_in, n_mem = 2, 3
    res = IsingXX(n_in + n_mem, seed=4)
    ang = _angles(12, n_in, seed=9)
    U = res.unitary(0.9)
    z_signs, _ = sign_tables(n_in + n_mem)
    rho_mem = np.zeros((2 ** n_mem, 2 ** n_mem), complex)
    rho_mem[0, 0] = 1.0
    mbar, rows = 0.0, []
    for t in range(len(ang)):
        psi = product_state(np.clip(ang[t] + k_fb * np.tanh(mbar), -np.pi, np.pi))
        rho = U @ np.kron(np.outer(psi, psi.conj()), rho_mem) @ U.conj().T
        rho_mem = partial_trace_first(rho, n_in, n_mem)
        d = np.real(np.diag(rho))
        rows.append(z_signs @ d)
        mbar = float(np.mean(rows[-1]))
    out = qrc_features(ang, n_in, n_mem, res, taus=(0.9,), readout="Z", memory="recurrent", k_fb=k_fb,
                       method="batched")
    assert np.max(np.abs(out - np.array(rows))) < 1e-11


@pytest.mark.skipif(not HAS_CUPY, reason="no CUDA device")
@pytest.mark.parametrize("memory,k_fb", [("reset", 0.0), ("reset", 1.0), ("recurrent", 0.5)])
def test_cupy_matches_numpy(memory, k_fb):
    res = XXZRandomField(6, seed=3, delta=1.1, W=1.5)
    ang = _angles(30, 2)
    kw = dict(taus=(1.3,), memory=memory, k_fb=k_fb, method="batched")
    a = qrc_features(ang, 2, 4, res, backend="numpy", **kw)
    b = qrc_features(ang, 2, 4, res, backend="cupy", precision="double", **kw)
    assert isinstance(b, np.ndarray)
    assert np.max(np.abs(a - b)) < 1e-11
    c = qrc_features(ang, 2, 4, res, backend="cupy", **kw)          # auto = single on the GPU
    assert np.max(np.abs(a - c)) < 1e-5


def test_unknown_memory_mode_rejected():
    with pytest.raises(ValueError, match="memory"):
        qrc_features(_angles(3, 1), 1, 1, IsingXX(2, 0), memory="bogus")


def test_recurrent_single_precision_does_not_drift():
    res = XXZRandomField(6, seed=5, delta=0.8, W=1.0)
    ang = _angles(300, 2, seed=2)
    kw = dict(taus=(1.1,), memory="recurrent", method="batched")
    a = qrc_features(ang, 2, 4, res, **kw)
    b = qrc_features(ang, 2, 4, res, precision="single", **kw)
    assert np.max(np.abs(a - b)) < 2e-5


def test_real_and_complex_unitaries_agree():
    res = XXZRandomField(5, seed=1, delta=0.4, W=2.0, hx=0.3)
    H = res.hamiltonian()
    assert np.isrealobj(H)
    w, V = np.linalg.eigh(H.astype(complex))
    ref = (V * np.exp(-1j * w * 0.7)) @ V.conj().T
    assert np.max(np.abs(res.unitary(0.7) - ref)) < 1e-12


@pytest.mark.parametrize("backend", ["numpy"] + (["cupy"] if HAS_CUPY else []))
def test_single_precision_unitary_is_unitary(backend):
    from qrc_bench.backend import get_xp, to_numpy
    xp = get_xp(backend)
    res = XXZRandomField(6, seed=0, delta=0.5, W=1.0)
    U = to_numpy(res.unitary(0.9, xp=xp, dtype=np.complex64))
    assert U.dtype == np.complex64
    assert np.max(np.abs(U @ U.conj().T - np.eye(64))) < 1e-5
    assert np.max(np.abs(U - res.unitary(0.9))) < 1e-5


def test_unitary_cache_returns_same_matrix_for_same_tau():
    res = IsingXX(5, seed=0)
    assert res.unitary(0.7) is res.unitary(0.7)
