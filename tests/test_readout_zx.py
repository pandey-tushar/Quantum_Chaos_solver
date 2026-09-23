"""Multi-basis readout 'ZX': the Z block is unchanged and an X-basis block is measured beside it.

Z and ZZ all commute, so one Z-basis distribution already gives every feature of the 'ZZ' readout.
'ZX' adds a second, non-commuting basis: P_X = diag(H^q rho H^q), read with the same sign tables.
The total shot budget is fixed, so two bases mean half the shots each -- the comparison is
"more bases" against "more shots per basis".
"""
from functools import reduce

import numpy as np
import pytest

from qrc_bench.qrc import qrc_features
from qrc_bench.readouts import readout_width
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.reservoirs.xxz import XXZRandomField
from qrc_bench.simulate.ops import X as PAULI_X
from qrc_bench.simulate.ops import partial_trace_first, pauli_string, product_state, sign_tables

try:
    import cupy  # noqa: F401
    HAS_CUPY = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUPY = False

H1 = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)


def _hadamard(q):
    return reduce(np.kron, [H1] * q)


def _angles(T, n_in, seed=0):
    return np.random.default_rng(seed).uniform(-np.pi / 2, np.pi / 2, (T, n_in))


# --- (a) a product state, no evolution ---------------------------------------------------------

def test_product_state_x_expectations():
    """U = identity: R_Y(a)|0> has <X> = sin(a), <Z> = cos(a), and the state is a product."""
    n_in, q = 3, 3
    ang = _angles(5, n_in, seed=1)
    res = IsingXX(q, seed=0)
    F = qrc_features(ang, n_in, 0, res, taus=(0.0,), readout="ZX", mem_depth=1)
    w = readout_width("ZZ", q)
    assert F.shape == (5, 2 * w)
    pairs = [(i, k) for i in range(q) for k in range(i + 1, q)]
    for t, a in enumerate(ang):
        assert np.max(np.abs(F[t, :q] - np.cos(a))) < 1e-12
        assert np.max(np.abs(F[t, w:w + q] - np.sin(a))) < 1e-12
        zz = [np.cos(a[i]) * np.cos(a[k]) for i, k in pairs]
        xx = [np.sin(a[i]) * np.sin(a[k]) for i, k in pairs]
        assert np.max(np.abs(F[t, q:w] - zz)) < 1e-12
        assert np.max(np.abs(F[t, w + q:] - xx)) < 1e-12


# --- (b) against an explicit rho_full ----------------------------------------------------------

def _rho_full(psis, U, n_in, n_mem):
    """The dense reference of simulate/dense.py, but keeping rho rather than its diagonal."""
    rho_mem = np.zeros((2 ** n_mem, 2 ** n_mem), dtype=complex)
    rho_mem[0, 0] = 1.0
    rho = None
    for k, psi in enumerate(psis):
        rho = U @ np.kron(np.outer(psi, psi.conj()), rho_mem) @ U.conj().T
        if k < len(psis) - 1:
            rho_mem = partial_trace_first(rho, n_in, n_mem)
    return rho


@pytest.mark.parametrize("depth", [2, 3])      # depth 3 drives a step whose X probabilities are skipped
def test_x_block_matches_an_explicit_density_matrix(depth):
    n_in, n_mem, tau = 2, 2, 0.9
    q = n_in + n_mem
    res = IsingXX(q, seed=3)
    U = res.unitary(tau)
    ang = _angles(6, n_in, seed=2)
    F = qrc_features(ang, n_in, n_mem, res, taus=(tau,), readout="ZX", mem_depth=depth)
    z_signs, zz_signs = sign_tables(q)
    Hq = _hadamard(q)
    w = readout_width("ZZ", q)
    for t in range(len(ang)):
        psis = [product_state(ang[tt]) for tt in range(max(0, t - depth + 1), t + 1)]
        rho = _rho_full(psis, U, n_in, n_mem)
        P_X = np.real(np.diag(Hq @ rho @ Hq))
        assert abs(P_X.sum() - 1.0) < 1e-12
        ref = np.concatenate([z_signs @ P_X, zz_signs @ P_X])
        assert np.max(np.abs(F[t, w:] - ref)) < 1e-12
        for i in range(q):
            xi = np.trace(rho @ pauli_string({i: PAULI_X}, q)).real
            assert abs(F[t, w + i] - xi) < 1e-12


# --- (c) the Z side is untouched ---------------------------------------------------------------

@pytest.mark.parametrize("memory,mem_depth", [("recurrent", 1), ("reset", 3)])
def test_z_block_is_bit_identical_to_the_zz_readout(memory, mem_depth):
    n_in, n_mem = 2, 3
    res = XXZRandomField(n_in + n_mem, seed=7, delta=0.7, W=2.0)
    ang = _angles(16, n_in, seed=5)
    kw = dict(taus=(1.0, 0.5), mem_depth=mem_depth, k_fb=1.5, memory=memory, method="batched")
    zz = qrc_features(ang, n_in, n_mem, res, readout="ZZ", **kw)
    zx = qrc_features(ang, n_in, n_mem, res, readout="ZX", **kw)
    w = readout_width("ZZ", n_in + n_mem)
    assert zx.shape == (len(ang), 4 * w)                       # two taus x two bases
    for b in range(2):                                         # one tau block after the other
        assert np.array_equal(zx[:, 2 * w * b:2 * w * b + w], zz[:, w * b:w * (b + 1)])


# --- (d) widths ---------------------------------------------------------------------------------

def test_widths():
    from qrc_bench.tuning import Comparison

    assert readout_width("ZX", 12) == 156
    assert readout_width("ZX", 4) == 20
    cmp = Comparison(kind="window", n_series=6, input_window=3, n_mem=6, readout="ZX")
    assert (cmp.q, cmp.width) == (12, 156)


# --- (e) the shot budget is split across the bases ----------------------------------------------

class _Recorder:
    """Stands in for the sampler: records the shot count and the distribution, changes nothing."""

    def __init__(self):
        self.shots, self.probs = [], []

    def __call__(self, P, shots, rng, xp):
        self.shots.append(shots)
        self.probs.append(np.array(P, copy=True))
        return P


@pytest.mark.parametrize("memory,mem_depth,k_fb", [("recurrent", 1, 1.5), ("reset", 2, 0.0)])
def test_total_shot_budget_is_split_equally_across_bases(monkeypatch, memory, mem_depth, k_fb):
    res = IsingXX(4, seed=2)
    ang = _angles(4, 2, seed=6)
    kw = dict(taus=(1.0,), mem_depth=mem_depth, memory=memory, k_fb=k_fb, shots=100)

    rec_zz = _Recorder()
    monkeypatch.setattr("qrc_bench.qrc.sample_probs", rec_zz)
    qrc_features(ang, 2, 2, res, readout="ZZ", **kw)
    assert rec_zz.shots and set(rec_zz.shots) == {100}

    rec_zx = _Recorder()
    monkeypatch.setattr("qrc_bench.qrc.sample_probs", rec_zx)
    qrc_features(ang, 2, 2, res, readout="ZX", **kw)
    assert set(rec_zx.shots) == {50}
    assert len(rec_zx.shots) == 2 * len(rec_zz.shots)          # one call per basis
    for k in range(0, len(rec_zx.probs), 2):                   # Z first, then the X basis
        assert np.array_equal(rec_zx.probs[k], rec_zz.probs[k // 2])
        assert not np.array_equal(rec_zx.probs[k], rec_zx.probs[k + 1])


# --- (f) the fast Walsh-Hadamard transform ------------------------------------------------------

@pytest.mark.parametrize("q", [3, 5])
def test_fwht_equals_the_kronecker_hadamard(q):
    from qrc_bench.simulate.batched import hadamard_rows

    D = 2 ** q
    rng = np.random.default_rng(q)
    U = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D)))
    ref = _hadamard(q) @ U
    out = hadamard_rows(U.copy(), 0, np)
    assert out.shape == U.shape
    assert np.max(np.abs(out - ref)) < 1e-12


@pytest.mark.skipif(not HAS_CUPY, reason="no CUDA device")
@pytest.mark.parametrize("q", [3, 5])
def test_fwht_on_cupy_single_precision(q):
    from qrc_bench.simulate.batched import hadamard_rows

    D = 2 ** q
    rng = np.random.default_rng(q + 1)
    U = (rng.standard_normal((D, D)) + 1j * rng.standard_normal((D, D))).astype(np.complex64)
    ref = _hadamard(q) @ U
    out = cupy.asnumpy(hadamard_rows(cupy.asarray(U), 0, cupy))
    assert out.dtype == np.complex64
    assert np.max(np.abs(out - ref)) < 1e-5


def test_fwht_is_applied_to_the_row_index_of_the_driver_unitary():
    """U4 is (R, d1, D, d2); the X frame must be H^q on U's row index, not on anything else."""
    from qrc_bench.simulate.batched import Driver

    n_in, n_mem, q = 2, 2, 4
    U = IsingXX(q, seed=8).unitary(0.6)
    drv = Driver(U, n_in, n_mem, np, np.complex128, x_basis=True)
    ref = _hadamard(q) @ U
    got = drv.U4_X.transpose(0, 2, 1, 3).reshape(2 ** q, 2 ** q)
    assert np.max(np.abs(got - ref)) < 1e-12


# --- (g) several reservoirs side by side --------------------------------------------------------

def test_multi_reservoir_zx_matches_one_call_per_reservoir():
    n_in, n_mem, T, s = 2, 2, 24, 2
    ang = np.random.default_rng(0).uniform(-1, 1, (2, s, T, n_in))
    reservoirs = [XXZRandomField(n_in + n_mem, seed, delta=0.7, W=2.0) for seed in (11, 12)]
    kw = dict(taus=(1.0,), readout="ZX", memory="recurrent", k_fb=1.5, mem_depth=3,
              backend="numpy", precision="double", chunk_size=24, streams=True)
    together = qrc_features(ang, n_in, n_mem, reservoirs, **kw)
    assert together.shape == (2, s, T, 2 * readout_width("ZZ", n_in + n_mem))
    for r, res in enumerate(reservoirs):
        assert np.max(np.abs(together[r] - qrc_features(ang[r], n_in, n_mem, res, **kw))) < 1e-12


@pytest.mark.skipif(not HAS_CUPY, reason="no CUDA device")
@pytest.mark.parametrize("memory", ["reset", "recurrent"])
def test_cupy_single_precision_zx_matches_numpy_double(memory):
    n_in, n_mem, T, s = 2, 3, 20, 2
    ang = np.random.default_rng(4).uniform(-1, 1, (2, s, T, n_in))
    reservoirs = [XXZRandomField(n_in + n_mem, seed, delta=0.6, W=1.3) for seed in (3, 4)]
    kw = dict(taus=(0.8,), readout="ZX", memory=memory, mem_depth=3, k_fb=0.5, streams=True)
    a = qrc_features(ang, n_in, n_mem, reservoirs, backend="numpy", precision="double", **kw)
    b = qrc_features(ang, n_in, n_mem, reservoirs, backend="cupy", precision="single", **kw)
    assert isinstance(b, np.ndarray) and b.shape == a.shape
    assert np.max(np.abs(a - b)) < 1e-5


# --- (h) CLI ------------------------------------------------------------------------------------

def test_cli_accepts_the_zx_readout():
    from qrc_bench.cli import build_parser
    from qrc_bench.tuning import Comparison

    a = build_parser().parse_args(["experiment", "--task", "lorenz96", "--kind", "window",
                                   "--input-window", "3", "--n-mem", "6", "--readout", "ZX"])
    assert a.readout == "ZX"
    cmp = Comparison(kind="window", n_series=6, input_window=a.input_window, n_mem=a.n_mem,
                     readout=a.readout)
    assert (cmp.q, cmp.width) == (12, 156)
    lay = build_parser().parse_args(["layout", "--n-series", "6", "--n-mem", "6", "--readout", "ZX"])
    assert lay.readout == "ZX"


# --- (i) the reference simulators only know the Z diagonal --------------------------------------

@pytest.mark.parametrize("method", ["branch", "dense"])
def test_reference_methods_refuse_the_multi_basis_readout(method):
    with pytest.raises(ValueError, match="multi-basis readout requires"):
        qrc_features(_angles(4, 1), 1, 1, IsingXX(2, 0), readout="ZX", method=method)
