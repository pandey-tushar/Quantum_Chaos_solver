"""Backend-aware defaults and multi-stream batching (several data series through one reservoir at once)."""
import numpy as np
import pytest

from qrc_bench.encoders import dense_rxrz
from qrc_bench.qrc import qrc_features, resolve_budget_mb, resolve_precision
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.reservoirs.xxz import XXZRandomField
from qrc_bench.tuning import Comparison, Sim
from qrc_bench.tuning.spaces import build_streams

try:
    import cupy
    HAS_CUPY = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUPY = False
BACKENDS = ["numpy"] + (["cupy"] if HAS_CUPY else [])


def test_auto_precision_follows_backend():
    assert resolve_precision("auto", "numpy") == "double"
    assert resolve_precision("auto", "cupy") == "single"
    assert resolve_precision("double", "cupy") == "double"
    with pytest.raises(ValueError):
        resolve_precision("half", "numpy")


def test_auto_budget():
    assert resolve_budget_mb("auto", "numpy") == 512
    assert resolve_budget_mb(100, "cupy") == 100
    if HAS_CUPY:
        b = resolve_budget_mb("auto", "cupy")
        free = cupy.cuda.Device().mem_info[0] / 2 ** 20
        assert 32 <= b <= free


@pytest.mark.parametrize("backend", BACKENDS)
def test_single_precision_eigensolver_on_request(backend):
    from qrc_bench.backend import get_xp, to_numpy
    res = IsingXX(7, 1)
    U1 = to_numpy(res.unitary(1.3, xp=get_xp(backend), dtype=np.complex64))
    U2 = res.unitary(1.3)
    assert np.max(np.abs(U1 - U2)) < 1e-4


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("memory,k_fb,mem_depth", [("reset", 0.0, 3), ("reset", 1.2, 2), ("recurrent", 0.0, 1),
                                                   ("recurrent", 0.9, 1)])
def test_streams_equal_separate_runs(backend, memory, k_fb, mem_depth):
    rng = np.random.default_rng(0)
    S, T, n_in, n_mem = 3, 17, 2, 3
    ang = rng.uniform(-1, 1, (S, T, n_in))
    res = XXZRandomField(n_in + n_mem, 4, delta=0.6, W=1.3)
    kw = dict(taus=(0.8, 0.4), memory=memory, k_fb=k_fb, mem_depth=mem_depth, backend=backend, precision="double",
              chunk_size=5)
    together = qrc_features(ang, n_in, n_mem, res, streams=True, **kw)
    assert together.shape[:2] == (S, T)
    for s in range(S):
        ref = qrc_features(ang[s], n_in, n_mem, res, taus=(0.8, 0.4), memory=memory, k_fb=k_fb,
                           mem_depth=mem_depth, method="dense" if memory == "reset" else "batched")
        assert np.max(np.abs(together[s] - ref)) < 1e-11


def test_streams_with_dense_encoding():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (2, 12, 5))
    ang = np.stack([dense_rxrz(x, 1.0) for x in X])
    res = IsingXX(5, 0)
    out = qrc_features(ang, 3, 2, res, memory="recurrent", k_fb=0.5, streams=True)
    for s in range(2):
        assert np.allclose(out[s], qrc_features(ang[s], 3, 2, res, memory="recurrent", k_fb=0.5), atol=1e-12)


@pytest.mark.parametrize("model", ["qrc", "esn", "random_features", "poly2"])
def test_build_streams_matches_per_series_builds(model):
    from qrc_bench.tuning.spaces import MODELS
    rng = np.random.default_rng(2)
    Xs = [rng.standard_normal((40, 2)) for _ in range(3)]
    kind = "window" if model in ("qrc", "random_features", "poly2") else "recurrent"
    cmp = Comparison(kind=kind, n_series=2, input_window=2, n_mem=1)
    params = {"qrc": {"tau": 1.0, "scale": 0.8, "reservoir": {"v": 1.2}, "k_fb": 0.0},
              "esn": {"sr": 0.9, "leak": 0.3, "in_scale": 0.5, "density": 0.2},
              "random_features": {"scale": 1.0, "bias_scale": 0.5}, "poly2": {}}[model]
    out = build_streams(model, params, Xs, 7, cmp, Sim())
    for X, F in zip(Xs, out):
        assert np.allclose(F, MODELS[model].build(params, X, 7, cmp, Sim()), atol=1e-12)


def test_angle_shapes_are_explicit():
    res = IsingXX(4, 0)
    with pytest.raises(ValueError, match="angles"):
        qrc_features(np.zeros((5, 3)), 2, 2, res)
    assert qrc_features(np.zeros((6, 2, 2)), 2, 2, res).shape[0] == 6                  # dense, one stream
    assert qrc_features(np.zeros((6, 2, 2)), 2, 2, res, streams=True).shape[:2] == (6, 2)  # R_Y, six streams


def test_auto_backend_uses_gpu_only_where_it_pays(monkeypatch):
    from qrc_bench import backend
    monkeypatch.setattr(backend, "gpu_available", lambda: True)
    assert backend.resolve_backend("auto", q=8) == "numpy"
    assert backend.resolve_backend("auto", q=10) == "cupy"
    assert backend.resolve_backend("numpy", q=14) == "numpy"
    monkeypatch.setattr(backend, "gpu_available", lambda: False)
    assert backend.resolve_backend("auto", q=14) == "numpy"
    with pytest.raises(ValueError):
        backend.resolve_backend("tpu", q=5)


def test_qrc_features_accepts_auto_backend():
    res = IsingXX(4, 0)
    ang = np.random.default_rng(0).uniform(-1, 1, (8, 2))
    assert np.allclose(qrc_features(ang, 2, 2, res, backend="auto"), qrc_features(ang, 2, 2, res), atol=1e-12)


def test_experiment_cli_resumes_by_default():
    from qrc_bench.cli import build_parser
    base = ["experiment", "--task", "henon", "--kind", "window", "--input-window", "2", "--n-mem", "1"]
    assert build_parser().parse_args(base).resume is True
    assert build_parser().parse_args(base + ["--no-resume"]).resume is False
    assert build_parser().parse_args(base).backend == "auto"
