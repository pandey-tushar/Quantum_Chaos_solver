"""Several reservoirs driven side by side in one simulate call.

The loop is kernel-launch-bound, so a trial's reservoir seeds are cheaper as one batch than as
R separate calls. Batching must not change any number: a list of reservoirs has to reproduce the
per-reservoir calls exactly, and one reservoir has to keep today's shapes and results.
"""
import numpy as np
import pytest

from qrc_bench.qrc import qrc_features
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.reservoirs.xxz import XXZRandomField
from qrc_bench.simulate import batched
from qrc_bench.tuning import Comparison, Protocol, Sim, tune_model
from qrc_bench.tuning.spaces import MODELS, build_streams, build_trial

try:
    import cupy  # noqa: F401
    HAS_CUPY = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUPY = False

TINY = dict(task="henon", task_kwargs={"n_series": 2}, n_steps=160, n_trials=2, n_folds=2, washout=10,
            tune_data_seeds=(100, 101), eval_data_seeds=(0, 1), eval_res_seeds=(0, 1))


@pytest.mark.parametrize("memory", ["reset", "recurrent"])
@pytest.mark.parametrize("k_fb", [0.0, 1.5])
def test_reservoir_list_matches_one_call_per_reservoir(memory, k_fb):
    n_in, n_mem, T, s = 2, 2, 40, 2
    ang = np.random.default_rng(0).uniform(-1, 1, (2, s, T, n_in))       # (R, s, T, n_in)
    reservoirs = [XXZRandomField(n_in + n_mem, seed, delta=0.7, W=2.0) for seed in (11, 12)]
    # chunk_size 24 leaves several steps per reset batch, so a batch really is (stream, step) sized
    # and its reservoir-major ordering is exercised rather than trivial
    kw = dict(taus=(1.0, 0.5), memory=memory, k_fb=k_fb, mem_depth=3, backend="numpy", precision="double",
              chunk_size=24, streams=True)
    together = qrc_features(ang, n_in, n_mem, reservoirs, **kw)
    assert together.shape[:3] == (2, s, T)
    for r, res in enumerate(reservoirs):
        one = qrc_features(ang[r], n_in, n_mem, res, **kw)
        assert one.shape == together[r].shape
        assert np.max(np.abs(together[r] - one)) < 1e-12


def test_reservoir_list_rejects_a_mismatched_leading_axis():
    res = [IsingXX(4, 0), IsingXX(4, 1)]
    ang = np.zeros((3, 2, 5, 2))
    with pytest.raises(ValueError, match="reservoirs"):
        qrc_features(ang, 2, 2, res, streams=True)
    with pytest.raises(ValueError, match="streams"):
        qrc_features(np.zeros((2, 5, 2)), 2, 2, res)


def test_driver_takes_one_or_many_unitaries_and_checks_the_batch():
    n_in, n_mem = 2, 2
    U = np.stack([IsingXX(4, s).unitary(1.0) for s in (0, 1)])
    drv = batched.Driver(U, n_in, n_mem, np, np.complex128)
    assert drv.R == 2 and drv.U4.shape == (2, 2 ** n_in, drv.D, 2 ** n_mem)
    psi = np.zeros((4, 2 ** n_in), complex)
    assert drv._W(psi).shape == (4, drv.D, 2 ** n_mem)
    with pytest.raises(AssertionError):
        drv._W(np.zeros((3, 2 ** n_in), complex))
    one = batched.Driver(U[0], n_in, n_mem, np, np.complex128)
    assert one.R == 1
    both = psi + np.arange(4)[:, None]                  # first two elements belong to reservoir 0
    assert np.array_equal(one._W(both)[:2], drv._W(both)[:2])


@pytest.mark.parametrize("model", ["qrc", "random_features", "esn"])
def test_build_trial_matches_the_per_seed_builds(model):
    rng = np.random.default_rng(2)
    Xs = [rng.standard_normal((40, 2)) for _ in range(3)]
    kind = "recurrent" if model == "esn" else "window"
    cmp = Comparison(kind=kind, n_series=2, input_window=2, n_mem=1)
    params = {"qrc": {"tau": 1.0, "scale": 0.8, "reservoir": {"v": 1.2}, "k_fb": 0.0},
              "esn": {"sr": 0.9, "leak": 0.3, "in_scale": 0.5, "density": 0.2, "bias_scale": 0.5},
              "random_features": {"scale": 1.0, "bias_scale": 0.5}}[model]
    seeds = (5, 6)
    built = build_trial(model, params, Xs, seeds, cmp, Sim())
    assert set(built) == {(s, i) for s in seeds for i in range(len(Xs))}
    for seed in seeds:
        for i, F in enumerate(build_streams(model, params, Xs, seed, cmp, Sim())):
            assert np.allclose(built[(seed, i)], F, atol=1e-12)
        if model != "qrc":
            for i, X in enumerate(Xs):
                assert np.allclose(built[(seed, i)], MODELS[model].build(params, X, seed, cmp, Sim()), atol=1e-12)


def test_build_trial_falls_back_to_per_series_when_lengths_differ():
    rng = np.random.default_rng(3)
    Xs = [rng.standard_normal((30, 2)), rng.standard_normal((25, 2))]
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    params = {"tau": 1.0, "scale": 0.8, "reservoir": {"v": 1.2}, "k_fb": 0.0}
    built = build_trial("qrc", params, Xs, (5,), cmp, Sim())
    for i, X in enumerate(Xs):
        assert np.allclose(built[(5, i)], MODELS["qrc"].build(params, X, 5, cmp, Sim()), atol=1e-12)


def test_tune_model_keeps_every_data_and_reservoir_run():
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(tune_res_seeds=(100, 101), **TINY)
    rec = tune_model("qrc", cmp, proto, Sim(), log=lambda s: None)
    assert set(rec["best_val_per_run"]) == {f"ds{ds}_rs{rs}" for ds in (100, 101) for rs in (100, 101)}
    assert np.isclose(rec["best_val"], np.mean(list(rec["best_val_per_run"].values())))


@pytest.mark.skipif(not HAS_CUPY, reason="no CUDA device")
def test_cupy_single_precision_multi_reservoir_matches_numpy():
    n_in, n_mem, T, s = 2, 3, 24, 2
    ang = np.random.default_rng(4).uniform(-1, 1, (2, s, T, n_in))
    reservoirs = [XXZRandomField(n_in + n_mem, seed, delta=0.6, W=1.3) for seed in (3, 4)]
    kw = dict(taus=(0.8,), memory="recurrent", k_fb=0.5, streams=True)
    a = qrc_features(ang, n_in, n_mem, reservoirs, backend="numpy", precision="double", **kw)
    b = qrc_features(ang, n_in, n_mem, reservoirs, backend="cupy", precision="single", **kw)
    assert isinstance(b, np.ndarray) and b.shape == a.shape
    assert np.max(np.abs(a - b)) < 1e-5
