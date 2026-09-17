import numpy as np
import pytest

from qrc_bench import registry
from qrc_bench.encoders import dense_rxrz, encode, n_encoded_qubits
from qrc_bench.layout import derive
from qrc_bench.qrc import qrc_features
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.simulate.ops import product_state
from qrc_bench.tuning import Comparison, Protocol, Sim, tune_model


@pytest.mark.parametrize("name,kw,n", [("lorenz96", {"n_series": 10}, 10), ("switching_multi", {"n_series": 9}, 9),
                                       ("henon", {"n_series": 12}, 12)])
def test_large_tasks_are_standardised_and_seeded(name, kw, n):
    task = registry.get("task", name)
    X = task(400, 3, **kw)
    assert X.shape == (400, n) and np.all(np.isfinite(X))
    assert np.allclose(X.mean(0), 0, atol=1e-8) and np.allclose(X.std(0), 1, atol=1e-8)
    assert np.array_equal(X, task(400, 3, **kw)) and not np.allclose(X, task(400, 4, **kw))


def test_lorenz96_is_chaotic_not_periodic():
    X = registry.get("task", "lorenz96")(600, 0, n_series=10)
    lag1 = np.mean([np.corrcoef(X[:-1, k], X[1:, k])[0, 1] for k in range(10)])
    assert 0.3 < lag1 < 0.999


def test_switching_multi_shares_one_hidden_regime():
    X = registry.get("task", "switching_multi")(2000, 1, n_series=4, p_switch=0.0, coupling=0.0)
    # with no switching every series is the same AR(1) sign regime: positive lag-1 correlation
    assert all(np.corrcoef(X[:-1, k], X[1:, k])[0, 1] > 0.5 for k in range(4))


def test_dense_encoding_pairs_series_per_qubit():
    X = np.arange(10.0).reshape(2, 5) / 10
    A = dense_rxrz(X, scale=2.0)
    assert A.shape == (2, 3, 2) and n_encoded_qubits(5, "dense_rxrz") == 3
    assert np.allclose(A[0, :, 0], [0.0, 0.4, 0.8]) and np.allclose(A[0, :, 1], [0.2, 0.6, 0.0])   # odd: pad 0
    assert encode(X, "per_series", 2.0).shape == (2, 5)


def test_dense_product_state_is_rz_ry():
    a, b = 0.7, -1.2
    psi = product_state(np.array([[a, b]]))
    assert np.allclose(psi, [np.cos(a / 2) * np.exp(-1j * b / 2), np.sin(a / 2) * np.exp(1j * b / 2)])


@pytest.mark.parametrize("k_fb", [0.0, 1.0])
def test_dense_encoding_batched_matches_reference(k_fb):
    X = np.random.default_rng(0).uniform(-1, 1, (12, 4))
    ang = dense_rxrz(X, 1.0)
    res = IsingXX(4, 1)
    a = qrc_features(ang, 2, 2, res, method="dense", k_fb=k_fb)
    b = qrc_features(ang, 2, 2, res, method="batched", k_fb=k_fb)
    assert np.max(np.abs(a - b)) < 1e-11


def test_layout_and_comparison_with_dense_encoding():
    assert derive(9, encoding="dense_rxrz", n_mem=3).q == 8
    cmp = Comparison(kind="window", n_series=9, input_window=2, n_mem=3, encoding="dense_rxrz")
    assert (cmp.n_in, cmp.q) == (5, 8)
    proto = Protocol(task="lorenz96", task_kwargs={"n_series": 9}, n_steps=160, n_trials=1, n_folds=2, washout=10,
                     tune_data_seeds=(100,), tune_res_seeds=(100,), eval_data_seeds=(0,), eval_res_seeds=(0,))
    rec = tune_model("qrc", cmp, proto, Sim(), log=lambda s: None)
    assert np.isfinite(rec["best_val"])


def test_dense_encoding_cupy_matches_numpy():
    pytest.importorskip("cupy")
    X = np.random.default_rng(1).uniform(-1, 1, (20, 5))
    ang = dense_rxrz(X, 1.3)
    res = IsingXX(5, 2)
    for memory, k_fb in (("reset", 0.8), ("recurrent", 0.8)):
        a = qrc_features(ang, 3, 2, res, memory=memory, k_fb=k_fb)
        b = qrc_features(ang, 3, 2, res, memory=memory, k_fb=k_fb, backend="cupy", precision="double")
        assert np.max(np.abs(a - b)) < 1e-11
