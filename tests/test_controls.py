import numpy as np

from qrc_bench import registry
from qrc_bench.baselines import random_features, windowed


def _X():
    return np.random.default_rng(0).standard_normal((50, 3))


def test_random_features_shape_range_and_seed():
    X = _X()
    F = random_features(X, width=40, seed=1, window=2, scale=0.7, bias_scale=0.3)
    assert F.shape == (50, 40) and np.all(np.abs(F) < 1)
    assert np.array_equal(F, random_features(X, width=40, seed=1, window=2, scale=0.7, bias_scale=0.3))
    assert not np.allclose(F, random_features(X, width=40, seed=2, window=2, scale=0.7, bias_scale=0.3))


def test_random_features_see_only_their_window():
    X = _X()
    Y = X.copy()
    Y[:30] += 5.0                       # change only steps older than the window of step 40
    a = random_features(X, width=10, seed=0, window=3)
    b = random_features(Y, width=10, seed=0, window=3)
    assert np.allclose(a[35:], b[35:])
    assert not np.allclose(a[:30], b[:30])


def test_random_features_are_tanh_of_affine_window_map():
    X = _X()
    F = random_features(X, width=5, seed=3, window=2, scale=1.0, bias_scale=0.0)
    Z = windowed(X, 2)
    # zero bias: odd function of the window
    assert np.allclose(random_features(-X, width=5, seed=3, window=2, scale=1.0, bias_scale=0.0), -F)
    assert F.shape[0] == Z.shape[0]


def test_registered():
    assert "random_features" in registry.names("baseline")
