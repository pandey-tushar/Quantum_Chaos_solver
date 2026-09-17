import numpy as np
import pytest

from qrc_bench.evaluate import (fold_bounds, persistence_nrmse, ridge_forecast, test_nrmse, val_nrmse,
                                val_nrmse_folds)
from qrc_bench.tasks.henon import coupled_henon


def _data():
    X = coupled_henon(300, seed=0, n_series=3)
    F = np.hstack([X, X ** 2, np.roll(X, 1, axis=0)])
    return F, X


def test_unstandardised_matches_paper_ridge():
    F, X = _data()
    assert np.isclose(test_nrmse(F, X, 1, 1e-2), ridge_forecast(F, X, 1, alphas=(1e-2,)))


@pytest.mark.parametrize("fn", [val_nrmse, test_nrmse])
def test_standardised_scores_ignore_feature_scale_and_offset(fn):
    F, X = _data()
    G = F * np.linspace(0.001, 1000, F.shape[1]) + np.linspace(-50, 50, F.shape[1])
    a = fn(F, X, 1, 1e-1, standardize=True)
    b = fn(G, X, 1, 1e-1, standardize=True)
    assert np.isclose(a, b, rtol=1e-8)


def test_standardised_fit_has_intercept():
    F, X = _data()
    # a target offset is absorbed by the centred fit
    assert np.isclose(test_nrmse(F, X + 7.0, 1, 1e-3, standardize=True),
                      test_nrmse(F, X, 1, 1e-3, standardize=True), rtol=1e-8)


def test_constant_feature_column_is_harmless():
    F, X = _data()
    G = np.hstack([F, np.ones((len(F), 1))])
    assert np.isclose(test_nrmse(G, X, 1, 1e-2, standardize=True), test_nrmse(F, X, 1, 1e-2, standardize=True))


def test_rolling_folds_stay_before_the_test_slice():
    bounds = fold_bounds(1000, n_folds=3, val_frac=0.10, test_frac=0.15)
    assert bounds == [(550, 650), (650, 750), (750, 850)]


def test_fold_objective_is_mean_of_folds_and_uses_washout():
    F, X = _data()
    folds = fold_bounds(len(F), n_folds=2, val_frac=0.10, test_frac=0.15)
    v = val_nrmse_folds(F, X, 1, 1e-2, folds=folds, standardize=True, washout=20)
    assert np.isfinite(v)
    G = F.copy()
    G[:20] = 1e6                                    # garbage inside the washout must not matter
    assert np.isclose(val_nrmse_folds(G, X, 1, 1e-2, folds=folds, standardize=True, washout=20), v)


def test_persistence_nrmse_exact():
    Y = np.arange(200.0)[:, None]
    # predicting Y[t+h] = Y[t] misses by h everywhere on the test slice
    _, nval = 140, 170
    yte = Y[nval + 3:]
    assert np.isclose(persistence_nrmse(Y, 3), 3.0 / np.std(yte))
