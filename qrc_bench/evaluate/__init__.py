"""Chronological splits, ridge readout, NRMSE.

NRMSE is RMSE / std of the scored target, averaged over output columns, so 1.0 is the
trivial-mean predictor. ``ridge_forecast`` and ``val_mse`` keep the cartography paper's exact
protocol (no standardisation, no intercept). The tuning protocol uses ``standardize=True``:
features are z-scored and the target centred with statistics of the fitted rows only, so
one ridge penalty range treats every model's features alike.
"""
from __future__ import annotations

import numpy as np

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _fit(F, Y, alpha):
    return np.linalg.solve(F.T @ F + alpha * np.eye(F.shape[1]), F.T @ Y)


def _splits(T, train_frac, val_frac):
    return int(round(train_frac * T)), int(round((train_frac + val_frac) * T))


def _nrmse(P, Y):
    return float(np.mean(np.sqrt(np.mean((P - Y) ** 2, axis=0)) / (np.std(Y, axis=0) + 1e-12)))


def fit_predict(Ftr, Ytr, Fev, alpha, standardize=False):
    """Ridge fit on (Ftr, Ytr), prediction for Fev. Standardised: z-score features and centre
    the target with training statistics (constant columns become zero)."""
    if not standardize:
        return Fev @ _fit(Ftr, Ytr, alpha)
    mu, sd = Ftr.mean(axis=0), Ftr.std(axis=0)
    sd = np.where(sd > 1e-12 * np.maximum(1.0, np.abs(mu)), sd, 1.0)
    ym = Ytr.mean(axis=0)
    W = _fit((Ftr - mu) / sd, Ytr - ym, alpha)
    return ((Fev - mu) / sd) @ W + ym


def ridge_forecast(F, Y, horizon=1, train_frac=0.7, val_frac=0.15, alphas=ALPHAS) -> float:
    """The paper's protocol: test NRMSE of predicting Y[t+h] from F[t], alpha chosen on validation
    from a grid, readout refit on train + validation, test scored once."""
    T, h = len(F), horizon
    ntr, nval = _splits(T, train_frac, val_frac)
    if T - h <= nval + 5:
        return float("nan")
    Fva, Yva = F[ntr:nval - h], Y[ntr + h:nval]
    best_a, best_v = alphas[0], np.inf
    for a in alphas:
        v = np.mean((Fva @ _fit(F[:ntr - h], Y[h:ntr], a) - Yva) ** 2)
        if v < best_v:
            best_v, best_a = v, a
    W = _fit(F[:nval - h], Y[h:nval], best_a)
    Yte = Y[nval + h:T]
    rmse = np.sqrt(np.mean((F[nval:T - h] @ W - Yte) ** 2, axis=0))
    return float(np.mean(rmse / (np.std(Yte, axis=0) + 1e-12)))


def val_nrmse(F, Y, horizon=1, alpha=1e-3, train_frac=0.7, val_frac=0.15, standardize=False, washout=0) -> float:
    """Validation NRMSE of a readout fit on the training slice at a fixed alpha. Never touches test."""
    T, h = len(F), horizon
    ntr, nval = _splits(T, train_frac, val_frac)
    return _score_fold(F, Y, h, alpha, ntr, nval, standardize, washout)


def test_nrmse(F, Y, horizon=1, alpha=1e-3, train_frac=0.7, val_frac=0.15, standardize=False, washout=0) -> float:
    """Test NRMSE of a readout refit on everything before the test slice. Call once, after tuning."""
    T, h = len(F), horizon
    _, nval = _splits(T, train_frac, val_frac)
    if T - h <= nval + 5:
        return float("nan")
    return _score_fold(F, Y, h, alpha, nval, T, standardize, washout)


test_nrmse.__test__ = False   # not a pytest test


def fold_bounds(T: int, n_folds: int = 3, val_frac: float = 0.10, test_frac: float = 0.15):
    """Rolling-origin validation folds [(train_end, val_end), ...], all ending before the test slice."""
    val_len = int(round(val_frac * T))
    test_start = T - int(round(test_frac * T))
    bounds = [(test_start - (n_folds - k) * val_len, test_start - (n_folds - k - 1) * val_len)
              for k in range(n_folds)]
    if bounds[0][0] <= 0:
        raise ValueError(f"{n_folds} folds of {val_len} steps do not fit before the test slice")
    return bounds


def val_nrmse_folds(F, Y, horizon=1, alpha=1e-3, folds=None, standardize=True, washout=0) -> float:
    """Mean validation NRMSE over rolling-origin folds (train on [washout, e), validate on [e, v))."""
    folds = folds or fold_bounds(len(F))
    return float(np.mean([_score_fold(F, Y, horizon, alpha, e, v, standardize, washout) for e, v in folds]))


def test_start(T: int, test_frac: float = 0.15) -> int:
    return T - int(round(test_frac * T))


test_start.__test__ = False


def score_test(F, Y, horizon=1, alpha=1e-3, test_frac=0.15, standardize=True, washout=0) -> float:
    """Final evaluation: fit on [washout, test_start), score the test slice once."""
    return _score_fold(F, Y, horizon, alpha, test_start(len(F), test_frac), len(F), standardize, washout)


def _score_fold(F, Y, h, alpha, fit_end, eval_end, standardize, washout):
    if eval_end - h <= fit_end or fit_end - h <= washout:
        return float("inf")
    P = fit_predict(F[washout:fit_end - h], Y[washout + h:fit_end], F[fit_end:eval_end - h], alpha, standardize)
    return _nrmse(P, Y[fit_end + h:eval_end])


def persistence_nrmse(Y, horizon=1, train_frac=0.7, val_frac=0.15) -> float:
    """No-change forecast Y[t+h] = Y[t] on the paper's test slice (a floor every model should beat)."""
    T, h = len(Y), horizon
    _, nval = _splits(T, train_frac, val_frac)
    return _nrmse(Y[nval:T - h], Y[nval + h:T])


def persistence_test(Y, horizon=1, test_frac=0.15) -> float:
    """No-change forecast on the tuning protocol's test slice."""
    s = test_start(len(Y), test_frac)
    return _nrmse(Y[s:len(Y) - horizon], Y[s + horizon:])


def val_mse(F, Y, horizon=1, alpha=1e-3, train_frac=0.7, val_frac=0.15) -> float:
    """The paper's validation MSE at a fixed alpha, used to choose its grid settings. Never touches test."""
    T, h = len(F), horizon
    ntr, nval = _splits(T, train_frac, val_frac)
    if nval - h <= ntr or T <= nval:
        return float("inf")
    W = _fit(F[:ntr - h], Y[h:ntr], alpha)
    return float(np.mean((F[ntr:nval - h] @ W - Y[ntr + h:nval]) ** 2))
