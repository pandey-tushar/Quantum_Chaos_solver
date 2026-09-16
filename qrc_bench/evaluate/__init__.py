"""Chronological 70/15/15 split, ridge readout, NRMSE."""
from __future__ import annotations

import numpy as np

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def _fit(F, Y, alpha):
    return np.linalg.solve(F.T @ F + alpha * np.eye(F.shape[1]), F.T @ Y)


def _splits(T, train_frac, val_frac):
    return int(round(train_frac * T)), int(round((train_frac + val_frac) * T))


def ridge_forecast(F, Y, horizon=1, train_frac=0.7, val_frac=0.15, alphas=ALPHAS) -> float:
    """Test NRMSE of predicting Y[t+h] from F[t].

    Ridge alpha is chosen on validation, the readout is refit on train + validation,
    and the test slice is scored once. NRMSE is RMSE / std of the test target, averaged
    over output columns, so 1.0 is the trivial-mean predictor.
    """
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


def val_mse(F, Y, horizon=1, alpha=1e-3, train_frac=0.7, val_frac=0.15) -> float:
    """Validation MSE at a fixed ridge alpha; used to choose hyperparameters. Never touches test."""
    T, h = len(F), horizon
    ntr, nval = _splits(T, train_frac, val_frac)
    if nval - h <= ntr or T <= nval:
        return float("inf")
    W = _fit(F[:ntr - h], Y[h:ntr], alpha)
    return float(np.mean((F[ntr:nval - h] @ W - Y[ntr + h:nval]) ** 2))
