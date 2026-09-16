"""The cartography paper's protocols (arXiv:2607.09905), rebuilt from pipeline modules.

These reproduce the paper's per-seed-pair numbers exactly and are the regression
reference for every refactor: grid search on validation, same candidate order and
tie-breaking as the original scripts.
"""
from __future__ import annotations

import numpy as np

from qrc_bench.baselines import esn_features, linear_features, poly2_features, windowed
from qrc_bench.encoders import angle_ry
from qrc_bench.evaluate import ridge_forecast, val_mse
from qrc_bench.qrc import qrc_features
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.tasks.ar import regime_switching
from qrc_bench.tasks.henon import coupled_henon


def _argmin_val(candidates, Y, h):
    """First candidate with the strictly lowest validation MSE (paper's tie-breaking)."""
    best = None
    for key, F in candidates:
        v = val_mse(F, Y, h)
        if best is None or v < best[0]:
            best = (v, key, F)
    return best[1], best[2]


def case_i_pair(coupling: float, data_seed: int, res_seed: int, n_steps: int = 800,
                n_series: int = 5, n_mem: int = 3, obs_noise: float = 0.1, horizon: int = 1,
                scales=(np.pi / 4, np.pi / 2), taus=(1.0, 2.0), method: str = "branch",
                backend: str = "numpy") -> dict:
    """Case I concat ablation: QRC (ZZ or ZZ_QR2, scale, tau chosen on validation), Poly2, Poly2 + QRC."""
    X = coupled_henon(n_steps, data_seed, n_series=n_series, coupling=coupling, obs_noise=obs_noise)
    res = IsingXX(n_series + n_mem, res_seed, seed_offset=10_000)
    blocks = {}
    for sc in scales:
        ang = angle_ry(X, sc)
        for tp in sorted({t for tau in taus for t in (tau, tau / 2.0)}):
            blocks[(sc, tp)] = qrc_features(ang, n_series, n_mem, res, taus=(tp,), readout="ZZ",
                                            method=method, backend=backend)
    candidates = []
    for readout in ("ZZ", "ZZ_QR2"):
        for sc in scales:
            for tau in taus:
                F = blocks[(sc, tau)]
                if readout == "ZZ_QR2":
                    F = np.concatenate([F, blocks[(sc, tau / 2.0)]], axis=1)
                candidates.append(((readout, round(sc, 4), tau), F))
    (readout, sc, tau), Fq = _argmin_val(candidates, X, horizon)
    Fpoly = poly2_features(X, window=3)
    return {"Poly2": ridge_forecast(Fpoly, X, horizon),
            "QRC": ridge_forecast(Fq, X, horizon),
            "Poly2+QRC": ridge_forecast(np.concatenate([Fpoly, Fq], axis=1), X, horizon),
            "qrc_cfg": {"readout": readout, "scale": sc, "tau": tau},
            "D": {"QRC": int(Fq.shape[1]), "Poly2": int(Fpoly.shape[1])}}


def case_ii_pair(data_seed: int, res_seed: int, p_switch: float = 0.05, n_steps: int = 1200,
                 n_mem: int = 4, window: int = 5, horizon: int = 1, k_fbs=(0.5, 1.0, 2.0),
                 taus=(1.0, 2.0), esn_srs=(0.7, 0.9, 0.99), esn_leaks=(0.2, 0.5, 0.9),
                 method: str = "dense", backend: str = "numpy") -> dict:
    """Case II narrow tuning: feedback QRC (tau, k_fb), open loop (tau), ESN (sr, leak), Poly2, linear."""
    X = regime_switching(n_steps, data_seed, p_switch=p_switch)
    ang = angle_ry(X, 1.0)
    res = IsingXX(1 + n_mem, res_seed, seed_offset=20_000)

    def feats(tau, k):
        return windowed(qrc_features(ang, 1, n_mem, res, taus=(tau,), readout="ZZ", k_fb=k,
                                     method=method, backend=backend), window)

    open_c, fb_c = [], []
    for tau in taus:
        open_c.append(((tau, 0.0), feats(tau, 0.0)))
        fb_c.extend(((tau, k), feats(tau, k)) for k in k_fbs)
    open_cfg, F_open = _argmin_val(open_c, X, horizon)
    fb_cfg, F_fb = _argmin_val(fb_c, X, horizon)
    n_res = F_fb.shape[1] // window
    esn_c = [((sr, lk), windowed(esn_features(X, n_res, res_seed, sr=sr, leak=lk), window))
             for sr in esn_srs for lk in esn_leaks]
    esn_cfg, F_esn = _argmin_val(esn_c, X, horizon)
    return {"FB_QRC": ridge_forecast(F_fb, X, horizon),
            "OpenLoop_QRC": ridge_forecast(F_open, X, horizon),
            "ESN": ridge_forecast(F_esn, X, horizon),
            "Poly2": ridge_forecast(poly2_features(X, window), X, horizon),
            "Linear": ridge_forecast(linear_features(X, window), X, horizon),
            "cfg": {"FB_QRC": {"tau": fb_cfg[0], "k_fb": fb_cfg[1]}, "OpenLoop_QRC": {"tau": open_cfg[0]},
                    "ESN": {"sr": esn_cfg[0], "leak": esn_cfg[1], "n_res": n_res}}}
