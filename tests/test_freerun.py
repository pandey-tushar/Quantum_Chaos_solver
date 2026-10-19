"""Free-running forecasts: every stepper reproduces its batch builder, the readout is the one the
test score fits, and the valid prediction time is measured in Lyapunov times of the same integrator."""
import json

import numpy as np
import pytest

from qrc_bench.baselines import esn_features, esn_step, esn_weights
from qrc_bench.cli import build_parser
from qrc_bench.evaluate import fit_readout, score_test, test_start
from qrc_bench.qrc import RecurrentQRC, qrc_features
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.tasks.lorenz96 import _l96_rhs, lorenz96
from qrc_bench.tuning import Comparison, Protocol, Sim, run_comparison
from qrc_bench.tuning.spaces import MODELS

try:
    import cupy  # noqa: F401
    HAS_CUPY = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUPY = False

L96 = dict(task="lorenz96", task_kwargs={"n_series": 4, "sample_every": 5}, n_steps=160, n_trials=2, n_folds=2,
           washout=10, tune_data_seeds=(100, 101), tune_res_seeds=(100,), eval_data_seeds=(0, 1),
           eval_res_seeds=(0, 1))


# --- Lorenz-96 integrator and Lyapunov exponent ---------------------------------------------

def _lorenz96_before_refactor(n_steps, seed, n_series=10, forcing=8.0, dt=0.01, sample_every=5, obs_noise=0.0,
                              burn=1000):
    rng = np.random.default_rng(seed)
    x = forcing + 0.01 * rng.standard_normal(n_series)
    out = np.zeros((n_steps + burn, n_series))
    for i in range(n_steps + burn):
        for _ in range(sample_every):
            k1 = _l96_rhs(x, forcing)
            k2 = _l96_rhs(x + 0.5 * dt * k1, forcing)
            k3 = _l96_rhs(x + 0.5 * dt * k2, forcing)
            k4 = _l96_rhs(x + dt * k3, forcing)
            x = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i] = x
    X = out[burn:]
    if obs_noise > 0:
        X = X + obs_noise * X.std(0) * rng.standard_normal(X.shape)
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


@pytest.mark.parametrize("kw", [dict(n_series=6, sample_every=5, burn=50), dict(n_series=5, obs_noise=0.1, burn=20)])
def test_rk4_factoring_keeps_lorenz96_bit_identical(kw):
    assert np.array_equal(lorenz96(200, 3, **kw), _lorenz96_before_refactor(200, 3, **kw))


def test_lyapunov_exponent_matches_the_literature_for_k40():
    from qrc_bench.freerun import lyapunov_exponent
    lam = lyapunov_exponent(40, forcing=8.0)
    assert 1.55 <= lam <= 1.80


def test_lyapunov_exponent_k6_is_positive_and_deterministic():
    from qrc_bench.freerun import lyapunov_exponent
    a = lyapunov_exponent(6, forcing=8.0, t_total=200.0)
    assert a > 0 and a == lyapunov_exponent(6, forcing=8.0, t_total=200.0)


def test_lyapunov_time_in_sample_steps():
    from qrc_bench.freerun import lyapunov_time_steps
    proto = Protocol(**L96)                                   # dt 0.01 (default), sample_every 5
    assert np.isclose(lyapunov_time_steps(2.0, proto), 1.0 / (2.0 * 0.01 * 5))


# --- readout ----------------------------------------------------------------------------------

def test_fit_readout_is_the_ridge_score_test_fits():
    rng = np.random.default_rng(0)
    F, Y = rng.standard_normal((200, 7)), rng.standard_normal((200, 3))
    alpha, washout, h = 0.3, 10, 1
    s = test_start(len(F), 0.15)
    for standardize in (False, True):
        # the pre-refactor _score_fold
        Ftr, Ytr, Fev = F[washout:s - h], Y[washout + h:s], F[s:len(F) - h]
        if standardize:
            mu, sd = Ftr.mean(0), Ftr.std(0)
            ym = Ytr.mean(0)
            Z = (Ftr - mu) / sd
            P = ((Fev - mu) / sd) @ np.linalg.solve(Z.T @ Z + alpha * np.eye(7), Z.T @ (Ytr - ym)) + ym
        else:
            P = Fev @ np.linalg.solve(Ftr.T @ Ftr + alpha * np.eye(7), Ftr.T @ Ytr)
        Yte = Y[s + h:]
        ref = np.mean(np.sqrt(np.mean((P - Yte) ** 2, 0)) / (np.std(Yte, 0) + 1e-12))
        assert abs(score_test(F, Y, h, alpha, 0.15, standardize, washout) - ref) < 1e-12
        assert np.max(np.abs(fit_readout(F, Y, h, alpha, standardize, washout, s)(Fev) - P)) < 1e-12


# --- steppers ---------------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["qrc", "random_features", "poly2", "linear"])
def test_window_stepper_reproduces_batch_features(model):
    from qrc_bench.freerun import WindowStepper
    cmp = Comparison(kind="window", n_series=3, input_window=3, n_mem=2)
    params = {"qrc": {"tau": 0.9, "scale": 0.8, "reservoir": {}, "k_fb": 0.0},
              "random_features": {"scale": 0.7, "bias_scale": 0.4}}.get(model, {})
    sim = Sim(backend="numpy", precision="double")
    X = np.random.default_rng(1).standard_normal((40, 3))
    F = MODELS[model].build(params, X, 5, cmp, sim)
    ts = np.array([2, 11, 30, 39])
    buf = np.stack([X[t - 2:t + 1] for t in ts])
    out = WindowStepper(model, params, cmp, sim, 5).features(buf, step=1)
    assert out.shape == (len(ts), F.shape[1])
    assert np.max(np.abs(out - F[ts])) < 1e-12


def test_window_stepper_refuses_feedback():
    from qrc_bench.freerun import WindowStepper
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    params = {"tau": 0.9, "scale": 0.8, "reservoir": {}, "k_fb": 0.5}
    with pytest.raises(ValueError, match="k_fb"):
        WindowStepper("qrc", params, cmp, Sim(backend="numpy"), 0)


BACKENDS = ["numpy"] + (["cupy"] if HAS_CUPY else [])


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("k_fb", [0.0, 1.2])
@pytest.mark.parametrize("shots", [None, 256])
@pytest.mark.parametrize("readout", ["ZZ", "ZX"])
def test_recurrent_stepper_is_simulate_bit_for_bit(backend, k_fb, shots, readout):
    n_in, n_mem, S, T = 2, 2, 3, 12
    res = IsingXX(n_in + n_mem, seed=3)
    ang = np.random.default_rng(2).uniform(-1.5, 1.5, (S, T, n_in))
    kw = dict(taus=(0.9,), readout=readout, k_fb=k_fb, backend=backend, shots=shots, shot_seed=7)
    ref = qrc_features(ang, n_in, n_mem, res, memory="recurrent", streams=True, **kw)
    st = RecurrentQRC(n_in, n_mem, res, S, **kw)
    out = np.stack([st.step(ang[:, t]) for t in range(T)], axis=1)
    assert out.dtype == ref.dtype and np.array_equal(out, ref)


@pytest.mark.parametrize("k_fb", [0.0, 1.2])
def test_recurrent_snapshots_fork_into_one_batch(k_fb):
    """Warm one stream, snapshot at two steps, restore both as a batch of two and continue on the
    true inputs: each continues as the uninterrupted run (to round-off: the batch is regrouped)."""
    n_in, n_mem, T = 2, 2, 14
    res = IsingXX(n_in + n_mem, seed=4)
    ang = np.random.default_rng(5).uniform(-1.5, 1.5, (T, n_in))
    kw = dict(taus=(0.9, 0.45), k_fb=k_fb, backend="numpy")
    ref = qrc_features(ang, n_in, n_mem, res, memory="recurrent", **kw)
    st = RecurrentQRC(n_in, n_mem, res, 1, **kw)
    snaps = {}
    for t in range(9):
        st.step(ang[t:t + 1])
        snaps[t] = st.snapshot()
    st.restore([snaps[5], snaps[8]])
    for k in range(1, 6):
        out = st.step(np.stack([ang[5 + k], ang[8 + k]]))
        assert np.max(np.abs(out - ref[[5 + k, 8 + k]])) < 1e-13


def test_esn_step_reproduces_esn_features():
    X = np.random.default_rng(3).standard_normal((50, 4))
    p = dict(sr=0.8, leak=0.4, in_scale=0.6, density=0.3, bias_scale=1.5)
    F = esn_features(X, 12, 9, **p)
    W_in, W, b = esn_weights(4, 12, 9, p["sr"], p["in_scale"], p["density"], p["bias_scale"])
    ts = np.array([1, 17, 49])
    r = np.stack([esn_step(F[t - 1], X[t], W_in, W, b, p["leak"]) for t in ts])
    assert np.array_equal(r, F[ts])


# --- free run ---------------------------------------------------------------------------------

def _oracle(persist):
    """A stand-in for fit_readout: linear features at window 1 are the state itself, so the true
    next state is found by looking the feature row up in the series (or repeated: persistence)."""
    def fake(F, Y, horizon, alpha, standardize, washout, fit_end):
        def predict(Fnew):
            if persist:
                return Fnew.copy()
            idx = [np.flatnonzero((Y == f).all(axis=1))[0] for f in Fnew]
            return Y[np.array(idx) + 1]
        return predict
    return fake


def test_free_run_oracle_is_valid_for_the_whole_horizon(monkeypatch):
    from qrc_bench import freerun
    cmp = Comparison(kind="window", n_series=4, input_window=1, n_mem=1)
    proto = Protocol(**L96)
    monkeypatch.setattr(freerun, "fit_readout", _oracle(persist=False))
    out = freerun.free_run("linear", {}, 1.0, cmp, proto, Sim(), 0, 0, n_starts=4, horizon_steps=8, threshold=0.5,
                           lt_steps=4.0)
    assert out["vpt_mean"] == 2.0 and out["vpt_per_start"] == [2.0] * 4
    assert len(out["nrmse_curve"]) == 8 and max(out["nrmse_curve"]) == 0.0


def test_free_run_persistence_degrades(monkeypatch):
    from qrc_bench import freerun
    cmp = Comparison(kind="window", n_series=4, input_window=1, n_mem=1)
    proto = Protocol(**L96)
    monkeypatch.setattr(freerun, "fit_readout", _oracle(persist=True))
    out = freerun.free_run("linear", {}, 1.0, cmp, proto, Sim(), 0, 0, n_starts=5, horizon_steps=20, threshold=0.5,
                           lt_steps=4.0)
    curve = np.array(out["nrmse_curve"])
    assert out["vpt_mean"] > 0
    assert np.all(np.diff(curve[:4]) > 0)


@pytest.mark.parametrize("kind,model,params", [
    ("window", "qrc", {"tau": 0.9, "scale": 0.8, "reservoir": {}, "k_fb": 0.0}),
    ("recurrent", "qrc", {"tau": 0.9, "scale": 0.8, "reservoir": {}, "k_fb": 1.0}),
    ("recurrent", "esn", {"sr": 0.8, "leak": 0.5, "in_scale": 0.5, "density": 0.3, "bias_scale": 1.0}),
    ("window", "random_features", {"scale": 0.7, "bias_scale": 0.4})])
def test_free_run_every_model_kind(kind, model, params):
    from qrc_bench.freerun import free_run
    cmp = Comparison(kind=kind, n_series=4, input_window=2, n_mem=1, shots=512)
    out = free_run(model, params, 1e-2, cmp, Protocol(**L96), Sim(), 1, 0, n_starts=3, horizon_steps=6,
                   threshold=0.5, lt_steps=3.0)
    assert len(out["vpt_per_start"]) == 3 and len(out["nrmse_curve"]) == 6
    assert np.isfinite(out["nrmse_curve"]).all() and 0.0 <= out["vpt_mean"] <= 2.0


def test_run_freerun_writes_record_and_stats(tmp_path, monkeypatch):
    from qrc_bench import freerun
    cmp = Comparison(kind="window", n_series=4, input_window=2, n_mem=1)
    models = ["qrc", "random_features", "poly2", "linear"]
    run_comparison(cmp, Protocol(**L96), Sim(), models=models, out_dir=tmp_path, log=lambda s: None)
    monkeypatch.setattr(freerun, "lyapunov_exponent", lambda *a, **k: 2.0)     # 10 steps per Lyapunov time
    lines = []
    freerun.run_freerun(tmp_path / "comparison_window.json", n_starts=3, horizon_lt=1.0, log=lines.append)
    saved = json.loads((tmp_path / "freerun_window.json").read_text())
    for key in ("source", "lyapunov", "settings", "models", "stats", "environment"):
        assert key in saved
    assert np.isclose(saved["lyapunov"]["time_steps"], 10.0) and saved["settings"]["horizon_steps"] == 10
    for m in models:
        rows = saved["models"][m]["rows"]
        assert [(r["data_seed"], r["res_seed"]) for r in rows] == [(0, 0), (0, 1), (1, 0), (1, 1)]
        assert all(len(r["nrmse_curve"]) == 10 for r in rows)
        assert set(saved["models"][m]["summary"]) >= {"vpt_mean", "vpt_std", "nrmse_at_lt"}
        assert set(saved["models"][m]["summary"]["nrmse_at_lt"]) == {"1"}
    assert set(saved["stats"]) == {f"qrc_vs_{m}" for m in models[1:]}
    assert "p_holm_cluster" in saved["stats"]["qrc_vs_poly2"]
    assert any("VPT" in s for s in lines)


def test_cli_freerun_arguments_parse():
    a = build_parser().parse_args(["freerun", "--comparison", "x/comparison_window.json"])
    assert (a.comparison, a.starts, a.horizon_lt, a.threshold, a.out) == ("x/comparison_window.json", 20, 10.0,
                                                                          0.5, None)
    a = build_parser().parse_args(["freerun", "--comparison", "c.json", "--starts", "2", "--horizon-lt", "1",
                                   "--threshold", "0.3", "--out", "o.json"])
    assert (a.starts, a.horizon_lt, a.threshold, a.out) == (2, 1.0, 0.3, "o.json")


def test_restore_refuses_snapshots_of_different_shapes():
    """Factor states grow a column per step until they become density matrices: snapshots taken at
    different ranks cannot share one batch."""
    from qrc_bench.simulate.batched import Driver, RecurrentStepper
    n_in, n_mem = 1, 3                                         # d1 = 2, d2 = 8: factor rank 2, then 4
    U = IsingXX(n_in + n_mem, seed=1).unitary(0.7)
    st = RecurrentStepper(Driver(U, n_in, n_mem, np, np.complex128), 1, lambda P: P)
    st.step(np.array([[0.3]]))
    a = st.snapshot()
    st.step(np.array([[0.5]]))
    b = st.snapshot()
    assert a[0][0] == b[0][0] == "factor" and a[0][1].shape != b[0][1].shape
    with pytest.raises(ValueError, match="shape"):
        st.restore([a, b])
