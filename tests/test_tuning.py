import numpy as np

from qrc_bench.cli import build_parser
from qrc_bench.evaluate import ridge_forecast, test_nrmse, val_nrmse
from qrc_bench.tasks.ar import regime_switching
from qrc_bench.tuning import Context, tune_all, tune_model


def _X():
    return regime_switching(160, seed=0)


def test_fixed_alpha_scores_match_ridge_forecast():
    X = _X()
    F = np.hstack([X, X ** 2])
    assert np.isclose(test_nrmse(F, X, 1, 1e-2), ridge_forecast(F, X, 1, alphas=(1e-2,)))
    assert np.isfinite(val_nrmse(F, X, 1, 1e-2))


def test_same_protocol_for_every_model():
    ctx = Context(_X(), res_seed=0, window=2, lag_window=2, n_mem_range=(1, 2), esn_units=6)
    rec = tune_all(["qrc", "esn", "poly2", "linear"], ctx, n_trials=3, sampler_seed=1, log=lambda s: None)
    assert {r["n_trials"] for r in rec["models"].values()} == {3}
    assert {r["sampler_seed"] for r in rec["models"].values()} == {1}
    for r in rec["models"].values():
        assert "alpha" in r["search_space"]
        assert np.isfinite(r["val_nrmse"]) and np.isfinite(r["test_nrmse"])
    assert rec["models"]["poly2"]["width"] == 5          # 2 lags -> 2 + 3
    assert rec["models"]["esn"]["best_params"]["n_res"] == 6


def test_qrc_respects_q_cap_and_open_loop():
    ctx = Context(_X(), res_seed=0, n_mem_range=(1, 20), feedback=False, q_max=3)
    rec = tune_model("qrc", ctx, n_trials=2, log=lambda s: None)
    assert rec["search_space"]["n_mem"]["attributes"]["high"] == 2
    assert rec["best_params"]["k_fb"] == 0.0


def test_enqueue_defaults_runs_defaults_first():
    ctx = Context(_X(), res_seed=0, esn_units=5)
    rec = tune_model("esn", ctx, n_trials=1, enqueue_defaults=True, log=lambda s: None)
    assert rec["best_params"]["sr"] == 0.9 and rec["best_alpha"] == 1e-3


def test_cli_tune_arguments_parse():
    a = build_parser().parse_args(["tune", "--task", "switching", "--task-arg", "p_switch=0.05",
                                   "--data-seeds", "0", "1", "--trials", "100", "--window", "5"])
    assert (a.trials, a.data_seeds, a.task_arg, a.models) == (100, [0, 1], ["p_switch=0.05"],
                                                                ["qrc", "esn", "poly2", "linear"])
