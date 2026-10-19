"""The fair-tuning protocol: one frozen setting per model, tuned on held-out seeds and folds,
matched inputs and feature width, then evaluated once on fresh seeds."""
import json

import numpy as np
import pytest

from qrc_bench.cli import build_parser
from qrc_bench.tuning import Comparison, Protocol, Sim, evaluate_frozen, run_comparison, tune_model

TINY = dict(task="henon", task_kwargs={"n_series": 2}, n_steps=160, n_trials=2, n_folds=2, washout=10,
            tune_data_seeds=(100, 101), tune_res_seeds=(100,), eval_data_seeds=(0, 1), eval_res_seeds=(0, 1))


def test_comparison_derives_matched_width():
    cmp = Comparison(kind="window", n_series=2, input_window=3, n_mem=2, readout="ZZ", n_taus=2)
    assert (cmp.n_in, cmp.q, cmp.width) == (2, 4, 2 * (4 + 6))
    assert cmp.default_models() == ("qrc", "random_features", "poly2", "linear")
    rec = Comparison(kind="recurrent", n_series=2, input_window=3, n_mem=2)
    assert "esn" in rec.default_models()


def test_window_comparison_rejects_recurrent_models():
    cmp = Comparison(kind="window", n_series=2, input_window=3, n_mem=2)
    with pytest.raises(ValueError, match="esn"):
        cmp.check_models(["qrc", "esn"])


def test_protocol_rejects_overlapping_tuning_and_evaluation_seeds():
    with pytest.raises(ValueError, match="overlap"):
        Protocol(task="henon", tune_data_seeds=(0, 1), eval_data_seeds=(1, 2))
    with pytest.raises(ValueError, match="overlap"):
        Protocol(task="henon", tune_res_seeds=(3,), eval_res_seeds=(3,))


@pytest.mark.parametrize("model", ["qrc", "random_features", "poly2"])
def test_tuning_averages_over_tuning_seeds_and_records_time(model):
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**TINY)
    rec = tune_model(model, cmp, proto, Sim(), log=lambda s: None)
    assert rec["n_trials"] == 2 and "alpha" in rec["search_space"]
    assert len(rec["trial_seconds"]) == 2 and all(t > 0 for t in rec["trial_seconds"])
    n_runs = 2 if model == "poly2" else 2 * 1          # poly2 has no reservoir seed
    assert len(rec["best_val_per_run"]) == n_runs
    assert np.isclose(rec["best_val"], np.mean(list(rec["best_val_per_run"].values())))


def test_frozen_evaluation_uses_only_eval_seeds_and_best_params():
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**TINY)
    rec = tune_model("qrc", cmp, proto, Sim(), log=lambda s: None)
    ev = evaluate_frozen("qrc", rec["best_params"], rec["best_alpha"], cmp, proto, Sim())
    assert [(r["data_seed"], r["res_seed"]) for r in ev] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(r["width"] == cmp.width for r in ev)
    assert all(np.isfinite(r["test_nrmse"]) for r in ev)


@pytest.mark.parametrize("kind,models", [("window", ["qrc", "random_features", "poly2", "linear"]),
                                         ("recurrent", ["qrc", "esn", "linear"])])
def test_run_comparison_same_budget_matched_width_and_stats(tmp_path, kind, models):
    cmp = Comparison(kind=kind, n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**TINY)
    out = run_comparison(cmp, proto, Sim(), models=models, out_dir=tmp_path, log=lambda s: None)
    tunings = [out["models"][m]["tuning"] for m in models]
    assert {t["n_trials"] for t in tunings} == {2}
    assert {t["sampler_seed"] for t in tunings} == {proto.sampler_seed}
    other = "random_features" if kind == "window" else "esn"
    assert out["models"]["qrc"]["eval"][0]["width"] == out["models"][other]["eval"][0]["width"] == cmp.width
    assert set(out["stats"]) == {f"qrc_vs_{m}" for m in models if m != "qrc"}
    assert "p_holm_cluster" in out["stats"][f"qrc_vs_{models[1]}"]
    assert len(out["persistence"]) == 2
    assert out["environment"]["numpy"] and "git_commit" in out["environment"]
    saved = json.loads((tmp_path / f"comparison_{kind}.json").read_text())
    assert saved["protocol"]["n_trials"] == 2


def test_resume_does_not_rerun_finished_trials(tmp_path):
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**TINY)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    a = tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    b = tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    assert a["n_trials"] == b["n_trials"] == 2 and a["best_val"] == b["best_val"]


def test_cli_experiment_arguments_parse():
    a = build_parser().parse_args(["experiment", "--task", "henon", "--task-arg", "n_series=9", "--kind", "window",
                                   "--input-window", "3", "--n-mem", "2", "--trials", "100", "--backend", "cupy",
                                   "--precision", "single", "--out", "results/x"])
    assert (a.kind, a.input_window, a.n_mem, a.trials, a.precision) == ("window", 3, 2, 100, "single")


def test_resume_refuses_a_changed_protocol(tmp_path):
    from dataclasses import replace
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**TINY)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    with pytest.raises(ValueError, match="different"):
        tune_model("linear", cmp, replace(proto, washout=20), Sim(), storage=storage, log=lambda s: None)
    with pytest.raises(ValueError, match="different"):
        tune_model("linear", replace(cmp, input_window=3), proto, Sim(), storage=storage, log=lambda s: None)
    # more trials with an otherwise identical protocol is a valid resume
    rec = tune_model("linear", cmp, replace(proto, n_trials=3), Sim(), storage=storage, log=lambda s: None)
    assert rec["n_trials"] == 3


PER_DIM = {k: v for k, v in TINY.items() if k != "n_trials"}


def test_budget_scales_with_search_dimension():
    """Equal trials across spaces of different size favours the small space, so the floor
    budget is trials_per_dim x the model's own number of tuned knobs (alpha included)."""
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1, reservoir="xxz")
    proto = Protocol(**PER_DIM, trials_per_dim=2, patience=1, max_trials=12)
    qrc = tune_model("qrc", cmp, proto, Sim(), log=lambda s: None)
    p2 = tune_model("poly2", cmp, proto, Sim(), log=lambda s: None)
    assert (qrc["budget"]["n_dims"], qrc["budget"]["floor"]) == (5, 10)     # tau, scale, delta, W, alpha
    assert (p2["budget"]["n_dims"], p2["budget"]["floor"]) == (1, 2)        # alpha only
    for rec in (qrc, p2):
        assert rec["budget"]["floor"] <= rec["n_trials"] <= proto.max_trials
        assert rec["budget"]["rule"] == "per_dim" and rec["budget"]["stopped"] in ("patience", "max_trials")


def test_fixed_trial_budget_is_still_available_for_smoke_runs():
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    rec = tune_model("linear", cmp, Protocol(**TINY), Sim(), log=lambda s: None)
    assert rec["n_trials"] == 2 and rec["budget"]["rule"] == "fixed"


def test_budget_never_exceeds_max_trials():
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**PER_DIM, trials_per_dim=50, patience=50, max_trials=3)
    rec = tune_model("linear", cmp, proto, Sim(), log=lambda s: None)
    assert rec["n_trials"] == 3 and rec["budget"]["stopped"] == "max_trials"


def test_run_level_stats_flagged_when_a_model_is_deterministic():
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    out = run_comparison(cmp, Protocol(**TINY), Sim(), models=["qrc", "random_features", "poly2"],
                         log=lambda s: None)
    assert out["stats"]["qrc_vs_poly2"]["run_level_independent"] is False
    assert out["stats"]["qrc_vs_random_features"]["run_level_independent"] is True


def test_cli_budget_arguments_parse():
    a = build_parser().parse_args(["experiment", "--task", "henon", "--kind", "window",
                                   "--input-window", "2", "--n-mem", "8"])
    assert (a.trials, a.trials_per_dim, a.max_trials, a.patience) == (None, 25, 300, 50)


def test_esn_bias_lets_features_be_even():
    """Without a bias every ESN feature is odd in the input (F(-x) = -F(x)), so the readout
    cannot form squares or products; the bias removes that handicap."""
    from qrc_bench.baselines import esn_features
    X = np.random.default_rng(0).standard_normal((200, 3))
    odd = esn_features(X, 20, 0, bias_scale=0.0)
    assert np.allclose(esn_features(-X, 20, 0, bias_scale=0.0), -odd)
    assert not np.allclose(esn_features(-X, 20, 0, bias_scale=1.0), -esn_features(X, 20, 0, bias_scale=1.0))


def test_classical_spaces_tune_a_bias_over_wide_ranges():
    import optuna
    from qrc_bench.tuning.spaces import MODELS
    cmp = Comparison(kind="recurrent", n_series=2, input_window=2, n_mem=1)
    for model in ("esn", "random_features"):
        trial = optuna.create_study().ask()
        MODELS[model].space(trial, cmp)
        assert trial.distributions["bias_scale"].high >= 10.0
    trial = optuna.create_study().ask()
    MODELS["random_features"].space(trial, cmp)
    assert trial.distributions["scale"].low <= 1e-3        # the old 0.05 bound was where it pinned


def test_resuming_a_converged_study_runs_no_more_trials(tmp_path):
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**PER_DIM, trials_per_dim=2, patience=1, max_trials=8)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    a = tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    b = tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    assert b["n_trials"] == a["n_trials"] and b["budget"]["stopped"] == a["budget"]["stopped"]


def test_series_are_generated_once_per_experiment():
    """Every model tunes and evaluates on the same series; generating a 2000-step Lorenz-96 series
    costs ~9 s, so the series cache must hand back the same array for the same request."""
    from qrc_bench.tuning.study import _series
    proto = Protocol(**TINY)
    assert _series(proto, 0) is _series(proto, 0)
    assert _series(proto, 0) is not _series(proto, 1)
    from dataclasses import replace
    assert _series(replace(proto, n_steps=proto.n_steps + 1), 0).shape[0] == proto.n_steps + 1


def test_pinned_search_edges_were_widened():
    """Every tuned optimum of the finite-shot grid sat on a range edge: v and W at 10, the ESN
    spectral radius at 0.05 and bias at 10, ridge alpha at 1e-6. Widened on both sides together."""
    import optuna
    from qrc_bench import registry
    from qrc_bench.tuning.spaces import ALPHA_RANGE, MODELS
    assert ALPHA_RANGE[0] <= 1e-9
    for res, knob in (("xxz", "W"), ("ising_xx", "v")):
        trial = optuna.create_study().ask()
        registry.get("reservoir", res).suggest(trial)
        assert trial.distributions[knob].high >= 30.0
    trial = optuna.create_study().ask()
    MODELS["esn"].space(trial, Comparison(kind="recurrent", n_series=2, input_window=2, n_mem=1))
    assert trial.distributions["sr"].low <= 0.01 and trial.distributions["bias_scale"].high >= 30.0
