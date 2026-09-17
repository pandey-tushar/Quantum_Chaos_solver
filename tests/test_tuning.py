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
