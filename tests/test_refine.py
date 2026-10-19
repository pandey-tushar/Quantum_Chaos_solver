"""Optional SPSA refinement after the TPE stage: every model gets the same local search from its
best TPE trial, run as ordinary Optuna trials so records, curves and resume need no new paths."""
import json
from dataclasses import replace

import numpy as np
import optuna
import pytest
from optuna.distributions import FloatDistribution

from qrc_bench.cli import build_parser
from qrc_bench.tuning import Comparison, Protocol, Sim, run_comparison, tune_model
from qrc_bench.tuning import study as study_mod
from qrc_bench.tuning.study import BUDGET_KEYS, _from_cube, _spsa, _to_cube
from tests.test_tuning import PER_DIM, TINY

TARGET = np.array([0.3, 0.7, 0.5])


def _quad(x):
    assert np.all((x >= 0) & (x <= 1))
    return float(np.sum((x - TARGET) ** 2))


def test_spsa_finds_a_quadratic_minimum_inside_the_cube():
    x, v, points = _spsa(_quad, np.full(3, 0.5), budget=96, seed=0)
    assert np.all(np.abs(x - TARGET) < 0.05) and np.isclose(v, _quad(x))
    assert len(points) == 96 and all(np.all((p >= 0) & (p <= 1)) for p in points)


def test_spsa_is_deterministic_per_seed_and_uses_the_whole_even_budget():
    a = _spsa(_quad, np.full(3, 0.5), budget=7, seed=1)[2]
    b = _spsa(_quad, np.full(3, 0.5), budget=7, seed=1)[2]
    c = _spsa(_quad, np.full(3, 0.5), budget=7, seed=2)[2]
    assert len(a) == 6 and np.array_equal(a, b) and not np.array_equal(a, c)


def test_spsa_treats_a_failed_evaluation_as_the_worst_seen():
    x, v, points = _spsa(lambda u: float("inf") if u[0] > 0.5 else _quad(u), np.full(3, 0.5), budget=20, seed=0)
    assert np.isfinite(v) and x[0] <= 0.5 and len(points) == 20


def test_unit_cube_mapping():
    log = FloatDistribution(1e-6, 1e2, log=True)
    assert np.isclose(_from_cube(0.0, log), 1e-6) and np.isclose(_from_cube(1.0, log), 1e2)
    assert np.isclose(_from_cube(0.5, log), 1e-2)
    lin = FloatDistribution(-2.0, 2.0)
    assert np.isclose(_from_cube(0.25, lin), -1.0)
    for d in (log, lin):
        for u in (0.0, 0.3, 0.5, 1.0):
            assert abs(_to_cube(_from_cube(u, d), d) - u) < 1e-12


def test_protocol_rejects_an_unknown_refine():
    with pytest.raises(ValueError, match="refine"):
        Protocol(task="henon", refine="bogus")


@pytest.mark.parametrize("model", ["linear", "qrc"])
def test_no_refine_leaves_tuning_unchanged(model):
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    rec = tune_model(model, cmp, Protocol(**TINY, refine="none"), Sim(), log=lambda s: None)
    assert rec["n_trials"] == 2
    assert (rec["budget"]["refine_evals"], rec["budget"]["refine_gain"]) == (0, 0.0)
    assert rec["sampler"] == "TPE"


def test_spsa_stage_adds_its_trials_after_tpe():
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    base = Protocol(**PER_DIM, trials_per_dim=2, patience=1, max_trials=12)
    tpe = tune_model("linear", cmp, base, Sim(), log=lambda s: None)
    rec = tune_model("linear", cmp, replace(base, refine="spsa", refine_frac=1.0), Sim(), log=lambda s: None)
    assert rec["budget"]["floor"] == 2 and rec["budget"]["refine_evals"] == 2 and rec["sampler"] == "TPE+SPSA"
    assert rec["n_trials"] == tpe["n_trials"] + 2 and rec["trial_values"][:-2] == tpe["trial_values"]
    assert np.all(np.diff(rec["best_so_far"]) <= 0)
    assert rec["best_val"] <= min(rec["trial_values"][:-2])
    assert rec["budget"]["refine_gain"] >= 0
    assert np.isclose(rec["budget"]["refine_gain"], min(rec["trial_values"][:-2]) - rec["best_val"])


def test_resume_refuses_to_add_refine_to_a_stored_study(tmp_path):
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    tune_model("linear", cmp, Protocol(**TINY), Sim(), storage=storage, log=lambda s: None)
    with pytest.raises(ValueError, match="different"):
        tune_model("linear", cmp, Protocol(**TINY, refine="spsa"), Sim(), storage=storage, log=lambda s: None)


@pytest.mark.parametrize("budget", [dict(TINY), dict(PER_DIM, trials_per_dim=2, patience=1, max_trials=12)])
def test_resuming_a_refined_study_runs_no_more_trials(tmp_path, budget):
    """The refine trials do not count as TPE trials on resume, and a finished stage is replayed, not rerun."""
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**budget, refine="spsa", refine_frac=1.0)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    a = tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    b = tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    assert b["n_trials"] == a["n_trials"] and b["best_val"] == a["best_val"] and b["budget"] == a["budget"]


def test_resume_accepts_a_study_stored_before_refine_existed(tmp_path):
    """A study fingerprinted before the refine fields existed resumes with refine="none"."""
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**TINY)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    old = {k: v for k, v in proto.as_dict().items() if k not in BUDGET_KEYS + ("refine", "refine_frac")}
    optuna.create_study(storage=storage, study_name="linear").set_user_attr(
        "fingerprint", json.dumps({"comparison": cmp.as_dict(), "sim": Sim().as_dict(), "protocol": old},
                                  sort_keys=True, default=list))
    assert tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)["n_trials"] == 2


@pytest.mark.parametrize("leftover", ["fail", "waiting"])
def test_resume_after_an_interrupted_refine_stage(tmp_path, monkeypatch, leftover):
    """Interrupted at the second SPSA evaluation, inside the objective (a FAIL trial is left) or
    between enqueue and optimize (a WAITING trial is left): the resumed run matches an uninterrupted one."""
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**TINY, refine="spsa", refine_frac=1.0)
    ref = tune_model("linear", cmp, proto, Sim(), log=lambda s: None)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    with monkeypatch.context() as m:
        if leftover == "fail":
            real, calls = study_mod.val_nrmse_folds, []

            def flaky(*a, **k):                    # 2 calls per linear trial: 2 TPE trials, then SPSA
                calls.append(1)
                if len(calls) > 6:
                    raise RuntimeError("interrupted")
                return real(*a, **k)
            m.setattr(study_mod, "val_nrmse_folds", flaky)
        else:
            real, calls = optuna.Study.optimize, []

            def flaky(self, *a, n_trials=None, **k):
                if n_trials == 1:
                    calls.append(1)
                    if len(calls) == 2:
                        raise RuntimeError("interrupted")
                return real(self, *a, n_trials=n_trials, **k)
            m.setattr(optuna.Study, "optimize", flaky)
        with pytest.raises(RuntimeError, match="interrupted"):
            tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    rec = tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    assert rec["n_trials"] == ref["n_trials"] and rec["best_val"] == ref["best_val"]
    assert rec["budget"]["refine_evals"] == ref["budget"]["refine_evals"]
    assert rec["trial_values"] == ref["trial_values"]
    trials = optuna.load_study(study_name="linear", storage=storage).trials
    assert not [t for t in trials if t.state == optuna.trial.TrialState.WAITING]


def test_refine_gain_is_zero_when_every_trial_failed(monkeypatch):
    monkeypatch.setattr(study_mod, "val_nrmse_folds", lambda *a, **k: float("inf"))
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    rec = tune_model("linear", cmp, Protocol(**TINY, refine="spsa", refine_frac=1.0), Sim(), log=lambda s: None)
    assert rec["budget"]["refine_gain"] == 0.0


def test_cli_refine_arguments_parse():
    base = ["experiment", "--task", "henon", "--kind", "window", "--input-window", "2", "--n-mem", "8"]
    a = build_parser().parse_args(base + ["--refine", "spsa", "--refine-frac", "0.3"])
    assert (a.refine, a.refine_frac) == ("spsa", 0.3)
    a = build_parser().parse_args(base)
    assert (a.refine, a.refine_frac) == ("none", 0.5)


def test_run_comparison_refines_every_model(tmp_path):
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    models = ["qrc", "random_features", "linear"]
    out = run_comparison(cmp, Protocol(**TINY, refine="spsa", refine_frac=1.0), Sim(), models=models,
                         out_dir=tmp_path, log=lambda s: None)
    for m in models:
        assert out["models"][m]["tuning"]["budget"]["refine_evals"] == 2
        assert out["models"][m]["tuning"]["n_trials"] == 4
    saved = json.loads((tmp_path / "comparison_window.json").read_text())
    assert saved["protocol"]["refine"] == "spsa"
