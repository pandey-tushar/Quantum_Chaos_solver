"""Finite-shot measurement: every readout comes from M shots per step, so the features and the
feedback signal both read the same empirical distribution P_hat = counts / M."""
import json

import numpy as np
import pytest

from qrc_bench.qrc import qrc_features
from qrc_bench.reservoirs.ising_xx import IsingXX
from qrc_bench.shots import sample_probs

try:
    import cupy  # noqa: F401
    HAS_CUPY = cupy.cuda.runtime.getDeviceCount() > 0
except Exception:
    HAS_CUPY = False


def _P(rows, k, seed=0):
    P = np.random.default_rng(seed).random((rows, k)) + 0.05
    return P / P.sum(axis=1, keepdims=True)


# --- sample_probs ------------------------------------------------------------------------------

def test_rows_are_normalised_and_counts_are_integers():
    P = _P(9, 8)
    M = 64
    out = sample_probs(P, M, np.random.default_rng(0), np)
    assert out.shape == P.shape and out.dtype == P.dtype
    assert np.max(np.abs(out.sum(axis=1) - 1.0)) < 1e-12
    counts = out * M
    assert np.max(np.abs(counts - np.round(counts))) < 1e-9


def test_one_shot_gives_one_hot_rows():
    out = sample_probs(_P(20, 4, seed=3), 1, np.random.default_rng(1), np)
    assert np.array_equal(np.sort(out, axis=1)[:, :-1], np.zeros((20, 3)))
    assert np.allclose(out.max(axis=1), 1.0)


def test_sampling_is_seed_deterministic():
    P = _P(6, 16, seed=5)
    a = sample_probs(P, 32, np.random.default_rng(7), np)
    b = sample_probs(P, 32, np.random.default_rng(7), np)
    c = sample_probs(P, 32, np.random.default_rng(8), np)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_a_deterministic_row_is_returned_exactly():
    P = _P(3, 8, seed=2)
    P[1] = 0.0
    P[1, 5] = 1.0
    for M in (1, 7, 1000):
        out = sample_probs(P, M, np.random.default_rng(0), np)
        assert out[1, 5] == 1.0 and out[1].sum() == 1.0


def test_error_falls_as_one_over_sqrt_shots():
    P = _P(32, 16, seed=11)
    rms = {M: np.sqrt(np.mean((sample_probs(P, M, np.random.default_rng(4), np) - P) ** 2))
           for M in (100, 10_000)}
    assert 7.0 <= rms[100] / rms[10_000] <= 14.0


def test_per_outcome_variance_matches_a_multinomial():
    p = np.array([0.5, 0.25, 0.15, 0.10])
    M, draws = 50, 2000
    out = sample_probs(np.broadcast_to(p, (draws, 4)).copy(), M, np.random.default_rng(0), np)
    var = out.var(axis=0)
    assert np.allclose(out.mean(axis=0), p, atol=0.02)
    assert np.all(np.abs(var - p * (1 - p) / M) < 0.25 * p * (1 - p) / M)


@pytest.mark.parametrize("bad", [0, -5, 2.5, "64"])
def test_shots_must_be_a_positive_int(bad):
    with pytest.raises(ValueError, match="shots"):
        sample_probs(_P(2, 4), bad, np.random.default_rng(0), np)


def test_rows_are_sub_batched_without_changing_the_answer():
    """rows x M is capped per launch; a budget that forces several sub-batches must still
    produce valid rows for every row of the batch."""
    P = _P(40, 8, seed=6)
    out = sample_probs(P, 4096, np.random.default_rng(0), np, max_uniforms=8192)
    assert np.max(np.abs(out.sum(axis=1) - 1.0)) < 1e-12
    assert np.all(out >= 0.0)


@pytest.mark.skipif(not HAS_CUPY, reason="no CUDA device")
def test_cupy_sampling_is_valid_and_scales_the_same_way():
    P = _P(32, 16, seed=11).astype(np.float32)
    Pd = cupy.asarray(P)
    rms = {}
    for M in (100, 10_000):
        out = sample_probs(Pd, M, cupy.random.default_rng(4), cupy)
        assert out.shape == Pd.shape and out.dtype == Pd.dtype
        host = cupy.asnumpy(out)
        assert np.max(np.abs(host.sum(axis=1) - 1.0)) < 1e-5
        assert np.max(np.abs(host * M - np.round(host * M))) < 1e-2
        rms[M] = np.sqrt(np.mean((host - P) ** 2))
    assert 7.0 <= rms[100] / rms[10_000] <= 14.0


# --- qrc_features ------------------------------------------------------------------------------

def _angles(T, n_in, seed=0):
    return np.random.default_rng(seed).uniform(-np.pi / 2, np.pi / 2, (T, n_in))


@pytest.mark.parametrize("memory", ["reset", "recurrent"])
def test_exact_features_are_unchanged_by_the_shots_argument(memory):
    res = IsingXX(4, seed=1)
    ang = _angles(10, 2, seed=1)
    kw = dict(taus=(1.0,), readout="ZZ", mem_depth=3, memory=memory)
    base = qrc_features(ang, 2, 2, res, **kw)
    assert np.array_equal(qrc_features(ang, 2, 2, res, shots=None, **kw), base)
    if memory == "reset":                       # the reference method is exact and reset-only
        ref = qrc_features(ang, 2, 2, res, method="branch", **kw)
        assert np.max(np.abs(base - ref)) < 1e-11


@pytest.mark.parametrize("memory", ["reset", "recurrent"])
def test_shot_noise_falls_as_one_over_sqrt_shots(memory):
    res = IsingXX(4, seed=1)
    ang = _angles(12, 2, seed=2)
    kw = dict(taus=(1.0,), readout="ZZ", mem_depth=3, memory=memory)
    exact = qrc_features(ang, 2, 2, res, **kw)
    rms = {M: np.sqrt(np.mean((qrc_features(ang, 2, 2, res, shots=M, shot_seed=3, **kw) - exact) ** 2))
           for M in (100, 10_000)}
    assert rms[100] / 30.0 <= rms[10_000] <= rms[100] / (10.0 / 3.0)


def test_shot_noise_is_seeded_by_shot_seed():
    res = IsingXX(4, seed=1)
    ang = _angles(8, 2, seed=4)
    kw = dict(taus=(1.0,), readout="ZZ", mem_depth=2, shots=256)
    a = qrc_features(ang, 2, 2, res, shot_seed=0, **kw)
    assert np.array_equal(qrc_features(ang, 2, 2, res, shot_seed=0, **kw), a)
    assert not np.array_equal(qrc_features(ang, 2, 2, res, shot_seed=1, **kw), a)


@pytest.mark.parametrize("method", ["branch", "dense"])
def test_reference_methods_refuse_shots(method):
    with pytest.raises(ValueError, match="shots require"):
        qrc_features(_angles(4, 1), 1, 1, IsingXX(2, 0), method=method, shots=64)


class _FirstCallOneHot:
    """Stands in for the sampler: the first batch it sees becomes |0...0>, later ones pass through."""

    def __init__(self):
        self.calls = 0

    def __call__(self, P, shots, rng, xp):
        self.calls += 1
        if self.calls > 1:
            return P
        out = xp.zeros_like(P)
        out[:, 0] = 1.0
        return out


@pytest.mark.parametrize("memory,mem_depth", [("recurrent", 1), ("reset", 2)])
def test_feedback_consumes_the_sampled_distribution(monkeypatch, memory, mem_depth):
    """With step 0 sampled to |0...0>, mean <Z> of step 0 is exactly 1, so step 1's R_Y drive
    must be shifted by k_fb * tanh(1) -- and step 0's own features must be those of |0...0>."""
    k_fb = 1.5
    res = IsingXX(4, seed=2)
    ang = _angles(2, 2, seed=8) / 2.0
    kw = dict(taus=(1.0,), readout="ZZ", mem_depth=mem_depth, memory=memory)
    fake = _FirstCallOneHot()
    monkeypatch.setattr("qrc_bench.qrc.sample_probs", fake)
    out = qrc_features(ang, 2, 2, res, k_fb=k_fb, shots=64, **kw)
    assert fake.calls == 2
    shifted = ang.copy()
    shifted[1] += k_fb * np.tanh(1.0)
    ref = qrc_features(shifted, 2, 2, res, k_fb=0.0, **kw)
    assert np.max(np.abs(out[1] - ref[1])) < 1e-10
    assert np.max(np.abs(out[0] - 1.0)) < 1e-12         # every Z and ZZ of |0...0> is +1


# --- tuning and CLI ----------------------------------------------------------------------------

TINY = dict(task="henon", task_kwargs={"n_series": 2}, n_steps=160, n_trials=2, n_folds=2, washout=10,
            tune_data_seeds=(100, 101), tune_res_seeds=(100,), eval_data_seeds=(0, 1), eval_res_seeds=(0, 1))


def test_comparison_records_shots():
    from qrc_bench.tuning import Comparison
    assert Comparison(kind="window", n_series=2, input_window=2, n_mem=1).as_dict()["shots"] is None
    assert Comparison(kind="window", n_series=2, input_window=2, n_mem=1, shots=64).as_dict()["shots"] == 64


def test_shots_are_part_of_the_resume_fingerprint(tmp_path):
    from dataclasses import replace

    from qrc_bench.tuning import Comparison, Protocol, Sim, tune_model
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1)
    proto = Protocol(**TINY)
    storage = f"sqlite:///{tmp_path / 'studies.db'}"
    tune_model("linear", cmp, proto, Sim(), storage=storage, log=lambda s: None)
    with pytest.raises(ValueError, match="different"):
        tune_model("linear", replace(cmp, shots=64), proto, Sim(), storage=storage, log=lambda s: None)


def test_build_trial_with_shots_gives_finite_features_of_the_right_width():
    from qrc_bench import registry
    from qrc_bench.tuning import Comparison, Sim
    from qrc_bench.tuning.spaces import build_trial

    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1, shots=64)
    X = registry.get("task", "henon")(120, 0, n_series=2)
    params = {"tau": 1.0, "scale": 0.5, "reservoir": {}, "k_fb": 0.0}
    built = build_trial("qrc", params, [X], [0], cmp, Sim())
    single = built[(0, 0)]
    assert single.shape == (120, cmp.width) and np.all(np.isfinite(single))
    multi = build_trial("qrc", params, [X], [0, 1], cmp, Sim())          # batched multi-reservoir
    assert multi[(0, 0)].shape == (120, cmp.width) and np.all(np.isfinite(multi[(1, 0)]))


def test_run_comparison_stores_shots(tmp_path):
    from qrc_bench.tuning import Comparison, Protocol, Sim, run_comparison
    cmp = Comparison(kind="window", n_series=2, input_window=2, n_mem=1, shots=64)
    out = run_comparison(cmp, Protocol(**TINY), Sim(), models=["qrc", "linear"], out_dir=tmp_path,
                         log=lambda s: None)
    saved = json.loads((tmp_path / "comparison_window.json").read_text())
    assert saved["comparison"]["shots"] == 64
    assert all(np.isfinite(r["test_nrmse"]) for r in out["models"]["qrc"]["eval"])


def test_cli_shots_default_is_1024_per_qubit():
    from qrc_bench.cli import build_parser, resolve_shots
    from qrc_bench.tuning import Comparison
    a = build_parser().parse_args(["experiment", "--task", "henon", "--kind", "window",
                                   "--input-window", "2", "--n-mem", "8"])
    assert a.shots == "auto"
    cmp = Comparison(kind="window", n_series=4, input_window=a.input_window, n_mem=a.n_mem)
    assert cmp.q == 12
    assert resolve_shots(a.shots, cmp.q, log=lambda s: None) == 1024 * 12


def test_cli_shots_zero_means_exact_and_warns():
    from qrc_bench.cli import build_parser, resolve_shots
    a = build_parser().parse_args(["experiment", "--task", "henon", "--kind", "window",
                                   "--input-window", "2", "--n-mem", "8", "--shots", "0"])
    lines = []
    assert resolve_shots(a.shots, 12, log=lines.append) is None
    assert len(lines) == 1 and "exact" in lines[0].lower()


def test_cli_shots_explicit_budget():
    from qrc_bench.cli import build_parser, resolve_shots
    a = build_parser().parse_args(["experiment", "--task", "henon", "--kind", "window",
                                   "--input-window", "2", "--n-mem", "8", "--shots", "500"])
    assert resolve_shots(a.shots, 12, log=lambda s: None) == 500
