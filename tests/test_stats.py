import numpy as np
import pytest

from qrc_bench.stats import holm, paired_comparison


def test_clear_difference_is_detected_at_both_levels():
    rng = np.random.default_rng(0)
    ds = np.repeat(np.arange(5), 5)
    base = rng.normal(0.8, 0.05, 25) + 0.05 * ds
    a, b = base - 0.05 + rng.normal(0, 0.005, 25), base
    r = paired_comparison(a, b, ds, n_boot=2000, seed=0)
    assert r["n_runs"] == 25 and r["n_clusters"] == 5
    assert np.isclose(r["mean_diff"], np.mean(a - b))
    assert r["wins_a"] == 25 and r["wins_b"] == 0
    assert r["ci95_run"][1] < 0 and r["ci95_cluster"][1] < 0
    assert r["p_t_run"] < 1e-6 and r["p_t_cluster"] < 0.01


def test_cluster_level_is_honest_about_pseudoreplication():
    # a big run-level signal carried by one data seed only
    ds = np.repeat(np.arange(5), 5)
    b = np.full(25, 0.8)
    a = b.copy()
    a[ds == 0] -= 0.3
    a += np.random.default_rng(1).normal(0, 1e-3, 25)
    r = paired_comparison(a, b, ds, n_boot=2000, seed=0)
    assert r["p_t_cluster"] > 0.05


def test_ties_counted():
    ds = np.array([0, 0, 1, 1])
    r = paired_comparison(np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 2.5, 2.0, 4.0]), ds, tie_tol=1e-9)
    assert (r["wins_a"], r["wins_b"], r["ties"]) == (1, 1, 2)


def test_holm_adjustment():
    adj = holm([0.01, 0.04, 0.03])
    assert np.allclose(adj, [0.03, 0.06, 0.06])
    with pytest.raises(ValueError):
        holm([])
