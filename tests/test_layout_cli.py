import pytest

from qrc_bench import registry
from qrc_bench.cli import main
from qrc_bench.layout import derive


def test_case_i_layout():
    lay = derive(5, n_mem=3, n_taus=2, poly2_window=3)
    assert (lay.q, lay.qrc_features, lay.poly2_features) == (8, 72, 135)


def test_case_i_q11_layout():
    lay = derive(5, n_mem=6, n_taus=2, poly2_window=3)
    assert (lay.q, lay.qrc_features) == (11, 132)


def test_case_ii_layout():
    lay = derive(1, n_mem=4, qrc_window=5, poly2_window=5)
    assert (lay.q, lay.qrc_features, lay.poly2_features) == (5, 75, 20)


def test_match_features_picks_smallest_memory():
    lay = derive(5, mem_rule="match-features", n_taus=2, poly2_window=2)   # Poly2 width 65
    assert (lay.n_mem, lay.qrc_features, lay.poly2_features) == (3, 72, 65)


def test_match_features_unreachable_within_cap():
    # Case I Poly2 (135) is out of reach: q = 11 gives 132
    with pytest.raises(ValueError, match="Poly2 width 135"):
        derive(5, mem_rule="match-features", n_taus=2, poly2_window=3)


def test_ratio_rule():
    assert derive(4, mem_rule="ratio", mem_ratio=0.5).n_mem == 2


def test_q_cap():
    with pytest.raises(ValueError, match="q_max"):
        derive(9, n_mem=3)


def test_registry_names():
    assert registry.names("reservoir") == ["ising_xx", "xxz", "xxz_uniform"]
    assert {"henon", "switching", "drift"} <= set(registry.names("task"))
    assert {"esn", "poly2", "linear"} <= set(registry.names("baseline"))


def test_cli_layout(capsys):
    main(["layout", "--n-series", "5", "--n-mem", "3", "--n-taus", "2", "--poly2-window", "3"])
    assert '"qrc_features": 72' in capsys.readouterr().out
