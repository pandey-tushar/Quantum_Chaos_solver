"""The rebuilt protocols must reproduce the cartography paper's per-seed-pair JSONs."""
import json
from pathlib import Path

import pytest

from qrc_bench.experiments.cartography import case_i_pair, case_ii_pair

DATA = Path(__file__).parent / "data"
TOL = 1e-9


@pytest.mark.slow
def test_case_ii_ds0_rs0_matches_paper():
    ref = json.loads((DATA / "caseII_phase2_ps0.05_h1_10x10.json").read_text())["nrmse"]
    out = case_ii_pair(0, 0)
    for k in ("FB_QRC", "OpenLoop_QRC", "ESN"):
        assert abs(out[k] - ref[k][0]) < TOL, (k, out[k], ref[k][0])
    for k in ("Poly2", "Linear"):          # per data seed in the paper JSON
        assert abs(out[k] - ref[k][0]) < TOL, (k, out[k], ref[k][0])


@pytest.mark.slow
def test_case_i_c01_ds0_rs0_matches_paper():
    ref = json.loads((DATA / "caseI_phase3_c0.1_n0.1_ds0_rs0.json").read_text())["results_per_horizon"]["1"]
    out = case_i_pair(0.1, 0, 0)
    for k in ("Poly2", "QRC", "Poly2+QRC"):
        assert abs(out[k] - ref[k]["nrmse"]) < TOL, (k, out[k], ref[k]["nrmse"])
    assert out["qrc_cfg"] == ref["qrc_cfg"]
    assert out["D"] == {"QRC": 72, "Poly2": 135}
