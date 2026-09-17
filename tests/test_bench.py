from qrc_bench.bench import benchmark
from qrc_bench.cli import build_parser


def test_benchmark_records_time_per_step_and_agreement():
    rows = benchmark(layouts=[(1, 2)], steps=6, methods=["branch", "batched"], backends=["numpy"],
                     precisions=["double"], memories=["reset"], repeats=1)
    assert {r["method"] for r in rows} == {"branch", "batched"}
    for r in rows:
        assert r["q"] == 3 and r["steps"] == 6 and r["ms_per_step"] > 0
        assert r["max_abs_diff_vs_reference"] < 1e-11


def test_cli_bench_parses():
    a = build_parser().parse_args(["bench", "--layouts", "1:4", "5:6", "--steps", "100"])
    assert a.layouts == ["1:4", "5:6"] and a.steps == 100
