"""Command line: python -m qrc_bench {list, layout, reproduce}."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from qrc_bench import registry
from qrc_bench.layout import ENCODINGS, MEM_RULES, Q_MAX, derive
from qrc_bench.readouts import READOUTS


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, cwd=Path(__file__).parent).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _write(out: str | None, payload: dict):
    text = json.dumps(payload, indent=2)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text)


def cmd_list(_args):
    for kind in registry.KINDS:
        print(f"{kind:10s} {', '.join(registry.names(kind))}")


def cmd_layout(args):
    lay = derive(args.n_series, encoding=args.encoding, mem_rule=args.mem_rule, n_mem=args.n_mem,
                 mem_ratio=args.mem_ratio, readout=args.readout, n_taus=args.n_taus,
                 qrc_window=args.qrc_window, poly2_window=args.poly2_window, q_max=args.q_max)
    print(json.dumps(lay.as_dict(), indent=2))


def cmd_reproduce(args):
    from qrc_bench.experiments.cartography import case_i_pair, case_ii_pair

    t0 = time.time()
    if args.case == "I":
        result = case_i_pair(args.coupling, args.data_seed, args.res_seed, method=args.method,
                             backend=args.backend)
    else:
        result = case_ii_pair(args.data_seed, args.res_seed, p_switch=args.p_switch,
                              method=args.method, backend=args.backend)
    config = {k: v for k, v in vars(args).items() if k != "func"}
    _write(args.out, {"config": config, "result": result,
                      "git_commit": _git_commit(), "wall_time_s": round(time.time() - t0, 2)})


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="qrc_bench", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="registered tasks, reservoirs and baselines")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("layout", help="derive the qubit layout and feature widths")
    p.add_argument("--n-series", type=int, required=True)
    p.add_argument("--encoding", choices=ENCODINGS, default="per_series")
    p.add_argument("--mem-rule", choices=MEM_RULES, default="fixed")
    p.add_argument("--n-mem", type=int)
    p.add_argument("--mem-ratio", type=float)
    p.add_argument("--readout", choices=READOUTS, default="ZZ")
    p.add_argument("--n-taus", type=int, default=1)
    p.add_argument("--qrc-window", type=int, default=1)
    p.add_argument("--poly2-window", type=int, default=1)
    p.add_argument("--q-max", type=int, default=Q_MAX)
    p.set_defaults(func=cmd_layout)

    p = sub.add_parser("reproduce", help="one seed pair of a cartography-paper protocol")
    p.add_argument("--case", choices=["I", "II"], required=True)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--res-seed", type=int, default=0)
    p.add_argument("--coupling", type=float, default=0.1, help="Case I")
    p.add_argument("--p-switch", type=float, default=0.05, help="Case II")
    p.add_argument("--method", choices=["branch", "dense"], default="branch")
    p.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    p.add_argument("--out", help="write the result JSON here too")
    p.set_defaults(func=cmd_reproduce)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
