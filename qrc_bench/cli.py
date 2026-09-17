"""Command line: python -m qrc_bench {list, layout, reproduce, experiment, bench}."""
from __future__ import annotations

import argparse
import ast
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


def _task_kwargs(pairs) -> dict:
    out = {}
    for s in pairs or []:
        k, _, v = s.partition("=")
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def _seeds(values):
    return tuple(int(v) for v in values)


def cmd_experiment(args):
    from qrc_bench.tuning import Comparison, Protocol, Sim, run_comparison

    cmp = Comparison(kind=args.kind, n_series=args.n_series or registry.get("task", args.task)(
                         60, 0, **_task_kwargs(args.task_arg)).shape[1],
                     input_window=args.input_window, n_mem=args.n_mem, readout=args.readout, n_taus=args.n_taus,
                     reservoir=args.reservoir, encoding=args.encoding, feedback=not args.no_feedback)
    proto = Protocol(task=args.task, task_kwargs=_task_kwargs(args.task_arg), n_steps=args.n_steps,
                     horizon=args.horizon, tune_data_seeds=_seeds(args.tune_data_seeds),
                     tune_res_seeds=_seeds(args.tune_res_seeds), eval_data_seeds=_seeds(args.eval_data_seeds),
                     eval_res_seeds=_seeds(args.eval_res_seeds), n_trials=args.trials,
                     sampler_seed=args.sampler_seed, n_folds=args.folds, washout=args.washout)
    sim = Sim(method=args.method, backend=args.backend, precision=args.precision)
    run_comparison(cmp, proto, sim, models=args.models, out_dir=args.out, resume=args.resume)


def cmd_bench(args):
    from qrc_bench.bench import benchmark

    layouts = [tuple(int(v) for v in s.split(":")) for s in args.layouts]
    rows = benchmark(layouts=layouts, steps=args.steps, methods=args.methods, backends=args.backends,
                     precisions=args.precisions, memories=args.memories, repeats=args.repeats, log=print)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({"rows": rows, "git_commit": _git_commit()}, indent=2))


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
    p.add_argument("--method", choices=["batched", "branch", "dense"], default="batched")
    p.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    p.add_argument("--out", help="write the result JSON here too")
    p.set_defaults(func=cmd_reproduce)

    p = sub.add_parser("experiment", help="tune every model under one protocol, freeze, evaluate on fresh seeds")
    p.add_argument("--task", required=True)
    p.add_argument("--task-arg", action="append", metavar="KEY=VALUE", help="task parameter, repeatable")
    p.add_argument("--n-series", type=int, help="default: read from the task")
    p.add_argument("--kind", choices=["window", "recurrent"], required=True)
    p.add_argument("--input-window", type=int, required=True, help="steps of input (reset-QRC depth, lags)")
    p.add_argument("--n-mem", type=int, required=True)
    p.add_argument("--readout", choices=READOUTS, default="ZZ")
    p.add_argument("--n-taus", type=int, default=1)
    p.add_argument("--reservoir", default="ising_xx")
    p.add_argument("--encoding", choices=ENCODINGS, default="per_series")
    p.add_argument("--no-feedback", action="store_true", help="recurrent kind: fix k_fb = 0")
    p.add_argument("--models", nargs="+", help="default: all models of the comparison kind")
    p.add_argument("--n-steps", type=int, default=1200)
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--tune-data-seeds", nargs="+", default=[100, 101, 102])
    p.add_argument("--tune-res-seeds", nargs="+", default=[100, 101, 102])
    p.add_argument("--eval-data-seeds", nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--eval-res-seeds", nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--trials", type=int, default=100, help="same budget for every model")
    p.add_argument("--sampler-seed", type=int, default=0)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--washout", type=int, default=50)
    p.add_argument("--method", choices=["batched", "branch", "dense"], default="batched")
    p.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    p.add_argument("--precision", choices=["auto", "double", "single"], default="auto",
                   help="auto: single on cupy, double on numpy")
    p.add_argument("--out", help="directory for the comparison JSON (and Optuna studies with --resume)")
    p.add_argument("--resume", action="store_true", help="keep studies in SQLite under --out and resume them")
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("bench", help="time the reservoir simulators per step")
    p.add_argument("--layouts", nargs="+", default=["1:4", "5:3", "5:6"], metavar="N_IN:N_MEM")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--methods", nargs="+", choices=["batched", "branch", "dense"], default=["branch", "batched"])
    p.add_argument("--backends", nargs="+", choices=["numpy", "cupy"], default=["numpy", "cupy"])
    p.add_argument("--precisions", nargs="+", choices=["double", "single"], default=["double", "single"])
    p.add_argument("--memories", nargs="+", choices=["reset", "recurrent"], default=["reset", "recurrent"])
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--out", help="write rows as JSON")
    p.set_defaults(func=cmd_bench)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
