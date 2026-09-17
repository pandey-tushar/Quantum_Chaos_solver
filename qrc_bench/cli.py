"""Command line: python -m qrc_bench {list, layout, reproduce, tune}."""
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


def cmd_tune(args):
    from qrc_bench.tuning import Context, tune_all

    task = registry.get("task", args.task)
    task_kwargs = _task_kwargs(args.task_arg)
    outdir = Path(args.out) if args.out else None
    for ds in args.data_seeds:
        X = task(args.n_steps, ds, **task_kwargs)
        for rs in args.res_seeds:
            print(f"=== tune {args.task} ds{ds} rs{rs}: {X.shape[1]} series, {args.trials} trials per model")
            ctx = Context(X, rs, horizon=args.horizon, window=args.window, lag_window=args.lag_window,
                          reservoir=args.reservoir, n_mem_range=(args.n_mem_min, args.n_mem_max),
                          feedback=not args.no_feedback, esn_units=args.esn_units, method=args.method,
                          backend=args.backend, q_max=args.q_max)
            storage = f"sqlite:///{outdir / f'optuna_{args.task}_ds{ds}_rs{rs}.db'}" if outdir and args.resume else None
            t0 = time.time()
            rec = tune_all(args.models, ctx, n_trials=args.trials, sampler_seed=args.sampler_seed,
                           enqueue_defaults=args.enqueue_defaults, storage=storage)
            payload = {"task": args.task, "task_kwargs": task_kwargs, "n_steps": args.n_steps,
                       "data_seed": ds, "res_seed": rs, **rec,
                       "git_commit": _git_commit(), "wall_time_s": round(time.time() - t0, 2)}
            if outdir:
                outdir.mkdir(parents=True, exist_ok=True)
                (outdir / f"tune_{args.task}_ds{ds}_rs{rs}.json").write_text(json.dumps(payload, indent=2))


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

    p = sub.add_parser("tune", help="Optuna-tune every model with one shared protocol")
    p.add_argument("--task", required=True)
    p.add_argument("--task-arg", action="append", metavar="KEY=VALUE", help="task parameter, repeatable")
    p.add_argument("--n-steps", type=int, default=1200)
    p.add_argument("--data-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--res-seeds", type=int, nargs="+", default=[0])
    p.add_argument("--horizon", type=int, default=1)
    p.add_argument("--models", nargs="+", choices=["qrc", "esn", "poly2", "linear"],
                   default=["qrc", "esn", "poly2", "linear"])
    p.add_argument("--trials", type=int, default=100, help="same budget for every model")
    p.add_argument("--sampler-seed", type=int, default=0)
    p.add_argument("--enqueue-defaults", action="store_true", help="first trial of every model = its defaults")
    p.add_argument("--reservoir", default="ising_xx")
    p.add_argument("--n-mem-min", type=int, default=1)
    p.add_argument("--n-mem-max", type=int, default=6)
    p.add_argument("--no-feedback", action="store_true", help="fix k_fb = 0 (open loop)")
    p.add_argument("--esn-units", type=int, help="fix ESN size instead of tuning it")
    p.add_argument("--window", type=int, default=1, help="stacked steps of reservoir features (QRC, ESN)")
    p.add_argument("--lag-window", type=int, default=1, help="input window of Poly2 and linear")
    p.add_argument("--method", choices=["branch", "dense"], default="branch")
    p.add_argument("--backend", choices=["numpy", "cupy"], default="numpy")
    p.add_argument("--q-max", type=int, default=Q_MAX)
    p.add_argument("--out", help="directory for per-seed-pair JSONs")
    p.add_argument("--resume", action="store_true", help="keep studies in SQLite under --out and resume them")
    p.set_defaults(func=cmd_tune)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
