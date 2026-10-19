"""Free-running (autonomous) forecasts of frozen models and their valid prediction time.

A model's readout is fit exactly as for the test score (rows [washout, test_start), tuned alpha,
standardised), then, from a start t0 in the test slice, it predicts x_{t0+1} from the features of
the true history and keeps going on its own: each prediction is fed back as the next input. The
error after h steps, divided by each series' test-slice std and taken as the RMS over the series,
first exceeds ``threshold`` after the valid prediction time (VPT), counted in steps that were still
valid and reported in Lyapunov times of the task's own integrator.

How each model is advanced one input at a time, over a batch of starts:
  window models  the features at a step depend only on the last ``input_window`` inputs, so the
                 stepper keeps those windows and calls the model's own batch builder on them;
                 the window QRC draws a fresh shot realisation every step
  recurrent QRC  batched.RecurrentStepper, warmed on the true inputs once, snapshotted at every
                 start and restored as one batch; it is the code ``simulate`` itself runs
  ESN            esn_step, the update esn_features applies, from the state at each start
"""
from __future__ import annotations

import inspect
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np

from qrc_bench import registry
from qrc_bench.baselines import esn_step, esn_weights
from qrc_bench.encoders import encode
from qrc_bench.evaluate import fit_readout, test_start
from qrc_bench.qrc import RecurrentQRC, qrc_features
from qrc_bench.tasks.lorenz96 import lorenz96, rk4_step
from qrc_bench.tuning.spaces import MODELS, Comparison, Protocol, Sim
from qrc_bench.tuning.study import _series, compare_to_qrc, environment

D0 = 1e-8               # Benettin: separation the partner trajectory is rescaled to every step
BURN_TIME = 100.0       # time units integrated before measuring, to start on the attractor
T_TOTAL = 2000.0        # time units over which the Lyapunov exponent is averaged
LT_MARKS = (1, 2, 5)    # Lyapunov times at which the summary reports the NRMSE curve


def lyapunov_exponent(n_series: int, forcing: float = 8.0, dt: float = 0.01, t_total: float = T_TOTAL,
                      seed: int = 0) -> float:
    """Largest Lyapunov exponent of Lorenz-96 (Benettin): a reference and a partner D0 away are
    advanced together by the lorenz96 task's RK4 step, the separation is rescaled back to D0 after
    every step, and lambda is the mean log growth per unit time over ``t_total``."""
    rng = np.random.default_rng(seed)
    x = forcing + 0.01 * rng.standard_normal(n_series)
    for _ in range(int(round(BURN_TIME / dt))):
        x = rk4_step(x, forcing, dt)
    u = rng.standard_normal(n_series)
    xy = np.stack([x, x + D0 * u / np.linalg.norm(u)], axis=1)     # columns: reference, partner
    n = int(round(t_total / dt))
    total = 0.0
    for _ in range(n):
        xy = rk4_step(xy, forcing, dt)
        d = xy[:, 1] - xy[:, 0]
        nd = np.linalg.norm(d)
        total += np.log(nd / D0)
        xy[:, 1] = xy[:, 0] + d * (D0 / nd)
    return total / (n * dt)


def _l96_settings(proto: Protocol) -> dict:
    if proto.task != "lorenz96":
        raise ValueError(f"the Lyapunov time is defined for the lorenz96 task, got {proto.task!r}")
    defaults = {k: p.default for k, p in inspect.signature(lorenz96).parameters.items()
                if p.default is not inspect.Parameter.empty}
    return defaults | proto.task_kwargs


def lyapunov_time_steps(lam: float, proto: Protocol) -> float:
    """One Lyapunov time 1 / lambda in samples of the series: 1 / (lambda dt sample_every)."""
    kw = _l96_settings(proto)
    return 1.0 / (lam * kw["dt"] * kw["sample_every"])


def _shot_seed(res_seed: int, step: int) -> int:
    return int(np.random.SeedSequence([res_seed, step]).generate_state(1)[0])


def _taus(params, cmp):
    return tuple(params["tau"] / 2 ** k for k in range(cmp.n_taus))


def _reservoir(params, cmp, seed):
    return registry.get("reservoir", cmp.reservoir)(cmp.q, seed, **params["reservoir"])


class WindowStepper:
    """Features of window models from the last ``input_window`` inputs of each stream.

    ``features(buf, step)``: buf (B, W, n) -> (B, F), what the batch builder gives at a step whose
    last W inputs are that window. Classical models run their builder on the windows laid end to
    end: row k W + W - 1 of it sees exactly window k. The QRC runs its reset reservoir on the B
    windows as streams and keeps the last step, with a fresh shot realisation per ``step``. The
    window kind never tunes feedback (spaces._qrc_space), so only k_fb = 0 is supported."""

    def __init__(self, model, params, cmp: Comparison, sim: Sim, seed: int):
        if model == "qrc" and params["k_fb"] != 0.0:
            raise ValueError("window QRC stepping supports k_fb = 0 only")
        self.model, self.params, self.cmp, self.sim, self.seed = model, params, cmp, sim, seed
        self.res = _reservoir(params, cmp, seed) if model == "qrc" else None

    def features(self, buf, step):
        B, W, n = buf.shape
        cmp, p = self.cmp, self.params
        if self.model != "qrc":
            return MODELS[self.model].build(p, buf.reshape(B * W, n), self.seed, cmp, self.sim)[W - 1::W]
        ang = encode(buf.reshape(B * W, n), cmp.encoding, p["scale"])
        ang = ang.reshape(B, W, *ang.shape[1:])
        return qrc_features(ang, cmp.n_in, cmp.n_mem, self.res, taus=_taus(p, cmp), readout=cmp.readout,
                            mem_depth=W, memory="reset", method=self.sim.method, backend=self.sim.backend,
                            precision=self.sim.precision, streams=True, shots=cmp.shots,
                            shot_seed=_shot_seed(self.seed, step))[:, -1]


def _runner(model, params, cmp: Comparison, sim: Sim, X, seed, t0s):
    """(F, step): features on the true series (as many rows as were built, at least up to every
    start), and step(x, h) -> the features after feeding the predictions x (B, n) at step h."""
    if model == "qrc" and cmp.kind == "recurrent":
        st = RecurrentQRC(cmp.n_in, cmp.n_mem, _reservoir(params, cmp, seed), 1, taus=_taus(params, cmp),
                          readout=cmp.readout, k_fb=params["k_fb"], backend=sim.backend, precision=sim.precision,
                          shots=cmp.shots, shot_seed=seed)
        ang = encode(X, cmp.encoding, params["scale"])
        starts, rows, snaps = set(t0s.tolist()), [], {}
        for t in range(t0s.max() + 1):
            rows.append(st.step(ang[t:t + 1])[0])
            if t in starts:
                snaps[t] = st.snapshot()
        st.restore([snaps[t] for t in t0s])
        return np.array(rows), lambda x, h: st.step(encode(x, cmp.encoding, params["scale"]))

    F = MODELS[model].build(params, X, seed, cmp, sim)
    if model == "esn":
        weights = esn_weights(X.shape[1], cmp.width, seed, params["sr"], params["in_scale"], params["density"],
                              params["bias_scale"])
        state = {"r": F[t0s]}

        def step(x, h):
            state["r"] = np.stack([esn_step(r, xi, *weights, params["leak"]) for r, xi in zip(state["r"], x)])
            return state["r"]
        return F, step

    W = cmp.input_window
    ws = WindowStepper(model, params, cmp, sim, seed)
    state = {"buf": np.stack([X[t - W + 1:t + 1] for t in t0s])}

    def step(x, h):
        state["buf"] = np.concatenate([state["buf"][:, 1:], x[:, None]], axis=1)
        return ws.features(state["buf"], h)
    return F, step


def free_run(model, params, alpha, cmp: Comparison, proto: Protocol, sim: Sim, data_seed, res_seed, n_starts,
             horizon_steps, threshold, lt_steps) -> dict:
    """VPT and NRMSE curve of one frozen model on one (data seed, reservoir seed) run, from
    ``n_starts`` starts spread evenly over the test slice. VPTs are in Lyapunov times
    (``lt_steps`` samples each); a start that never crosses the threshold counts the full horizon."""
    if proto.horizon != 1:
        raise ValueError(f"a free run feeds each prediction back as the next input: horizon must be 1, "
                         f"got {proto.horizon}")
    X = _series(proto, data_seed)
    T, H = len(X), horizon_steps
    ts = test_start(T, proto.test_frac)
    if T - 1 - H < ts:
        raise ValueError(f"a {H}-step horizon does not fit in the {T - ts}-step test slice")
    t0s = np.round(np.linspace(ts, T - 1 - H, n_starts)).astype(int)
    F, step = _runner(model, params, cmp, sim, X, res_seed, t0s)
    predict = fit_readout(F, X[:len(F)], proto.horizon, alpha, proto.standardize, proto.washout, ts)
    sd = X[ts:].std(axis=0)
    feats = F[t0s]
    err = np.empty((len(t0s), H))                      # per start: RMS over series of the scaled error
    for h in range(1, H + 1):
        x = predict(feats)
        err[:, h - 1] = np.sqrt(np.mean(((x - X[t0s + h]) / sd) ** 2, axis=1))
        if h < H:
            feats = step(x, h)
    fail = err > threshold
    vpt = np.where(fail.any(axis=1), fail.argmax(axis=1), H) / lt_steps
    return {"vpt_mean": float(vpt.mean()), "vpt_per_start": vpt.tolist(),
            "nrmse_curve": np.sqrt(np.mean(err ** 2, axis=0)).tolist()}


def _from_record(cls, d):
    names = {f.name for f in fields(cls)}
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items() if k in names})


def run_freerun(comparison_json_path, n_starts=20, horizon_lt=10.0, threshold=0.5, out=None, log=print) -> dict:
    """Free runs of every model of a finished comparison at its frozen best setting, on every evaluation
    (data seed, reservoir seed) as evaluate_frozen does, with paired statistics as run_comparison's.
    The statistics compare -VPT, so the existing lower-is-better convention holds: wins_a counts runs
    where the QRC's VPT is the longer one. Writes freerun_<kind>.json next to the comparison."""
    path = Path(comparison_json_path)
    rec = json.loads(path.read_text())
    cmp = _from_record(Comparison, rec["comparison"])
    proto = _from_record(Protocol, rec["protocol"])
    sim = _from_record(Sim, rec["sim"])
    kw = _l96_settings(proto)
    t_start = time.perf_counter()
    lam = lyapunov_exponent(kw["n_series"], kw["forcing"], kw["dt"])
    lt = lyapunov_time_steps(lam, proto)
    H = int(round(horizon_lt * lt))
    log(f"=== free run, {cmp.kind}: lambda {lam:.4f}, Lyapunov time {lt:.2f} steps, horizon {H} steps, "
        f"{n_starts} starts, threshold {threshold}")
    result = {"source": {"comparison": str(path), "git_commit": rec.get("environment", {}).get("git_commit")},
              "lyapunov": {"exponent": lam, "time_steps": lt, "method": "Benettin, RK4 step of the lorenz96 task",
                           "t_total": T_TOTAL, "n_series": kw["n_series"], "forcing": kw["forcing"], "dt": kw["dt"],
                           "sample_every": kw["sample_every"]},
              "settings": {"n_starts": n_starts, "horizon_lt": horizon_lt, "horizon_steps": H,
                           "threshold": threshold, "vpt_unit": "Lyapunov times"},
              "models": {}}
    for m, entry in rec["models"].items():
        tuning, spec = entry["tuning"], MODELS[m]
        done, rows = {}, []
        for ds in proto.eval_data_seeds:
            for rs in proto.eval_res_seeds:
                k = (ds, rs if spec.uses_seed else proto.eval_res_seeds[0])
                if k not in done:
                    t0 = time.perf_counter()
                    done[k] = free_run(m, tuning["best_params"], tuning["best_alpha"], cmp, proto, sim, k[0], k[1],
                                       n_starts, H, threshold, lt)
                    done[k]["seconds"] = round(time.perf_counter() - t0, 3)
                rows.append({"data_seed": ds, "res_seed": rs} | done[k])
        vpts = np.array([r["vpt_mean"] for r in rows])
        curves = np.array([r["nrmse_curve"] for r in rows])
        marks = {str(k): float(curves[:, int(round(k * lt)) - 1].mean()) for k in LT_MARKS if int(round(k * lt)) <= H}
        result["models"][m] = {"rows": rows, "summary": {"vpt_mean": float(vpts.mean()), "vpt_std": float(vpts.std()),
                                                         "nrmse_at_lt": marks}}
        log(f"  {m:15s} VPT {vpts.mean():.3f} +- {vpts.std():.3f} Lyapunov times  "
            f"[{sum(d['seconds'] for d in done.values()):.0f}s]")
    stats = compare_to_qrc({m: {(r["data_seed"], r["res_seed"]): -r["vpt_mean"] for r in v["rows"]}
                            for m, v in result["models"].items()})
    result["stats"] = stats
    result["stats_convention"] = ("paired_comparison on -VPT (lower is better): "
                                  "wins_a = runs where the QRC's VPT is longer")
    result["environment"] = environment()
    result["total_seconds"] = round(time.perf_counter() - t_start, 3)
    target = Path(out) if out else path.parent / f"freerun_{cmp.kind}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    for n, s in stats.items():
        note = "" if s["run_level_independent"] else "  [run-level stats not independent]"
        log(f"  {n:22s} VPT {-s['mean_a']:.3f} vs {-s['mean_b']:.3f}  diff {-s['mean_diff']:+.3f}  "
            f"seed wins {s['cluster_wins_a']}/{s['n_clusters']}  p_cluster(Holm) {s['p_holm_cluster']:.3g}{note}")
    return result
