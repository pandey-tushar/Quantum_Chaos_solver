"""Build every table and every inline number of the paper from the pipeline's JSON records.

Writes tables/*.tex (\\input by main.tex) and tables/numbers.tex (one \\newcommand per inline number).
Usage: python make_tables.py [--results DIR]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(r"D:\Projects\Quantum_Chaos_solver_pipeline\results")

# the eight cells: (kind, reservoir, readout) -> record directory
CELLS = [
    ("window", "xxz", "ZZ", "l96_window_xxz_shots"),
    ("window", "xxz", "ZX", "l96_window_xxz_zx_shots"),
    ("window", "ising", "ZZ", "l96_window_ising_shots"),
    ("window", "ising", "ZX", "l96_window_ising_zx_shots"),
    ("recurrent", "xxz", "ZZ", "l96_recurrent_xxz_shots"),
    ("recurrent", "xxz", "ZX", "l96_recurrent_xxz_zx_shots"),
    ("recurrent", "ising", "ZZ", "l96_recurrent_ising_shots"),
    ("recurrent", "ising", "ZX", "l96_recurrent_ising_zx_shots"),
]
EXACT = {"window": "l96_window_xxz_exact", "recurrent": "l96_recurrent_xxz_exact"}   # xxz, Z/ZZ, shots null
WIDE = {"window": "l96_window_ising_zx_wide", "recurrent": "l96_recurrent_ising_zx_wide"}  # widened ranges
REFINE = "l96_window_ising_zx_refine"
ESN_NO_BIAS = "l96_recurrent_xxz"          # first exact recurrent run: ESN without the bias knob

CLASSICAL = ["random_features", "poly2", "linear", "esn"]
LABEL = {"qrc": "QRC", "random_features": "Random features", "poly2": "Poly2", "linear": "Linear",
         "esn": "ESN"}
RES_TEX = {"xxz": "XXZ", "ising": "Ising"}
READ_TEX = {"ZZ": "Z/ZZ", "ZX": "+X/XX"}
FLAT_TOL = 0.02
ALPHA = 0.05


def load(results: Path, name: str) -> dict:
    kind = "window" if "window" in name else "recurrent"
    return json.loads((results / name / f"comparison_{kind}.json").read_text())


def test_values(rec: dict, model: str) -> list[float]:
    return [r["test_nrmse"] for r in rec["models"][model]["eval"]]


def mean_sd(rec: dict, model: str) -> tuple[float, float]:
    v = test_values(rec, model)
    return st.mean(v), st.stdev(v)


def pm(rec: dict, model: str, bold: bool = False) -> str:
    if model not in rec["models"]:
        return "--"
    m, s = mean_sd(rec, model)
    return f"$\\mathbf{{{m:.3f}}}\\pm{s:.3f}$" if bold else f"${m:.3f}\\pm{s:.3f}$"


def pval(p: float) -> str:
    if p >= 0.01:
        return f"${p:.3f}$"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def sci(x: float) -> str:
    mant, exp = f"{x:.1e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


def cell_name(kind, res, readout) -> str:
    return f"{kind} & {RES_TEX[res]} & {READ_TEX[readout]}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def same_protocol(a: dict, b: dict, ignore=()) -> bool:
    """Protocols equal up to the refine keys (absent in older records, 'none' in newer) and ``ignore``."""
    skip = {"refine", "refine_frac", *ignore}
    return {k: v for k, v in a.items() if k not in skip} == {k: v for k, v in b.items() if k not in skip}


def space(rec: dict, model: str, knob: str) -> tuple[float, float]:
    a = rec["models"][model]["tuning"]["search_space"][knob]["attributes"]
    return a["low"], a["high"]


# ---------------------------------------------------------------- one-step tables

def cells_nrmse(recs) -> str:
    rows = []
    for (kind, res, readout, _), rec in zip(CELLS, recs):
        means = {m: mean_sd(rec, m)[0] for m in rec["models"]}
        best = min(means, key=means.get)
        cols = [pm(rec, m, m == best) for m in ["qrc"] + CLASSICAL]
        rows.append(f"{cell_name(kind, res, readout)} & {rec['comparison']['width']} & " + " & ".join(cols) + r" \\")
    return "\n".join([
        r"\begin{tabular}{lllrccccc}", r"\toprule",
        r"Kind & Res. & Readout & Width & QRC & RF & Poly2 & Linear & ESN \\",
        r"\midrule", *rows[:4], r"\midrule", *rows[4:], r"\bottomrule", r"\end{tabular}", ""])


def cells_stats(recs, part: str) -> str:
    """part 'wins': QRC wins in runs (data seeds) per classical model; part 'p': Holm p per model."""
    rows = []
    for (kind, res, readout, _), rec in zip(CELLS, recs):
        cols = []
        for m in CLASSICAL:
            s = rec["stats"].get(f"qrc_vs_{m}")
            if s is None:
                cols.append("--")
            elif part == "wins":
                dag = "" if s["run_level_independent"] else r"$^\dagger$"
                cols.append(f"{s['wins_a']}/{s['n_runs']} ({s['cluster_wins_a']}/{s['n_clusters']}){dag}")
            else:
                cols.append(pval(s["p_holm_cluster"]))
        rows.append(f"{cell_name(kind, res, readout)} & " + " & ".join(cols) + r" \\")
    head = " & ".join(f"vs {LABEL[m]}" for m in CLASSICAL)
    return "\n".join([
        r"\begin{tabular}{lll" + "c" * len(CLASSICAL) + "}", r"\toprule",
        rf"Kind & Res. & Readout & {head} \\",
        r"\midrule", *rows[:4], r"\midrule", *rows[4:], r"\bottomrule", r"\end{tabular}", ""])


def basis_gain(recs) -> str:
    by = {(k, r, ro): rec for (k, r, ro, _), rec in zip(CELLS, recs)}
    rows = []
    for kind in ("window", "recurrent"):
        for res in ("xxz", "ising"):
            zz, zx = by[(kind, res, "ZZ")], by[(kind, res, "ZX")]
            cols = ["--" if m not in zz["models"] else f"${mean_sd(zx, m)[0] - mean_sd(zz, m)[0]:+.3f}$"
                    for m in ["qrc", "random_features", "esn"]]
            gap_zz = mean_sd(zz, "qrc")[0] - mean_sd(zz, "random_features")[0]
            gap_zx = mean_sd(zx, "qrc")[0] - mean_sd(zx, "random_features")[0]
            rows.append(f"{kind} & {RES_TEX[res]} & {mean_sd(zz, 'qrc')[0]:.3f} & {mean_sd(zx, 'qrc')[0]:.3f} & "
                        + " & ".join(cols) + f" & {gap_zz:.3f} & {gap_zx:.3f}" + r" \\")
    return "\n".join([
        r"\begin{tabular}{llccccccc}", r"\toprule",
        r" & & \multicolumn{2}{c}{QRC NRMSE} & \multicolumn{3}{c}{change Z/ZZ $\to$ +X/XX} "
        r"& \multicolumn{2}{c}{QRC $-$ RF} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
        r"Kind & Res. & Z/ZZ & +X/XX & QRC & RF & ESN & Z/ZZ & +X/XX \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""])


def basis_macros(recs, out: list):
    by = {(k, r, ro): rec for (k, r, ro, _), rec in zip(CELLS, recs)}
    q = {k: mean_sd(rec, "qrc")[0] for k, rec in by.items()}
    rf = {k: mean_sd(rec, "random_features")[0] for k, rec in by.items()}
    assert all(q[(k, r, "ZX")] < q[(k, r, "ZZ")] for k in ("window", "recurrent") for r in ("xxz", "ising"))
    ratios = {kind: [(q[(kind, r, "ZZ")] - q[(kind, r, "ZX")]) / (rf[(kind, r, "ZZ")] - rf[(kind, r, "ZX")])
                     for r in ("xxz", "ising")] for kind in ("window", "recurrent")}
    out += [macro("ratioWinMin", f"{min(ratios['window']):.1f}"), macro("ratioWinMax", f"{max(ratios['window']):.1f}"),
            macro("ratioRecMin", f"{100 / max(ratios['recurrent']):.0f}"),
            macro("ratioRecMax", f"{100 / min(ratios['recurrent']):.0f}")]
    gap = {k: q[k] - rf[k] for k in by}
    win = [1 - gap[("window", r, "ZX")] / gap[("window", r, "ZZ")] for r in ("xxz", "ising")]
    rec = [gap[("recurrent", r, "ZZ")] - gap[("recurrent", r, "ZX")] for r in ("xxz", "ising")]
    out += [macro("gapShrinkWinPctMin", f"{100 * min(win):.0f}"), macro("gapShrinkWinPctMax", f"{100 * max(win):.0f}"),
            macro("gapShrinkRecMin", f"{min(rec):.2f}"), macro("gapShrinkRecMax", f"{max(rec):.2f}")]
    # the basis change is the largest one-factor change of the QRC in the grid
    factors = (("window", "recurrent"), ("xxz", "ising"), ("ZZ", "ZX"))
    change = {}
    for i, (a, b) in enumerate(factors):
        for k in by:
            if k[i] == a:
                k2 = list(k)
                k2[i] = b
                change[(i, k)] = abs(q[k] - q[tuple(k2)])
    assert max(change, key=change.get)[0] == 2


# ---------------------------------------------------------------- shots, refinement, widened ranges

def shots_table(results: Path, recs, out: list) -> str:
    by = {(k, r, ro): rec for (k, r, ro, _), rec in zip(CELLS, recs)}
    pairs = {kind: (load(results, EXACT[kind]), by[(kind, "xxz", "ZZ")]) for kind in ("window", "recurrent")}
    rows = []
    for kind, (ex, sh) in pairs.items():
        assert ex["comparison"].get("shots") is None and sh["comparison"]["shots"] == 12288
        assert same_protocol(ex["protocol"], sh["protocol"])
        for m in ex["models"]:
            if m != "qrc":
                assert test_values(ex, m) == test_values(sh, m), (kind, m)     # classical rows identical
    for m in ["qrc"] + CLASSICAL:
        cols = []
        for kind in ("window", "recurrent"):
            ex, sh = pairs[kind]
            if m not in ex["models"]:
                cols += ["--", "--", "--"]
                continue
            a, b = mean_sd(ex, m)[0], mean_sd(sh, m)[0]
            cols += [f"{a:.3f}", f"{b:.3f}", "$0$" if a == b else f"${b - a:+.3f}$"]
        rows.append(f"{LABEL[m]} & " + " & ".join(cols) + r" \\")
    for kind, (ex, sh) in pairs.items():
        out.append(macro(f"shotLoss{kind.capitalize()}", num(mean_sd(sh, "qrc")[0] - mean_sd(ex, "qrc")[0])))
    s = pairs["recurrent"][0]["stats"]["qrc_vs_random_features"]
    assert s["p_holm_cluster"] > ALPHA
    out += [macro("exactWinsRfRecurrent", str(s["wins_a"])),
            macro("exactClusterWinsRfRecurrent", str(s["cluster_wins_a"])),
            macro("exactPRfRecurrent", pval(s["p_holm_cluster"]).strip("$"))]
    return "\n".join([
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r" & \multicolumn{3}{c}{window, XXZ, Z/ZZ} & \multicolumn{3}{c}{recurrent, XXZ, Z/ZZ} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Model & exact & 12288 shots & change & exact & 12288 shots & change \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""])


def refine_table(rec, out: list) -> str:
    rows, flats = [], {}
    for m in ["qrc", "random_features", "poly2", "linear"]:
        t = rec["models"][m]["tuning"]
        b = t["budget"]
        n_tpe = t["n_trials"] - b["refine_evals"]
        tpe = t["trial_values"][:n_tpe]
        best = min(tpe)
        assert abs(best - t["best_val"]) < 1e-12 and b["refine_gain"] == 0.0
        assert b["refine_evals"] == 2 * (int(rec["protocol"]["refine_frac"] * b["floor"]) // 2)
        flats[m] = sum(v <= best + FLAT_TOL for v in tpe) / n_tpe
        rows.append(f"{LABEL[m]} & {b['n_dims']} & {n_tpe} & {b['refine_evals']} & {best:.4f} & "
                    f"{min(t['trial_values'][n_tpe:]):.4f} & {b['refine_gain']:.1f} & {100 * flats[m]:.0f}\\%" + r" \\")
    classical = [flats[m] for m in ("random_features", "poly2", "linear")]
    assert min(classical) == flats["random_features"] and max(classical) == flats["linear"] > flats["qrc"]
    out += [macro("flatQrc", f"{100 * flats['qrc']:.0f}"), macro("flatRf", f"{100 * flats['random_features']:.0f}"),
            macro("flatLin", f"{100 * flats['linear']:.0f}"),
            macro("refineCommit", rec["environment"]["git_commit"][:7])]
    return "\n".join([
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Model & knobs & TPE trials & SPSA evals & best TPE val. & best SPSA val. & gain & "
        rf"within {FLAT_TOL} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""])


def wide_table(results: Path, recs, out: list) -> str:
    by = {(k, r, ro): rec for (k, r, ro, _), rec in zip(CELLS, recs)}
    rows = []
    for kind in ("window", "recurrent"):
        orig, wide = by[(kind, "ising", "ZX")], load(results, WIDE[kind])
        assert wide["comparison"] == orig["comparison"] and same_protocol(wide["protocol"], orig["protocol"], ["alpha_range"])
        for tag, rec in (("orig.", orig), ("wide", wide)):
            cols = [num(mean_sd(rec, m)[0]) if m in rec["models"] else "--"
                    for m in ("qrc", "random_features", "poly2", "esn")]
            for m in ("random_features", "poly2", "esn"):
                s = rec["stats"].get(f"qrc_vs_{m}")
                cols.append("--" if s is None else f"{s['cluster_wins_a']}/5, {pval(s['p_holm_cluster'])}")
            cols.append(f"{rec['models']['qrc']['tuning']['best_params']['reservoir']['v']:.2f}")
            if "esn" in rec["models"]:
                p = rec["models"]["esn"]["tuning"]["best_params"]
                cols += [f"{p['sr']:.3f}", f"{p['bias_scale']:.2f}"]
            else:
                cols += ["--", "--"]
            rows.append(f"{kind} & {tag} & " + " & ".join(cols) + r" \\")
        for m in ("random_features", "esn"):                     # no verdict change against RF or ESN
            if m in wide["models"]:
                assert wide["stats"][f"qrc_vs_{m}"]["cluster_wins_a"] == 0
                assert wide["stats"][f"qrc_vs_{m}"]["p_holm_cluster"] < ALPHA
        assert wide["stats"]["qrc_vs_poly2"]["p_holm_cluster"] > ALPHA
    w_rec, w_win = load(results, WIDE["recurrent"]), load(results, WIDE["window"])
    o_rec = by[("recurrent", "ising", "ZX")]
    assert o_rec["stats"]["qrc_vs_poly2"]["p_holm_cluster"] < ALPHA
    assert w_rec["models"]["random_features"]["tuning"]["budget"]["stopped"] == "max_trials"
    lo_v, hi_v = space(w_win, "qrc", "v")
    lo_sr, hi_sr = space(w_rec, "esn", "sr")
    _, hi_b = space(w_rec, "esn", "bias_scale")
    lo_a = {space(r, m, "alpha")[0] for r in (w_rec, w_win) for m in r["models"]}
    assert len(lo_a) == 1 and space(w_rec, "qrc", "v") == (lo_v, hi_v)
    out += [macro("wideVLow", f"{lo_v:g}"), macro("wideVHigh", f"{hi_v:g}"), macro("wideSrLow", f"{lo_sr:g}"),
            macro("wideSrHigh", f"{hi_sr:g}"), macro("wideBiasHigh", f"{hi_b:g}"),
            macro("wideAlphaLow", f"10^{{{round(__import__('math').log10(lo_a.pop()))}}}"),
            macro("vHigh", f"{space(o_rec, 'qrc', 'v')[1]:g}"),
            macro("wideOldPolyPRec", pval(o_rec["stats"]["qrc_vs_poly2"]["p_holm_cluster"]).strip("$")),
            macro("wideNewPolyPRec", pval(w_rec["stats"]["qrc_vs_poly2"]["p_holm_cluster"]).strip("$"))]
    return "\n".join([
        r"\begin{tabular}{llcccccccccc}", r"\toprule",
        r" & & \multicolumn{4}{c}{test NRMSE} & \multicolumn{3}{c}{QRC seed wins, Holm $p$} & & "
        r"\multicolumn{2}{c}{ESN} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-9}\cmidrule(lr){11-12}",
        r"Kind & ranges & QRC & RF & Poly2 & ESN & vs RF & vs Poly2 & vs ESN & $v$ & radius & bias \\",
        r"\midrule", *rows[:2], r"\midrule", *rows[2:], r"\bottomrule", r"\end{tabular}", ""])


# ---------------------------------------------------------------- free run

def load_freerun(results: Path, name: str, rec: dict) -> dict:
    """The free-run record of a cell, checked against the cell's comparison record."""
    kind = rec["comparison"]["kind"]
    fr = json.loads((results / name / f"freerun_{kind}.json").read_text())
    assert Path(fr["source"]["comparison"]).parent.name == name
    assert fr["source"]["git_commit"] == rec["environment"]["git_commit"]
    assert set(fr["models"]) == set(rec["models"])
    return fr


def vpt_values(fr: dict, model: str) -> list[float]:
    return [r["vpt_mean"] for r in fr["models"][model]["rows"]]


def vpt_table(frs) -> str:
    rows = []
    for (kind, res, readout, _), fr in zip(CELLS, frs):
        lt = fr["lyapunov"]["time_steps"]
        means = {m: st.mean(vpt_values(fr, m)) for m in fr["models"]}
        best = max(means, key=means.get)
        cols = []
        for m in ["qrc"] + CLASSICAL:
            if m not in fr["models"]:
                cols.append("--")
                continue
            v = vpt_values(fr, m)
            mu, sd = st.mean(v), st.stdev(v)
            assert abs(mu - fr["models"][m]["summary"]["vpt_mean"]) < 1e-12
            core = rf"\mathbf{{{mu:.2f}}}" if m == best else f"{mu:.2f}"
            cols.append(f"${core}\\pm{sd:.2f}$ ({mu * lt:.1f})")
        rows.append(f"{RES_TEX[res]} & {READ_TEX[readout]} & " + " & ".join(cols) + r" \\")
    group = [r"\multicolumn{7}{l}{\emph{" + k + r"}} \\" for k in ("window", "recurrent")]
    return "\n".join([
        r"\begin{tabular}{llccccc}", r"\toprule",
        r"Res. & Readout & QRC & RF & Poly2 & Linear & ESN \\",
        r"\midrule", group[0], *rows[:4], r"\midrule", group[1], *rows[4:], r"\bottomrule", r"\end{tabular}", ""])


def vpt_macros(frs, out: list):
    by = {(k, r, ro): fr for (k, r, ro, _), fr in zip(CELLS, frs)}
    f0 = frs[0]
    for fr in frs:                       # one Lyapunov setting for the whole grid
        assert fr["lyapunov"] == f0["lyapunov"] and fr["settings"] == f0["settings"]
    lam, lt, s = f0["lyapunov"]["exponent"], f0["lyapunov"]["time_steps"], f0["settings"]
    out += [macro("lyapExp", f"{lam:.3f}"), macro("lyapTimeSteps", f"{lt:.2f}"),
            macro("horizonLt", f"{s['horizon_lt']:g}"), macro("nStarts", str(s["n_starts"])),
            macro("vptThreshold", f"{s['threshold']:g}"), macro("sampleEvery", str(f0["lyapunov"]["sample_every"]))]
    means = {(c, m): st.mean(vpt_values(fr, m)) for c, fr in by.items() for m in fr["models"]}
    top = max(means, key=means.get)
    assert max(means.values()) < 1.0 and top[1] == "esn"
    out.append(macro("vptBest", f"{means[top]:.2f}"))
    for c, fr in by.items():
        kind, _, ro = c
        assert means[(c, "random_features")] > means[(c, "qrc")]
        if kind == "recurrent":
            assert means[(c, "esn")] > means[(c, "random_features")]
        if ro == "ZX":
            sp = fr["stats"]["qrc_vs_poly2"]
            assert sp["cluster_wins_a"] == sp["n_clusters"] and sp["wins_a"] == sp["n_runs"]
        else:                            # one basis: ties with linear (window) and Poly2 (recurrent)
            assert fr["stats"]["qrc_vs_" + ("linear" if kind == "window" else "poly2")]["p_holm_cluster"] > ALPHA
    out.append(macro("vptSigRfCells", str(sum(fr["stats"]["qrc_vs_random_features"]["p_holm_cluster"] < ALPHA
                                              for fr in frs))))
    poly = {k: by[(k, "xxz", "ZZ")]["models"]["poly2"]["summary"]["nrmse_at_lt"]["1"] for k in ("window", "recurrent")}
    out.append(macro("polyNrmseOneLtWindow", sci(poly["window"])))


# ---------------------------------------------------------------- remaining inline numbers

def general_macros(results: Path, recs, out: list):
    by = {(k, r, ro): rec for (k, r, ro, _), rec in zip(CELLS, recs)}
    assert all(t["tuning"]["budget"]["stopped"] == "patience" for rec in recs for t in rec["models"].values())
    # QRC vs random features in every cell
    rfs = [rec["stats"]["qrc_vs_random_features"] for rec in recs]
    assert all(s["wins_a"] == 0 and s["cluster_wins_a"] == 0 for s in rfs)
    out += [macro("maxPRf", pval(max(s["p_holm_cluster"] for s in rfs)).strip("$")),
            macro("minGapRf", num(min(s["mean_diff"] for s in rfs))),
            macro("maxGapRf", num(max(s["mean_diff"] for s in rfs)))]
    n_tests = sum(len(rec["stats"]) for rec in recs)
    raw_rf = max(s["p_t_cluster"] for s in rfs)
    assert raw_rf * n_tests < ALPHA
    out += [macro("nTests", str(n_tests)), macro("bonfRf", pval(raw_rf * n_tests).strip("$"))]
    # best QRC cell vs Poly2
    s = by[("window", "ising", "ZX")]["stats"]["qrc_vs_poly2"]
    assert s["p_holm_cluster"] > ALPHA and s["ci95_cluster"][0] < 0 < s["ci95_cluster"][1]
    out += [macro("bestPolyRunWins", str(s["wins_a"])), macro("bestPolySeedWins", str(s["cluster_wins_a"])),
            macro("bestPolyP", pval(s["p_holm_cluster"]).strip("$")),
            macro("bestPolyCiLo", num(s["ci95_cluster"][0], 4)), macro("bestPolyCiHi", num(s["ci95_cluster"][1], 4))]
    # ESN in the recurrent kind
    rec_cells = [by[("recurrent", r, ro)] for r in ("xxz", "ising") for ro in ("ZZ", "ZX")]
    esn = [rec["stats"]["qrc_vs_esn"] for rec in rec_cells]
    assert all(e["wins_a"] == 0 and e["cluster_wins_a"] == 0 for e in esn)
    out += [macro("minGapEsn", num(min(e["mean_diff"] for e in esn))),
            macro("maxGapEsn", num(max(e["mean_diff"] for e in esn)))]
    esn_p = [rec["models"]["esn"]["tuning"]["best_params"] for rec in rec_cells]
    leaks = [p["leak"] for p in esn_p]
    out += [macro("esnSrLow", f"{space(rec_cells[0], 'esn', 'sr')[0]:g}"),
            macro("esnLeakMin", f"{min(leaks):.2f}"), macro("esnLeakMax", f"{max(leaks):.2f}"),
            macro("esnKeepMin", f"{1 - max(leaks):.2f}"), macro("esnKeepMax", f"{1 - min(leaks):.2f}")]
    # Poly2 results that may not survive a grid-wide correction
    weak = sorted(rec["stats"]["qrc_vs_poly2"]["p_holm_cluster"] for rec in recs
                  if 0.005 < rec["stats"]["qrc_vs_poly2"]["p_holm_cluster"] < ALPHA)
    out += [macro("polyWeakPLo", pval(weak[0]).strip("$")), macro("polyWeakPHi", pval(weak[-1]).strip("$"))]
    c, czx = recs[0]["comparison"], by[("window", "xxz", "ZX")]["comparison"]
    out += [macro("shots", str(c["shots"])), macro("nQubits", str(c["q"])), macro("widthZZ", str(c["width"])),
            macro("widthZX", str(czx["width"])),
            macro("widthPoly", str(recs[0]["models"]["poly2"]["eval"][0]["width"])),
            macro("widthLin", str(recs[0]["models"]["linear"]["eval"][0]["width"]))]
    T = by[("recurrent", "xxz", "ZZ")]["protocol"]["n_steps"]
    out += [macro("recSteps", str(T)), macro("replayDrives", sci(T * (T + 1) / 2))]
    # ESN without the bias knob (first exact recurrent run) against the clean exact run
    nb, ex = load(results, ESN_NO_BIAS), load(results, EXACT["recurrent"])
    assert "bias_scale" not in nb["models"]["esn"]["tuning"]["best_params"]
    assert ex["models"]["qrc"]["tuning"]["best_params"] == nb["models"]["qrc"]["tuning"]["best_params"]
    assert max(abs(a - b) for a, b in zip(test_values(ex, "qrc"), test_values(nb, "qrc"))) < 1e-5   # float32 GPU
    assert ex["stats"]["qrc_vs_esn"]["wins_a"] == 0
    out += [macro("noBiasEsn", num(mean_sd(nb, "esn")[0])), macro("noBiasQrc", num(mean_sd(nb, "qrc")[0])),
            macro("noBiasQrcWins", str(nb["stats"]["qrc_vs_esn"]["wins_a"])),
            macro("biasEsn", num(mean_sd(ex, "esn")[0]))]
    sds = [mean_sd(rec, "qrc")[1] for rec in recs]
    pers = [v for rec in recs for v in rec["persistence"].values()]
    out += [macro("qrcSdMin", num(min(sds))), macro("qrcSdMax", num(max(sds))),
            macro("persMin", f"{min(pers):.2f}"), macro("persMax", f"{max(pers):.2f}"),
            macro("gpu", nb["environment"]["gpu"].replace("NVIDIA GeForce ", ""))]


def record_names() -> list[str]:
    return ([c[3] for c in CELLS] + [EXACT["window"], EXACT["recurrent"], ESN_NO_BIAS, REFINE,
                                      WIDE["window"], WIDE["recurrent"]])


def repro_table(results: Path, out: list) -> str:
    rows, total = [], 0.0
    for n in record_names():
        rec = load(results, n)
        total += rec["total_seconds"]
        q = rec["models"]["qrc"]["tuning"]["study_seconds"]
        rows.append(rf"\texttt{{{n.replace('_', chr(92) + '_')}}} & \texttt{{{rec['environment']['git_commit'][:7]}}} & "
                    f"{rec['total_seconds'] / 3600:.2f} & {q / 3600:.2f}" + r" \\")
    out.append(macro("totalHours", f"{total / 3600:.1f}"))
    return "\n".join([
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Record & commit & total (h) & QRC tuning (h) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=RESULTS)
    results = ap.parse_args().results
    recs = [load(results, c[3]) for c in CELLS]
    for (kind, res, readout, _), rec in zip(CELLS, recs):
        c = rec["comparison"]
        assert (c["kind"], c["readout"], c["shots"], c["q"]) == (kind, readout, 12288, 12), c
        assert c["reservoir"] == {"xxz": "xxz", "ising": "ising_xx"}[res]
    macros: list[str] = []
    out = HERE / "tables"
    out.mkdir(exist_ok=True)
    for stale in ("cells_stats.tex", "vpt_stats.tex"):
        (out / stale).unlink(missing_ok=True)
    (out / "cells_nrmse.tex").write_text(cells_nrmse(recs))
    (out / "cells_wins.tex").write_text(cells_stats(recs, "wins"))
    (out / "cells_p.tex").write_text(cells_stats(recs, "p"))
    (out / "basis_gain.tex").write_text(basis_gain(recs))
    basis_macros(recs, macros)
    (out / "shots.tex").write_text(shots_table(results, recs, macros))
    (out / "refine.tex").write_text(refine_table(load(results, REFINE), macros))
    (out / "wide.tex").write_text(wide_table(results, recs, macros))
    (out / "repro.tex").write_text(repro_table(results, macros))
    general_macros(results, recs, macros)
    frs = [load_freerun(results, c[3], rec) for c, rec in zip(CELLS, recs)]
    (out / "vpt.tex").write_text(vpt_table(frs))
    (out / "vpt_wins.tex").write_text(cells_stats(frs, "wins"))
    (out / "vpt_p.tex").write_text(cells_stats(frs, "p"))
    vpt_macros(frs, macros)
    commits = {fr["environment"]["git_commit"][:7] for fr in frs}
    assert len(commits) == 1
    macros += [macro("freerunCommit", commits.pop()),
               macro("freerunHours", f"{sum(fr['total_seconds'] for fr in frs) / 3600:.2f}")]
    (out / "numbers.tex").write_text("% generated by make_tables.py; do not edit\n" + "".join(macros))
    print("wrote", sorted(p.name for p in out.glob("*.tex")))


if __name__ == "__main__":
    main()
