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
EXACT = {"window": "l96_window_xxz", "recurrent": "l96_recurrent_xxz_v2"}   # xxz, Z/ZZ, shots null
REFINE = "l96_window_ising_zx_refine"
ESN_NO_BIAS = "l96_recurrent_xxz"          # first exact recurrent run: ESN without the bias knob

CLASSICAL = ["random_features", "poly2", "linear", "esn"]
LABEL = {"qrc": "QRC", "random_features": "Random features", "poly2": "Poly2", "linear": "Linear",
         "esn": "ESN"}
RES_TEX = {"xxz": "XXZ", "ising": "Ising"}
READ_TEX = {"ZZ": "Z/ZZ", "ZX": "+X/XX"}
FLAT_TOL = 0.02


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
    core = f"{m:.3f}\\pm{s:.3f}"
    return f"$\\mathbf{{{m:.3f}}}\\pm{s:.3f}$" if bold else f"${core}$"


def pval(p: float) -> str:
    if p >= 0.01:
        return f"${p:.3f}$"
    mant, exp = f"{p:.1e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def cell_name(kind, res, readout) -> str:
    return f"{kind} & {RES_TEX[res]} & {READ_TEX[readout]}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def num(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def cells_nrmse(recs) -> str:
    rows = []
    for (kind, res, readout, _), rec in zip(CELLS, recs):
        means = {m: mean_sd(rec, m)[0] for m in rec["models"]}
        best = min(means, key=means.get)
        width = rec["comparison"]["width"]
        cols = [pm(rec, m, m == best) for m in ["qrc"] + CLASSICAL]
        rows.append(f"{cell_name(kind, res, readout)} & {width} & " + " & ".join(cols) + r" \\")
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
            macro("sampleLt", f"{1 / lt:.2f}"), macro("horizonLt", f"{s['horizon_lt']:g}"),
            macro("horizonSteps", str(s["horizon_steps"])), macro("nStarts", str(s["n_starts"])),
            macro("vptThreshold", f"{s['threshold']:g}"),
            macro("sampleEvery", str(f0["lyapunov"]["sample_every"]))]
    means = {(c, m): st.mean(vpt_values(fr, m)) for c, fr in by.items() for m in fr["models"]}
    top = max(means, key=means.get)
    assert max(means.values()) < 1.0 and top[1] == "esn" and top[0][2] == "ZX"
    out += [macro("vptBest", f"{means[top]:.2f}"),
            macro("vptMaxRun", f"{max(max(vpt_values(fr, m)) for fr in frs for m in fr['models']):.2f}")]
    for c, fr in by.items():
        kind, _, ro = c
        rf = means[(c, "random_features")]
        assert rf > means[(c, "qrc")]
        if kind == "recurrent":
            assert means[(c, "esn")] > rf
        if ro == "ZX":
            sp = fr["stats"]["qrc_vs_poly2"]
            assert sp["cluster_wins_a"] == sp["n_clusters"] and sp["wins_a"] == sp["n_runs"]
    def prange(kind, ro, comp):
        ps = [by[(kind, r, ro)]["stats"][f"qrc_vs_{comp}"]["p_holm_cluster"] for r in ("xxz", "ising")]
        return pval(min(ps)).strip("$"), pval(max(ps)).strip("$")
    for kind in ("window", "recurrent"):
        K = kind.capitalize()
        lo, hi = prange(kind, "ZX", "random_features")
        out += [macro(f"vptPRfZX{K}Lo", lo), macro(f"vptPRfZX{K}Hi", hi)]
        lo, hi = prange(kind, "ZZ", "random_features")
        out += [macro(f"vptPRfZZ{K}Lo", lo), macro(f"vptPRfZZ{K}Hi", hi)]
        lo, hi = prange(kind, "ZZ", "poly2")
        out += [macro(f"vptPPolyZZ{K}Lo", lo), macro(f"vptPPolyZZ{K}Hi", hi)]
        poly1 = by[(kind, "xxz", "ZZ")]["models"]["poly2"]["summary"]["nrmse_at_lt"]["1"]
        mant, exp = f"{poly1:.1e}".split("e")
        out.append(macro(f"polyNrmseOneLt{K}", f"{mant}\\times10^{{{int(exp)}}}"))
    n_sig = sum(fr["stats"]["qrc_vs_random_features"]["p_holm_cluster"] < 0.05 for fr in frs)
    out.append(macro("vptSigRfCells", str(n_sig)))
    rz = [by[("recurrent", r, "ZZ")]["stats"]["qrc_vs_random_features"]["cluster_wins_a"] for r in ("xxz", "ising")]
    out.append(macro("vptRecZZRfSeedWinsMax", str(max(rz))))
    lin = [means[(c, "linear")] for c in by]
    out += [macro("vptLinMax", f"{max(lin):.2f}")]
    qzz = [means[(("window", r, "ZZ"), "qrc")] for r in ("xxz", "ising")]
    out += [macro("vptQrcWinZZMin", f"{min(qzz):.2f}"), macro("vptQrcWinZZMax", f"{max(qzz):.2f}")]


def review_macros(results: Path, recs, frs, out: list):
    """Numbers added in the review pass: multiplicity, edge optima, ESN memory, shrinking gaps, VPT ties."""
    by = {(k, r, ro): rec for (k, r, ro, _), rec in zip(CELLS, recs)}
    fby = {(k, r, ro): fr for (k, r, ro, _), fr in zip(CELLS, frs)}
    n_tests = sum(len(rec["stats"]) for rec in recs)
    raw_rf = max(rec["stats"]["qrc_vs_random_features"]["p_t_cluster"] for rec in recs)
    out += [macro("nTests", str(n_tests)), macro("bonfRf", pval(min(1.0, raw_rf * n_tests)).strip("$"))]
    assert raw_rf * n_tests < 0.05
    weak = sorted(rec["stats"]["qrc_vs_poly2"]["p_holm_cluster"] for rec in recs
                  if 0.005 < rec["stats"]["qrc_vs_poly2"]["p_holm_cluster"] < 0.05)
    out += [macro("polyWeakPLo", pval(weak[0]).strip("$")), macro("polyWeakPHi", pval(weak[-1]).strip("$"))]
    # ridge penalty at the lower edge of its range
    lo = by[("window", "xxz", "ZZ")]["models"]["qrc"]["tuning"]["search_space"]["alpha"]["attributes"]["low"]
    edge = {(c, m): rec["models"][m]["tuning"]["best_alpha"] for c, rec in by.items() for m in rec["models"]
            if rec["models"][m]["tuning"]["best_alpha"] < 1.5 * lo}
    assert set(edge) == {(c, "random_features") for c in by if c[2] == "ZZ"} | {(("window", "xxz", "ZX"), "qrc")}
    assert lo == 1e-6
    out += [macro("alphaLow", "10^{-6}"),
            macro("alphaEdgeMax", f"{max(edge.values()) / lo:.2f}" + r"\times10^{-6}")]
    # ESN memory
    leaks = [by[("recurrent", r, ro)]["models"]["esn"]["tuning"]["best_params"]["leak"]
             for r in ("xxz", "ising") for ro in ("ZZ", "ZX")]
    out += [macro("esnLeakMin", f"{min(leaks):.2f}"), macro("esnLeakMax", f"{max(leaks):.2f}"),
            macro("esnKeepMin", f"{1 - max(leaks):.2f}"), macro("esnKeepMax", f"{1 - min(leaks):.2f}")]
    # recurrent gap to random features, one basis -> two
    shr = []
    for r in ("xxz", "ising"):
        g = [mean_sd(by[("recurrent", r, ro)], "qrc")[0] - mean_sd(by[("recurrent", r, ro)], "random_features")[0]
             for ro in ("ZZ", "ZX")]
        shr.append(g[0] - g[1])
    out += [macro("gapShrinkRecMin", f"{min(shr):.2f}"), macro("gapShrinkRecMax", f"{max(shr):.2f}")]
    # persistence NRMSE over all eight cells
    pers = [v for rec in recs for v in rec["persistence"].values()]
    out += [macro("persMin", f"{min(pers):.2f}"), macro("persMax", f"{max(pers):.2f}")]
    # exact-reference commits
    for kind in ("window", "recurrent"):
        K = kind.capitalize()
        out += [macro(f"exactCommit{K}", load(results, EXACT[kind])["environment"]["git_commit"][:7]),
                macro(f"shotsCommit{K}", by[(kind, "xxz", "ZZ")]["environment"]["git_commit"][:7])]
    # VPT ties of the one-basis QRC
    for (kind, comp, tag) in (("window", "linear", "Lin"), ("recurrent", "poly2", "Poly")):
        for r in ("xxz", "ising"):
            st_ = fby[(kind, r, "ZZ")]["stats"][f"qrc_vs_{comp}"]
            R = RES_TEX[r].capitalize()
            out += [macro(f"vptTie{tag}{R}Wins", str(st_["wins_a"])),
                    macro(f"vptTie{tag}{R}Seeds", str(st_["cluster_wins_a"])),
                    macro(f"vptTie{tag}{R}P", pval(st_["p_holm_cluster"]).strip("$"))]


def basis_gain(recs, out_macros: list) -> str:
    by = {(k, r, ro): rec for (k, r, ro, _), rec in zip(CELLS, recs)}
    rows = []
    for kind in ("window", "recurrent"):
        for res in ("xxz", "ising"):
            zz, zx = by[(kind, res, "ZZ")], by[(kind, res, "ZX")]
            cols = []
            for m in ["qrc", "random_features", "esn"]:
                if m not in zz["models"]:
                    cols.append("--")
                    continue
                d = mean_sd(zx, m)[0] - mean_sd(zz, m)[0]
                cols.append(f"${d:+.3f}$")
                tag = {"qrc": "Qrc", "random_features": "Rf", "esn": "Esn"}[m]
                out_macros.append(macro(f"gain{tag}{kind.capitalize()}{RES_TEX[res].capitalize()}",
                                        num(-d)))
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


def shots_table(results: Path, recs, out_macros: list) -> str:
    by = {(k, r, ro): rec for (k, r, ro, _), rec in zip(CELLS, recs)}
    pairs = {kind: (load(results, EXACT[kind]), by[(kind, "xxz", "ZZ")]) for kind in ("window", "recurrent")}
    for kind, (ex, sh) in pairs.items():
        assert ex["comparison"].get("shots") is None and sh["comparison"]["shots"] == 12288
    rows = []
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
        K = kind.capitalize()
        a, b = mean_sd(ex, "qrc")[0], mean_sd(sh, "qrc")[0]
        out_macros += [macro(f"exactQrc{K}", num(a)), macro(f"shotsQrc{K}", num(b)),
                       macro(f"shotLoss{K}", num(b - a))]
        s = ex["stats"]["qrc_vs_random_features"]
        out_macros += [macro(f"exactWinsRf{K}", f"{s['wins_a']}"),
                       macro(f"exactClusterWinsRf{K}", f"{s['cluster_wins_a']}"),
                       macro(f"exactPRf{K}", pval(s["p_holm_cluster"]).strip("$"))]
        # the window exact run used a narrower random-feature range; record it
        sp = ex["models"]["random_features"]["tuning"]["search_space"]
        out_macros += [macro(f"exactRfScaleLow{K}", f"{sp['scale']['attributes']['low']:g}"),
                       macro(f"exactRfBiasHigh{K}", f"{sp['bias_scale']['attributes']['high']:g}")]
    return "\n".join([
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r" & \multicolumn{3}{c}{window, XXZ, Z/ZZ} & \multicolumn{3}{c}{recurrent, XXZ, Z/ZZ} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Model & exact & 12288 shots & change & exact & 12288 shots & change \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""])


def refine_table(rec, out_macros: list) -> str:
    rows = []
    for m in ["qrc", "random_features", "poly2", "linear"]:
        t = rec["models"][m]["tuning"]
        b = t["budget"]
        n_tpe = t["n_trials"] - b["refine_evals"]
        tpe = t["trial_values"][:n_tpe]
        best = min(tpe)
        assert abs(best - t["best_val"]) < 1e-12 and b["refine_gain"] == 0.0
        flat = sum(v <= best + FLAT_TOL for v in tpe) / n_tpe
        rows.append(f"{LABEL[m]} & {b['n_dims']} & {n_tpe} & {b['refine_evals']} & {best:.4f} & "
                    f"{min(t['trial_values'][n_tpe:]):.4f} & {b['refine_gain']:.1f} & {100 * flat:.0f}\\%" + r" \\")
        tag = {"qrc": "Qrc", "random_features": "Rf", "poly2": "Poly", "linear": "Lin"}[m]
        out_macros += [macro(f"flat{tag}", f"{100 * flat:.0f}"), macro(f"tpeTrials{tag}", str(n_tpe)),
                       macro(f"spsaEvals{tag}", str(b["refine_evals"]))]
    return "\n".join([
        r"\begin{tabular}{lccccccc}", r"\toprule",
        r"Model & knobs & TPE trials & SPSA evals & best TPE val. & best SPSA val. & gain & "
        rf"within {FLAT_TOL} \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""])


def repro_table(results: Path) -> str:
    names = [c[3] for c in CELLS] + [EXACT["window"], ESN_NO_BIAS, EXACT["recurrent"], REFINE]
    rows = []
    for n in names:
        rec = load(results, n)
        env = rec["environment"]
        q = rec["models"]["qrc"]["tuning"]["study_seconds"]
        rows.append(rf"\texttt{{{n.replace('_', chr(92) + '_')}}} & \texttt{{{env['git_commit'][:7]}}} & "
                    f"{rec['total_seconds'] / 3600:.2f} & {q / 3600:.2f}" + r" \\")
    return "\n".join([
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Record & commit & total (h) & QRC tuning (h) \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}", ""])


def general_macros(results: Path, recs, out: list):
    by = {(k, r, ro): rec for (k, r, ro, _), rec in zip(CELLS, recs)}
    # headline: QRC vs random features in every cell
    rf_runs = [rec["stats"]["qrc_vs_random_features"]["wins_a"] for rec in recs]
    rf_clusters = [rec["stats"]["qrc_vs_random_features"]["cluster_wins_a"] for rec in recs]
    out += [macro("maxQrcRunWinsRf", str(max(rf_runs))), macro("maxQrcSeedWinsRf", str(max(rf_clusters)))]
    pmax = max(rec["stats"]["qrc_vs_random_features"]["p_holm_cluster"] for rec in recs)
    out.append(macro("maxPRf", pval(pmax).strip("$")))
    diffs = [rec["stats"]["qrc_vs_random_features"]["mean_diff"] for rec in recs]
    out += [macro("minGapRf", num(min(diffs))), macro("maxGapRf", num(max(diffs)))]
    # best QRC cell vs Poly2
    best = by[("window", "ising", "ZX")]
    s = best["stats"]["qrc_vs_poly2"]
    out += [macro("bestQrc", num(mean_sd(best, "qrc")[0])), macro("bestPoly", num(mean_sd(best, "poly2")[0])),
            macro("bestRf", num(mean_sd(best, "random_features")[0])),
            macro("bestPolyRunWins", f"{s['wins_a']}"), macro("bestPolyRunLosses", f"{s['wins_b']}"),
            macro("bestPolyTies", f"{s['ties']}"), macro("bestPolySeedWins", f"{s['cluster_wins_a']}"),
            macro("bestPolyP", pval(s["p_holm_cluster"]).strip("$")),
            macro("bestPolyCiLo", num(s["ci95_cluster"][0], 4)), macro("bestPolyCiHi", num(s["ci95_cluster"][1], 4))]
    # ESN margin in the recurrent kind
    esn = [by[("recurrent", r, ro)]["stats"]["qrc_vs_esn"]["mean_diff"] for r in ("xxz", "ising") for ro in ("ZZ", "ZX")]
    out += [macro("minGapEsn", num(min(esn))), macro("maxGapEsn", num(max(esn)))]
    out += [macro("esnZZ", num(mean_sd(by[("recurrent", "xxz", "ZZ")], "esn")[0])),
            macro("esnZX", num(mean_sd(by[("recurrent", "xxz", "ZX")], "esn")[0]))]
    # pinned reservoir knobs and their ranges
    def bp(cell):
        return by[cell]["models"]["qrc"]["tuning"]["best_params"]
    def space(cell, k):
        return by[cell]["models"]["qrc"]["tuning"]["search_space"][k]["attributes"]
    out += [macro("vWinIsingZX", f"{bp(('window', 'ising', 'ZX'))['reservoir']['v']:.2f}"),
            macro("vRecIsingZX", f"{bp(('recurrent', 'ising', 'ZX'))['reservoir']['v']:.2f}"),
            macro("WRecXxzZX", f"{bp(('recurrent', 'xxz', 'ZX'))['reservoir']['W']:.2f}"),
            macro("vHigh", f"{space(('window', 'ising', 'ZX'), 'v')['high']:g}"),
            macro("WHigh", f"{space(('recurrent', 'xxz', 'ZX'), 'W')['high']:g}")]
    esn_bp = [by[("recurrent", r, ro)]["models"]["esn"]["tuning"]["best_params"]
              for r in ("xxz", "ising") for ro in ("ZZ", "ZX")]
    esn_sp = by[("recurrent", "xxz", "ZZ")]["models"]["esn"]["tuning"]["search_space"]
    out += [macro("esnSrMax", f"{max(p['sr'] for p in esn_bp):.3f}"),
            macro("esnSrLow", f"{esn_sp['sr']['attributes']['low']:g}"),
            macro("esnBiasMin", f"{min(p['bias_scale'] for p in esn_bp):.2f}"),
            macro("esnBiasHigh", f"{esn_sp['bias_scale']['attributes']['high']:g}")]
    # recurrent feedback gains, shots
    kfb = [by[("recurrent", r, ro)]["models"]["qrc"]["tuning"]["best_params"]["k_fb"]
           for r in ("xxz", "ising") for ro in ("ZZ", "ZX")]
    out += [macro("kfbMin", f"{min(kfb):.2f}"), macro("kfbMax", f"{max(kfb):.2f}")]
    c = recs[0]["comparison"]
    out += [macro("shots", str(c["shots"])), macro("shotsPerBasisZX", str(by[("window", "xxz", "ZX")]["comparison"]["shots"] // 2)),
            macro("nQubits", str(c["q"])), macro("widthZZ", str(c["width"])),
            macro("widthZX", str(by[("window", "xxz", "ZX")]["comparison"]["width"])),
            macro("widthPoly", str(recs[0]["models"]["poly2"]["eval"][0]["width"])),
            macro("widthLin", str(recs[0]["models"]["linear"]["eval"][0]["width"]))]
    # restarting-protocol replay cost of one recurrent series
    T = by[("recurrent", "xxz", "ZZ")]["protocol"]["n_steps"]
    drives = T * (T + 1) // 2
    out += [macro("recSteps", str(T)), macro("replayDrives", f"{drives / 1e6:.1f}\\times10^{{6}}")]
    # ESN without the bias knob (first exact recurrent run)
    nb = load(results, ESN_NO_BIAS)
    assert "bias_scale" not in nb["models"]["esn"]["tuning"]["best_params"]
    v2 = load(results, EXACT["recurrent"])
    s = nb["stats"]["qrc_vs_esn"]
    out += [macro("noBiasEsn", num(mean_sd(nb, "esn")[0])), macro("noBiasQrc", num(mean_sd(nb, "qrc")[0])),
            macro("noBiasQrcWins", str(s["wins_a"])), macro("biasEsn", num(mean_sd(v2, "esn")[0]))]
    assert v2["models"]["qrc"]["tuning"]["best_params"] == nb["models"]["qrc"]["tuning"]["best_params"]
    out += [macro("vTwoQrcStudySeconds", f"{v2['models']['qrc']['tuning']['study_seconds']:.2f}"),
            macro("noBiasQrcStudyHours", f"{nb['models']['qrc']['tuning']['study_seconds'] / 3600:.2f}")]
    # basis change: gaps to random features, gain ratios, and check it is the largest one-factor QRC change
    q = {k: mean_sd(rec, "qrc")[0] for k, rec in by.items()}
    rf = {k: mean_sd(rec, "random_features")[0] for k, rec in by.items()}
    for kind in ("window", "recurrent"):
        for res in ("xxz", "ising"):
            for ro in ("ZZ", "ZX"):
                out.append(macro(f"gapRf{kind.capitalize()}{RES_TEX[res].capitalize()}{ro}",
                                 num(q[(kind, res, ro)] - rf[(kind, res, ro)])))
    ratios = {kind: [(q[(kind, r, "ZZ")] - q[(kind, r, "ZX")]) / (rf[(kind, r, "ZZ")] - rf[(kind, r, "ZX")])
                     for r in ("xxz", "ising")] for kind in ("window", "recurrent")}
    out += [macro("ratioWinMin", f"{min(ratios['window']):.1f}"), macro("ratioWinMax", f"{max(ratios['window']):.1f}"),
            macro("ratioRecMin", f"{100 / max(ratios['recurrent']):.0f}"),
            macro("ratioRecMax", f"{100 / min(ratios['recurrent']):.0f}")]
    factors = {"kind": ("window", "recurrent"), "res": ("xxz", "ising"), "ro": ("ZZ", "ZX")}
    best_change = {}
    for i, f in enumerate(factors):
        for k in by:
            if k[i] == factors[f][0]:
                k2 = list(k)
                k2[i] = factors[f][1]
                best_change[(f, k)] = abs(q[k] - q[tuple(k2)])
    top = max(best_change, key=best_change.get)
    assert top[0] == "ro", top
    sds = [mean_sd(rec, "qrc")[1] for rec in recs]
    out += [macro("qrcSdMin", num(min(sds))), macro("qrcSdMax", num(max(sds)))]
    ex_w = load(results, EXACT["window"])
    out += [macro("exactRfWindow", num(mean_sd(ex_w, "random_features")[0])),
            macro("shotsRfWindow", num(mean_sd(by[("window", "xxz", "ZZ")], "random_features")[0]))]
    # TPE trial counts per model over the eight cells, and the stopping rule
    for m, tag in (("qrc", "Qrc"), ("random_features", "Rf"), ("esn", "Esn")):
        n = [rec["models"][m]["tuning"]["n_trials"] for rec in recs if m in rec["models"]]
        out += [macro(f"trialsMin{tag}", str(min(n))), macro(f"trialsMax{tag}", str(max(n)))]
    assert all(t["tuning"]["budget"]["stopped"] == "patience" for rec in recs for t in rec["models"].values())
    # wall time
    names = [c[3] for c in CELLS] + [EXACT["window"], ESN_NO_BIAS, EXACT["recurrent"], REFINE]
    tot = sum(load(results, n)["total_seconds"] for n in names)
    out.append(macro("totalHours", f"{tot / 3600:.1f}"))
    out.append(macro("gpu", nb["environment"]["gpu"].replace("NVIDIA GeForce ", "")))


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
    (out / "cells_nrmse.tex").write_text(cells_nrmse(recs))
    (out / "cells_stats.tex").unlink(missing_ok=True)
    (out / "cells_wins.tex").write_text(cells_stats(recs, "wins"))
    (out / "cells_p.tex").write_text(cells_stats(recs, "p"))
    (out / "basis_gain.tex").write_text(basis_gain(recs, macros))
    (out / "shots.tex").write_text(shots_table(results, recs, macros))
    (out / "refine.tex").write_text(refine_table(load(results, REFINE), macros))
    (out / "repro.tex").write_text(repro_table(results))
    general_macros(results, recs, macros)
    frs = [load_freerun(results, c[3], rec) for c, rec in zip(CELLS, recs)]
    (out / "vpt.tex").write_text(vpt_table(frs))
    (out / "vpt_stats.tex").unlink(missing_ok=True)
    (out / "vpt_wins.tex").write_text(cells_stats(frs, "wins"))
    (out / "vpt_p.tex").write_text(cells_stats(frs, "p"))
    review_macros(results, recs, frs, macros)
    vpt_macros(frs, macros)
    commits = {fr["environment"]["git_commit"][:7] for fr in frs}
    assert len(commits) == 1
    macros += [macro("freerunCommit", commits.pop()),
               macro("freerunHours", f"{sum(fr['total_seconds'] for fr in frs) / 3600:.2f}")]
    (out / "numbers.tex").write_text("% generated by make_tables.py; do not edit\n" + "".join(macros))
    print("wrote", sorted(p.name for p in out.glob("*.tex")))


if __name__ == "__main__":
    main()
