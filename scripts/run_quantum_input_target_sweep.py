#!/usr/bin/env python3
"""
run_quantum_input_target_sweep.py - Sweep across many target
Hamiltonians to test the hypothesis: QRC wins on thermal/chaotic
targets, loses on integrable/localized targets.

Spans the thermal -> MBL transition (Heisenberg disorder strength)
and the integrable -> generic transition (XXZ delta).  All cells at
q=5 reservoir, 3 seeds for speed.

Configs (25 cells total):
  - Heisenberg, seeds {42, 100, 200}, disorder in {0.1, 0.5, 1.0, 2.0, 4.0}
       = 15 cells (clean -> strong-MBL)
  - XXZ delta in {0.0, 0.5, 1.0, 1.5, 2.0}, seeds {42, 100}
       = 10 cells (XY -> Heisenberg -> Ising-like)

Calls the existing experiment script per config; outputs land in
results/quantum_input_sweep/<config_tag>/summary.json.  Run
target_feature_analysis.py afterwards to aggregate and plot
features-vs-gap.

Expected wall-clock: ~12-15 min (q=5 with 3 seeds).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
SCRIPT = ROOT / "scripts" / "run_quantum_input_experiment.py"
PY = ROOT / "venv" / "Scripts" / "python.exe"

# ---------------------------------------------------------------------------
# Sweep configs
# ---------------------------------------------------------------------------

def build_configs():
    configs = []
    # Heisenberg disorder sweep: 3 seeds x 5 disorder strengths
    for seed in [42, 100, 200]:
        for dis in [0.1, 0.5, 1.0, 2.0, 4.0]:
            tag = f"heis_s{seed}_d{dis:g}"
            configs.append({
                "tag": tag,
                "args": ["--target-system", "heisenberg",
                          "--target-seed", str(seed),
                          "--target-disorder", str(dis)],
            })
    # XXZ delta sweep: 2 seeds x 5 deltas
    for seed in [42, 100]:
        for dlt in [0.0, 0.5, 1.0, 1.5, 2.0]:
            tag = f"xxz_s{seed}_d{dlt:g}"
            configs.append({
                "tag": tag,
                "args": ["--target-system", "xxz",
                          "--target-seed", str(seed),
                          "--xxz-delta", str(dlt),
                          "--target-disorder", "0.5"],
            })
    return configs


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-qubits", type=int, default=5)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--out-root", type=str,
                    default=str(ROOT / "results" / "quantum_input_sweep"))
    args = p.parse_args()

    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    configs = build_configs()
    print(f"=== Sweep over {len(configs)} target Hamiltonians ===")
    t_start = time.time()
    log = []
    for i, cfg in enumerate(configs):
        out_dir = out_root / cfg["tag"]
        if (out_dir / "summary.json").exists():
            print(f"[{i+1}/{len(configs)}] {cfg['tag']}  (cached, skip)")
            log.append({"tag": cfg["tag"], "cached": True})
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(PY), str(SCRIPT),
            "--n-qubits", str(args.n_qubits),
            "--n-seeds", str(args.n_seeds),
            "--n-steps", str(args.n_steps),
            "--out-dir", str(out_dir),
        ] + cfg["args"]
        t0 = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        ok = proc.returncode == 0 and (out_dir / "summary.json").exists()
        status = "OK" if ok else "FAIL"
        print(f"[{i+1}/{len(configs)}] {cfg['tag']:<24}  {status:<4}  {dt:>5.1f}s")
        if not ok:
            print(f"  stderr (last 5 lines):\n{chr(10).join(proc.stderr.splitlines()[-5:])}")
        log.append({"tag": cfg["tag"], "ok": ok, "dt_s": dt})

    total = time.time() - t_start
    with open(out_root / "sweep_log.json", "w") as f:
        json.dump({"total_s": total, "n_configs": len(configs), "log": log}, f, indent=2)
    print(f"\nDone in {total/60:.1f} min.  Log saved to {out_root / 'sweep_log.json'}.")
    print(f"\nNext: run scripts/target_feature_analysis.py --dirs "
           f"{' '.join(str(out_root/cfg['tag']) for cfg in configs)}")


if __name__ == "__main__":
    main()
