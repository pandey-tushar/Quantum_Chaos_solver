#!/usr/bin/env python3
"""
generate_swe_data.py - Produce SWE reference trajectory + per-seed trajectories.

Output structure (saved as a single .npz under data/swe/):
    grid_meta:   dict-like (Nx, Ny, Lx, Ly, dx, dy)
    physics:     dict (g, f0, H, nu)
    dt_save:     time between saved snapshots (s)
    ref_snaps:   (n_t, 3, Nx, Ny) -- long reference trajectory used to fit POD
    seed_snaps:  (n_seeds, n_t_seed, 3, Nx, Ny) -- per-seed trajectories for QRC train/test
    seeds:       (n_seeds,)

Run:
    python scripts/generate_swe_data.py --nx 64 --n-seeds 5 --t-end 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from swe_solver import (
    make_grid, integrate_swe, random_balanced_ic, cfl_dt,
    total_energy, total_mass,
)


def main():
    p = argparse.ArgumentParser(description="Generate SWE training/eval data.")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--lx", type=float, default=1.0e6)
    p.add_argument("--ly", type=float, default=1.0e6)
    p.add_argument("--g",  type=float, default=9.81)
    p.add_argument("--f0", type=float, default=1.0e-4)
    p.add_argument("--H",  type=float, default=100.0)
    p.add_argument("--nu", type=float, default=1.0e7,
                   help="Biharmonic hyperviscosity (m^4/s)")
    p.add_argument("--amp", type=float, default=2.0,
                   help="Amplitude of random IC h-perturbation")
    p.add_argument("--k-max", type=int, default=4,
                   help="Max wavenumber in random IC")

    # Reference run (used only to fit POD modes)
    p.add_argument("--ref-t-hours", type=float, default=200.0,
                   help="Reference run length in simulated hours")
    p.add_argument("--ref-seed", type=int, default=999,
                   help="Seed for the reference run; kept distinct from per-seed runs")

    # Per-seed runs (used for QRC train/test)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--seed-t-hours", type=float, default=80.0)
    p.add_argument("--save-every-hours", type=float, default=0.5,
                   help="Save snapshot every N hours of simulation time")

    p.add_argument("--cfl-safety", type=float, default=0.3)
    p.add_argument("--out", type=str, default=None,
                   help="Output .npz path (default: data/swe/swe_data.npz)")
    args = p.parse_args()

    grid = make_grid(Nx=args.nx, Ny=args.ny, Lx=args.lx, Ly=args.ly)
    dt = cfl_dt(grid, g=args.g, H=args.H, safety=args.cfl_safety)
    dt_save = args.save_every_hours * 3600.0
    save_every = max(1, int(round(dt_save / dt)))

    print(f"Grid: {args.nx}x{args.ny}, dx={grid['dx']:.0f} m, dy={grid['dy']:.0f} m")
    print(f"dt = {dt:.2f} s ({args.cfl_safety*100:.0f}% CFL); save_every = {save_every} -> "
          f"dt_save = {save_every*dt/3600:.2f} h (target {args.save_every_hours} h)")
    print(f"Physics: g={args.g}, f0={args.f0}, H={args.H}, nu={args.nu:.1e}")

    rhs_kwargs = dict(g=args.g, f0=args.f0, H=args.H, nu=args.nu)

    # --- Reference run --------------------------------------------------------
    ref_seconds = args.ref_t_hours * 3600.0
    ref_n_steps = int(round(ref_seconds / dt))
    print(f"\n[Reference] seed={args.ref_seed}, "
          f"{args.ref_t_hours:.1f} h = {ref_n_steps} steps, "
          f"~{ref_n_steps // save_every + 1} snapshots")

    state0 = random_balanced_ic(grid, H=args.H, amp=args.amp, k_max=args.k_max,
                                f0=args.f0, g=args.g, seed=args.ref_seed)
    t0 = time.time()
    ref_times, ref_snaps = integrate_swe(state0, grid, dt=dt, n_steps=ref_n_steps,
                                          save_every=save_every, **rhs_kwargs)
    print(f"  ran in {time.time()-t0:.1f}s; "
          f"{ref_snaps.shape[0]} snapshots, "
          f"E_drift={(total_energy(ref_snaps[-1], grid, g=args.g, H_ref=args.H) /
                     total_energy(ref_snaps[0], grid, g=args.g, H_ref=args.H) - 1)*100:+.2f}%, "
          f"finite={np.all(np.isfinite(ref_snaps))}")

    # --- Per-seed runs --------------------------------------------------------
    seed_seconds = args.seed_t_hours * 3600.0
    seed_n_steps = int(round(seed_seconds / dt))
    print(f"\n[Per-seed] {args.n_seeds} seeds, "
          f"{args.seed_t_hours:.1f} h each = {seed_n_steps} steps, "
          f"~{seed_n_steps // save_every + 1} snapshots/seed")

    seeds = list(range(args.n_seeds))
    seed_snaps_all = []
    seed_times = None
    for s in seeds:
        state0 = random_balanced_ic(grid, H=args.H, amp=args.amp, k_max=args.k_max,
                                    f0=args.f0, g=args.g, seed=s)
        t0 = time.time()
        seed_t, seed_sn = integrate_swe(state0, grid, dt=dt, n_steps=seed_n_steps,
                                          save_every=save_every, **rhs_kwargs)
        elapsed = time.time() - t0
        if not np.all(np.isfinite(seed_sn)):
            raise RuntimeError(f"seed {s}: NaN/Inf detected; reduce CFL or amp")
        seed_snaps_all.append(seed_sn)
        seed_times = seed_t
        print(f"  seed {s}: {elapsed:.1f}s, {seed_sn.shape[0]} snapshots, "
              f"final |u|max={np.abs(seed_sn[-1, 0]).max():.2f}")

    seed_snaps_all = np.stack(seed_snaps_all, axis=0)        # (n_seeds, n_t, 3, Nx, Ny)

    # --- Save -----------------------------------------------------------------
    out_path = Path(args.out) if args.out else (ROOT / "data" / "swe" / "swe_data.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        ref_snaps=ref_snaps.astype(np.float32),
        ref_times=ref_times,
        seed_snaps=seed_snaps_all.astype(np.float32),
        seed_times=seed_times,
        seeds=np.array(seeds),
        # grid descriptor as plain arrays so npz is happy
        Nx=args.nx, Ny=args.ny, Lx=args.lx, Ly=args.ly,
        dx=grid["dx"], dy=grid["dy"],
        g=args.g, f0=args.f0, H=args.H, nu=args.nu,
        dt=dt, dt_save=save_every * dt,
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {out_path}  ({size_mb:.1f} MB)")
    print(f"  ref_snaps: {ref_snaps.shape}  (POD training)")
    print(f"  seed_snaps: {seed_snaps_all.shape}  (QRC train/test)")


if __name__ == "__main__":
    main()
