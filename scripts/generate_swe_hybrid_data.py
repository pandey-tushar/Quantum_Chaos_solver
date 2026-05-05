#!/usr/bin/env python3
"""
generate_swe_hybrid_data.py - Paired (truth, physics) SWE trajectories for Block B.

For each seed we run two SWE simulations from the *same* initial condition:
  - Truth:    nu = nu_truth      (lighter dissipation; resolves nonlinear mixing)
  - Physics:  nu = nu_phys (>>)  (over-damped; loses fine-scale features)

The "physics" run is the model the QRC will correct. The "truth" run is the
target. Their difference at each step is the *subgrid residual* the QRC must
learn -- mimicking the NeuralGCM-style story where a coarse physics core is
augmented by a learned correction.

We also save a longer reference truth trajectory (single seed) used to fit
the POD basis on the resolved dynamics.

Output (data/swe/swe_hybrid_data.npz):
    ref_truth_snaps:   (n_t_ref, 3, Nx, Ny)         POD-fitting source
    seed_truth_snaps:  (n_seeds, n_t_seed, 3, Nx, Ny)
    seed_phys_snaps:   (n_seeds, n_t_seed, 3, Nx, Ny)
    seeds, dt_save, grid metadata, physics constants.

Run:
    python scripts/generate_swe_hybrid_data.py --n-seeds 8 \
        --ref-t-hours 400 --seed-t-hours 600 --save-every-hours 2.0 \
        --nu-truth 1e7 --nu-phys 5e8
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
    total_energy,
)


def main():
    p = argparse.ArgumentParser(description="Generate paired SWE hybrid data.")
    p.add_argument("--nx", type=int, default=48)
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--lx", type=float, default=1.0e6)
    p.add_argument("--ly", type=float, default=1.0e6)
    p.add_argument("--g",  type=float, default=9.81)
    p.add_argument("--f0", type=float, default=1.0e-4)
    p.add_argument("--H",  type=float, default=100.0)
    p.add_argument("--nu-truth", type=float, default=1.0e7,
                   help="Hyperviscosity for truth run (m^4/s)")
    p.add_argument("--nu-phys",  type=float, default=5.0e8,
                   help="Hyperviscosity for physics-only run")
    p.add_argument("--linear-phys", action="store_true",
                   help="Use linearised SWE for physics run (drops nonlinear "
                        "advection; substantially crude vs full nonlinear truth)")
    p.add_argument("--amp", type=float, default=3.0)
    p.add_argument("--k-max", type=int, default=4)

    p.add_argument("--ref-t-hours", type=float, default=400.0)
    p.add_argument("--ref-seed", type=int, default=999)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--seed-t-hours", type=float, default=600.0)
    p.add_argument("--save-every-hours", type=float, default=2.0)

    p.add_argument("--cfl-safety", type=float, default=0.3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    grid = make_grid(Nx=args.nx, Ny=args.ny, Lx=args.lx, Ly=args.ly)
    dt = cfl_dt(grid, g=args.g, H=args.H, safety=args.cfl_safety)
    save_every = max(1, int(round(args.save_every_hours * 3600.0 / dt)))
    print(f"Grid: {args.nx}x{args.ny}, dt={dt:.2f}s, save_every={save_every} "
          f"({save_every*dt/3600:.2f}h)")
    print(f"Truth nu = {args.nu_truth:.1e}, Physics nu = {args.nu_phys:.1e} "
          f"(ratio {args.nu_phys/args.nu_truth:.0f}x)")

    rhs_truth = dict(g=args.g, f0=args.f0, H=args.H, nu=args.nu_truth, linear=False)
    rhs_phys  = dict(g=args.g, f0=args.f0, H=args.H, nu=args.nu_phys,
                     linear=args.linear_phys)
    print(f"  physics linearised = {args.linear_phys}")

    # ---- Reference truth trajectory (for POD fitting) ----
    ref_n = int(round(args.ref_t_hours * 3600.0 / dt))
    print(f"\n[Reference truth] seed={args.ref_seed}, "
          f"{args.ref_t_hours:.0f}h = {ref_n} steps")
    s0 = random_balanced_ic(grid, H=args.H, amp=args.amp, k_max=args.k_max,
                             f0=args.f0, g=args.g, seed=args.ref_seed)
    t0 = time.time()
    ref_t, ref_snaps = integrate_swe(s0, grid, dt=dt, n_steps=ref_n,
                                       save_every=save_every, **rhs_truth)
    print(f"  done in {time.time()-t0:.1f}s, {ref_snaps.shape[0]} snaps, "
          f"E_drift={(total_energy(ref_snaps[-1], grid, g=args.g, H_ref=args.H)/total_energy(ref_snaps[0], grid, g=args.g, H_ref=args.H)-1)*100:+.2f}%")

    # ---- Per-seed paired trajectories ----
    seed_n = int(round(args.seed_t_hours * 3600.0 / dt))
    print(f"\n[Per-seed paired] {args.n_seeds} seeds x ({args.seed_t_hours:.0f}h truth + same physics)"
          f", {seed_n} steps each, ~{seed_n//save_every + 1} snaps")

    truth_all, phys_all = [], []
    seed_times = None
    for s in range(args.n_seeds):
        s_ic = random_balanced_ic(grid, H=args.H, amp=args.amp, k_max=args.k_max,
                                    f0=args.f0, g=args.g, seed=s)
        t0 = time.time()
        seed_t, truth_sn = integrate_swe(s_ic, grid, dt=dt, n_steps=seed_n,
                                           save_every=save_every, **rhs_truth)
        truth_dt = time.time() - t0

        t0 = time.time()
        _,      phys_sn  = integrate_swe(s_ic, grid, dt=dt, n_steps=seed_n,
                                           save_every=save_every, **rhs_phys)
        phys_dt = time.time() - t0

        if not (np.all(np.isfinite(truth_sn)) and np.all(np.isfinite(phys_sn))):
            raise RuntimeError(f"seed {s}: NaN/Inf detected")
        truth_all.append(truth_sn); phys_all.append(phys_sn)
        seed_times = seed_t
        # Track divergence to confirm physics differs meaningfully from truth.
        diff = np.linalg.norm(truth_sn[-1] - phys_sn[-1]) / (np.linalg.norm(truth_sn[-1]) + 1e-12)
        print(f"  seed {s}: truth {truth_dt:.1f}s, phys {phys_dt:.1f}s, "
              f"final-step relative truth-vs-phys diff = {diff*100:.1f}%")

    truth_all = np.stack(truth_all, axis=0)
    phys_all  = np.stack(phys_all,  axis=0)

    out_path = Path(args.out) if args.out else (ROOT / "data" / "swe" / "swe_hybrid_data.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ref_truth_snaps=ref_snaps.astype(np.float32),
        ref_truth_times=ref_t,
        seed_truth_snaps=truth_all.astype(np.float32),
        seed_phys_snaps=phys_all.astype(np.float32),
        seed_times=seed_times,
        seeds=np.arange(args.n_seeds),
        Nx=args.nx, Ny=args.ny, Lx=args.lx, Ly=args.ly,
        dx=grid["dx"], dy=grid["dy"],
        g=args.g, f0=args.f0, H=args.H,
        nu_truth=args.nu_truth, nu_phys=args.nu_phys,
        linear_phys=int(args.linear_phys),
        dt=dt, dt_save=save_every * dt,
    )
    print(f"\nSaved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"  ref_truth_snaps: {ref_snaps.shape}")
    print(f"  seed_truth_snaps: {truth_all.shape}")
    print(f"  seed_phys_snaps:  {phys_all.shape}")


if __name__ == "__main__":
    main()
