#!/usr/bin/env python3
"""
generate_qg_hybrid_data.py - Paired (truth, physics) QG trajectories.

QG analogue of `generate_swe_hybrid_data.py`. For each seed we run two
single-layer barotropic QG simulations from the *same* initial vorticity
field:
  - Truth:    full nonlinear J(psi, q) + beta * d(psi)/dx + hyperviscosity
  - Physics:  linearised (Jacobian dropped) + beta * d(psi)/dx + hyperviscosity

The "physics" run is the deliberately-crude model the QRC will correct;
the "truth" run is the target. Their per-step difference is the
nonlinear-advection contribution -- exactly the kind of subgrid term a
NeuralGCM-style hybrid is designed to learn.

We save the streamfunction `psi` alongside the vorticity `q` so the
downstream POD basis has access to both physical fields (analogous to
(u, v, h) in the SWE setup). This produces 2-channel snapshots
(q, psi) of shape (n_t, 2, Nx, Ny) consumable by `PODReducer` without
modification.

Output (data/qg/qg_hybrid_data.npz):
    ref_truth_snaps:   (n_t_ref, 2, Nx, Ny)
    seed_truth_snaps:  (n_seeds, n_t_seed, 2, Nx, Ny)
    seed_phys_snaps:   (n_seeds, n_t_seed, 2, Nx, Ny)
    seeds, dt_save, grid metadata, physics constants.

Run:
    python scripts/generate_qg_hybrid_data.py --n-seeds 8 \\
        --ref-t-days 60 --seed-t-days 90 --save-every-hours 6.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qg_solver import (
    make_qg_grid, integrate_qg, random_vortex_ic, cfl_dt,
    vorticity_to_streamfunction, velocity_from_psi,
    enstrophy, kinetic_energy,
)


def _stack_q_psi(q_snaps: np.ndarray, grid) -> np.ndarray:
    """Stack vorticity + streamfunction so POD sees both fields."""
    n_t = q_snaps.shape[0]
    out = np.empty((n_t, 2) + q_snaps.shape[1:], dtype=q_snaps.dtype)
    for t in range(n_t):
        out[t, 0] = q_snaps[t]
        out[t, 1] = vorticity_to_streamfunction(q_snaps[t], grid)
    return out


def main():
    p = argparse.ArgumentParser(description="Generate paired QG hybrid data.")
    p.add_argument("--nx", type=int, default=64)
    p.add_argument("--ny", type=int, default=64)
    p.add_argument("--lx", type=float, default=1.0e6)
    p.add_argument("--ly", type=float, default=1.0e6)
    p.add_argument("--beta", type=float, default=1.6e-11,
                   help="beta-plane parameter (mid-latitude default)")
    p.add_argument("--r-drag", type=float, default=0.0,
                   help="Linear bottom drag rate (s^-1)")
    p.add_argument("--nu", type=float, default=1.0e16,
                   help="Hyperviscosity coefficient")
    p.add_argument("--p-hyper", type=int, default=4,
                   help="Hyperviscosity order (-Laplacian)^p")
    p.add_argument("--amp", type=float, default=1.0e-5,
                   help="RMS vorticity of initial condition")
    p.add_argument("--k-peak", type=int, default=6)
    p.add_argument("--k-width", type=int, default=2)

    p.add_argument("--ref-t-days", type=float, default=60.0)
    p.add_argument("--ref-seed", type=int, default=999)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--seed-t-days", type=float, default=90.0)
    p.add_argument("--save-every-hours", type=float, default=6.0)

    p.add_argument("--cfl-safety", type=float, default=0.3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    grid = make_qg_grid(Nx=args.nx, Ny=args.ny, Lx=args.lx, Ly=args.ly)

    # Estimate a representative u_max from a sample IC for CFL
    sample = random_vortex_ic(grid, amp=args.amp, k_peak=args.k_peak,
                                k_width=args.k_width, seed=args.ref_seed)
    psi = vorticity_to_streamfunction(sample, grid)
    u, v = velocity_from_psi(psi, grid)
    u_max = float(np.sqrt(np.max(u ** 2 + v ** 2)))
    dt = cfl_dt(grid, u_max=max(u_max, 0.5), safety=args.cfl_safety)
    save_every = max(1, int(round(args.save_every_hours * 3600.0 / dt)))

    print(f"Grid: {args.nx}x{args.ny}, dx={grid['dx']:.0f}m, dy={grid['dy']:.0f}m")
    print(f"Sample IC u_max = {u_max:.2f} m/s")
    print(f"dt = {dt:.0f}s ({args.cfl_safety*100:.0f}% advective CFL); "
          f"save_every = {save_every} -> dt_save = {save_every*dt/3600:.2f} h "
          f"(target {args.save_every_hours} h)")
    print(f"Physics: beta={args.beta:.1e}, r={args.r_drag}, "
          f"nu={args.nu:.1e}, p_hyper={args.p_hyper}")

    rhs_truth = dict(beta=args.beta, r=args.r_drag, nu=args.nu,
                       p_hyper=args.p_hyper, linear=False)
    rhs_phys  = dict(beta=args.beta, r=args.r_drag, nu=args.nu,
                       p_hyper=args.p_hyper, linear=True)

    # ---- Reference run for POD fitting ----
    ref_seconds = args.ref_t_days * 86400.0
    ref_n = int(round(ref_seconds / dt))
    print(f"\n[Reference truth] seed={args.ref_seed}, "
          f"{args.ref_t_days:.1f} days = {ref_n} steps")
    q0 = random_vortex_ic(grid, amp=args.amp, k_peak=args.k_peak,
                            k_width=args.k_width, seed=args.ref_seed)
    t0 = time.time()
    ref_t, ref_q = integrate_qg(q0, grid, dt=dt, n_steps=ref_n,
                                  save_every=save_every, **rhs_truth)
    ref_snaps = _stack_q_psi(ref_q, grid)
    print(f"  done in {time.time()-t0:.1f}s, {ref_snaps.shape[0]} snapshots, "
          f"finite={np.all(np.isfinite(ref_snaps))}, "
          f"KE drift = {(kinetic_energy(ref_q[-1], grid)/kinetic_energy(ref_q[0], grid)-1)*100:+.2f}%")

    # ---- Per-seed paired runs ----
    seed_seconds = args.seed_t_days * 86400.0
    seed_n = int(round(seed_seconds / dt))
    print(f"\n[Per-seed paired] {args.n_seeds} seeds x ({args.seed_t_days:.1f} days "
          f"truth + same physics), {seed_n} steps each, "
          f"~{seed_n // save_every + 1} snapshots/seed")

    truth_all, phys_all = [], []
    seed_times = None
    for s in range(args.n_seeds):
        ic = random_vortex_ic(grid, amp=args.amp, k_peak=args.k_peak,
                                k_width=args.k_width, seed=s)
        t0 = time.time()
        seed_t, q_truth = integrate_qg(ic, grid, dt=dt, n_steps=seed_n,
                                          save_every=save_every, **rhs_truth)
        truth_dt = time.time() - t0

        t0 = time.time()
        _, q_phys = integrate_qg(ic, grid, dt=dt, n_steps=seed_n,
                                    save_every=save_every, **rhs_phys)
        phys_dt = time.time() - t0

        if not (np.all(np.isfinite(q_truth)) and np.all(np.isfinite(q_phys))):
            raise RuntimeError(f"seed {s}: NaN/Inf detected")

        snap_truth = _stack_q_psi(q_truth, grid)
        snap_phys  = _stack_q_psi(q_phys, grid)
        truth_all.append(snap_truth); phys_all.append(snap_phys)
        seed_times = seed_t

        diff = (np.linalg.norm(q_truth[-1] - q_phys[-1])
                / (np.linalg.norm(q_truth[-1]) + 1e-12))
        print(f"  seed {s}: truth {truth_dt:.1f}s, phys {phys_dt:.1f}s, "
              f"final-step relative truth-vs-phys diff = {diff*100:.1f}%")

    truth_all = np.stack(truth_all, axis=0)
    phys_all  = np.stack(phys_all,  axis=0)

    out_path = Path(args.out) if args.out else (ROOT / "data" / "qg" / "qg_hybrid_data.npz")
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
        beta=args.beta, r_drag=args.r_drag, nu=args.nu,
        p_hyper=args.p_hyper,
        linear_phys=1,
        dt=dt, dt_save=save_every * dt,
    )
    print(f"\nSaved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"  ref_truth_snaps: {ref_snaps.shape}")
    print(f"  seed_truth_snaps: {truth_all.shape}")
    print(f"  seed_phys_snaps:  {phys_all.shape}")


if __name__ == "__main__":
    main()
