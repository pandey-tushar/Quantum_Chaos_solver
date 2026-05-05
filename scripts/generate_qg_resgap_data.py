#!/usr/bin/env python3
"""
generate_qg_resgap_data.py - QG truth/physics paired data with a
*resolution* gap (rather than linearisation gap).

Truth: full nonlinear QG integrated at high resolution N_truth x N_truth.
Physics: full nonlinear QG integrated at low resolution N_phys x N_phys.
The truth is then spectrally low-pass filtered and subsampled onto the
low-resolution grid for storage and downstream POD. The "residual"
between truth_coarse and physics is the contribution of unresolved
scales to the resolved (large-scale) evolution -- the classical
subgrid closure target. This is the gap the atmospheric-community
treats as canonical (NeuralGCM, see also Pathak et al. spectral
truncation work).

In contrast to generate_qg_hybrid_data.py (which used a linearisation
gap), this gap is genuinely nonlinear because the resolved-scale
evolution depends on the unresolved-scale dynamics through nonlinear
advection terms. It is not well-approximated by a linear function of
the resolved state, so a learned correction with nonlinear features
should in principle have something to do.

Output (data/qg/qg_resgap_data.npz):
    ref_truth_snaps:   (n_t_ref, 2, N_phys, N_phys)   coarse-grained truth
    seed_truth_snaps:  (n_seeds, n_t_seed, 2, N_phys, N_phys)
    seed_phys_snaps:   (n_seeds, n_t_seed, 2, N_phys, N_phys)
    seeds, dt_save, grid metadata, gap_type='resolution'
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
    kinetic_energy,
)


# ---------------------------------------------------------------------------
# Spectral coarse-graining
# ---------------------------------------------------------------------------

def _coarse_grain(field: np.ndarray, n_low: int) -> np.ndarray:
    """Spectral low-pass filter + uniform subsample, doubly periodic.

    For an input field on an Nx_h x Ny_h grid (with Nx_h = Ny_h, both
    divisible by n_low), zero out spectral content with |kx|, |ky| above
    the low-resolution Nyquist, IFFT to a smoothed high-res field, then
    uniform-subsample onto the n_low x n_low grid.

    Field values are preserved (no energy rescaling) so the coarse-
    grained field is the same physical field band-limited to the
    low-res spectrum.
    """
    Nx_h = field.shape[0]
    if Nx_h % n_low != 0:
        raise ValueError(f"high resolution {Nx_h} must be a multiple of {n_low}")
    f_hat = np.fft.fft2(field)
    kx = np.fft.fftfreq(Nx_h, d=1.0)
    KX, KY = np.meshgrid(kx, kx, indexing="ij")
    k_cutoff = (n_low / 2.0) / Nx_h - 1e-9
    mask = (np.abs(KX) < k_cutoff) & (np.abs(KY) < k_cutoff)
    f_lp = np.real(np.fft.ifft2(f_hat * mask))
    stride = Nx_h // n_low
    return f_lp[::stride, ::stride]


def _stack_q_psi_at_low_res(q_low: np.ndarray, grid_low) -> np.ndarray:
    """For a single low-res vorticity field, return (q, psi) stacked."""
    K2 = grid_low["K2"]
    psi_hat = -np.fft.fft2(q_low) / K2
    psi_hat[0, 0] = 0.0
    psi_low = np.real(np.fft.ifft2(psi_hat))
    return np.stack([q_low, psi_low], axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Generate QG paired data with a resolution gap.")
    p.add_argument("--n-truth", type=int, default=128, help="Truth grid size")
    p.add_argument("--n-phys",  type=int, default=64,  help="Physics (and storage) grid size")
    p.add_argument("--lx", type=float, default=1.0e6)
    p.add_argument("--ly", type=float, default=1.0e6)
    p.add_argument("--beta", type=float, default=1.6e-11)
    p.add_argument("--r-drag", type=float, default=0.0)
    p.add_argument("--nu", type=float, default=1.0e16)
    p.add_argument("--p-hyper", type=int, default=4)
    p.add_argument("--amp", type=float, default=1.0e-5)
    p.add_argument("--k-peak", type=int, default=6)
    p.add_argument("--k-width", type=int, default=2)

    p.add_argument("--ref-t-days", type=float, default=20.0)
    p.add_argument("--ref-seed", type=int, default=999)
    p.add_argument("--n-seeds", type=int, default=8)
    p.add_argument("--seed-t-days", type=float, default=12.0)
    p.add_argument("--save-every-hours", type=float, default=3.0)

    p.add_argument("--cfl-safety", type=float, default=0.3)
    p.add_argument("--out", type=str, default=None)
    args = p.parse_args()

    if args.n_truth % args.n_phys != 0:
        raise SystemExit(f"--n-truth ({args.n_truth}) must be a multiple of --n-phys ({args.n_phys})")

    grid_truth = make_qg_grid(Nx=args.n_truth, Ny=args.n_truth, Lx=args.lx, Ly=args.ly)
    grid_phys  = make_qg_grid(Nx=args.n_phys,  Ny=args.n_phys,  Lx=args.lx, Ly=args.ly)

    # Estimate u_max for CFL on the high-res grid (governs both runs' dt)
    sample = random_vortex_ic(grid_truth, amp=args.amp, k_peak=args.k_peak,
                                k_width=args.k_width, seed=args.ref_seed)
    psi = vorticity_to_streamfunction(sample, grid_truth)
    u, v = velocity_from_psi(psi, grid_truth)
    u_max = float(np.sqrt(np.max(u ** 2 + v ** 2)))
    dt_truth = cfl_dt(grid_truth, u_max=max(u_max, 0.5), safety=args.cfl_safety)
    # Use the same dt for physics (low-res CFL is more permissive)
    dt = dt_truth
    save_every = max(1, int(round(args.save_every_hours * 3600.0 / dt)))

    print(f"Truth grid:   {args.n_truth}x{args.n_truth}, dx={grid_truth['dx']:.0f} m")
    print(f"Physics grid: {args.n_phys}x{args.n_phys},   dx={grid_phys['dx']:.0f} m")
    print(f"Sample IC u_max = {u_max:.2f} m/s")
    print(f"dt = {dt:.0f} s (high-res CFL); save_every = {save_every} -> "
          f"dt_save = {save_every*dt/3600:.2f} h")

    rhs_kwargs = dict(beta=args.beta, r=args.r_drag, nu=args.nu,
                       p_hyper=args.p_hyper, linear=False)

    # ---- Reference run ----
    ref_n = int(round(args.ref_t_days * 86400.0 / dt))
    print(f"\n[Reference truth] high-res, seed={args.ref_seed}, "
          f"{args.ref_t_days:.1f} days = {ref_n} steps")
    q0_h = random_vortex_ic(grid_truth, amp=args.amp, k_peak=args.k_peak,
                              k_width=args.k_width, seed=args.ref_seed)
    t0 = time.time()
    ref_t, ref_q_high = integrate_qg(q0_h, grid_truth, dt=dt, n_steps=ref_n,
                                        save_every=save_every, **rhs_kwargs)
    ref_dt = time.time() - t0
    # Coarse-grain each snapshot
    n_ref = ref_q_high.shape[0]
    ref_snaps = np.empty((n_ref, 2, args.n_phys, args.n_phys), dtype=np.float32)
    for i in range(n_ref):
        q_low = _coarse_grain(ref_q_high[i], args.n_phys)
        ref_snaps[i] = _stack_q_psi_at_low_res(q_low, grid_phys)
    print(f"  done in {ref_dt:.1f}s, {n_ref} snapshots, "
          f"finite={np.all(np.isfinite(ref_snaps))}, "
          f"high-res KE drift={(kinetic_energy(ref_q_high[-1], grid_truth)/kinetic_energy(ref_q_high[0], grid_truth)-1)*100:+.2f}%")

    # ---- Per-seed paired runs ----
    seed_n = int(round(args.seed_t_days * 86400.0 / dt))
    print(f"\n[Per-seed paired] {args.n_seeds} seeds, "
          f"{args.seed_t_days:.1f} days each = {seed_n} steps")

    truth_all, phys_all = [], []
    seed_times = None
    for s in range(args.n_seeds):
        # High-res IC (truth)
        q0_h = random_vortex_ic(grid_truth, amp=args.amp, k_peak=args.k_peak,
                                  k_width=args.k_width, seed=s)
        # Low-res IC (physics): coarse-grain the same high-res IC
        q0_l = _coarse_grain(q0_h, args.n_phys)

        t0 = time.time()
        _, truth_q_h = integrate_qg(q0_h, grid_truth, dt=dt, n_steps=seed_n,
                                       save_every=save_every, **rhs_kwargs)
        truth_dt = time.time() - t0

        t0 = time.time()
        seed_t, phys_q_l = integrate_qg(q0_l, grid_phys, dt=dt, n_steps=seed_n,
                                           save_every=save_every, **rhs_kwargs)
        phys_dt = time.time() - t0
        seed_times = seed_t

        # Coarse-grain truth onto low-res grid
        n_t = truth_q_h.shape[0]
        truth_low = np.empty((n_t, 2, args.n_phys, args.n_phys), dtype=np.float32)
        phys_low  = np.empty((n_t, 2, args.n_phys, args.n_phys), dtype=np.float32)
        for i in range(n_t):
            q_low_t = _coarse_grain(truth_q_h[i], args.n_phys)
            truth_low[i] = _stack_q_psi_at_low_res(q_low_t, grid_phys)
            phys_low[i]  = _stack_q_psi_at_low_res(phys_q_l[i],   grid_phys)

        if not (np.all(np.isfinite(truth_low)) and np.all(np.isfinite(phys_low))):
            raise RuntimeError(f"seed {s}: NaN/Inf detected")

        truth_all.append(truth_low); phys_all.append(phys_low)
        # Final-step truth-vs-phys diff
        diff_final = (np.linalg.norm(truth_low[-1, 0] - phys_low[-1, 0])
                        / (np.linalg.norm(truth_low[-1, 0]) + 1e-12))
        print(f"  seed {s}: truth {truth_dt:.1f}s, phys {phys_dt:.1f}s, "
              f"final-step relative truth-vs-phys (vorticity) diff = {diff_final*100:.1f}%")

    truth_all = np.stack(truth_all, axis=0)
    phys_all  = np.stack(phys_all,  axis=0)

    out_path = Path(args.out) if args.out else (ROOT / "data" / "qg" / "qg_resgap_data.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        ref_truth_snaps=ref_snaps,
        ref_truth_times=ref_t,
        seed_truth_snaps=truth_all.astype(np.float32),
        seed_phys_snaps=phys_all.astype(np.float32),
        seed_times=seed_times,
        seeds=np.arange(args.n_seeds),
        Nx=args.n_phys, Ny=args.n_phys, Lx=args.lx, Ly=args.ly,
        dx=grid_phys["dx"], dy=grid_phys["dy"],
        n_truth=args.n_truth, n_phys=args.n_phys,
        beta=args.beta, r_drag=args.r_drag, nu=args.nu,
        p_hyper=args.p_hyper,
        # The downstream pipeline reads `linear_phys` to build rhs_phys; we
        # set 0 here because the resolution-gap physics is fully nonlinear.
        linear_phys=0,
        gap_type="resolution",
        dt=dt, dt_save=save_every * dt,
    )
    print(f"\nSaved {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"  ref_truth_snaps: {ref_snaps.shape}")
    print(f"  seed_truth_snaps: {truth_all.shape}")
    print(f"  seed_phys_snaps:  {phys_all.shape}")


if __name__ == "__main__":
    main()
