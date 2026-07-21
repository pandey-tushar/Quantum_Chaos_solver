#!/usr/bin/env python3
"""
r1_experiments.py -- Journal-revision (R1) experiment harness for the
4-qubit / 3-layer statevector QPINN (arXiv paper 1, qst_submission).

Single-file harness in the repo style: argparse subcommands, correctness
gates (self-test), JSON outputs with the full config + provenance (git rev,
git-dirty flag, source/baseline SHA256) + result SHA256 printed to stdout,
and a picklable multiprocessing runner.

SHORT RUNS: the self-test finishes in seconds. For the train/jacobian modes
a projected wall time is printed up front from the measured per-eval costs
(loss ~0.33 s/layer; PS gradient = 2*n_params evals); any run projected over
--max-minutes (default 10) ABORTS unless you pass --yes. Use small --n-iters
/ --n-train for quick trials (e.g. train --n-iters 2 --n-train 6).

It answers four referee threads:
  E1 (variance)  : gradient variance across random inits (barren-plateau
                   test) with a NORMALIZED, McClean-comparable cost.
  E2 (train)     : optimizer independence -- FD / SPSA / parameter-shift /
                   layer-wise all driven from the same circuit.
  E3 (jacobian)  : Jacobian effective dimension (SVD / participation ratio).
  E3 (fourier)   : Fourier expressivity of <Z_j>(t) vs the Lorenz target
                   spectrum (bandwidth comparison).

PROVENANCE
----------
The circuit, expectation, physics loss, forward-difference gradient and
Lorenz dynamics are reused BY IMPORT from the published entry point
scripts/run_qpinn_gradient_logging.py (module is main-guarded, so importing
it runs no training). At L=3 this harness reproduces that script's loss
EXACTLY (gate G-PARITY). For variable n_layers we use a parametric mirror
build_qpinn_circuit_L (same gate structure, 15 params/layer) that equals
the baseline builder at L=3 gate-for-gate; the mirror pattern is taken from
scripts/probe_r1.py.

Parameter-shift correctness: every theta parameter enters exactly one Pauli
rotation (rx/ry/rz), so d<Z_j>(t)/dtheta_k is EXACT via the +/- pi/2 two-eval
shift rule. The physics loss is quadratic in the observables, so the EXACT
loss gradient is obtained by the PS observable-Jacobian followed by the
analytic chain rule through the loss (verified to ~4e-8 rel vs central FD).

HARD LIMITS: 4 qubits fixed. No commits. Short runs only.
"""
from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# Pin BLAS/OpenMP to a single thread PER PROCESS *before* numpy is imported.
# The hot path is a 16-dim (4-qubit) statevector sim: it is Python/latency
# bound, not FLOP bound, so multi-threaded BLAS gives no speedup here but
# DOES oversubscribe when the parallel runner forks up to --jobs workers on a
# fixed core budget (each worker otherwise spins up N BLAS threads -> jobs*N
# threads contending for the cores). Pinning to 1 thread/process makes the
# sustained --jobs 6 throughput match the single-eval microbenchmark instead
# of degrading ~2.5x under contention. Set only if unset (respect the user).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Import the published baseline (main-guarded -> importing runs no training).
# --------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import run_qpinn_gradient_logging as qp  # noqa: E402

# Baseline module constants (ground truth).
N_QUBITS = qp.N_QUBITS               # 4  (NEVER exceed)
N_PARAMS_PER_LAYER = qp.N_PARAMS_PER_LAYER   # 15
BASELINE_N_LAYERS = qp.N_LAYERS      # 3

# Baseline problem defaults (run_seed defaults).
X0 = np.array([1.0, 1.0, 1.0])
DEFAULT_T_END = 3.0
DEFAULT_N_TRAIN = 50
LAM = 10.0                           # boundary weight, as published
DT_TIME = 1e-3                       # central-in-time eps inside physics loss
LORENZ_JAC_PARAMS = (10.0, 28.0, 8.0 / 3.0)   # sigma, rho, beta

# Affine output map (predict_state ranges), reused for the exact chain rule.
X_RANGE = (-20.0, 20.0)
Y_RANGE = (-30.0, 30.0)
Z_RANGE = (0.0, 50.0)
OUT_SCALES = np.array([(X_RANGE[1] - X_RANGE[0]) / 2.0,
                       (Y_RANGE[1] - Y_RANGE[0]) / 2.0,
                       (Z_RANGE[1] - Z_RANGE[0]) / 2.0])
OUT_LO = np.array([X_RANGE[0], Y_RANGE[0], Z_RANGE[0]])

RESULTS_DIR = ROOT / "results" / "r1"


# ==========================================================================
# Parametric circuit (4 qubits fixed, n_layers variable).
# Mirrors run_qpinn_gradient_logging.build_qpinn_circuit gate-for-gate;
# at n_layers == 3 it is identical. Pattern from scripts/probe_r1.py.
# ==========================================================================

def n_params(n_layers: int) -> int:
    return n_layers * N_PARAMS_PER_LAYER


def build_qpinn_circuit_L(t, theta, t_max, n_layers, n_qubits=N_QUBITS):
    """Parametric QPINN circuit at time t (variable layer count)."""
    from qiskit import QuantumCircuit
    if n_qubits > 4:
        raise ValueError("qubit count capped at 4")
    qc = QuantumCircuit(n_qubits)
    # Fixed time encoding (identical to baseline).
    theta_t = 2.0 * np.pi * t / t_max
    qc.ry(float(theta_t), 0)
    qc.rz(float(theta_t), 1)
    qc.rx(float(theta_t), 2)
    qc.ry(float(np.pi * t / t_max), 3)
    # Variational layers (15 params/layer: 12 single-qubit + 3 RZ phases).
    p = 0
    for _ in range(n_layers):
        for q in range(n_qubits):
            qc.rx(float(theta[p + 0]), q)
            qc.ry(float(theta[p + 1]), q)
            qc.rz(float(theta[p + 2]), q)
            p += 3
        for q in range(n_qubits - 1):
            qc.cx(q, q + 1)
            qc.rz(float(theta[p]), q + 1)
            p += 1
    assert p == n_params(n_layers)
    return qc


def _z_expectations_from_state(qc, qubits=(0, 1, 2)):
    """(<Z_q0>, <Z_q1>, ...) computed from a SINGLE Qiskit statevector build.

    REFERENCE PATH (kept for provenance / cross-checking): numerically
    identical to calling qp.expectation_z once per qubit. z_vector() no longer
    uses this -- it uses the direct-numpy simulator below, which reproduces
    this to machine precision (verified in the self-test, gate G-STATEVEC) and
    is ~5x faster per evaluation. Guarded by G-PARITY (loss must still match
    the baseline to <=1e-10)."""
    from qiskit.quantum_info import Statevector
    probs = np.abs(Statevector.from_instruction(qc).data) ** 2
    n = qc.num_qubits
    states = np.arange(2 ** n)
    return np.array([float(np.sum((1.0 - 2.0 * ((states >> q) & 1)) * probs))
                     for q in qubits])


# --------------------------------------------------------------------------
# Direct-numpy 4-qubit statevector simulator (the hot path).
# --------------------------------------------------------------------------
# The QPINN ansatz is a fixed, tiny circuit: rx/ry/rz single-qubit rotations
# and nearest-neighbour cx on 4 qubits (16-dim state). Qiskit 2.x builds a
# fresh QuantumCircuit object and runs Statevector.from_instruction for EVERY
# evaluation, applying each gate through generic dense-Operator machinery
# (Operator.__init__, op_shape.auto, argsort, reshape, isinstance...). Under
# parameter-shift that is ~90*2 shift evals * (t, t+-eps) time-stencil *
# n_train collocation points per iteration -- millions of gate applications,
# each dominated by Python object churn, NOT by the 16-dim linear algebra.
#
# This applies the SAME unitaries directly to a length-16 complex vector via
# precomputed index pairs (a single-qubit gate touches exactly the two
# amplitudes whose q-th bit differs; cx swaps the two amplitudes whose target
# bit differs among control==1 rows). Qiskit's little-endian convention
# (qubit 0 == least-significant bit of the statevector index) is reproduced
# exactly, so the resulting state -- hence <Z_j> -- matches Statevector to
# machine precision (~1e-15). Gate G-STATEVEC re-checks this every self-test.

# Precomputed amplitude-index pairs (i0 has bit q = 0; i1 = i0 | 1<<q).
_Q_IDX = tuple(
    (np.array([i for i in range(2 ** N_QUBITS) if not (i >> q) & 1]),
     np.array([i for i in range(2 ** N_QUBITS) if not (i >> q) & 1]) | (1 << q))
    for q in range(N_QUBITS)
)
# Precomputed cx source/dest index arrays for the nearest-neighbour pairs used
# by the ansatz (control q, target q+1): rows with control bit set, target bit
# flipped.
_CX_IDX = {}
for _q in range(N_QUBITS - 1):
    _src = np.array([i for i in range(2 ** N_QUBITS) if (i >> _q) & 1])
    _CX_IDX[(_q, _q + 1)] = (_src, _src ^ (1 << (_q + 1)))
# +/-1 Z-signs per qubit over all 16 basis states, for <Z_q> = sum(sign*probs).
_Z_STATES = np.arange(2 ** N_QUBITS)
_Z_SIGN = np.array([1.0 - 2.0 * ((_Z_STATES >> q) & 1) for q in (0, 1, 2)])


def _apply_1q(psi, g00, g01, g10, g11, q):
    """In-place single-qubit gate (2x2 = [[g00,g01],[g10,g11]]) on qubit q.
    Scalars are passed unpacked to avoid allocating a 2x2 array per gate."""
    i0, i1 = _Q_IDX[q]
    a = psi[i0]
    b = psi[i1]
    psi[i0] = g00 * a + g01 * b
    psi[i1] = g10 * a + g11 * b


def _apply_cx(psi, ctrl, tgt):
    """In-place cx (control ctrl, target tgt), little-endian."""
    src, dst = _CX_IDX[(ctrl, tgt)]
    tmp = psi[src].copy()
    psi[src] = psi[dst]
    psi[dst] = tmp


def _sim_state(t, theta, t_max, n_layers):
    """Direct-numpy statevector for the QPINN ansatz. Matches
    build_qpinn_circuit_L gate-for-gate; Statevector-equivalent to ~1e-15."""
    psi = np.zeros(2 ** N_QUBITS, dtype=complex)
    psi[0] = 1.0
    theta_t = 2.0 * np.pi * t / t_max
    # Fixed time encoding (ry q0, rz q1, rx q2, ry q3).
    ct, st = np.cos(theta_t / 2.0), np.sin(theta_t / 2.0)
    _apply_1q(psi, ct, -st, st, ct, 0)                       # ry(theta_t) q0
    ez = np.exp(-0.5j * theta_t)
    _apply_1q(psi, ez, 0.0, 0.0, ez.conjugate(), 1)          # rz(theta_t) q1
    _apply_1q(psi, ct, -1j * st, -1j * st, ct, 2)            # rx(theta_t) q2
    th3 = np.pi * t / t_max
    c3, s3 = np.cos(th3 / 2.0), np.sin(th3 / 2.0)
    _apply_1q(psi, c3, -s3, s3, c3, 3)                       # ry(pi t/t_max) q3
    p = 0
    for _ in range(n_layers):
        for q in range(N_QUBITS):
            a = 0.5 * theta[p + 0]                            # rx
            c, s = np.cos(a), np.sin(a)
            _apply_1q(psi, c, -1j * s, -1j * s, c, q)
            a = 0.5 * theta[p + 1]                            # ry
            c, s = np.cos(a), np.sin(a)
            _apply_1q(psi, c, -s, s, c, q)
            e = np.exp(-0.5j * theta[p + 2])                 # rz
            _apply_1q(psi, e, 0.0, 0.0, e.conjugate(), q)
            p += 3
        for q in range(N_QUBITS - 1):
            _apply_cx(psi, q, q + 1)
            e = np.exp(-0.5j * theta[p])                     # rz on q+1
            _apply_1q(psi, e, 0.0, 0.0, e.conjugate(), q + 1)
            p += 1
    return psi


def z_vector(t, theta, t_max, n_layers):
    """(<Z_0>, <Z_1>, <Z_2>) at time t (direct-numpy simulator).

    Statevector-equivalent to the Qiskit path
    _z_expectations_from_state(build_qpinn_circuit_L(...)) to ~1e-15; kept in
    sync by gate G-STATEVEC and guarded end-to-end by G-PARITY / G-PS-FD."""
    psi = _sim_state(t, theta, t_max, n_layers)
    probs = psi.real ** 2 + psi.imag ** 2
    return _Z_SIGN @ probs


def predict_state_L(t, theta, t_max, n_layers):
    """State [x,y,z] via the affine map from the z-vector."""
    return z_vector(t, theta, t_max, n_layers) * OUT_SCALES + OUT_LO + OUT_SCALES


def physics_loss_L(theta, t_points, x0, t_max, n_layers, lam=LAM):
    """Baseline physics loss L_phys + lam*L_bc, for variable layer count.

    At n_layers == 3 this equals qp.physics_loss exactly (see G-PARITY).
    """
    eps = DT_TIME
    L_phys = 0.0
    for t in t_points:
        u = predict_state_L(t, theta, t_max, n_layers)
        u_dot = (predict_state_L(t + eps, theta, t_max, n_layers)
                 - predict_state_L(t - eps, theta, t_max, n_layers)) / (2.0 * eps)
        F_u = qp.lorenz_rhs(u)
        L_phys += float(np.sum((u_dot - F_u) ** 2))
    L_phys /= len(t_points)
    u0 = predict_state_L(0.0, theta, t_max, n_layers)
    L_bc = float(np.sum((u0 - x0) ** 2))
    return L_phys + lam * L_bc


# ==========================================================================
# Cost functions (both selectable).
# ==========================================================================

def cost_paper(theta, t_points, x0, t_max, n_layers):
    """Published physics loss (L_phys + lam*L_bc)."""
    return physics_loss_L(theta, t_points, x0, t_max, n_layers)


def cost_bounded(theta, t_points, x0, t_max, n_layers):
    """McClean-comparable normalized cost in [0,1]:
        C = mean_i [1 - <Z_0>(t_i)] / 2 .
    """
    vals = [(1.0 - z_vector(t, theta, t_max, n_layers)[0]) / 2.0
            for t in t_points]
    return float(np.mean(vals))


COSTS = {"paper": cost_paper, "bounded": cost_bounded}


# ==========================================================================
# Gradients.
# ==========================================================================

def fd_gradient(cost_fn, theta, t_points, x0, t_max, n_layers, eps=1e-4):
    """Forward finite-difference gradient of an arbitrary cost (baseline path
    when cost_fn is cost_paper: matches qp.finite_diff_per_parameter)."""
    base = cost_fn(theta, t_points, x0, t_max, n_layers)
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy()
        tp[i] += eps
        grad[i] = (cost_fn(tp, t_points, x0, t_max, n_layers) - base) / eps
    return base, grad


def central_fd_gradient(cost_fn, theta, t_points, x0, t_max, n_layers, eps=1e-6):
    """Central finite-difference gradient (reference for gates)."""
    grad = np.zeros_like(theta)
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += eps
        tm = theta.copy(); tm[i] -= eps
        gp = cost_fn(tp, t_points, x0, t_max, n_layers)
        gm = cost_fn(tm, t_points, x0, t_max, n_layers)
        grad[i] = (gp - gm) / (2.0 * eps)
    return grad


def ps_z_jacobian(t, theta, t_max, n_layers, shift=np.pi / 2.0):
    """EXACT Jacobian d(<Z_0>,<Z_1>,<Z_2>)(t) / d theta via the +/- pi/2
    parameter-shift rule (2 evals/param). Shape (3, n_params)."""
    npar = len(theta)
    J = np.zeros((3, npar))
    for k in range(npar):
        tp = theta.copy(); tp[k] += shift
        tm = theta.copy(); tm[k] -= shift
        J[:, k] = (z_vector(t, tp, t_max, n_layers)
                   - z_vector(t, tm, t_max, n_layers)) / 2.0
    return J


def ps_loss_gradient(theta, t_points, x0, t_max, n_layers, lam=LAM):
    """EXACT gradient of the PAPER physics loss via parameter-shift on the
    observables + analytic chain rule through the quadratic loss.

    The loss is nonlinear in <Z_j>, so the two-eval shift rule cannot be
    applied to the loss directly; it is applied to each observable (exact),
    then differentiated through analytically. Verified vs central FD.
    """
    eps = DT_TIME
    sigma, rho, beta = LORENZ_JAC_PARAMS
    npar = len(theta)
    g = np.zeros(npar)
    for t in t_points:
        u = predict_state_L(t, theta, t_max, n_layers)
        up = predict_state_L(t + eps, theta, t_max, n_layers)
        um = predict_state_L(t - eps, theta, t_max, n_layers)
        u_dot = (up - um) / (2.0 * eps)
        F_u = qp.lorenz_rhs(u)
        r = u_dot - F_u                                   # residual (3,)
        d_state = OUT_SCALES[:, None] * ps_z_jacobian(t, theta, t_max, n_layers)
        d_udot = (OUT_SCALES[:, None] * ps_z_jacobian(t + eps, theta, t_max, n_layers)
                  - OUT_SCALES[:, None] * ps_z_jacobian(t - eps, theta, t_max, n_layers)
                  ) / (2.0 * eps)
        x, y, z = u
        JF = np.array([[-sigma,  sigma,  0.0],
                       [rho - z, -1.0,   -x],
                       [y,       x,      -beta]])
        d_F = JF @ d_state
        d_r = d_udot - d_F                                # (3, n_params)
        g += 2.0 * (r[:, None] * d_r).sum(axis=0)
    g /= len(t_points)
    # boundary term
    u0 = predict_state_L(0.0, theta, t_max, n_layers)
    d_u0 = OUT_SCALES[:, None] * ps_z_jacobian(0.0, theta, t_max, n_layers)
    r0 = u0 - x0
    g += lam * 2.0 * (r0[:, None] * d_u0).sum(axis=0)
    return g


def spsa_gradient(cost_fn, theta, t_points, x0, t_max, n_layers, c, rng):
    """One SPSA simultaneous-perturbation gradient estimate (2 loss evals)."""
    delta = rng.choice([-1.0, 1.0], size=len(theta))
    lp = cost_fn(theta + c * delta, t_points, x0, t_max, n_layers)
    lm = cost_fn(theta - c * delta, t_points, x0, t_max, n_layers)
    ghat = (lp - lm) / (2.0 * c) / delta
    return ghat, lp, lm


# ==========================================================================
# Helpers: seeding, IO, hashing, git.
# ==========================================================================

# Measured unit cost (file header): one loss eval ~1.0 s at L=3, n_train=50,
# scaling ~0.33 s per layer and linearly in the collocation-point count.
SEC_PER_LOSS_EVAL_L1_FULL = 0.33   # seconds per loss eval, per layer, at n_train=50


def sec_per_loss_eval(n_layers: int, n_train: int) -> float:
    return SEC_PER_LOSS_EVAL_L1_FULL * n_layers * (n_train / DEFAULT_N_TRAIN)


def estimate_train_wall_s(optimizer, n_iters, n_layers, n_train, n_seeds, jobs):
    """Projected wall time (s) for a train run, from measured per-eval costs.
    Evals/iter: fd = n_params+1; ps(paper) = 1 + 4*n_params (loss + PS over 3
    time samples per collocation point ~ absorbed into per-eval scaling, so we
    use the dominant 2*n_params shift evals *2 for the +/-eps time stencil);
    spsa = 2; layerwise = (n_params_per_layer+1) per iter over n_layers stages.
    A deliberate over-estimate is fine -- it only gates a confirmation."""
    tle = sec_per_loss_eval(n_layers, n_train)
    npar = n_params(n_layers)
    if optimizer == "fd":
        evals_per_iter = npar + 1
        total_iters = n_iters
    elif optimizer == "ps":
        evals_per_iter = 1 + 4 * npar   # PS Jacobian at t, t+eps, t-eps
        total_iters = n_iters
    elif optimizer == "spsa":
        evals_per_iter = 2
        total_iters = n_iters
    elif optimizer == "layerwise":
        evals_per_iter = N_PARAMS_PER_LAYER + 1
        total_iters = n_iters * n_layers   # one stage per layer
    else:
        evals_per_iter, total_iters = npar + 1, n_iters
    per_seed = evals_per_iter * total_iters * tle
    return per_seed * n_seeds / max(jobs, 1)


def estimate_jacobian_wall_s(n_inits, n_layers, n_train, n_files=0):
    """PS Jacobian is 2*n_params evals per time point per init."""
    tle = sec_per_loss_eval(n_layers, n_train)
    evals = 2 * n_params(n_layers) * n_train
    return evals * tle * (n_inits + n_files)


def guard_wall_time(estimate_s: float, max_minutes: float, yes: bool, label: str):
    """Print the projection; abort (return False) if it exceeds the budget and
    --yes was not passed. Makes long runs opt-in, per the module docstring."""
    print(f"[estimate] {label}: projected wall time ~ {estimate_s:.0f} s "
          f"(~{estimate_s / 60.0:.1f} min); budget --max-minutes={max_minutes}")
    if estimate_s > max_minutes * 60.0 and not yes:
        print(f"[abort] projected {estimate_s / 60.0:.1f} min exceeds "
              f"--max-minutes {max_minutes}. Re-run with --yes to proceed, "
              f"or lower --n-iters / --n-train / --n-layers.")
        return False
    return True


def init_theta(seed: int, n_layers: int) -> np.ndarray:
    """Reproducible random init, same distribution as baseline run_seed."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0.0, 2.0 * np.pi, n_params(n_layers))


def make_t_points(t_end: float, n_train: int) -> np.ndarray:
    return np.linspace(0.0, t_end, n_train)


def git_rev() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True if the working tree has uncommitted or untracked changes (this
    harness is currently untracked, so a git_rev alone is NOT a valid
    provenance anchor -- see source_sha256)."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"], cwd=str(ROOT),
                             capture_output=True, text=True, timeout=15)
        return bool(out.stdout.strip())
    except Exception:
        return True


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except Exception:
        return "unknown"


def provenance() -> dict:
    """Byte-level provenance so a result JSON is auditable even when the
    generating script is untracked (git_rev then does not pin the code)."""
    return {
        "git_rev": git_rev(),
        "git_dirty": git_dirty(),
        "source_sha256": _sha256_file(Path(__file__)),
        "baseline_sha256": _sha256_file(Path(qp.__file__)),
    }


def _canonical_no_timing(obj):
    """Deep-copy obj with any 'timing' keys removed (for G-DET hashing)."""
    if isinstance(obj, dict):
        return {k: _canonical_no_timing(v) for k, v in obj.items()
                if k != "timing"}
    if isinstance(obj, list):
        return [_canonical_no_timing(v) for v in obj]
    return obj


def result_hash(payload: dict) -> str:
    """SHA256 over the payload EXCLUDING wall times (timing key)."""
    stripped = _canonical_no_timing(payload)
    blob = json.dumps(stripped, sort_keys=True, separators=(",", ":"),
                      default=float).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def save_result(mode: str, tag: str, payload: dict, out_dir: Path = None) -> Path:
    """Write results/r1/<mode>_<tag>.json; print path + SHA256(no timing)."""
    out_dir = out_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = provenance()
    payload.setdefault("git_rev", prov["git_rev"])
    payload.setdefault("provenance", prov)
    digest = result_hash(payload)
    payload["sha256_no_timing"] = digest
    out_path = out_dir / f"{mode}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"[saved] {out_path}")
    print(f"[sha256:no-timing] {digest}")
    return out_path


# ==========================================================================
# Incremental checkpointing + resume.
# ==========================================================================
# Every mode writes each finished UNIT (train: one seed; variance/jacobian:
# one init; fourier: one draw) to <out_dir>/_checkpoints/<mode>_<hash>/ the
# moment it is done, so a kill loses at most one in-flight unit. For `train`
# we ALSO checkpoint WITHIN a seed every CKPT_EVERY_ITERS iters (or every
# ~CKPT_EVERY_SECONDS wall seconds, whichever comes first): the exact optimizer
# state (iter index, theta, Adam m/v or the SPSA rng byte-state + a_base, plus
# layerwise stage/inner counters) and the loss history so far, so a killed seed
# RESUMES from its last saved iteration -- never restarts from iter 0. All
# writes are atomic (temp file + os.replace), so a kill mid-write can never
# corrupt a checkpoint. Re-invoking the SAME config in the SAME checkpoint dir
# loads finished units, resumes partial train seeds, and reassembles the final
# JSON in the identical schema -- indistinguishable from an uninterrupted run.

CKPT_EVERY_ITERS = 10          # checkpoint at least this often (iterations)
CKPT_EVERY_SECONDS = 5.0       # ...or this often (wall seconds), whichever first

# In-process test hook (used only by the G-RESUME gate): if set to an int N,
# _train_one raises _TrainInterrupted right AFTER completing iteration N, so a
# partial checkpoint exists for the resume test. None in all normal runs.
_TEST_STOP_AFTER_ITER = None


class _TrainInterrupted(Exception):
    """Raised by the in-process test hook to simulate a mid-seed kill."""
    def __init__(self, done_iters):
        super().__init__(f"interrupted after {done_iters} iters")
        self.done_iters = done_iters


def _config_hash(config: dict, exclude_keys) -> str:
    """Stable short hash of a run config with the per-unit / cosmetic keys
    removed (seed identity, unit count, tag, jobs, output location, checkpoint
    knobs). Two invocations that differ ONLY in those keys share a checkpoint
    dir, so the SAME seeds/inits resume regardless of how many were requested
    or how the run was tagged/parallelised."""
    reduced = {k: v for k, v in config.items() if k not in exclude_keys}
    blob = json.dumps(reduced, sort_keys=True, separators=(",", ":"),
                      default=float).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def checkpoint_dir(out_dir: Path, mode: str, config: dict,
                   exclude_keys) -> Path:
    """<out_dir>/_checkpoints/<mode>_<config_hash>/ (created on demand)."""
    h = _config_hash(config, exclude_keys)
    d = out_dir / "_checkpoints" / f"{mode}_{h}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _atomic_write_json(path: Path, obj: dict):
    """Write JSON to `path` atomically: dump to a unique temp file in the same
    directory (so os.replace is a rename, not a cross-device copy), flush +
    fsync, then os.replace. A kill at any point leaves EITHER the old file or
    the fully-written new file -- never a truncated/corrupt checkpoint."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}.{time.time_ns()}")
    with open(tmp, "w") as f:
        json.dump(obj, f, default=float)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _unit_ckpt_path(ckpt_dir: Path, key) -> Path:
    """Per-unit checkpoint file. `key` identifies the unit (seed int, init
    label str, or draw index int); sanitised for use as a filename."""
    safe = str(key).replace(os.sep, "_").replace("/", "_")
    return ckpt_dir / f"unit_{safe}.json"


def _load_unit(ckpt_dir: Path, key):
    """Return the saved checkpoint dict for a unit, or None if absent/unreadable
    (a corrupt/partial temp file is never named unit_*.json, so this only ever
    reads a completely-written checkpoint)."""
    p = _unit_ckpt_path(ckpt_dir, key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _save_unit(ckpt_dir: Path, key, ckpt: dict):
    _atomic_write_json(_unit_ckpt_path(ckpt_dir, key), ckpt)


def _resolve_ckpt_dir(args, mode, config, exclude_keys):
    """Return the checkpoint dir for this run (Path), or None if checkpointing
    is disabled via --no-checkpoint. Honours a --checkpoint-dir override; else
    defaults to <out_dir>/_checkpoints/<mode>_<config_hash>/. Default behaviour
    (no flags) is checkpointing ON -- resume is transparent when the same
    out_dir/config is reused."""
    if getattr(args, "no_checkpoint", False):
        return None
    if getattr(args, "checkpoint_dir", None):
        d = Path(args.checkpoint_dir)
        d.mkdir(parents=True, exist_ok=True)
        return d
    return checkpoint_dir(args.out_dir_path, mode, config, exclude_keys)


def _report_resume(ckpt_dir, keys, mode, unit_label):
    """Print a one-line resume summary and return (n_complete, n_partial) among
    the requested unit keys, so a re-invocation visibly states what it will skip
    vs. resume vs. compute fresh."""
    n_done = n_partial = 0
    for k in keys:
        u = _load_unit(ckpt_dir, k)
        if u is None:
            continue
        if u.get("complete"):
            n_done += 1
        elif u.get("done_iters", 0) > 0:
            n_partial += 1
    n_fresh = len(keys) - n_done - n_partial
    print(f"[checkpoint] {mode}: dir={ckpt_dir}")
    print(f"[checkpoint] {len(keys)} {unit_label}(s): {n_done} complete "
          f"(skip), {n_partial} partial (resume), {n_fresh} fresh")
    return n_done, n_partial


# ==========================================================================
# Parallel runner (Windows spawn-safe: all args picklable, top-level worker).
# ==========================================================================

def _variance_worker(spec: dict) -> dict:
    """Compute one init's gradient(s). Re-seeds from spec.

    Two quantities per init:
      * out[cname]        : gradient of the TIME-AVERAGED cost (FD). This is a
                            trained-cost diagnostic, NOT McClean-comparable --
                            averaging over the encoding period before
                            differentiating annihilates non-DC Fourier modes.
      * out['mcclean_bounded'] : list over collocation times of the EXACT
                            per-t gradient dC_t/dtheta_k of the bounded cost
                            C_t = (1 - <Z_0>(t))/2, via parameter-shift. Each
                            t is a FIXED circuit, so Var across inits of this
                            (pooled over t) is the McClean quantity.
    """
    seed = spec["seed"]
    # Resume: a finished init is loaded from its per-unit checkpoint and NOT
    # recomputed. Each init is fast, so per-unit granularity is sufficient here.
    ckpt_dir = spec.get("ckpt_dir")
    if ckpt_dir is not None:
        cached = _load_unit(Path(ckpt_dir), seed)
        if cached is not None and cached.get("complete"):
            return cached["result"]
    n_layers = spec["n_layers"]
    which = spec["cost"]              # "paper" | "bounded" | "both"
    t_points = make_t_points(spec["t_end"], spec["n_train"])
    theta = init_theta(seed, n_layers)
    cost_list = ["paper", "bounded"] if which == "both" else [which]
    out = {"seed": seed}
    for cname in cost_list:
        _, grad = fd_gradient(COSTS[cname], theta, t_points, X0,
                              spec["t_end"], n_layers)
        out[cname] = grad.tolist()
    if spec.get("mcclean") and ("bounded" in cost_list):
        # Exact fixed-circuit gradient of C_t = (1-<Z_0>(t))/2 at each t.
        per_t = [(-0.5 * ps_z_jacobian(t, theta, spec["t_end"], n_layers)[0]).tolist()
                 for t in t_points]
        out["mcclean_bounded"] = per_t
    if ckpt_dir is not None:
        _save_unit(Path(ckpt_dir), seed,
                   {"complete": True, "seed": seed, "result": out})
    return out


def _seed_train_worker(spec: dict) -> dict:
    """Run one training unit (single seed). Re-seeds from spec."""
    return _train_one(spec)


def run_units(worker, specs, jobs: int):
    """Map worker over specs, serial (jobs==1) or via a Pool (jobs>1).
    Results are returned in the SAME ORDER as specs (Pool.map preserves it),
    so serial and parallel produce identical output (gate G-PAR)."""
    if jobs <= 1:
        return [worker(s) for s in specs]
    with mp.Pool(processes=jobs) as pool:
        return pool.map(worker, specs)


# ==========================================================================
# Subcommand: variance (E1).
# ==========================================================================

def cmd_variance(args):
    n_layers = args.n_layers
    mcclean = bool(getattr(args, "mcclean", False))
    config = {
        "n_inits": args.n_inits, "n_layers": n_layers, "cost": args.cost,
        "n_qubits": N_QUBITS, "n_params": n_params(n_layers),
        "t_end": args.t_end, "n_train": args.n_train,
        "seed_base": args.seed_base, "jobs": args.jobs,
        "fd_eps": 1e-4, "mcclean": mcclean,
        "mcclean_grad_method": "parameter-shift (exact)" if mcclean else None,
    }
    ck = _resolve_ckpt_dir(args, "variance", config, _VARIANCE_HASH_EXCLUDE)
    specs = [{"seed": args.seed_base + i, "n_layers": n_layers,
              "cost": args.cost, "t_end": args.t_end, "n_train": args.n_train,
              "mcclean": mcclean, "ckpt_dir": (str(ck) if ck else None)}
             for i in range(args.n_inits)]
    if ck is not None:
        _report_resume(ck, [s["seed"] for s in specs], "variance", "init")
    t0 = time.perf_counter()
    per_init = run_units(_variance_worker, specs, args.jobs)
    wall = time.perf_counter() - t0

    cost_list = ["paper", "bounded"] if args.cost == "both" else [args.cost]
    stats = {}
    for cname in cost_list:
        G = np.array([np.array(pi[cname]) for pi in per_init])   # (n_inits, npar)
        var_k = np.var(G, axis=0, ddof=0)
        mean_k = np.mean(G, axis=0)
        stats[cname] = {
            "note": "gradient of the TIME-AVERAGED cost; trained-cost "
                    "diagnostic, NOT comparable to McClean 2^-n thresholds "
                    "(the encoding period is averaged out before "
                    "differentiation). See mcclean_stats.",
            "var_per_component": var_k.tolist(),
            "mean_per_component": mean_k.tolist(),
            "var_mean_over_components": float(np.mean(var_k)),
            "var_median_over_components": float(np.median(var_k)),
            "grad_norm_per_init": [float(np.linalg.norm(g)) for g in G],
        }

    # ---- McClean-comparable fixed-circuit variance (E1 headline) -----------
    mcclean_stats = None
    if mcclean and ("bounded" in cost_list):
        # per_init[i]["mcclean_bounded"] : (n_train, n_params) exact grads.
        # Fixed t -> fixed circuit. Var across inits at each (t) then pooled;
        # also pool all (init, t) samples for the headline per-component Var.
        # Shape: (n_inits, n_train, n_params)
        M = np.array([np.array(pi["mcclean_bounded"]) for pi in per_init])
        n_t = M.shape[1]
        # Var across inits at each fixed t, per component -> (n_train, n_params)
        var_kt = np.var(M, axis=0, ddof=0)
        # Headline: mean over components of the per-t variance, reported per t
        # and pooled (mean over t of that mean -- all fixed-circuit).
        var_mean_per_t = [float(np.mean(var_kt[j])) for j in range(n_t)]
        mcclean_stats = {
            "definition": "Var across inits of dC_t/dtheta_k at FIXED t "
                          "(fixed circuit), exact via parameter-shift; "
                          "directly comparable to the McClean 2^-n bound. "
                          "n_train-independent (does NOT average over t before "
                          "differentiating).",
            "var_mean_over_components_per_t": var_mean_per_t,
            "var_mean_over_components_and_t": float(np.mean(var_mean_per_t)),
            "var_median_over_components_and_t":
                float(np.median([np.median(var_kt[j]) for j in range(n_t)])),
            "n_times": n_t,
        }

    payload = {
        "mode": "variance",
        "config": config,
        "per_init_gradients": per_init,
        "stats": stats,
        "mcclean_stats": mcclean_stats,
        "timing": {"wall_s": wall},
    }
    save_result("variance", args.tag, payload, args.out_dir_path)
    for cname in cost_list:
        print(f"  cost={cname} (time-avg, NOT McClean): mean Var_k = "
              f"{stats[cname]['var_mean_over_components']:.3e}")
    if mcclean_stats is not None:
        print(f"  [McClean fixed-t, exact PS] mean Var_k[dC_t/dtheta_k] = "
              f"{mcclean_stats['var_mean_over_components_and_t']:.3e}")


# ==========================================================================
# Subcommand: train (E2).
# ==========================================================================

def _finalize_train(seed, opt, stages, iters, per_iter_wall, final_loss,
                    theta) -> dict:
    """Assemble the finished-seed result dict (identical schema to the
    pre-checkpoint harness). Used both by a fresh finish and by a resumed
    finish, so a resumed seed's record is byte-identical to an uninterrupted
    one modulo the wall-time list (which is excluded from sha256_no_timing)."""
    return {
        "seed": seed, "optimizer": opt,
        "stages": stages,  # None unless layerwise (== n_layers: all layers trained)
        "iters": iters,
        "final_loss": float(final_loss),
        "theta_final": theta.tolist() if hasattr(theta, "tolist") else list(theta),
        "timing": {"per_iter_wall_s": per_iter_wall,
                   "total_wall_s": float(np.sum(per_iter_wall))},
    }


def _train_one(spec: dict) -> dict:
    """Single-seed training run for one optimizer. Deterministic given spec.

    Checkpointing/resume: if spec["ckpt_dir"] is set, progress is written to
    that seed's checkpoint every CKPT_EVERY_ITERS iters (or every
    CKPT_EVERY_SECONDS wall seconds), and on entry any partial checkpoint for
    this seed is loaded so the optimization RESUMES from its last saved
    iteration -- it never restarts from iter 0. A checkpoint with
    complete=True short-circuits to reassembling the finished record from disk.
    With no ckpt_dir the behaviour is exactly the original (no I/O)."""
    opt = spec["optimizer"]
    seed = spec["seed"]
    n_iters = spec["n_iters"]
    n_layers = spec["n_layers"]
    cname = spec["cost"]
    t_end = spec["t_end"]
    n_train = spec["n_train"]
    cost_fn = COSTS[cname]
    t_points = make_t_points(t_end, n_train)
    npar = n_params(n_layers)

    ckpt_dir = spec.get("ckpt_dir")
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
    stop_after = _TEST_STOP_AFTER_ITER  # in-process test hook (None normally)

    # ---- Load any existing checkpoint for this seed -----------------------
    prog = None
    if ckpt_dir is not None:
        prog = _load_unit(ckpt_dir, seed)
    if prog is not None and prog.get("complete"):
        # Already fully done: reassemble the finished record, recompute nothing.
        theta = np.array(prog["theta"])
        return _finalize_train(seed, opt, prog.get("stages"), prog["iters"],
                               prog["per_iter_wall"], prog["final_loss"], theta)

    # ---- Restore or fresh-initialise the optimizer state -----------------
    if prog is not None:
        theta = np.array(prog["theta"])
        iters = prog["iters"]
        per_iter_wall = prog["per_iter_wall"]
        done_iters = prog["done_iters"]     # descent iters already completed
        st = prog["state"]                  # optimizer-specific state
    else:
        theta = init_theta(seed, n_layers)
        iters = []
        per_iter_wall = []
        done_iters = 0
        st = None

    stages = n_layers if opt == "layerwise" else None
    _last_ckpt = [time.perf_counter()]      # mutable holder for closure

    def _save_progress(done, state, force=False):
        """Atomically checkpoint mid-seed progress if K iters or ~5s elapsed."""
        if ckpt_dir is None:
            return
        now = time.perf_counter()
        if not force and (done % CKPT_EVERY_ITERS != 0) \
                and (now - _last_ckpt[0] < CKPT_EVERY_SECONDS):
            return
        _save_unit(ckpt_dir, seed, {
            "seed": seed, "optimizer": opt, "complete": False,
            "done_iters": done, "n_iters": n_iters, "stages": stages,
            "theta": theta.tolist(), "iters": iters,
            "per_iter_wall": per_iter_wall, "state": state,
        })
        _last_ckpt[0] = now

    def record(it, loss, grad_norm, loss_exact=None):
        rec = {"iter": it, "loss": float(loss),
               "grad_norm": (None if grad_norm is None
                             else float(grad_norm))}
        # loss_exact: for SPSA this is the TRUE cost_fn(theta_k) on sampled
        # iters (K=10, plus iter 0 and final); NaN otherwise. Absent for the
        # other optimizers whose "loss" already IS the exact loss.
        if loss_exact is not None:
            rec["loss_exact"] = float(loss_exact)
        iters.append(rec)

    if opt == "fd":
        # Baseline forward-difference path + baseline Adam (matches published).
        if st is None:
            m = np.zeros(npar); v = np.zeros(npar)
        else:
            m = np.array(st["m"]); v = np.array(st["v"])
        for it in range(done_iters + 1, n_iters + 1):
            ti = time.perf_counter()
            loss, grad = fd_gradient(cost_fn, theta, t_points, X0, t_end, n_layers)
            cap = np.percentile(np.abs(grad), 99) * 5.0
            grad_c = np.clip(grad, -cap, cap)
            theta, m, v = qp.adam_step(theta, grad_c, m, v, it,
                                       lr=1e-3 * (0.995 ** it))
            per_iter_wall.append(time.perf_counter() - ti)
            record(it, loss, float(np.linalg.norm(grad)))
            _save_progress(it, {"m": m.tolist(), "v": v.tolist()})
            if stop_after is not None and it >= stop_after:
                _save_progress(it, {"m": m.tolist(), "v": v.tolist()}, force=True)
                raise _TrainInterrupted(it)

    elif opt == "ps":
        # Parameter-shift EXACT gradient (paper cost only; bounded cost's PS
        # gradient is likewise exact but we route the published loss here).
        if cname != "paper":
            # bounded cost is linear in <Z_0>, so PS applies directly & exactly.
            pass
        if st is None:
            m = np.zeros(npar); v = np.zeros(npar)
        else:
            m = np.array(st["m"]); v = np.array(st["v"])
        for it in range(done_iters + 1, n_iters + 1):
            ti = time.perf_counter()
            loss = cost_fn(theta, t_points, X0, t_end, n_layers)
            if cname == "paper":
                grad = ps_loss_gradient(theta, t_points, X0, t_end, n_layers)
            else:
                # bounded: C = mean_i (1-<Z0>)/2  -> dC/dtheta = -0.5*mean_i dZ0
                acc = np.zeros(npar)
                for t in t_points:
                    acc += -0.5 * ps_z_jacobian(t, theta, t_end, n_layers)[0]
                grad = acc / len(t_points)
            cap = np.percentile(np.abs(grad), 99) * 5.0
            grad_c = np.clip(grad, -cap, cap)
            theta, m, v = qp.adam_step(theta, grad_c, m, v, it,
                                       lr=1e-3 * (0.995 ** it))
            per_iter_wall.append(time.perf_counter() - ti)
            record(it, loss, float(np.linalg.norm(grad)))
            _save_progress(it, {"m": m.tolist(), "v": v.tolist()})
            if stop_after is not None and it >= stop_after:
                _save_progress(it, {"m": m.tolist(), "v": v.tolist()}, force=True)
                raise _TrainInterrupted(it)

    elif opt == "spsa":
        # Standard Spall gains. a_base is calibrated on the MEDIAN of max|ghat|
        # over several SPSA probes at theta_0 (fix 1): a single first estimate
        # can land nearly orthogonal to the gradient, giving a tiny g_inf that
        # blows a_base up ~1000x and diverges that seed. The median is robust.
        rng = np.random.default_rng(seed + 10_000)  # descent-loop SPSA stream
        A = 0.1 * n_iters
        alpha, gamma = 0.602, 0.101
        c_base = 0.1
        a_target = 0.05
        if st is None:
            # --- fix 1: robust a_base from a dedicated, seed-derived sub-stream.
            # A SEPARATE rng (seed + 20_000) is used ONLY for the calibration
            # probes so it does NOT consume/shift the descent-loop rng above --
            # the descent trajectory stays a pure function of `seed`.
            cal_rng = np.random.default_rng(seed + 20_000)
            n_cal = 5
            c0 = c_base  # c_k at it==0 is c_base / 1**gamma == c_base
            g_infs = []
            for _ in range(n_cal):
                ghat_c, _, _ = spsa_gradient(cost_fn, theta, t_points, X0,
                                             t_end, n_layers, c0, cal_rng)
                g_infs.append(float(np.max(np.abs(ghat_c))))
            g_inf_med = float(np.median(g_infs)) + 1e-12
            a_base = a_target * ((A + 1) ** alpha) / g_inf_med
        else:
            # Restore a_base (calibration is done ONCE, never re-run) and the
            # EXACT descent-rng byte-state so the perturbation stream continues
            # from where it was killed -- resuming reproduces the uninterrupted
            # trajectory bit-for-bit.
            a_base = st["a_base"]
            rng.bit_generator.state = st["rng_state"]
        # `it` here is the 0-based SPSA loop index; done_iters == number of
        # loop iterations already completed, so resume continues at that index.
        for it in range(done_iters, n_iters):
            ti = time.perf_counter()
            c_k = c_base / (it + 1) ** gamma
            ghat, lp, lm = spsa_gradient(cost_fn, theta, t_points, X0,
                                         t_end, n_layers, c_k, rng)
            a_k = a_base / (it + 1 + A) ** alpha
            # fix 2: record the EXACT loss cost_fn(theta_k) at iter 0, every
            # K=10 iters, and the final iter, so the SPSA curve is comparable
            # to fd/ps on equal footing. These extra evals do NOT touch `rng`
            # (they take no perturbation), so determinism is preserved.
            K = 10
            loss_exact = None
            if (it % K == 0) or (it == n_iters - 1):
                loss_exact = cost_fn(theta, t_points, X0, t_end, n_layers)
            theta = theta - a_k * ghat
            # perturbed-loss proxy: 0.5*(L(theta+c*delta)+L(theta-c*delta)),
            # biased ABOVE the true loss; kept for continuity, superseded by
            # loss_exact for plotting.
            loss = 0.5 * (lp + lm)
            per_iter_wall.append(time.perf_counter() - ti)
            record(it + 1, loss, None,  # no exact grad -> grad_norm None
                   loss_exact=loss_exact)
            done = it + 1
            _save_progress(done, {"a_base": a_base,
                                  "rng_state": rng.bit_generator.state})
            if stop_after is not None and done >= stop_after:
                _save_progress(done, {"a_base": a_base,
                                      "rng_state": rng.bit_generator.state},
                               force=True)
                raise _TrainInterrupted(done)

    elif opt == "layerwise":
        # One sequential stage PER LAYER; stage s trains only layer s's 15
        # params (FD, 16 evals/iter) with the others frozen at init. Every
        # layer is trained (stages == n_layers), so the E3 capacity curve is
        # not spuriously flattened above L=3.
        if st is None:
            m = np.zeros(npar); v = np.zeros(npar)
            global_it = 0
        else:
            m = np.array(st["m"]); v = np.array(st["v"])
            global_it = st["global_it"]
        total = stages * n_iters
        # Flat global iteration index (0-based) maps to (stage, inner) so a
        # resume continues at the exact stage AND inner iteration where it was
        # killed. global_it == done_iters here (they advance together).
        for gi in range(done_iters, total):
            s = gi // n_iters
            lo = s * N_PARAMS_PER_LAYER
            hi = lo + N_PARAMS_PER_LAYER
            ti = time.perf_counter()
            # FD only over the active slice (16 evals: 1 base + 15).
            base = cost_fn(theta, t_points, X0, t_end, n_layers)
            grad = np.zeros(npar)
            for i in range(lo, hi):
                tp = theta.copy(); tp[i] += 1e-4
                grad[i] = (cost_fn(tp, t_points, X0, t_end, n_layers)
                           - base) / 1e-4
            cap = np.percentile(np.abs(grad[lo:hi]), 99) * 5.0
            grad_c = grad.copy()
            grad_c[lo:hi] = np.clip(grad[lo:hi], -cap, cap)
            global_it += 1
            theta, m, v = qp.adam_step(theta, grad_c, m, v, global_it,
                                       lr=1e-3 * (0.995 ** global_it))
            per_iter_wall.append(time.perf_counter() - ti)
            iters.append({"iter": global_it, "stage": s, "loss": float(base),
                          "grad_norm": float(np.linalg.norm(grad[lo:hi]))})
            _save_progress(global_it,
                           {"m": m.tolist(), "v": v.tolist(),
                            "global_it": global_it})
            if stop_after is not None and global_it >= stop_after:
                _save_progress(global_it,
                               {"m": m.tolist(), "v": v.tolist(),
                                "global_it": global_it}, force=True)
                raise _TrainInterrupted(global_it)
    else:
        raise ValueError(f"unknown optimizer {opt}")

    final_loss = cost_fn(theta, t_points, X0, t_end, n_layers)
    result = _finalize_train(seed, opt, stages, iters, per_iter_wall,
                             final_loss, theta)
    # Mark the unit complete on disk so a re-invocation is a pure reassembly.
    if ckpt_dir is not None:
        _save_unit(ckpt_dir, seed, {
            "seed": seed, "optimizer": opt, "complete": True,
            "done_iters": (stages * n_iters if opt == "layerwise" else n_iters),
            "n_iters": n_iters, "stages": stages,
            "theta": theta.tolist(), "iters": iters,
            "per_iter_wall": per_iter_wall, "final_loss": float(final_loss),
        })
    return result


# Config keys that identify a UNIT or are purely cosmetic/operational and so
# must NOT enter the checkpoint config_hash (else the same seeds would refuse to
# resume across a different --n-seeds / --tag / --jobs). Everything else is
# semantic and DOES key the checkpoint dir.
_TRAIN_HASH_EXCLUDE = {"n_seeds", "seed_base", "jobs"}
_VARIANCE_HASH_EXCLUDE = {"n_inits", "seed_base", "jobs"}
_JACOBIAN_HASH_EXCLUDE = {"n_inits", "seed_base", "params_file"}
_FOURIER_HASH_EXCLUDE = {"n_draws", "seed_base"}


def cmd_train(args):
    est = estimate_train_wall_s(args.optimizer, args.n_iters, args.n_layers,
                                args.n_train, args.n_seeds, args.jobs)
    if not guard_wall_time(est, args.max_minutes, args.yes,
                           f"train --optimizer {args.optimizer}"):
        return 2
    config = {
        "optimizer": args.optimizer, "n_iters": args.n_iters,
        "n_seeds": args.n_seeds, "seed_base": args.seed_base,
        "n_layers": args.n_layers, "cost": args.cost,
        "n_qubits": N_QUBITS, "n_params": n_params(args.n_layers),
        "t_end": args.t_end, "n_train": args.n_train, "jobs": args.jobs,
        # Result-shaping hyperparameters (were hardcoded; recorded here so
        # a train JSON is regenerable from config + provenance alone).
        "optimizer_hyperparams": {
            "fd_eps": 1e-4,
            "grad_clip": "clip to +/- 5 * 99th-percentile(|grad|)",
            "adam": {"lr_base": 1e-3, "lr_decay": 0.995,
                     "b1": 0.9, "b2": 0.999, "eps": 1e-8},
            "spsa": {"alpha": 0.602, "gamma": 0.101, "A": "0.1 * n_iters",
                     "c_base": 0.1, "a_target": 0.05, "seed_offset": 10000,
                     "a_base_calibration": "median of max|ghat| over 5 "
                                           "probes at theta_0",
                     "calibration_seed_offset": 20000,
                     "loss_exact_every_k": 10},
            "layerwise": {"stages": "n_layers (one stage per layer)"},
        },
    }
    ck = _resolve_ckpt_dir(args, "train", config, _TRAIN_HASH_EXCLUDE)
    specs = [{"optimizer": args.optimizer, "seed": args.seed_base + i,
              "n_iters": args.n_iters, "n_layers": args.n_layers,
              "cost": args.cost, "t_end": args.t_end, "n_train": args.n_train,
              "ckpt_dir": (str(ck) if ck else None)}
             for i in range(args.n_seeds)]
    if ck is not None:
        n_done, n_partial = _report_resume(ck, [s["seed"] for s in specs],
                                           "train", "seed")
    t0 = time.perf_counter()
    runs = run_units(_seed_train_worker, specs, args.jobs)
    wall = time.perf_counter() - t0
    payload = {
        "mode": "train",
        "config": config,
        "runs": runs,
        "timing": {"wall_s": wall},
    }
    save_result("train", f"{args.optimizer}_{args.tag}", payload, args.out_dir_path)
    for r in runs:
        pit = r["timing"]["per_iter_wall_s"]
        mean_it = float(np.mean(pit)) if pit else 0.0
        print(f"  seed={r['seed']} opt={r['optimizer']} "
              f"final_loss={r['final_loss']:.3e} mean_iter={mean_it:.2f}s")


# ==========================================================================
# Subcommand: jacobian (E3).
# ==========================================================================

def _jacobian_for_theta(theta, t_points, t_end, n_layers):
    """Stacked exact PS Jacobian over all times: (3*n_times, n_params)."""
    blocks = [ps_z_jacobian(t, theta, t_end, n_layers) for t in t_points]
    return np.vstack(blocks)


def cmd_jacobian(args):
    n_layers = args.n_layers
    n_files = 1 if args.params_file else 0
    est = estimate_jacobian_wall_s(args.n_inits, n_layers, args.n_train, n_files)
    if not guard_wall_time(est, args.max_minutes, args.yes, "jacobian"):
        return 2
    # Warn when the row count under-determines the Jacobian (PR/rank row-capped).
    if 3 * args.n_train < n_params(n_layers):
        print(f"[warn] 3*n_train={3 * args.n_train} < n_params="
              f"{n_params(n_layers)}: participation ratio / rank are "
              f"row-limited; increase --n-train for a faithful capacity curve.")
    t_points = make_t_points(args.t_end, args.n_train)

    config = {
        "n_inits": args.n_inits, "n_layers": n_layers,
        "n_qubits": N_QUBITS, "n_params": n_params(n_layers),
        "t_end": args.t_end, "n_train": args.n_train,
        "seed_base": args.seed_base, "params_file": args.params_file,
        "rank_threshold_rel": 1e-6, "method": "parameter-shift (exact)",
    }
    ck = _resolve_ckpt_dir(args, "jacobian", config, _JACOBIAN_HASH_EXCLUDE)

    thetas = []
    if args.params_file:
        loaded = np.array(json.loads(Path(args.params_file).read_text()))
        thetas.append(("file", loaded))
    for i in range(args.n_inits):
        thetas.append((f"seed{args.seed_base + i}",
                       init_theta(args.seed_base + i, n_layers)))
    if ck is not None:
        _report_resume(ck, [lbl for lbl, _ in thetas], "jacobian", "init")

    t0 = time.perf_counter()
    entries = []
    for label, theta in thetas:
        # Resume: a finished init's entry is loaded from disk, not recomputed.
        if ck is not None:
            cached = _load_unit(ck, label)
            if cached is not None and cached.get("complete"):
                entries.append(cached["result"])
                continue
        J = _jacobian_for_theta(theta, t_points, args.t_end, n_layers)
        sv = np.linalg.svd(J, compute_uv=False)
        sv2 = sv ** 2
        pr = float((sv2.sum() ** 2) / (np.sum(sv2 ** 2) + 1e-300))  # participation ratio
        rank = int(np.sum(sv > 1e-6 * (sv[0] if sv.size else 1.0)))
        entry = {
            "label": label,
            "singular_values": sv.tolist(),
            "participation_ratio": pr,
            "rank_1e-6": rank,
            "jacobian_shape": list(J.shape),
        }
        entries.append(entry)
        if ck is not None:
            _save_unit(ck, label, {"complete": True, "label": label,
                                   "result": entry})
    wall = time.perf_counter() - t0

    payload = {
        "mode": "jacobian",
        "config": config,
        "entries": entries,
        "timing": {"wall_s": wall},
    }
    save_result("jacobian", args.tag, payload, args.out_dir_path)
    for e in entries:
        print(f"  {e['label']}: eff-dim(PR)={e['participation_ratio']:.2f} "
              f"rank(1e-6)={e['rank_1e-6']} shape={e['jacobian_shape']}")


# ==========================================================================
# Subcommand: fourier (E3).
# ==========================================================================

def cmd_fourier(args):
    n_layers = args.n_layers
    n_grid = max(args.n_grid, 512)
    t_grid = np.linspace(0.0, args.t_end, n_grid)

    # Model spectra: aggregate |FFT| of <Z_j>(t) over random param draws.
    freqs = np.fft.rfftfreq(n_grid, d=(t_grid[1] - t_grid[0]))
    config = {
        "n_draws": args.n_draws, "n_layers": n_layers, "n_grid": n_grid,
        "n_qubits": N_QUBITS, "t_end": args.t_end, "seed_base": args.seed_base,
    }
    ck = _resolve_ckpt_dir(args, "fourier", config, _FOURIER_HASH_EXCLUDE)
    if ck is not None:
        _report_resume(ck, [args.seed_base + d for d in range(args.n_draws)],
                       "fourier", "draw")

    agg = {"Z0": np.zeros(len(freqs)), "Z1": np.zeros(len(freqs)),
           "Z2": np.zeros(len(freqs))}
    t0 = time.perf_counter()
    for d in range(args.n_draws):
        draw_seed = args.seed_base + d
        # Resume: a finished draw's per-key spectrum is loaded and re-added,
        # not recomputed. A UNIT = one draw (n_grid single-draw FFTs).
        cached = _load_unit(ck, draw_seed) if ck is not None else None
        if cached is not None and cached.get("complete"):
            for key in ("Z0", "Z1", "Z2"):
                agg[key] += np.array(cached["spectrum"][key])
            continue
        theta = init_theta(draw_seed, n_layers)
        zt = np.array([z_vector(t, theta, args.t_end, n_layers) for t in t_grid])
        draw_spec = {}
        for j, key in enumerate(("Z0", "Z1", "Z2")):
            spec = np.abs(np.fft.rfft(zt[:, j] - zt[:, j].mean()))
            agg[key] += spec
            draw_spec[key] = spec.tolist()
        if ck is not None:
            _save_unit(ck, draw_seed, {"complete": True, "seed": draw_seed,
                                       "spectrum": draw_spec})
    for key in agg:
        agg[key] /= max(args.n_draws, 1)
    model_wall = time.perf_counter() - t0

    # Target Lorenz spectrum from the SAME trajectory data the baseline trains
    # on (Lorenz from x0 over [0, t_end], baseline dt=0.01), each component.
    traj = qp.integrate(qp.lorenz_rhs, X0, args.t_end, dt=0.01)
    tgt_n = traj.shape[0]
    tgt_freqs = np.fft.rfftfreq(tgt_n, d=0.01)
    target = {}
    for j, key in enumerate(("x", "y", "z")):
        target[key] = np.abs(np.fft.rfft(traj[:, j] - traj[:, j].mean())).tolist()

    def bandwidth(freqs_arr, spec_arr, frac=0.99):
        s = np.asarray(spec_arr)
        cs = np.cumsum(s ** 2)
        if cs[-1] <= 0:
            return 0.0
        idx = int(np.searchsorted(cs, frac * cs[-1]))
        idx = min(idx, len(freqs_arr) - 1)
        return float(freqs_arr[idx])

    # ---- Analytic accessible-frequency bound (Schuld-style) ----------------
    # Each time-encoding Pauli rotation R(phi(t)), phi = omega * t, has
    # generator eigenvalues +/-1/2, contributing frequencies {-f, 0, +f} with
    # f = omega / (2*pi). Encoding: qubits 0,1,2 use 2*pi*t/t_end -> f=1/t_end;
    # qubit 3 uses pi*t/t_end -> f=0.5/t_end. The accessible spectrum of the
    # Fourier model is spanned by sums/differences of these, so the MAXIMUM
    # accessible frequency is their sum: (1+1+1+0.5)/t_end = 3.5/t_end. This is
    # the model-class expressivity bound; the random-draw spectrum below only
    # illustrates where random theta happens to place energy (< the bound).
    encoding_freqs_hz = [1.0 / args.t_end, 1.0 / args.t_end,
                         1.0 / args.t_end, 0.5 / args.t_end]
    f_max_analytic = float(sum(encoding_freqs_hz))          # 3.5 / t_end
    target_bw = {"x": bandwidth(tgt_freqs, target["x"]),
                 "y": bandwidth(tgt_freqs, target["y"]),
                 "z": bandwidth(tgt_freqs, target["z"])}

    payload = {
        "mode": "fourier",
        "config": {
            "n_draws": args.n_draws, "n_layers": n_layers, "n_grid": n_grid,
            "n_qubits": N_QUBITS, "n_params": n_params(n_layers),
            "t_end": args.t_end, "seed_base": args.seed_base,
            "target_dt": 0.01, "target_n": tgt_n,
            "freq_resolution_hz": 1.0 / args.t_end,
        },
        # Headline expressivity bound (analytic, resolution-independent).
        "expressivity": {
            "encoding_freqs_hz": encoding_freqs_hz,
            "f_max_accessible_hz": f_max_analytic,
            "target_bandwidth_99pct_hz": target_bw,
            "note": "f_max_accessible (analytic) is the model-class bound: a "
                    "trained theta can place energy anywhere up to this; the "
                    "band-limited model class (3.5/t_end) is compared to the "
                    "chaotic target bandwidth. The random-draw spectrum is an "
                    "illustration and understates the class (it is also "
                    "resolution-limited to 1/t_end per FFT bin).",
        },
        "model_freqs": freqs.tolist(),
        "model_spectrum": {k: v.tolist() for k, v in agg.items()},
        "target_freqs": tgt_freqs.tolist(),
        "target_spectrum": target,
        "bandwidth_99pct": {
            "model_Z0": bandwidth(freqs, agg["Z0"]),
            "model_Z1": bandwidth(freqs, agg["Z1"]),
            "model_Z2": bandwidth(freqs, agg["Z2"]),
            "target_x": target_bw["x"],
            "target_y": target_bw["y"],
            "target_z": target_bw["z"],
        },
        "timing": {"model_wall_s": model_wall},
    }
    save_result("fourier", args.tag, payload, args.out_dir_path)
    bw = payload["bandwidth_99pct"]
    print(f"  f_max accessible (analytic) = {f_max_analytic:.2f} Hz  "
          f"target BW(x) = {bw['target_x']:.2f} Hz")
    print(f"  [illustration] random-draw model BW(Z0) = {bw['model_Z0']:.2f} Hz")


# ==========================================================================
# Subcommand: self-test (gates).
# ==========================================================================

def _gate(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def cmd_selftest(args):
    print("=" * 70)
    print("R1 harness self-test")
    print("=" * 70)
    results = {}

    # small collocation set for speed; gates are structural not statistical.
    t_end = DEFAULT_T_END
    t_points_full = make_t_points(t_end, DEFAULT_N_TRAIN)   # for G-PARITY parity
    t_points_small = make_t_points(t_end, 6)

    # ---- G-PARITY: L=3 harness loss == baseline script loss, rel <=1e-10 ----
    seed = 12345
    theta3 = init_theta(seed, 3)
    l_harness = physics_loss_L(theta3, t_points_full, X0, t_end, 3)
    l_baseline = qp.physics_loss(theta3, t_points_full, X0, t_end)
    rel = abs(l_harness - l_baseline) / (abs(l_baseline) + 1e-300)
    results["G-PARITY"] = _gate(
        "G-PARITY", rel <= 1e-10,
        f"rel diff harness vs baseline loss = {rel:.2e} (<=1e-10)")

    # ---- G-STATEVEC: direct-numpy z_vector == Qiskit reference, <=1e-12 -----
    # z_vector() uses the direct-numpy simulator; here we re-derive the SAME
    # observables via the Qiskit Statevector reference path and require machine
    # precision agreement across random (n_layers, theta, t). This pins the
    # fast hot path to the audited Qiskit build gate-for-gate.
    max_sv = 0.0
    sv_rng = np.random.default_rng(20240607)
    for _ in range(24):
        nl = int(sv_rng.integers(1, 5))
        th_sv = sv_rng.uniform(0.0, 2.0 * np.pi, n_params(nl))
        t_sv = float(sv_rng.uniform(0.0, t_end))
        z_fast = z_vector(t_sv, th_sv, t_end, nl)
        z_ref = _z_expectations_from_state(
            build_qpinn_circuit_L(t_sv, th_sv, t_end, nl), (0, 1, 2))
        max_sv = max(max_sv, float(np.max(np.abs(z_fast - z_ref))))
    results["G-STATEVEC"] = _gate(
        "G-STATEVEC", max_sv <= 1e-12,
        f"max |numpy z_vector - Qiskit ref| = {max_sv:.2e} (<=1e-12)")

    # ---- G-PS-FD: PS loss grad == central FD (eps=1e-6), rel <=1e-4 ----
    theta_g = init_theta(999, 3)
    g_ps = ps_loss_gradient(theta_g, t_points_small, X0, t_end, 3)
    g_fd = central_fd_gradient(cost_paper, theta_g, t_points_small, X0, t_end, 3,
                               eps=1e-6)
    rel_g = float(np.linalg.norm(g_ps - g_fd) / (np.linalg.norm(g_fd) + 1e-300))
    results["G-PS-FD"] = _gate(
        "G-PS-FD", rel_g <= 1e-4,
        f"rel |g_ps - g_fd| = {rel_g:.2e} (<=1e-4)")

    # ---- G-SPSA: mean of >=50 SPSA estimates cos-sim >0.5 with exact grad ----
    theta_s = init_theta(2024, 3)
    g_exact = ps_loss_gradient(theta_s, t_points_small, X0, t_end, 3)
    n_spsa = 50
    rng = np.random.default_rng(7)
    acc = np.zeros_like(theta_s)
    c_spsa = 0.1
    for _ in range(n_spsa):
        gh, _, _ = spsa_gradient(cost_paper, theta_s, t_points_small, X0,
                                 t_end, 3, c_spsa, rng)
        acc += gh
    g_mean = acc / n_spsa
    cos = float(np.dot(g_mean, g_exact)
                / (np.linalg.norm(g_mean) * np.linalg.norm(g_exact) + 1e-300))
    results["G-SPSA"] = _gate(
        "G-SPSA", cos > 0.5,
        f"cos(mean SPSA[{n_spsa}], exact grad) = {cos:.3f} (>0.5)")

    # ---- G-DET: same CLI args + seed twice -> byte-identical result JSON ----
    # no_checkpoint=True keeps this a pure recompute-determinism test (both runs
    # recompute from scratch); resume is exercised separately by G-RESUME.
    tmp = args.out_dir_path / "_selftest_det"
    det_args = argparse.Namespace(
        n_inits=3, n_layers=2, cost="bounded", t_end=t_end, n_train=6,
        seed_base=100, jobs=1, tag="detA", out_dir_path=tmp, mcclean=False,
        checkpoint_dir=None, no_checkpoint=True)
    cmd_variance(det_args)
    hA = json.loads((tmp / "variance_detA.json").read_text())["sha256_no_timing"]
    det_args.tag = "detB"
    cmd_variance(det_args)
    hB = json.loads((tmp / "variance_detB.json").read_text())["sha256_no_timing"]
    results["G-DET"] = _gate(
        "G-DET", hA == hB,
        f"hash(run A)==hash(run B): {hA[:12]} == {hB[:12]}")

    # ---- G-PAR: --jobs 2 parallel == serial on the SAME 2 seeds ----
    specs = [{"seed": 500 + i, "n_layers": 2, "cost": "bounded",
              "t_end": t_end, "n_train": 6} for i in range(2)]
    serial = run_units(_variance_worker, specs, jobs=1)
    parallel = run_units(_variance_worker, specs, jobs=2)
    par_ok = (json.dumps(serial, sort_keys=True, default=float)
              == json.dumps(parallel, sort_keys=True, default=float))
    results["G-PAR"] = _gate(
        "G-PAR", par_ok,
        f"serial 2-seed result == jobs=2 result: {par_ok}")

    # ---- G-BOUNDED: bounded cost in [0,1] on 20 random thetas ----
    lo, hi = 1.0, 0.0
    ok_bounded = True
    for s in range(20):
        th = init_theta(3000 + s, 3)
        c = cost_bounded(th, t_points_small, X0, t_end, 3)
        lo = min(lo, c); hi = max(hi, c)
        if not (0.0 <= c <= 1.0):
            ok_bounded = False
    results["G-BOUNDED"] = _gate(
        "G-BOUNDED", ok_bounded,
        f"bounded cost range over 20 thetas = [{lo:.4f}, {hi:.4f}] subset [0,1]")

    # ---- G-RESUME: a killed+resumed train seed reproduces the uninterrupted
    # result bit-for-bit, and a re-invocation of a finished run is a pure
    # reassembly (recomputes nothing). --------------------------------------
    global _TEST_STOP_AFTER_ITER
    rt = args.out_dir_path / "_selftest_resume"

    def _mk_train_args(tag, out_dir, ckpt_dir=None, no_ckpt=False):
        return argparse.Namespace(
            optimizer="fd", n_iters=10, n_seeds=1, seed_base=999, n_layers=2,
            cost="paper", t_end=t_end, n_train=6, tag=tag, jobs=1,
            out_dir_path=out_dir, max_minutes=10.0, yes=True,
            checkpoint_dir=(str(ckpt_dir) if ckpt_dir else None),
            no_checkpoint=no_ckpt)

    def _final_of(out_dir, tag):
        r = json.loads((out_dir / f"train_fd_{tag}.json").read_text())["runs"][0]
        return np.array(r["theta_final"]), float(r["final_loss"])

    # (a) uninterrupted reference (checkpointing OFF so it is a clean baseline).
    ref_dir = rt / "ref"
    _TEST_STOP_AFTER_ITER = None
    cmd_train(_mk_train_args("ref", ref_dir, no_ckpt=True))
    ref_theta, ref_loss = _final_of(ref_dir, "ref")

    # (b) interrupted run: stop right after iter 4, leaving a partial checkpoint.
    run_dir = rt / "run"
    ck_dir = rt / "ckpt"
    interrupted = False
    _TEST_STOP_AFTER_ITER = 4
    try:
        cmd_train(_mk_train_args("run", run_dir, ckpt_dir=ck_dir))
    except _TrainInterrupted as e:
        interrupted = True
        killed_at = e.done_iters
    _TEST_STOP_AFTER_ITER = None
    part = _load_unit(ck_dir, 999)
    partial_ok = (interrupted and part is not None
                  and not part.get("complete")
                  and part.get("done_iters") == 4)

    # (c) resume: same config + same checkpoint dir -> continue iters 5..10.
    cmd_train(_mk_train_args("run", run_dir, ckpt_dir=ck_dir))
    res_theta, res_loss = _final_of(run_dir, "run")
    fin = _load_unit(ck_dir, 999)
    resumed_complete = bool(fin and fin.get("complete")
                            and fin.get("done_iters") == 10)

    d_theta = float(np.max(np.abs(res_theta - ref_theta)))
    rel_loss = abs(res_loss - ref_loss) / (abs(ref_loss) + 1e-300)

    # (d) re-invoke the finished run: pure reassembly, identical final JSON,
    # nothing recomputed. Compare sha256_no_timing of the two output JSONs.
    h_run1 = json.loads((run_dir / "train_fd_run.json").read_text())["sha256_no_timing"]
    cmd_train(_mk_train_args("run", run_dir, ckpt_dir=ck_dir))
    res2_theta, res2_loss = _final_of(run_dir, "run")
    h_run2 = json.loads((run_dir / "train_fd_run.json").read_text())["sha256_no_timing"]
    noop_ok = (h_run1 == h_run2
               and float(np.max(np.abs(res2_theta - res_theta))) == 0.0)

    resume_ok = (partial_ok and resumed_complete and noop_ok
                 and d_theta <= 1e-10 and rel_loss <= 1e-10)
    results["G-RESUME"] = _gate(
        "G-RESUME", resume_ok,
        f"killed@iter{killed_at if interrupted else '?'}/10, resumed->10; "
        f"max|dtheta|={d_theta:.2e} rel|dloss|={rel_loss:.2e} (<=1e-10); "
        f"re-invoke no-op hash-match={h_run1[:8] == h_run2[:8]}")

    all_pass = all(results.values())
    print("-" * 70)
    print(f"SELF-TEST: {'ALL PASS' if all_pass else 'FAILURES PRESENT'} "
          f"({sum(results.values())}/{len(results)})")
    # persist a summary too
    save_result("selftest", args.tag,
                {"mode": "selftest", "config": {"n_qubits": N_QUBITS},
                 "gates": {k: bool(v) for k, v in results.items()},
                 "all_pass": bool(all_pass)},
                args.out_dir_path)
    return 0 if all_pass else 1


# ==========================================================================
# CLI.
# ==========================================================================

def _add_common(p):
    p.add_argument("--tag", default="dev", help="output filename tag")
    p.add_argument("--jobs", type=int, default=1, help="parallel workers")
    p.add_argument("--out-dir", dest="out_dir", default=None,
                   help="override results dir (default results/r1)")
    p.add_argument("--max-minutes", type=float, default=10.0,
                   help="abort train/jacobian if projected wall time exceeds "
                        "this (unless --yes)")
    p.add_argument("--yes", action="store_true",
                   help="proceed even if the projected wall time exceeds "
                        "--max-minutes")
    # Checkpointing is ON by default (transparent resume when the same
    # out_dir/config is reused). These only override where or whether.
    p.add_argument("--checkpoint-dir", dest="checkpoint_dir", default=None,
                   help="override the per-unit checkpoint dir (default "
                        "<out_dir>/_checkpoints/<mode>_<config_hash>)")
    p.add_argument("--no-checkpoint", dest="no_checkpoint", action="store_true",
                   help="disable checkpointing/resume entirely (recompute all "
                        "units, write nothing incremental)")


def _resolve_out_dir(args):
    args.out_dir_path = Path(args.out_dir) if args.out_dir else RESULTS_DIR


def build_parser():
    ap = argparse.ArgumentParser(description="R1 QPINN experiment harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pv = sub.add_parser("variance", help="E1 gradient-variance / barren plateau")
    pv.add_argument("--n-inits", type=int, default=3)
    pv.add_argument("--n-layers", type=int, default=3)
    pv.add_argument("--cost", choices=["paper", "bounded", "both"],
                    default="bounded")
    pv.add_argument("--seed-base", type=int, default=0)
    pv.add_argument("--t-end", type=float, default=DEFAULT_T_END)
    pv.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    pv.add_argument("--mcclean", action="store_true",
                    help="also compute the McClean-comparable fixed-circuit "
                         "variance (exact PS, bounded cost; n_train-independent)")
    _add_common(pv)
    pv.set_defaults(func=cmd_variance)

    pt = sub.add_parser("train", help="E2 optimizer-independence training")
    pt.add_argument("--optimizer", choices=["fd", "spsa", "ps", "layerwise"],
                    required=True)
    pt.add_argument("--n-iters", type=int, default=200)
    pt.add_argument("--n-seeds", type=int, default=1)
    pt.add_argument("--seed-base", type=int, default=0)
    pt.add_argument("--n-layers", type=int, default=3)
    pt.add_argument("--cost", choices=["paper", "bounded"], default="paper")
    pt.add_argument("--t-end", type=float, default=DEFAULT_T_END)
    pt.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    _add_common(pt)
    pt.set_defaults(func=cmd_train)

    pj = sub.add_parser("jacobian", help="E3 Jacobian effective dimension")
    pj.add_argument("--n-inits", type=int, default=3)
    pj.add_argument("--n-layers", type=int, default=3)
    pj.add_argument("--seed-base", type=int, default=0)
    pj.add_argument("--params-file", default=None,
                    help="optional JSON file with a theta vector")
    pj.add_argument("--t-end", type=float, default=DEFAULT_T_END)
    pj.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    _add_common(pj)
    pj.set_defaults(func=cmd_jacobian)

    pf = sub.add_parser("fourier", help="E3 Fourier expressivity vs target")
    pf.add_argument("--n-draws", type=int, default=8)
    pf.add_argument("--n-layers", type=int, default=3)
    pf.add_argument("--n-grid", type=int, default=512)
    pf.add_argument("--seed-base", type=int, default=0)
    pf.add_argument("--t-end", type=float, default=DEFAULT_T_END)
    _add_common(pf)
    pf.set_defaults(func=cmd_fourier)

    ps = sub.add_parser("self-test", help="run all correctness gates")
    _add_common(ps)
    ps.set_defaults(func=cmd_selftest)

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    _resolve_out_dir(args)
    rc = args.func(args)
    return int(rc) if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
