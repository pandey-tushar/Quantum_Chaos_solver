"""
Windowed Variational Quantum Circuit (WVQC) — fast batched numpy simulator.

Key optimization: vectorize all training samples through the circuit at once.
State shape: (batch, 2^n_qubits). Gate application via tensor reshape — O(batch * 2^n)
instead of O(batch * 2^n * 2^n) from matrix-matrix multiply.

Task: [state(t-w+1)..state(t)] -> state(t+1)  (same as QRC)
Circuit: 4 qubits, 3 layers, RX/RY/RZ + ring CNOT
Loss: pure MSE — task-symmetric comparison to QRC
"""

import sys, json, time, argparse
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
from data_utils import load_classical_solution


# ── Batched statevector simulator ─────────────────────────────────────────────
# state: complex array (batch, 2**n_qubits)
# Gates are applied via tensor reshape — no full unitary construction needed.

def apply_single(state, gate, qubit, n):
    """Apply 2x2 gate to qubit `qubit` of batched state (batch, 2^n)."""
    batch = state.shape[0]
    s = state.reshape(batch, 2**qubit, 2, 2**(n-qubit-1))
    s = np.einsum('ab,sxby->sxay', gate, s)
    return s.reshape(batch, 2**n)

def apply_cnot(state, ctrl, tgt, n):
    """CNOT gate on batched state."""
    batch = state.shape[0]
    s = state.reshape([batch] + [2]*n)
    # Swap tgt qubit conditioned on ctrl
    axes_ctrl = ctrl + 1  # +1 for batch dim
    axes_tgt  = tgt  + 1
    idx0 = [slice(None)] * (n + 1)
    idx1 = [slice(None)] * (n + 1)
    idx0[axes_ctrl] = 1; idx1[axes_ctrl] = 1
    idx0[axes_tgt]  = 0; idx1[axes_tgt]  = 1
    tmp = s[tuple(idx0)].copy()
    s[tuple(idx0)] = s[tuple(idx1)]
    s[tuple(idx1)] = tmp
    return s.reshape(batch, 2**n)

def rx_mat(t): c=np.cos(t/2); s=np.sin(t/2); return np.array([[c,-1j*s],[-1j*s,c]])
def ry_mat(t): c=np.cos(t/2); s=np.sin(t/2); return np.array([[c,-s],[s,c]])
def rz_mat(t): return np.array([[np.exp(-1j*t/2),0],[0,np.exp(1j*t/2)]])

SCALES  = np.array([20., 30., 25.])
OFFSETS = np.array([ 0.,  0., 25.])

def encode(window_batch):
    """window_batch: (batch, window*3) -> enc_t0, enc_t1 each (batch, 3)"""
    states = window_batch.reshape(window_batch.shape[0], -1, 3)  # (B, w, 3)
    enc_t0 = np.clip(states[:, -1] / SCALES, -1, 1) * np.pi     # most recent
    enc_t1 = np.clip(states[:, -2] / SCALES, -1, 1) * np.pi     # one step back
    return enc_t0, enc_t1

def forward(theta, X, n_qubits=4, n_layers=3):
    """
    Batched forward pass.
    X: (batch, window*3)
    Returns predicted (x,y,z): (batch, 3)
    """
    batch = X.shape[0]
    enc_t0, enc_t1 = encode(X)  # (batch, 3)

    # Initial state |0000> for each sample
    state = np.zeros((batch, 2**n_qubits), dtype=complex)
    state[:, 0] = 1.0

    idx = 0
    for layer in range(n_layers):
        # Data re-uploading — different angle per sample, so apply per-sample
        for q, col in enumerate([0, 1, 2]):
            for b in range(batch):
                gate = ry_mat(enc_t0[b, col])
                state[b:b+1] = apply_single(state[b:b+1], gate, q, n_qubits)
        # enc_t1 on qubit 3
        for b in range(batch):
            gate = ry_mat(enc_t1[b, 0])
            state[b:b+1] = apply_single(state[b:b+1], gate, 3, n_qubits)

        # Trainable rotations (same theta for all samples — vectorizable)
        for q in range(n_qubits):
            gx = rx_mat(theta[idx]); state = apply_single(state, gx, q, n_qubits); idx+=1
            gy = ry_mat(theta[idx]); state = apply_single(state, gy, q, n_qubits); idx+=1
            gz = rz_mat(theta[idx]); state = apply_single(state, gz, q, n_qubits); idx+=1

        # Ring CNOT
        for q in range(n_qubits):
            state = apply_cnot(state, q, (q+1) % n_qubits, n_qubits)

    # Measure <Z_q> for q in {0,1,2}
    probs = np.abs(state)**2  # (batch, 2^n)
    evs = []
    for q in range(3):
        signs = np.array([(-1)**((i >> (n_qubits-1-q)) & 1) for i in range(2**n_qubits)])
        evs.append(np.sum(probs * signs, axis=1))  # (batch,)
    preds = np.stack(evs, axis=1) * SCALES + OFFSETS  # (batch, 3)
    return np.real(preds)

def mse_loss(theta, X, y):
    return float(np.mean((forward(theta, X) - y)**2))

def param_shift_grad(theta, X, y):
    g = np.zeros(len(theta))
    shift = np.pi / 2
    for i in range(len(theta)):
        tp = theta.copy(); tp[i] += shift
        tm = theta.copy(); tm[i] -= shift
        g[i] = (mse_loss(tp, X, y) - mse_loss(tm, X, y)) / 2
    return g


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data(window=5):
    sol    = load_classical_solution(project_root / "data")
    states = np.column_stack([sol['x'], sol['y'], sol['z']])
    train  = states[sol['t'] <= 3.0]
    test   = states[(sol['t'] > 3.0) & (sol['t'] <= 4.0)]

    def windows(s, w):
        X = np.array([s[i:i+w].flatten() for i in range(len(s)-w)])
        Y = np.array([s[i+w]             for i in range(len(s)-w)])
        return X, Y

    X_tr, y_tr = windows(train, window)
    X_te, y_te = windows(np.vstack([train[-window:], test]), window)
    return X_tr, y_tr, X_te, y_te


# ── Adam ──────────────────────────────────────────────────────────────────────

class Adam:
    def __init__(self, lr=0.05, b1=0.9, b2=0.999, eps=1e-8, decay=0.998):
        self.lr=lr; self.b1=b1; self.b2=b2; self.eps=eps; self.decay=decay
        self.m=self.v=None; self.t=0

    def step(self, theta, grad):
        if self.m is None:
            self.m=np.zeros_like(theta); self.v=np.zeros_like(theta)
        self.t+=1
        self.m=self.b1*self.m+(1-self.b1)*grad
        self.v=self.b2*self.v+(1-self.b2)*grad**2
        mh=self.m/(1-self.b1**self.t); vh=self.v/(1-self.b2**self.t)
        self.lr*=self.decay
        return theta - self.lr*mh/(np.sqrt(vh)+self.eps)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(seed=0, n_iters=200, window=5, lr=0.05):
    print(f"\n{'='*60}")
    print(f"Windowed VQC | seed={seed} | iters={n_iters} | w={window}", flush=True)
    print(f"{'='*60}", flush=True)

    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te = load_data(window)
    print(f"Train: {X_tr.shape}  Test: {X_te.shape}", flush=True)

    # Benchmark speed first
    theta = rng.uniform(-np.pi, np.pi, 3*4*3)
    t0 = time.time(); _ = mse_loss(theta, X_tr, y_tr); fwd=time.time()-t0
    t0 = time.time(); _ = param_shift_grad(theta, X_tr[:5], y_tr[:5]); gs=time.time()-t0
    est = gs/5*len(X_tr)*n_iters/60
    print(f"Forward pass: {fwd:.2f}s | Grad/sample: {gs/5*1000:.1f}ms | Est total: {est:.1f} min", flush=True)

    opt = Adam(lr=lr)
    history = {'loss': [], 'grad_norm': []}
    t0 = time.time()

    for it in range(n_iters):
        loss = mse_loss(theta, X_tr, y_tr)
        grad = param_shift_grad(theta, X_tr, y_tr)
        theta = opt.step(theta, grad)
        gnorm = float(np.linalg.norm(grad))
        history['loss'].append(float(loss)); history['grad_norm'].append(gnorm)
        if it % 10 == 0 or it == n_iters-1:
            print(f"  iter {it:3d}  loss={loss:8.3f}  |grad|={gnorm:.3f}  {time.time()-t0:.0f}s", flush=True)

    elapsed   = time.time() - t0
    train_mse = float(np.mean((forward(theta, X_tr) - y_tr)**2))
    test_mse  = float(np.mean((forward(theta, X_te) - y_te)**2))

    print(f"\n  Train MSE : {train_mse:.4f}")
    print(f"  Test  MSE : {test_mse:.4f}")
    print(f"  Time      : {elapsed:.1f}s  ({elapsed/3600:.2f}h)", flush=True)

    out = project_root / "results" / f"windowed_vqc_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    json.dump({'seed': seed, 'n_iters': n_iters, 'window': window,
               'train_mse': train_mse, 'test_mse': test_mse, 'time_s': elapsed,
               'history': history},
              open(out / "results.json", 'w'), indent=2)
    print(f"  Saved → {out}/results.json")
    return train_mse, test_mse, elapsed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed',   type=int,   default=0)
    ap.add_argument('--iters',  type=int,   default=200)
    ap.add_argument('--window', type=int,   default=5)
    ap.add_argument('--lr',     type=float, default=0.05)
    args = ap.parse_args()
    run(args.seed, args.iters, args.window, args.lr)
