# Paper 6 NEW DIRECTION: Channel-spectrum learning (2-copy advantage)

**Date:** 2026-05-31
**Script:** `scripts/diag_channel_learning_pilot.py`
**Status:** m=4 pilot CLEAN; m=5 running.

## Why we pivoted here

The trajectory high-body task gave only a marginal, non-robust win (k=4/h=20
only, and even that barely sub-1.0; k=6 failed across target seeds). Two
research rounds + my own analysis identified the root cause:

**A state-injection QRC measuring single channel/state outputs is a SINGLE-COPY
strategy — information-theoretically the same class as the classical learner.
There is no separation unless the QRC holds a coherent resource the classical
baseline lacks.**

The fix (Chen-Cotler-Huang-Li FOCS 2021, arXiv:2111.05881; Huang et al.
Science 2022, arXiv:2112.00778): give the QRC the **2-copy Choi state** of an
unknown channel. Then local reservoir measurements access Bell-basis
information and the channel's Pauli spectrum becomes linearly decodable with
**weight-independent** variance (~1/M), while a single-copy classical shadow
estimator pays **3^w/M** for a weight-w Pauli eigenvalue.

## Task

- Unknown m-qubit Pauli channel Λ(ρ)=Σ_i p_i P_i ρ P_i, rates p~Dirichlet(α).
- Target: the 4^m Pauli eigenvalues λ_i = Σ_j p_j χ(P_i,P_j) (Walsh-Hadamard
  of the rates), grouped by Pauli weight.
- QRC: inject the Choi state (2m qubits, Bell-diagonal) into a q-qubit
  scrambling reservoir, evolve, read local Pauli moments (1+2 body, Z/X/Y),
  M=q·1024 shots. Ridge readout across channels.
- Classical (single-copy): shadow estimate of each λ_i, variance 3^w/M.
  Eigenvalues are mutually UNCORRELATED across random channels (character
  orthogonality), so regression cannot denoise via cross-target correlation
  → classical NRMSE = sqrt(3^w/M)/std(λ_w). This kills the "classical
  amortizes too" counterargument.
- Metric: NRMSE per Pauli weight. Clean win = QRC <1 AND classical >1.

## Fairness (preempting referees)

- Both learners get M channel-uses / measurements. QRC's only extra resource
  is 2-copy (Choi) access measured jointly — the provable separation resource.
- Classical is single-copy with the optimal per-eigenvalue shadow estimator
  (3^w/M is the known floor). The exp(m) separation is for estimating ALL
  eigenvalues / nonlinear functionals; per-eigenvalue the floor is 3^w/M, so
  the per-weight comparison is fair and the gap is real, growing with w.
- TODO for the real result: simulate ACTUAL classical Clifford shadows
  (not just the analytic 3^w/M floor) to be airtight.

## m=4, q=9 pilot (N=400 channels, Dirichlet α=1, M=9216) — CLEAN

| weight | QRC_exact | QRC_shots | Classical | std(λ) | clean win? |
|--------|-----------|-----------|-----------|--------|-----------|
| 1 | 0.012 | 0.014 | 0.270 | 0.125 | no |
| 2 | 0.063 | 0.075 | 0.469 | 0.125 | no |
| 3 | 0.270 | 0.331 | 0.810 | 0.125 | no |
| 4 | **0.704** | **0.744** | **1.401** | 0.125 | **YES** |

Findings:
- The scrambled local readout DOES linearly decode eigenvalues up to w=4
  (my worry that 135 features can't span the 256-dim eigenvalue space was
  wrong — scrambling gives good overlap; ridge captures most signal).
- Shot noise negligible at M=9216 (0.744 vs 0.704).
- Clean separation at w=4: QRC 0.74 < 1 < classical 1.40. Gap grows with weight.

## m=5, q=11 — the decisive test (running)

Expect classical to fail at BOTH w=4 (~1.4) and w=5 (~2.4) due to smaller
eigenvalue std (~1/2^m). Open question: does QRC still decode w=4,5 <1 with
only 198 features over a 1024-dim eigenvalue space? If yes → two-cell headline
with a gap growing in weight. If QRC>1 at w=5, the decoding hits the
feature-count ceiling and we cap the claim at w=4.

## If confirmed — plan to a QTML headline

1. Seed sweep: 3 reservoir seeds × 3 channel-ensemble seeds.
2. Real Clifford-shadow classical baseline (not analytic floor).
3. Vary α (channel density) ∈ {0.3, 1, 3} for robustness.
4. Headline figure: NRMSE vs Pauli weight, QRC flat-ish <1, classical rising
   through 1 — the 2-copy-vs-single-copy signature.
5. Abstract: cite arXiv:2111.05881, arXiv:2112.00778 for the separation;
   arXiv:2604.23743 for the QRC architecture.
