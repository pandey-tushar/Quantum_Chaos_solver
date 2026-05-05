"""
Quantum Chaos Solver

A research framework for benchmarking fixed-Hamiltonian Quantum Reservoir
Computing (QRC) and Quantum Physics-Informed Neural Networks (QPINN) on
chaotic dynamical systems, with extensions to PDE forecast correction.

Version history:
    1.0.0  Initial QRC vs QPINN comparison on Lorenz (2024-12)
    1.1.0  arXiv 2604.23743 release: 5-seed stats, multi-system benchmark
           (Lorenz / Rossler / Lorenz-96), classical ESN baseline (2026-04)
    2.0.0  SWE+QRC PoC: nonlinear shallow water solver, POD reduction,
           autonomous QRC and physics-residual hybrid pipelines, K=q
           matched-complexity scaling experiments (2026-05)
"""

__version__ = "2.0.0"

