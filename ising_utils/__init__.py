"""
ising_utils package
===================

Utilities for converting QUBO problems to Ising Hamiltonians and evaluating
energies, for use with quantum solvers (D-Wave, Qiskit, etc.).

Public API
----------
IsingConverter
    Converts between QUBO matrices and Ising Hamiltonians (QUBO ↔ Ising).
IsingResult
    Dataclass holding J couplings, h fields, energy offset, and spin count.
evaluate_ising_energy
    Compute  E = Σ J_ij s_i s_j + Σ h_i s_i + offset.
evaluate_qubo_energy
    Compute  E = x^T Q x.
"""

from .converter import IsingConverter, IsingResult
from .energy import evaluate_ising_energy, evaluate_qubo_energy

__all__ = [
    "IsingConverter",
    "IsingResult",
    "evaluate_ising_energy",
    "evaluate_qubo_energy",
]
