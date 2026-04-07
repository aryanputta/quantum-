"""
Energy evaluation utilities for QUBO and Ising Hamiltonians.

Functions
---------
evaluate_qubo_energy       : E = x^T Q x
evaluate_ising_energy      : E = Σ_{i<j} J_ij s_i s_j + Σ_i h_i s_i + offset
qubo_to_ising_bitstring    : {0,1}^n  →  {-1,+1}^n
ising_to_qubo_bitstring    : {-1,+1}^n → {0,1}^n
verify_energy_equivalence  : exhaustive check over all 2^n bitstrings (small n)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Bitstring conversions
# ---------------------------------------------------------------------------

def qubo_to_ising_bitstring(x: np.ndarray) -> np.ndarray:
    """
    Convert a QUBO bitstring x ∈ {0,1}^n to an Ising spin vector s ∈ {-1,+1}^n.

    Uses the relation  s_i = 2 x_i - 1  (i.e. 0 → -1, 1 → +1).

    Parameters
    ----------
    x : array-like, shape (n,), dtype int or float with values in {0, 1}.

    Returns
    -------
    s : np.ndarray, shape (n,), values in {-1.0, +1.0}.
    """
    x = np.asarray(x, dtype=float)
    return 2.0 * x - 1.0


def ising_to_qubo_bitstring(s: np.ndarray) -> np.ndarray:
    """
    Convert an Ising spin vector s ∈ {-1,+1}^n to a QUBO bitstring x ∈ {0,1}^n.

    Uses the relation  x_i = (1 + s_i) / 2  (i.e. -1 → 0, +1 → 1).

    Parameters
    ----------
    s : array-like, shape (n,), dtype int or float with values in {-1, +1}.

    Returns
    -------
    x : np.ndarray, shape (n,), values in {0.0, 1.0}.
    """
    s = np.asarray(s, dtype=float)
    return (1.0 + s) / 2.0


# ---------------------------------------------------------------------------
# Energy functions
# ---------------------------------------------------------------------------

def evaluate_qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """
    Evaluate the QUBO energy  E = x^T Q x.

    Works for upper-triangular, lower-triangular, or full symmetric Q.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n).
    x : array-like, shape (n,), values in {0, 1}.

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    Q = np.asarray(Q, dtype=float)
    return float(x @ Q @ x)


def evaluate_ising_energy(
    J: dict,
    h: dict,
    offset: float,
    s: np.ndarray,
) -> float:
    """
    Evaluate the Ising energy  E = Σ_{i<j} J_{ij} s_i s_j + Σ_i h_i s_i + offset.

    Parameters
    ----------
    J      : dict mapping (i, j) → coupling strength (convention: i < j).
    h      : dict mapping i → local field.
    offset : constant energy offset.
    s      : array-like, shape (n,), spin values in {-1, +1}.

    Returns
    -------
    float
    """
    s = np.asarray(s, dtype=float)
    energy = offset

    # Two-body terms
    for (i, j), Jij in J.items():
        energy += Jij * s[i] * s[j]

    # One-body terms
    for i, hi in h.items():
        energy += hi * s[i]

    return float(energy)


# ---------------------------------------------------------------------------
# Equivalence verification
# ---------------------------------------------------------------------------

def verify_energy_equivalence(
    Q: np.ndarray,
    x: np.ndarray,  # kept for API compatibility but not used in exhaustive mode
    ising_result,
    max_n: int = 20,
) -> bool:
    """
    Verify that the QUBO and Ising formulations give the same energy for every
    possible bitstring (exhaustive, 2^n evaluations).

    This is only practical for small n (≤ ``max_n``).

    Parameters
    ----------
    Q            : np.ndarray QUBO matrix, shape (n, n).
    x            : not used in exhaustive mode (kept for API symmetry).
    ising_result : IsingResult (duck-typed; needs .J, .h, .offset).
    max_n        : maximum n for exhaustive check.

    Returns
    -------
    bool  — True iff QUBO and Ising energies agree within 1 × 10⁻⁸ for all
            bitstrings.

    Raises
    ------
    ValueError  if n > max_n.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    if n > max_n:
        raise ValueError(
            f"n={n} exceeds max_n={max_n}. Use check_energy_equivalence_sampled "
            "for large problems."
        )

    tol = 1e-8
    for bits in range(2 ** n):
        x_bits = np.array([(bits >> i) & 1 for i in range(n)], dtype=float)
        e_qubo = evaluate_qubo_energy(Q, x_bits)
        s_bits = qubo_to_ising_bitstring(x_bits)
        e_ising = evaluate_ising_energy(
            ising_result.J, ising_result.h, ising_result.offset, s_bits
        )
        if abs(e_qubo - e_ising) > tol:
            return False
    return True
