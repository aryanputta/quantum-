"""
QUBO ↔ Ising Hamiltonian converter.

Conversion formula
------------------
Given a QUBO energy  E_QUBO = x^T Q x  with x_i ∈ {0,1},
substitute  x_i = (1 + s_i) / 2,  s_i ∈ {-1, +1}.

We first symmetrise Q:
    Q_sym[i,j] = Q_sym[j,i] = (Q[i,j] + Q[j,i]) / 2   for i ≠ j
    Q_sym[i,i] = Q[i,i]

After substitution, the Ising parameters are:

    J_{ij}  = Q_sym[i,j] / 2              (i < j)
    h_i     = Q_sym[i,i] / 2  +  Σ_{j≠i} Q_sym[i,j] / 2
    offset  = Σ_i Q_sym[i,i] / 2  +  Σ_{i<j} Q_sym[i,j] / 2

Inverse:

    Q_sym[i,j]  = 2 J_{ij}               (i ≠ j)
    Q_sym[i,i]  = 2 h_i  −  2 Σ_{j≠i} J_{min(i,j),max(i,j)}

The roundtrip  QUBO → Ising → QUBO  is exact to floating-point precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .energy import (
    evaluate_qubo_energy,
    evaluate_ising_energy,
    qubo_to_ising_bitstring,
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class IsingResult:
    """
    Result of the QUBO-to-Ising conversion.

    Attributes
    ----------
    J : dict mapping (i, j) → coupling strength, i < j.
    h : dict mapping i → local field.
    offset : float  constant energy offset.
    n_spins : int   number of spin variables.
    """
    J: dict
    h: dict
    offset: float
    n_spins: int


# ---------------------------------------------------------------------------
# Converter
# ---------------------------------------------------------------------------

class IsingConverter:
    """Convert between QUBO matrices and Ising Hamiltonians."""

    # ------------------------------------------------------------------
    def qubo_to_ising(self, Q: np.ndarray) -> IsingResult:
        """
        Convert an (n×n) QUBO matrix to an Ising Hamiltonian.

        The input Q may be upper-triangular, lower-triangular, or fully
        symmetric — it is symmetrised internally.

        Parameters
        ----------
        Q : np.ndarray, shape (n, n).

        Returns
        -------
        IsingResult
        """
        Q = np.asarray(Q, dtype=float)
        n = Q.shape[0]
        if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
            raise ValueError(f"Q must be a square 2-D array; got shape {Q.shape}.")

        # Symmetrise: off-diagonal average, diagonal unchanged
        Q_sym = np.zeros((n, n), dtype=float)
        for i in range(n):
            Q_sym[i, i] = Q[i, i]
            for j in range(i + 1, n):
                sym_val = (Q[i, j] + Q[j, i]) / 2.0
                Q_sym[i, j] = sym_val
                Q_sym[j, i] = sym_val

        # J_{ij} = Q_sym[i,j] / 2   (i < j)
        #
        # Derivation: substituting x_i = (1 + s_i)/2 into E = x^T Q x,
        # the coupling coefficient on s_i*s_j (i<j) is 2*Q_sym[i,j] * 1/4 = Q_sym[i,j]/2.
        J: dict[tuple[int, int], float] = {}
        for i in range(n):
            for j in range(i + 1, n):
                val = Q_sym[i, j] / 2.0
                if val != 0.0:
                    J[(i, j)] = val

        # h_i = Q_sym[i,i] / 2  +  Σ_{j≠i} Q_sym[i,j] / 2
        #
        # Linear coefficient on s_i from diagonal: Q_sym[i,i]/2.
        # Linear coefficient on s_i from cross-term (i,j): Q_sym[i,j]/2.
        h: dict[int, float] = {}
        for i in range(n):
            hi = Q_sym[i, i] / 2.0
            for j in range(n):
                if j != i:
                    hi += Q_sym[i, j] / 2.0
            h[i] = hi

        # offset = Σ_i Q_sym[i,i] / 2  +  Σ_{i<j} Q_sym[i,j] / 2
        #
        # Constant terms: Q_sym[i,i]*(1+s_i)/2 contributes Q_sym[i,i]/2;
        # cross-terms 2*Q_sym[i,j]*(1+s_i+s_j+s_is_j)/4 contribute Q_sym[i,j]/2.
        offset = 0.0
        for i in range(n):
            offset += Q_sym[i, i] / 2.0
        for i in range(n):
            for j in range(i + 1, n):
                offset += Q_sym[i, j] / 2.0

        return IsingResult(J=J, h=h, offset=offset, n_spins=n)

    # ------------------------------------------------------------------
    def ising_to_qubo(self, result: IsingResult) -> np.ndarray:
        """
        Convert an IsingResult back to a QUBO matrix.

        Inverse of :meth:`qubo_to_ising`.  The returned Q is symmetric
        (not upper-triangular) so that the roundtrip is numerically exact.

        Parameters
        ----------
        result : IsingResult

        Returns
        -------
        np.ndarray, shape (n, n), symmetric.
        """
        n = result.n_spins
        Q = np.zeros((n, n), dtype=float)

        # Recover Q_sym from J and h:
        #   J_{ij} = Q_sym[i,j] / 2  →  Q_sym[i,j] = 2 J_{ij}
        #   h_i = Q_sym[i,i]/2 + Σ_{j≠i} Q_sym[i,j]/2
        #       = Q_sym[i,i]/2 + Σ_{j≠i} J_{ij}
        #       →  Q_sym[i,i] = 2(h_i - Σ_{j≠i} J_{ij})
        #
        # where J_{ij} is defined symmetrically (J_{ji} = J_{ij}).

        # Fill off-diagonal
        for (i, j), Jij in result.J.items():
            Q[i, j] = 2.0 * Jij
            Q[j, i] = 2.0 * Jij

        # Fill diagonal
        # From h_i = Q_sym[i,i]/2 + sum_{j!=i} Q_sym[i,j]/2
        #          = Q_sym[i,i]/2 + sum_{j!=i} J[i,j]
        # → Q_sym[i,i] = 2*(h_i - sum_{j!=i} J[i,j])
        for i in range(n):
            hi = result.h.get(i, 0.0)
            sum_J = sum(
                result.J.get((min(i, j), max(i, j)), 0.0)
                for j in range(n) if j != i
            )
            Q[i, i] = 2.0 * (hi - sum_J)

        return Q

    # ------------------------------------------------------------------
    def to_pauli_op_dict(self, result: IsingResult) -> dict:
        """
        Convert an IsingResult to a Pauli operator dictionary for Qiskit.

        Returns
        -------
        dict with keys:
          ``'ZZ'``     : list of (i, j, J_ij) tuples — two-qubit ZZ terms.
          ``'Z'``      : list of (i, h_i) tuples — single-qubit Z terms.
          ``'offset'`` : float — constant energy offset.
        """
        zz_terms = [(i, j, Jij) for (i, j), Jij in result.J.items()]
        z_terms  = [(i, hi) for i, hi in result.h.items()]
        return {
            "ZZ": zz_terms,
            "Z": z_terms,
            "offset": result.offset,
        }

    # ------------------------------------------------------------------
    def roundtrip_check(self, Q: np.ndarray, tol: float = 1e-8) -> bool:
        """
        Convert Q → Ising → Q′ and check that Q′ ≈ Q_sym.

        Note: the inverse always produces a *symmetric* matrix, so the check
        compares against the symmetrised version of the input.

        Parameters
        ----------
        Q   : np.ndarray, shape (n, n).
        tol : absolute tolerance for element-wise comparison.

        Returns
        -------
        bool  — True if the roundtrip is accurate within ``tol``.
        """
        Q = np.asarray(Q, dtype=float)
        n = Q.shape[0]

        # Symmetrise the reference
        Q_sym = np.zeros_like(Q)
        for i in range(n):
            Q_sym[i, i] = Q[i, i]
            for j in range(i + 1, n):
                sym_val = (Q[i, j] + Q[j, i]) / 2.0
                Q_sym[i, j] = sym_val
                Q_sym[j, i] = sym_val

        ising = self.qubo_to_ising(Q)
        Q_rt  = self.ising_to_qubo(ising)

        return bool(np.allclose(Q_rt, Q_sym, atol=tol, rtol=0.0))
