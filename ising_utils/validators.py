"""
Validation utilities for IsingResult objects.

Functions
---------
validate_ising_result             : structural and numerical checks.
check_energy_equivalence_sampled  : stochastic equivalence check vs QUBO.
"""

from __future__ import annotations

import numpy as np

from .converter import IsingResult
from .energy import (
    evaluate_qubo_energy,
    evaluate_ising_energy,
    qubo_to_ising_bitstring,
)


def validate_ising_result(result: IsingResult) -> dict:
    """
    Run structural and numerical sanity checks on an IsingResult.

    Checks performed
    ----------------
    ``type_check``
        J and h are dicts, offset is float-like, n_spins is a positive int.
    ``j_keys_ordered``
        Every key (i, j) in J satisfies i < j.
    ``j_indices_in_range``
        All indices in J are in [0, n_spins).
    ``h_indices_in_range``
        All indices in h are in [0, n_spins).
    ``no_nan_inf_j``
        No NaN or Inf values in J.
    ``no_nan_inf_h``
        No NaN or Inf values in h.
    ``offset_finite``
        offset is finite.

    Parameters
    ----------
    result : IsingResult

    Returns
    -------
    dict with keys ``'is_valid'`` (bool) and ``'checks'`` (dict of bool).
    """
    checks: dict[str, bool] = {}
    n = result.n_spins

    # Type / basic structure
    checks["type_check"] = (
        isinstance(result.J, dict)
        and isinstance(result.h, dict)
        and isinstance(n, int)
        and n > 0
        and np.isfinite(float(result.offset))
    )

    # J key ordering
    checks["j_keys_ordered"] = all(i < j for (i, j) in result.J.keys())

    # Index bounds
    checks["j_indices_in_range"] = all(
        0 <= i < n and 0 <= j < n for (i, j) in result.J.keys()
    )
    checks["h_indices_in_range"] = all(0 <= i < n for i in result.h.keys())

    # Finite values
    checks["no_nan_inf_j"] = all(np.isfinite(v) for v in result.J.values())
    checks["no_nan_inf_h"] = all(np.isfinite(v) for v in result.h.values())
    checks["offset_finite"] = bool(np.isfinite(float(result.offset)))

    is_valid = all(checks.values())
    return {"is_valid": is_valid, "checks": checks}


def check_energy_equivalence_sampled(
    Q: np.ndarray,
    ising_result: IsingResult,
    n_samples: int = 1000,
    tol: float = 1e-6,
    seed: int = 0,
) -> dict:
    """
    Stochastically verify that QUBO and Ising energies agree.

    Draws ``n_samples`` random bitstrings, evaluates both the QUBO energy
    and the corresponding Ising energy, and checks they match within ``tol``.

    Parameters
    ----------
    Q            : np.ndarray, shape (n, n) QUBO matrix.
    ising_result : IsingResult.
    n_samples    : number of random samples.
    tol          : absolute tolerance for energy comparison.
    seed         : random seed for reproducibility.

    Returns
    -------
    dict with keys:
      ``'all_match'``       : bool — True if every sample matches.
      ``'n_mismatches'``    : int  — number of failing samples.
      ``'max_abs_error'``   : float — maximum |E_QUBO - E_Ising|.
      ``'mean_abs_error'``  : float — mean |E_QUBO - E_Ising|.
      ``'n_samples'``       : int  — number of samples tested.
      ``'tolerance'``       : float — tolerance used.
    """
    Q = np.asarray(Q, dtype=float)
    n = Q.shape[0]
    rng = np.random.default_rng(seed)

    abs_errors: list[float] = []
    n_mismatches = 0

    for _ in range(n_samples):
        x = rng.integers(0, 2, size=n).astype(float)
        e_qubo = evaluate_qubo_energy(Q, x)
        s = qubo_to_ising_bitstring(x)
        e_ising = evaluate_ising_energy(
            ising_result.J, ising_result.h, ising_result.offset, s
        )
        err = abs(e_qubo - e_ising)
        abs_errors.append(err)
        if err > tol:
            n_mismatches += 1

    abs_errors_arr = np.array(abs_errors)
    return {
        "all_match": n_mismatches == 0,
        "n_mismatches": n_mismatches,
        "max_abs_error": float(np.max(abs_errors_arr)),
        "mean_abs_error": float(np.mean(abs_errors_arr)),
        "n_samples": n_samples,
        "tolerance": tol,
    }
