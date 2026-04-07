"""
Constraint penalty functions for QUBO formulation of routing problems.

All functions return (linear_terms, quadratic_terms) representing the penalty
contribution to the QUBO energy.  The caller multiplies the penalty weight
before or after adding to the main QUBO.

Convention
----------
linear_terms  : dict  {variable_index -> coefficient on x_i}
quadratic_terms : dict  {(i, j) with i < j -> coefficient on x_i * x_j}

The expansion used throughout is:

    (sum_k a_k x_k - b)^2
      = sum_k a_k^2 x_k^2  +  2 * sum_{k<l} a_k a_l x_k x_l  -  2b * sum_k a_k x_k  +  b^2

Because x_i ∈ {0,1} we have x_i^2 = x_i, so the squared linear term collapses
to the diagonal.
"""

from __future__ import annotations
from typing import Any

import networkx as nx


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_linear(terms: dict, idx: int, coeff: float) -> None:
    terms[idx] = terms.get(idx, 0.0) + coeff


def _add_quadratic(terms: dict, i: int, j: int, coeff: float) -> None:
    if i == j:
        raise ValueError("Quadratic term must have i != j.")
    key = (min(i, j), max(i, j))
    terms[key] = terms.get(key, 0.0) + coeff


def _expand_squared_sum(
    coefficients: dict[int, float],
    rhs: float,
    penalty: float,
    linear_out: dict,
    quadratic_out: dict,
) -> None:
    """
    Add penalty * (sum_k coefficients[k] * x_k - rhs)^2 to QUBO dicts.

    Parameters
    ----------
    coefficients : dict {var_idx: coefficient a_k}
    rhs          : right-hand side constant b
    penalty      : scalar multiplier P
    linear_out   : accumulator for linear QUBO terms (modified in-place)
    quadratic_out: accumulator for quadratic QUBO terms (modified in-place)
    """
    idx_list = list(coefficients.items())

    # Diagonal / linear part:  P * (a_k^2 - 2*b*a_k) * x_k  +  P*b^2 (constant, ignored)
    for idx, a_k in idx_list:
        _add_linear(linear_out, idx, penalty * (a_k * a_k - 2.0 * rhs * a_k))

    # Off-diagonal / quadratic part:  2P * a_k * a_l * x_k * x_l
    for m in range(len(idx_list)):
        for n in range(m + 1, len(idx_list)):
            idx_m, a_m = idx_list[m]
            idx_n, a_n = idx_list[n]
            if idx_m != idx_n:
                _add_quadratic(quadratic_out, idx_m, idx_n, penalty * 2.0 * a_m * a_n)


# ---------------------------------------------------------------------------
# Public constraint functions
# ---------------------------------------------------------------------------

def flow_conservation_penalty(
    G: nx.DiGraph,
    variables: dict,
    src: Any,
    dst: Any,
    penalty: float,
) -> tuple[dict, dict]:
    """
    Encode flow conservation as a QUBO penalty.

    For each node v the constraint is:

        outflow(v) - inflow(v) = demand(v)

    where demand(src)=+1, demand(dst)=-1, others=0.

    The penalty term added is::

        penalty * sum_v (outflow(v) - inflow(v) - demand(v))^2

    Parameters
    ----------
    G         : directed graph.
    variables : dict mapping edge tuple (u, v) -> variable index.
    src       : source node.
    dst       : destination node.
    penalty   : penalty weight P.

    Returns
    -------
    linear_terms, quadratic_terms
    """
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}

    demand = {src: 1.0, dst: -1.0}

    for v in G.nodes():
        # Build coefficient dict for (outflow - inflow) at v
        coeffs: dict[int, float] = {}

        for u in G.predecessors(v):
            edge = (u, v)
            if edge in variables:
                idx = variables[edge]
                coeffs[idx] = coeffs.get(idx, 0.0) - 1.0  # inflow  → negative

        for w in G.successors(v):
            edge = (v, w)
            if edge in variables:
                idx = variables[edge]
                coeffs[idx] = coeffs.get(idx, 0.0) + 1.0  # outflow → positive

        # Remove zero-coefficient entries
        coeffs = {k: c for k, c in coeffs.items() if c != 0.0}
        if not coeffs:
            continue

        rhs = demand.get(v, 0.0)
        _expand_squared_sum(coeffs, rhs, penalty, linear, quadratic)

    return linear, quadratic


def capacity_penalty(
    G: nx.DiGraph,
    variables: dict,
    penalty: float,
    congestion_threshold: float = 0.8,
) -> tuple[dict, dict]:
    """
    Penalise routing over edges whose congestion already exceeds a threshold.

    The penalty is a simple linear term for each over-capacity edge:

        penalty * congestion_e * x_e    for each e where congestion_e > threshold

    This encourages the solver to avoid already-loaded links.

    Parameters
    ----------
    G                    : graph with ``congestion`` edge attribute ∈ [0, 1].
    variables            : dict mapping edge tuple -> variable index.
    penalty              : penalty weight P.
    congestion_threshold : edges with congestion above this value are penalised.

    Returns
    -------
    linear_terms, quadratic_terms
    """
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}

    for (u, v), idx in variables.items():
        cong = float(G[u][v].get("congestion", 0.0))
        if cong > congestion_threshold:
            _add_linear(linear, idx, penalty * cong)

    return linear, quadratic


def path_validity_penalty(
    G: nx.DiGraph,
    variables: dict,
    src: Any,
    dst: Any,
    penalty: float,
) -> tuple[dict, dict]:
    """
    Encode path validity constraints as QUBO penalties.

    Three sub-constraints are enforced:

    1. Source must have exactly 1 unit of outflow.
    2. Destination must have exactly 1 unit of inflow.
    3. Intermediate nodes must have balanced flow (inflow == outflow).

    This is equivalent to ``flow_conservation_penalty`` but broken out
    explicitly so callers can apply it separately from the flow balance.

    Parameters
    ----------
    G         : directed graph.
    variables : dict mapping edge tuple (u, v) -> variable index.
    src       : source node.
    dst       : destination node.
    penalty   : penalty weight P.

    Returns
    -------
    linear_terms, quadratic_terms
    """
    # Delegate to flow_conservation_penalty which already encodes the full
    # set of constraints (outflow - inflow = demand at every node).
    return flow_conservation_penalty(G, variables, src, dst, penalty)
