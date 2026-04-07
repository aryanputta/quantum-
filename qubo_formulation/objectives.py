"""
Objective functions for QUBO formulation of routing problems.

Each objective returns (linear_terms, quadratic_terms) where:
  linear_terms:    dict mapping variable_index -> coefficient
  quadratic_terms: dict mapping (i, j) with i < j -> coefficient
"""

from __future__ import annotations
from typing import Any

import networkx as nx
import numpy as np


def _edge_attr(G: nx.Graph, u: Any, v: Any, attr: str, default: float = 0.0) -> float:
    """Safely read an edge attribute, falling back to default."""
    return float(G[u][v].get(attr, default))


def latency_objective(
    G: nx.Graph,
    variables: dict,
) -> tuple[dict, dict]:
    """
    Build linear latency objective terms.

    Reads the ``latency`` attribute on each edge.  The objective is purely
    linear (no quadratic cross-terms) because latency is a per-edge quantity.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose edges carry a ``latency`` attribute (defaults to 1.0).
    variables : dict
        Mapping ``edge_tuple -> variable_index``.

    Returns
    -------
    linear_terms : dict  {var_idx: coeff}
    quadratic_terms : dict  {(i, j): coeff}  — empty for this objective
    """
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}

    for (u, v), idx in variables.items():
        lat = _edge_attr(G, u, v, "latency", default=1.0)
        linear[idx] = linear.get(idx, 0.0) + lat

    return linear, quadratic


def congestion_objective(
    G: nx.Graph,
    variables: dict,
) -> tuple[dict, dict]:
    """
    Build linear congestion objective terms.

    Reads the ``congestion`` attribute on each edge (fraction of capacity in
    use, ∈ [0, 1]).  Higher congestion → higher cost.

    Parameters
    ----------
    G : networkx.Graph
        Graph whose edges carry a ``congestion`` attribute (defaults to 0.0).
    variables : dict
        Mapping ``edge_tuple -> variable_index``.

    Returns
    -------
    linear_terms : dict  {var_idx: coeff}
    quadratic_terms : dict  {(i, j): coeff}  — empty for this objective
    """
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}

    for (u, v), idx in variables.items():
        cong = _edge_attr(G, u, v, "congestion", default=0.0)
        linear[idx] = linear.get(idx, 0.0) + cong

    return linear, quadratic


def loss_objective(
    G: nx.Graph,
    variables: dict,
) -> tuple[dict, dict]:
    """
    Build linear packet-loss objective terms.

    Reads the ``loss`` attribute on each edge (loss probability ∈ [0, 1]).

    Parameters
    ----------
    G : networkx.Graph
        Graph whose edges carry a ``loss`` attribute (defaults to 0.0).
    variables : dict
        Mapping ``edge_tuple -> variable_index``.

    Returns
    -------
    linear_terms : dict  {var_idx: coeff}
    quadratic_terms : dict  {(i, j): coeff}  — empty for this objective
    """
    linear: dict[int, float] = {}
    quadratic: dict[tuple[int, int], float] = {}

    for (u, v), idx in variables.items():
        loss = _edge_attr(G, u, v, "loss", default=0.0)
        linear[idx] = linear.get(idx, 0.0) + loss

    return linear, quadratic


def combine_objectives(
    objectives: list[tuple[dict, dict]],
    weights: list[float],
) -> tuple[dict, dict]:
    """
    Combine multiple (linear, quadratic) objective dicts with scalar weights.

    Parameters
    ----------
    objectives : list of (linear_terms, quadratic_terms)
        Each element is a tuple returned by one of the objective functions.
    weights : list of float
        Scalar weight applied to each corresponding objective.

    Returns
    -------
    linear_terms : dict  {var_idx: coeff}
    quadratic_terms : dict  {(i, j): coeff}

    Raises
    ------
    ValueError
        If ``len(objectives) != len(weights)``.
    """
    if len(objectives) != len(weights):
        raise ValueError(
            f"Number of objectives ({len(objectives)}) must match "
            f"number of weights ({len(weights)})."
        )

    combined_linear: dict[int, float] = {}
    combined_quadratic: dict[tuple[int, int], float] = {}

    for (linear, quadratic), w in zip(objectives, weights):
        for idx, coeff in linear.items():
            combined_linear[idx] = combined_linear.get(idx, 0.0) + w * coeff
        for pair, coeff in quadratic.items():
            i, j = (min(pair), max(pair))
            combined_quadratic[(i, j)] = combined_quadratic.get((i, j), 0.0) + w * coeff

    return combined_linear, combined_quadratic
