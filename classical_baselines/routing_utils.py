"""
Shared helper functions used by all classical solvers.

Convention
----------
``variable_map`` throughout this module uses the **index-keyed** format::

    variable_map : dict  {int → (u, v)}

This matches the ``QUBOResult.variable_map`` produced by ``QUBOBuilder``,
the ``QAOADecoder``, and the Ising converter — all of which index variables
by integer and map back to edge tuples.
"""

from __future__ import annotations
from typing import Any

import numpy as np
import networkx as nx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _idx_to_edge(variable_map: dict) -> dict:
    """Return variable_map as {int: (u,v)} regardless of input direction."""
    # Already int-keyed?
    sample_key = next(iter(variable_map), None)
    if sample_key is None:
        return {}
    if isinstance(sample_key, int):
        return variable_map                         # already {int → edge}
    # edge-keyed: invert
    return {v: k for k, v in variable_map.items()}


def _edge_to_idx(variable_map: dict) -> dict:
    """Return variable_map as {(u,v): int} regardless of input direction."""
    sample_key = next(iter(variable_map), None)
    if sample_key is None:
        return {}
    if isinstance(sample_key, int):
        return {v: k for k, v in variable_map.items()}  # invert
    return variable_map                                   # already {edge → int}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def path_to_bitstring(path: list, variable_map: dict, n_vars: int) -> np.ndarray:
    """
    Convert a node path to a binary edge-assignment vector.

    For consecutive nodes ``(n_i, n_{i+1})`` in *path* the corresponding
    variable index in *variable_map* is set to 1.  Both directed orientations
    are checked, so the function works for directed and undirected graphs.

    Parameters
    ----------
    path : list
        Ordered list of node IDs, e.g. ``[0, 2, 5, 3]``.
    variable_map : dict
        ``{int: (u, v)}`` — maps variable index to edge tuple.
    n_vars : int
        Total number of QUBO binary variables.

    Returns
    -------
    np.ndarray
        Binary vector of length *n_vars*.
    """
    # Build edge→index lookup for fast membership testing
    e2i = _edge_to_idx(variable_map)
    bitstring = np.zeros(n_vars, dtype=np.int8)

    for u, v in zip(path[:-1], path[1:]):
        for key in [(u, v), (v, u)]:
            if key in e2i:
                bitstring[e2i[key]] = 1
                break

    return bitstring


def bitstring_to_routing_cost(
    bitstring: np.ndarray,
    variable_map: dict,
    G: nx.Graph,
    w_latency: float = 1.0,
    w_congestion: float = 2.0,
    w_loss: float = 3.0,
) -> tuple:
    """
    Decode a binary bitstring to aggregate routing metrics.

    Parameters
    ----------
    bitstring : np.ndarray
        Binary assignment vector.
    variable_map : dict
        ``{int: (u, v)}`` — maps variable index to edge tuple.
    G : networkx.Graph
        Graph with edge attributes ``latency``, ``congestion``, ``loss``.
    w_latency, w_congestion, w_loss : float
        Objective weights.

    Returns
    -------
    (routing_cost, latency, congestion_score, loss_score) : tuple[float, float, float, float]
    """
    i2e = _idx_to_edge(variable_map)

    total_latency = 0.0
    total_congestion = 0.0
    total_loss = 0.0

    for idx, active in enumerate(bitstring):
        if not active:
            continue
        edge = i2e.get(idx)
        if edge is None:
            continue
        u, v = edge[0], edge[1]
        if G.has_edge(u, v):
            edata = G[u][v]
            # Support both simple graphs and multi-graphs
            if isinstance(next(iter(edata.values()), None), dict):
                edata = next(iter(edata.values()))
            total_latency    += float(edata.get("latency",    1.0))
            total_congestion += float(edata.get("congestion", 0.0))
            total_loss       += float(edata.get("loss",       0.0))

    routing_cost = (
        w_latency    * total_latency
        + w_congestion * total_congestion
        + w_loss       * total_loss
    )
    return routing_cost, total_latency, total_congestion, total_loss


def check_flow_conservation(
    bitstring: np.ndarray,
    variable_map: dict,
    G: nx.Graph,
    src: Any,
    dst: Any,
) -> bool:
    """
    Verify that *bitstring* encodes a valid path from *src* to *dst*.

    Flow-conservation rules:

    * ``src``:  net outflow = +1  (one more outgoing than incoming)
    * ``dst``:  net outflow = -1  (one more incoming than outgoing)
    * others:  net outflow = 0

    Parameters
    ----------
    bitstring : np.ndarray
        Binary assignment vector.
    variable_map : dict
        ``{int: (u, v)}`` — maps variable index to edge tuple.
    G : networkx.Graph
        Network graph.
    src, dst
        Source and destination node identifiers.

    Returns
    -------
    bool
        ``True`` if flow conservation is satisfied and at least one edge
        is active.
    """
    i2e = _idx_to_edge(variable_map)
    net_flow: dict[Any, int] = {}

    for idx, active in enumerate(bitstring):
        if not active:
            continue
        edge = i2e.get(idx)
        if edge is None:
            continue
        u, v = edge[0], edge[1]
        net_flow[u] = net_flow.get(u, 0) + 1   # outgoing +1
        net_flow[v] = net_flow.get(v, 0) - 1   # incoming -1

    for node in set(list(net_flow.keys()) + [src, dst]):
        nf = net_flow.get(node, 0)
        if node == src:
            if nf != 1:
                return False
        elif node == dst:
            if nf != -1:
                return False
        else:
            if nf != 0:
                return False

    return int(np.sum(bitstring)) > 0


def compute_qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    """
    Compute QUBO objective value  E = x^T Q x.

    Parameters
    ----------
    Q : np.ndarray, shape (n, n)
        Upper-triangular or symmetric QUBO matrix.
    x : np.ndarray, shape (n,)
        Binary variable vector.

    Returns
    -------
    float
    """
    x = np.asarray(x, dtype=float)
    return float(x @ Q @ x)
