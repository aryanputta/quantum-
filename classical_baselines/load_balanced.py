"""
Load-balanced routing baseline solver.

Finds the top-k shortest paths and selects the one with the best
combined congestion + latency score.
"""

from __future__ import annotations
from typing import Any
import time

import numpy as np
import networkx as nx

from .solver_base import SolverResult
from .routing_utils import (
    path_to_bitstring,
    bitstring_to_routing_cost,
    check_flow_conservation,
    compute_qubo_energy,
)


def _k_shortest_paths(G: nx.Graph, src, dst, k: int, weight_fn):
    """
    Return up to *k* simple shortest paths ordered by composite weight.

    Uses :func:`networkx.shortest_simple_paths` which yields paths in
    non-decreasing order of length; we stop after *k* paths are collected.
    """
    # Build a weight attribute on a copy so networkx can use it
    H = nx.Graph() if not G.is_directed() else nx.DiGraph()
    for u, v, data in G.edges(data=True):
        w = weight_fn(u, v, data)
        H.add_edge(u, v, _w=w)
        if not G.is_directed():
            H.add_edge(v, u, _w=w)

    paths = []
    try:
        for path in nx.shortest_simple_paths(H, src, dst, weight="_w"):
            paths.append(path)
            if len(paths) >= k:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        pass
    return paths


class LoadBalancedSolver:
    """
    Load-balanced path selection solver.

    Enumerates the top-k shortest paths between *src* and *dst* using
    :func:`networkx.shortest_simple_paths`, scores each candidate by::

        score = max_congestion * w_congestion + total_latency * w_latency

    and picks the path with the lowest score.  This encourages choosing
    paths that avoid heavily loaded links while keeping latency reasonable.
    """

    def solve(
        self,
        G: nx.Graph,
        src: Any,
        dst: Any,
        qubo_result=None,
        n_paths: int = 3,
        w_latency: float = 1.0,
        w_congestion: float = 2.0,
        w_loss: float = 3.0,
    ) -> SolverResult:
        """
        Select the best load-balanced path and return a SolverResult.

        Parameters
        ----------
        G : networkx.Graph
            Network graph with ``latency``, ``congestion``, and ``loss`` edge
            attributes.
        src, dst
            Source and destination node identifiers.
        qubo_result : optional
            Object with a ``variable_map`` attribute and optionally a ``Q``
            matrix.
        n_paths : int
            Number of candidate paths to evaluate.
        w_latency, w_congestion, w_loss : float
            Weights used both for candidate generation and final scoring.

        Returns
        -------
        SolverResult
        """
        t_start = time.perf_counter()

        def _composite(u, v, data):
            # Handle multigraph edge data (dict-of-dicts)
            if isinstance(next(iter(data.values()), None), dict):
                data = next(iter(data.values()))
            lat = float(data.get("latency", 1.0))
            cong = float(data.get("congestion", 0.0))
            loss = float(data.get("loss", 0.0))
            return w_latency * lat + w_congestion * cong + w_loss * loss

        # ------------------------------------------------------------------ #
        # Enumerate candidate paths
        # ------------------------------------------------------------------ #
        candidate_paths = _k_shortest_paths(G, src, dst, n_paths, _composite)

        feasible = True
        if not candidate_paths:
            # No path exists
            feasible = False
            best_path = [src]
        else:
            # ---------------------------------------------------------------- #
            # Score each candidate: max_congestion * w_congestion +
            #                       total_latency  * w_latency
            # ---------------------------------------------------------------- #
            def _score_path(path):
                total_lat = 0.0
                max_cong = 0.0
                for u, v in zip(path[:-1], path[1:]):
                    if not G.has_edge(u, v):
                        continue
                    edata = G[u][v]
                    if isinstance(next(iter(edata.values()), None), dict):
                        edata = next(iter(edata.values()))
                    total_lat += float(edata.get("latency", 1.0))
                    cong = float(edata.get("congestion", 0.0))
                    if cong > max_cong:
                        max_cong = cong
                return max_cong * w_congestion + total_lat * w_latency

            best_path = min(candidate_paths, key=_score_path)

        # ------------------------------------------------------------------ #
        # Convert path to bitstring
        # ------------------------------------------------------------------ #
        if qubo_result is not None and hasattr(qubo_result, "variable_map"):
            variable_map = qubo_result.variable_map
            n_vars = qubo_result.n_variables if qubo_result is not None and hasattr(qubo_result, "n_variables") else len(variable_map)
        else:
            variable_map = {i: edge for i, edge in enumerate(G.edges())}
            n_vars = qubo_result.n_variables if qubo_result is not None and hasattr(qubo_result, "n_variables") else len(variable_map)

        bitstring = path_to_bitstring(best_path, variable_map, n_vars)

        # ------------------------------------------------------------------ #
        # Compute routing metrics
        # ------------------------------------------------------------------ #
        routing_cost, latency, congestion_score, loss_score = bitstring_to_routing_cost(
            bitstring, variable_map, G, w_latency, w_congestion, w_loss
        )

        if qubo_result is not None and hasattr(qubo_result, "Q"):
            best_energy = compute_qubo_energy(qubo_result.Q, bitstring)
        else:
            best_energy = routing_cost

        runtime = time.perf_counter() - t_start

        return SolverResult(
            solver_name="load_balanced",
            best_energy=best_energy,
            best_bitstring=bitstring,
            routing_cost=routing_cost,
            latency=latency,
            congestion_score=congestion_score,
            loss_score=loss_score,
            runtime_seconds=runtime,
            n_iterations=len(candidate_paths),
            energy_history=[best_energy],
            feasible=feasible,
            metadata={
                "path": best_path,
                "path_length": len(best_path) - 1,
                "n_candidates_evaluated": len(candidate_paths),
                "n_paths_requested": n_paths,
            },
        )
