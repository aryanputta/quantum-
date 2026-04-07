"""
Dijkstra shortest-path baseline solver.
"""

from __future__ import annotations
from typing import Any, Optional
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


class DijkstraSolver:
    """
    Shortest-path solver using Dijkstra's algorithm with a composite weight.

    The composite edge weight is::

        w = w_latency * latency + w_congestion * congestion + w_loss * loss

    This is equivalent to the linear part of the QUBO objective, making it a
    natural classical baseline.
    """

    def solve(
        self,
        G: nx.Graph,
        src: Any,
        dst: Any,
        qubo_result=None,
        w_latency: float = 1.0,
        w_congestion: float = 2.0,
        w_loss: float = 3.0,
    ) -> SolverResult:
        """
        Find the shortest path and return a standardised SolverResult.

        Parameters
        ----------
        G : networkx.Graph
            Network graph with ``latency``, ``congestion``, and ``loss`` edge
            attributes.
        src, dst
            Source and destination node identifiers.
        qubo_result : optional
            Object with a ``variable_map`` attribute (``edge_tuple -> int``)
            and optionally a ``Q`` matrix.  When supplied the bitstring is
            aligned to the QUBO variable ordering.
        w_latency, w_congestion, w_loss : float
            Weights for the composite edge cost.

        Returns
        -------
        SolverResult
        """
        t_start = time.perf_counter()

        # ------------------------------------------------------------------ #
        # Build composite weight on edges
        # ------------------------------------------------------------------ #
        def _composite(u, v, data):
            lat = float(data.get("latency", 1.0))
            cong = float(data.get("congestion", 0.0))
            loss = float(data.get("loss", 0.0))
            return w_latency * lat + w_congestion * cong + w_loss * loss

        # ------------------------------------------------------------------ #
        # Run Dijkstra
        # ------------------------------------------------------------------ #
        feasible = True
        try:
            path = nx.shortest_path(G, source=src, target=dst, weight=_composite)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # No path exists — return a zero bitstring with infinite cost
            feasible = False
            path = [src]

        # ------------------------------------------------------------------ #
        # Convert path to bitstring
        # ------------------------------------------------------------------ #
        if qubo_result is not None and hasattr(qubo_result, "variable_map"):
            # QUBOResult.variable_map is {int: (u,v)} — canonical format
            variable_map = qubo_result.variable_map
            n_vars = qubo_result.n_variables
        else:
            # Build a default {int: (u,v)} variable map from the graph edges
            variable_map = {i: edge for i, edge in enumerate(G.edges())}
            n_vars = len(variable_map)

        bitstring = path_to_bitstring(path, variable_map, n_vars)

        # ------------------------------------------------------------------ #
        # Compute routing metrics
        # ------------------------------------------------------------------ #
        routing_cost, latency, congestion_score, loss_score = bitstring_to_routing_cost(
            bitstring, variable_map, G, w_latency, w_congestion, w_loss
        )

        # QUBO energy (if Q matrix is available)
        if qubo_result is not None and hasattr(qubo_result, "Q"):
            best_energy = compute_qubo_energy(qubo_result.Q, bitstring)
        else:
            best_energy = routing_cost

        runtime = time.perf_counter() - t_start

        return SolverResult(
            solver_name="dijkstra",
            best_energy=best_energy,
            best_bitstring=bitstring,
            routing_cost=routing_cost,
            latency=latency,
            congestion_score=congestion_score,
            loss_score=loss_score,
            runtime_seconds=runtime,
            n_iterations=1,
            energy_history=[best_energy],
            feasible=feasible,
            metadata={
                "path": path,
                "path_length": len(path) - 1,
            },
        )
