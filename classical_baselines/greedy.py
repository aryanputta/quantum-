"""
Greedy hop-by-hop routing baseline solver.
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


class GreedySolver:
    """
    Greedy path-finding solver.

    At each hop the neighbour with the minimum composite edge cost is chosen.
    Cycles are prevented by maintaining a visited-node set.  If the greedy
    walk gets stuck before reaching the destination, the solver falls back to
    Dijkstra's algorithm.
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
        Greedily walk from *src* to *dst* and return a SolverResult.

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
        w_latency, w_congestion, w_loss : float
            Weights for the composite edge cost.

        Returns
        -------
        SolverResult
        """
        t_start = time.perf_counter()

        def _edge_cost(u, v):
            data = G[u][v]
            # Handle multigraph (dict-of-dicts)
            if isinstance(next(iter(data.values()), None), dict):
                data = next(iter(data.values()))
            lat = float(data.get("latency", 1.0))
            cong = float(data.get("congestion", 0.0))
            loss = float(data.get("loss", 0.0))
            return w_latency * lat + w_congestion * cong + w_loss * loss

        # ------------------------------------------------------------------ #
        # Greedy walk
        # ------------------------------------------------------------------ #
        path = [src]
        visited = {src}
        current = src
        used_fallback = False
        feasible = True

        while current != dst:
            neighbours = [
                n for n in G.neighbors(current)
                if n not in visited
            ]

            if not neighbours:
                # Stuck — fall back to Dijkstra from the original source
                used_fallback = True
                try:
                    def _composite(u, v, data):
                        lat = float(data.get("latency", 1.0))
                        cong = float(data.get("congestion", 0.0))
                        loss = float(data.get("loss", 0.0))
                        return w_latency * lat + w_congestion * cong + w_loss * loss

                    path = nx.shortest_path(
                        G, source=src, target=dst, weight=_composite
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    feasible = False
                    path = [src]
                break

            # Pick neighbour with minimum composite cost
            best_neighbour = min(neighbours, key=lambda n: _edge_cost(current, n))
            path.append(best_neighbour)
            visited.add(best_neighbour)
            current = best_neighbour

        # ------------------------------------------------------------------ #
        # Convert path to bitstring
        # ------------------------------------------------------------------ #
        if qubo_result is not None and hasattr(qubo_result, "variable_map"):
            variable_map = qubo_result.variable_map
            n_vars = qubo_result.n_variables if qubo_result is not None and hasattr(qubo_result, "n_variables") else len(variable_map)
        else:
            variable_map = {i: edge for i, edge in enumerate(G.edges())}
            n_vars = qubo_result.n_variables if qubo_result is not None and hasattr(qubo_result, "n_variables") else len(variable_map)

        bitstring = path_to_bitstring(path, variable_map, n_vars)

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
            solver_name="greedy",
            best_energy=best_energy,
            best_bitstring=bitstring,
            routing_cost=routing_cost,
            latency=latency,
            congestion_score=congestion_score,
            loss_score=loss_score,
            runtime_seconds=runtime,
            n_iterations=len(path),
            energy_history=[best_energy],
            feasible=feasible,
            metadata={
                "path": path,
                "path_length": len(path) - 1,
                "used_fallback": used_fallback,
            },
        )
