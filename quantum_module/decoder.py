"""
QAOA Measurement Decoder.
Converts measurement bitstrings back into routing decisions and evaluates solution quality.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DecodedSolution:
    bitstring: np.ndarray
    qubo_energy: float
    routing_cost: float
    latency: float
    congestion_score: float
    loss_score: float
    feasible: bool
    probability: float          # empirical probability from shot counts
    active_edges: list          # list of edge tuples in selected route
    metadata: dict = field(default_factory=dict)


class QAOADecoder:
    """
    Decode QAOA measurement results into routing solutions.
    """

    def __init__(self, Q: np.ndarray, variable_map: dict, G, src: int, dst: int,
                 w_latency: float = 1.0, w_congestion: float = 2.0, w_loss: float = 3.0):
        self.Q = Q
        self.variable_map = variable_map  # {var_index: (u, v)}
        self.G = G
        self.src = src
        self.dst = dst
        self.w_latency = w_latency
        self.w_congestion = w_congestion
        self.w_loss = w_loss

    def decode_counts(self, counts: dict) -> list[DecodedSolution]:
        """
        Decode all measurement outcomes into routing solutions.

        Parameters
        ----------
        counts : dict mapping bitstring → count

        Returns
        -------
        List of DecodedSolution sorted by qubo_energy (best first)
        """
        total_shots = sum(counts.values())
        solutions = []
        for bitstring_str, count in counts.items():
            x = self._parse_bitstring(bitstring_str)
            if x is None or len(x) != self.Q.shape[0]:
                continue
            energy = float(x @ self.Q @ x)
            routing_cost, lat, cong, loss = self._compute_routing_metrics(x)
            feasible = self._check_feasibility(x)
            active = [self.variable_map[i] for i in range(len(x)) if x[i] == 1
                      and i in self.variable_map]
            prob = count / total_shots
            solutions.append(DecodedSolution(
                bitstring=x,
                qubo_energy=energy,
                routing_cost=routing_cost,
                latency=lat,
                congestion_score=cong,
                loss_score=loss,
                feasible=feasible,
                probability=prob,
                active_edges=active,
            ))
        solutions.sort(key=lambda s: s.qubo_energy)
        return solutions

    def best_feasible(self, solutions: list[DecodedSolution]) -> Optional[DecodedSolution]:
        """Return best feasible solution, or best overall if none feasible."""
        feasible = [s for s in solutions if s.feasible]
        if feasible:
            return feasible[0]
        return solutions[0] if solutions else None

    def feasibility_rate(self, solutions: list[DecodedSolution]) -> float:
        """Fraction of probability mass on feasible solutions."""
        if not solutions:
            return 0.0
        return sum(s.probability for s in solutions if s.feasible)

    def approximation_ratio(self, solutions: list[DecodedSolution], best_known: float) -> float:
        """Compute approximation ratio: best_known / best_found_energy."""
        if not solutions or best_known == 0:
            return 0.0
        best = solutions[0].qubo_energy
        if best == 0:
            return 1.0
        return best_known / best if best > 0 else abs(best_known / best)

    # ------------------------------------------------------------------

    def _parse_bitstring(self, bs: str) -> Optional[np.ndarray]:
        """Parse a Qiskit bitstring (big-endian) to array of ints."""
        try:
            # Qiskit returns big-endian: reverse for qubit-0 at index 0
            return np.array([int(b) for b in reversed(bs)], dtype=int)
        except (ValueError, TypeError):
            return None

    def _compute_routing_metrics(self, x: np.ndarray):
        """Compute routing cost from binary edge assignment."""
        total_lat = 0.0
        total_cong = 0.0
        total_loss = 0.0
        for i, val in enumerate(x):
            if val == 1 and i in self.variable_map:
                u, v = self.variable_map[i]
                if self.G.has_edge(u, v):
                    d = self.G[u][v]
                    total_lat += d.get("latency", 0.0)
                    total_cong += d.get("congestion", 0.0)
                    total_loss += d.get("loss", 0.0)
        cost = (self.w_latency * total_lat
                + self.w_congestion * total_cong
                + self.w_loss * total_loss)
        return cost, total_lat, total_cong, total_loss

    def _check_feasibility(self, x: np.ndarray) -> bool:
        """Check flow conservation constraints."""
        # Build flow per node
        inflow = {}
        outflow = {}
        for i, val in enumerate(x):
            if val == 1 and i in self.variable_map:
                u, v = self.variable_map[i]
                outflow[u] = outflow.get(u, 0) + 1
                inflow[v] = inflow.get(v, 0) + 1

        all_nodes = set(inflow.keys()) | set(outflow.keys())
        for node in all_nodes:
            inf = inflow.get(node, 0)
            outf = outflow.get(node, 0)
            if node == self.src:
                if outf - inf != 1:
                    return False
            elif node == self.dst:
                if inf - outf != 1:
                    return False
            else:
                if inf != outf:
                    return False
        # Also check src has outflow and dst has inflow
        if outflow.get(self.src, 0) == 0 or inflow.get(self.dst, 0) == 0:
            return False
        return True
