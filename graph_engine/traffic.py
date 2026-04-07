"""
TrafficManager — generate traffic demand matrices, update graph congestion,
and apply traffic spikes for stress-testing quantum routing solutions.

The traffic model treats each node pair (i, j) as a potential flow demand.
Congestion on each edge is updated by summing traffic contributions from all
flows that traverse it under shortest-path routing, then normalising by link
bandwidth.
"""

from __future__ import annotations

import copy
import logging
import math
from typing import Any, Collection, Optional

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)

# Maximum allowed congestion value (capped to avoid unphysical saturation)
_MAX_CONGESTION = 1.0
# Minimum bandwidth guard (Mbps) to avoid divide-by-zero
_MIN_BANDWIDTH = 1e-3


def _safe_float(val: Any, default: float) -> float:
    """Return *val* coerced to float, or *default* on failure."""
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


class TrafficManager:
    """Manages traffic demand and edge congestion for routing experiments.

    Parameters
    ----------
    demand_scale:
        Multiplier applied to raw demand values in the traffic matrix.
        Expressed in Mbps.  Default: 100.0 Mbps per unit demand.
    routing_weight:
        Edge attribute used for shortest-path routing when propagating
        traffic flows.  Default: ``'latency'``.
    """

    def __init__(
        self,
        demand_scale: float = 100.0,
        routing_weight: str = "latency",
    ) -> None:
        self.demand_scale = demand_scale
        self.routing_weight = routing_weight

    # ------------------------------------------------------------------
    # Traffic matrix generation
    # ------------------------------------------------------------------

    def generate_traffic_matrix(
        self,
        G: nx.DiGraph,
        intensity: float = 0.5,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate an N×N traffic demand matrix for the graph.

        Each entry ``T[i, j]`` represents the demand (in Mbps) from node *i*
        to node *j*.  Diagonal entries are zero.  The intensity parameter
        scales the overall demand level.

        The demand distribution is log-normal (common in real network
        traffic) with the mean shifted by *intensity*.  Pairs without a
        reachable path receive zero demand.

        Parameters
        ----------
        G:
            Directed graph defining the network topology.
        intensity:
            Overall traffic intensity factor in ``[0, 1]``.  ``0.0`` gives
            near-zero demands; ``1.0`` gives maximum realistic demands.
        seed:
            Random seed for reproducibility.

        Returns
        -------
        np.ndarray
            Float64 matrix of shape ``(N, N)`` where ``N = G.number_of_nodes()``.
            Rows and columns correspond to ``sorted(G.nodes())``.
        """
        if not (0.0 <= intensity <= 1.0):
            raise ValueError(f"intensity must be in [0, 1], got {intensity}")

        nodes = sorted(G.nodes())
        n = len(nodes)
        node_index = {node: idx for idx, node in enumerate(nodes)}
        rng = np.random.default_rng(seed)

        T = np.zeros((n, n), dtype=np.float64)
        if n < 2:
            return T

        # Log-normal base demand: mean ≈ 1.0, moderate variance
        sigma = 0.6
        mu = intensity * self.demand_scale  # mean demand in Mbps

        # Draw raw demands and apply intensity scaling
        raw = rng.lognormal(mean=math.log(max(mu, 1e-3)), sigma=sigma, size=(n, n))
        np.fill_diagonal(raw, 0.0)

        # Zero out unreachable pairs
        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                if i == j:
                    continue
                if not nx.has_path(G, src, dst):
                    raw[i, j] = 0.0

        T = raw.astype(np.float64)
        log.debug(
            "generate_traffic_matrix: n=%d, intensity=%.2f, total_demand=%.1f Mbps",
            n,
            intensity,
            T.sum(),
        )
        return T

    # ------------------------------------------------------------------
    # Congestion update
    # ------------------------------------------------------------------

    def update_congestion(
        self,
        G: nx.DiGraph,
        traffic_matrix: np.ndarray,
    ) -> None:
        """Update edge congestion attributes in *G* based on *traffic_matrix*.

        The method routes each non-zero demand along the shortest weighted
        path and accumulates traffic on every traversed edge.  Edge
        congestion is then recomputed as:

            congestion = min(edge_load_Mbps / bandwidth_Mbps, 1.0)

        The graph is modified **in place**.

        Parameters
        ----------
        G:
            Directed graph whose ``congestion`` edge attributes will be
            updated.
        traffic_matrix:
            Float matrix of shape ``(N, N)`` as returned by
            :meth:`generate_traffic_matrix`.
        """
        nodes = sorted(G.nodes())
        n = len(nodes)

        if traffic_matrix.shape != (n, n):
            raise ValueError(
                f"traffic_matrix shape {traffic_matrix.shape} does not match "
                f"number of nodes {n}."
            )

        # Initialise edge loads to zero
        edge_load: dict[tuple, float] = {(u, v): 0.0 for u, v in G.edges()}

        weight = self.routing_weight

        for i, src in enumerate(nodes):
            for j, dst in enumerate(nodes):
                demand = traffic_matrix[i, j]
                if demand <= 0.0 or src == dst:
                    continue
                try:
                    path = nx.shortest_path(G, src, dst, weight=weight)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    continue

                # Accumulate demand on each edge along the path
                for k in range(len(path) - 1):
                    edge = (path[k], path[k + 1])
                    if edge in edge_load:
                        edge_load[edge] += demand

        # Write back congestion
        for (u, v), load in edge_load.items():
            bw = _safe_float(G[u][v].get("bandwidth"), 100.0)
            bw = max(bw, _MIN_BANDWIDTH)
            new_cong = min(load / bw, _MAX_CONGESTION)
            G[u][v]["congestion"] = new_cong

        total_load = sum(edge_load.values())
        log.debug(
            "update_congestion: total edge load = %.1f Mbps across %d edges",
            total_load,
            G.number_of_edges(),
        )

    # ------------------------------------------------------------------
    # Traffic spike
    # ------------------------------------------------------------------

    def apply_traffic_spike(
        self,
        G: nx.DiGraph,
        node_set: Collection[Any],
        factor: float = 2.0,
    ) -> None:
        """Increase congestion on all edges adjacent to *node_set* by *factor*.

        Congestion is multiplicatively scaled and capped at ``1.0``.  This
        simulates a sudden burst of traffic from or to a set of nodes (e.g.
        a DDoS event or a large file transfer).

        The graph is modified **in place**.

        Parameters
        ----------
        G:
            Directed graph to modify.
        node_set:
            Collection of node identifiers whose adjacent edges are targeted.
        factor:
            Multiplicative factor applied to existing congestion values.
            Must be > 0.
        """
        if factor <= 0:
            raise ValueError(f"factor must be > 0, got {factor}")

        node_set_s = set(node_set)
        affected = 0

        for node in node_set_s:
            if node not in G:
                log.warning("apply_traffic_spike: node %s not in graph, skipping.", node)
                continue

            # Outgoing edges
            for _, v, data in G.out_edges(node, data=True):
                old = _safe_float(data.get("congestion"), 0.0)
                data["congestion"] = min(old * factor, _MAX_CONGESTION)
                affected += 1

            # Incoming edges
            for u, _, data in G.in_edges(node, data=True):
                old = _safe_float(data.get("congestion"), 0.0)
                data["congestion"] = min(old * factor, _MAX_CONGESTION)
                affected += 1

        log.info(
            "apply_traffic_spike: factor=%.2f applied to %d edge(s) adjacent to %d node(s)",
            factor,
            affected,
            len(node_set_s),
        )

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self, G: nx.DiGraph) -> nx.DiGraph:
        """Return a deep copy of *G* for before/after comparison.

        The copy preserves all node and edge attributes, graph-level
        attributes, and the directed structure.  Modifying the returned
        snapshot does not affect the original graph.

        Parameters
        ----------
        G:
            Graph to snapshot.

        Returns
        -------
        nx.DiGraph
            Independent deep copy.
        """
        snap = copy.deepcopy(G)
        log.debug(
            "snapshot: captured graph with %d nodes, %d edges",
            snap.number_of_nodes(),
            snap.number_of_edges(),
        )
        return snap

    # ------------------------------------------------------------------
    # Congestion delta utility
    # ------------------------------------------------------------------

    def congestion_delta(
        self,
        before: nx.DiGraph,
        after: nx.DiGraph,
    ) -> dict[tuple, float]:
        """Compute per-edge congestion change between two graph snapshots.

        Parameters
        ----------
        before:
            Graph state before a traffic event.
        after:
            Graph state after a traffic event.

        Returns
        -------
        dict[tuple[Any, Any], float]
            Mapping of ``(u, v)`` → ``(after_congestion - before_congestion)``.
            Only edges present in *both* graphs are included.
        """
        delta: dict[tuple, float] = {}
        for u, v in before.edges():
            if after.has_edge(u, v):
                c_before = _safe_float(before[u][v].get("congestion"), 0.0)
                c_after = _safe_float(after[u][v].get("congestion"), 0.0)
                delta[(u, v)] = c_after - c_before
        return delta
