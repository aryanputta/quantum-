"""
GraphPreprocessor — normalise edge attributes, extract graph-level features,
derive routing pairs, and reduce large graphs to QUBO-tractable subgraphs.

All methods operate on ``networkx.DiGraph`` instances produced by
:class:`graph_engine.loader.NetworkLoader` and expect edges to carry the
standard attributes ``latency``, ``bandwidth``, ``congestion``, and ``loss``.
"""

from __future__ import annotations

import logging
import math
import random
from copy import deepcopy
from typing import Any

import networkx as nx
import numpy as np

log = logging.getLogger(__name__)

# Attribute names that are subject to normalisation
_NUMERIC_ATTRS = ("latency", "bandwidth", "congestion", "loss")


def _safe_get(data: dict, key: str, default: float = 0.0) -> float:
    """Return ``data[key]`` as float, or *default* on missing/invalid values."""
    val = data.get(key, default)
    try:
        f = float(val)
        if math.isfinite(f):
            return f
    except (TypeError, ValueError):
        pass
    return default


class GraphPreprocessor:
    """Utility class for graph preprocessing tasks in the routing pipeline.

    Methods are designed to be called in sequence:

    1. :meth:`normalize_weights` — scale edge attributes to [0, 1]
    2. :meth:`get_subgraph`       — reduce to QUBO-tractable size
    3. :meth:`compute_routing_pairs` — derive (src, dst) workload
    4. :meth:`extract_features`  — compute graph statistics for ML input
    """

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------

    def normalize_weights(self, G: nx.DiGraph) -> nx.DiGraph:
        """Return a copy of *G* with all numeric edge attributes scaled to [0, 1].

        Each attribute is scaled independently using min–max normalisation:

            x_norm = (x - x_min) / (x_max - x_min)

        If ``x_max == x_min`` (constant attribute) every value is set to
        ``0.5`` to avoid division-by-zero while preserving semantic meaning.

        The original graph is not modified.

        Parameters
        ----------
        G:
            Input directed graph.

        Returns
        -------
        nx.DiGraph
            New graph with the same topology but normalised edge attributes.
            Original raw values are preserved under keys with a ``_raw``
            suffix (e.g. ``latency_raw``).
        """
        H = deepcopy(G)

        if H.number_of_edges() == 0:
            log.warning("normalize_weights called on graph with no edges.")
            return H

        for attr in _NUMERIC_ATTRS:
            values = [
                _safe_get(data, attr)
                for _, _, data in H.edges(data=True)
            ]
            v_min = min(values)
            v_max = max(values)
            span = v_max - v_min

            for _, _, data in H.edges(data=True):
                raw = _safe_get(data, attr)
                # Preserve original value
                data[f"{attr}_raw"] = raw
                if span > 0.0:
                    data[attr] = (raw - v_min) / span
                else:
                    data[attr] = 0.5

        return H

    # ------------------------------------------------------------------
    # Routing pairs
    # ------------------------------------------------------------------

    def compute_routing_pairs(
        self,
        G: nx.DiGraph,
        k: int = 3,
        seed: int = 42,
    ) -> list[tuple[Any, Any]]:
        """Select up to *k* source–destination routing pairs from *G*.

        Selection strategy:
        * Prefer pairs whose shortest-path distance (hop count) is greater
          than 1 so they actually exercise multi-hop routing.
        * If the graph is small enough (≤ 20 nodes), enumerate all reachable
          pairs and rank them by path length (longest first), then take the
          top *k*.
        * For larger graphs a random sample of candidate pairs is evaluated.

        Parameters
        ----------
        G:
            Directed graph to derive pairs from.
        k:
            Maximum number of (src, dst) pairs to return.
        seed:
            Random seed for sampling in large graphs.

        Returns
        -------
        list[tuple]
            List of ``(src, dst)`` node-pair tuples, length ≤ *k*.
        """
        if G.number_of_nodes() < 2:
            log.warning("Graph has fewer than 2 nodes; no routing pairs possible.")
            return []

        nodes = list(G.nodes())
        rng = random.Random(seed)

        # Build candidate (src, dst) list
        if len(nodes) <= 20:
            # Exhaustive: all ordered pairs
            candidates = [
                (u, v) for u in nodes for v in nodes if u != v
            ]
        else:
            # Random sample of pairs (limit to keep this O(k·log n))
            sample_size = min(len(nodes) * (len(nodes) - 1), k * 50)
            candidates = []
            for _ in range(sample_size):
                u, v = rng.sample(nodes, 2)
                candidates.append((u, v))

        # Filter to reachable pairs and record path lengths
        scored: list[tuple[int, Any, Any]] = []
        for u, v in candidates:
            try:
                length = nx.shortest_path_length(G, u, v)
                if length > 0:
                    scored.append((length, u, v))
            except nx.NetworkXNoPath:
                pass
            except nx.NodeNotFound:
                pass

        if not scored:
            log.warning("No reachable pairs found; falling back to first two nodes.")
            return [(nodes[0], nodes[1])] if len(nodes) >= 2 else []

        # Sort by path length descending to prefer interesting (longer) routes
        scored.sort(key=lambda t: t[0], reverse=True)

        # Deduplicate while preserving order
        seen: set[tuple] = set()
        pairs: list[tuple[Any, Any]] = []
        for _, u, v in scored:
            pair = (u, v)
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                if len(pairs) == k:
                    break

        log.debug("compute_routing_pairs: selected %d pairs (k=%d)", len(pairs), k)
        return pairs

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(self, G: nx.DiGraph) -> dict[str, float]:
        """Compute a flat feature dictionary for *G* suitable as ML input.

        Features computed
        -----------------
        n_nodes : int
            Total number of nodes.
        n_edges : int
            Total number of directed edges.
        avg_degree : float
            Average out-degree.
        avg_latency : float
            Mean latency across all edges (uses ``latency`` attr).
        avg_congestion : float
            Mean congestion across all edges.
        max_congestion : float
            Maximum congestion value over all edges.
        avg_loss : float
            Mean packet-loss probability across all edges.
        density : float
            Graph density (directed).
        clustering_coeff : float
            Average clustering coefficient (undirected projection).
        diameter_approx : float
            Approximation of the graph diameter (BFS from a sampled set of
            source nodes, returning the maximum eccentricity observed).
            Returns ``-1.0`` for disconnected or trivial graphs.

        Parameters
        ----------
        G:
            Directed graph to analyse.

        Returns
        -------
        dict[str, float]
        """
        n = G.number_of_nodes()
        m = G.number_of_edges()

        features: dict[str, float] = {
            "n_nodes": float(n),
            "n_edges": float(m),
            "avg_degree": float(m / n) if n > 0 else 0.0,
            "density": float(nx.density(G)),
        }

        # Edge-attribute statistics
        latencies = [_safe_get(d, "latency") for _, _, d in G.edges(data=True)]
        congestions = [_safe_get(d, "congestion") for _, _, d in G.edges(data=True)]
        losses = [_safe_get(d, "loss") for _, _, d in G.edges(data=True)]

        if latencies:
            features["avg_latency"] = float(np.mean(latencies))
        else:
            features["avg_latency"] = 0.0

        if congestions:
            features["avg_congestion"] = float(np.mean(congestions))
            features["max_congestion"] = float(np.max(congestions))
        else:
            features["avg_congestion"] = 0.0
            features["max_congestion"] = 0.0

        if losses:
            features["avg_loss"] = float(np.mean(losses))
        else:
            features["avg_loss"] = 0.0

        # Clustering coefficient (undirected projection)
        try:
            undirected = G.to_undirected()
            features["clustering_coeff"] = float(nx.average_clustering(undirected))
        except Exception:
            features["clustering_coeff"] = 0.0

        # Diameter approximation via BFS eccentricity on a node sample
        features["diameter_approx"] = self._approximate_diameter(G)

        return features

    def _approximate_diameter(self, G: nx.DiGraph, n_samples: int = 10) -> float:
        """Estimate the graph diameter via BFS from a random node sample.

        Returns ``-1.0`` if the graph is not (weakly) connected or has < 2
        nodes.
        """
        if G.number_of_nodes() < 2:
            return -1.0
        if not nx.is_weakly_connected(G):
            return -1.0

        nodes = list(G.nodes())
        sample = nodes[:n_samples] if len(nodes) <= n_samples else (
            random.sample(nodes, n_samples)
        )

        max_ecc = 0
        for src in sample:
            lengths = nx.single_source_shortest_path_length(G, src)
            if lengths:
                ecc = max(lengths.values())
                if ecc > max_ecc:
                    max_ecc = ecc

        return float(max_ecc)

    # ------------------------------------------------------------------
    # Subgraph extraction for QUBO tractability
    # ------------------------------------------------------------------

    def get_subgraph(self, G: nx.DiGraph, max_nodes: int = 20) -> nx.DiGraph:
        """Extract a tractable subgraph of at most *max_nodes* nodes.

        Strategy:
        1. If ``|V| ≤ max_nodes``, return a copy of *G* unchanged.
        2. Otherwise, find the highest-degree node and perform a BFS
           expansion until *max_nodes* nodes have been collected, preserving
           all edges between selected nodes.  This keeps a dense,
           well-connected region of the graph which is most interesting for
           routing.

        Parameters
        ----------
        G:
            Source directed graph.
        max_nodes:
            Maximum number of nodes in the returned subgraph.

        Returns
        -------
        nx.DiGraph
            Induced subgraph with the same edge attributes.
        """
        if max_nodes < 2:
            raise ValueError(f"max_nodes must be >= 2, got {max_nodes}")

        n = G.number_of_nodes()
        if n <= max_nodes:
            return G.copy()

        # Start BFS from the node with the highest total degree
        undirected = G.to_undirected()
        seed_node = max(undirected.degree(), key=lambda x: x[1])[0]

        selected: list = []
        queue = [seed_node]
        visited: set = {seed_node}

        while queue and len(selected) < max_nodes:
            current = queue.pop(0)
            selected.append(current)
            # Explore both successors and predecessors for denser coverage
            neighbours = list(G.successors(current)) + list(G.predecessors(current))
            for nb in neighbours:
                if nb not in visited and len(selected) + len(queue) < max_nodes:
                    visited.add(nb)
                    queue.append(nb)

        # Fill up to max_nodes if BFS ended early (sparse graph)
        if len(selected) < max_nodes:
            remaining = [n for n in G.nodes() if n not in visited]
            for node in remaining:
                if len(selected) >= max_nodes:
                    break
                selected.append(node)

        subgraph = G.subgraph(selected).copy()
        log.info(
            "get_subgraph: reduced from %d → %d nodes, %d → %d edges",
            n,
            subgraph.number_of_nodes(),
            G.number_of_edges(),
            subgraph.number_of_edges(),
        )
        return subgraph
