"""
NetworkLoader — build or load directed, weighted NetworkX graphs from multiple
data sources used in the quantum routing optimisation pipeline.

All returned graphs are ``networkx.DiGraph`` instances whose edges carry four
attributes:

    latency     : float  — propagation latency in milliseconds
    bandwidth   : float  — link capacity in Mbps
    congestion  : float  — current utilisation ratio  ∈ [0, 1]
    loss        : float  — packet-loss probability     ∈ [0, 1]

Missing attributes on load are filled with sensible defaults so that
downstream QUBO formulation code never encounters a KeyError.
"""

from __future__ import annotations

import csv
import logging
import random
import re
from pathlib import Path
from typing import Optional

import networkx as nx

from data.sample_topologies import TOPOLOGIES, get_topology

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default edge-attribute values used when a source file omits them
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, float] = {
    "latency": 10.0,
    "bandwidth": 100.0,
    "congestion": 0.1,
    "loss": 0.001,
}


def _fill_missing_attrs(G: nx.DiGraph) -> None:
    """Fill any missing standard edge attributes with default values in-place."""
    for u, v, data in G.edges(data=True):
        for attr, default in _DEFAULTS.items():
            if attr not in data:
                data[attr] = default


def _ensure_connected(G: nx.DiGraph) -> nx.DiGraph:
    """Return the subgraph induced by the largest weakly-connected component.

    If the graph is already (weakly) connected, the original graph is
    returned unchanged.  This prevents QUBO formulation from operating on
    disconnected topologies.
    """
    if nx.is_weakly_connected(G):
        return G

    components = sorted(
        nx.weakly_connected_components(G), key=len, reverse=True
    )
    largest = components[0]
    log.warning(
        "Graph has %d weakly connected components; using largest (%d nodes).",
        len(components),
        len(largest),
    )
    return G.subgraph(largest).copy()


# ---------------------------------------------------------------------------
# NetworkLoader
# ---------------------------------------------------------------------------

class NetworkLoader:
    """Factory class for building NetworkX DiGraphs from various sources.

    All ``load_*`` methods apply consistent post-processing:
    * missing edge attributes are filled with defaults
    * the graph is reduced to its largest weakly-connected component when
      disconnected

    Parameters
    ----------
    ensure_connected:
        When *True* (default) each loaded graph is reduced to its largest
        weakly-connected component.
    default_latency_range:
        ``(min_ms, max_ms)`` used when randomising latency.
    default_bandwidth_range:
        ``(min_mbps, max_mbps)`` used when randomising bandwidth.
    default_congestion_range:
        ``(min, max)`` utilisation ratio.
    default_loss_range:
        ``(min, max)`` loss probability.
    """

    def __init__(
        self,
        ensure_connected: bool = True,
        default_latency_range: tuple[float, float] = (1.0, 50.0),
        default_bandwidth_range: tuple[float, float] = (10.0, 1000.0),
        default_congestion_range: tuple[float, float] = (0.0, 0.9),
        default_loss_range: tuple[float, float] = (0.0, 0.05),
    ) -> None:
        self.ensure_connected = ensure_connected
        self.latency_range = default_latency_range
        self.bandwidth_range = default_bandwidth_range
        self.congestion_range = default_congestion_range
        self.loss_range = default_loss_range

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rand_attrs(self, rng: random.Random) -> dict[str, float]:
        """Draw a random set of edge attributes using the configured ranges."""
        return {
            "latency": rng.uniform(*self.latency_range),
            "bandwidth": rng.uniform(*self.bandwidth_range),
            "congestion": rng.uniform(*self.congestion_range),
            "loss": rng.uniform(*self.loss_range),
        }

    def _post_process(self, G: nx.DiGraph) -> nx.DiGraph:
        """Fill missing attrs and optionally reduce to largest WCC."""
        _fill_missing_attrs(G)
        if self.ensure_connected and G.number_of_nodes() > 1:
            G = _ensure_connected(G)
        return G

    # ------------------------------------------------------------------
    # Public loader methods
    # ------------------------------------------------------------------

    def load_synthetic(
        self,
        n_nodes: int = 20,
        edge_prob: float = 0.3,
        seed: Optional[int] = 42,
    ) -> nx.DiGraph:
        """Generate a random Erdős–Rényi directed graph with realistic edge attrs.

        Parameters
        ----------
        n_nodes:
            Number of nodes in the graph.
        edge_prob:
            Probability that a directed edge (u→v) exists for each pair.
        seed:
            Random seed for reproducibility.  ``None`` uses system entropy.

        Returns
        -------
        nx.DiGraph
            Directed graph with ``latency``, ``bandwidth``, ``congestion``,
            and ``loss`` edge attributes.
        """
        if n_nodes < 2:
            raise ValueError(f"n_nodes must be >= 2, got {n_nodes}")
        if not (0.0 < edge_prob <= 1.0):
            raise ValueError(f"edge_prob must be in (0, 1], got {edge_prob}")

        rng = random.Random(seed)
        G: nx.DiGraph = nx.gnp_random_graph(
            n_nodes, edge_prob, seed=seed, directed=True
        )

        # Assign random realistic attributes to each edge
        for u, v in list(G.edges()):
            attrs = self._rand_attrs(rng)
            G[u][v].update(attrs)

        # Ensure at least a spanning tree exists (handle low edge_prob)
        nodes = list(G.nodes())
        rng.shuffle(nodes)
        for i in range(len(nodes) - 1):
            u, v = nodes[i], nodes[i + 1]
            if not G.has_edge(u, v):
                G.add_edge(u, v, **self._rand_attrs(rng))
            if not G.has_edge(v, u):
                G.add_edge(v, u, **self._rand_attrs(rng))

        log.debug(
            "Synthetic graph: %d nodes, %d edges (edge_prob=%.2f, seed=%s)",
            G.number_of_nodes(),
            G.number_of_edges(),
            edge_prob,
            seed,
        )
        return self._post_process(G)

    def load_from_edgelist(self, path: str | Path) -> nx.DiGraph:
        """Load a directed graph from a plain whitespace-separated edge list.

        Each non-comment line must start with two node identifiers.  Optional
        additional columns are ignored.  Comment lines start with ``#``.

        Parameters
        ----------
        path:
            Path to the edge-list file (plain text or ``.gz``).

        Returns
        -------
        nx.DiGraph
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Edge list not found: {path}")

        import gzip as _gzip

        G = nx.DiGraph()
        opener = _gzip.open if path.suffix == ".gz" else open

        edges_added = 0
        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except ValueError:
                    u, v = parts[0], parts[1]  # type: ignore[assignment]

                if not G.has_edge(u, v):
                    G.add_edge(u, v)
                    edges_added += 1

        log.info(
            "load_from_edgelist: %d nodes, %d edges from %s",
            G.number_of_nodes(),
            edges_added,
            path,
        )
        return self._post_process(G)

    def load_rocketfuel(self, path: str | Path) -> nx.DiGraph:
        """Parse a Rocketfuel ``.cch`` (compressed connectivity + hops) file.

        Rocketfuel CCH format lines look like::

            <router-name> <neighbor-name> <weight>

        Lines starting with ``#`` are comments.  Both gzip-compressed and
        plain-text files are supported.

        Parameters
        ----------
        path:
            Path to the ``.cch`` or ``.cch.gz`` file.

        Returns
        -------
        nx.DiGraph
            Node labels are the original router-name strings.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Rocketfuel file not found: {path}")

        import gzip as _gzip

        G = nx.DiGraph()
        # Node-name → integer ID mapping for compactness
        node_map: dict[str, int] = {}

        def _node_id(name: str) -> int:
            if name not in node_map:
                node_map[name] = len(node_map)
            return node_map[name]

        suffix = "".join(path.suffixes)
        opener = _gzip.open if ".gz" in suffix else open

        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue

                src_name = parts[0]
                dst_name = parts[1]
                src = _node_id(src_name)
                dst = _node_id(dst_name)

                # Weight field (hop count / IGP metric) → proxy for latency
                weight = 1.0
                if len(parts) >= 3:
                    try:
                        weight = float(parts[2])
                    except ValueError:
                        pass

                latency = max(0.1, weight * 0.5)  # heuristic: 0.5 ms per hop unit
                G.add_edge(
                    src,
                    dst,
                    latency=latency,
                    bandwidth=_DEFAULTS["bandwidth"],
                    congestion=_DEFAULTS["congestion"],
                    loss=_DEFAULTS["loss"],
                    router_src=src_name,
                    router_dst=dst_name,
                )

        log.info(
            "load_rocketfuel: %d nodes, %d edges from %s",
            G.number_of_nodes(),
            G.number_of_edges(),
            path,
        )
        return self._post_process(G)

    def load_snap(self, path: str | Path) -> nx.DiGraph:
        """Parse a SNAP-format edge list (e.g. email-EuAll.txt.gz).

        SNAP files have a header block of ``#``-prefixed comment lines
        followed by whitespace-separated ``FromNode ToNode`` pairs.

        Parameters
        ----------
        path:
            Path to the SNAP file (plain text or gzip-compressed).

        Returns
        -------
        nx.DiGraph
            Integer-labelled directed graph.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"SNAP file not found: {path}")

        import gzip as _gzip

        G = nx.DiGraph()
        opener = _gzip.open if path.suffix == ".gz" else open
        edges_seen: int = 0

        with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    u = int(parts[0])
                    v = int(parts[1])
                except ValueError:
                    continue

                if not G.has_edge(u, v):
                    G.add_edge(
                        u,
                        v,
                        latency=_DEFAULTS["latency"],
                        bandwidth=_DEFAULTS["bandwidth"],
                        congestion=_DEFAULTS["congestion"],
                        loss=_DEFAULTS["loss"],
                    )
                    edges_seen += 1

        log.info(
            "load_snap: %d nodes, %d edges from %s",
            G.number_of_nodes(),
            edges_seen,
            path,
        )
        return self._post_process(G)

    def load_sample_topology(self, name: str) -> nx.DiGraph:
        """Build a DiGraph from one of the pre-embedded sample topologies.

        Parameters
        ----------
        name:
            One of ``'ring_mesh_10'``, ``'isp_15'``, or ``'random_20'``.
            See :mod:`data.sample_topologies` for details.

        Returns
        -------
        nx.DiGraph
            Directed graph with full ``latency``, ``bandwidth``,
            ``congestion``, and ``loss`` edge attributes.

        Raises
        ------
        KeyError
            If *name* is not a known topology.
        """
        topo = get_topology(name)  # raises KeyError for unknown names

        G = nx.DiGraph(name=topo["name"], description=topo.get("description", ""))
        G.add_nodes_from(topo["nodes"])

        for edge_tuple in topo["edges"]:
            if len(edge_tuple) == 6:
                u, v, latency, bandwidth, congestion, loss = edge_tuple
            elif len(edge_tuple) == 2:
                u, v = edge_tuple
                latency, bandwidth, congestion, loss = (
                    _DEFAULTS["latency"],
                    _DEFAULTS["bandwidth"],
                    _DEFAULTS["congestion"],
                    _DEFAULTS["loss"],
                )
            else:
                log.warning("Unexpected edge tuple length %d, skipping.", len(edge_tuple))
                continue

            G.add_edge(
                u,
                v,
                latency=float(latency),
                bandwidth=float(bandwidth),
                congestion=float(congestion),
                loss=float(loss),
            )

        log.debug(
            "load_sample_topology('%s'): %d nodes, %d edges",
            name,
            G.number_of_nodes(),
            G.number_of_edges(),
        )
        return self._post_process(G)
