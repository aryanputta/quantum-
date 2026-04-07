"""
Graph and QUBO feature extraction for ML models.
Produces fixed-size feature vectors from variable-size graphs.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class GraphFeatureExtractor:
    """
    Extract fixed-size feature vectors from NetworkX graphs and QUBO instances.

    Feature vector layout (22 features):
      0:  n_nodes (normalized)
      1:  n_edges (normalized)
      2:  density
      3:  avg_degree (normalized)
      4:  max_degree (normalized)
      5:  avg_latency
      6:  std_latency
      7:  max_latency
      8:  avg_bandwidth (normalized)
      9:  avg_congestion
      10: max_congestion
      11: std_congestion
      12: avg_loss
      13: max_loss
      14: clustering_coeff
      15: n_qubits / n_nodes ratio (QUBO size indicator)
      16: qubo_density (frac of nonzero Q entries)
      17: qubo_energy_range (max - min diagonal)
      18: qubo_max_coupling
      19: traffic_intensity (mean demand)
      20: src_out_degree (normalized)
      21: dst_in_degree (normalized)
    """

    N_FEATURES = 22
    MAX_NODES = 200     # normalization constant

    def extract(
        self,
        G,
        Q: Optional[np.ndarray] = None,
        src: Optional[int] = None,
        dst: Optional[int] = None,
        traffic_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Extract feature vector.

        Parameters
        ----------
        G              : NetworkX directed graph with edge attrs
        Q              : QUBO matrix (optional)
        src, dst       : source/destination nodes (optional)
        traffic_matrix : demand matrix (optional)

        Returns
        -------
        np.ndarray of shape (N_FEATURES,)
        """
        import networkx as nx

        feats = np.zeros(self.N_FEATURES, dtype=np.float32)
        n = G.number_of_nodes()
        m = G.number_of_edges()

        feats[0] = n / self.MAX_NODES
        feats[1] = m / (self.MAX_NODES ** 2)
        feats[2] = nx.density(G)

        if n > 0:
            degrees = [d for _, d in G.degree()]
            feats[3] = np.mean(degrees) / self.MAX_NODES
            feats[4] = max(degrees) / self.MAX_NODES

        # Edge weight statistics
        latencies = [d.get("latency", 0.0) for _, _, d in G.edges(data=True)]
        bandwidths = [d.get("bandwidth", 100.0) for _, _, d in G.edges(data=True)]
        congestions = [d.get("congestion", 0.0) for _, _, d in G.edges(data=True)]
        losses = [d.get("loss", 0.0) for _, _, d in G.edges(data=True)]

        if latencies:
            feats[5] = float(np.mean(latencies)) / 100.0
            feats[6] = float(np.std(latencies)) / 100.0
            feats[7] = float(np.max(latencies)) / 100.0
        if bandwidths:
            feats[8] = float(np.mean(bandwidths)) / 1000.0
        if congestions:
            feats[9] = float(np.mean(congestions))
            feats[10] = float(np.max(congestions))
            feats[11] = float(np.std(congestions))
        if losses:
            feats[12] = float(np.mean(losses))
            feats[13] = float(np.max(losses))

        # Graph structure
        try:
            feats[14] = float(nx.average_clustering(G.to_undirected()))
        except Exception:
            feats[14] = 0.0

        # QUBO features
        if Q is not None:
            n_vars = Q.shape[0]
            feats[15] = n_vars / max(n, 1)
            nz = np.count_nonzero(Q)
            feats[16] = nz / max(n_vars ** 2, 1)
            diag = np.diag(Q)
            if len(diag) > 0:
                feats[17] = float(np.max(diag) - np.min(diag)) / (np.abs(np.max(diag)) + 1e-8)
            off_diag = Q - np.diag(diag)
            if off_diag.size > 0:
                feats[18] = float(np.max(np.abs(off_diag))) / (np.abs(np.max(diag)) + 1e-8)

        # Traffic features
        if traffic_matrix is not None:
            feats[19] = float(np.mean(traffic_matrix))

        # Source/destination features
        if src is not None and G.has_node(src):
            feats[20] = G.out_degree(src) / self.MAX_NODES
        if dst is not None and G.has_node(dst):
            feats[21] = G.in_degree(dst) / self.MAX_NODES

        return feats

    def extract_batch(
        self,
        instances: list,
    ) -> np.ndarray:
        """
        Extract features for a list of (G, Q, src, dst, traffic_matrix) tuples.

        Returns
        -------
        np.ndarray of shape (n_instances, N_FEATURES)
        """
        rows = []
        for item in instances:
            if isinstance(item, dict):
                G = item.get("G")
                Q = item.get("Q")
                src = item.get("src")
                dst = item.get("dst")
                tm = item.get("traffic_matrix")
            else:
                G, Q, src, dst, tm = item
            rows.append(self.extract(G, Q, src, dst, tm))
        return np.array(rows, dtype=np.float32)
