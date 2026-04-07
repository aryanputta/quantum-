"""
Tests for graph engine: loader, preprocessor, traffic manager.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNetworkLoader:

    def test_load_synthetic_basic(self):
        from graph_engine import NetworkLoader
        loader = NetworkLoader()
        G = loader.load_synthetic(10, edge_prob=0.3, seed=42)
        assert G.number_of_nodes() == 10
        assert G.number_of_edges() > 0

    def test_edge_attributes_present(self):
        from graph_engine import NetworkLoader
        loader = NetworkLoader()
        G = loader.load_synthetic(8, seed=1)
        for u, v, d in G.edges(data=True):
            assert "latency" in d, f"Edge ({u},{v}) missing latency"
            assert "bandwidth" in d, f"Edge ({u},{v}) missing bandwidth"
            assert "congestion" in d, f"Edge ({u},{v}) missing congestion"
            assert "loss" in d, f"Edge ({u},{v}) missing loss"

    def test_edge_attribute_ranges(self):
        from graph_engine import NetworkLoader
        loader = NetworkLoader()
        G = loader.load_synthetic(15, seed=7)
        for u, v, d in G.edges(data=True):
            assert d["latency"] > 0, "Latency must be positive"
            assert 0.0 <= d["congestion"] <= 1.0, "Congestion must be in [0,1]"
            assert 0.0 <= d["loss"] <= 1.0, "Loss must be in [0,1]"
            assert d["bandwidth"] > 0

    def test_load_sample_topology(self):
        from graph_engine import NetworkLoader
        loader = NetworkLoader()
        G = loader.load_sample_topology("ring_mesh_10")
        assert G.number_of_nodes() > 0

    def test_reproducibility(self):
        from graph_engine import NetworkLoader
        loader = NetworkLoader()
        G1 = loader.load_synthetic(10, seed=42)
        G2 = loader.load_synthetic(10, seed=42)
        assert list(G1.edges()) == list(G2.edges())


class TestGraphPreprocessor:

    def test_normalize_weights(self):
        from graph_engine import NetworkLoader, GraphPreprocessor
        G = NetworkLoader().load_synthetic(10, seed=3)
        preprocessor = GraphPreprocessor()
        G_norm = preprocessor.normalize_weights(G)
        for u, v, d in G_norm.edges(data=True):
            for key in ["latency", "congestion"]:
                if key in d:
                    assert 0.0 <= d[key] <= 1.0 + 1e-9, f"{key} out of [0,1] after normalization"

    def test_extract_features(self):
        from graph_engine import NetworkLoader, GraphPreprocessor
        G = NetworkLoader().load_synthetic(12, seed=5)
        preprocessor = GraphPreprocessor()
        feats = preprocessor.extract_features(G)
        required = ["n_nodes", "n_edges", "avg_degree", "avg_latency",
                    "avg_congestion", "density"]
        for key in required:
            assert key in feats, f"Feature '{key}' missing"

    def test_compute_routing_pairs(self):
        from graph_engine import NetworkLoader, GraphPreprocessor
        G = NetworkLoader().load_synthetic(10, seed=9)
        pairs = GraphPreprocessor().compute_routing_pairs(G, k=3)
        assert len(pairs) <= 3
        for src, dst in pairs:
            assert src != dst
            assert G.has_node(src)
            assert G.has_node(dst)

    def test_get_subgraph(self):
        from graph_engine import NetworkLoader, GraphPreprocessor
        G = NetworkLoader().load_synthetic(30, seed=2)
        G_sub = GraphPreprocessor().get_subgraph(G, max_nodes=15)
        assert G_sub.number_of_nodes() <= 15


class TestTrafficManager:

    def test_generate_traffic_matrix(self):
        import numpy as np
        from graph_engine import NetworkLoader, TrafficManager
        G = NetworkLoader().load_synthetic(8, seed=1)
        tm = TrafficManager()
        matrix = tm.generate_traffic_matrix(G, intensity=0.5, seed=42)
        n = G.number_of_nodes()
        assert matrix.shape == (n, n)
        assert (matrix >= 0).all()

    def test_update_congestion_changes_edges(self):
        import numpy as np
        from graph_engine import NetworkLoader, TrafficManager
        G = NetworkLoader().load_synthetic(8, seed=1)
        tm = TrafficManager()
        traffic = tm.generate_traffic_matrix(G, intensity=0.8, seed=1)
        before = {(u, v): d["congestion"] for u, v, d in G.edges(data=True)}
        tm.update_congestion(G, traffic)
        after = {(u, v): d["congestion"] for u, v, d in G.edges(data=True)}
        # At least some congestion values should change
        changes = sum(1 for k in before if abs(before[k] - after[k]) > 1e-6)
        assert changes > 0, "Congestion values unchanged after traffic update"
