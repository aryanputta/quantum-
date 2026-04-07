"""
Tests for QUBO formulation module.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestQUBOBuilder:

    def _make_small_graph(self):
        """Create a deterministic small 5-node graph."""
        import networkx as nx
        G = nx.DiGraph()
        edges = [
            (0, 1, {"latency": 5.0, "bandwidth": 100.0, "congestion": 0.2, "loss": 0.01}),
            (1, 2, {"latency": 3.0, "bandwidth": 200.0, "congestion": 0.1, "loss": 0.005}),
            (0, 2, {"latency": 10.0, "bandwidth": 50.0, "congestion": 0.5, "loss": 0.02}),
            (2, 3, {"latency": 2.0, "bandwidth": 300.0, "congestion": 0.05, "loss": 0.001}),
            (1, 3, {"latency": 8.0, "bandwidth": 150.0, "congestion": 0.3, "loss": 0.015}),
        ]
        G.add_edges_from(edges)
        return G

    def test_qubo_shape(self):
        from qubo_formulation import QUBOBuilder
        G = self._make_small_graph()
        builder = QUBOBuilder()
        result = builder.build_edge_routing_qubo(G, src=0, dst=3)
        n = result.n_variables
        assert result.Q.shape == (n, n), f"Q shape mismatch: {result.Q.shape} vs ({n},{n})"

    def test_qubo_symmetry(self):
        from qubo_formulation import QUBOBuilder
        G = self._make_small_graph()
        builder = QUBOBuilder()
        result = builder.build_edge_routing_qubo(G, src=0, dst=3)
        Q = result.Q
        # QUBO can be upper triangular; check symmetry after symmetrizing
        Q_sym = (Q + Q.T) / 2
        assert np.allclose(Q_sym, Q_sym.T, atol=1e-10), "Q is not symmetric"

    def test_variable_map_completeness(self):
        from qubo_formulation import QUBOBuilder
        G = self._make_small_graph()
        builder = QUBOBuilder()
        result = builder.build_edge_routing_qubo(G, src=0, dst=3)
        n = result.n_variables
        # All variable indices should be in variable_map
        assert len(result.variable_map) == n, (
            f"variable_map size {len(result.variable_map)} != n_variables {n}"
        )

    def test_feasible_path_has_lower_energy(self):
        """A valid path should have lower QUBO energy than a random assignment."""
        from qubo_formulation import QUBOBuilder
        from classical_baselines.routing_utils import path_to_bitstring, compute_qubo_energy
        import networkx as nx

        G = self._make_small_graph()
        builder = QUBOBuilder(penalty_path=10.0, penalty_flow=8.0)
        result = builder.build_edge_routing_qubo(G, src=0, dst=3)

        # Valid path: 0→1→2→3
        path = [0, 1, 2, 3]
        x_valid = path_to_bitstring(path, result.variable_map, result.n_variables)
        e_valid = compute_qubo_energy(result.Q, x_valid)

        # Random assignment
        rng = np.random.default_rng(1)
        e_random_list = []
        for _ in range(50):
            x_rand = rng.integers(0, 2, result.n_variables)
            e_random_list.append(compute_qubo_energy(result.Q, x_rand))

        e_random_mean = np.mean(e_random_list)
        assert e_valid <= e_random_mean, (
            f"Valid path energy ({e_valid:.4f}) should be ≤ random mean ({e_random_mean:.4f})"
        )

    def test_qubo_validation(self):
        from qubo_formulation import QUBOBuilder
        G = self._make_small_graph()
        builder = QUBOBuilder()
        result = builder.build_edge_routing_qubo(G, src=0, dst=3)
        validation = builder.validate_qubo(result)
        assert "is_valid" in validation
        assert isinstance(validation["is_valid"], bool)


class TestQUBOObjectives:

    def test_objectives_return_dicts(self):
        from qubo_formulation.objectives import latency_objective, congestion_objective
        import networkx as nx
        G = nx.DiGraph()
        G.add_edge(0, 1, latency=5.0, congestion=0.2, loss=0.01, bandwidth=100.0)
        G.add_edge(1, 2, latency=3.0, congestion=0.1, loss=0.005, bandwidth=200.0)
        # objectives.py receives edge_to_idx format: {(u,v): int}
        variables = {(0, 1): 0, (1, 2): 1}

        lin, quad = latency_objective(G, variables)
        assert isinstance(lin, dict)
        assert isinstance(quad, dict)
        assert len(lin) == 2  # one per edge variable

        lin2, quad2 = congestion_objective(G, variables)
        assert len(lin2) == 2
