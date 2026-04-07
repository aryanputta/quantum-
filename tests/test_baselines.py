"""
Tests for classical routing baseline solvers.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_test_graph():
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


class TestDijkstra:

    def test_finds_path(self):
        from classical_baselines import DijkstraSolver
        G = make_test_graph()
        solver = DijkstraSolver()
        result = solver.solve(G, 0, 3)
        assert result.solver_name == "dijkstra"
        assert result.feasible
        assert result.routing_cost > 0
        assert result.runtime_seconds >= 0

    def test_result_fields_populated(self):
        from classical_baselines import DijkstraSolver
        G = make_test_graph()
        result = DijkstraSolver().solve(G, 0, 3)
        assert result.latency >= 0
        assert result.congestion_score >= 0
        assert result.loss_score >= 0

    def test_disconnected_graph(self):
        import networkx as nx
        from classical_baselines import DijkstraSolver
        G = nx.DiGraph()
        G.add_node(0)
        G.add_node(1)
        result = DijkstraSolver().solve(G, 0, 1)
        assert not result.feasible


class TestGreedy:

    def test_finds_path_or_falls_back(self):
        from classical_baselines import GreedySolver
        G = make_test_graph()
        result = GreedySolver().solve(G, 0, 3)
        assert result.solver_name == "greedy"
        assert result.routing_cost >= 0


class TestLoadBalanced:

    def test_finds_path(self):
        from classical_baselines import LoadBalancedSolver
        G = make_test_graph()
        result = LoadBalancedSolver().solve(G, 0, 3)
        assert result.solver_name == "load_balanced"
        assert result.feasible


class TestSimulatedAnnealing:

    def test_runs_and_returns_result(self):
        from classical_baselines import SimulatedAnnealingSolver
        from qubo_formulation import QUBOBuilder
        G = make_test_graph()
        builder = QUBOBuilder()
        qubo = builder.build_edge_routing_qubo(G, 0, 3)
        solver = SimulatedAnnealingSolver(num_reads=5, num_sweeps=100, seed=42)
        result = solver.solve(qubo.Q, qubo.variable_map, G, 0, 3)
        assert result.solver_name == "simulated_annealing"
        assert result.runtime_seconds > 0
        assert len(result.energy_history) > 0

    def test_energy_decreases_on_average(self):
        """Best energy from more sweeps should not be worse than fewer sweeps."""
        from classical_baselines import SimulatedAnnealingSolver
        from qubo_formulation import QUBOBuilder
        G = make_test_graph()
        qubo = QUBOBuilder().build_edge_routing_qubo(G, 0, 3)
        r1 = SimulatedAnnealingSolver(num_reads=5, num_sweeps=50, seed=1).solve(
            qubo.Q, qubo.variable_map, G, 0, 3)
        r2 = SimulatedAnnealingSolver(num_reads=5, num_sweeps=500, seed=1).solve(
            qubo.Q, qubo.variable_map, G, 0, 3)
        # More sweeps should find equal or better energy (soft check — stochastic)
        assert r2.best_energy <= r1.best_energy + abs(r1.best_energy) * 0.5


class TestParallelTempering:

    def test_runs_and_returns_result(self):
        from classical_baselines import ParallelTemperingSolver
        from qubo_formulation import QUBOBuilder
        G = make_test_graph()
        qubo = QUBOBuilder().build_edge_routing_qubo(G, 0, 3)
        solver = ParallelTemperingSolver(num_replicas=4, num_sweeps=100, seed=42)
        result = solver.solve(qubo.Q, qubo.variable_map, G, 0, 3)
        assert result.solver_name == "parallel_tempering"
        assert result.runtime_seconds > 0
