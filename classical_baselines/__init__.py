"""
classical_baselines
===================

Classical routing and annealing baselines for quantum routing optimisation.

All solvers return a standardised :class:`SolverResult` so they can be
compared directly with quantum solver outputs.

Exported names
--------------
SolverResult
    Dataclass holding the result of any solver run.

DijkstraSolver
    Shortest-path solver using Dijkstra's algorithm with composite edge weight.

GreedySolver
    Greedy hop-by-hop solver with Dijkstra fallback when stuck.

LoadBalancedSolver
    Top-k shortest path selector that minimises max congestion + total latency.

SimulatedAnnealingSolver
    QUBO minimiser using simulated annealing with a linear beta schedule.

ParallelTemperingSolver
    QUBO minimiser using parallel tempering (replica-exchange Monte Carlo).
"""

from .solver_base import SolverResult
from .dijkstra import DijkstraSolver
from .greedy import GreedySolver
from .load_balanced import LoadBalancedSolver
from .simulated_annealing import SimulatedAnnealingSolver
from .parallel_tempering import ParallelTemperingSolver

__all__ = [
    "SolverResult",
    "DijkstraSolver",
    "GreedySolver",
    "LoadBalancedSolver",
    "SimulatedAnnealingSolver",
    "ParallelTemperingSolver",
]
