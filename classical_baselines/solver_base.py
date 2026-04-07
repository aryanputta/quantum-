from dataclasses import dataclass, field
from typing import Optional
import time
import numpy as np


@dataclass
class SolverResult:
    solver_name: str
    best_energy: float          # QUBO/routing objective value
    best_bitstring: np.ndarray  # binary assignment vector
    routing_cost: float         # latency + congestion + loss weighted sum
    latency: float
    congestion_score: float
    loss_score: float
    runtime_seconds: float
    n_iterations: int
    energy_history: list        # per-iteration best energy
    feasible: bool              # does solution satisfy constraints
    metadata: dict = field(default_factory=dict)
