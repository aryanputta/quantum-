"""
Sweep configuration for reproducible experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class SweepConfig:
    """
    Configuration for a parameter sweep experiment.
    All combinations are run with fixed seeds for reproducibility.
    """
    graph_sizes: List[int] = field(default_factory=lambda: [8, 12, 16, 20])
    traffic_intensities: List[float] = field(default_factory=lambda: [0.2, 0.5, 0.8])
    congestion_penalties: List[float] = field(default_factory=lambda: [1.0, 5.0, 10.0])
    qaoa_depths: List[int] = field(default_factory=lambda: [1, 2, 3])
    repeat_per_config: int = 10
    base_seed: int = 42
    save_dir: str = "experiments/runs"
    solvers: List[str] = field(default_factory=lambda: [
        "dijkstra", "greedy", "load_balanced",
        "simulated_annealing", "parallel_tempering"
    ])
    run_quantum: bool = True

    def iter_configs(self):
        """
        Iterate over all sweep configurations.
        Yields dict with config for one experiment point.
        """
        for n in self.graph_sizes:
            for intensity in self.traffic_intensities:
                for penalty in self.congestion_penalties:
                    for p in (self.qaoa_depths if self.run_quantum else [1]):
                        yield {
                            "n_nodes": n,
                            "traffic_intensity": intensity,
                            "congestion_penalty": penalty,
                            "qaoa_depth": p,
                            "seed": self.base_seed + n * 1000 + int(intensity * 100),
                        }

    def n_configs(self) -> int:
        n_q = len(self.qaoa_depths) if self.run_quantum else 1
        return (len(self.graph_sizes) * len(self.traffic_intensities)
                * len(self.congestion_penalties) * n_q)
