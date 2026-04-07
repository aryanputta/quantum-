"""
Benchmark metrics computation.
Computes routing quality, approximation ratio, and solver comparison tables.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Single trial result from one solver run."""
    solver: str
    trial: int
    best_energy: float
    routing_cost: float
    latency: float
    congestion_score: float
    loss_score: float
    runtime_seconds: float
    feasible: bool
    n_iterations: int
    n_qubits: Optional[int] = None
    circuit_depth: Optional[int] = None
    error_rate: Optional[float] = None
    extra: dict = field(default_factory=dict)


@dataclass
class SolverStats:
    """Aggregated statistics for one solver across all trials."""
    solver: str
    n_trials: int
    mean_energy: float
    std_energy: float
    best_energy: float
    mean_routing_cost: float
    mean_latency: float
    mean_congestion: float
    mean_loss: float
    mean_runtime: float
    std_runtime: float
    feasibility_rate: float
    approximation_ratio: float      # vs best known
    tts99: Optional[float]          # seconds, or None if not computable


class BenchmarkMetrics:
    """
    Compute and aggregate benchmark metrics from solver trial results.
    """

    def __init__(self, target_quality_fraction: float = 0.95):
        """
        Parameters
        ----------
        target_quality_fraction : fraction of best known energy defining 'success'
                                  e.g. 0.95 means within 5% of best known
        """
        self.target_quality_fraction = target_quality_fraction

    def add_result(self, results: list[RunResult], result: RunResult):
        results.append(result)

    def aggregate(self, results: list[RunResult], best_known: Optional[float] = None) -> list[SolverStats]:
        """
        Aggregate results per solver.

        Parameters
        ----------
        results    : list of RunResult from all trials and solvers
        best_known : best energy across all solvers (used for approximation ratio)

        Returns
        -------
        List of SolverStats, one per solver
        """
        df = self._to_dataframe(results)

        if best_known is None:
            best_known = df["best_energy"].min()

        solver_stats = []
        for solver_name, group in df.groupby("solver"):
            energies = group["best_energy"].values
            runtimes = group["runtime_seconds"].values
            costs = group["routing_cost"].values
            feasible = group["feasible"].values

            best_e = float(energies.min())
            mean_e = float(energies.mean())
            std_e = float(energies.std()) if len(energies) > 1 else 0.0

            # Approximation ratio: best_known / best_found (≤1 means worse than known)
            if best_e != 0 and best_known is not None:
                approx_ratio = float(best_known / best_e)
            else:
                approx_ratio = 1.0

            # TTS99
            tts = self._compute_tts99(
                energies, runtimes, best_known, self.target_quality_fraction
            )

            solver_stats.append(SolverStats(
                solver=str(solver_name),
                n_trials=len(group),
                mean_energy=mean_e,
                std_energy=std_e,
                best_energy=best_e,
                mean_routing_cost=float(costs.mean()),
                mean_latency=float(group["latency"].mean()),
                mean_congestion=float(group["congestion_score"].mean()),
                mean_loss=float(group["loss_score"].mean()),
                mean_runtime=float(runtimes.mean()),
                std_runtime=float(runtimes.std()) if len(runtimes) > 1 else 0.0,
                feasibility_rate=float(feasible.mean()),
                approximation_ratio=approx_ratio,
                tts99=tts,
            ))

        solver_stats.sort(key=lambda s: s.mean_energy)
        return solver_stats

    def to_dataframe(self, stats: list[SolverStats]) -> pd.DataFrame:
        """Convert SolverStats list to pandas DataFrame."""
        rows = []
        for s in stats:
            rows.append({
                "Solver": s.solver,
                "Trials": s.n_trials,
                "Mean Energy": round(s.mean_energy, 4),
                "Std Energy": round(s.std_energy, 4),
                "Best Energy": round(s.best_energy, 4),
                "Mean Routing Cost": round(s.mean_routing_cost, 4),
                "Mean Latency": round(s.mean_latency, 4),
                "Mean Congestion": round(s.mean_congestion, 4),
                "Mean Loss": round(s.mean_loss, 6),
                "Mean Runtime (s)": round(s.mean_runtime, 4),
                "Feasibility Rate": round(s.feasibility_rate, 3),
                "Approx Ratio": round(s.approximation_ratio, 4),
                "TTS99 (s)": round(s.tts99, 4) if s.tts99 is not None else "N/A",
            })
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------

    def _to_dataframe(self, results: list[RunResult]) -> pd.DataFrame:
        return pd.DataFrame([{
            "solver": r.solver,
            "trial": r.trial,
            "best_energy": r.best_energy,
            "routing_cost": r.routing_cost,
            "latency": r.latency,
            "congestion_score": r.congestion_score,
            "loss_score": r.loss_score,
            "runtime_seconds": r.runtime_seconds,
            "feasible": int(r.feasible),
            "n_iterations": r.n_iterations,
        } for r in results])

    def _compute_tts99(
        self,
        energies: np.ndarray,
        runtimes: np.ndarray,
        best_known: float,
        target_fraction: float,
    ) -> Optional[float]:
        """
        Estimate TTS99: time to achieve 99% success probability.

        TTS99 = t_run * log(1 - 0.99) / log(1 - p_success)

        where p_success = fraction of runs achieving energy ≤ threshold.
        threshold = best_known + |best_known| * (1 - target_fraction)
        """
        if len(energies) < 2:
            return None

        # Same convention as TTS99Calculator:
        # negative best_known → threshold = best_known * target_fraction (less negative)
        # positive best_known → threshold = best_known * (2 - target_fraction) (slightly above)
        if best_known < 0:
            threshold = best_known * target_fraction
        elif best_known > 0:
            threshold = best_known * (2.0 - target_fraction)
        else:
            threshold = 0.0
        # For minimization: success if energy <= threshold
        n_success = int(np.sum(energies <= threshold))
        p_success = n_success / len(energies)

        if p_success <= 0.0:
            return None
        if p_success >= 1.0:
            return float(runtimes.mean())

        t_run = float(runtimes.mean())
        tts99 = t_run * np.log(1 - 0.99) / np.log(1 - p_success)
        return float(tts99)


class SolverComparison:
    """
    Compare two solver configurations (e.g., with vs without ML).
    """

    def __init__(self, metrics: BenchmarkMetrics):
        self.metrics = metrics

    def compare(self, baseline_results: list[RunResult], enhanced_results: list[RunResult],
                 baseline_name: str = "baseline", enhanced_name: str = "enhanced") -> dict:
        """
        Compare two solver configurations across shared metrics.

        Returns dict with improvement statistics.
        """
        baseline_stats = self.metrics.aggregate(baseline_results)
        enhanced_stats = self.metrics.aggregate(enhanced_results)

        def get_stat(stats, name):
            for s in stats:
                if s.solver == name:
                    return s
            return None

        b = get_stat(baseline_stats, baseline_name) or (baseline_stats[0] if baseline_stats else None)
        e = get_stat(enhanced_stats, enhanced_name) or (enhanced_stats[0] if enhanced_stats else None)

        if b is None or e is None:
            return {}

        energy_improvement = (b.mean_energy - e.mean_energy) / (abs(b.mean_energy) + 1e-10) * 100
        runtime_improvement = (b.mean_runtime - e.mean_runtime) / (abs(b.mean_runtime) + 1e-10) * 100
        tts_improvement = None
        if b.tts99 and e.tts99:
            tts_improvement = (b.tts99 - e.tts99) / (abs(b.tts99) + 1e-10) * 100

        return {
            "energy_improvement_pct": round(energy_improvement, 2),
            "runtime_improvement_pct": round(runtime_improvement, 2),
            "tts99_improvement_pct": round(tts_improvement, 2) if tts_improvement else None,
            "feasibility_improvement": round(e.feasibility_rate - b.feasibility_rate, 3),
            "baseline": b,
            "enhanced": e,
        }
