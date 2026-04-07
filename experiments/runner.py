"""
Experiment Runner.
Executes parameter sweeps and saves results for analysis.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np

from .sweep import SweepConfig

logger = logging.getLogger(__name__)


class ExperimentRunner:
    """
    Orchestrates reproducible sweeps across routing conditions.

    Saves per-run results, configs, seeds, and per-solver energies.
    Supports exact reruns via fixed seeds.
    """

    def __init__(self, pipeline, config: SweepConfig = None, save_dir: str = "experiments/runs"):
        self.pipeline = pipeline
        self.sweep_config = config or SweepConfig(save_dir=save_dir)
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def run_sweep(self, verbose: bool = True) -> list:
        """
        Execute full parameter sweep.

        Returns list of experiment records (dicts).
        """
        configs = list(self.sweep_config.iter_configs())
        total = len(configs) * self.sweep_config.repeat_per_config

        logger.info("Starting sweep: %d configurations × %d repeats = %d total runs",
                    len(configs), self.sweep_config.repeat_per_config, total)

        all_records = []
        config_idx = 0

        for cfg in configs:
            config_idx += 1
            n = cfg["n_nodes"]
            intensity = cfg["traffic_intensity"]
            penalty = cfg["congestion_penalty"]
            p = cfg["qaoa_depth"]
            seed = cfg["seed"]

            if verbose:
                print(f"\n[CONFIG {config_idx}/{len(configs)}] "
                      f"n={n}, intensity={intensity}, penalty={penalty}, p_qaoa={p}")

            # Update pipeline QUBO penalty
            self.pipeline.qubo_builder.penalty_capacity = penalty
            self.pipeline.p = p
            self.pipeline.qaoa_builder.p = p

            for trial in range(self.sweep_config.repeat_per_config):
                trial_seed = seed + trial
                record = self._run_single(cfg, trial, trial_seed, verbose)
                all_records.append(record)

        # Save summary
        self._save_records(all_records)
        logger.info("Sweep complete. %d records saved to %s", len(all_records), self.save_dir)
        return all_records

    def run_single_config(
        self,
        n_nodes: int = 12,
        traffic_intensity: float = 0.5,
        congestion_penalty: float = 5.0,
        qaoa_depth: int = 2,
        n_trials: int = 10,
        seed: int = 42,
        run_quantum: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Run a single configuration with n_trials repeats.
        Returns aggregated benchmark results.
        """
        cfg = {
            "n_nodes": n_nodes,
            "traffic_intensity": traffic_intensity,
            "congestion_penalty": congestion_penalty,
            "qaoa_depth": qaoa_depth,
            "seed": seed,
        }
        records = []
        for trial in range(n_trials):
            record = self._run_single(cfg, trial, seed + trial, verbose=verbose and trial == 0)
            records.append(record)

        return {
            "config": cfg,
            "n_trials": n_trials,
            "records": records,
        }

    # ------------------------------------------------------------------

    def _run_single(self, cfg: dict, trial: int, seed: int, verbose: bool) -> dict:
        """Run one trial for a given config."""
        n = cfg["n_nodes"]
        intensity = cfg["traffic_intensity"]
        t_start = time.time()

        try:
            # Build graph
            G = self.pipeline.loader.load_synthetic(n, edge_prob=0.35, seed=seed)
            G = self.pipeline.preprocessor.get_subgraph(G, max_nodes=n)
            pairs = self.pipeline.preprocessor.compute_routing_pairs(G, k=1)

            if not pairs:
                nodes = list(G.nodes())
                src, dst = nodes[0], nodes[-1]
            else:
                src, dst = pairs[0]

            # Traffic update
            from graph_engine import TrafficManager
            tm = TrafficManager()
            traffic = tm.generate_traffic_matrix(G, intensity=intensity, seed=seed)
            tm.update_congestion(G, traffic)

            # Build instance
            inst = self.pipeline.build_instance(G, src, dst, traffic_matrix=traffic)

            # Run classical solvers
            classical_results = self.pipeline.run_classical_solvers(
                inst, self.sweep_config.solvers
            )

            # Collect per-solver metrics
            solver_data = {}
            for sname, sres in classical_results.items():
                solver_data[sname] = {
                    "energy": float(sres.best_energy),
                    "routing_cost": float(sres.routing_cost),
                    "latency": float(sres.latency),
                    "congestion": float(sres.congestion_score),
                    "loss": float(sres.loss_score),
                    "runtime": float(sres.runtime_seconds),
                    "feasible": bool(sres.feasible),
                    "n_iterations": int(sres.n_iterations),
                }

            record = {
                "config": cfg,
                "trial": trial,
                "seed": seed,
                "n_nodes": n,
                "n_edges": G.number_of_edges(),
                "src": int(src),
                "dst": int(dst),
                "n_variables": inst.qubo_result.n_variables if inst.qubo_result else 0,
                "solvers": solver_data,
                "total_time": time.time() - t_start,
                "status": "ok",
            }

            if verbose:
                best_e = min(d["energy"] for d in solver_data.values()) if solver_data else float("nan")
                print(f"  trial={trial+1}  vars={record['n_variables']}  best_E={best_e:.4f}")

        except Exception as exc:
            logger.error("Trial failed: %s", exc, exc_info=True)
            record = {
                "config": cfg, "trial": trial, "seed": seed,
                "status": "error", "error": str(exc),
                "total_time": time.time() - t_start,
            }

        return record

    def _save_records(self, records: list):
        """Save all records to JSON."""
        path = os.path.join(self.save_dir, f"sweep_{int(time.time())}.json")
        # Convert numpy types for JSON serialization
        def _convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def _clean(d):
            if isinstance(d, dict):
                return {k: _clean(v) for k, v in d.items()}
            if isinstance(d, list):
                return [_clean(v) for v in d]
            return _convert(d)

        with open(path, "w") as f:
            json.dump(_clean(records), f, indent=2)
        logger.info("Experiment records saved: %s", path)
        print(f"  [SAVE] Experiment records saved: {path}")
