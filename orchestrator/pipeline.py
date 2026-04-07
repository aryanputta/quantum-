"""
HQMRE Hybrid Orchestrator Pipeline.
Coordinates graph ingestion → QUBO → Ising → solver dispatch → benchmarking.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RoutingInstance:
    """A single routing problem instance."""
    G: object                       # NetworkX graph
    src: int
    dst: int
    qubo_result: Optional[object] = None   # QUBOResult
    ising_result: Optional[object] = None  # IsingResult
    traffic_matrix: Optional[np.ndarray] = None
    instance_id: str = ""
    metadata: dict = field(default_factory=dict)


class HQMREPipeline:
    """
    End-to-end hybrid orchestrator.

    Manages the full pipeline:
    1. Graph loading
    2. QUBO construction (with optional ML-adjusted weights)
    3. Ising conversion
    4. Solver dispatch (classical + quantum variants)
    5. Result collection and benchmarking
    """

    SOLVER_KEYS = [
        "dijkstra",
        "greedy",
        "load_balanced",
        "simulated_annealing",
        "parallel_tempering",
        "qaoa_baseline",
        "qaoa_hardware_aware",
        "qaoa_ml_hardware",
    ]

    def __init__(self, config: dict = None):
        self.config = config or {}
        self._setup_components()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_components(self):
        """Lazy-import and initialize all subsystem components."""
        from graph_engine import NetworkLoader, GraphPreprocessor
        from qubo_formulation import QUBOBuilder
        from ising_utils import IsingConverter
        from classical_baselines import (
            DijkstraSolver, GreedySolver, LoadBalancedSolver,
            SimulatedAnnealingSolver, ParallelTemperingSolver,
        )
        from quantum_module import QAOACircuitBuilder, QAOAOptimizer, QAOADecoder
        from hardware_mapping import BackendManager, HardwareAwareTranspiler, CircuitStats
        from ml_optimizer import MLTrainer, QAOAAnglePredictor, QUBOWeightPredictor
        from benchmark import BenchmarkMetrics, TTS99Calculator, BenchmarkReporter

        cfg = self.config

        # ── Graph & QUBO ──────────────────────────────────────────────
        self.loader = NetworkLoader()
        self.preprocessor = GraphPreprocessor()
        self.qubo_builder = QUBOBuilder(
            penalty_path=cfg.get("penalty_path_validity", 10.0),
            penalty_flow=cfg.get("penalty_flow_conservation", 8.0),
            penalty_capacity=cfg.get("penalty_capacity", 5.0),
            w_latency=cfg.get("latency_weight", 1.0),
            w_congestion=cfg.get("congestion_weight", 2.0),
            w_loss=cfg.get("loss_weight", 3.0),
        )
        self.ising_converter = IsingConverter()

        # ── Classical solvers ─────────────────────────────────────────
        sa_cfg = cfg.get("sa", {})
        pt_cfg = cfg.get("pt", {})
        self.solvers = {
            "dijkstra": DijkstraSolver(),
            "greedy": GreedySolver(),
            "load_balanced": LoadBalancedSolver(),
            "simulated_annealing": SimulatedAnnealingSolver(
                num_reads=sa_cfg.get("num_reads", 100),
                num_sweeps=sa_cfg.get("num_sweeps", 2000),
                beta_range=tuple(sa_cfg.get("beta_range", [0.1, 10.0])),
                seed=sa_cfg.get("seed", 42),
            ),
            "parallel_tempering": ParallelTemperingSolver(
                num_replicas=pt_cfg.get("num_replicas", 8),
                num_sweeps=pt_cfg.get("num_sweeps", 2000),
                beta_min=pt_cfg.get("beta_min", 0.1),
                beta_max=pt_cfg.get("beta_max", 10.0),
                swap_interval=pt_cfg.get("swap_interval", 50),
                seed=pt_cfg.get("seed", 42),
            ),
        }

        # ── Quantum ───────────────────────────────────────────────────
        q_cfg = cfg.get("quantum", {})
        self.p = q_cfg.get("p_depth", 3)
        self.shots = q_cfg.get("shots", 4096)
        self.qaoa_builder = QAOACircuitBuilder(p=self.p)
        self.backend_manager = BackendManager()
        self.transpiler = HardwareAwareTranspiler(
            optimization_level=cfg.get("optimization_level", 3),
        )

        # ── ML ────────────────────────────────────────────────────────
        ml_cfg = cfg.get("ml", {})
        device = ml_cfg.get("device", "auto")
        self.ml_trainer = MLTrainer(p=self.p, device=device,
                                    checkpoint_dir=ml_cfg.get("checkpoint_dir", "ml_optimizer/checkpoints"))
        self.angle_predictor = self.ml_trainer.angle_predictor
        self.qubo_predictor = self.ml_trainer.qubo_predictor
        self._ml_trained = False

        # ── Benchmark ─────────────────────────────────────────────────
        bm_cfg = cfg.get("benchmark", {})
        self.benchmark_metrics = BenchmarkMetrics(
            target_quality_fraction=bm_cfg.get("tts_target_quality", 0.95)
        )
        self.tts_calculator = TTS99Calculator()
        self.reporter = BenchmarkReporter(
            results_dir=cfg.get("results_dir", "results")
        )

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def build_instance(
        self,
        G,
        src: int,
        dst: int,
        traffic_matrix=None,
        use_ml_weights: bool = False,
    ) -> RoutingInstance:
        """Build a RoutingInstance: QUBO + Ising from graph."""
        inst = RoutingInstance(G=G, src=src, dst=dst,
                               traffic_matrix=traffic_matrix)

        # QUBO construction
        if use_ml_weights and self._ml_trained:
            adj = self.qubo_predictor.predict_adjustments(G, src=src, dst=dst,
                                                          traffic_matrix=traffic_matrix)
            logger.debug("ML weight adjustments: %s", adj)
            qubo_result = self.qubo_builder.build_edge_routing_qubo(G, src, dst)
            Q_adj = self.qubo_predictor.apply_adjustments(
                qubo_result.Q, qubo_result.variable_map, adj, G, src, dst
            )
            qubo_result = type(qubo_result)(
                Q=Q_adj,
                linear=np.diag(Q_adj),
                variable_map=qubo_result.variable_map,
                n_variables=qubo_result.n_variables,
                metadata={**qubo_result.metadata, "ml_adjusted": True},
            )
        else:
            qubo_result = self.qubo_builder.build_edge_routing_qubo(G, src, dst)

        inst.qubo_result = qubo_result

        # Ising conversion
        try:
            inst.ising_result = self.ising_converter.qubo_to_ising(qubo_result.Q)
        except Exception as exc:
            logger.warning("Ising conversion failed: %s", exc)

        return inst

    def run_classical_solvers(
        self, inst: RoutingInstance, solvers: list = None
    ) -> dict:
        """Run classical solvers on instance. Returns {solver_name: SolverResult}."""
        to_run = solvers or ["dijkstra", "greedy", "load_balanced",
                             "simulated_annealing", "parallel_tempering"]
        results = {}
        for name in to_run:
            if name not in self.solvers:
                continue
            solver = self.solvers[name]
            logger.info("Running solver: %s", name)
            try:
                if name in ("dijkstra", "greedy", "load_balanced"):
                    r = solver.solve(inst.G, inst.src, inst.dst,
                                     qubo_result=inst.qubo_result)
                else:
                    r = solver.solve(
                        inst.qubo_result.Q,
                        inst.qubo_result.variable_map,
                        inst.G, inst.src, inst.dst,
                    )
                results[name] = r
            except Exception as exc:
                logger.error("Solver %s failed: %s", name, exc)
        return results

    def run_qaoa(
        self,
        inst: RoutingInstance,
        use_hardware_aware: bool = False,
        use_ml_angles: bool = False,
        use_noise: bool = False,
    ) -> Optional[object]:
        """
        Run QAOA solver on instance.

        Returns QAOASolverResult or None on failure.
        """
        if inst.ising_result is None:
            logger.error("Ising result not available for QAOA")
            return None

        t_start = time.time()
        n_qubits = inst.ising_result.n_spins

        if n_qubits == 0:
            logger.warning("Zero qubits — skipping QAOA")
            return None

        try:
            # ── Initial angles ────────────────────────────────────────
            if use_ml_angles and self._ml_trained:
                gamma0, beta0 = self.angle_predictor.predict(
                    inst.G, inst.qubo_result.Q, inst.src, inst.dst,
                    inst.traffic_matrix
                )
                logger.debug("ML warm-start angles: gamma=%s beta=%s", gamma0, beta0)
            else:
                gamma0 = None
                beta0 = None

            # ── Backend ───────────────────────────────────────────────
            noise_model = None
            backend = self.backend_manager.get_aer_simulator()

            if use_hardware_aware:
                fake_backend = self.backend_manager.get_fake_backend("fake_montreal")
                if fake_backend:
                    backend_props = self.backend_manager.extract_properties(fake_backend)
                    if use_noise:
                        noise_model = self.backend_manager.make_noise_model_from_properties(backend_props)

            # ── Build circuit ─────────────────────────────────────────
            J = inst.ising_result.J
            h = inst.ising_result.h
            offset = inst.ising_result.offset

            qc = self.qaoa_builder.build_circuit(J, h, n_qubits)

            # Transpile if hardware-aware
            circuit_stats = None
            if use_hardware_aware and backend is not None:
                try:
                    tr = self.transpiler.transpile(qc, backend)
                    from hardware_mapping import CircuitStats
                    circuit_stats = CircuitStats.from_transpilation_result(tr, n_qubits)
                    qc_exec = tr.transpiled_circuit
                except Exception as exc:
                    logger.warning("Transpilation failed: %s — using logical circuit", exc)
                    qc_exec = qc
            else:
                qc_exec = qc

            # ── Optimize ──────────────────────────────────────────────
            optimizer = QAOAOptimizer(
                p=self.p,
                shots=self.shots,
                optimizer=self.config.get("optimizer", "COBYLA"),
                max_iter=self.config.get("max_iter", 300),
                backend_name="aer_simulator",
                use_noise=use_noise and noise_model is not None,
                noise_model=noise_model,
            )

            from quantum_module import QAOAOptimizer
            opt_result = optimizer.optimize(
                qc_exec, self.qaoa_builder, J, h, offset,
                initial_gamma=gamma0, initial_beta=beta0,
            )

            # ── Decode ────────────────────────────────────────────────
            bound_qc = self.qaoa_builder.bind_parameters(
                qc, opt_result.optimal_gamma, opt_result.optimal_beta
            )
            counts = optimizer._run_circuit(bound_qc)
            from quantum_module import QAOADecoder
            decoder = QAOADecoder(
                inst.qubo_result.Q, inst.qubo_result.variable_map,
                inst.G, inst.src, inst.dst,
            )
            decoded = decoder.decode_counts(counts)
            best = decoder.best_feasible(decoded)

            runtime = time.time() - t_start

            return {
                "opt_result": opt_result,
                "decoded_solutions": decoded,
                "best_solution": best,
                "circuit_stats": circuit_stats,
                "runtime": runtime,
                "n_qubits": n_qubits,
                "feasibility_rate": decoder.feasibility_rate(decoded),
            }

        except Exception as exc:
            logger.error("QAOA run failed: %s", exc, exc_info=True)
            return None

    def train_ml(self, n_instances: int = 200):
        """Train ML models from synthetic routing instances."""
        def graph_factory(n, seed):
            G = self.loader.load_synthetic(n, edge_prob=0.35, seed=seed)
            G_sub = self.preprocessor.get_subgraph(G, max_nodes=n)
            pairs = self.preprocessor.compute_routing_pairs(G_sub, k=1)
            if not pairs:
                import networkx as nx
                nodes = list(G_sub.nodes())
                src, dst = nodes[0], nodes[-1]
            else:
                src, dst = pairs[0]
            return G_sub, src, dst

        instances, target_angles, target_weights = self.ml_trainer.generate_training_data(
            graph_factory, self.qubo_builder, n_instances=n_instances
        )
        self.ml_trainer.train(instances, target_angles, target_weights)
        self._ml_trained = True
        logger.info("ML training complete")

    def run_full_benchmark(
        self,
        G,
        src: int,
        dst: int,
        n_trials: int = 20,
        solvers_to_run: list = None,
        run_quantum: bool = True,
    ) -> dict:
        """
        Run full benchmark: all solvers × n_trials, collect metrics.

        Returns dict of results ready for BenchmarkMetrics.aggregate().
        """
        if solvers_to_run is None:
            solvers_to_run = ["dijkstra", "greedy", "load_balanced",
                              "simulated_annealing", "parallel_tempering"]

        from benchmark.metrics import RunResult
        all_results = []
        solver_energies = {}
        solver_convergence = {}

        logger.info("Building routing instance...")
        inst = self.build_instance(G, src, dst)

        # ── Classical solvers (repeated trials) ──────────────────────
        for solver_name in solvers_to_run:
            solver_energies[solver_name] = []
            solver_convergence[solver_name] = []
            for trial in range(n_trials):
                # Re-seed for reproducibility
                if solver_name == "simulated_annealing":
                    self.solvers[solver_name].seed = trial
                elif solver_name == "parallel_tempering":
                    self.solvers[solver_name].seed = trial

                classic_results = self.run_classical_solvers(inst, [solver_name])
                if solver_name in classic_results:
                    r = classic_results[solver_name]
                    all_results.append(RunResult(
                        solver=solver_name, trial=trial,
                        best_energy=r.best_energy, routing_cost=r.routing_cost,
                        latency=r.latency, congestion_score=r.congestion_score,
                        loss_score=r.loss_score, runtime_seconds=r.runtime_seconds,
                        feasible=r.feasible, n_iterations=r.n_iterations,
                    ))
                    solver_energies[solver_name].append(r.best_energy)
                    if hasattr(r, "energy_history") and r.energy_history:
                        solver_convergence[solver_name] = r.energy_history

        # ── QAOA variants ─────────────────────────────────────────────
        if run_quantum:
            for variant, use_hw, use_ml in [
                ("qaoa_baseline", False, False),
                ("qaoa_hardware_aware", True, False),
                ("qaoa_ml_hardware", True, True),
            ]:
                solver_energies[variant] = []
                for trial in range(min(n_trials, 5)):  # QAOA is slower
                    logger.info("QAOA variant=%s trial=%d", variant, trial)
                    res = self.run_qaoa(inst, use_hardware_aware=use_hw, use_ml_angles=use_ml)
                    if res and res["best_solution"]:
                        bs = res["best_solution"]
                        all_results.append(RunResult(
                            solver=variant, trial=trial,
                            best_energy=bs.qubo_energy, routing_cost=bs.routing_cost,
                            latency=bs.latency, congestion_score=bs.congestion_score,
                            loss_score=bs.loss_score, runtime_seconds=res["runtime"],
                            feasible=bs.feasible, n_iterations=res["opt_result"].n_function_evals,
                            n_qubits=res["n_qubits"],
                            circuit_depth=res["circuit_stats"].depth_physical if res["circuit_stats"] else None,
                            extra={"feasibility_rate": res["feasibility_rate"]},
                        ))
                        solver_energies[variant].append(bs.qubo_energy)
                        solver_convergence[variant] = [
                            e for _, e in res["opt_result"].convergence_history
                        ]

        return {
            "all_results": all_results,
            "solver_energies": solver_energies,
            "solver_convergence": solver_convergence,
            "instance": inst,
        }
