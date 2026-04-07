#!/usr/bin/env python3
"""
HQMRE — Hardware-Aware Quantum ML Routing Engine
=================================================
Main entry point.  Run this script to see the complete system in action.

Usage:
    python run.py                    # full demo with all solvers
    python run.py --quick            # fast demo (small graph, fewer trials)
    python run.py --sweep            # full parameter sweep
    python run.py --train-ml         # train ML models first
    python run.py --no-quantum       # skip QAOA (classical only)
    python run.py --nodes 12 --trials 15

All results and plots are saved to ./results/
"""

import argparse
import logging
import os
import sys
import time

import numpy as np

# ── Make all project modules importable ──────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("hqmre")


# ─────────────────────────────────────────────────────────────────────────────
def print_banner():
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║   HQMRE — Hardware-Aware Quantum ML Routing Engine                          ║
║   Hybrid QUBO/Ising Solver Benchmark for Telecom Network Routing            ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")


def print_section(title: str):
    print(f"\n{'─'*78}")
    print(f"  {title}")
    print(f"{'─'*78}")


def check_imports():
    """Verify key dependencies are available."""
    print_section("Dependency Check")
    deps = {
        "numpy": "numpy",
        "networkx": "networkx",
        "scipy": "scipy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "torch": "torch",
        "qiskit": "qiskit",
        "qiskit_aer": "qiskit_aer",
    }
    all_ok = True
    for name, mod in deps.items():
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  ✓  {name:<20} v{ver}")
        except ImportError:
            print(f"  ✗  {name:<20} NOT INSTALLED")
            all_ok = False

    # CUDA check
    try:
        import torch
        if torch.cuda.is_available():
            print(f"\n  ✓  CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA version: {torch.version.cuda}")
        else:
            print("\n     CUDA not available — ML will use CPU")
    except Exception:
        pass

    if not all_ok:
        print("\n  Some dependencies missing. Run:  pip install -r requirements.txt\n")
    return all_ok


def run_demo(args):
    """Full end-to-end demonstration."""
    from graph_engine import NetworkLoader, GraphPreprocessor, TrafficManager
    from qubo_formulation import QUBOBuilder
    from ising_utils import IsingConverter
    from classical_baselines import (
        DijkstraSolver, GreedySolver, LoadBalancedSolver,
        SimulatedAnnealingSolver, ParallelTemperingSolver,
    )
    from benchmark.metrics import BenchmarkMetrics, RunResult
    from benchmark.tts99 import TTS99Calculator
    from benchmark.reporter import BenchmarkReporter

    os.makedirs("results", exist_ok=True)
    reporter = BenchmarkReporter(results_dir="results")

    # ── 1. Build Graph ────────────────────────────────────────────────────────
    print_section("1. Network Graph Construction")
    n_nodes = args.nodes
    seed = args.seed

    loader = NetworkLoader()
    G = loader.load_synthetic(n_nodes, edge_prob=0.35, seed=seed)
    preprocessor = GraphPreprocessor()
    G = preprocessor.get_subgraph(G, max_nodes=n_nodes)

    feats = preprocessor.extract_features(G)
    print(f"  Topology:        Synthetic ISP-like ({n_nodes} nodes)")
    print(f"  Nodes:           {feats['n_nodes']}")
    print(f"  Edges:           {feats['n_edges']}")
    print(f"  Density:         {feats['density']:.4f}")
    print(f"  Avg degree:      {feats['avg_degree']:.2f}")
    print(f"  Avg latency:     {feats['avg_latency']:.2f} ms")
    print(f"  Avg congestion:  {feats['avg_congestion']:.4f}")
    print(f"  Avg loss:        {feats['avg_loss']:.6f}")

    # Apply traffic
    tm = TrafficManager()
    traffic = tm.generate_traffic_matrix(G, intensity=0.5, seed=seed)
    tm.update_congestion(G, traffic)
    print(f"\n  Traffic matrix applied (intensity=0.5)")

    # Choose src/dst
    pairs = preprocessor.compute_routing_pairs(G, k=1)
    if pairs:
        src, dst = pairs[0]
    else:
        nodes = list(G.nodes())
        src, dst = nodes[0], nodes[-1]
    print(f"  Routing demand:  node {src}  →  node {dst}")

    # ── 2. QUBO Formulation ───────────────────────────────────────────────────
    print_section("2. QUBO Formulation")
    builder = QUBOBuilder(penalty_path=10.0, penalty_flow=8.0, penalty_capacity=5.0,
                          w_latency=1.0, w_congestion=2.0, w_loss=3.0)
    qubo_result = builder.build_edge_routing_qubo(G, src, dst)

    print(f"  Binary variables (edges):  {qubo_result.n_variables}")
    print(f"  QUBO matrix shape:         {qubo_result.Q.shape}")
    diag = np.diag(qubo_result.Q)
    print(f"  Q diagonal range:          [{diag.min():.4f}, {diag.max():.4f}]")
    off_diag = qubo_result.Q - np.diag(diag)
    nz = np.count_nonzero(off_diag)
    print(f"  Non-zero off-diagonal:     {nz}")

    validation = builder.validate_qubo(qubo_result)
    print(f"  QUBO validation:           {'PASSED' if validation['is_valid'] else 'FAILED'}")

    # ── 3. Ising Conversion ───────────────────────────────────────────────────
    print_section("3. QUBO → Ising Conversion")
    converter = IsingConverter()
    ising_result = converter.qubo_to_ising(qubo_result.Q)

    print(f"  Spins (qubits):            {ising_result.n_spins}")
    print(f"  ZZ couplings |J|:          {len(ising_result.J)}")
    print(f"  Local fields |h|:          {len(ising_result.h)}")
    print(f"  Energy offset:             {ising_result.offset:.4f}")

    ok = converter.roundtrip_check(qubo_result.Q)
    print(f"  QUBO↔Ising roundtrip:      {'VERIFIED ✓' if ok else 'MISMATCH ✗'}")

    # ── 4. Classical Baselines ────────────────────────────────────────────────
    print_section("4. Classical Solver Benchmarks")
    n_trials = args.trials
    print(f"  Running {n_trials} trials per solver...\n")

    solvers = {
        "Dijkstra":           DijkstraSolver(),
        "Greedy":             GreedySolver(),
        "LoadBalanced":       LoadBalancedSolver(),
        "Sim. Annealing":     SimulatedAnnealingSolver(num_reads=50, num_sweeps=1000),
        "Parallel Tempering": ParallelTemperingSolver(num_replicas=6, num_sweeps=500),
    }

    all_results = []
    solver_energies = {}
    solver_convergence = {}

    fmt = "  {:<22} E={:>10.4f}  cost={:>8.4f}  lat={:>6.2f}  cong={:>6.4f}  " \
          "feas={:>5}  t={:.4f}s"

    for sname, solver in solvers.items():
        solver_energies[sname] = []
        solver_convergence[sname] = []
        for trial in range(n_trials):
            try:
                if sname in ("Dijkstra", "Greedy", "LoadBalanced"):
                    r = solver.solve(G, src, dst, qubo_result=qubo_result)
                else:
                    solver.seed = trial
                    r = solver.solve(qubo_result.Q, qubo_result.variable_map, G, src, dst)
                solver_energies[sname].append(r.best_energy)
                if hasattr(r, "energy_history") and r.energy_history:
                    solver_convergence[sname] = r.energy_history
                all_results.append(RunResult(
                    solver=sname, trial=trial,
                    best_energy=r.best_energy, routing_cost=r.routing_cost,
                    latency=r.latency, congestion_score=r.congestion_score,
                    loss_score=r.loss_score, runtime_seconds=r.runtime_seconds,
                    feasible=r.feasible, n_iterations=r.n_iterations,
                ))
            except Exception as exc:
                logger.warning("Solver %s trial %d failed: %s", sname, trial, exc)

        if solver_energies[sname]:
            best_e = min(solver_energies[sname])
            mean_e = np.mean(solver_energies[sname])
            # Use last successful result for display
            last_r = all_results[-1] if all_results else None
            if last_r:
                print(fmt.format(
                    sname, mean_e, last_r.routing_cost, last_r.latency,
                    last_r.congestion_score, str(last_r.feasible)[:5],
                    last_r.runtime_seconds
                ))

    # ── 5. QAOA Quantum Solver ────────────────────────────────────────────────
    if not args.no_quantum and ising_result.n_spins <= 20:
        print_section("5. QAOA Quantum Solver")
        from quantum_module import QAOACircuitBuilder, QAOAOptimizer, QAOADecoder

        p_depth = args.p
        print(f"  QAOA depth p={p_depth}, shots={args.shots}, qubits={ising_result.n_spins}")

        qaoa_variants = [
            ("QAOA (baseline)",      False, False),
            ("QAOA (hardware-aware)", True,  False),
        ]

        for variant_name, use_hw, use_ml in qaoa_variants:
            print(f"\n  Running {variant_name}...")
            try:
                t0 = time.time()
                qaoa_builder = QAOACircuitBuilder(p=p_depth)
                qc = qaoa_builder.build_circuit(
                    ising_result.J, ising_result.h, ising_result.n_spins
                )

                # Hardware-aware transpilation
                circuit_stats = None
                qc_exec = qc
                if use_hw:
                    from hardware_mapping import BackendManager, HardwareAwareTranspiler, CircuitStats
                    bm = BackendManager()
                    fake_backend = bm.get_fake_backend("fake_montreal")
                    backend = bm.get_aer_simulator()
                    if fake_backend:
                        tr = HardwareAwareTranspiler(optimization_level=3).transpile(qc, fake_backend)
                        circuit_stats = CircuitStats.from_transpilation_result(tr, ising_result.n_spins)
                        print(f"    Logical depth:   {circuit_stats.depth_logical}")
                        print(f"    Physical depth:  {circuit_stats.depth_physical}")
                        print(f"    CX gates (phys): {circuit_stats.n_cx_physical}")
                        print(f"    SWAP overhead:   {circuit_stats.swap_overhead}")

                optimizer = QAOAOptimizer(
                    p=p_depth, shots=args.shots,
                    optimizer="COBYLA", max_iter=args.qaoa_iter,
                )
                opt_result = optimizer.optimize(
                    qc, qaoa_builder,
                    ising_result.J, ising_result.h, ising_result.offset,
                )

                bound_qc = qaoa_builder.bind_parameters(
                    qc, opt_result.optimal_gamma, opt_result.optimal_beta
                )
                counts = optimizer._run_circuit(bound_qc)
                decoder = QAOADecoder(
                    qubo_result.Q, qubo_result.variable_map, G, src, dst
                )
                decoded = decoder.decode_counts(counts)
                best = decoder.best_feasible(decoded)
                feas_rate = decoder.feasibility_rate(decoded)

                t_elapsed = time.time() - t0

                if best:
                    solver_energies[variant_name] = [best.qubo_energy]
                    all_results.append(RunResult(
                        solver=variant_name, trial=0,
                        best_energy=best.qubo_energy, routing_cost=best.routing_cost,
                        latency=best.latency, congestion_score=best.congestion_score,
                        loss_score=best.loss_score, runtime_seconds=t_elapsed,
                        feasible=best.feasible, n_iterations=opt_result.n_function_evals,
                        n_qubits=ising_result.n_spins,
                    ))
                    print(f"    Optimal γ: {np.round(opt_result.optimal_gamma, 3)}")
                    print(f"    Optimal β: {np.round(opt_result.optimal_beta, 3)}")
                    print(f"    Best energy:        {best.qubo_energy:.4f}")
                    print(f"    Routing cost:       {best.routing_cost:.4f}")
                    print(f"    Feasible:           {best.feasible}")
                    print(f"    Feasibility rate:   {feas_rate:.3f}")
                    print(f"    Active edges:       {best.active_edges}")
                    print(f"    Func evaluations:   {opt_result.n_function_evals}")
                    print(f"    Total time:         {t_elapsed:.3f}s")
                    solver_convergence[variant_name] = [
                        e for _, e in opt_result.convergence_history
                    ]

            except Exception as exc:
                logger.warning("%s failed: %s", variant_name, exc, exc_info=True)

    # ── 6. ML Optimizer Training ──────────────────────────────────────────────
    if not args.no_ml and not args.quick:
        print_section("6. ML Optimizer (CUDA-Aware Training)")
        from ml_optimizer import MLTrainer

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Device: {device.upper()}")

            ml_trainer = MLTrainer(p=args.p, device=device)

            def graph_factory(n, seed):
                _G = loader.load_synthetic(n, edge_prob=0.35, seed=seed)
                _G = preprocessor.get_subgraph(_G, max_nodes=n)
                _pairs = preprocessor.compute_routing_pairs(_G, k=1)
                if _pairs:
                    _s, _d = _pairs[0]
                else:
                    _nodes = list(_G.nodes())
                    _s, _d = _nodes[0], _nodes[-1]
                return _G, _s, _d

            n_instances = 50 if args.quick else 150
            print(f"  Generating {n_instances} training instances...")
            instances, target_angles, target_weights = ml_trainer.generate_training_data(
                graph_factory, builder, n_instances=n_instances, seed=seed
            )
            print(f"  Training AnglePredictor and QUBOWeightPredictor...")
            history = ml_trainer.train(
                instances, target_angles, target_weights,
                angle_epochs=50 if args.quick else 100,
                qubo_epochs=40 if args.quick else 80,
                verbose=False,
            )

            final_angle_loss = history["angle"]["val_loss"][-1]
            final_qubo_loss = history["qubo_weight"]["train_loss"][-1]
            print(f"  AnglePredictor  — final val loss:  {final_angle_loss:.6f}")
            print(f"  QUBOPredictor   — final train loss: {final_qubo_loss:.6f}")

            # Test prediction
            gamma_pred, beta_pred = ml_trainer.angle_predictor.predict(G, qubo_result.Q, src, dst)
            print(f"  Predicted γ init: {np.round(gamma_pred, 3)}")
            print(f"  Predicted β init: {np.round(beta_pred, 3)}")

            adj = ml_trainer.qubo_predictor.predict_adjustments(G, qubo_result.Q, src, dst)
            print(f"  QUBO weight adjustments:")
            for k, v in adj.items():
                print(f"    {k:<25} {v:.4f}")

            ml_trainer.save_models()
            print(f"  Models saved to ml_optimizer/checkpoints/")

        except Exception as exc:
            logger.warning("ML training failed: %s", exc, exc_info=True)

    # ── 7. Benchmark Aggregation ──────────────────────────────────────────────
    print_section("7. Benchmark Results")
    bm = BenchmarkMetrics()
    best_known = min(e for energies in solver_energies.values() for e in energies) if solver_energies else 0.0
    stats = bm.aggregate(all_results, best_known=best_known)
    stats_df = bm.to_dataframe(stats)
    reporter.print_solver_table(stats_df, title="HQMRE Solver Comparison")
    reporter.save_benchmark_csv(stats_df)

    # ── 8. TTS99 Analysis ─────────────────────────────────────────────────────
    print_section("8. TTS99 Analysis")
    from benchmark.tts99 import TTS99Calculator, TTS99Result
    tts_calc = TTS99Calculator()
    tts_input = {}
    for stat in stats:
        sname = stat.solver
        energies_arr = np.array(solver_energies.get(sname, [stat.best_energy]))
        mean_rt = stat.mean_runtime
        tts_input[sname] = {
            "energies": energies_arr,
            "runtimes": np.full(len(energies_arr), mean_rt),
        }
    tts_results = tts_calc.compute_all(tts_input, best_known=best_known)
    reporter.print_tts99_table(tts_calc.summary_table(tts_results))

    # ── 9. Plots ──────────────────────────────────────────────────────────────
    print_section("9. Generating Plots")
    try:
        reporter.plot_approximation_ratio(stats, save=True)
        reporter.plot_tts99_comparison(tts_results, save=True)
        if any(solver_energies.values()):
            reporter.plot_energy_distribution(solver_energies, save=True)
        if any(solver_convergence.values()):
            reporter.plot_convergence(solver_convergence, save=True)
        print("  All plots saved to ./results/")
    except Exception as exc:
        logger.warning("Plotting failed: %s", exc)

    print_section("Run Complete")
    print("  Results saved to  ./results/")
    print("  Run `python run.py --help` to see all options.\n")


def run_sweep(args):
    """Run full parameter sweep."""
    print_section("Parameter Sweep")
    from orchestrator import HQMREPipeline
    from experiments import ExperimentRunner, SweepConfig
    import yaml

    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    flat_cfg = {}
    for section in cfg.values():
        if isinstance(section, dict):
            flat_cfg.update(section)

    pipeline = HQMREPipeline(flat_cfg)

    sweep = SweepConfig(
        graph_sizes=args.sweep_sizes,
        repeat_per_config=args.sweep_trials,
        run_quantum=not args.no_quantum,
    )
    runner = ExperimentRunner(pipeline, sweep)
    print(f"  Total configurations: {sweep.n_configs()}")
    print(f"  Repeats per config:   {sweep.repeat_per_config}")
    records = runner.run_sweep(verbose=True)
    print(f"\n  Sweep complete: {len(records)} total trials")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="HQMRE — Hardware-Aware Quantum ML Routing Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Fast demo: small graph, fewer iterations")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full parameter sweep")
    parser.add_argument("--no-quantum", action="store_true",
                        help="Skip QAOA quantum solvers")
    parser.add_argument("--no-ml", action="store_true",
                        help="Skip ML training")
    parser.add_argument("--train-ml", action="store_true",
                        help="Train and save ML models before benchmark")
    parser.add_argument("--nodes", type=int, default=10,
                        help="Number of graph nodes (default 10)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per solver (default 10)")
    parser.add_argument("--p", type=int, default=2,
                        help="QAOA circuit depth (default 2)")
    parser.add_argument("--shots", type=int, default=2048,
                        help="QAOA measurement shots (default 2048)")
    parser.add_argument("--qaoa-iter", type=int, default=150,
                        help="QAOA optimizer iterations (default 150)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--sweep-sizes", nargs="+", type=int, default=[8, 12, 16],
                        help="Graph sizes for sweep")
    parser.add_argument("--sweep-trials", type=int, default=5,
                        help="Trials per sweep config")

    args = parser.parse_args()

    # Quick mode overrides
    if args.quick:
        args.nodes = args.nodes if args.nodes != 10 else 8
        args.trials = 3
        args.qaoa_iter = 50
        args.shots = 512

    print_banner()
    check_imports()

    if args.sweep:
        run_sweep(args)
    else:
        run_demo(args)


if __name__ == "__main__":
    main()
