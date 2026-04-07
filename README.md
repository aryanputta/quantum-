# Hardware-Aware Quantum ML Routing Engine (HQMRE)

> A production-grade hybrid quantum–classical benchmark system for telecom
> network routing optimization via explicit QUBO/Ising reformulation,
> hardware-aware QAOA, and ML-assisted coefficient adaptation.

---

## Overview

HQMRE is a unified optimization and benchmarking framework that reformulates
the telecom routing problem as a **Quadratic Unconstrained Binary Optimization
(QUBO)** instance, converts it into an **Ising Hamiltonian**, and solves it
using a diverse family of classical and quantum algorithms under reproducible
experimental conditions.

The primary research contribution is **not** a claim of quantum advantage.
It is instead a disciplined, apples-to-apples comparison across five solver
families—graph-theoretic, annealing-style, and gate-model quantum—on the same
QUBO backbone, with rigorous metrics including **TTS99** (time-to-solution at
99% success probability) and feasibility rate.

---

## Problem Statement

Given a directed graph G = (V, E) where each edge e carries attributes
{latency, bandwidth, congestion, loss}, and a routing demand (src, dst),
find a path P ⊆ E that minimizes the composite objective:

```
min  Σ_{e ∈ P}  [ w_l · lat_e  +  w_c · cong_e  +  w_p · loss_e ]

s.t. flow conservation at every node
     capacity constraints on congested edges
```

This is reformulated as a QUBO (binary variables x_e ∈ {0,1} for each edge):

```
min  x^T Q x
```

where Q ∈ ℝ^{n×n} encodes both the objective and all constraints as quadratic
penalty terms, and then converted to an Ising Hamiltonian via the substitution
x_i = (1 + s_i)/2 with s_i ∈ {−1, +1}.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                         HQMRE Pipeline                               │
│                                                                      │
│  Real Network Data        QUBO Formulation       Ising Hamiltonian  │
│  (SNAP / Rocketfuel)  →  Q = objectives +   →   J_{ij}, h_i, c     │
│  NetworkX DiGraph         penalty terms                              │
│        │                       │                       │             │
│        │              ┌────────┴────────┐              │             │
│        │              │  Classical      │              │             │
│        │              │  Solvers        │     ┌────────┴──────────┐ │
│        │              │  ─ Dijkstra     │     │  Quantum Module   │ │
│        └──────────────│  ─ Greedy       │     │  ─ QAOA (p layers)│ │
│                       │  ─ Load-Bal.    │     │  ─ Noise model    │ │
│                       │  ─ Sim. Ann.    │     │  ─ HW transpile   │ │
│                       │  ─ Par. Temp.   │     │  ─ IBM backend    │ │
│                       └────────┬────────┘     └────────┬──────────┘ │
│                                │                        │            │
│             ┌──────────────────┴────────────────────────┘            │
│             │            Benchmark Module                             │
│             │   routing cost · latency · congestion · loss            │
│             │   approximation ratio · TTS99 · feasibility rate        │
│             │   circuit depth (logical vs physical)                   │
│             └──────────────────────────────────────────────────────── │
│                                                                      │
│   ML Optimizer (PyTorch, CUDA-aware)                                 │
│   ─ AnglePredictor: graph features → QAOA warm-start angles         │
│   ─ QUBOWeightPredictor: traffic features → penalty adjustment       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Repository Structure

```
quantum-/
├── config.yaml                  # All hyperparameters; single source of truth
├── requirements.txt
├── run.py                       # Main entry point (see Usage)
│
├── data/
│   ├── download_datasets.py     # Downloads SNAP / Rocketfuel / CAIDA data
│   └── sample_topologies.py     # Embedded reference topologies (no I/O needed)
│
├── graph_engine/
│   ├── loader.py                # NetworkLoader: synthetic + real dataset loaders
│   ├── preprocessor.py          # GraphPreprocessor: features, normalization, subgraph
│   └── traffic.py               # TrafficManager: demand matrix, congestion updates
│
├── qubo_formulation/
│   ├── qubo_builder.py          # QUBOBuilder → QUBOResult (Q matrix + variable map)
│   ├── objectives.py            # Latency / congestion / loss objective terms
│   └── constraints.py           # Flow conservation + capacity penalty expansion
│
├── ising_utils/
│   ├── converter.py             # IsingConverter: QUBO ↔ Ising (exact roundtrip)
│   ├── energy.py                # Energy evaluation in both representations
│   └── validators.py            # Numerical consistency checks
│
├── classical_baselines/
│   ├── dijkstra.py              # Composite-weight shortest path
│   ├── greedy.py                # Greedy hop-by-hop with Dijkstra fallback
│   ├── load_balanced.py         # Top-k path selection by max-congestion score
│   ├── simulated_annealing.py   # SA with linear β schedule; warm-started
│   └── parallel_tempering.py    # Replica-exchange PT with Metropolis swaps
│
├── quantum_module/
│   ├── qaoa.py                  # QAOACircuitBuilder: parameterized circuit from Ising H
│   ├── optimizer.py             # QAOAOptimizer: COBYLA / SPSA / L-BFGS-B
│   └── decoder.py               # QAOADecoder: bitstrings → routing decisions
│
├── hardware_mapping/
│   ├── backend_utils.py         # BackendManager: fake IBM backends, noise models
│   ├── transpiler.py            # HardwareAwareTranspiler: SABRE layout + routing
│   └── circuit_stats.py         # Logical vs physical circuit statistics
│
├── ml_optimizer/
│   ├── models.py                # AnglePredictor, QUBOCoefficientPredictor (PyTorch)
│   ├── feature_extractor.py     # 22-dimensional graph + QUBO feature vector
│   ├── angle_predictor.py       # End-to-end QAOA warm-start predictor
│   ├── qubo_predictor.py        # QUBO weight adjustment predictor
│   └── trainer.py               # MLTrainer: data generation + joint training
│
├── orchestrator/
│   └── pipeline.py              # HQMREPipeline: end-to-end coordination
│
├── benchmark/
│   ├── metrics.py               # BenchmarkMetrics: aggregation + approximation ratio
│   ├── tts99.py                 # TTS99Calculator: Wilson CI, TTS50/TTS99
│   └── reporter.py              # BenchmarkReporter: tables, plots, CSV export
│
├── experiments/
│   ├── sweep.py                 # SweepConfig: reproducible parameter sweeps
│   └── runner.py                # ExperimentRunner: sweep executor + JSON export
│
├── results/                     # Auto-populated: plots, tables, CSVs
└── tests/
    ├── test_qubo.py             # QUBO shape, symmetry, feasibility ordering
    ├── test_ising.py            # Energy equivalence (exhaustive + sampled)
    ├── test_baselines.py        # Solver correctness on known small graphs
    ├── test_benchmark.py        # TTS99 formula, BenchmarkMetrics aggregation
    └── test_graph_engine.py     # Loader, preprocessor, traffic manager
```

---

## QUBO Formulation

### Binary Variables

One binary variable x_e ∈ {0, 1} per directed edge e = (u, v).

### Objective

```
objective(x) = Σ_e  [ w_l · lat_e + w_c · cong_e + w_p · loss_e ] · x_e
```

This is a linear function of x, placed on the diagonal of Q.

### Constraints (Penalty Expansion)

Flow conservation at node v with demand d_v ∈ {−1, 0, +1}:

```
penalty · ( Σ_{(v,w)∈E} x_{vw}  −  Σ_{(u,v)∈E} x_{uv}  −  d_v )²
```

The squared term is expanded into diagonal (linear) and off-diagonal
(quadratic) QUBO entries using the identity x_i² = x_i (binary variables).
The penalty is **auto-scaled** to exceed the maximum possible path cost,
guaranteeing that no infeasible solution can have lower QUBO energy than
any feasible one:

```
penalty ≥ 5 × max_edge_cost × |V|
```

### Ising Conversion

Standard substitution x_i = (1 + s_i)/2 gives:

```
J_{ij} = Q_sym[i,j] / 2          (i < j)
h_i    = Q_sym[i,i] / 2  +  Σ_{j≠i} Q_sym[i,j] / 2
offset = Σ_i Q_sym[i,i]/2  +  Σ_{i<j} Q_sym[i,j]/2
```

The conversion is exact to floating-point precision; roundtrip verification
(QUBO → Ising → QUBO) is performed automatically.

---

## Solver Comparison

| Solver | Algorithm | Works on | Notes |
|---|---|---|---|
| Dijkstra | Graph-theoretic | G directly | Composite edge weight; optimal for single-commodity |
| Greedy | Greedy search | G directly | Hop-by-hop; Dijkstra fallback |
| LoadBalanced | k-shortest paths | G directly | Minimizes max-congestion score |
| Simulated Annealing | QUBO minimizer | Q matrix | Linear β schedule; Dijkstra warm-start |
| Parallel Tempering | QUBO minimizer | Q matrix | Geometric β ladder; Metropolis replica swaps |
| QAOA (baseline) | Gate-model quantum | Ising H | Variational; COBYLA optimizer |
| QAOA + HW-aware | Gate-model quantum | Ising H | SABRE transpilation; FakeMontreal noise |
| QAOA + ML + HW | Gate-model quantum | Ising H | ML warm-start angles + HW transpilation |

---

## Metrics

### Solution Quality
- **Routing cost**: `w_l · Σ lat + w_c · Σ cong + w_p · Σ loss` over active edges
- **Feasibility rate**: fraction of trials producing flow-conserving paths
- **Approximation ratio**: `E_best_known / E_solver` (1.0 = optimal)

### Speed and Reliability
- **TTS99**: time-to-solution at 99% success probability, computed as:

  ```
  TTS_99 = t_run × log(1 − 0.99) / log(1 − p_success)
  ```

  where p_success is estimated empirically over n_trials independent runs and
  a Wilson score 95% confidence interval is reported.

- **Convergence history**: per-sweep best energy logged for all annealing solvers

### Quantum Circuit Statistics
- Logical depth, physical depth, CX gate count (logical vs transpiled)
- SWAP overhead introduced by hardware routing
- Readout and gate error rates from backend properties

---

## ML Optimizer

The ML module provides two concrete functions, neither decorative:

### 1. QAOA Angle Prediction (`AnglePredictor`)

Predicts warm-start (γ, β) initialization for QAOA from a 22-dimensional
feature vector of graph statistics (density, degree distribution, edge weight
statistics, QUBO sparsity, backend noise). Reduces the number of optimizer
function evaluations needed to reach a target energy.

**Training**: supervised regression on (features, optimal-angles) pairs
generated from solved instances. Architecture: MLP with BatchNorm and dropout.
CUDA-aware: automatically uses GPU if available.

### 2. QUBO Weight Prediction (`QUBOCoefficientPredictor`)

Predicts multiplicative adjustment factors for the six QUBO penalty and
objective weights `{w_l, w_c, w_p, p_flow, p_path, p_cap}` from graph and
traffic features. When traffic conditions change (new congestion pattern),
instead of recomputing the full QUBO, the predictor returns a weight
adjustment vector in ≪1 ms, and `apply_adjustments()` rescales the cached Q
matrix in O(n²).

This makes QUBO adaptation sub-linear in graph size for recurring traffic
patterns.

---

## Usage

### Installation

```bash
pip install -r requirements.txt
```

### Quick Demo (all solvers, classical only)

```bash
python run.py --quick --no-quantum
```

### Full Demo with QAOA

```bash
python run.py --nodes 10 --trials 10 --p 2 --shots 2048
```

### Train ML Models First

```bash
python run.py --train-ml --no-quantum
```

### Full Parameter Sweep

```bash
python run.py --sweep --sweep-sizes 8 12 16 --sweep-trials 5 --no-quantum
```

### Run Tests

```bash
python -m pytest tests/ -v
```

### Download Real Datasets

```bash
python data/download_datasets.py --dataset snap
python data/download_datasets.py --dataset rocketfuel
```

---

## Experiment Reproducibility

Every experiment saves:

- Random seed used
- Full config at time of run
- Per-trial energy values and runtimes
- Solver-specific metadata (β schedule, replica count, circuit stats)
- JSON export to `experiments/runs/sweep_<timestamp>.json`

Fixed-seed reruns are guaranteed identical by construction.

---

## Running Tests

```bash
python -m pytest tests/ -v --tb=short
```

The test suite covers:

| Test file | What it verifies |
|---|---|
| `test_ising.py` | QUBO↔Ising roundtrip; energy equivalence for all 2^n bitstrings (n≤6) |
| `test_qubo.py` | Q shape, symmetry, variable map completeness, feasible < random energy |
| `test_baselines.py` | All five solvers produce valid `SolverResult` on a known 5-node graph |
| `test_benchmark.py` | TTS99 formula correctness; BenchmarkMetrics aggregation |
| `test_graph_engine.py` | Loader reproducibility; edge attribute ranges; traffic update |

---

## Configuration

All hyperparameters are centralized in `config.yaml`. Key sections:

```yaml
qubo:
  penalty_path_validity: 10.0   # overridden by auto-scaling at runtime
  latency_weight: 1.0
  congestion_weight: 2.0
  loss_weight: 3.0

quantum:
  p_depth: 3
  shots: 4096
  backend: "aer_simulator"

ml:
  angle_predictor:
    hidden_dims: [128, 256, 128]
    device: "auto"              # auto → CUDA > MPS > CPU
```

---

## Dependencies

| Package | Role |
|---|---|
| `qiskit` ≥ 1.0 | Quantum circuit construction |
| `qiskit-aer` ≥ 0.14 | Simulation (statevector, shot-based, noisy) |
| `torch` ≥ 2.2 | ML models; CUDA-aware training |
| `networkx` ≥ 3.2 | Graph representation and algorithms |
| `scipy` | Classical optimization (COBYLA, L-BFGS-B) |
| `numpy` | QUBO/Ising numerical operations |
| `pandas` + `matplotlib` | Benchmark tables and plots |
| `rich` | Terminal output formatting |

---

## Results

Outputs are saved to `results/`:

| File | Contents |
|---|---|
| `benchmark_results.csv` | Full solver comparison table |
| `approximation_ratio.png` | Bar chart of approx. ratios vs best known |
| `tts99.png` | TTS99 comparison (log scale) |
| `energy_distribution.png` | Box plots of energy across trials |
| `convergence.png` | Per-solver energy convergence curves |

---

## Honest Assessment

This system is designed to expose failure modes alongside successes:

- **QAOA on large graphs** (> 20 qubits): circuit depth grows quickly;
  transpilation overhead and decoherence limit practical performance.
- **SA/PT with small penalty budgets**: infeasible solutions dominate the QUBO
  landscape; auto-scaled penalties are required for reliable feasibility.
- **ML warm-start**: meaningful only after sufficient training data (≥ 100
  similar instances); untrained predictors fall back to random initialization.
- **Hardware-aware transpilation**: reduces circuit infidelity but adds SWAP
  gates; the net effect depends on problem size and coupling map topology.

These trade-offs are quantified in the benchmark output, not hidden.

---

## License

MIT
