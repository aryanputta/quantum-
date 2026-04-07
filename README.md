# Hardware-Aware Quantum ML Routing Engine (HQMRE)

A hybrid quantum–classical benchmark system for telecom network routing
optimization via explicit QUBO/Ising reformulation, hardware-aware QAOA,
and ML-assisted coefficient adaptation.

---

## Problem Formulation

Given a directed graph $G = (V, E)$ where each edge $e \in E$ carries
attributes $\{\text{lat}_e,\, \text{bw}_e,\, \text{cong}_e,\, \text{loss}_e\}$
and a routing demand $(s, t)$, find a path $P \subseteq E$ minimizing:

$$\min_{P} \;\sum_{e \in P} \bigl[\, w_l \cdot \text{lat}_e \;+\; w_c \cdot \text{cong}_e \;+\; w_p \cdot \text{loss}_e \,\bigr]$$

subject to flow conservation at every node.

---

## QUBO Formulation

Binary variables $x_e \in \{0, 1\}$ are assigned to each directed edge.
The routing problem is cast as a **Quadratic Unconstrained Binary
Optimization**:

$$\min_{x \in \{0,1\}^n} \; x^\top Q x$$

where $Q \in \mathbb{R}^{n \times n}$ encodes both the objective and all
constraints. The constraint penalty expands the squared flow-conservation
residual at each node $v$ with demand $d_v \in \{-1, 0, +1\}$:

$$Q \;\leftarrow\; Q \;+\; \lambda \sum_{v \in V}
\!\left(\sum_{(v,w)\in E} x_{vw} \;-\; \sum_{(u,v)\in E} x_{uv} \;-\; d_v\right)^{\!2}$$

Because $x_i \in \{0,1\}$, the identity $x_i^2 = x_i$ collapses squared
linear terms onto the diagonal. The penalty is auto-scaled to satisfy:

$$\lambda \;\geq\; 5 \times \max_{e \in E}\bigl(w_l \cdot \text{lat}_e + w_c \cdot \text{cong}_e + w_p \cdot \text{loss}_e\bigr) \times |V|$$

guaranteeing that no infeasible solution achieves lower energy than any
feasible one.

Solved via `scipy.optimize` (COBYLA / SLSQP) with:

- **Flow conservation equality**: $\sum_e a_{ve}\, x_e = d_v$ for each node $v$
- **Non-negativity bounds**: $x_e \geq 0$
- **Analytical gradient**: $\partial(x^\top Q x)/\partial x_i = (Q + Q^\top)_i \cdot x$

---

## QUBO ↔ Ising Conversion

Standard substitution $x_i = (1 + s_i)/2$ with $s_i \in \{-1, +1\}$
gives the Ising Hamiltonian $H = \sum_{i<j} J_{ij}\,s_i s_j + \sum_i h_i\, s_i + c$
with parameters:

$$J_{ij} = \frac{Q_{\text{sym}}[i,j]}{2}, \qquad
h_i = \frac{Q_{\text{sym}}[i,i]}{2} + \sum_{j \neq i} \frac{Q_{\text{sym}}[i,j]}{2}$$

$$c = \sum_i \frac{Q_{\text{sym}}[i,i]}{2} + \sum_{i < j} \frac{Q_{\text{sym}}[i,j]}{2}$$

where $Q_{\text{sym}} = (Q + Q^\top)/2$. The roundtrip QUBO $\to$ Ising
$\to$ QUBO is verified to floating-point precision for all instances.

---

## Phase Diagrams

Binary and ternary routing-phase maps are built by:

1. Sampling $G_{\min}$ across the composition simplex
2. Computing the lower convex hull of the $(x,\, G_{\min})$ surface
3. Identifying miscibility gaps as regions where $G_{\min}$ lies above the hull

Ternary diagrams use barycentric grid sampling and a 3D convex-hull projection.

---

## Solvers

| Solver | Domain | Algorithm |
|---|---|---|
| **Dijkstra** | $G$ directly | Composite-weight shortest path |
| **Greedy** | $G$ directly | Hop-by-hop; Dijkstra fallback |
| **Load-Balanced** | $G$ directly | Top-$k$ paths; min max-congestion |
| **Simulated Annealing** | $Q$ matrix | Linear $\beta$ schedule; Dijkstra warm-start |
| **Parallel Tempering** | $Q$ matrix | Geometric $\beta$ ladder; Metropolis replica swaps |
| **QAOA (baseline)** | Ising $H$ | Variational; COBYLA optimizer |
| **QAOA + HW-aware** | Ising $H$ | SABRE transpilation; FakeMontreal noise |
| **QAOA + ML + HW** | Ising $H$ | ML warm-start $(\gamma, \beta)$ + HW transpilation |

**Simulated Annealing** runs $R$ independent chains each for $S$ sweeps with
a linear inverse-temperature schedule $\beta_k = \beta_{\min} + k\,\Delta\beta$.
At each sweep, variable $i$ is flipped with Metropolis probability:

$$P(\text{flip}\, i) = \min\!\left(1,\; e^{-\beta_k \,\Delta E_i}\right), \qquad
\Delta E_i = (1 - 2x_i)\bigl[(Q + Q^\top)_i \cdot x - Q_{ii}\,x_i\bigr]$$

**Parallel Tempering** runs $M$ replicas at geometrically-spaced temperatures.
Every $\tau$ sweeps, adjacent replicas $(i, i+1)$ attempt a configuration
swap via the exchange criterion:

$$P(\text{swap}) = \min\!\left(1,\; e^{(\beta_{i+1} - \beta_i)(E_i - E_{i+1})}\right)$$

---

## QAOA

The parameterized quantum circuit applies $p$ alternating cost and mixer
unitaries to an initial uniform superposition:

$$|\psi(\gamma, \beta)\rangle = \prod_{l=1}^{p} e^{-i\beta_l H_B}\, e^{-i\gamma_l H_C} \,|{+}\rangle^{\otimes n}$$

where $H_C = \sum_{i<j} J_{ij}\, Z_i Z_j + \sum_i h_i\, Z_i$ is the Ising
cost Hamiltonian and $H_B = \sum_i X_i$ is the transverse-field mixer.
The variational objective minimizes the expectation value:

$$(\gamma^*, \beta^*) = \arg\min_{\gamma,\beta} \langle\psi(\gamma,\beta)|\,H_C\,|\psi(\gamma,\beta)\rangle$$

---

## Hardware Mapping

Circuits are transpiled to target backends using the SABRE layout and
routing methods at optimization level 3. For each experiment the following
statistics are recorded:

| Quantity | Symbol | Logged |
|---|---|---|
| Logical circuit depth | $d_L$ | yes |
| Physical circuit depth | $d_P$ | yes |
| Logical CX count | $\text{CX}_L$ | yes |
| Physical CX count | $\text{CX}_P$ | yes |
| SWAP overhead | $\Delta_{\text{SWAP}}$ | yes |
| Avg gate error | $\bar\epsilon_g$ | yes |
| Avg readout error | $\bar\epsilon_r$ | yes |

---

## Machine Learning Layer

### QAOA Angle Prediction

Fits initial parameters $(\gamma_0, \beta_0)$ from observed
$(G,\, Q,\, \text{backend})$ data. The predictor maps a 22-dimensional
feature vector $\phi$ to warm-start angles:

$$\hat\gamma,\,\hat\beta = f_\theta(\phi(G, Q, \text{noise}))$$

$f_\theta$ is a feedforward network trained with MSE loss and
`ReduceLROnPlateau` scheduling. Reduces optimizer function evaluations
needed to reach a target energy. CUDA-aware: auto-selects GPU $>$ MPS $>$ CPU.

### QUBO Coefficient Prediction

Predicts multiplicative adjustment factors
$\alpha \in \mathbb{R}^6$ for the six QUBO weights
$\{w_l,\, w_c,\, w_p,\, \lambda_{\text{flow}},\, \lambda_{\text{path}},\, \lambda_{\text{cap}}\}$
from graph and traffic features:

$$Q_{\text{adapted}} = \text{diag}(\alpha_{\text{obj}})\, Q_{\text{diag}} \;+\; \alpha_{\text{penalty}}\, Q_{\text{off-diag}}$$

When traffic conditions change, the predictor returns $\alpha$ in $\ll 1\,\text{ms}$
and rescaling is $O(n^2)$ — avoiding full QUBO recomputation for recurring
traffic patterns.

### Feature Engineering

Beyond raw $(T, x_i)$ inputs, the pipeline computes:

| Feature | Formula |
|---|---|
| Graph size terms | $\|V\|/V_{\max}$, $\|E\|/V_{\max}^2$, density |
| Degree statistics | $\bar{d}/V_{\max}$, $d_{\max}/V_{\max}$ |
| Latency statistics | $\bar{\text{lat}}/100$, $\sigma_{\text{lat}}/100$, $\text{lat}_{\max}/100$ |
| Congestion statistics | $\bar{c}$, $c_{\max}$, $\sigma_c$ |
| QUBO sparsity | $\text{nnz}(Q)/n^2$ |
| QUBO coupling range | $\max|Q_{ij}| / (\max|Q_{ii}| + \varepsilon)$ |
| Traffic intensity | $\bar{T}$ (mean demand) |
| Endpoint degrees | $d^+(s)/V_{\max}$, $d^-(t)/V_{\max}$ |

---

## Metrics

### Solution Quality

- **Routing cost**: $C(P) = \sum_{e \in P} (w_l \cdot \text{lat}_e + w_c \cdot \text{cong}_e + w_p \cdot \text{loss}_e)$
- **Approximation ratio**: $\rho = E_{\text{best known}} / E_{\text{solver}}$
- **Feasibility rate**: fraction of trials producing flow-conserving paths

### Time-to-Solution at 99%

$$\text{TTS}_{99} = t_{\text{run}} \cdot \frac{\log(1 - 0.99)}{\log(1 - p_{\text{success}})}$$

where $p_{\text{success}}$ is the empirical fraction of runs achieving
energy $\leq E_{\text{target}}$, and $E_{\text{target}} = 0.95 \times E_{\text{best}}$
(i.e., within 5% of the best known solution). A Wilson score 95% confidence
interval is reported alongside each TTS estimate.

Applied consistently across SA, PT, and all QAOA variants.

---

## Repository Structure

```
quantum-/
├── config.yaml                  # All hyperparameters — single source of truth
├── run.py                       # Main entry point
├── data/
│   ├── download_datasets.py     # SNAP / Rocketfuel / CAIDA loaders
│   └── sample_topologies.py     # Embedded reference topologies
├── graph_engine/                # NetworkLoader · GraphPreprocessor · TrafficManager
├── qubo_formulation/            # QUBOBuilder · objectives · constraints
├── ising_utils/                 # IsingConverter · energy · validators
├── classical_baselines/         # Dijkstra · Greedy · LoadBalanced · SA · PT
├── quantum_module/              # QAOACircuitBuilder · QAOAOptimizer · QAOADecoder
├── hardware_mapping/            # BackendManager · HardwareAwareTranspiler · CircuitStats
├── ml_optimizer/                # AnglePredictor · QUBOCoefficientPredictor · MLTrainer
├── orchestrator/                # HQMREPipeline
├── benchmark/                   # BenchmarkMetrics · TTS99Calculator · BenchmarkReporter
├── experiments/                 # SweepConfig · ExperimentRunner
├── results/                     # Auto-populated: plots · tables · CSVs
└── tests/                       # 41 unit tests across 5 test files
```

---

## Usage

```bash
pip install -r requirements.txt

# Quick demo — all classical solvers, benchmark table, TTS99, plots
python run.py --quick --no-quantum

# Full demo with QAOA (p=2, 2048 shots)
python run.py --nodes 10 --trials 10 --p 2 --shots 2048

# Train ML models, then run with ML warm-start
python run.py --train-ml
python run.py --nodes 12 --trials 15

# Parameter sweep (graph size × traffic intensity × congestion penalty)
python run.py --sweep --sweep-sizes 8 12 16 --sweep-trials 5

# Run test suite
python -m pytest tests/ -v
```

---

## Outputs

All results are saved to `results/`:

| File | Contents |
|---|---|
| `benchmark_results.csv` | Full solver comparison table |
| `approximation_ratio.png` | $\rho$ per solver vs best known |
| `tts99.png` | TTS$_{99}$ comparison (log scale) |
| `energy_distribution.png` | Box plots of $E$ across trials |
| `convergence.png` | Per-solver energy convergence curves |

Experiment records (per-trial energies, runtimes, seeds, configs) are
exported to `experiments/runs/sweep_<timestamp>.json` for exact reruns.

---

## Honest Assessment

| Limitation | Detail |
|---|---|
| QAOA on $n > 20$ qubits | Circuit depth grows as $O(p \cdot \|E\|)$; decoherence limits reliability |
| SA/PT with small $\lambda$ | Infeasible solutions dominate; auto-scaling is mandatory |
| ML warm-start | Meaningful only after $\geq 100$ similar training instances |
| HW transpilation | Reduces infidelity but adds SWAP gates; net gain depends on coupling map |

These trade-offs are quantified in the benchmark output, not hidden.

---

## Dependencies

| Package | Version | Role |
|---|---|---|
| `qiskit` | ≥ 1.0 | Quantum circuit construction |
| `qiskit-aer` | ≥ 0.14 | Statevector / shot-based / noisy simulation |
| `torch` | ≥ 2.2 | ML models; CUDA-aware training |
| `networkx` | ≥ 3.2 | Graph representation and algorithms |
| `scipy` | ≥ 1.12 | Classical optimization (COBYLA, SLSQP) |
| `numpy` | ≥ 1.26 | QUBO/Ising numerical operations |
| `pandas` + `matplotlib` | — | Benchmark tables and plots |

---

## License

MIT
