# Hardware-Aware Quantum ML Routing Engine (HQMRE)

HQMRE is a hybrid quantum-classical benchmark for telecom network routing. It takes a routing problem that becomes combinatorially hard at scale, reformulates it as a QUBO and Ising optimization problem, and compares classical solvers, annealing methods, and QAOA variants under the same latency, congestion, bandwidth, and packet-loss constraints.

The point of the repo is not to make vague claims about quantum advantage. The point is to build a serious benchmark that shows where quantum optimization helps, where it does not, and what hardware overhead it introduces.

## Why This Project Matters

Finding the best route on a small map is easy. Finding the best route across a large communication network is not. Once millions of possible paths compete under latency, congestion, and loss constraints, exact search becomes expensive enough that real systems fall back to heuristics.

HQMRE turns that systems bottleneck into a measurable routing benchmark:

- route selection is lifted into a QUBO
- the QUBO is converted into an Ising Hamiltonian
- QAOA is implemented in Qiskit
- hardware-aware transpilation makes circuit overhead visible
- ML warm-starts test whether repeated traffic patterns reduce optimizer effort

## Solvers Compared

| Solver | Role |
| --- | --- |
| Dijkstra | shortest-path baseline |
| Greedy | fast local routing baseline |
| Load-Balanced | congestion-aware graph baseline |
| Simulated Annealing | classical QUBO search |
| Parallel Tempering | multi-temperature QUBO search |
| QAOA | variational quantum baseline |
| QAOA + HW-aware | transpiled and noise-aware quantum run |
| QAOA + ML + HW | warm-started hardware-aware quantum run |

## Showcase and Demo

This repo is designed to be demoed locally from the CLI. No web stack is required.

Install dependencies:

```bash
pip install -r requirements.txt
```

Quick local demo:

```bash
python run.py --quick --no-quantum
```

QAOA demo:

```bash
python run.py --nodes 10 --trials 10 --p 2 --shots 2048
```

ML warm-start demo:

```bash
python run.py --train-ml
python run.py --nodes 12 --trials 15
```

For a presentation-ready walkthrough, talking points, and artifact order, see [SHOWCASE.md](SHOWCASE.md).

## What HQMRE Produces

Typical outputs include:

- `results/benchmark_results.csv`
- `results/approximation_ratio.png`
- `results/tts99.png`
- `results/energy_distribution.png`
- `results/convergence.png`
- `experiments/runs/sweep_<timestamp>.json`

These artifacts make it easy to present solution quality, runtime, and hardware cost side by side.

## Repository Layout

```text
quantum-/
|-- config.yaml
|-- run.py
|-- data/
|-- graph_engine/
|-- qubo_formulation/
|-- ising_utils/
|-- classical_baselines/
|-- quantum_module/
|-- hardware_mapping/
|-- ml_optimizer/
|-- orchestrator/
|-- benchmark/
|-- experiments/
|-- results/
`-- tests/
```

## Honest Framing

- QAOA does not automatically beat classical routing baselines.
- Exact or classical metaheuristic methods can still win on small and medium instances.
- Hardware-aware transpilation can erase nominal gains through depth and two-qubit overhead.
- ML warm-starts help only when enough similar instances exist.
- The value of HQMRE is that it makes those tradeoffs measurable.

## License

MIT
