# HQMRE Showcase Guide

## Project Ad Copy

HQMRE is a hybrid quantum-classical telecom routing benchmark that reformulates route selection as a QUBO and Ising optimization problem, runs QAOA in Qiskit, and measures solution quality, runtime, and hardware overhead against strong classical baselines.

It is built to answer a concrete systems question: when routing becomes combinatorially hard under latency, congestion, bandwidth, and packet-loss constraints, does quantum optimization provide useful structure, or do classical methods still dominate once real hardware costs are counted?

## 30-Second Explanation

Classical routing works well on small graphs, but large communication networks force heavy approximation because the number of valid paths explodes. HQMRE converts routing into a quantum optimization problem, then compares QAOA, annealing, and graph-based baselines on the same network instances. The result is not just a simulator run. It is a benchmark that shows where the quantum method helps, where it fails, and what it costs to execute.

## 5-Minute Live Demo

1. Install dependencies.

```bash
pip install -r requirements.txt
```

2. Run a quick classical benchmark to show the baseline pipeline.

```bash
python run.py --quick --no-quantum
```

3. Run a QAOA benchmark on a small routing instance.

```bash
python run.py --nodes 10 --trials 10 --p 2 --shots 2048
```

4. Show the generated artifacts in `results/` and explain the comparison.

5. If time permits, train the ML warm-start layer and rerun.

```bash
python run.py --train-ml
python run.py --nodes 12 --trials 15
```

## What To Show During the Demo

- `results/benchmark_results.csv` for the solver table
- `results/approximation_ratio.png` for solution quality
- `results/tts99.png` for time-to-solution
- `results/convergence.png` for optimizer behavior
- hardware statistics such as depth, CX count, and SWAP overhead

## Talking Points

- The routing problem is encoded as a QUBO with flow-conservation penalties.
- The QUBO is converted into an Ising Hamiltonian for QAOA.
- Qiskit is used to build and simulate the circuit.
- Hardware-aware transpilation matters because logical quality and physical quality are not the same thing.
- The repo is honest about failure cases. Quantum methods do not automatically win.

## Good One-Line Summary

Hardware-aware QAOA benchmark for telecom routing under realistic latency, congestion, and packet-loss constraints.
