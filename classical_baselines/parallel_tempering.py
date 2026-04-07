"""
Parallel Tempering (Replica Exchange) baseline solver operating on the QUBO matrix.

Multiple SA replicas run at different inverse temperatures; periodically
adjacent replicas attempt to swap configurations using the Metropolis criterion.
This helps escape local minima that trap single-temperature SA.
"""

from __future__ import annotations
from typing import Any
import time

import numpy as np
import networkx as nx

from .solver_base import SolverResult
from .routing_utils import (
    bitstring_to_routing_cost,
    check_flow_conservation,
    compute_qubo_energy,
)
from .simulated_annealing import SimulatedAnnealingSolver


class ParallelTemperingSolver:
    """
    Parallel tempering (replica-exchange) QUBO minimiser.

    *num_replicas* replicas are initialised with independent random
    bitstrings.  Their inverse temperatures are spaced geometrically between
    *beta_min* (hot, high-entropy replica) and *beta_max* (cold, low-energy
    replica).

    Every *swap_interval* sweeps, adjacent replicas (i, i+1) propose a swap
    using the standard Metropolis exchange criterion::

        accept = min(1, exp((beta_{i+1} - beta_i) * (E_i - E_{i+1})))

    The global best solution and energy across all replicas and all sweeps
    is returned.  ``energy_history`` records the global best energy after
    each sweep.

    Parameters may be supplied at construction time and overridden per call.

    Parameters
    ----------
    num_replicas : int
    num_sweeps : int
    beta_min, beta_max : float
    swap_interval : int
    seed : int
    w_latency, w_congestion, w_loss : float
    """

    def __init__(
        self,
        num_replicas: int = 8,
        num_sweeps: int = 2000,
        beta_min: float = 0.1,
        beta_max: float = 10.0,
        swap_interval: int = 50,
        seed: int = 42,
        w_latency: float = 1.0,
        w_congestion: float = 2.0,
        w_loss: float = 3.0,
    ) -> None:
        self.num_replicas = num_replicas
        self.num_sweeps = num_sweeps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.swap_interval = swap_interval
        self.seed = seed
        self.w_latency = w_latency
        self.w_congestion = w_congestion
        self.w_loss = w_loss

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _sa_sweep(
        Q: np.ndarray,
        Qsym: np.ndarray,
        x: np.ndarray,
        energy: float,
        beta: float,
        order: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        """
        Perform one SA sweep (iterate all variables once) in-place on *x*.

        Parameters
        ----------
        Q, Qsym : np.ndarray
            QUBO matrix and its symmetric counterpart (Q + Q^T).
        x : np.ndarray, shape (n,), dtype float64
            Current bitstring — **mutated in-place**.
        energy : float
            Current QUBO energy.
        beta : float
            Inverse temperature for this sweep.
        order : np.ndarray
            Variable visitation order (permuted outside this function).
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        x : np.ndarray  (same object, modified in-place)
        energy : float
        """
        for k in order:
            row_dot = float(Qsym[k, :] @ x) - float(Q[k, k]) * float(x[k])
            delta_e = (1.0 - 2.0 * float(x[k])) * row_dot

            if delta_e <= 0.0:
                x[k] = 1.0 - x[k]
                energy += delta_e
            else:
                if rng.random() < np.exp(-beta * delta_e):
                    x[k] = 1.0 - x[k]
                    energy += delta_e

        return x, energy

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    def solve(
        self,
        Q: np.ndarray,
        variable_map: dict,
        G: nx.Graph,
        src: Any,
        dst: Any,
        num_replicas: int = None,
        num_sweeps: int = None,
        beta_min: float = None,
        beta_max: float = None,
        swap_interval: int = None,
        seed: int = None,
        w_latency: float = None,
        w_congestion: float = None,
        w_loss: float = None,
    ) -> SolverResult:
        """
        Run parallel tempering on the QUBO and return a SolverResult.

        Parameters
        ----------
        Q : np.ndarray, shape (n, n)
            QUBO matrix.
        variable_map : dict
            Mapping ``edge_tuple -> variable_index``.
        G : networkx.Graph
            Network graph (used to decode routing metrics).
        src, dst
            Source and destination nodes.
        num_replicas : int
            Number of parallel replicas (temperature rungs).
        num_sweeps : int
            Total number of sweeps for each replica.
        beta_min, beta_max : float
            Geometric range of inverse temperatures.
        swap_interval : int
            Sweeps between replica-exchange attempts.
        seed : int
            Master random seed.
        w_latency, w_congestion, w_loss : float
            Weights for decoding routing cost.

        Returns
        -------
        SolverResult
        """
        # Resolve per-call overrides vs instance defaults
        num_replicas = num_replicas if num_replicas is not None else self.num_replicas
        num_sweeps   = num_sweeps   if num_sweeps   is not None else self.num_sweeps
        beta_min     = beta_min     if beta_min     is not None else self.beta_min
        beta_max     = beta_max     if beta_max     is not None else self.beta_max
        swap_interval = swap_interval if swap_interval is not None else self.swap_interval
        seed         = seed         if seed         is not None else self.seed
        w_latency    = w_latency    if w_latency    is not None else self.w_latency
        w_congestion = w_congestion if w_congestion is not None else self.w_congestion
        w_loss       = w_loss       if w_loss       is not None else self.w_loss

        t_start = time.perf_counter()

        n = Q.shape[0]
        rng = np.random.default_rng(seed)

        # Geometric beta schedule — index 0 is hottest (beta_min), last is coldest
        if num_replicas == 1:
            betas = np.array([beta_max])
        else:
            betas = np.geomspace(beta_min, beta_max, num_replicas)

        Qsym = Q + Q.T  # pre-compute symmetric form for fast ΔE

        # Initialise replicas with random bitstrings; seed the coldest replica
        # (index -1) with a Dijkstra warm-start for guaranteed feasibility.
        replicas = [
            rng.integers(0, 2, size=n).astype(np.float64)
            for _ in range(num_replicas)
        ]
        dijkstra_x0 = SimulatedAnnealingSolver._dijkstra_warmstart(
            G, src, dst, variable_map, n, w_latency, w_congestion, w_loss
        )
        if dijkstra_x0 is not None:
            replicas[-1] = dijkstra_x0   # coldest replica = best-known feasible
        energies = [float(x @ Q @ x) for x in replicas]

        global_best_energy = min(energies)
        global_best_x = replicas[np.argmin(energies)].copy()
        energy_history: list[float] = []

        # Seed feasible tracking from Dijkstra warm-start
        if dijkstra_x0 is not None:
            best_feasible_energy = float(dijkstra_x0 @ Q @ dijkstra_x0)
            best_feasible_x = dijkstra_x0.copy()
        else:
            best_feasible_energy = np.inf
            best_feasible_x = None

        n_swap_accepts = 0
        n_swap_attempts = 0

        for sweep_idx in range(num_sweeps):
            # ---------------------------------------------------------------- #
            # SA sweep for every replica
            # ---------------------------------------------------------------- #
            for r in range(num_replicas):
                order = rng.permutation(n)
                replicas[r], energies[r] = self._sa_sweep(
                    Q, Qsym, replicas[r], energies[r], betas[r], order, rng
                )
                if energies[r] < global_best_energy:
                    global_best_energy = energies[r]
                    global_best_x = replicas[r].copy()
                # Track best feasible replica.
                # Recompute true QUBO energy (incremental accumulators drift over
                # thousands of sweeps due to floating-point cancellation errors).
                bs_r = replicas[r].astype(np.int8)
                if check_flow_conservation(bs_r, variable_map, G, src, dst):
                    true_e = float(replicas[r] @ Q @ replicas[r])
                    if true_e < best_feasible_energy:
                        best_feasible_energy = true_e
                        best_feasible_x = replicas[r].copy()

            energy_history.append(global_best_energy)

            # ---------------------------------------------------------------- #
            # Replica swap attempts every swap_interval sweeps
            # ---------------------------------------------------------------- #
            if (sweep_idx + 1) % swap_interval == 0:
                # Attempt swaps between all adjacent pairs (random order)
                pair_order = rng.permutation(num_replicas - 1)
                for i in pair_order:
                    j = i + 1
                    # Metropolis criterion:
                    # accept = exp((beta_j - beta_i) * (E_i - E_j))
                    delta = (betas[j] - betas[i]) * (energies[i] - energies[j])
                    n_swap_attempts += 1
                    if delta >= 0.0 or rng.random() < np.exp(delta):
                        # Swap configurations
                        replicas[i], replicas[j] = replicas[j], replicas[i]
                        energies[i], energies[j] = energies[j], energies[i]
                        n_swap_accepts += 1

        # ------------------------------------------------------------------ #
        # Decode routing metrics — prefer best feasible solution.
        # ------------------------------------------------------------------ #
        report_x = best_feasible_x if best_feasible_x is not None else global_best_x
        report_energy = (best_feasible_energy if best_feasible_x is not None
                         else global_best_energy)
        best_bitstring = report_x.astype(np.int8)

        routing_cost, latency, congestion_score, loss_score = bitstring_to_routing_cost(
            best_bitstring, variable_map, G, w_latency, w_congestion, w_loss
        )

        feasible = check_flow_conservation(best_bitstring, variable_map, G, src, dst)

        runtime = time.perf_counter() - t_start

        swap_acceptance_rate = (
            n_swap_accepts / n_swap_attempts if n_swap_attempts > 0 else 0.0
        )

        return SolverResult(
            solver_name="parallel_tempering",
            best_energy=report_energy,
            best_bitstring=best_bitstring,
            routing_cost=routing_cost,
            latency=latency,
            congestion_score=congestion_score,
            loss_score=loss_score,
            runtime_seconds=runtime,
            n_iterations=num_sweeps * num_replicas,
            energy_history=energy_history,
            feasible=feasible,
            metadata={
                "num_replicas": num_replicas,
                "num_sweeps": num_sweeps,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "betas": betas.tolist(),
                "swap_interval": swap_interval,
                "swap_acceptance_rate": swap_acceptance_rate,
                "n_swap_attempts": n_swap_attempts,
                "n_swap_accepts": n_swap_accepts,
                "seed": seed,
                "n_vars": n,
            },
        )
