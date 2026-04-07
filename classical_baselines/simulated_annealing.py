"""
Simulated Annealing baseline solver operating directly on the QUBO matrix.
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


class SimulatedAnnealingSolver:
    """
    Simulated annealing minimiser for QUBO problems.

    Runs *num_reads* independent SA chains each for *num_sweeps* sweeps with
    a linear inverse-temperature (beta) schedule.  At every sweep every
    variable is visited once in a random order; a bit-flip is accepted with
    the Metropolis criterion.

    The global best solution across all reads and sweeps is returned as the
    SolverResult.  ``energy_history`` records the global best energy after
    every sweep of the best-read chain (identified retrospectively).

    Parameters may be supplied at construction time (stored as instance
    defaults) and overridden per ``solve()`` call.

    Parameters
    ----------
    num_reads : int
        Number of independent SA restarts.
    num_sweeps : int
        Number of sweeps per restart.
    beta_range : (float, float)
        Linear schedule from beta_min to beta_max.
    seed : int
        Master random seed.
    w_latency, w_congestion, w_loss : float
        Objective weights used when decoding routing cost from the bitstring.
    """

    def __init__(
        self,
        num_reads: int = 100,
        num_sweeps: int = 2000,
        beta_range: tuple = (0.1, 10.0),
        seed: int = 42,
        w_latency: float = 1.0,
        w_congestion: float = 2.0,
        w_loss: float = 3.0,
    ) -> None:
        self.num_reads = num_reads
        self.num_sweeps = num_sweeps
        self.beta_range = beta_range
        self.seed = seed
        self.w_latency = w_latency
        self.w_congestion = w_congestion
        self.w_loss = w_loss

    # ---------------------------------------------------------------------- #
    # Internal helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _run_chain(
        Q: np.ndarray,
        x0: np.ndarray,
        betas: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[float, np.ndarray, list[float]]:
        """
        Run one SA chain.

        Parameters
        ----------
        Q : np.ndarray, shape (n, n)
            QUBO matrix.
        x0 : np.ndarray, shape (n,), dtype float64
            Initial bitstring.
        betas : np.ndarray, shape (num_sweeps,)
            Inverse temperature at each sweep.
        rng : np.random.Generator
            Seeded random number generator.

        Returns
        -------
        best_energy : float
        best_x : np.ndarray
        history : list[float]   -- best energy after each sweep
        """
        n = len(x0)
        x = x0.copy().astype(np.float64)

        # Pre-compute Q + Q^T to exploit the identity:
        #   ΔE when flipping bit k = (Q + Q^T)[k, :] @ x  *  (1 - 2*x[k])
        # For a symmetric or upper-triangular Q this still works correctly.
        Qsym = Q + Q.T  # shape (n, n)

        current_energy = float(x @ Q @ x)
        best_energy = current_energy
        best_x = x.copy()
        history: list[float] = []

        for beta in betas:
            # Random sweep order
            order = rng.permutation(n)
            for k in order:
                # ΔE for flipping variable k
                # E_new - E_old = (1 - 2*x[k]) * [Qsym[k,:] @ x - Q[k,k]*x[k]]
                # Simplified: delta_e = (1 - 2*x[k]) * (Qsym[k,:] @ x - Q[k,k]*x[k])
                row_dot = float(Qsym[k, :] @ x) - float(Q[k, k]) * float(x[k])
                delta_e = (1.0 - 2.0 * float(x[k])) * row_dot

                if delta_e <= 0.0:
                    x[k] = 1.0 - x[k]
                    current_energy += delta_e
                else:
                    if rng.random() < np.exp(-beta * delta_e):
                        x[k] = 1.0 - x[k]
                        current_energy += delta_e

            if current_energy < best_energy:
                best_energy = current_energy
                best_x = x.copy()

            history.append(best_energy)

        return best_energy, best_x, history

    # ---------------------------------------------------------------------- #
    # Public interface
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _dijkstra_warmstart(G, src, dst, variable_map, n_vars,
                            w_latency, w_congestion, w_loss):
        """Return a bitstring for the Dijkstra shortest path, or None."""
        try:
            import networkx as nx
            from .routing_utils import path_to_bitstring

            def composite_weight(u, v, d):
                return (w_latency * d.get("latency", 1.0)
                        + w_congestion * d.get("congestion", 0.0)
                        + w_loss * d.get("loss", 0.0))

            path = nx.shortest_path(G, src, dst, weight=composite_weight)
            return path_to_bitstring(path, variable_map, n_vars).astype(np.float64)
        except Exception:
            return None

    def solve(
        self,
        Q: np.ndarray,
        variable_map: dict,
        G: nx.Graph,
        src: Any,
        dst: Any,
        num_reads: int = None,
        num_sweeps: int = None,
        beta_range: tuple = None,
        seed: int = None,
        w_latency: float = None,
        w_congestion: float = None,
        w_loss: float = None,
    ) -> SolverResult:
        """
        Run simulated annealing on the QUBO and return a SolverResult.

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
        num_reads : int
            Number of independent SA runs.
        num_sweeps : int
            Number of sweeps per run.
        beta_range : (float, float)
            (beta_min, beta_max) — linear schedule from cold to hot inverse
            temperature.
        seed : int
            Master random seed.
        w_latency, w_congestion, w_loss : float
            Weights for decoding routing cost.

        Returns
        -------
        SolverResult
        """
        # Resolve per-call overrides vs instance defaults
        num_reads    = num_reads    if num_reads    is not None else self.num_reads
        num_sweeps   = num_sweeps   if num_sweeps   is not None else self.num_sweeps
        beta_range   = beta_range   if beta_range   is not None else self.beta_range
        seed         = seed         if seed         is not None else self.seed
        w_latency    = w_latency    if w_latency    is not None else self.w_latency
        w_congestion = w_congestion if w_congestion is not None else self.w_congestion
        w_loss       = w_loss       if w_loss       is not None else self.w_loss

        t_start = time.perf_counter()

        n = Q.shape[0]
        rng = np.random.default_rng(seed)

        beta_min, beta_max = beta_range
        betas = np.linspace(beta_min, beta_max, num_sweeps)

        global_best_energy = np.inf
        global_best_x = np.zeros(n, dtype=np.float64)
        best_history: list[float] = []

        # Track the best *feasible* solution separately.
        # QUBO energies for feasible solutions contain a negative constant
        # offset (missing penalty * b^2 term from the penalty expansion).
        # Raw QUBO minimisation can exploit infeasible configurations that
        # are even more negative.  Returning the best feasible solution
        # gives a physically meaningful result comparable across solvers.
        best_feasible_energy = np.inf
        best_feasible_x = None

        # Build a Dijkstra warm-start bitstring so the first read begins from
        # a known feasible solution.  This guarantees at least one feasible
        # candidate even if the SA chain subsequently drifts to infeasible states.
        dijkstra_x0 = self._dijkstra_warmstart(G, src, dst, variable_map, n,
                                                w_latency, w_congestion, w_loss)
        if dijkstra_x0 is not None:
            dijk_e = float(dijkstra_x0 @ Q @ dijkstra_x0)
            best_feasible_energy = dijk_e
            best_feasible_x = dijkstra_x0.copy()
            if dijk_e < global_best_energy:
                global_best_energy = dijk_e
                global_best_x = dijkstra_x0.copy()

        for read_idx in range(num_reads):
            # Read 0: warm-start from Dijkstra; subsequent reads: random
            if read_idx == 0 and dijkstra_x0 is not None:
                x0 = dijkstra_x0.astype(np.float64)
            else:
                x0 = rng.integers(0, 2, size=n).astype(np.float64)
            energy, x_final, history = self._run_chain(
                Q, x0, betas, rng
            )
            if energy < global_best_energy:
                global_best_energy = energy
                global_best_x = x_final.copy()
                best_history = history
            # Check feasibility for every read's final state.
            # Recompute true QUBO energy to avoid incremental drift.
            bs = x_final.astype(np.int8)
            if check_flow_conservation(bs, variable_map, G, src, dst):
                true_e = float(x_final @ Q @ x_final)
                if true_e < best_feasible_energy:
                    best_feasible_energy = true_e
                    best_feasible_x = x_final.copy()

        # ------------------------------------------------------------------ #
        # Decode routing metrics from best bitstring
        # ------------------------------------------------------------------ #
        # Prefer the best feasible solution found; fall back to global best.
        report_x = best_feasible_x if best_feasible_x is not None else global_best_x
        report_energy = (best_feasible_energy if best_feasible_x is not None
                         else global_best_energy)
        best_bitstring = report_x.astype(np.int8)

        routing_cost, latency, congestion_score, loss_score = bitstring_to_routing_cost(
            best_bitstring, variable_map, G, w_latency, w_congestion, w_loss
        )

        feasible = check_flow_conservation(best_bitstring, variable_map, G, src, dst)

        runtime = time.perf_counter() - t_start

        return SolverResult(
            solver_name="simulated_annealing",
            best_energy=report_energy,
            best_bitstring=best_bitstring,
            routing_cost=routing_cost,
            latency=latency,
            congestion_score=congestion_score,
            loss_score=loss_score,
            runtime_seconds=runtime,
            n_iterations=num_reads * num_sweeps,
            energy_history=best_history,
            feasible=feasible,
            metadata={
                "num_reads": num_reads,
                "num_sweeps": num_sweeps,
                "beta_min": beta_min,
                "beta_max": beta_max,
                "seed": seed,
                "n_vars": n,
            },
        )
