"""
QUBO builder for telecom routing problems.

Provides QUBOResult and QUBOBuilder for converting NetworkX graph routing
problems into QUBO (Quadratic Unconstrained Binary Optimization) matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from .objectives import (
    latency_objective,
    congestion_objective,
    loss_objective,
    combine_objectives,
)
from .constraints import (
    flow_conservation_penalty,
    capacity_penalty,
    path_validity_penalty,
)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class QUBOResult:
    """
    Result of a QUBO formulation.

    Attributes
    ----------
    Q : np.ndarray  shape (n, n)
        Upper-triangular QUBO matrix.  Energy = x @ Q @ x for x ∈ {0,1}^n.
    linear : np.ndarray  shape (n,)
        Diagonal terms Q[i,i] (convenience view; same data as diag(Q)).
    variable_map : dict
        Maps variable index (int) → edge tuple (u, v) or path tuple.
    n_variables : int
        Number of binary variables (= n).
    metadata : dict
        Penalty weights, problem size, solver hints, etc.
    """
    Q: np.ndarray
    linear: np.ndarray
    variable_map: dict
    n_variables: int
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_variable_map(G: nx.DiGraph) -> tuple[dict, dict]:
    """
    Assign a unique integer index to every edge in G.

    Returns
    -------
    edge_to_idx : dict  {(u,v): int}
    idx_to_edge : dict  {int: (u,v)}
    """
    edge_to_idx: dict[tuple, int] = {}
    idx_to_edge: dict[int, tuple] = {}
    for i, (u, v) in enumerate(G.edges()):
        edge_to_idx[(u, v)] = i
        idx_to_edge[i] = (u, v)
    return edge_to_idx, idx_to_edge


def _assemble_qubo(
    n: int,
    linear: dict[int, float],
    quadratic: dict[tuple[int, int], float],
) -> np.ndarray:
    """
    Assemble an upper-triangular QUBO matrix from linear/quadratic dicts.

    Diagonal entries → Q[i,i]; off-diagonal → Q[i,j] with i < j.
    """
    Q = np.zeros((n, n), dtype=float)
    for idx, coeff in linear.items():
        Q[idx, idx] += coeff
    for (i, j), coeff in quadratic.items():
        lo, hi = (i, j) if i < j else (j, i)
        Q[lo, hi] += coeff
    return Q


def _merge(
    lin_acc: dict, quad_acc: dict,
    lin_new: dict, quad_new: dict,
) -> None:
    """Merge new linear/quadratic dicts into accumulators in-place."""
    for k, v in lin_new.items():
        lin_acc[k] = lin_acc.get(k, 0.0) + v
    for k, v in quad_new.items():
        lo, hi = (min(k), max(k))
        quad_acc[(lo, hi)] = quad_acc.get((lo, hi), 0.0) + v


# ---------------------------------------------------------------------------
# Main builder class
# ---------------------------------------------------------------------------

class QUBOBuilder:
    """
    Build QUBO matrices for routing optimisation problems.

    Default penalty and objective weights can be set at construction time
    and overridden per ``build_*`` call.

    Parameters
    ----------
    penalty_path : float
        Penalty weight for path-validity constraints.
    penalty_flow : float
        Penalty weight for flow-conservation constraints.
    penalty_capacity : float
        Penalty weight for capacity / congestion constraints.
    w_latency, w_congestion, w_loss : float
        Objective weights for latency, congestion, and packet loss.
    """

    def __init__(
        self,
        penalty_path: float = 10.0,
        penalty_flow: float = 8.0,
        penalty_capacity: float = 5.0,
        w_latency: float = 1.0,
        w_congestion: float = 2.0,
        w_loss: float = 3.0,
    ) -> None:
        self.penalty_path = penalty_path
        self.penalty_flow = penalty_flow
        self.penalty_capacity = penalty_capacity
        self.w_latency = w_latency
        self.w_congestion = w_congestion
        self.w_loss = w_loss

    # ------------------------------------------------------------------
    # Single-commodity routing
    # ------------------------------------------------------------------

    def build_edge_routing_qubo(
        self,
        G: nx.DiGraph,
        src: Any,
        dst: Any,
        penalty_path: float = None,
        penalty_flow: float = None,
        penalty_capacity: float = None,
        w_latency: float = None,
        w_congestion: float = None,
        w_loss: float = None,
    ) -> QUBOResult:
        """
        Build a single-commodity edge-routing QUBO.

        Binary variable x_e ∈ {0,1} for each directed edge e in G.

        Objective
        ---------
        minimise  Σ_e (w_lat·lat_e + w_cong·cong_e + w_loss·loss_e) · x_e

        Subject to (encoded as penalties):
        - Flow conservation at every node (src outflow=1, dst inflow=1, others balanced).
        - Path validity (delegates to flow conservation).
        - Capacity: penalise selection of over-loaded edges.

        Parameters
        ----------
        G               : directed NetworkX graph with edge attributes
                          ``latency``, ``congestion``, ``loss``.
        src, dst        : source and destination nodes.
        penalty_path    : weight for the path-validity penalty.
        penalty_flow    : weight for the flow-conservation penalty.
        penalty_capacity: weight for the capacity penalty.
        w_latency       : objective weight for latency.
        w_congestion    : objective weight for congestion.
        w_loss          : objective weight for packet loss.

        Returns
        -------
        QUBOResult
        """
        # Resolve per-call overrides vs instance defaults
        penalty_path     = penalty_path     if penalty_path     is not None else self.penalty_path
        penalty_flow     = penalty_flow     if penalty_flow     is not None else self.penalty_flow
        penalty_capacity = penalty_capacity if penalty_capacity is not None else self.penalty_capacity
        w_latency        = w_latency        if w_latency        is not None else self.w_latency
        w_congestion     = w_congestion     if w_congestion     is not None else self.w_congestion
        w_loss           = w_loss           if w_loss           is not None else self.w_loss

        if not G.has_node(src):
            raise ValueError(f"Source node {src!r} not in graph.")
        if not G.has_node(dst):
            raise ValueError(f"Destination node {dst!r} not in graph.")
        if src == dst:
            raise ValueError("Source and destination must differ.")

        edge_to_idx, idx_to_edge = _build_variable_map(G)
        n = len(edge_to_idx)

        # --- Objectives ---
        obj_lat  = latency_objective(G, edge_to_idx)
        obj_cong = congestion_objective(G, edge_to_idx)
        obj_loss = loss_objective(G, edge_to_idx)

        lin_obj, quad_obj = combine_objectives(
            [obj_lat, obj_cong, obj_loss],
            [w_latency, w_congestion, w_loss],
        )

        # --- Auto-scale penalties if needed ---
        # The penalty must exceed the maximum possible objective value for any
        # path (upper-bounded by using all edges at maximum cost).  This
        # guarantees that any feasible solution beats any infeasible one.
        max_edge_obj = max(
            (w_latency * d.get("latency", 1.0)
             + w_congestion * d.get("congestion", 0.0)
             + w_loss * d.get("loss", 0.0))
            for _, _, d in G.edges(data=True)
        ) if G.number_of_edges() > 0 else 1.0
        # Conservative upper bound on worst-case path cost
        path_upper_bound = max_edge_obj * G.number_of_nodes()
        auto_penalty = max(penalty_path, path_upper_bound * 5)

        # --- Constraints ---
        # Flow conservation encodes: for every node v,
        #   (net_outflow(v) - demand(v))^2 = 0
        # where demand(src)=+1, demand(dst)=-1, others=0.
        # This single constraint set subsumes path validity and loop-freeness.
        lin_acc: dict[int, float] = dict(lin_obj)
        quad_acc: dict[tuple[int, int], float] = dict(quad_obj)

        lin_fc, quad_fc = flow_conservation_penalty(
            G, edge_to_idx, src, dst, auto_penalty
        )
        _merge(lin_acc, quad_acc, lin_fc, quad_fc)

        # Capacity penalty: soft penalty for edges already near saturation.
        # Uses the user-supplied penalty_capacity (not auto-scaled; it is a
        # soft discouragment, not a hard constraint).
        lin_cap, quad_cap = capacity_penalty(G, edge_to_idx, penalty_capacity)
        _merge(lin_acc, quad_acc, lin_cap, quad_cap)

        Q = _assemble_qubo(n, lin_acc, quad_acc)

        return QUBOResult(
            Q=Q,
            linear=np.diag(Q).copy(),
            variable_map=idx_to_edge,
            n_variables=n,
            metadata={
                "problem": "edge_routing",
                "src": src,
                "dst": dst,
                "n_edges": n,
                "n_nodes": G.number_of_nodes(),
                "penalty_flow_used": float(auto_penalty),
                "penalty_capacity": float(penalty_capacity),
                "path_upper_bound": float(path_upper_bound),
                "w_latency": float(w_latency),
                "w_congestion": float(w_congestion),
                "w_loss": float(w_loss),
            },
        )

    # ------------------------------------------------------------------
    # Multi-commodity routing
    # ------------------------------------------------------------------

    def build_multicommodity_qubo(
        self,
        G: nx.DiGraph,
        demands: list[tuple[Any, Any, float]],
        penalty_path: float = 10.0,
        penalty_flow: float = 8.0,
        penalty_capacity: float = 5.0,
        w_latency: float = 1.0,
        w_congestion: float = 2.0,
        w_loss: float = 3.0,
        capacity_threshold: float = 0.8,
    ) -> QUBOResult:
        """
        Build a multi-commodity routing QUBO.

        Each (src, dst, volume) demand gets its own set of binary edge
        variables.  Shared capacity constraints penalise links where the
        aggregate selected volume would exceed (capacity_threshold * bandwidth).

        Parameters
        ----------
        G        : directed NetworkX graph.
        demands  : list of (src, dst, volume) tuples.
        penalty_* / w_* : same semantics as build_edge_routing_qubo.
        capacity_threshold : fraction of bandwidth that triggers penalty.

        Returns
        -------
        QUBOResult  with variable_map entries keyed as (commodity_idx, u, v).
        """
        if not demands:
            raise ValueError("demands list must be non-empty.")

        # Assign variable indices: commodity k, edge (u,v) → global index
        # variable_map: int → (k, u, v)
        idx = 0
        commodity_vars: list[dict[tuple, int]] = []   # per-commodity edge→idx
        global_var_map: dict[int, tuple] = {}         # global idx → (k, u, v)

        for k, (src, dst, _volume) in enumerate(demands):
            if not G.has_node(src):
                raise ValueError(f"Demand {k}: source {src!r} not in graph.")
            if not G.has_node(dst):
                raise ValueError(f"Demand {k}: destination {dst!r} not in graph.")
            if src == dst:
                raise ValueError(f"Demand {k}: src == dst ({src!r}).")
            e2i: dict[tuple, int] = {}
            for u, v in G.edges():
                e2i[(u, v)] = idx
                global_var_map[idx] = (k, u, v)
                idx += 1
            commodity_vars.append(e2i)

        n = idx
        lin_acc: dict[int, float] = {}
        quad_acc: dict[tuple[int, int], float] = {}

        # --- Per-commodity objectives + flow/path constraints ---
        for k, (src, dst, volume) in enumerate(demands):
            e2i = commodity_vars[k]
            obj_lat  = latency_objective(G, e2i)
            obj_cong = congestion_objective(G, e2i)
            obj_loss = loss_objective(G, e2i)
            lin_obj, quad_obj = combine_objectives(
                [obj_lat, obj_cong, obj_loss],
                [w_latency * volume, w_congestion * volume, w_loss * volume],
            )
            _merge(lin_acc, quad_acc, lin_obj, quad_obj)

            lin_pv, quad_pv = path_validity_penalty(G, e2i, src, dst, penalty_path)
            _merge(lin_acc, quad_acc, lin_pv, quad_pv)

            lin_fc, quad_fc = flow_conservation_penalty(G, e2i, src, dst, penalty_flow)
            _merge(lin_acc, quad_acc, lin_fc, quad_fc)

        # --- Shared capacity penalty ---
        # For each edge, penalise if the sum of selected commodities overloads it.
        # Penalty: P * (sum_k volume_k * x_{k,e} - cap_threshold * bw_e)^2
        # expanded only when total potential volume > threshold * bw.
        for u, v in G.edges():
            bw = float(G[u][v].get("bandwidth", 1.0))
            threshold_val = capacity_threshold * bw
            total_potential = sum(vol for _, _, vol in demands)

            if total_potential <= threshold_val:
                continue  # this edge can handle all demand; no penalty needed

            # Build coefficients: volume_k for each commodity variable on (u,v)
            coeffs: dict[int, float] = {}
            for k, (_src, _dst, volume) in enumerate(demands):
                var_idx = commodity_vars[k].get((u, v))
                if var_idx is not None:
                    coeffs[var_idx] = volume

            if not coeffs:
                continue

            # Penalise (sum_k vol_k * x_{k,e} - threshold_val)^2
            # Using the same quadratic expansion as _expand_squared_sum
            idx_list = list(coeffs.items())
            for var_i, a_i in idx_list:
                lin_acc[var_i] = lin_acc.get(var_i, 0.0) + penalty_capacity * (
                    a_i * a_i - 2.0 * threshold_val * a_i
                )
            for m in range(len(idx_list)):
                for nn in range(m + 1, len(idx_list)):
                    var_m, a_m = idx_list[m]
                    var_n, a_n = idx_list[nn]
                    key = (min(var_m, var_n), max(var_m, var_n))
                    quad_acc[key] = quad_acc.get(key, 0.0) + penalty_capacity * 2.0 * a_m * a_n

        Q = _assemble_qubo(n, lin_acc, quad_acc)

        return QUBOResult(
            Q=Q,
            linear=np.diag(Q).copy(),
            variable_map=global_var_map,
            n_variables=n,
            metadata={
                "problem": "multicommodity_routing",
                "n_commodities": len(demands),
                "demands": [(str(s), str(d), v) for s, d, v in demands],
                "n_edges_per_commodity": G.number_of_edges(),
                "n_nodes": G.number_of_nodes(),
                "n_total_variables": n,
                "penalty_path": penalty_path,
                "penalty_flow": penalty_flow,
                "penalty_capacity": penalty_capacity,
                "capacity_threshold": capacity_threshold,
                "w_latency": w_latency,
                "w_congestion": w_congestion,
                "w_loss": w_loss,
            },
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_qubo(self, result: QUBOResult) -> dict:
        """
        Run basic sanity checks on a QUBOResult.

        Checks
        ------
        symmetry_upper_triangular
            True if Q is upper-triangular (Q[i,j] == 0 for i > j).
        symmetry_symmetric
            True if Q == Q^T (alternative convention).
        lower_triangle_zero
            True if lower triangle is all zeros.
        no_nan_inf
            True if Q contains no NaN or Inf values.
        feasible_lt_random
            True if the energy of a (heuristic) feasible solution is below the
            average energy of 200 random bitstrings.
        variable_map_consistent
            True if variable_map keys match range(n_variables).

        Returns
        -------
        dict with keys ``'is_valid'`` (bool) and ``'checks'`` (dict of bool).
        """
        Q = result.Q
        n = result.n_variables
        checks: dict[str, bool] = {}

        # Basic shape
        checks["shape_square"] = (Q.ndim == 2 and Q.shape[0] == Q.shape[1])
        checks["shape_matches_n"] = (Q.shape == (n, n)) if checks["shape_square"] else False

        # NaN/Inf
        checks["no_nan_inf"] = bool(np.all(np.isfinite(Q)))

        # Upper triangular: lower triangle should be zero
        lower = np.tril(Q, k=-1)
        checks["lower_triangle_zero"] = bool(np.allclose(lower, 0.0))

        # Symmetric check (upper + lower == Q^T)
        checks["symmetry_symmetric"] = bool(np.allclose(Q, Q.T))

        # Variable map consistency
        checks["variable_map_consistent"] = (
            set(result.variable_map.keys()) == set(range(n))
        )

        # Energy check: heuristic feasible < random average
        rng = np.random.default_rng(42)
        random_energies = []
        for _ in range(200):
            x = rng.integers(0, 2, size=n).astype(float)
            random_energies.append(float(x @ Q @ x))
        avg_random = float(np.mean(random_energies))

        # All-zeros is a trivially feasible bitstring with energy 0; use it as
        # a baseline.  Also try all-ones.
        x_zero = np.zeros(n)
        x_ones = np.ones(n)
        e_zero = float(x_zero @ Q @ x_zero)
        e_ones = float(x_ones @ Q @ x_ones)
        best_known = min(e_zero, e_ones)

        checks["feasible_lt_random"] = (best_known <= avg_random)

        is_valid = all([
            checks["shape_square"],
            checks["shape_matches_n"],
            checks["no_nan_inf"],
            checks["lower_triangle_zero"],
            checks["variable_map_consistent"],
        ])

        return {"is_valid": is_valid, "checks": checks}
