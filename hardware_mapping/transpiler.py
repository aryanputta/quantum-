"""
Hardware-Aware Transpiler.
Transpiles QAOA circuits to target backend with optimization.
Records logical vs physical circuit statistics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class TranspilationResult:
    logical_depth: int
    physical_depth: int
    logical_cx_count: int
    physical_cx_count: int
    logical_gate_count: int
    physical_gate_count: int
    swap_overhead: int          # added SWAP gates
    qubit_layout: dict          # logical → physical qubit mapping
    transpiled_circuit: object  # QuantumCircuit
    backend_name: str
    optimization_level: int


class HardwareAwareTranspiler:
    """
    Transpile circuits with backend-specific optimizations.
    """

    def __init__(
        self,
        optimization_level: int = 3,
        routing_method: str = "sabre",
        layout_method: str = "sabre",
    ):
        self.optimization_level = optimization_level
        self.routing_method = routing_method
        self.layout_method = layout_method

    def transpile(self, qc, backend, backend_props=None) -> TranspilationResult:
        """
        Transpile a logical circuit to target backend.

        Parameters
        ----------
        qc           : QuantumCircuit (logical)
        backend      : Qiskit backend
        backend_props: BackendProperties (for logging, optional)

        Returns
        -------
        TranspilationResult
        """
        try:
            from qiskit import transpile

            logical_stats = self._circuit_stats(qc)

            tc = transpile(
                qc,
                backend=backend,
                optimization_level=self.optimization_level,
                routing_method=self.routing_method,
                layout_method=self.layout_method,
            )

            physical_stats = self._circuit_stats(tc)

            # Extract qubit layout
            layout = {}
            try:
                if hasattr(tc, "layout") and tc.layout and tc.layout.final_layout:
                    layout = {v: p for v, p in tc.layout.final_layout.get_virtual_bits().items()}
                elif hasattr(tc, "_layout") and tc._layout:
                    layout = {}
            except Exception:
                layout = {}

            logical_cx = logical_stats.get("cx", 0)
            physical_cx = physical_stats.get("cx", 0)
            swap_overhead = max(0, physical_stats.get("swap", 0))

            bname = str(backend.name) if hasattr(backend, "name") else str(backend)

            return TranspilationResult(
                logical_depth=logical_stats["depth"],
                physical_depth=physical_stats["depth"],
                logical_cx_count=logical_cx,
                physical_cx_count=physical_cx,
                logical_gate_count=logical_stats["n_gates"],
                physical_gate_count=physical_stats["n_gates"],
                swap_overhead=swap_overhead,
                qubit_layout=layout,
                transpiled_circuit=tc,
                backend_name=bname,
                optimization_level=self.optimization_level,
            )
        except Exception as exc:
            logger.warning("Transpilation failed: %s — returning passthrough", exc)
            logical_stats = self._circuit_stats(qc)
            return TranspilationResult(
                logical_depth=logical_stats["depth"],
                physical_depth=logical_stats["depth"],
                logical_cx_count=logical_stats.get("cx", 0),
                physical_cx_count=logical_stats.get("cx", 0),
                logical_gate_count=logical_stats["n_gates"],
                physical_gate_count=logical_stats["n_gates"],
                swap_overhead=0,
                qubit_layout={},
                transpiled_circuit=qc,
                backend_name="fallback",
                optimization_level=0,
            )

    def compare_stats(self, result: TranspilationResult) -> dict:
        """Return a comparison dict of logical vs physical circuit stats."""
        return {
            "depth_increase": result.physical_depth - result.logical_depth,
            "depth_ratio": (result.physical_depth / result.logical_depth
                            if result.logical_depth > 0 else 1.0),
            "cx_increase": result.physical_cx_count - result.logical_cx_count,
            "gate_increase": result.physical_gate_count - result.logical_gate_count,
            "swap_gates_added": result.swap_overhead,
            "optimization_level": result.optimization_level,
        }

    # ------------------------------------------------------------------

    def _circuit_stats(self, qc) -> dict:
        ops = qc.count_ops()
        return {
            "depth": qc.depth(),
            "n_gates": sum(ops.values()),
            "cx": ops.get("cx", 0),
            "swap": ops.get("swap", 0),
            "ops": dict(ops),
        }
