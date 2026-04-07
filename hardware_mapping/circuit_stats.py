"""
Circuit statistics logger for benchmark integration.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class CircuitStats:
    n_qubits: int
    depth_logical: int
    depth_physical: int
    n_cx_logical: int
    n_cx_physical: int
    n_gates_logical: int
    n_gates_physical: int
    swap_overhead: int
    backend_name: str
    optimization_level: int
    gate_counts_logical: dict = field(default_factory=dict)
    gate_counts_physical: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "depth_logical": self.depth_logical,
            "depth_physical": self.depth_physical,
            "n_cx_logical": self.n_cx_logical,
            "n_cx_physical": self.n_cx_physical,
            "n_gates_logical": self.n_gates_logical,
            "n_gates_physical": self.n_gates_physical,
            "swap_overhead": self.swap_overhead,
            "backend_name": self.backend_name,
            "optimization_level": self.optimization_level,
        }

    @classmethod
    def from_transpilation_result(cls, tr, n_qubits: int) -> "CircuitStats":
        from .transpiler import TranspilationResult
        return cls(
            n_qubits=n_qubits,
            depth_logical=tr.logical_depth,
            depth_physical=tr.physical_depth,
            n_cx_logical=tr.logical_cx_count,
            n_cx_physical=tr.physical_cx_count,
            n_gates_logical=tr.logical_gate_count,
            n_gates_physical=tr.physical_gate_count,
            swap_overhead=tr.swap_overhead,
            backend_name=tr.backend_name,
            optimization_level=tr.optimization_level,
        )
