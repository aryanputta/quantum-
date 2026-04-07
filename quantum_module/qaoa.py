"""
QAOA Circuit Builder.
Constructs parameterized QAOA circuits from Ising Hamiltonians.
Supports variable depth p and multiple backend targets.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class QAOACircuitBuilder:
    """
    Build QAOA circuits from an Ising Hamiltonian.

    Parameters
    ----------
    p : int
        Number of QAOA layers (depth).
    """

    def __init__(self, p: int = 1):
        self.p = p

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_circuit(
        self,
        ising_J: dict,
        ising_h: dict,
        n_qubits: int,
        gamma: Optional[list] = None,
        beta: Optional[list] = None,
    ):
        """
        Build a parameterized QAOA circuit.

        Parameters
        ----------
        ising_J : dict  {(i,j): coupling}
        ising_h : dict  {i: local_field}
        n_qubits : int
        gamma    : list of p floats (cost layer angles) — uses symbolic params if None
        beta     : list of p floats (mixer angles) — uses symbolic params if None

        Returns
        -------
        QuantumCircuit (Qiskit)
        """
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit import ParameterVector
        except ImportError as exc:
            raise ImportError("Qiskit required: pip install qiskit") from exc

        use_params = gamma is None or beta is None

        if use_params:
            gamma_params = ParameterVector("γ", self.p)
            beta_params = ParameterVector("β", self.p)
        else:
            gamma_params = list(gamma)
            beta_params = list(beta)

        qc = QuantumCircuit(n_qubits)

        # ── Initial state: uniform superposition ──────────────────────
        qc.h(range(n_qubits))

        # ── QAOA layers ───────────────────────────────────────────────
        for layer in range(self.p):
            g = gamma_params[layer]
            b = beta_params[layer]

            # Cost unitary U_C(γ) = exp(-i γ H_C)
            self._apply_cost_unitary(qc, ising_J, ising_h, g, n_qubits)

            # Mixer unitary U_B(β) = exp(-i β H_B) with H_B = sum X_i
            self._apply_mixer_unitary(qc, b, n_qubits)

        # Measure all
        qc.measure_all()

        if use_params:
            self._gamma_params = gamma_params
            self._beta_params = beta_params

        return qc

    def bind_parameters(self, qc, gamma: list, beta: list):
        """Bind numeric values to symbolic circuit parameters."""
        param_dict = {}
        for i, g in enumerate(gamma):
            param_dict[self._gamma_params[i]] = g
        for i, b in enumerate(beta):
            param_dict[self._beta_params[i]] = b
        return qc.assign_parameters(param_dict)

    def get_parameter_names(self):
        """Return flat list of parameter names: [γ_0,...,γ_{p-1}, β_0,...,β_{p-1}]."""
        return (
            [f"gamma_{i}" for i in range(self.p)]
            + [f"beta_{i}" for i in range(self.p)]
        )

    def circuit_stats(self, qc) -> dict:
        """Return depth, gate counts, two-qubit gate count."""
        from qiskit import transpile
        ops = qc.count_ops()
        return {
            "depth": qc.depth(),
            "n_gates": sum(ops.values()),
            "two_qubit_gates": ops.get("cx", 0) + ops.get("cz", 0) + ops.get("rzz", 0),
            "gate_counts": dict(ops),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _apply_cost_unitary(self, qc, J: dict, h: dict, gamma, n_qubits: int):
        """Apply cost Hamiltonian unitary: exp(-i gamma H_C)."""
        # ZZ interaction terms
        for (i, j), coupling in J.items():
            if i < j and i < n_qubits and j < n_qubits:
                angle = 2.0 * coupling
                if hasattr(angle, "__mul__"):  # Parameter
                    angle = 2.0 * coupling * gamma
                else:
                    angle = 2.0 * coupling * gamma
                qc.cx(i, j)
                qc.rz(angle, j)
                qc.cx(i, j)

        # Z single-qubit terms
        for i, field in h.items():
            if i < n_qubits:
                if hasattr(gamma, "__mul__"):
                    angle = 2.0 * field * gamma
                else:
                    angle = 2.0 * field * gamma
                qc.rz(angle, i)

    def _apply_mixer_unitary(self, qc, beta, n_qubits: int):
        """Apply transverse-field mixer: exp(-i beta sum_i X_i)."""
        for i in range(n_qubits):
            if hasattr(beta, "__mul__"):
                angle = 2.0 * beta
            else:
                angle = 2.0 * beta
            qc.rx(angle, i)
