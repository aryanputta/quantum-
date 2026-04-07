"""
Backend Manager.
Query IBM Quantum / Aer backend properties for hardware-aware optimization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class BackendProperties:
    name: str
    n_qubits: int
    coupling_map: list          # list of [i, j] pairs
    basis_gates: list
    avg_gate_error: float       # average single-qubit gate error
    avg_cx_error: float         # average CX error
    avg_readout_error: float    # average measurement error
    t1_us: float                # avg T1 coherence (microseconds)
    t2_us: float                # avg T2 coherence
    is_simulator: bool
    raw_properties: dict = field(default_factory=dict)


class BackendManager:
    """
    Manage Qiskit backends and extract properties for hardware-aware transpilation.
    """

    def __init__(self):
        self._aer_backend = None
        self._ibm_backend = None

    def get_aer_simulator(self, noise_model=None):
        """Return an Aer simulator backend."""
        try:
            from qiskit_aer import AerSimulator
            if noise_model:
                return AerSimulator(noise_model=noise_model)
            return AerSimulator()
        except ImportError:
            logger.warning("qiskit-aer not available")
            return None

    def get_fake_backend(self, name: str = "fake_montreal"):
        """
        Return a fake IBM backend for realistic noise simulation without credentials.
        Falls back to generic Aer if unavailable.
        """
        fake_backends = {
            "fake_montreal": self._try_fake_montreal,
            "fake_nairobi": self._try_fake_nairobi,
            "fake_lagos": self._try_fake_lagos,
        }
        getter = fake_backends.get(name, self._try_fake_montreal)
        backend = getter()
        if backend is None:
            logger.warning("Fake backend %s unavailable, using Aer", name)
            return self.get_aer_simulator()
        return backend

    def extract_properties(self, backend) -> BackendProperties:
        """Extract key properties from a backend."""
        try:
            name = backend.name if hasattr(backend, "name") else str(backend)
            is_sim = "simulator" in name.lower() or "fake" in name.lower() or "aer" in name.lower()

            # Number of qubits
            config = backend.configuration() if hasattr(backend, "configuration") else None
            n_qubits = config.n_qubits if config else getattr(backend, "num_qubits", 5)

            # Coupling map
            if config and hasattr(config, "coupling_map") and config.coupling_map:
                coupling_map = config.coupling_map
            else:
                coupling_map = [[i, i + 1] for i in range(n_qubits - 1)]

            # Basis gates
            basis_gates = config.basis_gates if config and hasattr(config, "basis_gates") else ["cx", "u1", "u2", "u3"]

            # Error rates
            avg_gate_err = 0.001
            avg_cx_err = 0.01
            avg_ro_err = 0.02
            t1, t2 = 50.0, 30.0  # defaults

            try:
                props = backend.properties()
                if props:
                    gate_errors = []
                    cx_errors = []
                    ro_errors = []
                    t1s = []
                    t2s = []
                    for q in range(n_qubits):
                        try:
                            ro_errors.append(props.readout_error(q) or 0.02)
                            t1s.append((props.t1(q) or 50e-6) * 1e6)
                            t2s.append((props.t2(q) or 30e-6) * 1e6)
                        except Exception:
                            pass
                    for gate_name in ["u1", "u2", "u3", "sx", "rz"]:
                        for q in range(n_qubits):
                            try:
                                err = props.gate_error(gate_name, [q])
                                if err is not None:
                                    gate_errors.append(err)
                            except Exception:
                                pass
                    for i, j in coupling_map:
                        try:
                            err = props.gate_error("cx", [i, j])
                            if err is not None:
                                cx_errors.append(err)
                        except Exception:
                            pass
                    if gate_errors:
                        avg_gate_err = float(sum(gate_errors) / len(gate_errors))
                    if cx_errors:
                        avg_cx_err = float(sum(cx_errors) / len(cx_errors))
                    if ro_errors:
                        avg_ro_err = float(sum(ro_errors) / len(ro_errors))
                    if t1s:
                        t1 = float(sum(t1s) / len(t1s))
                    if t2s:
                        t2 = float(sum(t2s) / len(t2s))
            except Exception as e:
                logger.debug("Could not extract detailed properties: %s", e)

            return BackendProperties(
                name=str(name),
                n_qubits=n_qubits,
                coupling_map=coupling_map,
                basis_gates=basis_gates,
                avg_gate_error=avg_gate_err,
                avg_cx_error=avg_cx_err,
                avg_readout_error=avg_ro_err,
                t1_us=t1,
                t2_us=t2,
                is_simulator=is_sim,
            )
        except Exception as exc:
            logger.warning("Property extraction failed: %s — using defaults", exc)
            return BackendProperties(
                name="unknown", n_qubits=5, coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4]],
                basis_gates=["cx", "u3"], avg_gate_error=0.001, avg_cx_error=0.01,
                avg_readout_error=0.02, t1_us=50.0, t2_us=30.0, is_simulator=True,
            )

    def make_noise_model_from_properties(self, props: BackendProperties):
        """Build Qiskit Aer noise model from backend properties."""
        try:
            from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
            import numpy as np

            nm = NoiseModel()

            # Single-qubit gate errors
            for gate in ["u1", "u2", "u3", "sx", "x"]:
                err = depolarizing_error(props.avg_gate_error, 1)
                nm.add_all_qubit_quantum_error(err, [gate])

            # Two-qubit gate errors
            cx_err = depolarizing_error(props.avg_cx_error, 2)
            nm.add_all_qubit_quantum_error(cx_err, ["cx"])

            # Readout errors
            for q in range(props.n_qubits):
                p = props.avg_readout_error
                ro_err = ReadoutError([[1 - p, p], [p, 1 - p]])
                nm.add_readout_error(ro_err, [q])

            return nm
        except Exception as exc:
            logger.warning("Could not build noise model: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Fake backend helpers
    # ------------------------------------------------------------------

    def _try_fake_montreal(self):
        try:
            from qiskit_ibm_runtime.fake_provider import FakeMontreal
            return FakeMontreal()
        except ImportError:
            try:
                from qiskit.providers.fake_provider import FakeMontreal
                return FakeMontreal()
            except ImportError:
                return None

    def _try_fake_nairobi(self):
        try:
            from qiskit_ibm_runtime.fake_provider import FakeNairobi
            return FakeNairobi()
        except ImportError:
            try:
                from qiskit.providers.fake_provider import FakeNairobi
                return FakeNairobi()
            except ImportError:
                return None

    def _try_fake_lagos(self):
        try:
            from qiskit_ibm_runtime.fake_provider import FakeLagos
            return FakeLagos()
        except ImportError:
            try:
                from qiskit.providers.fake_provider import FakeLagos
                return FakeLagos()
            except ImportError:
                return None
