"""
QAOA Parameter Optimizer.
Handles variational parameter optimization for QAOA circuits.
Supports multiple classical optimizers and backend execution modes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    optimal_gamma: np.ndarray
    optimal_beta: np.ndarray
    optimal_energy: float
    n_function_evals: int
    convergence_history: list   # list of (eval_idx, energy) tuples
    runtime_seconds: float
    success: bool
    backend_name: str
    shots: int
    metadata: dict = field(default_factory=dict)


class QAOAOptimizer:
    """
    Variational optimizer for QAOA parameters.

    Parameters
    ----------
    p            : QAOA depth
    shots        : measurement shots per evaluation
    optimizer    : 'COBYLA', 'SPSA', 'Nelder-Mead', 'L-BFGS-B'
    max_iter     : maximum optimizer iterations
    backend_name : 'statevector_simulator', 'aer_simulator', or IBM backend name
    use_noise    : whether to apply noise model
    """

    def __init__(
        self,
        p: int = 1,
        shots: int = 4096,
        optimizer: str = "COBYLA",
        max_iter: int = 300,
        backend_name: str = "aer_simulator",
        use_noise: bool = False,
        noise_model=None,
        coupling_map=None,
        basis_gates=None,
    ):
        self.p = p
        self.shots = shots
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.backend_name = backend_name
        self.use_noise = use_noise
        self.noise_model = noise_model
        self.coupling_map = coupling_map
        self.basis_gates = basis_gates
        self._backend = None

    # ------------------------------------------------------------------

    def optimize(
        self,
        qc_parameterized,
        circuit_builder,
        ising_J: dict,
        ising_h: dict,
        offset: float = 0.0,
        initial_gamma: Optional[np.ndarray] = None,
        initial_beta: Optional[np.ndarray] = None,
        callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        Optimize QAOA parameters using variational principle.

        Parameters
        ----------
        qc_parameterized : parameterized QuantumCircuit from QAOACircuitBuilder
        circuit_builder  : QAOACircuitBuilder instance (for bind_parameters)
        ising_J, ising_h : Ising coefficients
        offset           : constant energy offset
        initial_gamma/beta : warm-start values (random if None)

        Returns
        -------
        OptimizationResult
        """
        from scipy.optimize import minimize

        # ── Initial parameters ─────────────────────────────────────────
        if initial_gamma is None:
            rng = np.random.default_rng(42)
            initial_gamma = rng.uniform(0, np.pi, self.p)
        if initial_beta is None:
            rng = np.random.default_rng(43)
            initial_beta = rng.uniform(0, np.pi / 2, self.p)

        x0 = np.concatenate([initial_gamma, initial_beta])
        convergence_history = []
        eval_count = [0]
        best_energy = [np.inf]
        t_start = time.time()

        def objective(params):
            gamma = params[: self.p]
            beta = params[self.p :]
            bound_qc = circuit_builder.bind_parameters(qc_parameterized, gamma, beta)
            energy = self._evaluate_energy(bound_qc, ising_J, ising_h, offset)
            eval_count[0] += 1
            if energy < best_energy[0]:
                best_energy[0] = energy
            convergence_history.append((eval_count[0], energy))
            if callback:
                callback(eval_count[0], energy, params)
            return energy

        # ── Run optimizer ──────────────────────────────────────────────
        try:
            if self.optimizer_name == "COBYLA":
                result = minimize(
                    objective, x0, method="COBYLA",
                    options={"maxiter": self.max_iter, "rhobeg": 0.5},
                )
            elif self.optimizer_name == "SPSA":
                result = self._spsa_minimize(objective, x0)
            else:
                result = minimize(
                    objective, x0, method=self.optimizer_name,
                    options={"maxiter": self.max_iter},
                )
            success = result.success if hasattr(result, "success") else True
            opt_params = result.x
        except Exception as exc:
            logger.warning("Optimizer failed: %s — returning best found so far", exc)
            opt_params = x0
            success = False

        runtime = time.time() - t_start
        opt_gamma = opt_params[: self.p]
        opt_beta = opt_params[self.p :]

        return OptimizationResult(
            optimal_gamma=opt_gamma,
            optimal_beta=opt_beta,
            optimal_energy=best_energy[0],
            n_function_evals=eval_count[0],
            convergence_history=convergence_history,
            runtime_seconds=runtime,
            success=success,
            backend_name=self.backend_name,
            shots=self.shots,
            metadata={"optimizer": self.optimizer_name, "p": self.p},
        )

    # ------------------------------------------------------------------

    def _evaluate_energy(self, bound_qc, J: dict, h: dict, offset: float) -> float:
        """Execute circuit and compute expectation value of Ising Hamiltonian."""
        counts = self._run_circuit(bound_qc)
        total = sum(counts.values())
        energy = 0.0
        for bitstring, count in counts.items():
            # bitstring is big-endian (Qiskit convention): reverse for qubit indexing
            s = np.array([1 - 2 * int(b) for b in reversed(bitstring)], dtype=float)
            e = offset
            for (i, j), coupling in J.items():
                if i < len(s) and j < len(s):
                    e += coupling * s[i] * s[j]
            for i, field in h.items():
                if i < len(s):
                    e += field * s[i]
            energy += count * e
        return energy / total

    def _run_circuit(self, qc) -> dict:
        """Execute circuit on chosen backend, return measurement counts."""
        try:
            from qiskit_aer import AerSimulator
            from qiskit import transpile

            if self._backend is None:
                if self.use_noise and self.noise_model is not None:
                    self._backend = AerSimulator(noise_model=self.noise_model)
                else:
                    self._backend = AerSimulator()

            backend = self._backend
            tc = transpile(qc, backend)
            job = backend.run(tc, shots=self.shots)
            counts = job.result().get_counts()
            return counts
        except Exception as exc:
            logger.warning("Qiskit execution failed: %s — using statevector fallback", exc)
            return self._statevector_fallback(qc)

    def _statevector_fallback(self, qc) -> dict:
        """Fallback: sample from statevector if Aer unavailable."""
        try:
            from qiskit_aer import StatevectorSimulator
            from qiskit import transpile

            sv_backend = StatevectorSimulator()
            # Remove measurements for statevector
            qc_no_meas = qc.remove_final_measurements(inplace=False)
            tc = transpile(qc_no_meas, sv_backend)
            job = sv_backend.run(tc)
            sv = job.result().get_statevector()
            probs = np.abs(np.array(sv)) ** 2
            n_qubits = qc.num_qubits
            counts = {}
            for idx, p in enumerate(probs):
                if p > 1e-10:
                    bs = format(idx, f"0{n_qubits}b")
                    counts[bs] = int(round(p * self.shots))
            return counts
        except Exception as exc2:
            logger.error("Statevector fallback also failed: %s", exc2)
            return {}

    def _spsa_minimize(self, objective, x0: np.ndarray):
        """Simultaneous Perturbation Stochastic Approximation."""
        x = x0.copy()
        a = 0.1
        c = 0.1
        alpha = 0.602
        gamma_spsa = 0.101
        best_x = x.copy()
        best_val = objective(x)

        class _Result:
            pass

        for k in range(1, self.max_iter + 1):
            ak = a / (k ** alpha)
            ck = c / (k ** gamma_spsa)
            delta = 2 * np.random.randint(0, 2, size=len(x)) - 1
            x_plus = x + ck * delta
            x_minus = x - ck * delta
            g_hat = (objective(x_plus) - objective(x_minus)) / (2 * ck) * delta
            x = x - ak * g_hat
            val = objective(x)
            if val < best_val:
                best_val = val
                best_x = x.copy()

        r = _Result()
        r.x = best_x
        r.success = True
        return r
