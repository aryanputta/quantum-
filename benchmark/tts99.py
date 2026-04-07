"""
TTS99 Calculator — Time-To-Solution at 99% success probability.
Applied consistently across all solver families.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TTS99Result:
    solver: str
    target_energy: float
    target_fraction: float
    p_success: float            # empirical success probability
    n_trials: int
    n_success: int
    mean_runtime_per_trial: float
    tts99: Optional[float]      # seconds, None if not computable
    tts50: Optional[float]      # TTS at 50% (median)
    confidence_interval: tuple  # 95% CI on p_success via Wilson interval
    notes: str = ""


class TTS99Calculator:
    """
    Compute TTS99 for a solver given per-trial results.

    TTS_p = t_run * log(1 - p_desired) / log(1 - p_success)

    Assumptions:
    - Trials are i.i.d.
    - Single-run time = mean observed runtime
    - Success = energy ≤ target_energy (configurable threshold)
    """

    def __init__(self, target_success_prob: float = 0.99):
        self.target_success_prob = target_success_prob

    def compute(
        self,
        solver_name: str,
        energies: np.ndarray,
        runtimes: np.ndarray,
        target_energy: float,
        target_fraction: float = 0.95,
    ) -> TTS99Result:
        """
        Compute TTS99 for one solver.

        Parameters
        ----------
        solver_name    : identifier string
        energies       : per-trial best energies (lower = better)
        runtimes       : per-trial runtimes in seconds
        target_energy  : best known energy (reference)
        target_fraction: threshold = target_energy * target_fraction
                         (for minimization: threshold = target * (1 + ε))
        """
        n = len(energies)
        if n == 0:
            return TTS99Result(
                solver=solver_name, target_energy=target_energy,
                target_fraction=target_fraction, p_success=0.0, n_trials=0,
                n_success=0, mean_runtime_per_trial=0.0, tts99=None, tts50=None,
                confidence_interval=(0.0, 1.0),
                notes="No trials provided",
            )

        # Compute success threshold.
        #
        # Interpretation: a run is "successful" if it finds a solution within
        # (1 - target_fraction) × 100 % of the best known energy.
        #
        # For a minimiser with target_energy < 0:
        #   threshold = target_energy * target_fraction
        #   e.g. target=-10, fraction=0.95 → threshold=-9.5
        #   (solutions ≤ -9.5 are within 5 % of optimal)
        #
        # For target_energy ≥ 0:
        #   threshold = target_energy * (2 - target_fraction)
        #   e.g. target=10, fraction=0.95 → threshold=10.5
        #   (anything ≤ 10.5 is within 5 % of optimal from above)
        if target_energy < 0:
            threshold = target_energy * target_fraction        # less negative = easier
        elif target_energy > 0:
            threshold = target_energy * (2.0 - target_fraction)  # slightly above target
        else:
            threshold = 0.0

        n_success = int(np.sum(energies <= threshold))
        p_hat = n_success / n
        t_run = float(np.mean(runtimes))

        # Wilson score confidence interval on p_success
        ci = self._wilson_ci(n_success, n)

        # TTS computation
        tts99 = self._tts(p_hat, t_run, 0.99)
        tts50 = self._tts(p_hat, t_run, 0.50)

        notes = []
        notes.append(f"threshold={threshold:.4f}")
        notes.append(f"n_success={n_success}/{n}")
        notes.append(f"t_run={t_run:.4f}s")
        if p_hat == 0:
            notes.append("WARNING: p_success=0, TTS99 undefined (∞)")
        if p_hat == 1.0:
            notes.append("p_success=1.0: every run succeeded")

        return TTS99Result(
            solver=solver_name,
            target_energy=float(target_energy),
            target_fraction=target_fraction,
            p_success=p_hat,
            n_trials=n,
            n_success=n_success,
            mean_runtime_per_trial=t_run,
            tts99=tts99,
            tts50=tts50,
            confidence_interval=ci,
            notes=" | ".join(notes),
        )

    def compute_all(
        self,
        solver_results: dict,
        best_known: float,
        target_fraction: float = 0.95,
    ) -> dict[str, TTS99Result]:
        """
        Compute TTS99 for all solvers.

        Parameters
        ----------
        solver_results : {solver_name: {'energies': array, 'runtimes': array}}
        best_known     : reference energy (minimum across all solvers)

        Returns
        -------
        dict mapping solver name → TTS99Result
        """
        results = {}
        for name, data in solver_results.items():
            energies = np.array(data["energies"])
            runtimes = np.array(data["runtimes"])
            results[name] = self.compute(
                name, energies, runtimes, best_known, target_fraction
            )
        return results

    def summary_table(self, tts_results: dict[str, TTS99Result]) -> str:
        """Return a formatted text table of TTS99 results."""
        header = f"{'Solver':<30} {'p_success':>10} {'TTS99(s)':>12} {'TTS50(s)':>12} {'n_success':>10}"
        lines = [header, "-" * len(header)]
        for name, r in sorted(tts_results.items(), key=lambda x: x[1].tts99 or 1e18):
            tts99_str = f"{r.tts99:.4f}" if r.tts99 is not None else "∞"
            tts50_str = f"{r.tts50:.4f}" if r.tts50 is not None else "N/A"
            lines.append(
                f"{name:<30} {r.p_success:>10.4f} {tts99_str:>12} {tts50_str:>12} {r.n_success:>5}/{r.n_trials:<5}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------

    def _tts(self, p_success: float, t_run: float, p_desired: float) -> Optional[float]:
        """TTS formula. Returns None if undefined."""
        if p_success <= 0.0:
            return None
        if p_success >= 1.0:
            return t_run
        return t_run * np.log(1.0 - p_desired) / np.log(1.0 - p_success)

    def _wilson_ci(self, k: int, n: int, z: float = 1.96) -> tuple:
        """Wilson score 95% confidence interval for binomial proportion."""
        if n == 0:
            return (0.0, 1.0)
        p_hat = k / n
        denom = 1 + z ** 2 / n
        center = (p_hat + z ** 2 / (2 * n)) / denom
        margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z ** 2 / (4 * n ** 2)) / denom
        return (max(0.0, center - margin), min(1.0, center + margin))
