"""
Benchmark Reporter.
Generates formatted terminal output, tables, and plots for solver comparison.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class BenchmarkReporter:
    """
    Format and display benchmark results.
    Provides rich terminal output and matplotlib plots.
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def print_solver_table(self, stats_df, title: str = "Solver Benchmark Results"):
        """Print a formatted comparison table to terminal."""
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box

            console = Console()
            table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold cyan")

            for col in stats_df.columns:
                table.add_column(str(col), justify="right")

            for _, row in stats_df.iterrows():
                table.add_row(*[str(v) for v in row.values])

            console.print(table)
        except ImportError:
            # Fallback to plain print
            print(f"\n{'='*80}")
            print(f"  {title}")
            print(f"{'='*80}")
            print(stats_df.to_string(index=False))
            print(f"{'='*80}\n")

    def print_tts99_table(self, tts_summary: str):
        """Print TTS99 summary table."""
        print("\n" + "=" * 70)
        print("  TTS99 Analysis (Time-to-Solution at 99% Success Probability)")
        print("=" * 70)
        print(tts_summary)
        print("=" * 70 + "\n")

    def print_circuit_stats(self, circuit_stats: dict):
        """Print circuit statistics comparison."""
        print("\n" + "=" * 60)
        print("  Circuit Statistics: Logical vs Hardware-Transpiled")
        print("=" * 60)
        for key, val in circuit_stats.items():
            print(f"  {key:<30} {val}")
        print("=" * 60 + "\n")

    def plot_convergence(self, solver_histories: dict, title: str = "Convergence History", save: bool = True):
        """
        Plot energy convergence curves for all solvers.

        Parameters
        ----------
        solver_histories : {solver_name: list of (iter, energy)} or list of energies
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.cm as cm

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = cm.tab10(np.linspace(0, 1, len(solver_histories)))

            for (name, history), color in zip(solver_histories.items(), colors):
                if not history:
                    continue
                if isinstance(history[0], (tuple, list)):
                    iters = [h[0] for h in history]
                    energies = [h[1] for h in history]
                else:
                    iters = list(range(len(history)))
                    energies = list(history)
                ax.plot(iters, energies, label=name, color=color, linewidth=1.5)

            ax.set_xlabel("Iteration / Function Evaluation")
            ax.set_ylabel("Best Energy")
            ax.set_title(title)
            ax.legend(loc="upper right")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            if save:
                path = os.path.join(self.results_dir, "convergence.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                logger.info("Convergence plot saved: %s", path)
                print(f"  [PLOT] Convergence chart saved: {path}")

            plt.show()
            plt.close()
        except Exception as exc:
            logger.warning("Convergence plot failed: %s", exc)

    def plot_approximation_ratio(self, solver_stats, save: bool = True):
        """Bar chart of approximation ratios."""
        try:
            import matplotlib.pyplot as plt

            names = [s.solver for s in solver_stats]
            ratios = [s.approximation_ratio for s in solver_stats]
            colors = ["green" if r >= 0.9 else "orange" if r >= 0.7 else "red" for r in ratios]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(names, ratios, color=colors, edgecolor="black", linewidth=0.5)
            ax.axhline(y=1.0, color="blue", linestyle="--", label="Optimal")
            ax.set_ylabel("Approximation Ratio (higher = better)")
            ax.set_title("Solver Approximation Ratio vs Best Known Solution")
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
            for bar, ratio in zip(bars, ratios):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01, f"{ratio:.3f}",
                        ha="center", va="bottom", fontsize=9)
            plt.tight_layout()

            if save:
                path = os.path.join(self.results_dir, "approximation_ratio.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  [PLOT] Approximation ratio chart saved: {path}")
            plt.show()
            plt.close()
        except Exception as exc:
            logger.warning("Approx ratio plot failed: %s", exc)

    def plot_tts99_comparison(self, tts_results: dict, save: bool = True):
        """Bar chart of TTS99 values."""
        try:
            import matplotlib.pyplot as plt

            names = list(tts_results.keys())
            tts_vals = [r.tts99 if r.tts99 is not None else float("nan") for r in tts_results.values()]

            fig, ax = plt.subplots(figsize=(10, 5))
            bars = ax.bar(names, tts_vals, color="steelblue", edgecolor="black")
            ax.set_ylabel("TTS99 (seconds)")
            ax.set_title("Time-to-Solution at 99% Success Probability")
            ax.set_xticklabels(names, rotation=30, ha="right")
            ax.set_yscale("log")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            if save:
                path = os.path.join(self.results_dir, "tts99.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  [PLOT] TTS99 chart saved: {path}")
            plt.show()
            plt.close()
        except Exception as exc:
            logger.warning("TTS99 plot failed: %s", exc)

    def save_benchmark_csv(self, stats_df, filename: str = "benchmark_results.csv"):
        path = os.path.join(self.results_dir, filename)
        stats_df.to_csv(path, index=False)
        print(f"  [CSV] Benchmark results saved: {path}")

    def plot_energy_distribution(self, solver_energies: dict, save: bool = True):
        """Box-plot of energy distributions across trials."""
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            data = [np.array(v) for v in solver_energies.values()]
            labels = list(solver_energies.keys())
            ax.boxplot(data, labels=labels, notch=False, patch_artist=True)
            ax.set_ylabel("Energy (QUBO objective)")
            ax.set_title("Energy Distribution Across Trials per Solver")
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.grid(axis="y", alpha=0.3)
            plt.tight_layout()

            if save:
                path = os.path.join(self.results_dir, "energy_distribution.png")
                plt.savefig(path, dpi=150, bbox_inches="tight")
                print(f"  [PLOT] Energy distribution saved: {path}")
            plt.show()
            plt.close()
        except Exception as exc:
            logger.warning("Energy distribution plot failed: %s", exc)
