"""
Tests for benchmark metrics and TTS99 computation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestTTS99:

    def test_basic_tts99(self):
        from benchmark.tts99 import TTS99Calculator
        calc = TTS99Calculator(target_success_prob=0.99)
        energies = np.array([-10.0] * 6 + [-8.0] * 4)  # 60% success rate
        runtimes = np.ones(10) * 0.1
        result = calc.compute("test_solver", energies, runtimes, -10.0, 0.95)
        assert result.p_success == pytest.approx(0.6, abs=0.01)
        assert result.tts99 is not None
        assert result.tts99 > 0

    def test_all_success(self):
        from benchmark.tts99 import TTS99Calculator
        calc = TTS99Calculator()
        energies = np.full(10, -10.0)
        runtimes = np.ones(10) * 0.5
        result = calc.compute("solver", energies, runtimes, -10.0, 0.95)
        assert result.p_success == pytest.approx(1.0)
        assert result.tts99 == pytest.approx(0.5)

    def test_no_success(self):
        from benchmark.tts99 import TTS99Calculator
        calc = TTS99Calculator()
        energies = np.full(10, -5.0)  # none reach -10 threshold
        runtimes = np.ones(10)
        result = calc.compute("solver", energies, runtimes, -10.0, 0.95)
        assert result.p_success == 0.0
        assert result.tts99 is None

    def test_wilson_ci_bounds(self):
        from benchmark.tts99 import TTS99Calculator
        calc = TTS99Calculator()
        lo, hi = calc._wilson_ci(5, 10)
        assert 0.0 <= lo <= hi <= 1.0

    def test_tts99_formula_manual(self):
        """Verify TTS formula: TTS = t * log(1-0.99) / log(1-p)."""
        from benchmark.tts99 import TTS99Calculator
        calc = TTS99Calculator()
        p = 0.5
        t = 1.0
        expected = t * np.log(1 - 0.99) / np.log(1 - p)
        computed = calc._tts(p, t, 0.99)
        assert abs(computed - expected) < 1e-9


class TestBenchmarkMetrics:

    def _make_results(self):
        from benchmark.metrics import RunResult
        results = []
        for trial in range(10):
            results.append(RunResult(
                solver="sa", trial=trial, best_energy=-8.0 + np.random.normal(0, 0.5),
                routing_cost=5.0, latency=2.0, congestion_score=1.0, loss_score=0.01,
                runtime_seconds=0.1, feasible=True, n_iterations=1000,
            ))
            results.append(RunResult(
                solver="dijkstra", trial=trial, best_energy=-7.0,
                routing_cost=6.0, latency=3.0, congestion_score=0.5, loss_score=0.005,
                runtime_seconds=0.001, feasible=True, n_iterations=1,
            ))
        return results

    def test_aggregate_returns_per_solver(self):
        np.random.seed(42)
        from benchmark.metrics import BenchmarkMetrics
        metrics = BenchmarkMetrics()
        results = self._make_results()
        stats = metrics.aggregate(results)
        names = [s.solver for s in stats]
        assert "sa" in names
        assert "dijkstra" in names

    def test_feasibility_rate(self):
        from benchmark.metrics import BenchmarkMetrics, RunResult
        metrics = BenchmarkMetrics()
        results = [
            RunResult("solver", i, -5.0, 1.0, 1.0, 0.1, 0.0, 0.1, i % 2 == 0, 10)
            for i in range(10)
        ]
        stats = metrics.aggregate(results)
        assert stats[0].feasibility_rate == pytest.approx(0.5, abs=0.01)

    def test_approximation_ratio_perfect(self):
        from benchmark.metrics import BenchmarkMetrics, RunResult
        metrics = BenchmarkMetrics()
        results = [RunResult("solver", i, -10.0, 1.0, 1.0, 0.1, 0.0, 0.1, True, 10)
                   for i in range(5)]
        stats = metrics.aggregate(results, best_known=-10.0)
        assert stats[0].approximation_ratio == pytest.approx(1.0, abs=0.01)
