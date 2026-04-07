"""
Tests for QUBO ↔ Ising conversion with energy equivalence verification.
"""

import numpy as np
import pytest

sys_path_added = False


def setup_path():
    global sys_path_added
    if not sys_path_added:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        sys_path_added = True


@pytest.fixture(autouse=True)
def ensure_path():
    setup_path()


class TestIsingConversion:

    def test_qubo_to_ising_small(self):
        setup_path()
        from ising_utils.converter import IsingConverter
        Q = np.array([[1.0, -2.0], [-2.0, 3.0]])
        converter = IsingConverter()
        result = converter.qubo_to_ising(Q)
        assert result.n_spins == 2
        assert isinstance(result.J, dict)
        assert isinstance(result.h, dict)
        assert isinstance(result.offset, float)

    def test_roundtrip_symmetry(self):
        setup_path()
        from ising_utils.converter import IsingConverter
        rng = np.random.default_rng(42)
        n = 5
        Q = rng.uniform(-1, 1, (n, n))
        Q = (Q + Q.T) / 2  # symmetrize
        converter = IsingConverter()
        assert converter.roundtrip_check(Q, tol=1e-6), "QUBO→Ising→QUBO roundtrip failed"

    def test_energy_equivalence_2qubits(self):
        setup_path()
        from ising_utils.converter import IsingConverter
        from ising_utils.energy import (
            evaluate_qubo_energy, evaluate_ising_energy,
            qubo_to_ising_bitstring,
        )
        Q = np.array([[2.0, -1.0], [-1.0, 2.0]])
        converter = IsingConverter()
        ising = converter.qubo_to_ising(Q)

        for x0 in [0, 1]:
            for x1 in [0, 1]:
                x = np.array([x0, x1])
                s = qubo_to_ising_bitstring(x)
                e_qubo = evaluate_qubo_energy(Q, x)
                e_ising = evaluate_ising_energy(ising.J, ising.h, ising.offset, s)
                assert abs(e_qubo - e_ising) < 1e-6, (
                    f"Energy mismatch for x={x}: QUBO={e_qubo}, Ising={e_ising}"
                )

    def test_energy_equivalence_sampled(self):
        setup_path()
        from ising_utils.converter import IsingConverter
        from ising_utils.energy import verify_energy_equivalence
        rng = np.random.default_rng(7)
        n = 4
        Q = rng.uniform(-2, 2, (n, n))
        Q = (Q + Q.T) / 2
        converter = IsingConverter()
        ising = converter.qubo_to_ising(Q)
        assert verify_energy_equivalence(Q, None, ising), "Energy equivalence check failed"

    def test_ising_to_qubo_roundtrip(self):
        setup_path()
        from ising_utils.converter import IsingConverter
        rng = np.random.default_rng(99)
        n = 6
        Q = rng.uniform(-3, 3, (n, n))
        Q = np.triu(Q) + np.triu(Q, 1).T  # upper triangular, symmetric
        converter = IsingConverter()
        ising = converter.qubo_to_ising(Q)
        Q_rec = converter.ising_to_qubo(ising)
        assert Q_rec.shape == (n, n), "Reconstructed Q has wrong shape"
        assert np.allclose(Q, Q_rec, atol=1e-6), f"Q reconstruction error: max={np.max(np.abs(Q - Q_rec)):.2e}"


class TestQUBOEnergy:

    def test_zero_vector(self):
        setup_path()
        from ising_utils.energy import evaluate_qubo_energy
        Q = np.eye(4)
        x = np.zeros(4, dtype=int)
        assert evaluate_qubo_energy(Q, x) == 0.0

    def test_all_ones(self):
        setup_path()
        from ising_utils.energy import evaluate_qubo_energy
        Q = np.ones((3, 3))
        x = np.ones(3, dtype=int)
        # x^T Q x = sum of all Q entries = 9
        assert evaluate_qubo_energy(Q, x) == pytest.approx(9.0)

    def test_diagonal_only(self):
        setup_path()
        from ising_utils.energy import evaluate_qubo_energy
        Q = np.diag([1.0, 2.0, 3.0])
        x = np.array([1, 1, 0])
        assert evaluate_qubo_energy(Q, x) == pytest.approx(3.0)  # 1 + 2 + 0
