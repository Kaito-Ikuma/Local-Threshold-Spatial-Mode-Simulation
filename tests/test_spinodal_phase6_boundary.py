from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spinodal_phase0 import Phase0Task, run_phase0_case
from spinodal_phase12 import periodic_local_average
from spinodal_phase6_boundary import (
    BoundaryRun,
    boundary_local_average,
    exponential_profile,
    finite_L_profile,
    fit_cosh_profile,
    fit_exponential_profile,
    simulate_boundary_pair,
)


class SpinodalPhase6BoundaryTests(unittest.TestCase):
    def test_synthetic_exponential_recovers_xi(self) -> None:
        x = np.linspace(0.0, 80.0, 161)
        values = exponential_profile(x, 0.23, 13.5)
        fit = fit_exponential_profile(x, values)
        self.assertAlmostEqual(float(fit["xi"]), 13.5, places=8)
        self.assertAlmostEqual(float(fit["amplitude"]), 0.23, places=8)

    def test_synthetic_finite_L_recovers_xi(self) -> None:
        L = 100.0
        x = np.linspace(0.0, L, 201)
        values = finite_L_profile(x, -0.17, 18.0, L)
        fit = fit_cosh_profile(x, values, L=L)
        self.assertAlmostEqual(float(fit["xi"]), 18.0, places=7)
        self.assertAlmostEqual(float(fit["amplitude"]), -0.17, places=7)

    def test_ghost_dirichlet_average_matches_hand_calculation(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        average, count = boundary_local_average(
            values, 1, left_value=10.0, boundary_type="ghost_dirichlet"
        )
        self.assertAlmostEqual(float(average[0]), 6.0)
        self.assertAlmostEqual(float(average[1]), 2.0)
        self.assertTrue(np.all(count == 2))

    def test_open_fixed_denominator_coordination_profile(self) -> None:
        values = np.arange(1.0, 10.0)
        average, count = boundary_local_average(
            values, 2, left_value=100.0, boundary_type="open_fixed_denominator"
        )
        self.assertEqual(int(count[0]), 2)
        self.assertEqual(int(count[1]), 3)
        self.assertEqual(int(count[2]), 4)
        self.assertLess(int(count[0]), 4)
        self.assertAlmostEqual(float(average[0]), (2.0 + 3.0) / 4.0)

    def test_open_renormalized_uses_available_count(self) -> None:
        values = np.arange(1.0, 10.0)
        average, count = boundary_local_average(
            values, 2, left_value=100.0, boundary_type="open_renormalized"
        )
        self.assertEqual(int(count[0]), 2)
        self.assertAlmostEqual(float(average[0]), (2.0 + 3.0) / 2.0)

    def test_existing_periodic_average_is_unchanged(self) -> None:
        values = np.arange(8.0)
        expected = sum(np.roll(values, shift) for shift in (-2, -1, 1, 2)) / 4.0
        np.testing.assert_allclose(periodic_local_average(values, 2), expected)

    def test_small_boundary_pair_reaches_finite_profile(self) -> None:
        phase0 = run_phase0_case(Phase0Task(R=6, delta_list=(1e-2,)))
        row = phase0.delta_table.iloc[0]
        run = BoundaryRun(
            R=6,
            N=128,
            delta=1e-2,
            epsilon_fraction=0.05,
            boundary_type="ghost_dirichlet",
            right_boundary="reflecting",
            m_star=float(row["m_star"]),
            m_spinodal=phase0.spinodal.m_spinodal,
            Delta=float(row["Delta"]),
            mu=phase0.spinodal.mu,
            sigma_eff=phase0.spinodal.sigma_eff,
            Gamma0_theory=float(row["Gamma0_theory"]),
            xi_dyn=float(row["xi_theory"]),
        )
        result = simulate_boundary_pair(
            run, steady_tol=1e-9, max_steps=1000, minimum_tau_multiplier=5.0
        )
        self.assertTrue(np.all(np.isfinite(result["response"])))
        self.assertGreater(float(np.max(np.abs(result["response"]))), 0.0)
        self.assertTrue(bool(result["converged"]))


if __name__ == "__main__":
    unittest.main()
