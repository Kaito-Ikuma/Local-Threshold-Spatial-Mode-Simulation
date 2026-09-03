from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spinodal_poster_final_validation import (
    build_epsilon_robustness,
    build_fitwindow_robustness,
    bulk_normalized_profile,
    fit_dynamic_exponent,
)


class PosterFinalValidationTests(unittest.TestCase):
    def test_synthetic_dynamic_exponent_recovers_two(self) -> None:
        xi = np.array([3.0, 5.0, 8.0, 13.0])
        tau = 1.7 * xi**2
        fit = fit_dynamic_exponent(xi, tau)
        self.assertAlmostEqual(float(fit["z"]), 2.0, places=12)

    def test_equal_epsilon_lengths_have_unit_ratio(self) -> None:
        table = pd.DataFrame(
            {
                "R": [12, 12, 12], "delta": [1e-4] * 3,
                "epsilon_fraction": [0.025, 0.05, 0.10],
                "boundary_type": ["ghost_dirichlet"] * 3,
                "xi_bnd_cosh": [23.0] * 3,
                "fit_reliable": [True] * 3, "converged": [True] * 3,
            }
        )
        result = build_epsilon_robustness(table)
        np.testing.assert_allclose(result["relative_ratio"], 1.0)

    def test_two_R_window_is_the_unit_reference(self) -> None:
        systematics = pd.DataFrame(
            {
                "R": [12] * 3, "delta": [1e-4] * 3,
                "epsilon_fraction": [0.05] * 3,
                "boundary_type": ["ghost_dirichlet"] * 3,
                "fit_x_min_factor_R": [1.0, 2.0, 3.0],
                "xi_cosh": [20.0, 22.0, 24.0],
                "r2": [0.99] * 3, "rmse": [1e-7] * 3,
            }
        )
        xi = pd.DataFrame(
            {"R": [12], "delta": [1e-4], "epsilon_fraction": [0.05],
             "boundary_type": ["ghost_dirichlet"], "fit_reliable": [True],
             "converged": [True]}
        )
        result = build_fitwindow_robustness(systematics, xi)
        reference = result[np.isclose(result["fit_x_min_factor_R"], 2.0)]
        self.assertAlmostEqual(float(reference["relative_ratio"].iloc[0]), 1.0)

    def test_synthetic_bulk_exponential_collapses_to_exp_minus_X(self) -> None:
        R = 6
        xi = 9.5
        x = np.arange(0.0, 101.0)
        response = 0.3 * np.exp(-(x - 2.0 * R) / xi)
        result = bulk_normalized_profile(x, response, R=R, xi_bulk=xi)
        np.testing.assert_allclose(
            result["normalized_response"], np.exp(-result["X_bulk"]), rtol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
