from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spinodal_poster_final_validation import (
    _load_combined_micro_modes,
    build_epsilon_robustness,
    build_fitwindow_robustness,
    build_fully_numeric_combined_analysis,
    build_fully_numeric_dynamic_z_tables,
    bulk_normalized_profile,
    fit_dynamic_exponent,
    make_fully_numeric_combined_figures,
)


class PosterFinalValidationTests(unittest.TestCase):
    @staticmethod
    def synthetic_fully_numeric_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
        n_map = {6: 512, 12: 1024, 24: 2048, 48: 4096, 96: 8192}
        delta_values = (1e-5, 3e-5, 1e-4, 3e-4)
        q0_rows = []
        xi_rows = []
        for R, N in n_map.items():
            kappa = (R + 1) * (2 * R + 1) / 12.0
            for delta in delta_values:
                q0_rows.append(
                    {
                        "R": R, "N": N, "delta": delta,
                        "epsilon_fraction": 0.05,
                        "Gamma0_num": 0.7 * delta**0.5,
                        "Gamma0_source": "deterministic q=0 numerical",
                        "reliable": True,
                    }
                )
                xi_rows.append(
                    {
                        "R": R, "N": N, "delta": delta,
                        "epsilon_fraction": 0.05,
                        "boundary_type": "ghost_dirichlet",
                        "xi_bnd_cosh": np.sqrt(kappa / 0.7) * delta**-0.25,
                        "fit_x_min": 2.0 * R,
                        "fit_reliable": True,
                        "converged": True,
                    }
                )
        return pd.DataFrame(xi_rows), pd.DataFrame(q0_rows)

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

    def test_refinement_replaces_duplicate_with_larger_M_and_keeps_existing(self) -> None:
        columns = {
            "R": 24, "mode_index": 0, "q": 0.0, "qR": 0.0,
            "epsilon_fraction": 0.05, "Gamma_micro": 0.8,
            "Gamma_micro_se": 0.01, "reliable": True,
            "escape_fraction": 0.0, "preparation_drift": 0.0,
            "baseline_drift": 0.0, "method_B_C_relative_difference": 0.0,
            "N": 2048,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_path = root / "resultB_micro_mode_results.csv"
            refinement = root / "B_refinement" / "R024"
            pseudo = root / "phase5_R_sweep" / "R024" / "pseudospinodal_fine" / "analysis"
            refinement.mkdir(parents=True)
            pseudo.mkdir(parents=True)
            pd.DataFrame(
                [
                    {**columns, "delta": 0.036, "M_total": 8192},
                    {**columns, "delta": 0.041, "M_total": 8192},
                ]
            ).to_csv(base_path, index=False)
            pd.DataFrame(
                [{**columns, "delta": 0.036, "M_total": 32768}]
            ).to_csv(refinement / "phase5_mode_results.csv", index=False)
            pd.DataFrame(
                [{"T_obs": 50, "delta_ps_estimate": 0.031,
                  "delta_ps_se": 0.0, "delta_ps_ci_low": 0.031,
                  "delta_ps_ci_high": 0.031}]
            ).to_csv(pseudo / "phase5_pseudospinodal_time_dependence.csv", index=False)

            merged = _load_combined_micro_modes(
                base_path, root / "B_refinement", root / "phase5_R_sweep",
                T_obs=50, epsilon_fraction=0.05,
            )
            self.assertEqual(len(merged), 2)
            duplicate = merged[np.isclose(merged["matched_distance"], 0.005)].iloc[0]
            self.assertEqual(int(duplicate["M_total"]), 32768)
            self.assertEqual(duplicate["data_source"], "poster_B_refinement")
            self.assertTrue(np.isclose(merged["matched_distance"], 0.010).any())

    def test_fully_numeric_builder_rejects_theory_fallback(self) -> None:
        xi, q0 = self.synthetic_fully_numeric_inputs()
        q0.loc[0, "Gamma0_source"] = "Phase0 theory fallback"
        with self.assertRaisesRegex(ValueError, "rejects non-numerical"):
            build_fully_numeric_dynamic_z_tables(xi, q0)

    def test_fully_numeric_builder_recovers_two_for_each_R(self) -> None:
        xi, q0 = self.synthetic_fully_numeric_inputs()
        points, summary = build_fully_numeric_dynamic_z_tables(xi, q0)
        self.assertEqual(len(points), 20)
        self.assertTrue(points["included"].all())
        self.assertEqual(set(summary["R"]), {6, 12, 24, 48, 96})
        self.assertTrue((summary["n_points"] == 4).all())
        np.testing.assert_allclose(summary["z"], 2.0, rtol=0.0, atol=1e-12)

    def test_normalized_combined_collapse_uses_kappa_and_all_20_points(self) -> None:
        xi, q0 = self.synthetic_fully_numeric_inputs()
        points, _ = build_fully_numeric_dynamic_z_tables(xi, q0)
        combined, summary = build_fully_numeric_combined_analysis(
            points, lattice_spacing=1.0
        )
        self.assertEqual(len(combined), 20)
        self.assertEqual(set(combined["R"]), {6, 12, 24, 48, 96})
        r12 = combined[combined["R"] == 12].iloc[0]
        self.assertAlmostEqual(float(r12["kappa_R"]), 27.083333333333332)
        self.assertAlmostEqual(
            float(r12["xi_scaled"]),
            float(r12["xi_bnd"]) / np.sqrt(float(r12["kappa_R"])),
        )
        self.assertTrue(np.isfinite(combined["collapse_ratio"]).all())
        self.assertTrue((combined["collapse_ratio"] > 0.0).all())
        np.testing.assert_allclose(combined["collapse_ratio"], 1.0, atol=1e-12)
        self.assertAlmostEqual(float(summary["combined_fit_all_20"]["z"]), 2.0, places=12)
        self.assertTrue(summary["all_20_points_present"])
        self.assertTrue(summary["all_20_points_passed_reliability_criteria"])
        self.assertEqual(summary["Gamma0_source"], "deterministic q=0 numerical")

    def test_combined_figure_filenames_are_generated(self) -> None:
        xi, q0 = self.synthetic_fully_numeric_inputs()
        points, _ = build_fully_numeric_dynamic_z_tables(xi, q0)
        combined, summary = build_fully_numeric_combined_analysis(
            points, lattice_spacing=1.0
        )
        with tempfile.TemporaryDirectory() as temporary:
            paths = make_fully_numeric_combined_figures(
                Path(temporary), combined, summary
            )
            names = {path.name for path in paths}
            self.assertEqual(
                names,
                {
                    "03_fully_numeric_dynamic_z_combined.png",
                    "03_fully_numeric_dynamic_z_combined_raw.png",
                },
            )
            self.assertTrue(all(path.is_file() for path in paths))


if __name__ == "__main__":
    unittest.main()
