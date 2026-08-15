from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from spinodal_phase0 import kappa_R_theory
from spinodal_phase34 import (
    MISSING_INPUT_MESSAGE,
    build_collapse_values,
    build_nested_windows,
    compute_dynamic_lengths,
    fit_power_law,
    fit_q2_q4_dispersion,
    load_phase34_inputs,
    top_hat_kernel_expansion,
)


class SpinodalPhase34Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.delta = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])

    def test_01_synthetic_gamma_exponent(self) -> None:
        gamma = 3.0 * self.delta**0.5
        fit = fit_power_law(self.delta, gamma)
        self.assertAlmostEqual(float(fit["amplitude"]), 3.0, places=12)
        self.assertAlmostEqual(float(fit["exponent"]), 0.5, places=12)

    def test_02_tau_is_derived_with_opposite_exponent(self) -> None:
        gamma = 3.0 * self.delta**0.5
        fit = fit_power_law(self.delta, 1.0 / gamma)
        self.assertAlmostEqual(float(fit["exponent"]), -0.5, places=12)

    def test_03_dynamic_length_exponent(self) -> None:
        gamma = 3.0 * self.delta**0.5
        xi, _ = compute_dynamic_lengths(gamma, np.full_like(gamma, 27.0))
        fit = fit_power_law(self.delta, xi)
        self.assertAlmostEqual(float(fit["exponent"]), -0.25, places=12)

    def test_04_tau_vs_xi_dynamic_exponent(self) -> None:
        gamma = 3.0 * self.delta**0.5
        tau = 1.0 / gamma
        xi, _ = compute_dynamic_lengths(gamma, np.full_like(gamma, 27.0))
        fit = fit_power_law(xi, tau)
        self.assertAlmostEqual(float(fit["exponent"]), 2.0, places=12)

    def test_05_kappa_R12(self) -> None:
        self.assertAlmostEqual(kappa_R_theory(12, 1.0), 27.08333333333333, places=13)

    def test_06_top_hat_q4_coefficients(self) -> None:
        coefficients = top_hat_kernel_expansion(12, 1.0)
        self.assertAlmostEqual(coefficients["eta_R"], 210.7986111111111, places=12)
        self.assertAlmostEqual(coefficients["c4_R"], 155.9548611111111, places=12)

    def test_07_q2_q4_fit_recovers_synthetic_coefficients(self) -> None:
        gamma0 = 0.02
        kappa = 27.08333333333333
        c4 = 155.9548611111111
        q = np.array([0.0, 0.006, 0.012, 0.018, 0.024])
        gamma = gamma0 + kappa * q**2 + c4 * q**4
        fit = fit_q2_q4_dispersion(q, gamma, gamma0)
        self.assertAlmostEqual(float(fit["D2"]), kappa, places=10)
        self.assertAlmostEqual(float(fit["D4"]), c4, places=8)

    def test_08_synthetic_collapse_identity(self) -> None:
        gamma0 = 0.02
        D = 27.0
        q = np.array([0.0, 0.01, 0.02, 0.03])
        gamma = gamma0 + D * q**2
        collapse = build_collapse_values(q, gamma, gamma0, D)
        np.testing.assert_allclose(
            collapse["collapse_y_tau_ratio"],
            collapse["collapse_theory"],
            rtol=0.0,
            atol=2e-16,
        )

    def test_09_primary_window_is_fixed_by_delta_max(self) -> None:
        windows = build_nested_windows(
            self.delta,
            primary_delta_max=3e-4,
            min_window_points=3,
        )
        primary = [window for window in windows if window["primary_window"]]
        self.assertEqual(len(primary), 1)
        self.assertEqual(primary[0]["n_points"], 4)
        self.assertAlmostEqual(primary[0]["delta_max"], 3e-4)

        shifted = build_nested_windows(
            self.delta,
            primary_delta_max=1e-4,
            min_window_points=3,
        )
        shifted_primary = [window for window in shifted if window["primary_window"]]
        self.assertEqual(shifted_primary[0]["n_points"], 3)
        self.assertAlmostEqual(shifted_primary[0]["delta_max"], 1e-4)

    def test_10_missing_required_csv_column_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            phase0 = root / "phase0"
            phase12 = root / "phase12"
            phase0.mkdir()
            phase12.mkdir()
            (phase0 / "phase0_summary.json").write_text(
                json.dumps(
                    {
                        "z_spinodal": -1.0,
                        "sigma_eff": 0.2,
                        "kappa_R_theory": 27.0,
                        "inputs": {"R": 12, "a": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            pd.DataFrame(
                {
                    "delta": [1e-4],
                    "Delta": [0.1],
                    "Gamma0_theory": [0.03],
                    "xi_theory": [30.0],
                }
            ).to_csv(phase0 / "phase0_delta_table.csv", index=False)
            pd.DataFrame(
                {
                    "task_group": ["main"],
                    "delta": [1e-4],
                    "Delta": [0.1],
                    "N": [1024],
                    "R": [12],
                    "a": [1.0],
                    "mode_index": [0],
                    "q": [0.0],
                    "qR": [0.0],
                    "kernel_hat": [1.0],
                    "Gamma0_theory": [0.03],
                    "reliable": [True],
                }
            ).to_csv(phase12 / "phase12_mode_results.csv", index=False)
            pd.DataFrame(
                {
                    "delta": [1e-4],
                    "D_fit": [27.0],
                    "kappa_R_theory": [27.0],
                    "qR_max_fit": [0.35],
                    "n_modes_used": [5],
                }
            ).to_csv(phase12 / "phase12_dispersion_fits.csv", index=False)
            pd.DataFrame(
                {
                    "delta": [1e-4],
                    "mode_index": [0],
                    "q": [0.0],
                    "qR": [0.0],
                    "minus_log_kernel": [0.0],
                }
            ).to_csv(phase12 / "phase12_kernel_relation.csv", index=False)
            (phase12 / "phase12_validation_summary.json").write_text(
                json.dumps({"epsilon_convergence": {"performed": False}}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Gamma_from_lambda"):
                load_phase34_inputs(phase0, phase12)

    def test_11_missing_phase12_outputs_do_not_trigger_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(
                FileNotFoundError,
                MISSING_INPUT_MESSAGE.replace(".", r"\."),
            ):
                load_phase34_inputs(
                    Path(temporary) / "phase0",
                    Path(temporary) / "phase12",
                )


if __name__ == "__main__":
    unittest.main()
