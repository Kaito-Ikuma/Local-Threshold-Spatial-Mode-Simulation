from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from spinodal_gaussian_R_sweep import (
    build_R_N_mapping,
    qR_for_mode,
    validate_common_qR,
)
from spinodal_phase0 import Phase0Task, kappa_R_theory, run_phase0_case
from spinodal_phase34 import compute_dynamic_lengths, fit_power_law
from spinodal_phase5_analysis import kernel_hat


class GaussianRangeSweepTests(unittest.TestCase):
    def test_01_primary_N_mapping_keeps_N_over_R_constant(self) -> None:
        mapping = build_R_N_mapping((6, 12, 24, 48))
        self.assertEqual(mapping, {6: 512, 12: 1024, 24: 2048, 48: 4096})
        self.assertEqual({N / R for R, N in mapping.items()}, {1024 / 12})

    def test_02_same_mode_has_same_qR(self) -> None:
        mapping = build_R_N_mapping((6, 12, 24, 48))
        grid = validate_common_qR(mapping, range(7))
        for mode in range(7):
            values = [grid[R][mode] for R in mapping]
            self.assertLess(np.ptp(values), 1e-14)
            self.assertAlmostEqual(values[0], qR_for_mode(mode, 512, 6), places=14)

    def test_03_kappa_formula_for_every_primary_R(self) -> None:
        for R in (6, 12, 24, 48):
            expected = (R + 1) * (2 * R + 1) / 12.0
            self.assertAlmostEqual(kappa_R_theory(R, 1.0), expected, places=14)

    def test_04_exact_kernel_longwave_slope_approaches_kappa(self) -> None:
        for R in (6, 12, 24, 48):
            N = 2_000_000
            modes = np.arange(0, 7)
            q = 2.0 * math.pi * modes / N
            gamma = np.array([-math.log(kernel_hat(int(mode), N, R)) for mode in modes])
            design = np.column_stack((np.ones_like(q), q**2))
            slope = float(np.linalg.lstsq(design, gamma, rcond=None)[0][1])
            self.assertLess(abs(slope / kappa_R_theory(R, 1.0) - 1.0), 2e-7)

    def test_05_synthetic_gamma_exponent_is_R_independent(self) -> None:
        delta = np.array([3e-4, 1e-4, 3e-5, 1e-5])
        for R in (6, 12, 24, 48):
            amplitude = 1.0 + R / 10.0
            fit = fit_power_law(delta, amplitude * np.sqrt(delta))
            self.assertAlmostEqual(float(fit["exponent"]), 0.5, places=13)

    def test_06_synthetic_xi_exponent_and_z(self) -> None:
        delta = np.array([3e-4, 1e-4, 3e-5, 1e-5])
        for R in (6, 12, 24, 48):
            gamma = 2.5 * np.sqrt(delta)
            xi, _ = compute_dynamic_lengths(
                gamma, np.full_like(gamma, kappa_R_theory(R, 1.0))
            )
            xi_fit = fit_power_law(delta, xi)
            z_fit = fit_power_law(xi, 1.0 / gamma)
            self.assertAlmostEqual(float(xi_fit["exponent"]), -0.25, places=13)
            self.assertAlmostEqual(float(z_fit["exponent"]), 2.0, places=13)

    def test_07_R12_phase0_baseline_is_unchanged(self) -> None:
        result = run_phase0_case(Phase0Task(R=12, delta_list=(1e-3,)))
        self.assertAlmostEqual(result.spinodal.sigma_eff, 0.2127596452964393, places=14)
        self.assertAlmostEqual(result.spinodal.mu, 0.5333093426005172, places=14)
        self.assertAlmostEqual(result.spinodal.m_spinodal, -0.7609681085504881, places=14)
        self.assertAlmostEqual(result.spinodal.kappa_R_theory, 27.083333333333332, places=13)


if __name__ == "__main__":
    unittest.main()
