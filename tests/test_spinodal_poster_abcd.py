from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spinodal_poster_abcd import fit_matched_power_law, fit_measured_dispersion


class PosterABCDAnalysisTests(unittest.TestCase):
    def test_result_a_synthetic_half_exponent(self) -> None:
        matched = np.array([0.005, 0.0075, 0.010, 0.015, 0.020, 0.030, 0.040])
        gamma = 1.7 * matched**0.5
        fit = fit_matched_power_law(matched, gamma)
        self.assertAlmostEqual(float(fit["amplitude"]), 1.7, places=12)
        self.assertAlmostEqual(float(fit["exponent"]), 0.5, places=12)

    def test_result_b_synthetic_dispersion(self) -> None:
        q = np.array([0.0, 0.01, 0.02, 0.03, 0.04])
        gamma0 = 0.08
        D = 31.25
        gamma = gamma0 + D * q**2
        fit = fit_measured_dispersion(q, gamma)
        self.assertAlmostEqual(float(fit["Gamma0"]), gamma0, places=12)
        self.assertAlmostEqual(float(fit["D"]), D, places=10)
        self.assertAlmostEqual(float(fit["r2"]), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
