from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from spatial_mode_ensemble_validation import Config
from spinodal_phase0 import (
    DEFAULT_DELTAS,
    Phase0Task,
    compute_spinodal,
    run_phase0_case,
)


class SpinodalPhase0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run_phase0_case(Phase0Task())

    def test_01_B2_R12_benchmark(self) -> None:
        spinodal = self.result.spinodal
        self.assertAlmostEqual(spinodal.sigma_eff, 0.2127596452964393, places=14)
        self.assertAlmostEqual(spinodal.mu, 0.5333093426005172, places=14)
        self.assertAlmostEqual(spinodal.z_spinodal, -1.1774100225154747, places=14)
        self.assertAlmostEqual(spinodal.m_spinodal, -0.760968108550488, places=14)
        self.assertAlmostEqual(spinodal.Delta_spinodal, 0.1553260629521548, places=14)

    def test_02_spinodal_fixed_point_residual(self) -> None:
        self.assertLess(abs(self.result.spinodal.fixed_point_residual), 1e-12)

    def test_03_spinodal_tangency_residual(self) -> None:
        self.assertLess(abs(self.result.spinodal.tangency_residual), 1e-12)

    def test_04_B_at_or_below_one_is_rejected(self) -> None:
        for B in (1.0, 0.72, 0.0):
            with self.subTest(B=B):
                with self.assertRaisesRegex(
                    ValueError,
                    "No spinodal exists for B <= 1 in this mean-field map",
                ):
                    compute_spinodal(Config(B=B), "stay_to_evacuate")

    def test_05_spinodal_branches_are_symmetric(self) -> None:
        config = Config(R=12, B=2.0, sigma_J=1.0, sigma_phi=0.06)
        negative = compute_spinodal(config, "stay_to_evacuate")
        positive = compute_spinodal(config, "evacuate_to_stay")
        self.assertAlmostEqual(positive.m_spinodal, -negative.m_spinodal, places=14)
        self.assertAlmostEqual(
            positive.Delta_spinodal,
            -negative.Delta_spinodal,
            places=14,
        )
        stay_case = run_phase0_case(
            Phase0Task(branch="stay_to_evacuate", delta_list=(1e-3,))
        )
        evacuate_case = run_phase0_case(
            Phase0Task(branch="evacuate_to_stay", delta_list=(1e-3,))
        )
        stay_row = stay_case.delta_table.iloc[0]
        evacuate_row = evacuate_case.delta_table.iloc[0]
        self.assertAlmostEqual(evacuate_row["m_star"], -stay_row["m_star"], places=13)
        self.assertAlmostEqual(evacuate_row["Delta"], -stay_row["Delta"], places=14)
        self.assertAlmostEqual(
            evacuate_row["Lambda_star"],
            stay_row["Lambda_star"],
            places=13,
        )

    def test_06_stay_metastable_points_are_stable(self) -> None:
        table = self.result.delta_table
        self.assertEqual(tuple(table["delta"]), DEFAULT_DELTAS)
        self.assertTrue(table["stable"].all())
        self.assertTrue(((table["Lambda_star"] > 0.0) & (table["Lambda_star"] < 1.0)).all())
        self.assertTrue((table["m_star"] <= self.result.spinodal.m_spinodal).all())

    def test_07_theory_scales_diverge_toward_spinodal(self) -> None:
        table = self.result.delta_table.sort_values("delta", ascending=False)
        far = table.iloc[0]
        near = table.iloc[-1]
        self.assertGreater(near["Lambda_star"], far["Lambda_star"])
        self.assertLess(near["Gamma0_theory"], far["Gamma0_theory"])
        self.assertGreater(near["tau0_theory"], far["tau0_theory"])
        self.assertGreater(near["xi_theory"], far["xi_theory"])

    def test_08_import_and_calculation_do_not_require_mpi4py(self) -> None:
        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing_path if existing_path else "")
        with tempfile.TemporaryDirectory() as cache_dir:
            env["MPLCONFIGDIR"] = cache_dir
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "import sys; "
                        "sys.modules['mpi4py'] = None; "
                        "from spinodal_phase0 import Phase0Task, run_phase0_case; "
                        "assert run_phase0_case(Phase0Task()).spinodal.B == 2.0"
                    ),
                ],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_09_existing_default_B_is_unchanged(self) -> None:
        self.assertEqual(Config().B, 0.72)


if __name__ == "__main__":
    unittest.main()
