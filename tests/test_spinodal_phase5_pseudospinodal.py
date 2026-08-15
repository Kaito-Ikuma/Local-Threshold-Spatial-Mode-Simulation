from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from spinodal_phase5_core import build_work_units, simulate_microscopic_block
from spinodal_phase5_pseudospinodal_mpi import (
    aggregate_scan_results,
    build_parser,
    build_scan_tasks,
    gaussian_theory_points,
)


class Phase5PseudospinodalTests(unittest.TestCase):
    def test_theory_matches_phase0_B2_R12_reference(self) -> None:
        point = gaussian_theory_points(
            (1e-3,),
            B=2.0,
            R=12,
            sigma_J=1.0,
            sigma_phi=0.06,
            branch="stay_to_evacuate",
        )[0]
        self.assertAlmostEqual(point.mu, 0.5333093426005172, places=14)
        self.assertAlmostEqual(point.m_spinodal, -0.7609681085504881, places=14)
        self.assertAlmostEqual(point.Delta, 0.15432606295215484, places=14)
        self.assertAlmostEqual(point.m_star, -0.7949349292162027, places=12)

    def test_small_scan_aggregates_every_trial(self) -> None:
        args = build_parser().parse_args(
            [
                "--deltas",
                "0.01,0.03",
                "--N",
                "32",
                "--R",
                "3",
                "--M-total",
                "8",
                "--block-size",
                "4",
                "--kernel",
                "aggregated_exact",
                "--T-fixed",
                "2",
                "--preparation-steps",
                "2",
                "--burn-steps-per-stage",
                "1",
            ]
        )
        tasks = build_scan_tasks(args)
        results = [
            simulate_microscopic_block(unit)
            for task in tasks
            for unit in build_work_units(task)
        ]
        rows, detail = aggregate_scan_results(tasks, results)
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(detail), 2)
        self.assertTrue(all(row["M_total"] == 8 for row in rows))
        self.assertTrue(all(row["n_blocks"] == 2 for row in rows))
        self.assertTrue(
            all(0.0 <= row["escape_t0"] <= 1.0 for row in rows)
        )
        self.assertTrue(
            all("preparation_m_stage_2" in row for row in rows)
        )

    def test_module_does_not_import_matplotlib(self) -> None:
        code = f"""
import builtins
import sys

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('matplotlib intentionally unavailable')
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
sys.path.insert(0, {str(SRC_DIR)!r})
import spinodal_phase5_pseudospinodal_mpi
assert not any(
    name == 'matplotlib' or name.startswith('matplotlib.')
    for name in sys.modules
)
"""
        subprocess.run(
            [sys.executable, "-c", code],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
