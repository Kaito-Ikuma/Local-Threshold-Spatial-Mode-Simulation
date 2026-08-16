from __future__ import annotations

import tempfile
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from spinodal_R_sweep_analysis import (
    build_combined_summary,
    classify_response_coordinate,
    fine_grid_from_bracket,
    plan_fine_scan,
)
from spinodal_gaussian_R_sweep import build_R_N_mapping, qR_for_mode
from spinodal_phase5_core import (
    build_work_units,
    checkpoint_is_valid,
    save_block_checkpoint,
    simulate_microscopic_block,
    task_fingerprint,
)
from spinodal_phase5_mpi import (
    build_analytic_phase5_references,
    build_phase5_tasks,
)


def make_task(R: int, N: int, prefix: str):
    summary, phase0, modes, _ = build_analytic_phase5_references(
        deltas=(0.08,),
        modes=(0, 1),
        N=N,
        B=2.0,
        R=R,
        sigma_J=1.0,
        sigma_phi=0.06,
        phi_bar=0.0,
        lattice_spacing=1.0,
        branch="stay_to_evacuate",
        qR_max_fit=2.0,
    )
    return build_phase5_tasks(
        summary,
        phase0,
        modes,
        deltas=(0.08,),
        modes=(0,),
        epsilon_fractions=(0.05,),
        N=N,
        M_total=1,
        block_size=1,
        kernel="aggregated_exact",
        initialization="prepared_metastable",
        T_min=2,
        tau_multiplier=1.0,
        T_fixed=2,
        fit_start=0,
        fit_end=2,
        preparation_width=0.02,
        preparation_steps=1,
        burn_steps_per_stage=0,
        float_dtype="float64",
        base_seed=123,
        stage="pilot",
        save_structure_factor=False,
        track_survival=True,
        task_id_prefix=prefix,
    )[0]


class MicroscopicRangeSweepTests(unittest.TestCase):
    def test_01_range_is_in_task_id_and_fingerprint(self) -> None:
        task6 = make_task(6, 32, "R006_")
        task12 = make_task(12, 64, "R012_")
        self.assertTrue(task6.task_id.startswith("R006_"))
        self.assertTrue(task12.task_id.startswith("R012_"))
        self.assertNotEqual(task_fingerprint(task6), task_fingerprint(task12))

    def test_02_wrong_range_checkpoint_is_rejected(self) -> None:
        task6 = make_task(6, 32, "")
        task12 = make_task(12, 64, "")
        unit6 = build_work_units(task6)[0]
        unit12 = build_work_units(task12)[0]
        self.assertEqual(unit6.unit_id, unit12.unit_id)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "block.npz"
            save_block_checkpoint(simulate_microscopic_block(unit6), path)
            self.assertFalse(checkpoint_is_valid(path, unit12))

    def test_03_analytic_reference_uses_requested_range(self) -> None:
        summary6, _, modes6, dispersion6 = build_analytic_phase5_references(
            deltas=(0.08,), modes=(0, 1), N=512, B=2.0, R=6,
            sigma_J=1.0, sigma_phi=0.06, phi_bar=0.0,
            lattice_spacing=1.0, branch="stay_to_evacuate", qR_max_fit=0.35,
        )
        summary48, _, modes48, dispersion48 = build_analytic_phase5_references(
            deltas=(0.08,), modes=(0, 1), N=4096, B=2.0, R=48,
            sigma_J=1.0, sigma_phi=0.06, phi_bar=0.0,
            lattice_spacing=1.0, branch="stay_to_evacuate", qR_max_fit=0.35,
        )
        self.assertEqual(summary6["inputs"]["R"], 6)
        self.assertEqual(summary48["inputs"]["R"], 48)
        self.assertNotEqual(summary6["mu"], summary48["mu"])
        self.assertNotEqual(dispersion6["kappa_R_theory"].iloc[0], dispersion48["kappa_R_theory"].iloc[0])
        self.assertAlmostEqual(modes6["qR"].iloc[1], modes48["qR"].iloc[1], places=14)

    def test_04_primary_R_N_mapping_and_qR(self) -> None:
        mapping = build_R_N_mapping((6, 12, 24, 48))
        qR = [qR_for_mode(4, N, R) for R, N in mapping.items()]
        self.assertEqual(mapping[12], 1024)
        self.assertLess(np.ptp(qR), 1e-14)

    def test_05_pseudospinodal_interpolation_and_fine_grid(self) -> None:
        plan = plan_fine_scan(
            (0.04, 0.05, 0.06, 0.07),
            (0.40, 0.20, 0.08, 0.03),
        )
        self.assertEqual(plan["status"], "bracketed")
        self.assertEqual(plan["delta_lower"], 0.05)
        self.assertEqual(plan["delta_upper"], 0.06)
        grid = fine_grid_from_bracket(0.05, 0.06)
        self.assertLessEqual(max(np.diff(grid)), 0.002 + 1e-15)

    def test_06_unbracketed_scan_has_fixed_extension_rule(self) -> None:
        low = plan_fine_scan((0.04, 0.05), (0.05, 0.02))
        high = plan_fine_scan((0.12, 0.16), (0.30, 0.20))
        self.assertEqual(low["extension_direction"], "smaller_delta")
        self.assertAlmostEqual(low["extension_delta"], 0.04 / 1.5)
        self.assertEqual(high["extension_direction"], "larger_delta")
        self.assertAlmostEqual(high["extension_delta"], 0.16 * 1.5)

    def test_07_fixed_and_matched_coordinates_cannot_be_mixed(self) -> None:
        fixed = classify_response_coordinate((0.08, 0.10, 0.12), delta_ps=0.06)
        matched = classify_response_coordinate((0.065, 0.070, 0.080), delta_ps=0.06)
        self.assertEqual(fixed, "fixed_gaussian_delta")
        self.assertEqual(matched, "operational_pseudospinodal_matched")
        with self.assertRaisesRegex(ValueError, "neither fixed nor operational"):
            classify_response_coordinate((0.08, 0.09, 0.12), delta_ps=0.06)

    def test_08_partial_combination_keeps_gaussian_rows(self) -> None:
        gaussian = pd.DataFrame(
            {
                "R": [12], "N": [1024], "N_over_R": [1024 / 12],
                "sigma_eff": [0.2], "mu": [0.5], "kappa_R": [27.0],
                "p_Gamma": [0.5], "p_xi": [-0.25], "z": [2.0],
                "D_over_kappa_nearest": [1.0],
            }
        )
        combined = build_combined_summary(
            gaussian, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        )
        self.assertEqual(len(combined), 1)
        self.assertEqual(int(combined["R"].iloc[0]), 12)
        self.assertTrue(np.isnan(combined["micro_delta_ps_T50"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
