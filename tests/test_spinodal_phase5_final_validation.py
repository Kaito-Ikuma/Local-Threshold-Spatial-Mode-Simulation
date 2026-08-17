from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from spinodal_phase5_core import (
    Phase5Task,
    build_work_units,
    checkpoint_is_valid,
    legacy_task_fingerprint,
    load_block_checkpoint,
    save_block_checkpoint,
    simulate_microscopic_block,
    task_fingerprint,
)
from spinodal_phase5_final_validation import (
    D_PRECISION_R,
    FINITE_SIZE_PAIRS,
    SEEDS,
    D_production_decision,
    D_uncertainty_transform,
    aggregate_seed_rows,
    required_condition_report,
    time_extension_decision,
    validate_R96_mapping,
)


class Phase5FinalValidationTests(unittest.TestCase):
    def make_task(self, **changes) -> Phase5Task:
        task = Phase5Task(
            task_id="V_task_d000_m000_e00",
            task_group="validation_test",
            delta_index=0,
            epsilon_index=0,
            delta=0.08,
            Delta=0.10,
            m_star=-0.8,
            m_spinodal=-0.2,
            Gamma_closure=0.5,
            N=32,
            R=3,
            lattice_spacing=1.0,
            mode_index=0,
            epsilon_fraction=0.05,
            M_total=8,
            block_size=4,
            T=4,
            fit_start=0,
            fit_end=3,
            mu=0.5,
            sigma_J=1.0,
            sigma_phi=0.06,
            phi_bar=0.0,
            branch="stay_to_evacuate",
            microscopic_kernel="aggregated_exact",
            initialization_mode="bernoulli_meanfield",
            preparation_width=0.02,
            preparation_steps=2,
            burn_steps_per_stage=1,
            base_seed=20260815,
            track_survival=True,
        )
        return replace(task, **changes)

    def test_01_finite_size_task_fingerprint_contains_N(self) -> None:
        first = self.make_task(N=32, R=3)
        second = self.make_task(N=64, R=3)
        self.assertNotEqual(task_fingerprint(first), task_fingerprint(second))

    def test_02_same_R_different_N_checkpoint_is_rejected(self) -> None:
        first = build_work_units(self.make_task(N=32))[0]
        second_task = self.make_task(N=64)
        second = replace(build_work_units(second_task)[0], unit_id=first.unit_id)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.npz"
            save_block_checkpoint(simulate_microscopic_block(first), path)
            self.assertFalse(checkpoint_is_valid(path, second))

    def test_03_required_finite_size_pair_completeness(self) -> None:
        report = required_condition_report(FINITE_SIZE_PAIRS, FINITE_SIZE_PAIRS[:-1])
        self.assertFalse(report["complete"])
        self.assertEqual(report["missing"], [FINITE_SIZE_PAIRS[-1]])

    def test_04_M_extension_reuses_existing_blocks(self) -> None:
        initial = build_work_units(self.make_task(M_total=8))[0]
        extended = build_work_units(self.make_task(M_total=16))[0]
        self.assertEqual(initial.unit_id, extended.unit_id)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.npz"
            save_block_checkpoint(simulate_microscopic_block(initial), path)
            self.assertTrue(checkpoint_is_valid(path, extended))

            # Existing M6 schema-v2 blocks predate resume_fingerprint.  They
            # remain reusable when only the block-aligned M_total increases.
            with np.load(path, allow_pickle=False) as archive:
                payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
            metadata = json.loads(str(payload["metadata_json"].item()))
            metadata.pop("resume_fingerprint")
            payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
            np.savez_compressed(path, **payload)
            self.assertTrue(checkpoint_is_valid(path, extended))

    def test_05_D_over_kappa_uncertainty_transformation(self) -> None:
        result = D_uncertainty_transform(30.0, 3.0, 24.0, 36.0, 20.0)
        self.assertAlmostEqual(result["D_over_kappa"], 1.5)
        self.assertAlmostEqual(result["D_over_kappa_SE"], 0.15)
        self.assertAlmostEqual(result["D_over_kappa_CI_low"], 1.2)
        self.assertAlmostEqual(result["D_over_kappa_CI_high"], 1.8)
        self.assertAlmostEqual(result["z_from_unity"], 0.5 / 0.15)

        needs_more = D_production_decision(32768, 0.337, True)
        self.assertFalse(needs_more["production_M_sufficient"])
        self.assertTrue(needs_more["needs_M65536"])
        self.assertEqual(needs_more["production_precision_status"], "M65536_escalation_required")

        target_met = D_production_decision(32768, 0.212, True)
        self.assertTrue(target_met["production_M_sufficient"])
        self.assertFalse(target_met["needs_M65536"])

        maximum_reached = D_production_decision(65536, 0.30, True)
        self.assertFalse(maximum_reached["production_M_sufficient"])
        self.assertTrue(maximum_reached["production_run_finalized"])
        self.assertFalse(maximum_reached["precision_target_met"])
        self.assertEqual(
            maximum_reached["production_precision_status"],
            "maximum_M_reached_precision_target_not_met",
        )

    def test_06_time_extension_plans_only_unbracketed_scan(self) -> None:
        decision = time_extension_decision((0.04, 0.05), (0.05, 0.02))
        self.assertEqual(decision["status"], "extension_required")
        self.assertAlmostEqual(decision["suggested_new_deltas"][0], 0.04 / 1.25)

    def test_07_bracketed_time_scan_has_no_new_delta(self) -> None:
        decision = time_extension_decision((0.04, 0.05), (0.2, 0.02))
        self.assertEqual(decision["status"], "bracketed")
        self.assertEqual(decision["suggested_new_deltas"], ())

    def test_08_R96_mapping_is_8192(self) -> None:
        mapping = validate_R96_mapping()
        self.assertEqual(mapping["N"], 8192)
        self.assertTrue(mapping["N_over_R_consistent"])

    def test_09_R96_qR_is_consistent(self) -> None:
        self.assertTrue(validate_R96_mapping()["qR_consistent"])

    def test_10_independent_seed_changes_fingerprint(self) -> None:
        self.assertNotEqual(
            task_fingerprint(self.make_task(base_seed=SEEDS[0])),
            task_fingerprint(self.make_task(base_seed=SEEDS[1])),
        )

    def test_11_different_seed_checkpoint_is_rejected(self) -> None:
        first = build_work_units(self.make_task(base_seed=SEEDS[0]))[0]
        second_task = self.make_task(base_seed=SEEDS[1])
        second = replace(build_work_units(second_task)[0], unit_id=first.unit_id)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "block.npz"
            save_block_checkpoint(simulate_microscopic_block(first), path)
            self.assertFalse(checkpoint_is_valid(path, second))

    def test_12_seed_aggregation(self) -> None:
        import pandas as pd

        rows = pd.DataFrame(
            {
                "R": [12, 12, 12],
                "seed": list(SEEDS),
                "Gamma_ratio": [0.98, 1.00, 1.02],
                "Gamma_ratio_ci_low": [0.95, 0.97, 0.99],
                "Gamma_ratio_ci_high": [1.01, 1.03, 1.05],
            }
        )
        result = aggregate_seed_rows(rows)
        self.assertAlmostEqual(result["between_seed_mean"].iloc[0], 1.0)
        self.assertTrue(bool(result["seed_reproducible_soft"].iloc[0]))

    def test_13_required_R_completeness(self) -> None:
        report = required_condition_report(D_PRECISION_R, (12, 24))
        self.assertEqual(report["missing"], [48])

    def test_14_legacy_checkpoint_still_loads(self) -> None:
        task = self.make_task(M_total=4, block_size=4, track_survival=False)
        unit = build_work_units(task)[0]
        result = simulate_microscopic_block(unit)
        metadata = {key: value for key, value in asdict(result).items() if not isinstance(value, np.ndarray)}
        metadata.pop("checkpoint_schema_version")
        metadata.pop("survival_tracking_enabled")
        metadata.pop("survive_to_T_count")
        metadata.pop("resume_fingerprint")
        metadata["task_fingerprint"] = legacy_task_fingerprint(task)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            np.savez_compressed(
                path,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
                A_q=result.A_q,
                mean_m_plus=result.mean_m_plus,
                mean_m_minus=result.mean_m_minus,
                baseline_m=result.baseline_m,
                escape_fraction=result.escape_fraction,
                preparation_magnetization=result.preparation_magnetization,
                structure_factor=result.structure_factor,
            )
            loaded = load_block_checkpoint(path, unit)
            self.assertEqual(loaded.checkpoint_schema_version, 1)


if __name__ == "__main__":
    unittest.main()
