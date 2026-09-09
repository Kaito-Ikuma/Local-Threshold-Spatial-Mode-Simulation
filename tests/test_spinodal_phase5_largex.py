from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from spinodal_phase5_core import (
    Phase5Task,
    build_work_units,
    checkpoint_path,
    folded_harmonic_mode,
    long_wavelength_q_grid,
    make_work_unit_rng,
    q_grid_signature,
    resume_fingerprint,
    save_block_checkpoint,
    simulate_microscopic_block,
    task_fingerprint,
    work_unit_rng_entropy,
)
from spinodal_phase5_largex_analysis import (
    adaptive_fit_end,
    aggregate_cohorts,
    common_block_bootstrap_gammas,
    estimate_memory,
    origin_D_fit,
    verify_common_randomness,
)


class LargeXTests(unittest.TestCase):
    def task(self, **changes) -> Phase5Task:
        base = Phase5Task(
            task_id="largeX_d000_m000_e00", task_group="test",
            delta_index=0, epsilon_index=0, delta=0.04, Delta=0.08,
            m_star=-0.8, m_spinodal=-0.7, Gamma_closure=0.2,
            N=32, R=3, lattice_spacing=1.0, mode_index=0,
            epsilon_fraction=0.05, M_total=16, block_size=8,
            T=6, fit_start=0, fit_end=3, mu=0.5, sigma_J=1.0,
            sigma_phi=0.06, phi_bar=0.0, branch="stay_to_evacuate",
            initialization_mode="bernoulli_meanfield", track_survival=True,
            base_seed=991,
        )
        return replace(base, **changes)

    def test_q_grid_and_calibration_validation_split(self) -> None:
        rows = long_wavelength_q_grid(4096, 24)
        self.assertEqual([row["mode_index"] for row in rows], list(range(10)))
        self.assertEqual([row["mode_index"] for row in rows if row["set_type"] == "calibration"], [1, 2, 3, 4])
        self.assertEqual([row["mode_index"] for row in rows if row["set_type"] == "validation"], [5, 6, 7, 8, 9])
        self.assertAlmostEqual(rows[-1]["qR"], 0.3313398501832985)
        first = q_grid_signature(rows, N=4096, R=24, lattice_spacing=1.0, calibration_qR_max=0.15, validation_qR_max=0.35)
        second = q_grid_signature(rows, N=4096, R=24, lattice_spacing=1.0, calibration_qR_max=0.15, validation_qR_max=0.35)
        self.assertEqual(first, second)

    def test_independent_entropy_is_historical_and_common_omits_only_mode(self) -> None:
        independent = build_work_units(self.task(mode_index=2))[1]
        self.assertEqual(work_unit_rng_entropy(independent), [991, 0, 2, 0, 1])
        common0 = build_work_units(self.task(rng_coupling_mode="common_modes"))[1]
        common4 = build_work_units(self.task(mode_index=4, task_id="largeX_d000_m004_e00", rng_coupling_mode="common_modes"))[1]
        self.assertEqual(work_unit_rng_entropy(common0), [991, 0, 0, 1])
        self.assertEqual(work_unit_rng_entropy(common0), work_unit_rng_entropy(common4))
        np.testing.assert_array_equal(make_work_unit_rng(common0).random(20), make_work_unit_rng(common4).random(20))

    def test_legacy_fingerprint_is_unchanged_but_largex_settings_are_hashed(self) -> None:
        task = self.task(track_survival=False)
        payload = asdict(task)
        for key in (
            "rng_coupling_mode", "survival_criterion_primary",
            "survival_criterion_sensitivity", "q_set_type", "q_grid_signature",
            "fit_protocol", "campaign_version", "harmonic_orders",
            "survivor_cohort_end",
            "epsilon_linearity_validated",
        ):
            payload.pop(key)
        expected = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        self.assertEqual(task_fingerprint(task), expected)
        large = replace(task, campaign_version="largeX-v1", q_set_type="calibration", q_grid_signature="abc")
        self.assertNotEqual(task_fingerprint(task), task_fingerprint(large))
        self.assertNotEqual(task_fingerprint(large), task_fingerprint(replace(large, mode_index=1)))
        self.assertEqual(resume_fingerprint(large), resume_fingerprint(replace(large, M_total=32)))

    def test_common_modes_share_thresholds_but_checkpoint_names_do_not_collide(self) -> None:
        task0 = self.task(rng_coupling_mode="common_modes", M_total=8)
        task1 = replace(task0, task_id="largeX_d000_m001_e00", mode_index=1)
        unit0, unit1 = build_work_units(task0)[0], build_work_units(task1)[0]
        result0, result1 = simulate_microscopic_block(unit0), simulate_microscopic_block(unit1)
        self.assertEqual(result0.threshold_checksum, result1.threshold_checksum)
        report = verify_common_randomness({0: [result0], 1: [result1]})
        self.assertTrue(report["threshold_and_entropy_match"])
        with tempfile.TemporaryDirectory() as directory:
            path0, path1 = checkpoint_path(Path(directory), unit0), checkpoint_path(Path(directory), unit1)
            self.assertNotEqual(path0, path1)
            save_block_checkpoint(result0, path0); save_block_checkpoint(result1, path1)
            self.assertTrue(path0.is_file() and path1.is_file())

    def test_common_block_bootstrap_preserves_pairing(self) -> None:
        task0 = self.task(rng_coupling_mode="common_modes", M_total=24, block_size=8)
        task1 = replace(task0, task_id="largeX_d000_m001_e00", mode_index=1)
        blocks0 = [simulate_microscopic_block(unit) for unit in build_work_units(task0)]
        blocks1 = [simulate_microscopic_block(unit) for unit in build_work_units(task1)]
        times = np.arange(task0.T + 1)
        for index, block in enumerate(blocks0):
            blocks0[index] = replace(block, A_q=np.exp(-(0.10 + 0.01 * index) * times))
        for index, block in enumerate(blocks1):
            blocks1[index] = replace(block, A_q=np.exp(-(0.20 + 0.01 * index) * times))
        modes, samples, block_ids = common_block_bootstrap_gammas(
            {0: blocks0, 1: blocks1}, fit_end=3, replicates=200, seed=7
        )
        self.assertEqual(modes, [0, 1]); self.assertEqual(block_ids, [0, 1, 2])
        self.assertGreater(np.corrcoef(samples, rowvar=False)[0, 1], 0.99)

    def test_D_X_and_unresolved_sign(self) -> None:
        q = np.array([0.1, 0.2, 0.3])
        gamma0, D = 0.2, 3.0
        gamma = gamma0 + D * q**2
        self.assertAlmostEqual(origin_D_fit(q, gamma, gamma0), D)
        self.assertAlmostEqual(np.max(q * np.sqrt(D / gamma0)), 0.3 * np.sqrt(15.0))
        self.assertLess(origin_D_fit(q, gamma0 - q**2, gamma0), 0.0)

    def test_fixed_survivor_cohort_is_frozen_at_requested_end(self) -> None:
        result = simulate_microscopic_block(
            build_work_units(self.task(M_total=64, block_size=64, survivor_cohort_end=2))[0]
        )
        self.assertEqual(result.survivor_cohort_end, 2)
        self.assertEqual(result.survive_to_T_count, int(result.survival_count[2]))
        aggregate = aggregate_cohorts([result])
        self.assertEqual(aggregate.fixed_survivor_count, int(result.survival_count[2]))

    def test_adaptive_window_harmonic_alias_and_memory(self) -> None:
        amplitude = np.exp(-0.25 * np.arange(21))
        survival = np.ones(21); survival[11:] = 0.85
        self.assertEqual(adaptive_fit_end(amplitude, survival), 10)
        self.assertEqual(folded_harmonic_mode(7, 5, 32), 3)
        estimate = estimate_memory(4096, 32, 50, 76)
        self.assertGreater(estimate["estimated_per_rank_MiB"], 0)
        self.assertLess(estimate["estimated_node_total_GiB"], 10)


if __name__ == "__main__":
    unittest.main()
