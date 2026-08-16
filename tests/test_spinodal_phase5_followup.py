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

from spinodal_phase5_analysis import aggregate_block_series
from spinodal_phase5_core import (
    Phase5Task,
    _escape_fraction,
    _escape_mask,
    build_work_units,
    checkpoint_is_valid,
    legacy_task_fingerprint,
    load_block_checkpoint,
    simulate_microscopic_block,
    task_fingerprint,
)
from spinodal_phase5_followup_analysis import (
    SURVIVAL_RERUN_ERROR,
    aggregate_survival_blocks,
    block_mean_and_se,
    bootstrap_gamma_eff,
    build_gamma_eff_table,
    effective_relaxation,
    interpolate_escape_crossing,
    require_survival,
)


class SpinodalPhase5FollowupTests(unittest.TestCase):
    def make_task(self, **changes) -> Phase5Task:
        task = Phase5Task(
            task_id="task_d000_m001_e00",
            task_group="followup_test",
            delta_index=0,
            epsilon_index=0,
            delta=0.07,
            Delta=0.085,
            m_star=-0.95,
            m_spinodal=-0.2,
            Gamma_closure=0.9,
            N=32,
            R=3,
            lattice_spacing=1.0,
            mode_index=1,
            epsilon_fraction=0.05,
            M_total=32,
            block_size=8,
            T=8,
            fit_start=0,
            fit_end=3,
            mu=0.53,
            sigma_J=1.0,
            sigma_phi=0.06,
            phi_bar=0.0,
            branch="stay_to_evacuate",
            microscopic_kernel="aggregated_exact",
            initialization_mode="bernoulli_meanfield",
            preparation_width=0.02,
            preparation_steps=3,
            burn_steps_per_stage=2,
            base_seed=9876,
        )
        return replace(task, **changes)

    def synthetic_blocks(self, gamma: float = 0.4):
        task = self.make_task(M_total=32, block_size=8)
        blocks = [simulate_microscopic_block(unit) for unit in build_work_units(task)]
        exact = 0.03 * np.exp(-gamma * np.arange(task.T + 1))
        for block in blocks:
            block.A_q = exact.copy()
        return blocks

    def test_01_gamma_eff_recovers_synthetic_exponential(self) -> None:
        gamma = 0.37
        amplitude = 0.2 * np.exp(-gamma * np.arange(9))
        result = effective_relaxation(amplitude)
        np.testing.assert_allclose(result["Gamma_eff"], gamma, rtol=0, atol=1e-14)

    def test_02_sign_flip_is_explicit(self) -> None:
        result = effective_relaxation(np.array([1.0, -0.5, -0.25]))
        self.assertTrue(bool(result["sign_flip"][0]))
        self.assertFalse(bool(result["sign_flip"][1]))

    def test_03_noise_floor_fails_snr_flag(self) -> None:
        blocks = self.synthetic_blocks()
        for index, block in enumerate(blocks):
            block.A_q[2:] = ((-1) ** index) * 1e-10
        table = build_gamma_eff_table(
            {blocks[0].task_id: blocks},
            min_snr=5.0,
            bootstrap_replicates=20,
            bootstrap_seed=1,
        )
        self.assertFalse(bool(table.loc[table["t"] == 2, "snr_reliable"].iloc[0]))

    def test_04_block_bootstrap_recovers_synthetic_gamma(self) -> None:
        blocks = self.synthetic_blocks(gamma=0.42)
        samples = bootstrap_gamma_eff(blocks, replicates=50, seed=4)
        np.testing.assert_allclose(samples, 0.42, atol=1e-13)

    def test_05_escape_mask_matches_legacy_fraction(self) -> None:
        task = self.make_task(m_spinodal=0.0)
        plus = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]], dtype=np.int8)
        minus = plus.copy()
        mask = _escape_mask(plus, minus, task)
        self.assertEqual(float(np.mean(mask)), _escape_fraction(plus, minus, task))

    def test_06_cumulative_escape_is_monotone(self) -> None:
        task = self.make_task(track_survival=True)
        block = simulate_microscopic_block(build_work_units(task)[0])
        self.assertTrue(np.all(np.diff(block.escape_fraction_cumulative) >= -1e-15))

    def test_07_survival_fraction_is_monotone(self) -> None:
        task = self.make_task(track_survival=True)
        block = simulate_microscopic_block(build_work_units(task)[0])
        self.assertTrue(np.all(np.diff(block.survival_fraction_cumulative) <= 1e-15))

    def test_08_survivor_aggregate_uses_sum_over_count(self) -> None:
        task = self.make_task(track_survival=True, M_total=16, block_size=8)
        blocks = [simulate_microscopic_block(unit) for unit in build_work_units(task)]
        blocks[0].survival_count[:] = 2
        blocks[1].survival_count[:] = 6
        blocks[0].survivor_amplitude_sum_current[:] = 2.0
        blocks[1].survivor_amplitude_sum_current[:] = 12.0
        aggregate = aggregate_survival_blocks(blocks)
        np.testing.assert_allclose(aggregate["A_surviving_current"], 14.0 / 8.0)

    def test_09_fixed_final_survivor_cohort_uses_saved_numerator(self) -> None:
        task = self.make_task(track_survival=True, M_total=16, block_size=8)
        blocks = [simulate_microscopic_block(unit) for unit in build_work_units(task)]
        blocks[0].survive_to_T_count = 3
        blocks[1].survive_to_T_count = 5
        blocks[0].survive_to_T_amplitude_sum[:] = 3.0
        blocks[1].survive_to_T_amplitude_sum[:] = 10.0
        aggregate = aggregate_survival_blocks(blocks)
        np.testing.assert_allclose(aggregate["A_survive_to_T"], 13.0 / 8.0)

    def test_10_no_escape_survivor_response_equals_unconditional(self) -> None:
        task = self.make_task(track_survival=True, m_star=-0.7, m_spinodal=1.0)
        blocks = [simulate_microscopic_block(unit) for unit in build_work_units(task)]
        aggregate = aggregate_survival_blocks(blocks)
        np.testing.assert_allclose(
            aggregate["A_surviving_current"], aggregate["A_unconditional"], atol=1e-15
        )

    def _write_legacy_checkpoint(self, path: Path, task: Phase5Task):
        unit = build_work_units(task)[0]
        result = simulate_microscopic_block(unit)
        metadata = {
            key: value
            for key, value in asdict(result).items()
            if not isinstance(value, np.ndarray)
        }
        for key in (
            "checkpoint_schema_version",
            "survival_tracking_enabled",
            "survive_to_T_count",
        ):
            metadata.pop(key)
        metadata["task_fingerprint"] = legacy_task_fingerprint(task)
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
        return unit

    def test_11_legacy_v1_loads_for_normal_analysis(self) -> None:
        task = self.make_task(M_total=8, block_size=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            unit = self._write_legacy_checkpoint(path, task)
            self.assertTrue(checkpoint_is_valid(path, unit))
            loaded = load_block_checkpoint(path, unit)
            self.assertEqual(loaded.checkpoint_schema_version, 1)
            self.assertEqual(len(aggregate_block_series([loaded])["A_q"]), task.T + 1)

    def test_12_legacy_survival_request_has_clear_error(self) -> None:
        task = self.make_task(M_total=8, block_size=8)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.npz"
            unit = self._write_legacy_checkpoint(path, task)
            loaded = load_block_checkpoint(path, unit)
            with self.assertRaisesRegex(ValueError, "rerun with survival tracking enabled"):
                require_survival([loaded])

    def test_13_burn_protocol_changes_fingerprint(self) -> None:
        self.assertNotEqual(
            task_fingerprint(self.make_task(burn_steps_per_stage=8)),
            task_fingerprint(self.make_task(burn_steps_per_stage=16)),
        )

    def test_14_interpolation_recovers_ten_percent_crossing(self) -> None:
        result = interpolate_escape_crossing([0.06, 0.07], [0.2, 0.0], 0.1)
        self.assertAlmostEqual(result["estimate"], 0.065)
        self.assertTrue(result["monotonicity_ok"])

    def test_15_unbracketed_crossing_is_null(self) -> None:
        result = interpolate_escape_crossing([0.06, 0.07], [0.3, 0.2], 0.1)
        self.assertIsNone(result["estimate"])

    def test_16_unperturbed_pair_stays_identical(self) -> None:
        task = self.make_task(
            epsilon_fraction=0.0,
            unperturbed=True,
            track_survival=True,
            M_total=8,
            block_size=8,
        )
        block = simulate_microscopic_block(build_work_units(task)[0])
        np.testing.assert_array_equal(block.mean_m_plus, block.mean_m_minus)
        np.testing.assert_array_equal(block.A_q, np.zeros(task.T + 1))


if __name__ == "__main__":
    unittest.main()
