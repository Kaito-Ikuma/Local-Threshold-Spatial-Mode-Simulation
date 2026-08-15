from __future__ import annotations

import tempfile
import sys
import unittest
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from spinodal_phase5_analysis import (
    aggregate_block_series,
    fit_microscopic_relaxation,
)
from spinodal_phase5_core import (
    Phase5Task,
    aggregated_exact_interactions,
    aggregated_exact_step,
    apply_paired_mode_perturbation,
    build_work_units,
    checkpoint_is_valid,
    checkpoint_path,
    direct_J_interactions,
    direct_J_step_reference,
    load_block_checkpoint,
    make_work_unit_rng,
    periodic_neighbor_sum_batch,
    periodic_neighbor_sum_roll_reference,
    save_block_checkpoint,
    simulate_microscopic_block,
)
from spinodal_phase5_mpi import weighted_lpt_assignment


class SpinodalPhase5Tests(unittest.TestCase):
    def make_task(self, **changes) -> Phase5Task:
        task = Phase5Task(
            task_id="task_d000_m001_e00",
            task_group="test",
            delta_index=0,
            epsilon_index=0,
            delta=1e-3,
            Delta=0.154326,
            m_star=-0.794935,
            m_spinodal=-0.760968,
            Gamma_closure=0.11,
            N=32,
            R=3,
            lattice_spacing=1.0,
            mode_index=1,
            epsilon_fraction=0.2,
            M_total=64,
            block_size=16,
            T=8,
            fit_start=0,
            fit_end=5,
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
            base_seed=123456,
        )
        return replace(task, **changes)

    def fixed_pair(self, samples: int = 40000):
        plus_pattern = np.array([-1, 1, 1, -1, 1, -1, -1, 1, -1], dtype=np.int8)
        minus_pattern = np.array([-1, 1, -1, -1, 1, 1, -1, 1, -1], dtype=np.int8)
        return (
            np.repeat(plus_pattern[None, :], samples, axis=0),
            np.repeat(minus_pattern[None, :], samples, axis=0),
        )

    def test_01_periodic_neighbor_sum_matches_roll_reference(self) -> None:
        rng = np.random.default_rng(7)
        states = rng.choice([-1, 1], size=(11, 37)).astype(np.int8)
        for R in (1, 3, 8):
            np.testing.assert_array_equal(
                periodic_neighbor_sum_batch(states, R),
                periodic_neighbor_sum_roll_reference(states, R),
            )

    def sampled_moments(self):
        plus, minus = self.fixed_pair(samples=25000)
        R = 2
        mu = 0.43
        sigma = 1.2
        aggregate_plus, aggregate_minus = aggregated_exact_interactions(
            plus,
            minus,
            R=R,
            mu=mu,
            sigma_J=sigma,
            rng=np.random.Generator(np.random.Philox(100)),
        )
        direct_plus, direct_minus = direct_J_interactions(
            plus,
            minus,
            R=R,
            mu=mu,
            sigma_J=sigma,
            rng=np.random.Generator(np.random.Philox(101)),
        )
        site = 4
        local_plus = periodic_neighbor_sum_batch(plus[:1], R)[0, site] / (2 * R)
        local_minus = periodic_neighbor_sum_batch(minus[:1], R)[0, site] / (2 * R)
        rho = periodic_neighbor_sum_batch((plus * minus)[:1], R)[0, site] / (2 * R)
        return {
            "aggregate_plus": aggregate_plus[:, site],
            "aggregate_minus": aggregate_minus[:, site],
            "direct_plus": direct_plus[:, site],
            "direct_minus": direct_minus[:, site],
            "mean_plus": mu * local_plus,
            "mean_minus": mu * local_minus,
            "variance": sigma**2 / (2 * R),
            "covariance": sigma**2 / (2 * R) * rho,
        }

    def test_02_aggregated_and_direct_means_match_theory(self) -> None:
        values = self.sampled_moments()
        for prefix in ("aggregate", "direct"):
            self.assertAlmostEqual(float(np.mean(values[f"{prefix}_plus"])), values["mean_plus"], delta=0.015)
            self.assertAlmostEqual(float(np.mean(values[f"{prefix}_minus"])), values["mean_minus"], delta=0.015)

    def test_03_aggregated_and_direct_variances_match_theory(self) -> None:
        values = self.sampled_moments()
        for prefix in ("aggregate", "direct"):
            self.assertAlmostEqual(float(np.var(values[f"{prefix}_plus"])), values["variance"], delta=0.016)
            self.assertAlmostEqual(float(np.var(values[f"{prefix}_minus"])), values["variance"], delta=0.016)

    def test_04_aggregated_and_direct_covariance_matches_theory(self) -> None:
        values = self.sampled_moments()
        for prefix in ("aggregate", "direct"):
            covariance = float(
                np.cov(
                    values[f"{prefix}_plus"],
                    values[f"{prefix}_minus"],
                    ddof=0,
                )[0, 1]
            )
            self.assertAlmostEqual(covariance, values["covariance"], delta=0.016)

    def test_05_direct_and_aggregated_one_step_are_statistically_equivalent(self) -> None:
        plus, minus = self.fixed_pair(samples=30000)
        thresholds = np.zeros_like(plus, dtype=float)
        kwargs = dict(R=2, mu=0.35, sigma_J=1.0, h=0.05)
        direct = direct_J_step_reference(
            plus,
            minus,
            thresholds,
            rng=np.random.Generator(np.random.Philox(1)),
            **kwargs,
        )
        aggregated = aggregated_exact_step(
            plus,
            minus,
            thresholds,
            rng=np.random.Generator(np.random.Philox(2)),
            **kwargs,
        )
        for direct_side, aggregate_side in zip(direct, aggregated):
            self.assertLess(
                float(np.max(np.abs(direct_side.mean(axis=0) - aggregate_side.mean(axis=0)))),
                0.025,
            )
        direct_response = 0.5 * (direct[0].mean(axis=0) - direct[1].mean(axis=0))
        aggregate_response = 0.5 * (
            aggregated[0].mean(axis=0) - aggregated[1].mean(axis=0)
        )
        cosine = np.cos(2.0 * np.pi * np.arange(plus.shape[1]) / plus.shape[1])
        direct_amplitude = 2.0 * float(np.dot(direct_response, cosine)) / plus.shape[1]
        aggregate_amplitude = 2.0 * float(np.dot(aggregate_response, cosine)) / plus.shape[1]
        self.assertAlmostEqual(direct_amplitude, aggregate_amplitude, delta=0.015)

    def test_06_small_end_to_end_gamma_statistical_equivalence(self) -> None:
        base = self.make_task(
            N=32,
            R=3,
            M_total=3000,
            block_size=3000,
            T=7,
            fit_end=4,
            m_star=-0.65,
            m_spinodal=-0.2,
            epsilon_fraction=0.15,
            Delta=0.10,
        )
        direct = simulate_microscopic_block(
            build_work_units(replace(base, microscopic_kernel="direct_J"))[0]
        )
        aggregated = simulate_microscopic_block(
            build_work_units(replace(base, microscopic_kernel="aggregated_exact"))[0]
        )
        direct_gamma = float(
            fit_microscopic_relaxation(direct.A_q, 0, 4)["Gamma_micro"]
        )
        aggregated_gamma = float(
            fit_microscopic_relaxation(aggregated.A_q, 0, 4)["Gamma_micro"]
        )
        self.assertTrue(np.isfinite(direct_gamma))
        self.assertTrue(np.isfinite(aggregated_gamma))
        self.assertLess(abs(direct_gamma - aggregated_gamma), 0.12)

    def test_07_quenched_thresholds_are_not_modified(self) -> None:
        rng = np.random.Generator(np.random.Philox(9))
        plus = rng.choice([-1, 1], size=(10, 32)).astype(np.int8)
        minus = rng.choice([-1, 1], size=(10, 32)).astype(np.int8)
        thresholds = rng.normal(size=(10, 32))
        before = thresholds.copy()
        direct_J_step_reference(
            plus,
            minus,
            thresholds,
            R=3,
            mu=0.5,
            sigma_J=1.0,
            h=0.1,
            rng=rng,
        )
        np.testing.assert_array_equal(thresholds, before)

    def test_08_shared_noise_correlation_is_not_independent(self) -> None:
        plus, minus = self.fixed_pair(samples=30000)
        interaction_plus, interaction_minus = aggregated_exact_interactions(
            plus,
            minus,
            R=2,
            mu=0.0,
            sigma_J=1.0,
            rng=np.random.Generator(np.random.Philox(25)),
        )
        measured = float(np.corrcoef(interaction_plus[:, 4], interaction_minus[:, 4])[0, 1])
        rho = periodic_neighbor_sum_batch((plus * minus)[:1], 2)[0, 4] / 4
        self.assertAlmostEqual(measured, rho, delta=0.025)

    def test_09_rng_is_independent_of_rank_and_recreation(self) -> None:
        unit = build_work_units(self.make_task())[2]
        first = make_work_unit_rng(unit).standard_normal(100)
        second = make_work_unit_rng(unit).standard_normal(100)
        np.testing.assert_array_equal(first, second)
        self.assertNotIn("rank", unit.unit_id)

    def test_10_assignment_does_not_change_block_results_or_aggregation(self) -> None:
        task = self.make_task(M_total=32, block_size=8, T=5, fit_end=3)
        units = build_work_units(task)
        serial = [simulate_microscopic_block(unit) for unit in units]
        assignments = weighted_lpt_assignment(units, 3)
        by_id = {result.unit_id: result for result in serial}
        reordered = [
            by_id[unit.unit_id]
            for rank_units in assignments
            for unit in rank_units
        ]
        np.testing.assert_array_equal(
            aggregate_block_series(serial)["A_q"],
            aggregate_block_series(reordered)["A_q"],
        )

    def test_11_checkpoint_and_resume_validation(self) -> None:
        unit = build_work_units(self.make_task(M_total=8, block_size=8, T=4, fit_end=3))[0]
        result = simulate_microscopic_block(unit)
        with tempfile.TemporaryDirectory() as temporary:
            path = checkpoint_path(Path(temporary), unit)
            save_block_checkpoint(result, path)
            self.assertTrue(checkpoint_is_valid(path, unit))
            loaded = load_block_checkpoint(path, unit)
            np.testing.assert_array_equal(loaded.A_q, result.A_q)
            changed_task = replace(unit.task, preparation_width=0.03)
            changed_unit = replace(unit, task=changed_task)
            self.assertFalse(checkpoint_is_valid(path, changed_unit))
            path.write_bytes(b"corrupt")
            self.assertFalse(checkpoint_is_valid(path, unit))

    def test_12_prepared_perturbation_achieves_requested_amplitude(self) -> None:
        rng = np.random.Generator(np.random.Philox(88))
        base = rng.choice([-1, 1], size=(4096, 32), p=[0.8, 0.2]).astype(np.int8)
        _, _, achieved = apply_paired_mode_perturbation(
            base,
            mode_index=2,
            epsilon=0.05,
            lattice_spacing=1.0,
            rng=rng,
        )
        self.assertAlmostEqual(achieved, 0.05, delta=0.004)

    def test_13_block_aggregation_matches_one_shot_weighted_statistics(self) -> None:
        task = self.make_task(M_total=24, block_size=8, T=5, fit_end=3)
        blocks = [simulate_microscopic_block(unit) for unit in build_work_units(task)]
        aggregate = aggregate_block_series(blocks)
        explicit = sum(block.block_n * block.A_q for block in blocks) / sum(
            block.block_n for block in blocks
        )
        np.testing.assert_array_equal(aggregate["A_q"], explicit)

    def test_14_compute_only_path_does_not_import_matplotlib(self) -> None:
        code = f"""
import builtins
import sys
import tempfile
from pathlib import Path

real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == 'matplotlib' or name.startswith('matplotlib.'):
        raise ModuleNotFoundError('matplotlib intentionally unavailable')
    return real_import(name, *args, **kwargs)

builtins.__import__ = blocked_import
sys.path.insert(0, {str(SRC_DIR)!r})
from spinodal_phase5_analysis import Phase5Analysis, write_phase5_analysis
import pandas as pd

empty = pd.DataFrame()
analysis = Phase5Analysis(empty, empty, empty, empty, empty, empty, {{}})
with tempfile.TemporaryDirectory() as directory:
    paths = write_phase5_analysis(analysis, Path(directory), make_figures=False)
    assert 'validation_summary' in paths
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
"""
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
