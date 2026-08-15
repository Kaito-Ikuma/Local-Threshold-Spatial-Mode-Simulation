from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from spinodal_phase0 import Phase0Task, kappa_R_theory, run_phase0_case
from spinodal_phase12 import (
    Phase0Reference,
    build_phase12_tasks,
    deterministic_closure_step,
    ensure_phase0_reference,
    simulate_deterministic_mode,
)
from spinodal_phase12_mpi import aggregate_task_payloads, run_task_subset


class SpinodalPhase12Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        phase0 = run_phase0_case(Phase0Task(delta_list=(1e-3, 1e-4)))
        summary = {
            "inputs": {
                "B": phase0.task.B,
                "R": phase0.task.R,
                "sigma_J": phase0.task.sigma_J,
                "sigma_phi": phase0.task.sigma_phi,
                "phi_bar": phase0.task.phi_bar,
                "a": phase0.task.lattice_spacing,
                "branch": phase0.task.branch,
            },
            **asdict(phase0.spinodal),
        }
        cls.reference = Phase0Reference(
            phase0_dir=Path("."),
            summary=summary,
            delta_table=phase0.delta_table,
            regenerated=False,
        )

    def make_task(
        self,
        mode: int,
        epsilon_fraction: float = 1e-3,
        delta: float = 1e-3,
        N: int = 256,
    ):
        return build_phase12_tasks(
            self.reference,
            deltas=(delta,),
            modes=(mode,),
            N=N,
            epsilon_fraction=epsilon_fraction,
            T_fixed=20,
            fit_end_fixed=10,
        )[0]

    def test_01_uniform_fixed_point_is_preserved(self) -> None:
        task = self.make_task(mode=0)
        uniform = np.full(task.N, task.m_star)
        updated = deterministic_closure_step(
            uniform,
            R=task.R,
            mu=task.mu,
            Delta=task.Delta,
            sigma_eff=task.sigma_eff,
        )
        self.assertLess(float(np.max(np.abs(updated - uniform))), 2e-14)

    def test_02_q0_eigenvalue_matches_Lambda_star(self) -> None:
        result = simulate_deterministic_mode(self.make_task(mode=0))
        error = abs(result.metrics["lambda_1step"] - result.metrics["Lambda_star"])
        self.assertLess(error, 2e-9)

    def test_03_finite_q_eigenvalue_matches_theory(self) -> None:
        result = simulate_deterministic_mode(self.make_task(mode=1))
        error = abs(result.metrics["lambda_1step"] - result.metrics["lambda_theory"])
        self.assertLess(error, 2e-9)

    def test_04_epsilon_halving_converges(self) -> None:
        errors = []
        for fraction in (0.1, 0.05, 0.025):
            result = simulate_deterministic_mode(
                self.make_task(mode=1, epsilon_fraction=fraction)
            )
            errors.append(
                abs(result.metrics["lambda_1step"] - result.metrics["lambda_theory"])
            )
        self.assertLess(errors[-1], errors[0])
        self.assertLessEqual(errors[1], 1.05 * errors[0])

    def test_05_exact_Gamma_kernel_relation(self) -> None:
        q0 = simulate_deterministic_mode(self.make_task(mode=0))
        q1 = simulate_deterministic_mode(self.make_task(mode=1))
        simulated = q1.metrics["Gamma_from_lambda"] - q0.metrics["Gamma_from_lambda"]
        theory = -np.log(abs(q1.metrics["kernel_hat"]))
        self.assertLess(abs(simulated - theory), 2e-8)

    def test_06_kappa_R12(self) -> None:
        self.assertAlmostEqual(kappa_R_theory(12, 1.0), 27.08333333333333, places=13)

    def test_07_round_robin_aggregation_matches_serial(self) -> None:
        tasks = build_phase12_tasks(
            self.reference,
            deltas=(1e-3, 1e-4),
            modes=(0, 1, 2),
            N=128,
            epsilon_fraction=1e-3,
            T_fixed=15,
            fit_end_fixed=10,
        )
        serial = aggregate_task_payloads(
            [run_task_subset(tasks, rank=0, world_size=1)],
            expected_count=len(tasks),
        )
        round_robin = aggregate_task_payloads(
            [
                run_task_subset(tasks, rank=0, world_size=2),
                run_task_subset(tasks, rank=1, world_size=2),
            ],
            expected_count=len(tasks),
        )
        self.assertEqual(len(serial), len(round_robin))
        for serial_result, mpi_result in zip(serial, round_robin):
            self.assertEqual(serial_result.task.task_index, mpi_result.task.task_index)
            self.assertEqual(
                serial_result.metrics["lambda_fit"],
                mpi_result.metrics["lambda_fit"],
            )

    def test_08_serial_fallback_does_not_require_mpi4py(self) -> None:
        env = os.environ.copy()
        existing_path = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(SRC_DIR) + (os.pathsep + existing_path if existing_path else "")
        cache_dir = Path(tempfile.gettempdir()) / "local_threshold_test_mpl"
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["MPLCONFIGDIR"] = str(cache_dir)
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; sys.modules['mpi4py'] = None; "
                    "import spinodal_phase12_mpi as driver; "
                    "assert driver.WORLD_SIZE == 1; assert driver.MPI is None"
                ),
            ],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_09_missing_phase0_outputs_are_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            phase0_dir = Path(temporary_dir) / "phase0"
            reference = ensure_phase0_reference(
                phase0_dir,
                required_deltas=(1e-3,),
                fallback_task=Phase0Task(delta_list=(1e-3,)),
            )
            self.assertTrue(reference.regenerated)
            self.assertTrue((phase0_dir / "phase0_summary.json").exists())
            self.assertTrue((phase0_dir / "phase0_delta_table.csv").exists())
            self.assertAlmostEqual(float(reference.delta_table.iloc[0]["delta"]), 1e-3)


if __name__ == "__main__":
    unittest.main()
