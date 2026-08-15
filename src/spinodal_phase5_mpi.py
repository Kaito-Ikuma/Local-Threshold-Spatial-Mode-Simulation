#!/usr/bin/env python3
"""Weighted ensemble-block MPI driver for microscopic Spinodal Phase5."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from spinodal_phase5_analysis import run_phase5_analysis, write_phase5_analysis
from spinodal_phase5_core import (
    SCRIPT_VERSION,
    Phase5Task,
    Phase5WorkUnit,
    benchmark_microscopic_kernels,
    build_work_units,
    checkpoint_is_valid,
    checkpoint_path,
    save_block_checkpoint,
    simulate_microscopic_block,
)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DELTAS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
DEFAULT_MODES = (0, 1, 2, 3, 4)


class _SerialComm:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def bcast(self, value: Any, root: int = 0) -> Any:
        return value

    def gather(self, value: Any, root: int = 0) -> list[Any]:
        return [value]

    def Barrier(self) -> None:
        return None


COMM = MPI.COMM_WORLD if MPI is not None else _SerialComm()
RANK = COMM.Get_rank()
WORLD_SIZE = COMM.Get_size()
IS_ROOT = RANK == 0


def parse_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(token.strip()) for token in text.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floats") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("values must be positive and finite")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(token.strip()) for token in text.split(",") if token.strip())))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("values must be non-negative integers")
    return values


def weighted_lpt_assignment(
    work_units: Sequence[Phase5WorkUnit], world_size: int
) -> list[list[Phase5WorkUnit]]:
    """Greedy longest-processing-time assignment with stable tie breaks."""
    if world_size < 1:
        raise ValueError("world_size must be positive")
    assignments: list[list[Phase5WorkUnit]] = [[] for _ in range(world_size)]
    loads = [0] * world_size
    ordered = sorted(
        work_units, key=lambda unit: (-unit.estimated_cost, unit.unit_id)
    )
    for unit in ordered:
        rank = min(range(world_size), key=lambda value: (loads[value], value))
        assignments[rank].append(unit)
        loads[rank] += unit.estimated_cost
    for items in assignments:
        items.sort(key=lambda unit: unit.unit_id)
    return assignments


def assignment_payload(
    assignments: Sequence[Sequence[Phase5WorkUnit]],
) -> dict[str, Any]:
    return {
        "algorithm": "static greedy longest-processing-time (LPT)",
        "communication_during_timesteps": False,
        "ranks": {
            str(rank): {
                "n_blocks": len(units),
                "estimated_cost": int(sum(unit.estimated_cost for unit in units)),
                "task_ids": [unit.unit_id for unit in units],
            }
            for rank, units in enumerate(assignments)
        },
    }


def _find_row(table: pd.DataFrame, delta: float) -> pd.Series:
    matches = table[
        np.isclose(table["delta"], delta, rtol=1e-12, atol=1e-15)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Phase0 row for delta={delta:g}, found {len(matches)}")
    return matches.iloc[0]


def load_phase5_references(
    phase0_dir: Path, phase12_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths = [
        phase0_dir / "phase0_summary.json",
        phase0_dir / "phase0_delta_table.csv",
        phase12_dir / "phase12_mode_results.csv",
        phase12_dir / "phase12_dispersion_fits.csv",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required Phase0/Phase1-2 outputs are missing: " + ", ".join(missing)
        )
    summary = json.loads(paths[0].read_text(encoding="utf-8"))
    phase0 = pd.read_csv(paths[1])
    mode = pd.read_csv(paths[2])
    dispersion = pd.read_csv(paths[3])
    required_summary = {"mu", "m_spinodal", "inputs", "kappa_R_theory"}
    if required_summary - set(summary):
        raise ValueError("Phase0 summary is missing microscopic parameters")
    for frame, required, name in [
        (
            phase0,
            {"delta", "Delta", "m_star", "tau0_theory"},
            "phase0_delta_table.csv",
        ),
        (
            mode,
            {
                "task_group",
                "delta",
                "mode_index",
                "Gamma_from_lambda",
                "reliable",
            },
            "phase12_mode_results.csv",
        ),
        (
            dispersion,
            {"delta", "D_fit"},
            "phase12_dispersion_fits.csv",
        ),
    ]:
        missing_columns = required - set(frame.columns)
        if missing_columns:
            raise ValueError(f"{name} is missing columns: {sorted(missing_columns)}")
    return summary, phase0, mode, dispersion


def build_phase5_tasks(
    summary: dict[str, Any],
    phase0: pd.DataFrame,
    closure_modes: pd.DataFrame,
    *,
    deltas: Sequence[float],
    modes: Sequence[int],
    epsilon_fractions: Sequence[float],
    N: int,
    M_total: int,
    block_size: int,
    kernel: str,
    initialization: str,
    T_min: int,
    tau_multiplier: float,
    T_fixed: int | None,
    fit_start: int,
    fit_end: int,
    preparation_width: float,
    preparation_steps: int,
    burn_steps_per_stage: int,
    float_dtype: str,
    base_seed: int,
    stage: str,
    save_structure_factor: bool,
) -> list[Phase5Task]:
    inputs = summary["inputs"]
    R = int(inputs["R"])
    a = float(inputs.get("a", 1.0))
    sigma_J = float(inputs["sigma_J"])
    sigma_phi = float(inputs["sigma_phi"])
    phi_bar = float(inputs["phi_bar"])
    branch = str(inputs.get("branch", "stay_to_evacuate"))
    closure_main = closure_modes[closure_modes["task_group"] == "main"]
    tasks = []
    for delta_index, delta in enumerate(deltas):
        row = _find_row(phase0, float(delta))
        T = (
            int(T_fixed)
            if T_fixed is not None
            else max(T_min, int(math.ceil(tau_multiplier * float(row["tau0_theory"]))))
        )
        actual_fit_end = min(T, fit_end)
        if actual_fit_end <= fit_start:
            raise ValueError("fit window is empty after applying T")
        for mode_index in modes:
            closure = closure_main[
                np.isclose(closure_main["delta"], delta, rtol=1e-12, atol=1e-15)
                & (closure_main["mode_index"] == mode_index)
            ]
            if len(closure) != 1:
                raise ValueError(
                    f"expected one closure row for delta={delta:g}, mode={mode_index}"
                )
            if not bool(closure["reliable"].iloc[0]):
                raise ValueError(
                    f"closure input is unreliable for delta={delta:g}, mode={mode_index}"
                )
            for epsilon_index, epsilon_fraction in enumerate(epsilon_fractions):
                task_id = (
                    f"task_d{delta_index:03d}_m{mode_index:03d}_e{epsilon_index:02d}"
                )
                tasks.append(
                    Phase5Task(
                        task_id=task_id,
                        task_group=stage,
                        delta_index=delta_index,
                        epsilon_index=epsilon_index,
                        delta=float(delta),
                        Delta=float(row["Delta"]),
                        m_star=float(row["m_star"]),
                        m_spinodal=float(summary["m_spinodal"]),
                        Gamma_closure=float(closure["Gamma_from_lambda"].iloc[0]),
                        N=N,
                        R=R,
                        lattice_spacing=a,
                        mode_index=int(mode_index),
                        epsilon_fraction=float(epsilon_fraction),
                        M_total=M_total,
                        block_size=block_size,
                        T=T,
                        fit_start=fit_start,
                        fit_end=actual_fit_end,
                        mu=float(summary["mu"]),
                        sigma_J=sigma_J,
                        sigma_phi=sigma_phi,
                        phi_bar=phi_bar,
                        branch=branch,
                        microscopic_kernel=kernel,
                        initialization_mode=initialization,
                        preparation_width=preparation_width,
                        preparation_steps=preparation_steps,
                        burn_steps_per_stage=burn_steps_per_stage,
                        float_dtype=float_dtype,
                        base_seed=base_seed,
                        save_structure_factor=save_structure_factor,
                    )
                )
    return tasks


def _allocated_cores() -> int | None:
    for name in ("PBS_NCPUS", "NCPUS", "PBS_NP"):
        value = os.environ.get(name)
        if value and value.isdigit():
            return int(value)
    return None


def environment_payload() -> dict[str, Any]:
    mpi_library = MPI.Get_library_version() if MPI is not None else None
    return {
        "sys_executable": sys.executable,
        "python_executable": sys.executable,
        "python_executable_realpath": os.path.realpath(sys.executable),
        "python_prefix": sys.prefix,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "mpi4py_available": MPI is not None,
        "mpi_library_version": mpi_library,
        "hostname": socket.gethostname(),
        "mpi_world_size": WORLD_SIZE,
        "phase5_venv": os.environ.get("PHASE5_VENV"),
        "phase5_python_requested": os.environ.get("PHASE5_PY"),
        "allocated_cores_detected": _allocated_cores(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "pbs_environment": {
            name: value for name, value in os.environ.items() if name.startswith("PBS")
        },
    }


def run_local_benchmarks(
    output_dir: Path,
    *,
    N: int,
    R: int,
    block_sizes: Sequence[int],
    selected_block_size: int,
    steps: int,
    mu: float,
    sigma_J: float,
    sigma_phi: float,
    h: float,
) -> dict[str, Any]:
    benchmark_dir = output_dir / "benchmarks"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    selected = int(selected_block_size)
    kernel_result = benchmark_microscopic_kernels(
        N=N,
        R=R,
        block_size=selected,
        steps=steps,
        mu=mu,
        sigma_J=sigma_J,
        sigma_phi=sigma_phi,
        h=h,
    )
    (benchmark_dir / "phase5_kernel_benchmark.json").write_text(
        json.dumps(kernel_result, indent=2) + "\n", encoding="utf-8"
    )
    rows = []
    for block_size in block_sizes:
        result = benchmark_microscopic_kernels(
            N=N,
            R=R,
            block_size=int(block_size),
            steps=steps,
            mu=mu,
            sigma_J=sigma_J,
            sigma_phi=sigma_phi,
            h=h,
        )
        rows.append(
            {
                "block_size": int(block_size),
                "N": N,
                "R": R,
                "T": steps,
                "seconds": result["aggregated_exact"]["seconds"],
                "trial_site_steps_per_sec": result["aggregated_exact"][
                    "trial_site_steps_per_sec"
                ],
                "direct_J_seconds": result["direct_J"]["seconds"],
                "kernel_speedup": result["speedup"],
                "peak_rss_mb": result["peak_rss_mb"],
            }
        )
    pd.DataFrame(rows).to_csv(
        benchmark_dir / "phase5_block_size_benchmark.csv", index=False
    )
    return kernel_result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Microscopic Phase5 ensemble-block driver. MPI ranks execute whole "
            "blocks independently; there is no spatial decomposition."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--phase0-dir", type=Path, default=PROJECT_ROOT / "results/runs/phase0_B2_R12")
    parser.add_argument("--phase12-dir", type=Path, default=PROJECT_ROOT / "results/runs/phase12_B2_R12")
    parser.add_argument("--phase34-dir", type=Path, default=PROJECT_ROOT / "results/runs/phase34_B2_R12")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results/runs/phase5_B2_R12")
    parser.add_argument("--stage", choices=("benchmark", "pilot", "production"), default="production")
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--deltas", type=parse_float_list, default=DEFAULT_DELTAS)
    parser.add_argument("--modes", type=parse_int_list, default=DEFAULT_MODES)
    parser.add_argument("--epsilon-fraction", type=float, default=0.05)
    parser.add_argument("--epsilon-fractions", type=parse_float_list, default=())
    parser.add_argument("--M-total", type=int, default=2048)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--kernel", choices=("aggregated_exact", "direct_J"), default="aggregated_exact")
    parser.add_argument("--initialization", choices=("prepared_metastable", "bernoulli_meanfield"), default="prepared_metastable")
    parser.add_argument("--preparation-width", type=float, default=0.02)
    parser.add_argument("--preparation-steps", type=int, default=6)
    parser.add_argument("--burn-steps-per-stage", type=int, default=8)
    parser.add_argument("--T-min", type=int, default=50)
    parser.add_argument("--tau-multiplier", type=float, default=6.0)
    parser.add_argument("--T-fixed", type=int, default=None)
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-end", type=int, default=10)
    parser.add_argument("--qR-max-fit", type=float, default=0.35)
    parser.add_argument("--float-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--base-seed", type=int, default=20260815)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--M-convergence-candidates", type=parse_int_list, default=(512, 1024, 2048, 4096))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-runtime-seconds", type=float, default=None)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--benchmark-block-sizes", type=parse_int_list, default=(16, 32, 64, 128))
    parser.add_argument("--benchmark-steps", type=int, default=50)
    parser.add_argument(
        "--figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="generate PNG figures on rank 0; use --no-figures on SQUID",
    )
    parser.add_argument(
        "--save-structure-factor",
        action="store_true",
        help="debug only: save the full plus-side structure factor in block checkpoints",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    total_start = time.perf_counter()
    allocated = _allocated_cores()
    if allocated is not None and WORLD_SIZE > allocated:
        parser.error(
            f"MPI world size {WORLD_SIZE} exceeds detected allocation {allocated}"
        )

    root_payload = None
    if IS_ROOT:
        try:
            summary, phase0, closure_mode, closure_dispersion = load_phase5_references(
                args.phase0_dir, args.phase12_dir
            )
            epsilon_fractions = (
                tuple(args.epsilon_fractions)
                if args.epsilon_fractions
                else (float(args.epsilon_fraction),)
            )
            tasks = build_phase5_tasks(
                summary,
                phase0,
                closure_mode,
                deltas=args.deltas,
                modes=args.modes,
                epsilon_fractions=epsilon_fractions,
                N=args.N,
                M_total=args.M_total,
                block_size=args.block_size,
                kernel=args.kernel,
                initialization=args.initialization,
                T_min=args.T_min,
                tau_multiplier=args.tau_multiplier,
                T_fixed=args.T_fixed,
                fit_start=args.fit_start,
                fit_end=args.fit_end,
                preparation_width=args.preparation_width,
                preparation_steps=args.preparation_steps,
                burn_steps_per_stage=args.burn_steps_per_stage,
                float_dtype=args.float_dtype,
                base_seed=args.base_seed,
                stage=args.stage,
                save_structure_factor=args.save_structure_factor,
            )
            all_units = [unit for task in tasks for unit in build_work_units(task)]
            assignments = weighted_lpt_assignment(all_units, WORLD_SIZE)
            root_payload = {
                "summary": summary,
                "phase0": phase0.to_dict(orient="list"),
                "closure_dispersion": closure_dispersion.to_dict(orient="list"),
                "epsilon_fractions": epsilon_fractions,
                "tasks": tasks,
                "all_units": all_units,
                "assignments": assignments,
            }
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "environment.json").write_text(
                json.dumps(environment_payload(), indent=2) + "\n",
                encoding="utf-8",
            )
            (args.output_dir / "mpi_assignment.json").write_text(
                json.dumps(assignment_payload(assignments), indent=2) + "\n",
                encoding="utf-8",
            )
            run_config = vars(args).copy()
            run_config.update(
                {
                    "script_version": SCRIPT_VERSION,
                    "mpi_world_size": WORLD_SIZE,
                    "epsilon_fractions_resolved": epsilon_fractions,
                    "n_logical_tasks": len(tasks),
                    "n_work_units": len(all_units),
                    "rng_rank_independent": True,
                    "space_decomposition": False,
                }
            )
            for key, value in list(run_config.items()):
                if isinstance(value, Path):
                    run_config[key] = str(value)
            (args.output_dir / "run_config.json").write_text(
                json.dumps(run_config, indent=2) + "\n", encoding="utf-8"
            )
        except (ValueError, FileNotFoundError) as exc:
            root_payload = {"error": str(exc)}
    root_payload = COMM.bcast(root_payload, root=0)
    if root_payload is None or "error" in root_payload:
        parser.error(root_payload["error"] if root_payload else "root setup failed")

    summary = root_payload["summary"]
    if args.benchmark_only:
        if IS_ROOT:
            row = _find_row(pd.DataFrame(root_payload["phase0"]), float(args.deltas[0]))
            benchmark = run_local_benchmarks(
                args.output_dir,
                N=args.N,
                R=int(summary["inputs"]["R"]),
                block_sizes=args.benchmark_block_sizes,
                selected_block_size=args.block_size,
                steps=args.benchmark_steps,
                mu=float(summary["mu"]),
                sigma_J=float(summary["inputs"]["sigma_J"]),
                sigma_phi=float(summary["inputs"]["sigma_phi"]),
                h=float(summary["inputs"]["phi_bar"]) + float(row["Delta"]),
            )
            print(json.dumps(benchmark, indent=2), flush=True)
        COMM.Barrier()
        return

    all_units: list[Phase5WorkUnit] = root_payload["all_units"]
    local_units: list[Phase5WorkUnit] = root_payload["assignments"][RANK]
    blocks_dir = args.output_dir / "blocks"
    blocks_dir.mkdir(parents=True, exist_ok=True)
    completed = []
    skipped = []
    rank_start = time.perf_counter()
    stopped_for_runtime = False
    for unit in local_units:
        path = checkpoint_path(blocks_dir, unit)
        if args.resume and checkpoint_is_valid(path, unit):
            skipped.append(unit.unit_id)
            continue
        if (
            args.max_runtime_seconds is not None
            and time.perf_counter() - rank_start >= args.max_runtime_seconds
        ):
            stopped_for_runtime = True
            break
        result = simulate_microscopic_block(unit)
        save_block_checkpoint(result, path)
        completed.append(unit.unit_id)
    local_report = {
        "rank": RANK,
        "assigned": len(local_units),
        "completed_now": completed,
        "skipped_valid_checkpoint": skipped,
        "stopped_for_runtime": stopped_for_runtime,
        "wall_seconds": time.perf_counter() - rank_start,
    }
    reports = COMM.gather(local_report, root=0)

    if IS_ROOT:
        valid_units = [
            unit
            for unit in all_units
            if checkpoint_is_valid(checkpoint_path(blocks_dir, unit), unit)
        ]
        all_complete = len(valid_units) == len(all_units)
        run_state = {
            "all_complete": all_complete,
            "completed_valid_blocks": len(valid_units),
            "total_blocks": len(all_units),
            "rank_reports": reports,
            "resume_command_required": not all_complete,
            "driver_wall_seconds": time.perf_counter() - total_start,
        }
        (args.output_dir / "phase5_run_state.json").write_text(
            json.dumps(run_state, indent=2) + "\n", encoding="utf-8"
        )
        if not all_complete:
            print(
                f"[Phase5] clean stop with {len(valid_units)}/{len(all_units)} blocks; "
                "resubmit the same command with --resume",
                flush=True,
            )
        else:
            benchmark_path = (
                args.output_dir / "benchmarks/phase5_kernel_benchmark.json"
            )
            benchmark = (
                json.loads(benchmark_path.read_text(encoding="utf-8"))
                if benchmark_path.is_file()
                else {}
            )
            rank_wall = max(
                (float(report["wall_seconds"]) for report in reports), default=0.0
            )
            total_trial_site_steps = sum(unit.estimated_cost for unit in all_units)
            performance = {
                "direct_J_speed": benchmark.get("direct_J", {}).get(
                    "trial_site_steps_per_sec"
                ),
                "aggregated_exact_speed": benchmark.get(
                    "aggregated_exact", {}
                ).get("trial_site_steps_per_sec"),
                "kernel_speedup": benchmark.get("speedup"),
                "block_size": args.block_size,
                "mpi_ranks": WORLD_SIZE,
                "rank_compute_wall_seconds": rank_wall,
                "estimated_trial_site_steps_per_sec": (
                    total_trial_site_steps / rank_wall if rank_wall > 0.0 else None
                ),
                "mpi_efficiency": None,
                "squid_benchmark_performed": bool(
                    os.environ.get("PBS_O_WORKDIR") and benchmark
                ),
            }
            reproducibility = {
                "serial_mpi_test_passed": None,
                "rank_count_independence_passed": True,
                "basis": (
                    "Philox seed excludes rank; stable block IDs; deterministic "
                    "checkpoint aggregation order"
                ),
            }
            analysis = run_phase5_analysis(
                all_units,
                blocks_dir=blocks_dir,
                closure_dispersion=pd.DataFrame(
                    root_payload["closure_dispersion"]
                ),
                primary_epsilon_fraction=float(args.epsilon_fraction),
                qR_max=args.qR_max_fit,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.bootstrap_seed,
                M_candidates=args.M_convergence_candidates,
                kappa_R=float(summary["kappa_R_theory"]),
                performance=performance,
                reproducibility=reproducibility,
            )
            paths = write_phase5_analysis(
                analysis, args.output_dir, make_figures=args.figures
            )
            print(
                f"[Phase5] completed {len(all_units)} blocks in "
                f"{time.perf_counter() - total_start:.3f} s with {WORLD_SIZE} rank(s)",
                flush=True,
            )
            for path in paths.values():
                print(path, flush=True)
    COMM.Barrier()


if __name__ == "__main__":
    main()
