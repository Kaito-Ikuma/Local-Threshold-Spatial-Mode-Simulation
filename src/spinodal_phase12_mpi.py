#!/usr/bin/env python3
"""Serial/MPI task driver for deterministic spinodal Phase1-2 sweeps."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from spinodal_phase0 import DEFAULT_DELTAS, Phase0Task
from spinodal_phase12 import (
    DEFAULT_MODES,
    SCRIPT_VERSION,
    Phase0Reference,
    Phase12ModeResult,
    Phase12Task,
    build_phase12_tasks,
    ensure_phase0_reference,
    simulate_deterministic_mode,
    write_phase12_outputs,
)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class _SerialComm:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def gather(self, value: Any, root: int = 0) -> list[Any]:
        return [value]

    def bcast(self, value: Any, root: int = 0) -> Any:
        return value

    def Barrier(self) -> None:
        return None


COMM = MPI.COMM_WORLD if MPI is not None else _SerialComm()
RANK = COMM.Get_rank()
WORLD_SIZE = COMM.Get_size()
IS_ROOT = RANK == 0
MPI_ACTIVE = MPI is not None and WORLD_SIZE > 1


def root_print(*args: Any, **kwargs: Any) -> None:
    if IS_ROOT:
        print(*args, **kwargs, flush=True)


def parse_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(token.strip()) for token in text.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated floating-point values") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive and finite")
    return values


def parse_int_list(text: str) -> tuple[int, ...]:
    try:
        values = tuple(sorted(set(int(token.strip()) for token in text.split(",") if token.strip())))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not values or values[0] < 0:
        raise argparse.ArgumentTypeError("modes must be non-negative integers")
    return values


def run_task_subset(
    tasks: Sequence[Phase12Task],
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    """Run indices rank, rank+P, ... without depending on global MPI state."""
    start = time.perf_counter()
    task_indices = list(range(rank, len(tasks), world_size))
    results = [simulate_deterministic_mode(tasks[index]) for index in task_indices]
    return {
        "rank": int(rank),
        "task_indices": task_indices,
        "results": results,
        "elapsed_seconds": time.perf_counter() - start,
    }


def aggregate_task_payloads(
    payloads: Sequence[dict[str, Any]],
    expected_count: int,
) -> list[Phase12ModeResult]:
    """Validate task coverage and restore order independently of rank count."""
    indices: list[int] = []
    results: list[Phase12ModeResult] = []
    for payload in payloads:
        indices.extend(int(value) for value in payload["task_indices"])
        results.extend(payload["results"])
    if sorted(indices) != list(range(expected_count)):
        raise RuntimeError(
            f"Task decomposition mismatch: completed={sorted(indices)}, "
            f"expected={list(range(expected_count))}"
        )
    return sorted(results, key=lambda result: result.task.task_index)


def _broadcast_phase0_reference(
    phase0_dir: Path,
    required_deltas: Sequence[float],
    fallback_task: Phase0Task,
) -> Phase0Reference:
    payload: dict[str, Any] | None = None
    if IS_ROOT:
        reference = ensure_phase0_reference(
            phase0_dir,
            required_deltas=required_deltas,
            fallback_task=fallback_task,
        )
        payload = {
            "summary": reference.summary,
            "delta_table": reference.delta_table.to_dict(orient="list"),
            "regenerated": reference.regenerated,
        }
    payload = COMM.bcast(payload, root=0)
    if payload is None:
        raise RuntimeError("Phase0 reference broadcast failed")
    return Phase0Reference(
        phase0_dir=phase0_dir,
        summary=payload["summary"],
        delta_table=pd.DataFrame(payload["delta_table"]),
        regenerated=bool(payload["regenerated"]),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure q=0 and finite-q relaxation in the deterministic Gaussian "
            "closure near the Phase0 spinodal. MPI distributes independent tasks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument(
        "--phase0-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "phase0_B2_R12",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "phase12_B2_R12",
    )
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--modes", type=parse_int_list, default=DEFAULT_MODES)
    parser.add_argument("--delta-list", type=parse_float_list, default=DEFAULT_DELTAS)
    parser.add_argument("--epsilon-fraction", type=float, default=0.05)
    parser.add_argument("--tau-multiplier", type=float, default=6.0)
    parser.add_argument("--T-min", type=int, default=50, dest="T_min")
    parser.add_argument("--T-fixed", type=int, default=None, dest="T_fixed")
    parser.add_argument("--fit-start", type=int, default=0)
    parser.add_argument("--fit-end-fixed", type=int, default=None)
    parser.add_argument("--qR-max-fit", type=float, default=0.35, dest="qR_max_fit")
    parser.add_argument(
        "--epsilon-fraction-scan",
        type=parse_float_list,
        default=(),
        help="optional convergence scan, e.g. 0.10,0.05,0.025",
    )
    parser.add_argument(
        "--epsilon-scan-deltas",
        type=parse_float_list,
        default=(1e-3, 1e-4, 1e-5),
    )
    parser.add_argument(
        "--epsilon-scan-modes",
        type=parse_int_list,
        default=(0, 1, 4),
    )
    parser.add_argument(
        "--save-timeseries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="save compact A_q(t) time series for main tasks",
    )
    parser.add_argument(
        "--debug-profiles",
        action="store_true",
        help="save complete u_plus/u_minus profile histories as compressed NPZ",
    )

    # Used only when Phase0 files are absent.
    parser.add_argument("--B", type=float, default=2.0, help="Phase0 fallback parameter")
    parser.add_argument("--R", type=int, default=12, help="Phase0 fallback parameter")
    parser.add_argument("--sigma-J", type=float, default=1.0, dest="sigma_J")
    parser.add_argument("--sigma-phi", type=float, default=0.06, dest="sigma_phi")
    parser.add_argument("--phi-bar", type=float, default=0.0, dest="phi_bar")
    parser.add_argument("--a", type=float, default=1.0, dest="lattice_spacing")
    parser.add_argument(
        "--branch",
        choices=("stay_to_evacuate", "evacuate_to_stay"),
        default="stay_to_evacuate",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    total_start = time.perf_counter()

    requested_deltas = tuple(float(value) for value in args.delta_list)
    scan_fractions = tuple(sorted(set(args.epsilon_fraction_scan), reverse=True))
    scan_deltas = tuple(float(value) for value in args.epsilon_scan_deltas) if scan_fractions else ()
    all_required_deltas = tuple(dict.fromkeys((*requested_deltas, *scan_deltas)))
    fallback_task = Phase0Task(
        B=args.B,
        R=args.R,
        sigma_J=args.sigma_J,
        sigma_phi=args.sigma_phi,
        phi_bar=args.phi_bar,
        lattice_spacing=args.lattice_spacing,
        branch=args.branch,
        delta_list=all_required_deltas,
    )

    reference = _broadcast_phase0_reference(
        args.phase0_dir,
        required_deltas=all_required_deltas,
        fallback_task=fallback_task,
    )

    tasks: list[Phase12Task] | None = None
    if IS_ROOT:
        tasks = build_phase12_tasks(
            reference=reference,
            deltas=requested_deltas,
            modes=args.modes,
            N=args.N,
            epsilon_fraction=args.epsilon_fraction,
            tau_multiplier=args.tau_multiplier,
            T_min=args.T_min,
            T_fixed=args.T_fixed,
            fit_start=args.fit_start,
            fit_end_fixed=args.fit_end_fixed,
            qR_max_fit=args.qR_max_fit,
            task_group="main",
            debug_profiles=args.debug_profiles,
        )
        next_index = len(tasks)
        for fraction in scan_fractions:
            scan_tasks = build_phase12_tasks(
                reference=reference,
                deltas=scan_deltas,
                modes=args.epsilon_scan_modes,
                N=args.N,
                epsilon_fraction=fraction,
                tau_multiplier=args.tau_multiplier,
                T_min=args.T_min,
                T_fixed=args.T_fixed,
                fit_start=args.fit_start,
                fit_end_fixed=args.fit_end_fixed,
                qR_max_fit=args.qR_max_fit,
                task_group="epsilon_scan",
                start_index=next_index,
                debug_profiles=args.debug_profiles,
            )
            tasks.extend(scan_tasks)
            next_index += len(scan_tasks)
        root_print(
            f"[Phase1-2] ranks={WORLD_SIZE}, tasks={len(tasks)}, "
            f"main={len(requested_deltas) * len(args.modes)}, output={args.output_dir}"
        )
        if reference.regenerated:
            root_print(f"[Phase1-2] regenerated missing Phase0 outputs in {args.phase0_dir}")
    tasks = COMM.bcast(tasks, root=0)
    if tasks is None:
        raise RuntimeError("Task broadcast failed")

    local_payload = run_task_subset(tasks, RANK, WORLD_SIZE)
    gathered = COMM.gather(local_payload, root=0)

    if IS_ROOT:
        results = aggregate_task_payloads(gathered, expected_count=len(tasks))
        rank_times = {
            str(int(payload["rank"])): float(payload["elapsed_seconds"])
            for payload in gathered
        }
        task_counts = {
            str(int(payload["rank"])): len(payload["task_indices"])
            for payload in gathered
        }
        max_rank_time = max(rank_times.values(), default=0.0)
        total_rank_time = sum(rank_times.values())
        runtime_metadata = {
            "mpi_available": MPI is not None,
            "mpi_active": MPI_ACTIVE,
            "world_size": WORLD_SIZE,
            "parallelization": "round-robin independent (delta, mode, epsilon_fraction) tasks",
            "task_count": len(tasks),
            "task_counts_by_rank": task_counts,
            "compute_seconds_by_rank": rank_times,
            "max_rank_compute_seconds": max_rank_time,
            "sum_rank_compute_seconds": total_rank_time,
            "task_parallel_speedup_estimate": (
                total_rank_time / max_rank_time if max_rank_time > 0.0 else None
            ),
            "phase0_regenerated": reference.regenerated,
        }
        output_start = time.perf_counter()
        paths = write_phase12_outputs(
            results=results,
            output_dir=args.output_dir,
            qR_max_fit=args.qR_max_fit,
            runtime_metadata=runtime_metadata,
            save_timeseries=args.save_timeseries,
        )
        elapsed = time.perf_counter() - total_start
        output_elapsed = time.perf_counter() - output_start
        summary_path = paths["validation_summary"]
        validation_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        validation_summary["runtime"]["output_seconds"] = output_elapsed
        validation_summary["runtime"]["driver_total_wall_seconds"] = elapsed
        summary_path.write_text(
            json.dumps(validation_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        root_print(f"[Phase1-2] completed in {elapsed:.3f} s")
        for path in paths.values():
            root_print(path)

    COMM.Barrier()


if __name__ == "__main__":
    main()
