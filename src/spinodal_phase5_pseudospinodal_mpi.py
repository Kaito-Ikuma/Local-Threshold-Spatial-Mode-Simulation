#!/usr/bin/env python3
"""Small MPI scan of microscopic preparation survival away from the spinodal.

This diagnostic intentionally bypasses the plotting-oriented Phase0/Phase1-2
drivers.  It evaluates the same Gaussian-map theory equations locally, then
uses the unchanged Phase5 microscopic core to measure whether the prepared
metastable branch survives until t=0.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from spinodal_phase5_core import (
    SCRIPT_VERSION,
    Phase5BlockResult,
    Phase5Task,
    Phase5WorkUnit,
    build_work_units,
    simulate_microscopic_block,
)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


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


BRANCH_SIGNS = {
    "stay_to_evacuate": -1,
    "evacuate_to_stay": 1,
}


@dataclass(frozen=True)
class GaussianTheoryPoint:
    delta: float
    Delta: float
    m_star: float
    m_spinodal: float
    Delta_spinodal: float
    mu: float
    sigma_eff: float
    Lambda_star: float
    Gamma_closure: float


def parse_float_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(token.strip()) for token in text.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not values or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("deltas must be positive and finite")
    return values


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bisect_root(function: Any, left: float, right: float) -> float:
    f_left = float(function(left))
    f_right = float(function(right))
    if abs(f_left) < 1e-14:
        return left
    if abs(f_right) < 1e-14:
        return right
    if f_left * f_right > 0.0:
        raise ValueError(
            f"could not bracket metastable root: f({left})={f_left}, "
            f"f({right})={f_right}"
        )
    for _ in range(200):
        middle = 0.5 * (left + right)
        f_middle = float(function(middle))
        if abs(f_middle) < 1e-14 or abs(right - left) < 1e-14:
            return middle
        if f_left * f_middle <= 0.0:
            right = middle
            f_right = f_middle
        else:
            left = middle
            f_left = f_middle
    return 0.5 * (left + right)


def gaussian_theory_points(
    deltas: Sequence[float],
    *,
    B: float,
    R: int,
    sigma_J: float,
    sigma_phi: float,
    branch: str,
) -> list[GaussianTheoryPoint]:
    """Evaluate the Phase0 equations without importing plotting modules."""
    if B <= 1.0:
        raise ValueError("B must exceed 1 for a spinodal")
    if R < 1 or sigma_J < 0.0 or sigma_phi < 0.0:
        raise ValueError("invalid interaction range or noise scale")
    if branch not in BRANCH_SIGNS:
        raise ValueError(f"unknown branch: {branch}")
    sigma_eff = math.sqrt(sigma_J * sigma_J / (2 * R) + sigma_phi * sigma_phi)
    if sigma_eff <= 0.0:
        raise ValueError("effective Gaussian width must be positive")
    mu = B * math.sqrt(2.0 * math.pi) * sigma_eff / 2.0
    sign = BRANCH_SIGNS[branch]
    z_spinodal = sign * math.sqrt(2.0 * math.log(B))
    m_spinodal = 2.0 * _normal_cdf(z_spinodal) - 1.0
    Delta_spinodal = sigma_eff * z_spinodal - mu * m_spinodal

    points = []
    for delta in deltas:
        if not math.isfinite(delta) or delta <= 0.0:
            raise ValueError("deltas must be positive and finite")
        Delta = Delta_spinodal + sign * float(delta)

        def residual(magnetization: float) -> float:
            z = (mu * magnetization + Delta) / sigma_eff
            return 2.0 * _normal_cdf(z) - 1.0 - magnetization

        bracket = (
            (-1.0, m_spinodal)
            if branch == "stay_to_evacuate"
            else (m_spinodal, 1.0)
        )
        m_star = _bisect_root(residual, *bracket)
        z_star = (mu * m_star + Delta) / sigma_eff
        Lambda_star = B * math.exp(-0.5 * z_star * z_star)
        if not 0.0 < Lambda_star < 1.0:
            raise ValueError(f"metastable root is not stable for delta={delta:g}")
        points.append(
            GaussianTheoryPoint(
                delta=float(delta),
                Delta=Delta,
                m_star=m_star,
                m_spinodal=m_spinodal,
                Delta_spinodal=Delta_spinodal,
                mu=mu,
                sigma_eff=sigma_eff,
                Lambda_star=Lambda_star,
                Gamma_closure=-math.log(Lambda_star),
            )
        )
    return points


def build_scan_tasks(args: argparse.Namespace) -> list[Phase5Task]:
    points = gaussian_theory_points(
        args.deltas,
        B=args.B,
        R=args.R,
        sigma_J=args.sigma_J,
        sigma_phi=args.sigma_phi,
        branch=args.branch,
    )
    return [
        Phase5Task(
            task_id=f"pseudospinodal_d{index:03d}",
            task_group="pseudospinodal_scan",
            delta_index=index,
            epsilon_index=0,
            delta=point.delta,
            Delta=point.Delta,
            m_star=point.m_star,
            m_spinodal=point.m_spinodal,
            Gamma_closure=point.Gamma_closure,
            N=args.N,
            R=args.R,
            lattice_spacing=args.lattice_spacing,
            mode_index=0,
            epsilon_fraction=args.epsilon_fraction,
            M_total=args.M_total,
            block_size=args.block_size,
            T=args.T_fixed,
            fit_start=0,
            fit_end=args.T_fixed,
            mu=point.mu,
            sigma_J=args.sigma_J,
            sigma_phi=args.sigma_phi,
            phi_bar=args.phi_bar,
            branch=args.branch,
            microscopic_kernel=args.kernel,
            initialization_mode="prepared_metastable",
            preparation_width=args.preparation_width,
            preparation_steps=args.preparation_steps,
            burn_steps_per_stage=args.burn_steps_per_stage,
            float_dtype=args.float_dtype,
            base_seed=args.base_seed,
        )
        for index, point in enumerate(points)
    ]


def aggregate_scan_results(
    tasks: Sequence[Phase5Task],
    results: Sequence[Phase5BlockResult],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    detail: dict[str, Any] = {}
    for task in tasks:
        blocks = sorted(
            (result for result in results if result.task_id == task.task_id),
            key=lambda result: result.block_id,
        )
        if not blocks:
            raise ValueError(f"no results for {task.task_id}")
        weights = np.asarray([result.block_n for result in blocks], dtype=float)
        M_used = int(np.sum(weights))

        def average_array(name: str) -> np.ndarray:
            values = np.stack(
                [np.asarray(getattr(result, name), dtype=float) for result in blocks]
            )
            return np.average(values, axis=0, weights=weights)

        preparation = average_array("preparation_magnetization")
        escape = average_array("escape_fraction")
        row: dict[str, Any] = {
            "delta": task.delta,
            "Delta": task.Delta,
            "m_star": task.m_star,
            "m_spinodal": task.m_spinodal,
            "N": task.N,
            "R": task.R,
            "M_total": M_used,
            "n_blocks": len(blocks),
            "block_size": task.block_size,
            "kernel": task.microscopic_kernel,
            "preparation_initial_m": float(preparation[0]),
            "preparation_final_m": float(preparation[-1]),
            "preparation_drift": float(preparation[-1] - preparation[0]),
            "escape_t0": float(escape[0]),
            "escape_t1": float(escape[min(1, len(escape) - 1)]),
            "escape_final": float(escape[-1]),
            "escape_max": float(np.max(escape)),
            "survival_t0": float(1.0 - escape[0]),
            "preparation_survives": bool(float(escape[0]) <= 0.1),
        }
        for index, value in enumerate(preparation):
            row[f"preparation_m_stage_{index}"] = float(value)
        rows.append(row)
        detail[task.task_id] = {
            "task": asdict(task),
            "preparation_Delta_stages": np.linspace(
                task.Delta - task.preparation_width
                if task.branch == "stay_to_evacuate"
                else task.Delta + task.preparation_width,
                task.Delta,
                task.preparation_steps,
            ).tolist(),
            "preparation_magnetization": preparation.tolist(),
            "escape_fraction": escape.tolist(),
        }
    return rows, detail


def write_outputs(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    detail: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "phase5_pseudospinodal_scan.csv"
    detail_path = output_dir / "phase5_pseudospinodal_detail.json"
    environment_path = output_dir / "environment.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    detail_path.write_text(
        json.dumps(
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "phase5_core_version": SCRIPT_VERSION,
                "purpose": "prepared-metastable survival / microscopic pseudospinodal diagnostic",
                "configuration": {
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "tasks": detail,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    environment_path.write_text(
        json.dumps(
            {
                "python_executable": sys.executable,
                "python_prefix": sys.prefix,
                "python_version": platform.python_version(),
                "mpi_library_version": MPI.Get_library_version() if MPI is not None else None,
                "mpi_world_size": WORLD_SIZE,
                "hostname": socket.gethostname(),
                "phase5_venv": os.environ.get("PHASE5_VENV"),
                "phase5_python_requested": os.environ.get("PHASE5_PY"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return csv_path, detail_path, environment_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan prepared-metastable survival without importing matplotlib.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--deltas", type=parse_float_list, default=(0.01, 0.02, 0.03, 0.05, 0.10))
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--M-total", type=int, default=64)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--T-fixed", type=int, default=2)
    parser.add_argument("--epsilon-fraction", type=float, default=0.05)
    parser.add_argument("--kernel", choices=("direct_J", "aggregated_exact"), default="direct_J")
    parser.add_argument("--B", type=float, default=2.0)
    parser.add_argument("--R", type=int, default=12)
    parser.add_argument("--sigma-J", type=float, default=1.0, dest="sigma_J")
    parser.add_argument("--sigma-phi", type=float, default=0.06, dest="sigma_phi")
    parser.add_argument("--phi-bar", type=float, default=0.0, dest="phi_bar")
    parser.add_argument("--a", type=float, default=1.0, dest="lattice_spacing")
    parser.add_argument("--branch", choices=tuple(BRANCH_SIGNS), default="stay_to_evacuate")
    parser.add_argument("--preparation-width", type=float, default=0.02)
    parser.add_argument("--preparation-steps", type=int, default=6)
    parser.add_argument("--burn-steps-per-stage", type=int, default=8)
    parser.add_argument("--float-dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--base-seed", type=int, default=20260815)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/runs/phase5_pseudospinodal_direct_J"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.T_fixed < 2:
        parser.error("--T-fixed must be at least 2")
    if args.M_total < 1 or args.block_size < 1:
        parser.error("--M-total and --block-size must be positive")
    if args.preparation_width <= 0.0:
        parser.error("--preparation-width must be positive")

    tasks = build_scan_tasks(args) if IS_ROOT else None
    tasks = COMM.bcast(tasks, root=0)
    units = [unit for task in tasks for unit in build_work_units(task)]
    assignments = [[] for _ in range(WORLD_SIZE)]
    loads = [0] * WORLD_SIZE
    for unit in sorted(units, key=lambda item: (-item.estimated_cost, item.unit_id)):
        target = min(range(WORLD_SIZE), key=lambda rank: (loads[rank], rank))
        assignments[target].append(unit)
        loads[target] += unit.estimated_cost

    local_results = [simulate_microscopic_block(unit) for unit in assignments[RANK]]
    gathered = COMM.gather(local_results, root=0)
    if IS_ROOT:
        all_results = [result for rank_results in gathered for result in rank_results]
        rows, detail = aggregate_scan_results(tasks, all_results)
        paths = write_outputs(args.output_dir, rows, detail, args)
        print(
            "delta preparation_final_m escape_t0 survival_t0 preparation_survives",
            flush=True,
        )
        for row in rows:
            print(
                row["delta"],
                row["preparation_final_m"],
                row["escape_t0"],
                row["survival_t0"],
                row["preparation_survives"],
                flush=True,
            )
        for path in paths:
            print(path, flush=True)
    COMM.Barrier()


if __name__ == "__main__":
    main()
