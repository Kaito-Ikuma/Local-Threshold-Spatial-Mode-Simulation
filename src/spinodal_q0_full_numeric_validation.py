#!/usr/bin/env python3
"""Run and aggregate the fully numerical q=0 validation campaign.

The numerical estimator is deliberately not implemented here.  Every new
condition is constructed and simulated by the existing Phase1-2 machinery;
this module only discovers matching metadata, records provenance, and builds
the two campaign master tables.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from spinodal_phase0 import Phase0Task
from spinodal_phase12 import (
    build_phase12_tasks,
    ensure_phase0_reference,
    simulate_deterministic_mode,
    write_phase12_outputs,
)


SCRIPT_VERSION = "2026.09.04-q0-full-numeric-v1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NEW_R_VALUES = (6, 24, 48, 96)
ALL_R_VALUES = (6, 12, 24, 48, 96)
TARGET_DELTAS = (1e-5, 3e-5, 1e-4, 3e-4)
GAMMA_SOURCE = "deterministic q=0 numerical"
EXPECTED_PHYSICS = {
    "B": 2.0,
    "sigma_J": 1.0,
    "sigma_phi": 0.06,
    "phi_bar": 0.0,
    "a": 1.0,
    "branch": "stay_to_evacuate",
}


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _relative_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f"invalid boolean values: {invalid}")
    return normalized.map(mapping).astype(bool)


def load_phase6_N_map(phase6_dir: Path) -> dict[int, int]:
    """Read and cross-check the actual Phase6 R-to-N metadata."""
    table_path = phase6_dir / "resultC_xi_boundary.csv"
    summary_path = phase6_dir / "phase6_validation_summary.json"
    if not table_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"Phase6 metadata is incomplete in {phase6_dir}")
    table = pd.read_csv(table_path, usecols=["R", "N"])
    pairs = table.drop_duplicates()
    if pairs["R"].duplicated().any():
        raise ValueError("Phase6 resultC_xi_boundary.csv has multiple N values for one R")
    csv_map = {int(row.R): int(row.N) for row in pairs.itertuples(index=False)}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata_map = {int(key): int(value) for key, value in summary.get("R_N", {}).items()}
    if csv_map != metadata_map:
        raise ValueError(f"Phase6 CSV/summary R-to-N mismatch: {csv_map} != {metadata_map}")
    missing = sorted(set(ALL_R_VALUES) - set(csv_map))
    if missing:
        raise ValueError(f"Phase6 metadata lacks R values: {missing}")
    return csv_map


def _phase0_matches(summary: dict[str, Any], R: int) -> bool:
    inputs = summary.get("inputs", {})
    if int(inputs.get("R", -1)) != R:
        return False
    for key in ("B", "sigma_J", "sigma_phi", "phi_bar"):
        if not math.isclose(
            float(inputs.get(key, math.nan)), float(EXPECTED_PHYSICS[key]),
            rel_tol=0.0, abs_tol=1e-14,
        ):
            return False
    a_value = inputs.get("a", inputs.get("lattice_spacing", math.nan))
    return (
        math.isclose(float(a_value), float(EXPECTED_PHYSICS["a"]), rel_tol=0.0, abs_tol=1e-14)
        and str(inputs.get("branch")) == EXPECTED_PHYSICS["branch"]
    )


def find_phase0_dir(search_root: Path, output_root: Path, R: int) -> tuple[Path, str]:
    """Prefer an existing matching reference and reserve generation for R=96."""
    preferred = (
        [search_root / "phase0_B2_R12"] if R == 12 else []
    ) + [search_root / "gaussian_R_sweep" / f"R{R:03d}" / "phase0"]
    seen: set[Path] = set()
    candidates = [*preferred, *sorted(search_root.rglob("phase0_summary.json"))]
    for candidate in candidates:
        phase0_dir = candidate.parent if candidate.name == "phase0_summary.json" else candidate
        if phase0_dir in seen or not (phase0_dir / "phase0_summary.json").is_file():
            continue
        seen.add(phase0_dir)
        summary = json.loads((phase0_dir / "phase0_summary.json").read_text(encoding="utf-8"))
        if _phase0_matches(summary, R) and (phase0_dir / "phase0_delta_table.csv").is_file():
            return phase0_dir, "existing Phase0 reference"
    return output_root / f"R{R:03d}" / "phase0_reference", "generated Phase0 reference"


def _validate_reference_inputs(summary: dict[str, Any], R: int) -> None:
    if not _phase0_matches(summary, R):
        raise ValueError(
            f"Phase0 reference for R={R} does not match B=2, sigma_J=1, "
            "sigma_phi=0.06, phi_bar=0, a=1, stay_to_evacuate"
        )


def run_q0_R(
    R: int,
    *,
    N: int,
    output_root: Path,
    phase0_search_root: Path,
) -> dict[str, Any]:
    """Run four q=0 tasks for one R through unchanged Phase1 machinery."""
    if R not in NEW_R_VALUES:
        raise ValueError(f"new q=0 runs are fixed to R={NEW_R_VALUES}; received R={R}")
    phase0_dir, phase0_source_kind = find_phase0_dir(phase0_search_root, output_root, R)
    existed_before = (phase0_dir / "phase0_summary.json").is_file()
    fallback = Phase0Task(
        B=2.0, R=R, sigma_J=1.0, sigma_phi=0.06, phi_bar=0.0,
        lattice_spacing=1.0, branch="stay_to_evacuate", delta_list=TARGET_DELTAS,
    )
    reference = ensure_phase0_reference(
        phase0_dir, required_deltas=TARGET_DELTAS, fallback_task=fallback
    )
    _validate_reference_inputs(reference.summary, R)
    tasks = build_phase12_tasks(
        reference=reference,
        deltas=TARGET_DELTAS,
        modes=(0,),
        N=N,
        epsilon_fraction=0.05,
        tau_multiplier=6.0,
        T_min=50,
        fit_start=0,
        qR_max_fit=0.35,
        task_group="main",
    )
    started = time.perf_counter()
    results = [simulate_deterministic_mode(task) for task in tasks]
    compute_seconds = time.perf_counter() - started
    output_dir = output_root / f"R{R:03d}"
    runtime = {
        "mpi_available": False,
        "mpi_active": False,
        "world_size": 1,
        "parallelization": "serial independent q=0 tasks",
        "task_count": len(tasks),
        "compute_seconds": compute_seconds,
        "phase0_regenerated": reference.regenerated,
        "phase0_source": _relative_path(phase0_dir),
    }
    paths = write_phase12_outputs(
        results, output_dir, qR_max_fit=0.35, runtime_metadata=runtime,
        save_timeseries=True,
    )
    summary_path = paths["validation_summary"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["q0_full_numeric_provenance"] = {
        "git_commit": _git_sha(),
        "R": R,
        "N_from_phase6": N,
        "phase0_source": _relative_path(phase0_dir),
        "phase0_source_kind": (
            "generated for this campaign" if not existed_before and reference.regenerated
            else phase0_source_kind
        ),
        "phase12_source": _relative_path(paths["mode_results"]),
        "delta_list": list(TARGET_DELTAS),
        "mode_indices": [0],
        "epsilon_fraction": 0.05,
        "tau_multiplier": 6.0,
        "T_min": 50,
        "fit_start": 0,
        "kernel_hat_q0_all_one": all(
            math.isclose(float(result.metrics["kernel_hat"]), 1.0, rel_tol=0.0, abs_tol=1e-15)
            for result in results
        ),
        "numerical_estimator": "Gamma_from_lambda = -ln(abs(lambda_fit))",
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary["q0_full_numeric_provenance"]


MASTER_COLUMNS = [
    "R", "N", "delta", "Delta", "m_star", "Lambda_star", "epsilon_fraction",
    "T", "fit_start", "fit_end", "lambda_theory", "lambda_fit", "lambda_fit_r2",
    "Gamma0_theory", "Gamma0_num", "Gamma0_logfit", "Gamma_logfit_r2",
    "method_A_B_relative_difference", "method_B_C_relative_difference",
    "Gamma_relative_error", "reliable", "reliability_reason", "Gamma0_source",
    "source_file", "git_commit",
]


def load_q0_rows(path: Path, *, R: int, expected_N: int) -> pd.DataFrame:
    """Load exactly the fixed four numerical q=0 conditions from one source."""
    if not path.is_file():
        return pd.DataFrame(columns=MASTER_COLUMNS)
    frame = pd.read_csv(path)
    required = {
        "R", "N", "delta", "Delta", "m_star", "Lambda_star", "epsilon_fraction",
        "T", "fit_start", "fit_end", "mode_index", "task_group", "lambda_theory",
        "lambda_fit", "lambda_fit_r2", "Gamma0_theory", "Gamma_from_lambda",
        "Gamma_logfit", "Gamma_logfit_r2", "method_A_B_relative_difference",
        "method_B_C_relative_difference", "Gamma_relative_error", "reliable",
        "reliability_reason", "kernel_hat",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    frame["reliable"] = _coerce_bool(frame["reliable"])
    selected = frame[
        (frame["R"] == R)
        & (frame["N"] == expected_N)
        & (frame["mode_index"] == 0)
        & (frame["task_group"] == "main")
        & np.isclose(frame["epsilon_fraction"], 0.05, rtol=0.0, atol=1e-12)
    ].copy()
    rows = []
    for delta in TARGET_DELTAS:
        match = selected[np.isclose(selected["delta"], delta, rtol=1e-12, atol=1e-15)]
        if match.empty:
            continue
        if len(match) != 1:
            raise ValueError(f"{path}: expected at most one R={R}, delta={delta:g} q=0 row, found {len(match)}")
        if not math.isclose(float(match["kernel_hat"].iloc[0]), 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError(f"{path}: q=0 kernel_hat is not 1 at delta={delta:g}")
        rows.append(match.iloc[0])
    result = pd.DataFrame(rows).rename(
        columns={"Gamma_from_lambda": "Gamma0_num", "Gamma_logfit": "Gamma0_logfit"}
    )
    if result.empty:
        return pd.DataFrame(columns=MASTER_COLUMNS)
    result["Gamma0_source"] = GAMMA_SOURCE
    result["source_file"] = _relative_path(path)
    result["git_commit"] = _git_sha()
    return result[MASTER_COLUMNS]


def aggregate_q0_outputs(
    *,
    output_root: Path,
    phase6_dir: Path,
    existing_R12_dir: Path,
) -> tuple[Path, Path, Path]:
    """Build the four-R and five-R masters without a theory fallback."""
    n_map = load_phase6_N_map(phase6_dir)
    new_frames = [
        load_q0_rows(
            output_root / f"R{R:03d}" / "phase12_mode_results.csv",
            R=R, expected_N=n_map[R],
        )
        for R in NEW_R_VALUES
    ]
    four_R = pd.concat(new_frames, ignore_index=True).sort_values(["R", "delta"])
    r12_path = existing_R12_dir / "phase12_mode_results.csv"
    R12 = load_q0_rows(r12_path, R=12, expected_N=n_map[12])
    five_R = pd.concat([four_R, R12], ignore_index=True).sort_values(["R", "delta"])
    four_path = output_root / "q0_numeric_all_R.csv"
    five_path = output_root / "q0_numeric_all_5R.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    four_R.to_csv(four_path, index=False)
    five_R.to_csv(five_path, index=False)

    provenance: dict[str, Any] = {}
    for R in ALL_R_VALUES:
        group = five_R[five_R["R"] == R]
        default_source = (
            existing_R12_dir / "phase12_mode_results.csv" if R == 12
            else output_root / f"R{R:03d}" / "phase12_mode_results.csv"
        )
        source_file = (
            str(group["source_file"].iloc[0]) if not group.empty
            else _relative_path(default_source)
        )
        phase0_dir, phase0_kind = find_phase0_dir(
            PROJECT_ROOT / "results" / "runs", output_root, R
        )
        try:
            phase0_dir.resolve().relative_to(output_root.resolve())
            phase0_kind = "generated for this campaign"
        except ValueError:
            pass
        provenance[str(R)] = {
            "phase0_source": _relative_path(phase0_dir),
            "phase0_source_kind": phase0_kind,
            "phase12_numerical_source": source_file,
            "phase12_source_kind": (
                "existing Phase12 q=0 numerical" if R == 12
                else "new Phase12 q=0 numerical"
            ),
            "N": int(group["N"].iloc[0]) if not group.empty else n_map[R],
            "condition_count": int(len(group)),
            "all_reliable": bool(not group.empty and group["reliable"].all()),
        }
    expected_pairs = {(R, delta) for R in ALL_R_VALUES for delta in TARGET_DELTAS}
    actual_pairs = {(int(row.R), float(row.delta)) for row in five_R.itertuples()}
    missing_pairs = sorted(expected_pairs - actual_pairs)
    invalid_sources = sorted(set(five_R["Gamma0_source"]) - {GAMMA_SOURCE})
    relative_spreads = []
    for delta, group in five_R.groupby("delta"):
        if len(group) < 2:
            continue
        mean = float(group["Gamma0_num"].mean())
        relative_spreads.append(
            {
                "delta": float(delta),
                "relative_range_over_mean": float(
                    (group["Gamma0_num"].max() - group["Gamma0_num"].min()) / mean
                ),
            }
        )
    summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_sha(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "phase6_N_map": {str(key): value for key, value in n_map.items()},
        "fixed_conditions": {
            "R_new": list(NEW_R_VALUES), "R_all": list(ALL_R_VALUES),
            "delta": list(TARGET_DELTAS), "mode_index": 0,
            "epsilon_fraction": 0.05, "tau_multiplier": 6.0,
            "T_min": 50, "fit_start": 0,
        },
        "Gamma0_definition": "Gamma_from_lambda = -ln(abs(lambda_fit))",
        "Gamma0_source_required": GAMMA_SOURCE,
        "phase0_theory_used_in_final_tau0": False,
        "per_R_provenance": provenance,
        "new_condition_count": int(len(four_R)),
        "all_condition_count": int(len(five_R)),
        "missing_conditions": [f"R={R},delta={delta:g}" for R, delta in missing_pairs],
        "invalid_Gamma0_sources": invalid_sources,
        "all_five_R_numerical_available": not missing_pairs and not invalid_sources,
        "all_five_R_reliable": bool(five_R["reliable"].all()),
        "max_Gamma0_relative_error": float(five_R["Gamma_relative_error"].abs().max()),
        "max_Gamma_method_difference": float(
            (five_R["Gamma0_num"] - five_R["Gamma0_logfit"]).abs().max()
        ),
        "max_Gamma_method_relative_difference": float(
            five_R["method_B_C_relative_difference"].abs().max()
        ),
        "kernel_hat_q0": 1.0,
        "Gamma0_R_dependence_at_fixed_absolute_delta": relative_spreads,
        "max_Gamma0_R_relative_range_over_mean": (
            max(row["relative_range_over_mean"] for row in relative_spreads)
            if relative_spreads else None
        ),
        "Gamma0_R_dependence_note": (
            "kernel_hat(0)=1 removes finite-q kernel dispersion, but Gamma0 at fixed "
            "absolute delta still varies with R because sigma_eff=sqrt(sigma_J^2/(2R)+sigma_phi^2) "
            "and the corresponding Phase0 fixed point are R dependent"
        ),
    }
    summary_path = output_root / "q0_full_numeric_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return four_path, five_path, summary_path


def parse_R_list(text: str) -> tuple[int, ...]:
    values = tuple(int(token.strip()) for token in text.split(",") if token.strip())
    if not values:
        raise argparse.ArgumentTypeError("R-list must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--R-list", type=parse_R_list, default=NEW_R_VALUES)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--no-aggregate", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("results/runs/poster_ABCD/q0_full_numeric_validation"))
    parser.add_argument("--phase6-dir", type=Path, default=Path("results/runs/poster_ABCD/phase6_boundary"))
    parser.add_argument("--phase0-search-root", type=Path, default=Path("results/runs"))
    parser.add_argument("--existing-R12-dir", type=Path, default=Path("results/runs/phase12_B2_R12"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        n_map = load_phase6_N_map(args.phase6_dir)
        if not args.aggregate_only:
            invalid = sorted(set(args.R_list) - set(NEW_R_VALUES))
            if invalid:
                raise ValueError(f"new-run R-list contains unsupported values: {invalid}")
            for R in args.R_list:
                provenance = run_q0_R(
                    R, N=n_map[R], output_root=args.output_root,
                    phase0_search_root=args.phase0_search_root,
                )
                print(
                    f"R={R}: N={n_map[R]}, phase0={provenance['phase0_source']}, "
                    f"phase12={provenance['phase12_source']}"
                )
        if not args.no_aggregate:
            for path in aggregate_q0_outputs(
                output_root=args.output_root, phase6_dir=args.phase6_dir,
                existing_R12_dir=args.existing_R12_dir,
            ):
                print(path)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
