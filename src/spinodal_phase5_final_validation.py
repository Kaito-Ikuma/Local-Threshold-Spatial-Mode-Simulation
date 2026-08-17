#!/usr/bin/env python3
"""Final Phase5 R-sweep validation planning and checkpoint-only analysis.

This module adds V1--V5 without changing the microscopic dynamics.  Planning
and CSV/JSON analysis deliberately avoid Matplotlib so they can run in the
compute-only SQUID virtual environment.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from spinodal_phase5_analysis import fit_microscopic_relaxation, kernel_hat
from spinodal_phase5_core import Phase5BlockResult, load_block_checkpoint
from spinodal_phase5_followup_analysis import (
    aggregate_survival_blocks,
    effective_relaxation,
    interpolate_escape_crossing,
)


SCRIPT_VERSION = "2026.08.17-phase5-final-validation-v2"
PRIMARY_R = (6, 12, 24, 48)
FINITE_SIZE_PAIRS = ((12, 512), (12, 1024), (12, 2048), (24, 1024), (24, 2048), (24, 4096))
FINITE_SIZE_NEW_PAIRS = ((12, 512), (12, 2048), (24, 1024), (24, 4096))
D_PRECISION_R = (12, 24, 48)
TIME_R = (6, 12, 24, 48)
TIME_OBSERVATIONS = (20, 30, 40, 50)
SEED_R = (12, 48)
SEEDS = (20260815, 20260817, 20260818)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _format_values(values: Iterable[float]) -> str:
    return ",".join(f"{float(value):.12g}" for value in values)


def _fine_grid(lower: float, upper: float, max_step: float) -> tuple[float, ...]:
    intervals = max(1, int(math.ceil((upper - lower) / max_step)))
    return tuple(float(value) for value in np.linspace(lower, upper, intervals + 1))


def _qR_for_mode(mode: int, N: int, R: int) -> float:
    return 2.0 * math.pi * int(mode) * int(R) / int(N)


def case_label(R: int, N: int) -> str:
    return f"R{int(R):03d}_N{int(N):04d}"


def required_condition_report(required: Iterable[Any], observed: Iterable[Any]) -> dict[str, Any]:
    required_set = set(required)
    observed_set = set(observed)
    return {
        "complete": required_set <= observed_set,
        "missing": sorted(required_set - observed_set),
        "unexpected": sorted(observed_set - required_set),
    }


def run_state_complete(run_dir: Path) -> tuple[bool, dict[str, Any]]:
    path = Path(run_dir) / "phase5_run_state.json"
    if not path.is_file():
        return False, {"reason": "missing_run_state", "path": str(path)}
    state = json.loads(path.read_text(encoding="utf-8"))
    completed = int(state.get("completed_valid_blocks", -1))
    total = int(state.get("total_blocks", -2))
    complete = bool(state.get("all_complete")) and completed == total and total >= 0
    return complete, {
        "path": str(path),
        "all_complete": bool(state.get("all_complete")),
        "completed_valid_blocks": completed,
        "total_blocks": total,
    }


def _read_csv(path: Path, required: Sequence[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return frame


def _central_ps_dir(r_sweep_dir: Path, R: int) -> Path:
    return r_sweep_dir / f"R{R:03d}" / "pseudospinodal_fine"


def finite_size_ps_dir(r_sweep_dir: Path, R: int, N: int) -> Path:
    if (R, N) in ((12, 1024), (24, 2048)):
        return _central_ps_dir(r_sweep_dir, R)
    return r_sweep_dir / "finite_size" / case_label(R, N) / "pseudospinodal_fine"


def finite_size_response_dir(r_sweep_dir: Path, R: int, N: int) -> Path:
    if (R, N) in ((12, 1024), (24, 2048)):
        return r_sweep_dir / f"R{R:03d}" / "response_matched"
    return r_sweep_dir / "finite_size" / case_label(R, N) / "response_matched"


def build_finite_size_fine_plan(
    r_sweep_dir: Path,
    *,
    pairs: Sequence[tuple[int, int]] = FINITE_SIZE_NEW_PAIRS,
    criterion: float = 0.10,
    max_step: float = 0.002,
    extension_factor: float = 1.5,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for R, N in pairs:
        root = r_sweep_dir / "finite_size" / case_label(R, N)
        paths = sorted(root.glob("pseudospinodal_coarse*/analysis/phase5_pseudospinodal_fine_scan.csv"))
        if not paths:
            rows.append({"R": R, "N": N, "status": "missing_coarse", "fine_deltas": ""})
            continue
        frames = []
        for path in paths:
            frame = _read_csv(path, ("delta", "T", "escape_fraction_cumulative_T"))
            frames.append(frame[frame["T"] == 50][["delta", "escape_fraction_cumulative_T"]])
        table = pd.concat(frames, ignore_index=True).groupby("delta", as_index=False).mean().sort_values("delta")
        crossing = interpolate_escape_crossing(table["delta"], table["escape_fraction_cumulative_T"], criterion)
        if crossing["estimate"] is not None:
            grid = _fine_grid(float(crossing["lower"]), float(crossing["upper"]), max_step)
            status = "bracketed"
            direction = None
            extension = None
        elif np.all(table["escape_fraction_cumulative_T"] < criterion):
            grid = ()
            status = "extension_required"
            direction = "smaller_delta"
            extension = float(table["delta"].min() / extension_factor)
        elif np.all(table["escape_fraction_cumulative_T"] > criterion):
            grid = ()
            status = "extension_required"
            direction = "larger_delta"
            extension = float(table["delta"].max() * extension_factor)
        else:
            grid = ()
            status = "ambiguous"
            direction = "manual_nonmonotonic_review"
            extension = None
        rows.append(
            {
                "R": R,
                "N": N,
                "N_over_R": N / R,
                "status": status,
                "delta_lower": crossing["lower"],
                "delta_upper": crossing["upper"],
                "coarse_estimate": crossing["estimate"],
                "fine_deltas": _format_values(grid),
                "extension_direction": direction,
                "extension_delta": extension,
                "monotonicity_ok": crossing["monotonicity_ok"],
            }
        )
    return pd.DataFrame(rows)


def _plan_value(plan: Path, R: int, N: int, column: str) -> str:
    frame = pd.read_csv(plan, dtype={column: str}, keep_default_na=False)
    row = frame[(frame["R"] == R) & (frame["N"] == N)]
    if len(row) != 1:
        raise ValueError(f"expected one plan row for R={R}, N={N}")
    value = str(row.iloc[0][column]).strip()
    if not value:
        status = row.iloc[0].get("status", "unknown")
        raise ValueError(f"plan value {column} unavailable for R={R}, N={N}; status={status}")
    return value


def collect_finite_size_pseudospinodal(r_sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for R, N in FINITE_SIZE_PAIRS:
        root = finite_size_ps_dir(r_sweep_dir, R, N)
        time_path = root / "analysis/phase5_pseudospinodal_time_dependence.csv"
        scan_path = root / "analysis/phase5_pseudospinodal_fine_scan.csv"
        if not time_path.is_file() or not scan_path.is_file():
            continue
        complete, _ = run_state_complete(root)
        time = _read_csv(time_path, ("T_obs", "delta_ps_estimate", "delta_ps_se", "delta_ps_ci_low", "delta_ps_ci_high"))
        primary = time[time["T_obs"] == 50]
        if len(primary) != 1:
            continue
        scan = _read_csv(scan_path, ("delta", "escape_fraction_cumulative_T", "max_baseline_drift", "preparation_final_m"))
        row = primary.iloc[0]
        lower = row.get("delta_lower_bracket", math.nan)
        upper = row.get("delta_upper_bracket", math.nan)

        def probability(delta: float) -> float:
            match = scan[np.isclose(scan["delta"], delta, rtol=0.0, atol=5e-12)]
            return float(match["escape_fraction_cumulative_T"].iloc[0]) if len(match) == 1 else math.nan

        estimate = float(row["delta_ps_estimate"])
        reliable = complete and math.isfinite(estimate) and bool(row.get("monotonicity_ok", True))
        rows.append(
            {
                "R": R,
                "N": N,
                "N_over_R": N / R,
                "delta_ps_T50": estimate,
                "delta_ps_se": float(row["delta_ps_se"]),
                "delta_ps_ci_low": float(row["delta_ps_ci_low"]),
                "delta_ps_ci_high": float(row["delta_ps_ci_high"]),
                "P_escape_lower": probability(float(lower)),
                "P_escape_upper": probability(float(upper)),
                "bootstrap_replicates": int(row.get("n_bootstrap_valid", 0) + row.get("n_bootstrap_failed", 0)),
                "bracket_width": float(upper - lower),
                "baseline_drift": float(scan["max_baseline_drift"].max()),
                "preparation_final_m": float(scan["preparation_final_m"].mean()),
                "reliable": reliable,
                "run_complete": complete,
                "source": str(root),
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(["R", "N"]).reset_index(drop=True)
    return table


def _checkpoint_groups(input_dir: Path) -> dict[str, list[Phase5BlockResult]]:
    paths = sorted((input_dir / "blocks").glob("*.npz"))
    groups: dict[str, list[Phase5BlockResult]] = {}
    for path in paths:
        block = load_block_checkpoint(path)
        groups.setdefault(block.task_id, []).append(block)
    for blocks in groups.values():
        blocks.sort(key=lambda item: item.block_id)
    return groups


def _select_blocks(input_dir: Path, delta: float, mode: int = 0) -> list[Phase5BlockResult]:
    candidates = [
        blocks
        for blocks in _checkpoint_groups(input_dir).values()
        if blocks and blocks[0].mode_index == mode and math.isclose(blocks[0].delta, delta, rel_tol=0.0, abs_tol=5e-10)
    ]
    if len(candidates) != 1:
        raise ValueError(f"expected one checkpoint task for delta={delta:.12g}, mode={mode} in {input_dir}")
    return candidates[0]


def _weighted(blocks: Sequence[Phase5BlockResult], attribute: str) -> np.ndarray:
    values = np.stack([np.asarray(getattr(block, attribute), dtype=float) for block in blocks])
    weights = np.asarray([block.block_n for block in blocks], dtype=float)
    return np.average(values, axis=0, weights=weights)


def _bootstrap_response_ratios(
    blocks: Sequence[Phase5BlockResult], closure: float, *, replicates: int, seed: int
) -> dict[str, float]:
    rng = np.random.Generator(np.random.Philox(seed))
    ratio_samples = []
    conditional_samples = []
    for _ in range(replicates):
        selected = [blocks[index] for index in rng.integers(0, len(blocks), len(blocks))]
        aggregate = aggregate_survival_blocks(selected)
        unconditional = np.asarray(aggregate["A_unconditional"], dtype=float)
        surviving = np.asarray(aggregate["A_survive_to_T"], dtype=float)
        if not np.all(np.isfinite(surviving[:4])):
            continue
        gamma_u = float(fit_microscopic_relaxation(unconditional, 0, 3)["Gamma_micro"])
        gamma_s = float(fit_microscopic_relaxation(surviving, 0, 3)["Gamma_micro"])
        if math.isfinite(gamma_u) and math.isfinite(gamma_s) and gamma_u != 0.0:
            ratio_samples.append(gamma_s / closure)
            conditional_samples.append(gamma_s / gamma_u)

    def summary(samples: Sequence[float]) -> tuple[float, float, float]:
        values = np.asarray(samples, dtype=float)
        if len(values) < 2:
            return math.nan, math.nan, math.nan
        return float(np.std(values, ddof=1)), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))

    ratio_se, ratio_low, ratio_high = summary(ratio_samples)
    cond_se, cond_low, cond_high = summary(conditional_samples)
    return {
        "Gamma_ratio_se": ratio_se,
        "Gamma_ratio_ci_low": ratio_low,
        "Gamma_ratio_ci_high": ratio_high,
        "survival_correction_se": cond_se,
        "survival_correction_ci_low": cond_low,
        "survival_correction_ci_high": cond_high,
    }


def response_metrics(
    input_dir: Path,
    *,
    target_delta: float,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260816,
) -> dict[str, Any]:
    complete, state = run_state_complete(input_dir)
    mode = _read_csv(input_dir / "phase5_mode_results.csv", ("delta", "mode_index", "Gamma_closure", "method_B_C_relative_difference", "N", "R", "M_total"))
    selected = mode[(mode["mode_index"] == 0) & np.isclose(mode["delta"], target_delta, rtol=0.0, atol=5e-10)]
    if len(selected) != 1:
        raise ValueError(f"expected q=0 mode row at delta={target_delta:.12g} in {input_dir}")
    item = selected.iloc[0]
    blocks = _select_blocks(input_dir, float(item["delta"]), 0)
    aggregate = aggregate_survival_blocks(blocks)
    unconditional = np.asarray(aggregate["A_unconditional"], dtype=float)
    surviving = np.asarray(aggregate["A_survive_to_T"], dtype=float)
    fit_u = fit_microscopic_relaxation(unconditional, 0, 3)
    fit_s = fit_microscopic_relaxation(surviving, 0, 3)
    gamma_u = float(fit_u["Gamma_micro"])
    gamma_s = float(fit_s["Gamma_micro"])
    closure = float(item["Gamma_closure"])
    rates = effective_relaxation(unconditional)["Gamma_eff"]
    baseline = _weighted(blocks, "baseline_m")
    bootstrap = _bootstrap_response_ratios(blocks, closure, replicates=bootstrap_replicates, seed=bootstrap_seed)
    return {
        "R": int(item["R"]),
        "N": int(item["N"]),
        "delta": float(item["delta"]),
        "M_total": int(item["M_total"]),
        "Gamma_unconditional": gamma_u,
        "Gamma_survive_to_T": gamma_s,
        "Gamma_closure": closure,
        "Gamma_survive_over_closure": gamma_s / closure,
        "Gamma_survive_over_unconditional": gamma_s / gamma_u,
        "Gamma_eff_0": float(rates[0]),
        "Gamma_eff_1": float(rates[1]),
        "Gamma_eff_2": float(rates[2]),
        "Gamma_eff_3": float(rates[3]),
        "Gamma_eff_3_over_0": float(rates[3] / rates[0]),
        "method_B_C_relative_difference": float(item["method_B_C_relative_difference"]),
        "escape_fraction_cumulative_T": float(aggregate["escape_fraction_cumulative"][-1]),
        "baseline_drift": float(np.max(np.abs(baseline - baseline[0]))),
        "run_complete": complete,
        "run_state": state,
        **bootstrap,
    }


def _ci_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    return all(math.isfinite(value) for value in (a_low, a_high, b_low, b_high)) and max(a_low, b_low) <= min(a_high, b_high)


def collect_finite_size_response(
    r_sweep_dir: Path,
    pseudo: pd.DataFrame,
    *,
    bootstrap_replicates: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for index, (R, N) in enumerate(FINITE_SIZE_PAIRS):
        ps = pseudo[(pseudo["R"] == R) & (pseudo["N"] == N)]
        if len(ps) != 1:
            continue
        root = finite_size_response_dir(r_sweep_dir, R, N)
        if not root.is_dir():
            continue
        target = float(ps["delta_ps_T50"].iloc[0]) + 0.010
        try:
            row = response_metrics(root, target_delta=target, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=20260816 + index)
        except (FileNotFoundError, ValueError):
            continue
        rows.append(row)
    table = pd.DataFrame(rows)
    escalation_rows = []
    if table.empty:
        return table, pd.DataFrame(columns=["R", "N", "needs_M8192", "reason"])
    table = table.sort_values(["R", "N"]).reset_index(drop=True)
    table["Gamma_finite_size_converged"] = False
    table["delta_ps_finite_size_converged"] = False
    needs: set[tuple[int, int]] = set()
    reasons: dict[tuple[int, int], set[str]] = {}
    for R, group in table.groupby("R"):
        ordered = group.sort_values("N")
        for left_index, right_index in zip(ordered.index[:-1], ordered.index[1:]):
            left = table.loc[left_index]
            right = table.loc[right_index]
            difference = abs(float(right["Gamma_survive_over_closure"]) - float(left["Gamma_survive_over_closure"]))
            scale = max(abs(float(left["Gamma_survive_over_closure"])), 1e-15)
            combined_se = math.sqrt(float(left["Gamma_ratio_se"]) ** 2 + float(right["Gamma_ratio_se"]) ** 2)
            clear = math.isfinite(combined_se) and difference > 1.96 * combined_se
            over_three = difference / scale > 0.03
            gamma_overlap = _ci_overlap(
                float(left["Gamma_ratio_ci_low"]), float(left["Gamma_ratio_ci_high"]),
                float(right["Gamma_ratio_ci_low"]), float(right["Gamma_ratio_ci_high"]),
            )
            gamma_converged = difference / scale < 0.03 or gamma_overlap
            table.at[right_index, "Gamma_finite_size_converged"] = gamma_converged
            if clear or over_three:
                for item in (left, right):
                    key = (int(item["R"]), int(item["N"]))
                    if int(item["M_total"]) < 8192:
                        needs.add(key)
                        reasons.setdefault(key, set()).add("adjacent_N_Gamma_ratio_difference")

            ps_left = pseudo[(pseudo["R"] == R) & (pseudo["N"] == int(left["N"]))].iloc[0]
            ps_right = pseudo[(pseudo["R"] == R) & (pseudo["N"] == int(right["N"]))].iloc[0]
            ps_relative = abs(float(ps_right["delta_ps_T50"]) - float(ps_left["delta_ps_T50"])) / max(abs(float(ps_left["delta_ps_T50"])), 1e-15)
            ps_overlap = _ci_overlap(
                float(ps_left["delta_ps_ci_low"]), float(ps_left["delta_ps_ci_high"]),
                float(ps_right["delta_ps_ci_low"]), float(ps_right["delta_ps_ci_high"]),
            )
            table.at[right_index, "delta_ps_finite_size_converged"] = ps_relative < 0.05 or ps_overlap
    for R, N in FINITE_SIZE_PAIRS:
        escalation_rows.append(
            {
                "R": R,
                "N": N,
                "needs_M8192": (R, N) in needs,
                "reason": ";".join(sorted(reasons.get((R, N), set()))) or "not_required",
            }
        )
    return table, pd.DataFrame(escalation_rows)


def D_uncertainty_transform(D: float, D_se: float, ci_low: float, ci_high: float, kappa: float) -> dict[str, float]:
    if kappa <= 0.0:
        raise ValueError("kappa must be positive")
    ratio = D / kappa
    ratio_se = D_se / kappa
    return {
        "D_over_kappa": ratio,
        "D_over_kappa_SE": ratio_se,
        "D_over_kappa_CI_low": ci_low / kappa,
        "D_over_kappa_CI_high": ci_high / kappa,
        "z_from_unity": (ratio - 1.0) / ratio_se if ratio_se > 0.0 else math.nan,
    }


def D_production_decision(
    M: int,
    ratio_se: float,
    run_complete: bool,
    *,
    minimum_M: int = 32768,
    maximum_M: int = 65536,
    target_se: float = 0.25,
) -> dict[str, Any]:
    """Classify V2 precision without conflating minimum M and SE quality."""
    minimum_reached = int(M) >= minimum_M
    maximum_reached = int(M) >= maximum_M
    precision_target_met = math.isfinite(float(ratio_se)) and float(ratio_se) <= target_se
    needs_escalation = bool(
        run_complete
        and minimum_reached
        and not maximum_reached
        and not precision_target_met
    )
    sufficient = bool(run_complete and minimum_reached and precision_target_met)
    finalized = bool(
        run_complete and minimum_reached and (precision_target_met or maximum_reached)
    )
    if not run_complete:
        status = "run_incomplete"
    elif not minimum_reached:
        status = "below_minimum_M"
    elif precision_target_met:
        status = "precision_target_met"
    elif not maximum_reached:
        status = "M65536_escalation_required"
    else:
        status = "maximum_M_reached_precision_target_not_met"
    return {
        "minimum_production_M_reached": minimum_reached,
        "maximum_production_M_reached": maximum_reached,
        "precision_target_SE": target_se,
        "precision_target_met": precision_target_met,
        "production_M_sufficient": sufficient,
        "production_run_finalized": finalized,
        "needs_M65536": needs_escalation,
        "production_precision_status": status,
    }


def _kernel_slope(mode: pd.DataFrame, R: int, N: int) -> tuple[float, float]:
    eligible = mode[(mode["qR"] <= 0.35) & np.isfinite(mode["Gamma_micro"])].sort_values("mode_index")
    q0 = eligible[eligible["mode_index"] == 0]
    if len(q0) != 1 or len(eligible) < 2:
        return math.nan, math.nan
    x = np.asarray([-math.log(abs(kernel_hat(int(m), N, R))) for m in eligible["mode_index"]], dtype=float)
    y = eligible["Gamma_micro"].to_numpy(dtype=float) - float(q0["Gamma_micro"].iloc[0])
    denominator = float(np.dot(x, x))
    if denominator <= 0.0:
        return math.nan, math.nan
    slope = float(np.dot(x, y) / denominator)
    predicted = slope * x
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum((y - predicted) ** 2)) / sst if sst > 0.0 else math.nan
    return slope, r2


def collect_D_precision(r_sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for R in D_PRECISION_R:
        root = r_sweep_dir / f"R{R:03d}" / "dispersion"
        fits_path = root / "phase5_dispersion_fits.csv"
        mode_path = root / "phase5_mode_results.csv"
        if not fits_path.is_file() or not mode_path.is_file():
            continue
        complete, _ = run_state_complete(root)
        fits = _read_csv(fits_path, ("D_micro", "D_micro_se", "D_micro_ci_low", "D_micro_ci_high", "kappa_R", "delta"))
        if len(fits) != 1:
            continue
        fit = fits.iloc[0]
        mode = pd.read_csv(mode_path)
        N = int(mode["N"].iloc[0])
        M = int(mode["M_total"].iloc[0])
        transformed = D_uncertainty_transform(
            float(fit["D_micro"]), float(fit["D_micro_se"]),
            float(fit["D_micro_ci_low"]), float(fit["D_micro_ci_high"]),
            float(fit["kappa_R"]),
        )
        kernel_slope, kernel_r2 = _kernel_slope(mode, R, N)
        production = D_production_decision(
            M,
            transformed["D_over_kappa_SE"],
            complete,
        )
        rows.append(
            {
                "R": R,
                "N": N,
                "M_total": M,
                "delta": float(fit["delta"]),
                "D": float(fit["D_micro"]),
                "D_SE": float(fit["D_micro_se"]),
                "D_CI_low": float(fit["D_micro_ci_low"]),
                "D_CI_high": float(fit["D_micro_ci_high"]),
                "kappa_R": float(fit["kappa_R"]),
                **transformed,
                "kernel_slope": kernel_slope,
                "kernel_slope_r2": kernel_r2,
                **production,
                "run_complete": complete,
            }
        )
    return pd.DataFrame(rows)


def _candidate_survival_sources(r_sweep_dir: Path, R: int) -> list[Path]:
    root = r_sweep_dir / f"R{R:03d}"
    return sorted(
        [path for path in root.glob("pseudospinodal_coarse*") if (path / "blocks").is_dir()]
        + [path for path in root.glob("pseudospinodal_fine") if (path / "blocks").is_dir()]
        + [path for path in root.glob("time_extension*") if (path / "blocks").is_dir()]
    )


def _best_delta_blocks(r_sweep_dir: Path, R: int) -> dict[float, list[Phase5BlockResult]]:
    selected: dict[float, list[Phase5BlockResult]] = {}
    for source in _candidate_survival_sources(r_sweep_dir, R):
        complete, _ = run_state_complete(source)
        if not complete:
            continue
        for blocks in _checkpoint_groups(source).values():
            if not blocks or not blocks[0].survival_tracking_enabled or blocks[0].epsilon_fraction != 0.0:
                continue
            delta = float(blocks[0].delta)
            current_M = sum(block.block_n for block in blocks)
            previous_M = sum(block.block_n for block in selected.get(delta, []))
            if current_M > previous_M:
                selected[delta] = blocks
    return selected


def _escape_curves(r_sweep_dir: Path, R: int) -> tuple[list[float], dict[float, np.ndarray], dict[float, list[Phase5BlockResult]]]:
    blocks_by_delta = _best_delta_blocks(r_sweep_dir, R)
    curves = {}
    for delta, blocks in blocks_by_delta.items():
        total = sum(block.block_n for block in blocks)
        survival = np.sum(np.stack([block.survival_count for block in blocks]), axis=0) / total
        curves[delta] = 1.0 - survival
    return sorted(curves), curves, blocks_by_delta


def time_extension_decision(
    deltas: Sequence[float],
    probabilities: Sequence[float],
    *,
    criterion: float = 0.10,
    extension_factor: float = 1.25,
) -> dict[str, Any]:
    crossing = interpolate_escape_crossing(deltas, probabilities, criterion)
    probability = np.asarray(probabilities, dtype=float)
    if crossing["estimate"] is not None:
        return {"status": "bracketed", "suggested_new_deltas": (), "reason": "existing_grid_brackets_criterion", **crossing}
    if np.all(probability < criterion):
        return {"status": "extension_required", "suggested_new_deltas": (min(deltas) / extension_factor,), "reason": "all_escape_probabilities_below_criterion", **crossing}
    if np.all(probability > criterion):
        return {"status": "extension_required", "suggested_new_deltas": (max(deltas) * extension_factor,), "reason": "all_escape_probabilities_above_criterion", **crossing}
    return {"status": "ambiguous", "suggested_new_deltas": (), "reason": "nonmonotonic_or_multiple_crossings", **crossing}


def plan_time_extension(
    r_sweep_dir: Path,
    *,
    R_values: Sequence[int] = (6, 12),
    observation_times: Sequence[int] = (20, 30, 40, 50),
    criterion: float = 0.10,
    extension_factor: float = 1.25,
) -> pd.DataFrame:
    rows = []
    for R in R_values:
        deltas, curves, blocks_by_delta = _escape_curves(r_sweep_dir, R)
        if not deltas:
            for T in observation_times:
                rows.append({"R": R, "T_obs": T, "status": "missing_data", "suggested_new_deltas": "", "reason": "no_complete_survival_blocks"})
            continue
        for T in observation_times:
            probabilities = np.asarray([curves[delta][T] for delta in deltas], dtype=float)
            decision = time_extension_decision(
                deltas,
                probabilities,
                criterion=criterion,
                extension_factor=extension_factor,
            )
            if decision["status"] == "bracketed":
                endpoints = (float(decision["lower"]), float(decision["upper"]))
                low_M = [
                    delta
                    for delta in endpoints
                    if sum(block.block_n for block in blocks_by_delta[delta]) < 8192
                ]
                if low_M:
                    decision["status"] = "M_upgrade_required"
                    decision["suggested_new_deltas"] = tuple(low_M)
                    decision["reason"] = "bracket_exists_but_endpoint_M_below_8192"
            rows.append(
                {
                    "R": R,
                    "T_obs": T,
                    "current_delta_min": min(deltas),
                    "current_delta_max": max(deltas),
                    "status": decision["status"],
                    "suggested_new_deltas": _format_values(decision["suggested_new_deltas"]),
                    "reason": decision["reason"],
                    "monotonicity_ok": decision["monotonicity_ok"],
                }
            )
    return pd.DataFrame(rows)


def _bootstrap_escape(blocks: Sequence[Phase5BlockResult], replicates: int, seed: int) -> np.ndarray:
    counts = np.stack([block.survival_count for block in blocks]).astype(float)
    sizes = np.asarray([block.block_n for block in blocks], dtype=float)
    rng = np.random.Generator(np.random.Philox(seed))
    output = np.empty((replicates, counts.shape[1]), dtype=float)
    for replicate in range(replicates):
        indices = rng.integers(0, len(blocks), len(blocks))
        output[replicate] = 1.0 - np.sum(counts[indices], axis=0) / np.sum(sizes[indices])
    return output


def collect_time_dependence(
    r_sweep_dir: Path,
    *,
    bootstrap_replicates: int = 1000,
    bootstrap_seed: int = 20260816,
) -> pd.DataFrame:
    rows = []
    for R in TIME_R:
        deltas, curves, blocks_by_delta = _escape_curves(r_sweep_dir, R)
        if not deltas:
            continue
        boot = {
            delta: _bootstrap_escape(blocks_by_delta[delta], bootstrap_replicates, bootstrap_seed + R * 100 + index)
            for index, delta in enumerate(deltas)
        }
        estimates_by_T: dict[int, float] = {}
        for T in TIME_OBSERVATIONS:
            crossing = interpolate_escape_crossing(deltas, [curves[delta][T] for delta in deltas], 0.10)
            lower_M = (
                sum(block.block_n for block in blocks_by_delta[float(crossing["lower"])])
                if crossing["lower"] is not None
                else 0
            )
            upper_M = (
                sum(block.block_n for block in blocks_by_delta[float(crossing["upper"])])
                if crossing["upper"] is not None
                else 0
            )
            samples = []
            failed = 0
            for replicate in range(bootstrap_replicates):
                candidate = interpolate_escape_crossing(deltas, [boot[delta][replicate, T] for delta in deltas], 0.10)
                if candidate["estimate"] is None:
                    failed += 1
                else:
                    samples.append(float(candidate["estimate"]))
            values = np.asarray(samples, dtype=float)
            se = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
            low = float(np.quantile(values, 0.025)) if len(values) > 1 else math.nan
            high = float(np.quantile(values, 0.975)) if len(values) > 1 else math.nan
            estimate = float(crossing["estimate"]) if crossing["estimate"] is not None else math.nan
            estimates_by_T[T] = estimate
            rows.append(
                {
                    "R": R,
                    "T_obs": T,
                    "delta_ps": estimate,
                    "SE": se,
                    "CI95_low": low,
                    "CI95_high": high,
                    "delta_lower_bracket": crossing["lower"],
                    "delta_upper_bracket": crossing["upper"],
                    "n_bootstrap_valid": len(samples),
                    "n_bootstrap_failed": failed,
                    "monotonicity_ok": crossing["monotonicity_ok"],
                    "M_lower_bracket": lower_M,
                    "M_upper_bracket": upper_M,
                    "production_M_sufficient": lower_M >= 8192 and upper_M >= 8192,
                }
            )
        delta30, delta50 = estimates_by_T.get(30, math.nan), estimates_by_T.get(50, math.nan)
        absolute = delta50 - delta30
        relative = absolute / delta50 if math.isfinite(delta50) and delta50 != 0.0 else math.nan
        for row in rows:
            if row["R"] == R:
                row["absolute_rounding_50_minus_30"] = absolute
                row["relative_time_rounding"] = relative
    return pd.DataFrame(rows)


def validate_R96_mapping(N: int = 8192) -> dict[str, Any]:
    reference_ratio = 1024 / 12
    qR_reference = [_qR_for_mode(mode, 1024, 12) for mode in range(7)]
    qR_R96 = [_qR_for_mode(mode, N, 96) for mode in range(7)]
    return {
        "R": 96,
        "N": N,
        "N_over_R": N / 96,
        "reference_N_over_R": reference_ratio,
        "N_over_R_consistent": math.isclose(N / 96, reference_ratio, rel_tol=0.0, abs_tol=1e-14),
        "qR_consistent": bool(np.allclose(qR_R96, qR_reference, rtol=0.0, atol=1e-14)),
    }


def collect_R96(r_sweep_dir: Path, *, bootstrap_replicates: int = 1000) -> pd.DataFrame:
    ps_root = r_sweep_dir / "R096/pseudospinodal_fine"
    time_path = ps_root / "analysis/phase5_pseudospinodal_time_dependence.csv"
    response_root = r_sweep_dir / "R096/response_matched"
    if not time_path.is_file() or not response_root.is_dir():
        return pd.DataFrame()
    time = pd.read_csv(time_path)
    primary = time[time["T_obs"] == 50]
    if len(primary) != 1 or pd.isna(primary["delta_ps_estimate"].iloc[0]):
        return pd.DataFrame()
    pseudospinodal_complete, _ = run_state_complete(ps_root)
    delta_ps = float(primary["delta_ps_estimate"].iloc[0])
    metrics = response_metrics(response_root, target_delta=delta_ps + 0.010, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=20260960)
    return pd.DataFrame([{"R": 96, "N": 8192, "delta_ps": delta_ps, "Gamma_ratio": metrics["Gamma_survive_over_closure"], "Gamma_eff_ratio": metrics["Gamma_eff_3_over_0"], "Method_B_C_difference": metrics["method_B_C_relative_difference"], "escape_fraction": metrics["escape_fraction_cumulative_T"], "run_complete": pseudospinodal_complete and metrics["run_complete"]}])


def aggregate_seed_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    output = rows.copy()
    output["between_seed_mean"] = math.nan
    output["between_seed_std"] = math.nan
    output["max_relative_seed_deviation"] = math.nan
    output["seed_reproducible_soft"] = False
    for R, group in output.groupby("R"):
        values = group["Gamma_ratio"].to_numpy(dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else math.nan
        max_deviation = float(np.max(np.abs(values - mean)) / abs(mean)) if mean != 0.0 else math.nan
        intervals = list(zip(group["Gamma_ratio_ci_low"], group["Gamma_ratio_ci_high"]))
        common_overlap = bool(intervals) and max(float(low) for low, _ in intervals) <= min(float(high) for _, high in intervals)
        mask = output["R"] == R
        output.loc[mask, "between_seed_mean"] = mean
        output.loc[mask, "between_seed_std"] = std
        output.loc[mask, "max_relative_seed_deviation"] = max_deviation
        output.loc[mask, "seed_reproducible_soft"] = bool(max_deviation < 0.03 or common_overlap)
    return output


def collect_seed_check(r_sweep_dir: Path, *, bootstrap_replicates: int = 1000) -> pd.DataFrame:
    rows = []
    for R in SEED_R:
        time_path = r_sweep_dir / f"R{R:03d}/pseudospinodal_fine/analysis/phase5_pseudospinodal_time_dependence.csv"
        if not time_path.is_file():
            continue
        time = pd.read_csv(time_path)
        primary = time[time["T_obs"] == 50]
        if len(primary) != 1 or pd.isna(primary["delta_ps_estimate"].iloc[0]):
            continue
        target = float(primary["delta_ps_estimate"].iloc[0]) + 0.010
        for seed in SEEDS:
            root = (
                r_sweep_dir / f"R{R:03d}/response_matched"
                if seed == 20260815
                else r_sweep_dir / f"R{R:03d}/seed_check/seed_{seed}"
            )
            if not root.is_dir():
                continue
            try:
                metrics = response_metrics(root, target_delta=target, bootstrap_replicates=bootstrap_replicates, bootstrap_seed=seed + R)
            except (FileNotFoundError, ValueError):
                continue
            rows.append(
                {
                    "R": R,
                    "seed": seed,
                    "Gamma_ratio": metrics["Gamma_survive_over_closure"],
                    "Gamma_ratio_se": metrics["Gamma_ratio_se"],
                    "Gamma_ratio_ci_low": metrics["Gamma_ratio_ci_low"],
                    "Gamma_ratio_ci_high": metrics["Gamma_ratio_ci_high"],
                    "Gamma_eff_ratio": metrics["Gamma_eff_3_over_0"],
                    "Method_B_C_difference": metrics["method_B_C_relative_difference"],
                    "escape_fraction": metrics["escape_fraction_cumulative_T"],
                    "baseline_drift": metrics["baseline_drift"],
                    "within_run_bootstrap_SE": metrics["Gamma_ratio_se"],
                    "run_complete": metrics["run_complete"],
                }
            )
    return aggregate_seed_rows(pd.DataFrame(rows))


def _write_table(frame: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def time_rounding_table(time: pd.DataFrame) -> pd.DataFrame:
    """Return one V3 summary row per R while preserving the long-form table."""
    columns = [
        "R", "delta_ps_T20", "delta_ps_T30", "delta_ps_T40", "delta_ps_T50",
        "absolute_rounding", "relative_rounding",
    ]
    if time.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for R, group in time.groupby("R"):
        values = {int(row.T_obs): float(row.delta_ps) for row in group.itertuples()}
        representative = group.iloc[0]
        rows.append(
            {
                "R": int(R),
                "delta_ps_T20": values.get(20, math.nan),
                "delta_ps_T30": values.get(30, math.nan),
                "delta_ps_T40": values.get(40, math.nan),
                "delta_ps_T50": values.get(50, math.nan),
                "absolute_rounding": float(representative["absolute_rounding_50_minus_30"]),
                "relative_rounding": float(representative["relative_time_rounding"]),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values("R").reset_index(drop=True)


def seed_summary_table(seeds: pd.DataFrame) -> pd.DataFrame:
    if seeds.empty:
        return pd.DataFrame(columns=["R", "between_seed_mean", "between_seed_std", "max_relative_seed_deviation", "seed_reproducible_soft"])
    columns = ["R", "between_seed_mean", "between_seed_std", "max_relative_seed_deviation", "seed_reproducible_soft"]
    return seeds[columns].drop_duplicates("R").sort_values("R").reset_index(drop=True)


def analyze_finite_size(args: argparse.Namespace) -> list[Path]:
    pseudo = collect_finite_size_pseudospinodal(args.r_sweep_dir)
    response, escalation = collect_finite_size_response(args.r_sweep_dir, pseudo, bootstrap_replicates=args.bootstrap_replicates)
    output = args.output_dir
    return [
        _write_table(pseudo, output / "finite_size_pseudospinodal.csv"),
        _write_table(response, output / "finite_size_response.csv"),
        _write_table(escalation, output / "finite_size_response_escalation_plan.csv"),
    ]


def analyze_D_precision(args: argparse.Namespace) -> list[Path]:
    table = collect_D_precision(args.r_sweep_dir)
    output = args.output_dir
    csv_path = _write_table(table, output / "high_precision_D_over_kappa.csv")
    complete_table = (
        table[table["run_complete"] & table["production_run_finalized"]]
        if not table.empty and {"run_complete", "production_run_finalized"} <= set(table.columns)
        else pd.DataFrame()
    )
    condition_report = required_condition_report(D_PRECISION_R, set(complete_table.get("R", [])))
    summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "required_R": list(D_PRECISION_R),
        "completeness": condition_report,
        "needs_M65536": (
            table[table["needs_M65536"]]["R"].astype(int).tolist()
            if not table.empty and "needs_M65536" in table
            else []
        ),
        "precision_target_unmet_at_maximum_M": (
            table[
                table["maximum_production_M_reached"]
                & ~table["precision_target_met"]
            ]["R"].astype(int).tolist()
            if not table.empty
            and {"maximum_production_M_reached", "precision_target_met"} <= set(table.columns)
            else []
        ),
        "production_not_sufficient": (
            table[~table["production_M_sufficient"]]["R"].astype(int).tolist()
            if not table.empty and "production_M_sufficient" in table
            else list(D_PRECISION_R)
        ),
        "production_status_by_R": (
            table[
                [
                    "R",
                    "M_total",
                    "D_over_kappa_SE",
                    "production_M_sufficient",
                    "production_run_finalized",
                    "needs_M65536",
                    "production_precision_status",
                ]
            ].to_dict(orient="records")
            if not table.empty
            else []
        ),
        "automatic_M65536_submission": False,
    }
    json_path = output / "high_precision_D_validation_summary.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    return [csv_path, json_path]


def analyze_all(args: argparse.Namespace) -> list[Path]:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    pseudo = collect_finite_size_pseudospinodal(args.r_sweep_dir)
    finite_response, escalation = collect_finite_size_response(args.r_sweep_dir, pseudo, bootstrap_replicates=args.bootstrap_replicates)
    D = collect_D_precision(args.r_sweep_dir)
    time = collect_time_dependence(args.r_sweep_dir, bootstrap_replicates=args.bootstrap_replicates)
    time_rounding = time_rounding_table(time)
    R96 = collect_R96(args.r_sweep_dir, bootstrap_replicates=args.bootstrap_replicates)
    seeds = collect_seed_check(args.r_sweep_dir, bootstrap_replicates=args.bootstrap_replicates)
    seed_summary = seed_summary_table(seeds)
    paths = [
        _write_table(pseudo, output / "finite_size_pseudospinodal.csv"),
        _write_table(finite_response, output / "finite_size_response.csv"),
        _write_table(escalation, output / "finite_size_response_escalation_plan.csv"),
        _write_table(D, output / "high_precision_D_over_kappa.csv"),
        _write_table(time, output / "completed_pseudospinodal_time_dependence.csv"),
        _write_table(time_rounding, output / "observation_time_rounding.csv"),
        _write_table(R96, output / "R96_validation.csv"),
        _write_table(seeds, output / "seed_reproducibility.csv"),
        _write_table(seed_summary, output / "seed_reproducibility_summary.csv"),
    ]

    complete_pseudo = pseudo[pseudo["run_complete"]] if not pseudo.empty and "run_complete" in pseudo else pd.DataFrame()
    complete_response = finite_response[finite_response["run_complete"]] if not finite_response.empty and "run_complete" in finite_response else pd.DataFrame()
    finite_conditions = set(zip(complete_pseudo.get("R", []), complete_pseudo.get("N", []))) & set(zip(complete_response.get("R", []), complete_response.get("N", [])))
    completed_time = (
        time[time["production_M_sufficient"]]
        if not time.empty and "production_M_sufficient" in time
        else pd.DataFrame()
    )
    time_conditions = set(zip(completed_time.get("R", []), completed_time.get("T_obs", [])))
    complete_seeds = seeds[seeds["run_complete"]] if not seeds.empty and "run_complete" in seeds else pd.DataFrame()
    seed_conditions = set(zip(complete_seeds.get("R", []), complete_seeds.get("seed", [])))
    complete_D = (
        D[D["run_complete"] & D["production_run_finalized"]]
        if not D.empty and {"run_complete", "production_run_finalized"} <= set(D.columns)
        else pd.DataFrame()
    )
    complete_R96 = R96[R96["run_complete"]] if not R96.empty and "run_complete" in R96 else pd.DataFrame()
    completeness = {
        "V1_finite_size": required_condition_report(FINITE_SIZE_PAIRS, finite_conditions),
        "V2_D_precision": required_condition_report(D_PRECISION_R, set(complete_D.get("R", []))),
        "V3_time_dependence": required_condition_report(((R, T) for R in TIME_R for T in TIME_OBSERVATIONS), time_conditions),
        "V4_R96": required_condition_report((96,), set(complete_R96.get("R", []))),
        "V5_seed": required_condition_report(((R, seed) for R in SEED_R for seed in SEEDS), seed_conditions),
    }
    run_complete_flags = []
    for frame in (finite_response, D, R96, seeds):
        if not frame.empty and "run_complete" in frame:
            run_complete_flags.extend(bool(value) for value in frame["run_complete"])
    all_conditions_present = all(section["complete"] for section in completeness.values())
    all_present_run_states_complete = all(run_complete_flags)
    complete = all_conditions_present and all_present_run_states_complete

    summary_frames = []
    if not pseudo.empty:
        summary_frames.append(pseudo.assign(section="finite_size")[["section", "R", "N", "delta_ps_T50"]])
    if not finite_response.empty:
        summary_frames.append(finite_response.assign(section="finite_size_response")[["section", "R", "N", "Gamma_survive_over_closure", "Gamma_finite_size_converged"]])
    if not complete_D.empty:
        summary_frames.append(complete_D.assign(section="D_precision")[["section", "R", "N", "D", "D_over_kappa", "D_over_kappa_SE", "D_over_kappa_CI_low", "D_over_kappa_CI_high", "z_from_unity"]])
    if not time_rounding.empty:
        summary_frames.append(time_rounding.assign(section="time_dependence")[["section", "R", "delta_ps_T20", "delta_ps_T30", "delta_ps_T40", "delta_ps_T50", "absolute_rounding", "relative_rounding"]])
    if not R96.empty:
        summary_frames.append(R96.assign(section="R96"))
    if not seeds.empty:
        summary_frames.append(seeds.assign(section="seed")[["section", "R", "seed", "Gamma_ratio", "between_seed_mean", "between_seed_std"]])
    summary_csv = pd.concat(summary_frames, ignore_index=True, sort=False) if summary_frames else pd.DataFrame(columns=["section"])
    paths.append(_write_table(summary_csv, output / "phase5_final_validation_summary.csv"))
    summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "simulation_physics_changed": False,
        "R96_mapping": validate_R96_mapping(),
        "completeness": completeness,
        "missing_conditions": {name: section["missing"] for name, section in completeness.items()},
        "all_present_run_states_complete": all_present_run_states_complete,
        "all_run_states_complete": complete,
        "all_complete": complete,
        "complete": complete,
        "D_precision": {"needs_M65536": D[D.get("needs_M65536", False) == True]["R"].astype(int).tolist() if not D.empty else []},
        "finite_size": {"needs_M8192": escalation[escalation.get("needs_M8192", False) == True][["R", "N"]].to_dict(orient="records") if not escalation.empty else []},
        "wording": {
            "finite_size": "observed range dependence cannot be explained solely by the tested finite-size variation only when the measured R trend exceeds the tested N variation",
            "pseudospinodal": "operational observation-time-dependent 10%-escape crossover; not a critical point",
            "R96": "large-range confirmation point; not proof of a mean-field limit",
        },
    }
    summary_path = output / "phase5_final_validation_summary.json"
    summary_path.write_text(json.dumps(_json_safe(summary), indent=2) + "\n", encoding="utf-8")
    paths.append(summary_path)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    fs_plan = sub.add_parser("plan-finite-size")
    fs_plan.add_argument("--r-sweep-dir", type=Path, default=Path("results/runs/phase5_R_sweep"))
    fs_plan.add_argument("--output", type=Path, default=Path("results/runs/phase5_R_sweep/finite_size/finite_size_fine_plan.csv"))

    fs_value = sub.add_parser("print-finite-size-plan")
    fs_value.add_argument("--plan", type=Path, required=True)
    fs_value.add_argument("--R", type=int, required=True)
    fs_value.add_argument("--N", type=int, required=True)
    fs_value.add_argument("--column", default="fine_deltas")

    fs_matched = sub.add_parser("print-finite-size-matched")
    fs_matched.add_argument("--table", type=Path, required=True)
    fs_matched.add_argument("--R", type=int, required=True)
    fs_matched.add_argument("--N", type=int, required=True)
    fs_matched.add_argument("--offset", type=float, default=0.010)

    fs_analyze = sub.add_parser("analyze-finite-size")
    fs_analyze.add_argument("--r-sweep-dir", type=Path, default=Path("results/runs/phase5_R_sweep"))
    fs_analyze.add_argument("--output-dir", type=Path, default=Path("results/runs/phase5_final_validation"))
    fs_analyze.add_argument("--bootstrap-replicates", type=int, default=1000)

    D_analyze = sub.add_parser("analyze-D")
    D_analyze.add_argument("--r-sweep-dir", type=Path, default=Path("results/runs/phase5_R_sweep"))
    D_analyze.add_argument("--output-dir", type=Path, default=Path("results/runs/phase5_final_validation"))

    time_plan = sub.add_parser("plan-time-extension")
    time_plan.add_argument("--r-sweep-dir", type=Path, default=Path("results/runs/phase5_R_sweep"))
    time_plan.add_argument("--output", type=Path, default=Path("results/runs/phase5_R_sweep/observation_time_extension_plan.csv"))

    time_value = sub.add_parser("print-time-extension")
    time_value.add_argument("--plan", type=Path, required=True)
    time_value.add_argument("--R", type=int, required=True)

    analyze = sub.add_parser("analyze")
    analyze.add_argument("--r-sweep-dir", type=Path, default=Path("results/runs/phase5_R_sweep"))
    analyze.add_argument("--final-validation-dir", type=Path, default=Path("results/runs/phase5_final_validation"), help="compatibility alias for the validation input/output root")
    analyze.add_argument("--output-dir", type=Path, default=Path("results/runs/phase5_final_validation"))
    analyze.add_argument("--bootstrap-replicates", type=int, default=1000)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "plan-finite-size":
        table = build_finite_size_fine_plan(args.r_sweep_dir)
        print(_write_table(table, args.output))
    elif args.command == "print-finite-size-plan":
        print(_plan_value(args.plan, args.R, args.N, args.column))
    elif args.command == "print-finite-size-matched":
        table = pd.read_csv(args.table)
        row = table[(table["R"] == args.R) & (table["N"] == args.N)]
        if len(row) != 1 or pd.isna(row["delta_ps_T50"].iloc[0]):
            raise ValueError(f"no finite-size delta_ps for R={args.R}, N={args.N}")
        print(f"{float(row['delta_ps_T50'].iloc[0]) + args.offset:.12g}")
    elif args.command == "analyze-finite-size":
        for path in analyze_finite_size(args):
            print(path)
    elif args.command == "analyze-D":
        for path in analyze_D_precision(args):
            print(path)
    elif args.command == "plan-time-extension":
        print(_write_table(plan_time_extension(args.r_sweep_dir), args.output))
    elif args.command == "print-time-extension":
        table = pd.read_csv(args.plan, keep_default_na=False)
        values = sorted({float(value) for text in table[(table["R"] == args.R) & (table["status"].isin(("extension_required", "M_upgrade_required")))]["suggested_new_deltas"] for value in str(text).split(",") if value})
        if not values:
            raise ValueError(f"no time-extension delta required for R={args.R}")
        print(_format_values(values))
    elif args.command == "analyze":
        for path in analyze_all(args):
            print(path)


if __name__ == "__main__":
    main()
