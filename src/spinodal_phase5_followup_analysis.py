#!/usr/bin/env python3
"""Phase5 follow-up diagnostics from compact block checkpoints.

The ``time`` command is deliberately checkpoint-only and accepts legacy
schema-v1 files.  Survival-conditioned and first-passage commands require
schema-v2 sufficient statistics and never try to reconstruct unavailable
trial histories.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from spinodal_phase5_analysis import fit_microscopic_relaxation
from spinodal_phase5_core import SCRIPT_VERSION, Phase5BlockResult, load_block_checkpoint


FIXED_WINDOWS = ((0, 3), (0, 5), (1, 3), (1, 5), (2, 5), (2, 7), (3, 7))
SURVIVAL_RERUN_ERROR = (
    "Survival-conditioned observables are unavailable in legacy Phase5 "
    "checkpoints; rerun with survival tracking enabled."
)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _bootstrap_summary(samples: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < 2:
        return math.nan, math.nan, math.nan
    return (
        float(np.std(finite, ddof=1)),
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def effective_relaxation(amplitude: Sequence[float]) -> dict[str, np.ndarray]:
    """Return signed one-step lambda and its logarithmic envelope rate."""
    values = np.asarray(amplitude, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("amplitude must be a one-dimensional series of length >=2")
    current = values[:-1]
    following = values[1:]
    with np.errstate(divide="ignore", invalid="ignore"):
        lambdas = following / current
        gamma = -np.log(np.abs(lambdas))
    invalid = (~np.isfinite(current)) | (~np.isfinite(following)) | (current == 0.0)
    lambdas[invalid] = np.nan
    gamma[invalid | (lambdas == 0.0)] = np.nan
    sign_flip = (
        np.signbit(current) != np.signbit(following)
    ) & (current != 0.0) & (following != 0.0)
    return {
        "lambda_eff": lambdas,
        "Gamma_eff": gamma,
        "sign_flip": sign_flip,
    }


def load_checkpoint_groups(input_dir: Path) -> dict[str, list[Phase5BlockResult]]:
    blocks_dir = Path(input_dir) / "blocks"
    paths = sorted(blocks_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no Phase5 checkpoints found under {blocks_dir}")
    grouped: dict[str, list[Phase5BlockResult]] = {}
    for path in paths:
        block = load_block_checkpoint(path)
        grouped.setdefault(block.task_id, []).append(block)
    for blocks in grouped.values():
        blocks.sort(key=lambda item: item.block_id)
    return grouped


def _weighted_block_mean(
    blocks: Sequence[Phase5BlockResult], attribute: str
) -> np.ndarray:
    values = np.stack(
        [np.asarray(getattr(block, attribute), dtype=float) for block in blocks]
    )
    weights = np.asarray([block.block_n for block in blocks], dtype=float)
    return np.average(values, axis=0, weights=weights)


def block_mean_and_se(
    blocks: Sequence[Phase5BlockResult], attribute: str = "A_q"
) -> tuple[np.ndarray, np.ndarray]:
    """Return weighted mean and block-to-block SE of compact block means."""
    values = np.stack(
        [np.asarray(getattr(block, attribute), dtype=float) for block in blocks]
    )
    weights = np.asarray([block.block_n for block in blocks], dtype=float)
    mean = np.average(values, axis=0, weights=weights)
    if len(blocks) < 2:
        return mean, np.full_like(mean, math.nan)
    normalized = weights / np.sum(weights)
    denominator = 1.0 - float(np.sum(normalized**2))
    variance = np.sum(normalized[:, None] * (values - mean) ** 2, axis=0)
    variance = variance / denominator if denominator > 0.0 else variance
    effective_n = 1.0 / float(np.sum(normalized**2))
    return mean, np.sqrt(np.maximum(variance, 0.0) / effective_n)


def bootstrap_gamma_eff(
    blocks: Sequence[Phase5BlockResult],
    *,
    replicates: int,
    seed: int,
    attribute: str = "A_q",
) -> np.ndarray:
    """Block-bootstrap the full effective-rate time series."""
    if replicates < 1:
        return np.empty((0, len(blocks[0].A_q) - 1), dtype=float)
    values = np.stack(
        [np.asarray(getattr(block, attribute), dtype=float) for block in blocks]
    )
    weights = np.asarray([block.block_n for block in blocks], dtype=float)
    rng = np.random.Generator(np.random.Philox(seed))
    output = np.full((replicates, values.shape[1] - 1), math.nan, dtype=float)
    for replicate in range(replicates):
        indices = rng.integers(0, len(blocks), len(blocks))
        amplitude = np.average(values[indices], axis=0, weights=weights[indices])
        output[replicate] = effective_relaxation(amplitude)["Gamma_eff"]
    return output


def build_gamma_eff_table(
    groups: dict[str, list[Phase5BlockResult]],
    *,
    min_snr: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group_index, (task_id, blocks) in enumerate(sorted(groups.items())):
        mean, se = block_mean_and_se(blocks)
        rate = effective_relaxation(mean)
        boot = bootstrap_gamma_eff(
            blocks,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + group_index,
        )
        escape = _weighted_block_mean(blocks, "escape_fraction")
        baseline = _weighted_block_mean(blocks, "baseline_m")
        snr = np.divide(
            np.abs(mean),
            se,
            out=np.full_like(mean, math.inf),
            where=np.isfinite(se) & (se > 0.0),
        )
        for t in range(len(mean) - 1):
            gamma_se, gamma_low, gamma_high = _bootstrap_summary(boot[:, t])
            sign_flip = bool(rate["sign_flip"][t])
            snr_reliable = bool(
                snr[t] >= min_snr
                and snr[t + 1] >= min_snr
                and math.isfinite(float(rate["Gamma_eff"][t]))
                and not sign_flip
            )
            rows.append(
                {
                    "task_id": task_id,
                    "delta": blocks[0].delta,
                    "mode_index": blocks[0].mode_index,
                    "epsilon_fraction": blocks[0].epsilon_fraction,
                    "t": t,
                    "A_q": mean[t],
                    "A_q_se": se[t],
                    "A_q_snr": snr[t],
                    "lambda_eff": rate["lambda_eff"][t],
                    "Gamma_eff": rate["Gamma_eff"][t],
                    "Gamma_eff_envelope": rate["Gamma_eff"][t],
                    "Gamma_eff_se": gamma_se,
                    "Gamma_eff_ci_low": gamma_low,
                    "Gamma_eff_ci_high": gamma_high,
                    "sign_flip": sign_flip,
                    "snr_reliable": snr_reliable,
                    "escape_fraction": escape[t],
                    "baseline_m": baseline[t],
                    "gamma_eff_min_snr": min_snr,
                }
            )
    return pd.DataFrame(rows)


def _window_diagnostics(
    amplitude: np.ndarray,
    gamma_rows: pd.DataFrame,
    start: int,
    end: int,
    *,
    plateau_max_cv: float,
) -> dict[str, Any]:
    fit = fit_microscopic_relaxation(amplitude, start, end)
    local = gamma_rows[(gamma_rows["t"] >= start) & (gamma_rows["t"] < end)]
    reliable = local[local["snr_reliable"]]
    gamma_values = reliable["Gamma_eff"].to_numpy(dtype=float)
    finite = gamma_values[np.isfinite(gamma_values)]
    mean_gamma = float(np.mean(finite)) if finite.size else math.nan
    variation = (
        float(np.std(finite, ddof=1) / abs(mean_gamma))
        if finite.size > 1 and mean_gamma != 0.0
        else math.nan
    )
    ci_overlap = False
    if len(reliable) == len(local) and len(local) > 0:
        lows = reliable["Gamma_eff_ci_low"].to_numpy(dtype=float)
        highs = reliable["Gamma_eff_ci_high"].to_numpy(dtype=float)
        ci_overlap = bool(
            np.all(np.isfinite(lows))
            and np.all(np.isfinite(highs))
            and float(np.max(lows)) <= float(np.min(highs))
        )
    candidate = bool(
        len(local) == end - start
        and len(reliable) == len(local)
        and finite.size == len(local)
        and not bool(local["sign_flip"].any())
        and math.isfinite(variation)
        and variation <= plateau_max_cv
        and ci_overlap
    )
    return {
        "Gamma_B": fit["Gamma_micro"],
        "Gamma_C": fit["Gamma_logfit"],
        "method_B_C_relative_difference": fit[
            "method_B_C_relative_difference"
        ],
        "R2_B": fit["fit_r2"],
        "R2_C": fit["logfit_r2"],
        "mean_Gamma_eff": mean_gamma,
        "Gamma_eff_variation": variation,
        "gamma_eff_ci_overlap": ci_overlap,
        "plateau_candidate": candidate,
    }


def build_extended_fit_table(
    groups: dict[str, list[Phase5BlockResult]],
    gamma_eff: pd.DataFrame,
    *,
    plateau_max_cv: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for task_id, blocks in sorted(groups.items()):
        amplitude = _weighted_block_mean(blocks, "A_q")
        local_gamma = gamma_eff[gamma_eff["task_id"] == task_id]
        candidates: list[int] = []
        for start, end in FIXED_WINDOWS:
            if end >= len(amplitude):
                continue
            values = _window_diagnostics(
                amplitude,
                local_gamma,
                start,
                end,
                plateau_max_cv=plateau_max_cv,
            )
            rows.append(
                {
                    "task_id": task_id,
                    "delta": blocks[0].delta,
                    "mode_index": blocks[0].mode_index,
                    "epsilon_fraction": blocks[0].epsilon_fraction,
                    "fit_start": start,
                    "fit_end": end,
                    **values,
                    "primary_window": start == 0 and end == 3,
                    "plateau_found": False,
                    "plateau_start": math.nan,
                    "plateau_end": math.nan,
                    "plateau_mean_gamma": math.nan,
                }
            )
            if values["plateau_candidate"]:
                candidates.append(len(rows) - 1)
        if candidates:
            selected = max(
                candidates,
                key=lambda index: (
                    rows[index]["fit_end"] - rows[index]["fit_start"],
                    -rows[index]["fit_start"],
                ),
            )
            rows[selected]["plateau_found"] = True
            rows[selected]["plateau_start"] = rows[selected]["fit_start"]
            rows[selected]["plateau_end"] = rows[selected]["fit_end"]
            rows[selected]["plateau_mean_gamma"] = rows[selected]["mean_Gamma_eff"]
    return pd.DataFrame(rows)


def _update_summary(
    path: Path,
    section: str,
    payload: dict[str, Any],
    warnings: Sequence[str] = (),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        summary = json.loads(path.read_text(encoding="utf-8"))
    else:
        summary = {
            "script_version": SCRIPT_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "warnings": [],
        }
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary[section] = payload
    summary["warnings"] = sorted(set(summary.get("warnings", [])) | set(warnings))
    path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def analyze_time_command(args: argparse.Namespace) -> None:
    groups = load_checkpoint_groups(args.input_dir)
    gamma = build_gamma_eff_table(
        groups,
        min_snr=args.gamma_eff_min_snr,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    windows = build_extended_fit_table(
        groups, gamma, plateau_max_cv=args.plateau_max_cv
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gamma_path = args.output_dir / "phase5_gamma_eff.csv"
    window_path = args.output_dir / "phase5_fit_window_extended.csv"
    gamma.to_csv(gamma_path, index=False)
    windows.to_csv(window_path, index=False)
    primary = windows[windows["primary_window"]]
    plateau = windows[windows["plateau_found"]]
    payload = {
        "source": str(args.input_dir),
        "source_checkpoint_schema_versions": sorted(
            {int(block.checkpoint_schema_version) for blocks in groups.values() for block in blocks}
        ),
        "new_simulation_performed": False,
        "gamma_eff_min_snr": args.gamma_eff_min_snr,
        "max_method_B_C_difference": _finite_or_none(
            float(primary["method_B_C_relative_difference"].max())
        ),
        "gamma_eff_plateau_found_by_delta_mode": [
            {
                "delta": float(row.delta),
                "mode_index": int(row.mode_index),
                "epsilon_fraction": float(row.epsilon_fraction),
                "start": int(row.plateau_start),
                "end": int(row.plateau_end),
                "gamma": float(row.plateau_mean_gamma),
            }
            for row in plateau.itertuples()
        ],
        "max_gamma_eff_time_variation": _finite_or_none(
            float(windows["Gamma_eff_variation"].max())
        ),
        "snr_failure_count": int((~gamma["snr_reliable"]).sum()),
        "sign_flip_count": int(gamma["sign_flip"].sum()),
        "interpretation": (
            "diagnoses the time range of a single-exponential approximation; "
            "the primary Gamma remains the predeclared 0:3 Method-B result"
        ),
    }
    _update_summary(args.summary_path, "time_relaxation", payload)
    print(gamma_path)
    print(window_path)
    print(args.summary_path)


def require_survival(blocks: Sequence[Phase5BlockResult]) -> None:
    if not blocks or any(
        not block.survival_tracking_enabled
        or block.checkpoint_schema_version < 2
        or block.survival_count.size == 0
        for block in blocks
    ):
        raise ValueError(SURVIVAL_RERUN_ERROR)


def aggregate_survival_blocks(
    blocks: Sequence[Phase5BlockResult],
) -> dict[str, np.ndarray | float | int]:
    """Aggregate survivor observables by numerator/count, never block means."""
    require_survival(blocks)
    M_total = int(sum(block.block_n for block in blocks))
    current_count = np.sum(
        np.stack([block.survival_count for block in blocks]), axis=0, dtype=np.int64
    )
    current_sum = np.sum(
        np.stack([block.survivor_amplitude_sum_current for block in blocks]), axis=0
    )
    final_count = int(sum(block.survive_to_T_count for block in blocks))
    final_sum = np.sum(
        np.stack([block.survive_to_T_amplitude_sum for block in blocks]), axis=0
    )
    current_amplitude = np.divide(
        current_sum,
        current_count,
        out=np.full_like(current_sum, math.nan, dtype=float),
        where=current_count > 0,
    )
    final_amplitude = (
        final_sum / final_count
        if final_count > 0
        else np.full_like(final_sum, math.nan, dtype=float)
    )
    return {
        "M_total": M_total,
        "A_unconditional": _weighted_block_mean(blocks, "A_q"),
        "baseline_m": _weighted_block_mean(blocks, "baseline_m"),
        "escape_fraction_instantaneous": _weighted_block_mean(
            blocks, "escape_fraction"
        ),
        "survival_count_current": current_count,
        "survival_fraction_current": current_count / M_total,
        "escape_fraction_cumulative": 1.0 - current_count / M_total,
        "A_surviving_current": current_amplitude,
        "survive_to_T_count": final_count,
        "survive_to_T_fraction": final_count / M_total,
        "A_survive_to_T": final_amplitude,
    }


def _safe_fit(values: np.ndarray, start: int = 0, end: int = 3) -> dict[str, Any]:
    if len(values) <= end or not np.all(np.isfinite(values[start : end + 1])):
        return {
            "Gamma_micro": math.nan,
            "Gamma_logfit": math.nan,
            "method_B_C_relative_difference": math.nan,
            "fit_r2": math.nan,
            "logfit_r2": math.nan,
        }
    return fit_microscopic_relaxation(values, start, end)


def analyze_survival_command(args: argparse.Namespace) -> None:
    groups = load_checkpoint_groups(args.input_dir)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for task_id, blocks in sorted(groups.items()):
        aggregate = aggregate_survival_blocks(blocks)
        uncond = np.asarray(aggregate["A_unconditional"], dtype=float)
        current = np.asarray(aggregate["A_surviving_current"], dtype=float)
        final = np.asarray(aggregate["A_survive_to_T"], dtype=float)
        rate_u = effective_relaxation(uncond)
        rate_c = effective_relaxation(current)
        rate_f = effective_relaxation(final)
        T = len(uncond) - 1
        for t in range(T + 1):
            rows.append(
                {
                    "task_id": task_id,
                    "delta": blocks[0].delta,
                    "mode_index": blocks[0].mode_index,
                    "epsilon_fraction": blocks[0].epsilon_fraction,
                    "t": t,
                    "A_unconditional": uncond[t],
                    "survival_count_current": int(aggregate["survival_count_current"][t]),
                    "survival_fraction_current": aggregate["survival_fraction_current"][t],
                    "A_surviving_current": current[t],
                    "survive_to_T_count": int(aggregate["survive_to_T_count"]),
                    "survive_to_T_fraction": aggregate["survive_to_T_fraction"],
                    "A_survive_to_T": final[t],
                    "Gamma_eff_unconditional": rate_u["Gamma_eff"][t] if t < T else math.nan,
                    "Gamma_eff_surviving_current": rate_c["Gamma_eff"][t] if t < T else math.nan,
                    "Gamma_eff_survive_to_T": rate_f["Gamma_eff"][t] if t < T else math.nan,
                    "escape_fraction_instantaneous": aggregate["escape_fraction_instantaneous"][t],
                    "escape_fraction_cumulative": aggregate["escape_fraction_cumulative"][t],
                }
            )
        fit_u = _safe_fit(uncond)
        fit_f = _safe_fit(final)
        gamma_u = float(fit_u["Gamma_micro"])
        gamma_f = float(fit_f["Gamma_micro"])
        relative = (
            abs(gamma_f - gamma_u) / abs(gamma_u)
            if math.isfinite(gamma_u) and gamma_u != 0.0 and math.isfinite(gamma_f)
            else math.nan
        )
        survivor_reliable = int(aggregate["survive_to_T_count"]) >= args.min_survivors
        if not survivor_reliable:
            warnings.append(f"insufficient final survivors for {task_id}")
        summary_rows.append(
            {
                "delta": blocks[0].delta,
                "mode_index": blocks[0].mode_index,
                "Gamma_unconditional": _finite_or_none(gamma_u),
                "Gamma_survive_to_T": _finite_or_none(gamma_f),
                "relative_difference": _finite_or_none(relative),
                "survive_to_T_count": int(aggregate["survive_to_T_count"]),
                "reliable": survivor_reliable,
                "interpretation": (
                    "fixed final-survivor cohorts use future information and diagnose "
                    "basin-internal trajectories; they are not the unconditional response"
                ),
            }
        )
    table = pd.DataFrame(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "phase5_survival_conditioned.csv"
    table.to_csv(path, index=False)
    max_escape = float(table["escape_fraction_cumulative"].max())
    _update_summary(
        args.summary_path,
        "survival",
        {
            "source": str(args.input_dir),
            "max_cumulative_escape": max_escape,
            "comparisons": summary_rows,
            "interpretation": (
                "compare unconditional and survive-to-T Gamma to separate escape "
                "contamination from slower relaxation inside the metastable basin"
            ),
        },
        warnings,
    )
    print(path)
    print(args.summary_path)


def _bootstrap_gamma(
    blocks: Sequence[Phase5BlockResult], replicates: int, seed: int
) -> np.ndarray:
    values = np.stack([block.A_q for block in blocks])
    weights = np.asarray([block.block_n for block in blocks], dtype=float)
    rng = np.random.Generator(np.random.Philox(seed))
    samples = []
    for _ in range(replicates):
        indices = rng.integers(0, len(blocks), len(blocks))
        amplitude = np.average(values[indices], axis=0, weights=weights[indices])
        gamma = float(_safe_fit(amplitude)["Gamma_micro"])
        if math.isfinite(gamma):
            samples.append(gamma)
    return np.asarray(samples, dtype=float)


def analyze_preparation_command(args: argparse.Namespace) -> None:
    parsed: list[tuple[int, Path]] = []
    for value in args.input:
        try:
            burn_text, path_text = value.split("=", 1)
            parsed.append((int(burn_text), Path(path_text)))
        except ValueError as exc:
            raise ValueError("--input must have BURN=PATH form") from exc
    rows: list[dict[str, Any]] = []
    for burn, input_dir in sorted(parsed):
        groups = load_checkpoint_groups(input_dir)
        for index, (task_id, blocks) in enumerate(sorted(groups.items())):
            aggregate = aggregate_survival_blocks(blocks)
            amplitude = np.asarray(aggregate["A_unconditional"], dtype=float)
            fit = _safe_fit(amplitude)
            preparation = _weighted_block_mean(blocks, "preparation_magnetization")
            baseline = np.asarray(aggregate["baseline_m"], dtype=float)
            samples = _bootstrap_gamma(
                blocks, args.bootstrap_replicates, args.bootstrap_seed + 1000 * burn + index
            )
            gamma_se, gamma_low, gamma_high = _bootstrap_summary(samples)
            gamma_rate = build_gamma_eff_table(
                {task_id: blocks},
                min_snr=args.gamma_eff_min_snr,
                bootstrap_replicates=max(50, min(args.bootstrap_replicates, 200)),
                bootstrap_seed=args.bootstrap_seed + index,
            )
            windows = build_extended_fit_table(
                {task_id: blocks}, gamma_rate, plateau_max_cv=args.plateau_max_cv
            )
            plateau = windows[windows["plateau_found"]]
            rows.append(
                {
                    "task_id": task_id,
                    "delta": blocks[0].delta,
                    "mode_index": blocks[0].mode_index,
                    "burn_steps_per_stage": burn,
                    "M_total": int(aggregate["M_total"]),
                    "preparation_initial_m": preparation[0],
                    "preparation_final_m": preparation[-1],
                    "preparation_drift": preparation[-1] - preparation[0],
                    "max_baseline_drift": float(np.max(np.abs(baseline - baseline[0]))),
                    "Gamma_B": fit["Gamma_micro"],
                    "Gamma_B_se": gamma_se,
                    "Gamma_B_ci_low": gamma_low,
                    "Gamma_B_ci_high": gamma_high,
                    "Gamma_C": fit["Gamma_logfit"],
                    "method_B_C_relative_difference": fit["method_B_C_relative_difference"],
                    "plateau_found": not plateau.empty,
                    "plateau_gamma": float(plateau["plateau_mean_gamma"].iloc[0]) if not plateau.empty else math.nan,
                    "escape_fraction_instantaneous_max": float(np.max(aggregate["escape_fraction_instantaneous"])),
                    "escape_fraction_cumulative_T": float(aggregate["escape_fraction_cumulative"][-1]),
                    "Gamma_relative_change_vs_burn8": math.nan,
                    "bootstrap_consistent_with_burn8": False,
                    "preparation_converged_soft_flag": False,
                    "reliable": math.isfinite(float(fit["Gamma_micro"])),
                }
            )
    table = pd.DataFrame(rows)
    reference = {
        (float(row.delta), int(row.mode_index)): row
        for row in table[table["burn_steps_per_stage"] == 8].itertuples()
    }
    for index, row in table.iterrows():
        key = (float(row["delta"]), int(row["mode_index"]))
        if key not in reference:
            continue
        ref = reference[key]
        relative = abs(float(row["Gamma_B"]) - float(ref.Gamma_B)) / abs(float(ref.Gamma_B))
        combined = math.sqrt(float(row["Gamma_B_se"]) ** 2 + float(ref.Gamma_B_se) ** 2)
        consistent = abs(float(row["Gamma_B"]) - float(ref.Gamma_B)) <= 1.96 * combined
        soft = bool(relative < 0.02 and consistent)
        table.at[index, "Gamma_relative_change_vs_burn8"] = relative
        table.at[index, "bootstrap_consistent_with_burn8"] = consistent
        table.at[index, "preparation_converged_soft_flag"] = soft
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "phase5_preparation_scan.csv"
    table.to_csv(path, index=False)

    def changes(first: int, second: int) -> list[dict[str, Any]]:
        output = []
        for key in sorted({(float(r.delta), int(r.mode_index)) for r in table.itertuples()}):
            a = table[(table["burn_steps_per_stage"] == first) & np.isclose(table["delta"], key[0]) & (table["mode_index"] == key[1])]
            b = table[(table["burn_steps_per_stage"] == second) & np.isclose(table["delta"], key[0]) & (table["mode_index"] == key[1])]
            if a.empty or b.empty:
                continue
            output.append({"delta": key[0], "mode_index": key[1], "relative_change": abs(float(b["Gamma_B"].iloc[0]) - float(a["Gamma_B"].iloc[0])) / abs(float(a["Gamma_B"].iloc[0]))})
        return output

    _update_summary(
        args.summary_path,
        "preparation",
        {
            "burn_values": sorted(table["burn_steps_per_stage"].unique().astype(int).tolist()),
            "Gamma_change_8_to_16": changes(8, 16),
            "Gamma_change_16_to_32": changes(16, 32),
            "baseline_drift_by_burn": table.groupby("burn_steps_per_stage")["max_baseline_drift"].max().to_dict(),
            "escape_by_burn": table.groupby("burn_steps_per_stage")["escape_fraction_cumulative_T"].max().to_dict(),
            "preparation_converged_soft_flag": bool(table[table["burn_steps_per_stage"] > 8]["preparation_converged_soft_flag"].all()),
            "rng_note": (
                "legacy_v1 single Philox stream retained; thresholds are paired by stable IDs, "
                "but different burn lengths consume different later RNG positions"
            ),
        },
    )
    print(path)
    print(args.summary_path)


def interpolate_escape_crossing(
    deltas: Sequence[float],
    probabilities: Sequence[float],
    criterion: float = 0.10,
) -> dict[str, Any]:
    delta = np.asarray(deltas, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    order = np.argsort(delta)
    delta = delta[order]
    probability = probability[order]
    monotonic = bool(np.all(np.diff(probability) <= 1e-12))
    candidates = []
    for index in range(len(delta) - 1):
        p0, p1 = probability[index], probability[index + 1]
        if p0 >= criterion >= p1:
            candidates.append(index)
    if len(candidates) != 1:
        return {"estimate": None, "lower": None, "upper": None, "monotonicity_ok": monotonic}
    index = candidates[0]
    p0, p1 = probability[index], probability[index + 1]
    if p0 == p1:
        estimate = 0.5 * (delta[index] + delta[index + 1])
    else:
        estimate = delta[index] + (criterion - p0) * (delta[index + 1] - delta[index]) / (p1 - p0)
    return {"estimate": float(estimate), "lower": float(delta[index]), "upper": float(delta[index + 1]), "monotonicity_ok": monotonic}


def _bootstrap_escape_curves(
    blocks: Sequence[Phase5BlockResult], replicates: int, seed: int
) -> np.ndarray:
    require_survival(blocks)
    counts = np.stack([block.survival_count for block in blocks]).astype(float)
    sizes = np.asarray([block.block_n for block in blocks], dtype=float)
    rng = np.random.Generator(np.random.Philox(seed))
    output = np.empty((replicates, counts.shape[1]), dtype=float)
    for replicate in range(replicates):
        indices = rng.integers(0, len(blocks), len(blocks))
        output[replicate] = 1.0 - np.sum(counts[indices], axis=0) / np.sum(sizes[indices])
    return output


def analyze_pseudospinodal_command(args: argparse.Namespace) -> None:
    groups = load_checkpoint_groups(args.input_dir)
    by_delta: dict[float, list[Phase5BlockResult]] = {}
    for blocks in groups.values():
        require_survival(blocks)
        if any(
            block.epsilon_fraction != 0.0 or abs(block.initial_amplitude) > 1e-15
            for block in blocks
        ):
            raise ValueError("primary pseudospinodal analysis requires unperturbed epsilon=0 checkpoints")
        by_delta.setdefault(float(blocks[0].delta), []).extend(blocks)
    deltas = sorted(by_delta)
    curves: dict[float, np.ndarray] = {}
    boot: dict[float, np.ndarray] = {}
    for index, delta in enumerate(deltas):
        blocks = sorted(by_delta[delta], key=lambda item: item.block_id)
        require_survival(blocks)
        total = sum(block.block_n for block in blocks)
        survival = np.sum(np.stack([block.survival_count for block in blocks]), axis=0) / total
        curves[delta] = 1.0 - survival
        boot[delta] = _bootstrap_escape_curves(
            blocks, args.bootstrap_replicates, args.bootstrap_seed + index
        )
    T = min(args.primary_T, min(len(value) - 1 for value in curves.values()))
    primary_prob = [curves[delta][T] for delta in deltas]
    primary_crossing = interpolate_escape_crossing(deltas, primary_prob, args.criterion_probability)
    fine_rows = []
    for delta in deltas:
        blocks = by_delta[delta]
        weights = np.asarray([block.block_n for block in blocks], dtype=float)
        preparation = np.average(
            np.stack([block.preparation_magnetization for block in blocks]), axis=0, weights=weights
        )
        baseline = _weighted_block_mean(blocks, "baseline_m")
        instant = _weighted_block_mean(blocks, "escape_fraction")
        samples = boot[delta][:, T]
        se, low, high = _bootstrap_summary(samples)
        fine_rows.append(
            {
                "delta": delta,
                "Delta": blocks[0].Delta,
                "M_total": int(np.sum(weights)),
                "T": T,
                "preparation_final_m": preparation[-1],
                "max_baseline_drift": float(np.max(np.abs(baseline - baseline[0]))),
                "escape_fraction_instantaneous_T": instant[T],
                "escape_fraction_cumulative_T": curves[delta][T],
                "survival_fraction_T": 1.0 - curves[delta][T],
                "escape_cumulative_se": se,
                "escape_cumulative_ci_low": low,
                "escape_cumulative_ci_high": high,
                "criterion_probability": args.criterion_probability,
                "criterion_T": T,
                "brackets_criterion": delta in {primary_crossing["lower"], primary_crossing["upper"]},
            }
        )
    time_rows = []
    for T_obs in args.observation_times:
        if T_obs > T:
            continue
        probabilities = [curves[delta][T_obs] for delta in deltas]
        crossing = interpolate_escape_crossing(deltas, probabilities, args.criterion_probability)
        estimates = []
        failed = 0
        for replicate in range(args.bootstrap_replicates):
            replicate_probabilities = [boot[delta][replicate, T_obs] for delta in deltas]
            result = interpolate_escape_crossing(deltas, replicate_probabilities, args.criterion_probability)
            if result["estimate"] is None:
                failed += 1
            else:
                estimates.append(float(result["estimate"]))
        estimate_samples = np.asarray(estimates, dtype=float)
        se, low, high = _bootstrap_summary(estimate_samples)
        time_rows.append(
            {
                "T_obs": T_obs,
                "criterion_probability": args.criterion_probability,
                "delta_lower_bracket": crossing["lower"],
                "delta_upper_bracket": crossing["upper"],
                "delta_ps_estimate": crossing["estimate"],
                "delta_ps_se": se,
                "delta_ps_ci_low": low,
                "delta_ps_ci_high": high,
                "n_bootstrap_valid": len(estimates),
                "n_bootstrap_failed": failed,
                "monotonicity_ok": crossing["monotonicity_ok"],
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fine_path = args.output_dir / "phase5_pseudospinodal_fine_scan.csv"
    time_path = args.output_dir / "phase5_pseudospinodal_time_dependence.csv"
    pd.DataFrame(fine_rows).to_csv(fine_path, index=False)
    time_table = pd.DataFrame(time_rows)
    time_table.to_csv(time_path, index=False)
    primary_row = time_table[time_table["T_obs"] == T]
    warnings = []
    if primary_crossing["estimate"] is None:
        warnings.append("10%-escape crossover is not bracketed at the primary observation time")
    if not primary_crossing["monotonicity_ok"]:
        warnings.append("escape probability is not globally monotone in delta at the primary observation time")
    _update_summary(
        args.summary_path,
        "pseudospinodal",
        {
            "escape_criterion": args.criterion_probability,
            "primary_T": T,
            "delta_bracket": [primary_crossing["lower"], primary_crossing["upper"]],
            "delta_ps_estimate": primary_crossing["estimate"],
            "delta_ps_CI": (
                [
                    _finite_or_none(float(primary_row["delta_ps_ci_low"].iloc[0])),
                    _finite_or_none(float(primary_row["delta_ps_ci_high"].iloc[0])),
                ]
                if not primary_row.empty
                else None
            ),
            "time_dependence": time_rows,
            "wording": "operational microscopic pseudospinodal-like 10%-escape crossover; not a true spinodal",
        },
        warnings,
    )
    print(fine_path)
    print(time_path)
    print(args.summary_path)


def _common_output_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260816)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Phase5 time/preparation/survival/pseudospinodal diagnostics"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    time_parser = subparsers.add_parser("time")
    time_parser.add_argument("--input-dir", type=Path, required=True)
    _common_output_arguments(time_parser)
    time_parser.add_argument("--gamma-eff-min-snr", type=float, default=5.0)
    time_parser.add_argument("--plateau-max-cv", type=float, default=0.15)
    time_parser.set_defaults(handler=analyze_time_command)

    preparation = subparsers.add_parser("preparation")
    preparation.add_argument("--input", action="append", required=True, help="BURN=PATH")
    _common_output_arguments(preparation)
    preparation.add_argument("--gamma-eff-min-snr", type=float, default=5.0)
    preparation.add_argument("--plateau-max-cv", type=float, default=0.15)
    preparation.set_defaults(handler=analyze_preparation_command)

    survival = subparsers.add_parser("survival")
    survival.add_argument("--input-dir", type=Path, required=True)
    _common_output_arguments(survival)
    survival.add_argument("--min-survivors", type=int, default=100)
    survival.set_defaults(handler=analyze_survival_command)

    pseudospinodal = subparsers.add_parser("pseudospinodal")
    pseudospinodal.add_argument("--input-dir", type=Path, required=True)
    _common_output_arguments(pseudospinodal)
    pseudospinodal.add_argument("--criterion-probability", type=float, default=0.10)
    pseudospinodal.add_argument("--primary-T", type=int, default=50)
    pseudospinodal.add_argument(
        "--observation-times",
        type=lambda text: tuple(int(value) for value in text.split(",")),
        default=(10, 20, 30, 40, 50),
    )
    pseudospinodal.set_defaults(handler=analyze_pseudospinodal_command)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.bootstrap_replicates < 2:
        parser.error("--bootstrap-replicates must be at least 2")
    if getattr(args, "summary_path", None) is None:
        args.summary_path = args.output_dir / "phase5_followup_validation_summary.json"
    try:
        args.handler(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
