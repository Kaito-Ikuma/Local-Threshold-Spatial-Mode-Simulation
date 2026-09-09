#!/usr/bin/env python3
"""Covariance-aware analysis and planning utilities for Phase5 large-X runs.

The microscopic dynamics and checkpoint writer remain in ``spinodal_phase5_core``.
This module only combines stable block checkpoints, preserving common-mode block
pairing in every bootstrap replicate.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from spinodal_phase5_analysis import fit_microscopic_relaxation, kernel_hat
from spinodal_phase5_core import (
    Phase5BlockResult,
    load_block_checkpoint,
    long_wavelength_q_grid,
    q_grid_signature,
)


CAMPAIGN_VERSION = "2026.09.09-phase5-largeX-v1"
PRIMARY_SURVIVAL = 0.90
SENSITIVITY_SURVIVAL = 0.80
CALIBRATION_QR_MAX = 0.15
VALIDATION_QR_MAX = 0.35
UNCONDITIONAL_FIXED_MAX_RELATIVE_DIFFERENCE = 0.10
MAX_METHOD_B_C_RELATIVE_DIFFERENCE = 0.25
MIN_CALIBRATION_R2 = 0.80
MAX_KERNEL_Q4_RELATIVE_DIFFERENCE = 0.05
MAX_HARMONIC_RATIO = 0.10
M_SCHEDULE = (8192, 16384, 32768, 65536)


@dataclass(frozen=True)
class CohortAggregate:
    unconditional: np.ndarray
    current_survivor: np.ndarray
    fixed_survivor: np.ndarray
    survival: np.ndarray
    escape: np.ndarray
    baseline: np.ndarray
    preparation: np.ndarray
    M: int
    fixed_survivor_count: int


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _git_commit(project_root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=project_root, text=True,
        capture_output=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def load_groups(input_dir: Path) -> dict[tuple[float, float, int], list[Phase5BlockResult]]:
    paths = sorted((Path(input_dir) / "blocks").glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no Phase5 checkpoints in {Path(input_dir) / 'blocks'}")
    groups: dict[tuple[float, float, int], list[Phase5BlockResult]] = {}
    for path in paths:
        block = load_block_checkpoint(path)
        key = (float(block.delta), float(block.epsilon_fraction), int(block.mode_index))
        groups.setdefault(key, []).append(block)
    for blocks in groups.values():
        blocks.sort(key=lambda block: block.block_id)
        if len({block.block_id for block in blocks}) != len(blocks):
            raise ValueError(f"duplicate block IDs for {blocks[0].task_id}")
    return groups


def aggregate_cohorts(blocks: Sequence[Phase5BlockResult]) -> CohortAggregate:
    if not blocks:
        raise ValueError("cannot aggregate an empty block list")
    ordered = sorted(blocks, key=lambda block: block.block_id)
    weights = np.asarray([block.block_n for block in ordered], dtype=float)
    M = int(weights.sum())
    unconditional = np.average(
        np.stack([block.A_q for block in ordered]), axis=0, weights=weights
    )
    baseline = np.average(
        np.stack([block.baseline_m for block in ordered]), axis=0, weights=weights
    )
    preparation = np.average(
        np.stack([block.preparation_magnetization for block in ordered]),
        axis=0, weights=weights,
    )
    escape = np.average(
        np.stack([block.escape_fraction_cumulative for block in ordered]),
        axis=0, weights=weights,
    )
    survivor_counts = np.stack([block.survival_count for block in ordered]).sum(axis=0)
    survivor_sums = np.stack(
        [block.survivor_amplitude_sum_current for block in ordered]
    ).sum(axis=0)
    current = np.divide(
        survivor_sums, survivor_counts,
        out=np.full_like(survivor_sums, math.nan, dtype=float),
        where=survivor_counts > 0,
    )
    final_count = int(sum(block.survive_to_T_count for block in ordered))
    fixed_sum = np.stack(
        [block.survive_to_T_amplitude_sum for block in ordered]
    ).sum(axis=0)
    fixed = (
        fixed_sum / final_count
        if final_count > 0
        else np.full_like(unconditional, math.nan)
    )
    return CohortAggregate(
        unconditional=np.asarray(unconditional),
        current_survivor=np.asarray(current),
        fixed_survivor=np.asarray(fixed),
        survival=np.asarray(survivor_counts / M),
        escape=np.asarray(escape),
        baseline=np.asarray(baseline),
        preparation=np.asarray(preparation),
        M=M,
        fixed_survivor_count=final_count,
    )


def adaptive_fit_end(
    q0_amplitude: Sequence[float],
    survival: Sequence[float],
    *,
    preliminary_end: int = 3,
    tau_multiplier: float = 3.0,
    survival_criterion: float = PRIMARY_SURVIVAL,
    maximum_end: int = 20,
) -> int:
    """Pre-specified q=0 tau/survival rule; it never optimizes goodness of fit."""
    amplitude = np.asarray(q0_amplitude, dtype=float)
    survive = np.asarray(survival, dtype=float)
    if len(amplitude) != len(survive) or len(amplitude) <= preliminary_end:
        raise ValueError("q0 amplitude and survival histories are incompatible")
    preliminary = fit_microscopic_relaxation(amplitude, 0, preliminary_end)
    gamma = float(preliminary["Gamma_micro"])
    tau_end = (
        int(math.ceil(tau_multiplier / gamma))
        if math.isfinite(gamma) and gamma > 0.0
        else preliminary_end
    )
    valid_times = np.flatnonzero(survive >= survival_criterion)
    survival_end = int(valid_times[-1]) if valid_times.size else preliminary_end
    return max(
        preliminary_end,
        min(tau_end, survival_end, maximum_end, len(amplitude) - 1),
    )


def origin_D_fit(q: Sequence[float], gamma: Sequence[float], gamma0: float) -> float:
    q_array = np.asarray(q, dtype=float)
    y = np.asarray(gamma, dtype=float) - float(gamma0)
    x = q_array * q_array
    denominator = float(np.dot(x, x))
    return float(np.dot(x, y) / denominator) if denominator > 0.0 else math.nan


def free_intercept_dispersion(q: Sequence[float], gamma: Sequence[float]) -> tuple[float, float]:
    x = np.asarray(q, dtype=float) ** 2
    y = np.asarray(gamma, dtype=float)
    design = np.column_stack((np.ones_like(x), x))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[0]), float(beta[1])


def _summary(samples: np.ndarray) -> tuple[float, float, float]:
    finite = np.asarray(samples, dtype=float)
    finite = finite[np.isfinite(finite)]
    if len(finite) < 2:
        return math.nan, math.nan, math.nan
    return (
        float(np.std(finite, ddof=1)),
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def common_block_bootstrap_gammas(
    mode_blocks: dict[int, Sequence[Phase5BlockResult]],
    *,
    fit_end: int,
    replicates: int,
    seed: int,
) -> tuple[list[int], np.ndarray, list[int]]:
    """Resample identical physical block IDs across every q."""
    modes = sorted(mode_blocks)
    maps = {
        mode: {block.block_id: block for block in blocks}
        for mode, blocks in mode_blocks.items()
    }
    common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    if not common_ids:
        raise ValueError("no block IDs are common to all q modes")
    rng = np.random.Generator(np.random.Philox(seed))
    # A multinomial count vector is exactly equivalent to drawing B block IDs
    # with replacement.  Reusing the same matrix for every mode preserves the
    # covariance and avoids a Python loop over every sampled block.
    counts = rng.multinomial(
        len(common_ids), np.full(len(common_ids), 1.0 / len(common_ids)),
        size=replicates,
    ).astype(float)
    estimates = np.full((replicates, len(modes)), math.nan, dtype=float)
    for column, mode in enumerate(modes):
        ordered = [maps[mode][block_id] for block_id in common_ids]
        block_weights = np.asarray([block.block_n for block in ordered], dtype=float)
        weighted_counts = counts * block_weights[None, :]
        denominator = weighted_counts.sum(axis=1)
        amplitude_by_block = np.stack([block.A_q for block in ordered])
        amplitudes = weighted_counts @ amplitude_by_block
        amplitudes /= denominator[:, None]
        x = amplitudes[:, :fit_end]
        y = amplitudes[:, 1 : fit_end + 1]
        lambda_value = np.divide(
            np.sum(x * y, axis=1), np.sum(x * x, axis=1),
            out=np.full(replicates, math.nan),
            where=np.sum(x * x, axis=1) > np.finfo(float).tiny,
        )
        valid = np.isfinite(lambda_value) & (lambda_value != 0.0)
        estimates[valid, column] = -np.log(np.abs(lambda_value[valid]))
    return modes, estimates, common_ids


def _q_set(mode: int, qR: float) -> str:
    if mode == 0:
        return "q0"
    if qR <= CALIBRATION_QR_MAX + 1e-14:
        return "calibration"
    if qR <= VALIDATION_QR_MAX + 1e-14:
        return "validation"
    return "excluded"


def verify_common_randomness(
    mode_blocks: dict[int, Sequence[Phase5BlockResult]],
) -> dict[str, Any]:
    maps = {
        mode: {block.block_id: block for block in blocks}
        for mode, blocks in mode_blocks.items()
    }
    common_ids = sorted(set.intersection(*(set(mapping) for mapping in maps.values())))
    mismatches: list[dict[str, Any]] = []
    for block_id in common_ids:
        checksums = {maps[mode][block_id].threshold_checksum for mode in maps}
        entropies = {
            tuple(maps[mode][block_id].rng_identifier.get("seed_sequence_entropy", []))
            for mode in maps
        }
        if len(checksums) != 1 or len(entropies) != 1:
            mismatches.append(
                {"block_id": block_id, "checksum_count": len(checksums), "entropy_count": len(entropies)}
            )
    return {
        "common_block_count": len(common_ids),
        "threshold_and_entropy_match": not mismatches,
        "mismatches": mismatches,
    }


def _harmonic_diagnostics(blocks: Sequence[Phase5BlockResult]) -> dict[str, Any]:
    if not blocks or not blocks[0].harmonic_mode_indices.size:
        return {}
    weights = np.asarray([block.block_n for block in blocks], dtype=float)
    values = np.average(
        np.stack([block.harmonic_amplitudes for block in blocks]), axis=0, weights=weights
    )
    target = np.average(np.stack([block.A_q for block in blocks]), axis=0, weights=weights)
    denominator = max(float(np.max(np.abs(target))), 1e-15)
    result: dict[str, Any] = {}
    for index, (order, folded) in enumerate(zip(
        blocks[0].harmonic_orders, blocks[0].harmonic_mode_indices
    )):
        order = int(order)
        aliases_target = int(folded) == int(blocks[0].mode_index)
        result[f"harmonic_{order}_folded_mode"] = int(folded)
        result[f"harmonic_{order}_aliases_target"] = aliases_target
        result[f"harmonic_{order}_max_ratio"] = (
            math.nan if aliases_target
            else float(np.max(np.abs(values[index]))) / denominator
        )
    return result


def analyze_run(
    input_dir: Path,
    output_dir: Path,
    *,
    delta_ps: float,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260909,
    make_figures: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    config_path = input_dir / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    state_path = input_dir / "phase5_run_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(state_path)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not bool(state.get("all_complete", False)):
        raise ValueError(
            f"Phase5 run is incomplete ({state.get('completed_valid_blocks')}/"
            f"{state.get('total_blocks')}); resume before large-X analysis"
        )
    groups = load_groups(input_dir)
    conditions = sorted({(delta, epsilon) for delta, epsilon, _ in groups})
    mode_rows: list[dict[str, Any]] = []
    scaling_rows: list[dict[str, Any]] = []
    bootstrap_correlations: dict[str, tuple[list[int], np.ndarray]] = {}
    common_rng_reports: dict[str, Any] = {}
    M_convergence_rows: list[dict[str, Any]] = []
    cohort_time_rows: list[dict[str, Any]] = []
    failures: list[str] = []

    for condition_index, (delta, epsilon) in enumerate(conditions):
        mode_blocks = {
            mode: blocks for (local_delta, local_epsilon, mode), blocks in groups.items()
            if local_delta == delta and local_epsilon == epsilon
        }
        if 0 not in mode_blocks:
            failures.append(f"delta={delta:g},epsilon={epsilon:g}: missing q=0")
            continue
        aggregates = {mode: aggregate_cohorts(blocks) for mode, blocks in mode_blocks.items()}
        recommended_fit_end = adaptive_fit_end(
            aggregates[0].unconditional,
            aggregates[0].survival,
            survival_criterion=float(config.get("survival_criterion_primary", PRIMARY_SURVIVAL)),
        )
        frozen_cohort_end = config.get("survivor_cohort_end")
        fit_end = (
            int(frozen_cohort_end)
            if frozen_cohort_end is not None
            else recommended_fit_end
        )
        gammas: dict[int, float] = {}
        gamma_fixed: dict[int, float] = {}
        gamma_current: dict[int, float] = {}
        fit_details: dict[int, dict[str, Any]] = {}
        current_fit_details: dict[int, dict[str, Any]] = {}
        fixed_fit_details: dict[int, dict[str, Any]] = {}
        for mode, aggregate in aggregates.items():
            fit = fit_microscopic_relaxation(aggregate.unconditional, 0, fit_end)
            fit_current = fit_microscopic_relaxation(aggregate.current_survivor, 0, fit_end)
            fit_fixed = fit_microscopic_relaxation(aggregate.fixed_survivor, 0, fit_end)
            gammas[mode] = float(fit["Gamma_micro"])
            gamma_current[mode] = float(fit_current["Gamma_micro"])
            gamma_fixed[mode] = float(fit_fixed["Gamma_micro"])
            fit_details[mode] = fit
            current_fit_details[mode] = fit_current
            fixed_fit_details[mode] = fit_fixed
            for time_index in range(len(aggregate.unconditional)):
                cohort_time_rows.append({
                    "condition": f"delta_{delta:.12g}_eps_{epsilon:.12g}",
                    "delta_G": delta, "epsilon_fraction": epsilon,
                    "mode_index": mode, "t": time_index,
                    "A_q_unconditional": aggregate.unconditional[time_index],
                    "A_q_current_survivor": aggregate.current_survivor[time_index],
                    "A_q_fixed_survivor_cohort": aggregate.fixed_survivor[time_index],
                    "survival_fraction_cumulative": aggregate.survival[time_index],
                    "escape_fraction_cumulative": aggregate.escape[time_index],
                    "baseline_m": aggregate.baseline[time_index],
                    "fixed_survivor_count": aggregate.fixed_survivor_count,
                    "M_used": aggregate.M,
                })

        modes, gamma_samples, common_ids = common_block_bootstrap_gammas(
            mode_blocks,
            fit_end=fit_end,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + condition_index,
        )
        q_values = np.asarray([mode_blocks[mode][0].q for mode in modes])
        qR_values = np.asarray([mode_blocks[mode][0].qR for mode in modes])
        gamma_vector = np.asarray([gammas[mode] for mode in modes])
        q0_column = modes.index(0)
        gamma0 = gammas[0]
        calibration_columns = [
            index for index, (mode, qR) in enumerate(zip(modes, qR_values))
            if _q_set(mode, float(qR)) == "calibration"
        ]
        validation_columns = [
            index for index, (mode, qR) in enumerate(zip(modes, qR_values))
            if _q_set(mode, float(qR)) == "validation"
        ]
        calibration_with_q0 = [q0_column, *calibration_columns]
        D = origin_D_fit(
            q_values[calibration_columns], gamma_vector[calibration_columns], gamma0
        ) if calibration_columns else math.nan
        free_gamma0, free_D = (
            free_intercept_dispersion(
                q_values[calibration_with_q0], gamma_vector[calibration_with_q0]
            ) if calibration_columns else (math.nan, math.nan)
        )
        D_samples = np.full(bootstrap_replicates, math.nan)
        xi_samples = np.full(bootstrap_replicates, math.nan)
        Y_samples = np.full((bootstrap_replicates, len(modes)), math.nan)
        residual_samples = np.full_like(Y_samples, math.nan)
        for replicate in range(bootstrap_replicates):
            local_gamma = gamma_samples[replicate]
            local_gamma0 = local_gamma[q0_column]
            local_D = (
                origin_D_fit(
                    q_values[calibration_columns],
                    local_gamma[calibration_columns],
                    local_gamma0,
                ) if calibration_columns else math.nan
            )
            D_samples[replicate] = local_D
            if local_D > 0.0 and local_gamma0 > 0.0:
                local_xi = math.sqrt(local_D / local_gamma0)
                xi_samples[replicate] = local_xi
                Y_samples[replicate] = local_gamma0 / local_gamma
                local_X = q_values * local_xi
                residual_samples[replicate] = Y_samples[replicate] - 1.0 / (1.0 + local_X**2)
        D_se, D_low, D_high = _summary(D_samples)
        xi_se, xi_low, xi_high = _summary(xi_samples)
        D_resolved = bool(math.isfinite(D) and D > 0.0 and D_low > 0.0)
        xi = math.sqrt(D / gamma0) if D_resolved and gamma0 > 0.0 else math.nan
        X_values = q_values * xi if math.isfinite(xi) else np.full_like(q_values, math.nan)
        Y_values = gamma0 / gamma_vector
        predicted = 1.0 / (1.0 + X_values**2)
        residuals = Y_values - predicted
        finite_gamma_samples = gamma_samples[np.all(np.isfinite(gamma_samples), axis=1)]
        if len(finite_gamma_samples) >= 2:
            with np.errstate(invalid="ignore", divide="ignore"):
                correlation = np.atleast_2d(
                    np.corrcoef(finite_gamma_samples, rowvar=False)
                )
        else:
            correlation = np.full((len(modes), len(modes)), math.nan)
        bootstrap_correlations[
            f"delta={delta:.12g},epsilon={epsilon:.12g}"
        ] = (list(modes), correlation)
        common_report = verify_common_randomness(mode_blocks)
        common_rng_reports[f"delta={delta:.12g},epsilon={epsilon:.12g}"] = common_report

        for column, mode in enumerate(modes):
            block = mode_blocks[mode][0]
            aggregate = aggregates[mode]
            gamma_se, gamma_low, gamma_high = _summary(gamma_samples[:, column])
            y_se, y_low, y_high = _summary(Y_samples[:, column])
            residual_se, residual_low, residual_high = _summary(
                residual_samples[:, column]
            )
            fixed_difference = abs(gammas[mode] - gamma_fixed[mode]) / max(
                abs(gammas[mode]), 1e-15
            )
            reasons: list[str] = []
            if aggregate.survival[fit_end] < PRIMARY_SURVIVAL:
                reasons.append("survival_below_primary_0.90")
            if not math.isfinite(gammas[mode]) or gammas[mode] <= 0.0:
                reasons.append("invalid_Gamma")
            if fixed_difference > UNCONDITIONAL_FIXED_MAX_RELATIVE_DIFFERENCE:
                reasons.append("unconditional_fixed_survivor_difference_above_10pct")
            if not bool(fit_details[mode]["fit_r2"] >= 0.8):
                reasons.append("fit_r2_below_0.8")
            if (
                math.isfinite(float(fit_details[mode]["method_B_C_relative_difference"]))
                and float(fit_details[mode]["method_B_C_relative_difference"])
                > MAX_METHOD_B_C_RELATIVE_DIFFERENCE
            ):
                reasons.append("method_B_C_difference_above_25pct")
            row = {
                "condition": f"delta_{delta:.12g}_eps_{epsilon:.12g}",
                "R": int(config["R"]), "N": int(config["N"]),
                "delta_G": delta, "delta_ps": delta_ps, "s": delta - delta_ps,
                "Gamma0": gamma0, "q": block.q, "qR": block.qR,
                "mode_index": mode, "set_type": _q_set(mode, block.qR),
                "Gamma_q": gammas[mode], "Gamma_q_se": gamma_se,
                "Gamma_q_ci_low": gamma_low, "Gamma_q_ci_high": gamma_high,
                "lambda_q": math.exp(-gammas[mode]),
                "Gamma_q_current_survivor": gamma_current[mode],
                "Gamma_q_fixed_survivor": gamma_fixed[mode],
                "lambda_q_current_survivor": current_fit_details[mode]["lambda_micro"],
                "lambda_q_fixed_survivor": fixed_fit_details[mode]["lambda_micro"],
                "current_survivor_fit_r2": current_fit_details[mode]["fit_r2"],
                "fixed_survivor_fit_r2": fixed_fit_details[mode]["fit_r2"],
                "current_survivor_method_B_C_relative_difference": current_fit_details[mode]["method_B_C_relative_difference"],
                "fixed_survivor_method_B_C_relative_difference": fixed_fit_details[mode]["method_B_C_relative_difference"],
                "unconditional_fixed_relative_difference": fixed_difference,
                "survival_fraction_fit_end": aggregate.survival[fit_end],
                "survival_fraction_T": aggregate.survival[-1],
                "escape_fraction": aggregate.escape[fit_end],
                "epsilon_fraction": epsilon, "M_used": aggregate.M,
                "n_blocks": len(mode_blocks[mode]), "fit_start": 0,
                "fit_end": fit_end, "fit_r2": fit_details[mode]["fit_r2"],
                "fit_end_recommended_from_q0": recommended_fit_end,
                "method_B_C_relative_difference": fit_details[mode]["method_B_C_relative_difference"],
                "baseline_drift": float(np.max(np.abs(aggregate.baseline - aggregate.baseline[0]))),
                "preparation_drift": (
                    float(aggregate.preparation[-1] - aggregate.preparation[0])
                    if aggregate.preparation.size else math.nan
                ),
                "X": X_values[column], "Y": Y_values[column],
                "Y_se": y_se, "Y_ci_low": y_low, "Y_ci_high": y_high,
                "Y_pred": predicted[column], "residual": residuals[column],
                "residual_se": residual_se, "residual_ci_low": residual_low,
                "residual_ci_high": residual_high,
                "relative_residual": residuals[column] / predicted[column] if predicted[column] else math.nan,
                "kernel_minus_log": -math.log(abs(kernel_hat(mode, int(config["N"]), int(config["R"])))),
                "kappa_q2": ((int(config["R"]) + 1) * (2 * int(config["R"]) + 1) / 12.0) * block.q**2,
                "reliable": not reasons, "reason": ";".join(reasons) if reasons else "ok",
                **_harmonic_diagnostics(mode_blocks[mode]),
            }
            denominator = row["kappa_q2"]
            row["kernel_q4_relative_difference"] = (
                (row["kernel_minus_log"] - denominator) / denominator
                if denominator > 0.0 else 0.0
            )
            mode_rows.append(row)

        condition_rows = mode_rows[-len(modes):]
        validation_rows = [row for row in condition_rows if row["set_type"] == "validation"]
        min_survival = min(row["survival_fraction_fit_end"] for row in condition_rows)
        fixed_agreement = max(
            row["unconditional_fixed_relative_difference"] for row in condition_rows
        ) <= UNCONDITIONAL_FIXED_MAX_RELATIVE_DIFFERENCE
        all_modes_reliable = all(bool(row["reliable"]) for row in condition_rows)
        if calibration_columns:
            observed_delta_gamma = gamma_vector[calibration_columns] - gamma0
            predicted_delta_gamma = D * q_values[calibration_columns] ** 2
            denominator = float(np.sum((observed_delta_gamma - np.mean(observed_delta_gamma)) ** 2))
            calibration_r2 = (
                1.0 - float(np.sum((observed_delta_gamma - predicted_delta_gamma) ** 2)) / denominator
                if denominator > np.finfo(float).tiny else math.nan
            )
        else:
            calibration_r2 = math.nan
        calibration_stable = bool(math.isfinite(calibration_r2) and calibration_r2 >= MIN_CALIBRATION_R2)
        val_se = np.asarray([row["Y_se"] for row in validation_rows], dtype=float)
        val_residual = np.asarray([row["residual"] for row in validation_rows], dtype=float)
        valid_uncertainty = np.isfinite(val_se) & (val_se > 0.0)
        weighted_rmse = (
            float(np.sqrt(np.average(val_residual[valid_uncertainty] ** 2, weights=1.0 / val_se[valid_uncertainty] ** 2)))
            if np.any(valid_uncertainty) else math.nan
        )
        chi_square = math.nan
        chi_square_dof = 0
        if validation_columns and bootstrap_replicates > len(validation_columns) + 2:
            covariance = np.cov(residual_samples[:, validation_columns], rowvar=False)
            covariance = np.atleast_2d(covariance)
            if np.all(np.isfinite(covariance)) and np.linalg.cond(covariance) < 1e12:
                chi_square = float(val_residual @ np.linalg.inv(covariance) @ val_residual)
                chi_square_dof = len(validation_columns)
        X_max = max((row["X"] for row in validation_rows), default=math.nan)
        precision_met = bool(
            validation_rows
            and all(
                math.isfinite(row["Y_se"])
                and row["Y_se"] <= 0.02
                and row["Y_se"] / max(abs(row["Y"]), 1e-15) <= 0.05
                for row in validation_rows
            )
        )
        validation_consistent = bool(
            validation_rows
            and all(
                math.isfinite(row["residual_se"])
                and abs(row["residual"]) <= 1.96 * row["residual_se"]
                for row in validation_rows
            )
        )
        q4_acceptable = all(
            abs(row["kernel_q4_relative_difference"]) <= MAX_KERNEL_Q4_RELATIVE_DIFFERENCE
            for row in validation_rows
        )
        survival_primary = min_survival >= PRIMARY_SURVIVAL
        survival_sensitivity = min_survival >= SENSITIVITY_SURVIVAL
        epsilon_validated = bool(config.get("epsilon_linearity_validated", False))
        reliable = bool(
            D_resolved and survival_primary and fixed_agreement
            and common_report["threshold_and_entropy_match"]
            and validation_rows and epsilon_validated and all_modes_reliable
            and calibration_stable and validation_consistent and precision_met
            and q4_acceptable
        )
        current_M = min(aggregate.M for aggregate in aggregates.values())
        next_M = next((value for value in M_SCHEDULE if value > current_M), None)
        scaling_rows.append({
            "condition": f"delta_{delta:.12g}_eps_{epsilon:.12g}",
            "R": int(config["R"]), "N": int(config["N"]),
            "delta_G": delta, "delta_ps": delta_ps, "s": delta - delta_ps,
            "Gamma0": gamma0, "D_cal": D, "D_cal_SE": D_se,
            "D_cal_CI_low": D_low, "D_cal_CI_high": D_high,
            "Gamma0_free_intercept": free_gamma0, "D_free_intercept": free_D,
            "calibration_R2": calibration_r2,
            "xi_cal": xi, "xi_cal_SE": xi_se,
            "xi_cal_CI_low": xi_low, "xi_cal_CI_high": xi_high,
            "D_cal_resolved": D_resolved, "X_max_validation": X_max,
            "reached_X_0p3": bool(reliable and X_max >= 0.3),
            "reached_X_0p5": bool(reliable and X_max >= 0.5),
            "validation_RMSE": weighted_rmse, "validation_chi_square": chi_square,
            "validation_chi_square_dof": chi_square_dof,
            "validation_consistent_at_95pct": validation_consistent,
            "kernel_q4_acceptable": q4_acceptable,
            "survival_quality_primary": survival_primary,
            "survival_quality_sensitivity": survival_sensitivity,
            "minimum_survival_fit_end": min_survival,
            "unconditional_fixed_agreement": fixed_agreement,
            "epsilon_linearity": epsilon_validated, "common_rng_verified": common_report["threshold_and_entropy_match"],
            "precision_goal_met": precision_met, "M_used": current_M,
            "next_M_if_needed": None if precision_met else next_M,
            "fit_end": fit_end, "n_common_blocks": len(common_ids),
            "fit_end_recommended_from_q0": recommended_fit_end,
            "bootstrap_replicates": bootstrap_replicates, "reliable": reliable,
            "reason": "ok" if reliable else ";".join(
                reason for flag, reason in (
                    (not D_resolved, "D_cal_not_positive_at_95pct"),
                    (not survival_primary, "survival_below_primary_0.90"),
                    (not fixed_agreement, "conditional_unconditional_disagree"),
                    (not common_report["threshold_and_entropy_match"], "common_RNG_mismatch"),
                    (not validation_rows, "no_validation_modes"),
                    (not epsilon_validated, "epsilon_linearity_not_prevalidated"),
                    (not all_modes_reliable, "one_or_more_mode_fits_unreliable"),
                    (not calibration_stable, "calibration_fit_unstable"),
                    (not validation_consistent, "validation_residual_inconsistent_at_95pct"),
                    (not precision_met, "validation_precision_goal_not_met"),
                    (not q4_acceptable, "kernel_q4_correction_above_5pct"),
                ) if flag
            ),
        })

        # Reuse stable prefix blocks to quantify append-only M convergence of
        # the actual held-out observable, not only individual Gamma(q).
        for M_candidate in M_SCHEDULE:
            if M_candidate > current_M:
                continue
            subset = {
                mode: [block for block in blocks if block.end_trial <= M_candidate]
                for mode, blocks in mode_blocks.items()
            }
            if any(not blocks for blocks in subset.values()):
                continue
            if M_candidate == current_M:
                local_modes, local_samples = modes, gamma_samples
                local_gamma_vector = gamma_vector
            else:
                local_modes, local_samples, _ = common_block_bootstrap_gammas(
                    subset, fit_end=fit_end, replicates=bootstrap_replicates,
                    seed=bootstrap_seed + condition_index + M_candidate,
                )
                local_gamma_vector = np.asarray([
                    fit_microscopic_relaxation(
                        aggregate_cohorts(subset[mode]).unconditional, 0, fit_end
                    )["Gamma_micro"]
                    for mode in local_modes
                ], dtype=float)
            local_q = np.asarray([subset[mode][0].q for mode in local_modes])
            local_qR = np.asarray([subset[mode][0].qR for mode in local_modes])
            local_q0_column = local_modes.index(0)
            local_calibration = [
                index for index, (mode, qR) in enumerate(zip(local_modes, local_qR))
                if _q_set(mode, float(qR)) == "calibration"
            ]
            local_validation = [
                index for index, (mode, qR) in enumerate(zip(local_modes, local_qR))
                if _q_set(mode, float(qR)) == "validation"
            ]
            if not local_calibration or not local_validation:
                continue
            local_gamma0 = local_gamma_vector[local_q0_column]
            local_D = origin_D_fit(
                local_q[local_calibration], local_gamma_vector[local_calibration], local_gamma0
            )
            local_xi = math.sqrt(local_D / local_gamma0) if local_D > 0 and local_gamma0 > 0 else math.nan
            max_validation_column = local_validation[-1]
            local_Y = local_gamma0 / local_gamma_vector[max_validation_column]
            local_Y_samples = local_samples[:, local_q0_column] / local_samples[:, max_validation_column]
            local_Y_se, local_Y_low, local_Y_high = _summary(local_Y_samples)
            M_convergence_rows.append({
                "condition": f"delta_{delta:.12g}_eps_{epsilon:.12g}",
                "M_used": M_candidate,
                "n_blocks": len(subset[0]),
                "Gamma0": local_gamma0,
                "D_cal": local_D,
                "xi_cal": local_xi,
                "X_max_validation": local_q[max_validation_column] * local_xi,
                "Y_at_max_validation_q": local_Y,
                "Y_SE": local_Y_se,
                "Y_CI_low": local_Y_low,
                "Y_CI_high": local_Y_high,
                "precision_goal_met": bool(
                    math.isfinite(local_Y_se) and local_Y_se <= 0.02
                    and local_Y_se / max(abs(local_Y), 1e-15) <= 0.05
                ),
            })

    mode_table = pd.DataFrame(mode_rows)
    scaling_table = pd.DataFrame(scaling_rows)
    if not scaling_table.empty:
        reliable_rows = scaling_table[scaling_table["reliable"]]
        maximum_reliable_X = (
            float(reliable_rows["X_max_validation"].max()) if not reliable_rows.empty else None
        )
        best = (
            reliable_rows.sort_values("X_max_validation").iloc[-1].to_dict()
            if not reliable_rows.empty else None
        )
    else:
        maximum_reliable_X = None
        best = None
    epsilon_recommendations = epsilon_linearity_recommendations(mode_table)
    M_convergence = pd.DataFrame(M_convergence_rows)
    covariance_rows = []
    for condition, (correlation_modes, matrix) in bootstrap_correlations.items():
        for row_index, mode_i in enumerate(correlation_modes):
            for column_index, mode_j in enumerate(correlation_modes):
                covariance_rows.append({
                    "condition": condition, "mode_i": mode_i, "mode_j": mode_j,
                    "Gamma_bootstrap_correlation": matrix[row_index, column_index],
                })
    covariance_table = pd.DataFrame(covariance_rows)
    cohort_time_table = pd.DataFrame(cohort_time_rows)
    baseline_path = Path(__file__).resolve().parent.parent / "results/runs/phase5_largeX/baseline_largeX_summary.json"
    baseline_condition = None
    baseline_gamma0 = math.nan
    if baseline_path.is_file():
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
        baseline_entries = baseline_payload.get("entries", [])
        baseline_condition = next(
            (entry for entry in baseline_entries if entry.get("scope") == "R24_closest_reliable_operational_condition"),
            None,
        )
        if baseline_condition:
            baseline_gamma0 = float(baseline_condition["Gamma0_micro"])
    q0_rows = mode_table[mode_table["set_type"] == "q0"] if not mode_table.empty else mode_table
    q0_scout = []
    for _, row in q0_rows.iterrows():
        q0_scout.append({
            "delta_G": row["delta_G"], "s": row["s"], "Gamma0": row["Gamma_q"],
            "Gamma0_over_baseline": row["Gamma_q"] / baseline_gamma0 if baseline_gamma0 > 0 else None,
            "survival_fraction_fit_end": row["survival_fraction_fit_end"],
            "escape_fraction": row["escape_fraction"], "fit_end": int(row["fit_end"]),
            "fit_end_recommended_from_q0": int(row["fit_end_recommended_from_q0"]),
            "passes_primary_survival": bool(row["survival_fraction_fit_end"] >= PRIMARY_SURVIVAL),
        })
    summary = {
        "campaign_version": CAMPAIGN_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(Path(__file__).resolve().parent.parent),
        "input_dir": str(input_dir),
        "campaign_parameters": config,
        "operational_coordinate": {
            "delta_G": "abs(Delta - Delta_sp_Gaussian)",
            "delta_ps": delta_ps,
            "criterion": "P_esc_cum(T_obs=50)=0.10",
            "s_role": "matched/operational coordinate only; not a true-spinodal distance",
        },
        "q_grid": config.get("q_grid", []),
        "baseline_condition": baseline_condition,
        "q0_scout": {
            "target_Gamma0_ratios": [1.0, 0.5, 1.0 / 3.0, 0.2, 0.1],
            "observed": q0_scout,
            "selection_basis": "measured Gamma0 plus survival/escape; s is not the primary selector",
        },
        "calibration_rule": "qR<=0.15; origin-constrained DeltaGamma=D_cal*q^2",
        "validation_rule": "0.15<qR<=0.35; held out from D_cal",
        "survival": {"primary": PRIMARY_SURVIVAL, "sensitivity": SENSITIVITY_SURVIVAL},
        "common_rng_mode": config.get("rng_coupling_mode"),
        "common_rng_verification": common_rng_reports,
        "bootstrap": {"type": "common-block paired bootstrap", "replicates": bootstrap_replicates, "seed": bootstrap_seed},
        "M_schedule": list(M_SCHEDULE),
        "M_convergence": M_convergence.to_dict(orient="records"),
        "epsilon_recommendations": epsilon_recommendations.to_dict(orient="records"),
        "best_condition": best,
        "maximum_reliable_X": maximum_reliable_X,
        "reached_X_0p3": bool(maximum_reliable_X is not None and maximum_reliable_X >= 0.3),
        "reached_X_0p5": bool(maximum_reliable_X is not None and maximum_reliable_X >= 0.5),
        "next_Gamma0_scout_point_required": bool(maximum_reliable_X is None or maximum_reliable_X < 0.5),
        "failures": failures + (
            scaling_table.loc[~scaling_table["reliable"], ["condition", "reason"]]
            .astype(str).agg(": ".join, axis=1).tolist() if not scaling_table.empty else []
        ),
        "warnings": [
            "finite-R microscopic results use an operational pseudospinodal, not a proven true spinodal",
            "conditional survivor response is diagnostic and cannot replace failed unconditional criteria",
            "q^4 kernel corrections are reported rather than assumed to vanish",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    mode_table.to_csv(output_dir / "largeX_mode_results.csv", index=False)
    scaling_table.to_csv(output_dir / "largeX_scaling_results.csv", index=False)
    epsilon_recommendations.to_csv(output_dir / "largeX_epsilon_linearity.csv", index=False)
    M_convergence.to_csv(output_dir / "largeX_M_convergence.csv", index=False)
    covariance_table.to_csv(output_dir / "largeX_common_random_covariance.csv", index=False)
    cohort_time_table.to_csv(output_dir / "largeX_survival_cohort_timeseries.csv", index=False)
    pd.DataFrame(kernel_diagnostic(config)).to_csv(
        output_dir / "largeX_kernel_diagnostic.csv", index=False
    )
    (output_dir / "largeX_validation_summary.json").write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if make_figures:
        make_largeX_figures(
            mode_table, scaling_table, M_convergence, bootstrap_correlations, output_dir
        )
    return mode_table, scaling_table, summary


def epsilon_linearity_recommendations(mode_table: pd.DataFrame) -> pd.DataFrame:
    columns = ["delta_G", "recommended_epsilon_fraction", "linearity_passed", "reason"]
    if mode_table.empty or mode_table["epsilon_fraction"].nunique() < 2:
        return pd.DataFrame(columns=columns)
    rows = []
    for delta, group in mode_table.groupby("delta_G"):
        epsilon_values = sorted(group["epsilon_fraction"].unique())
        reference = group[np.isclose(group["epsilon_fraction"], epsilon_values[0])]
        accepted = [epsilon_values[0]]
        for epsilon in epsilon_values[1:]:
            candidate = group[np.isclose(group["epsilon_fraction"], epsilon)]
            merged = reference.merge(candidate, on="mode_index", suffixes=("_ref", "_candidate"))
            compatible = True
            for _, row in merged.iterrows():
                difference = abs(row["Gamma_q_candidate"] - row["Gamma_q_ref"])
                tolerance = 1.96 * math.sqrt(row["Gamma_q_se_candidate"]**2 + row["Gamma_q_se_ref"]**2)
                compatible &= bool(math.isfinite(tolerance) and difference <= tolerance)
                harmonic_ratios = [
                    row.get(f"harmonic_{order}_max_ratio_candidate", math.nan)
                    for order in (3, 5)
                ]
                compatible &= all(
                    not math.isfinite(float(ratio)) or float(ratio) <= MAX_HARMONIC_RATIO
                    for ratio in harmonic_ratios
                )
            if compatible:
                accepted.append(epsilon)
        rows.append({
            "delta_G": delta,
            "recommended_epsilon_fraction": max(accepted),
            "linearity_passed": True,
            "reason": "largest epsilon compatible with 0.025 within combined 95% bootstrap uncertainty and non-aliased 3q/5q leakage<=10%",
        })
    return pd.DataFrame(rows, columns=columns)


def kernel_diagnostic(config: dict[str, Any]) -> list[dict[str, Any]]:
    R, N = int(config["R"]), int(config["N"])
    a = float(config.get("lattice_spacing", 1.0))
    grid = long_wavelength_q_grid(N, R, a)
    kappa = a * a * (R + 1) * (2 * R + 1) / 12.0
    rows = []
    for row in grid:
        mode, q = int(row["mode_index"]), float(row["q"])
        exact = -math.log(abs(kernel_hat(mode, N, R)))
        leading = kappa * q * q
        rows.append({**row, "minus_log_abs_K": exact, "kappa_q2": leading,
                     "relative_difference": (exact-leading)/leading if leading else 0.0})
    return rows


def estimate_memory(N: int, block_size: int, T: int, ranks: int) -> dict[str, Any]:
    sites = int(N) * int(block_size)
    components = {
        "spin_state_arrays_bytes": 8 * sites,
        "threshold_bytes": 8 * sites,
        "integer_neighbor_work_bytes": 32 * sites,
        "floating_kernel_work_bytes": 160 * sites,
        "survival_history_bytes": 8 * block_size * (T + 1),
    }
    per_rank = sum(components.values())
    return {
        "N": N, "block_size": block_size, "T": T, "ranks": ranks,
        "method": "conservative live-array accounting; verify with peak_rss_mb on SQUID",
        "components": components,
        "estimated_per_rank_MiB": per_rank / 2**20,
        "estimated_node_total_GiB": per_rank * ranks / 2**30,
    }


def build_baseline(poster_dir: Path, output_dir: Path) -> dict[str, Any]:
    poster_dir, output_dir = Path(poster_dir), Path(output_dir)
    precision = pd.read_csv(poster_dir / "resultB_precision_diagnostic.csv")
    collapse_path = poster_dir / "resultB_final_combined_collapse.csv"
    collapse = pd.read_csv(collapse_path)
    reliable = precision[precision["condition_reliable"].astype(bool)]
    global_row = reliable.loc[reliable["X_max"].idxmax()]
    r24 = reliable[reliable["R"] == 24]
    closest = r24.loc[r24["matched_distance"].idxmin()]
    r24_max = r24.loc[r24["X_max"].idxmax()]

    def payload(row: pd.Series, label: str) -> dict[str, Any]:
        modes = collapse[
            (collapse["R"] == row["R"])
            & np.isclose(collapse["delta"], row["delta"], rtol=0.0, atol=5e-12)
        ].sort_values("mode_index")
        return {
            "scope": label,
            "R": int(row["R"]), "N": int(row["N"]),
            "delta_G": float(row["delta"]),
            "delta_ps": float(row["delta"] - row["matched_distance"]),
            "s": float(row["matched_distance"]),
            "Gamma0_micro": float(modes[modes["mode_index"] == 0]["Gamma_micro"].iloc[0]),
            "Gamma_q": modes[["mode_index", "q", "qR", "Gamma_micro"]].to_dict(orient="records"),
            "D_micro": float(row["D_micro"]),
            "D_micro_se": float(row["sigma_D"]),
            "xi_dyn_micro": float(modes["xi_dyn_micro"].iloc[0]),
            "X_max": float(row["X_max"]), "M": int(row["M"]),
            "epsilon_fraction": 0.05,
            "escape_fraction_cumulative": float(row["cumulative_escape_fraction"]),
            "survival_fraction": float(row["survival_fraction"]),
            "fit_start": 0, "fit_end": 3,
            "source": f"{poster_dir / 'resultB_precision_diagnostic.csv'}; {collapse_path}",
        }

    entries = [
        payload(closest, "R24_closest_reliable_operational_condition"),
        payload(r24_max, "R24_maximum_X"),
        payload(global_row, "all_R_maximum_X_reported_approximately_0p165"),
    ]
    ps_path = Path(__file__).resolve().parent.parent / "results/runs/phase5_R_sweep/finite_size/R024_N4096/pseudospinodal_fine/analysis/phase5_pseudospinodal_time_dependence.csv"
    new_reference = None
    if ps_path.is_file():
        ps_table = pd.read_csv(ps_path)
        row = ps_table[ps_table["T_obs"] == 50]
        if len(row) == 1:
            new_reference = {
                "R": 24, "N": 4096, "T_obs": 50,
                "criterion_probability": 0.10,
                "delta_ps": float(row["delta_ps_estimate"].iloc[0]),
                "delta_ps_se": float(row["delta_ps_se"].iloc[0]),
                "source": str(ps_path),
            }
    summary = {
        "campaign_version": CAMPAIGN_VERSION,
        "statement": "The reported X≈0.165 is the R=48 global maximum; R=24 reaches X=0.161701 at its closest reliable condition.",
        "entries": entries,
        "new_campaign_operational_reference": new_reference,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "baseline_largeX_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    pd.DataFrame([{key: value for key, value in entry.items() if key != "Gamma_q"} for entry in entries]).to_csv(
        output_dir / "baseline_largeX_summary.csv", index=False
    )
    return summary


def make_largeX_figures(
    mode: pd.DataFrame, scaling: pd.DataFrame, M_convergence: pd.DataFrame,
    correlations: dict[str, tuple[list[int], np.ndarray]], output_dir: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    def save(fig: Any, name: str) -> None:
        fig.tight_layout(); fig.savefig(figures / name, dpi=180); plt.close(fig)

    fig, ax = plt.subplots()
    q0 = mode[mode["set_type"] == "q0"] if not mode.empty else mode
    if not q0.empty:
        twin = ax.twinx()
        ax.errorbar(q0["s"], q0["Gamma_q"], yerr=q0["Gamma_q_se"], fmt="o", label="Gamma0")
        twin.plot(q0["s"], q0["survival_fraction_fit_end"], "s--", color="tab:orange", label="survival")
        twin.axhline(PRIMARY_SURVIVAL, color="0.5", ls=":")
        ax.set(xlabel="s (operational coordinate)", ylabel="Gamma0")
        twin.set_ylabel("survival at fit end")
    save(fig, "01_q0_scout_gamma_survival.png")

    fig, ax = plt.subplots()
    if not q0.empty:
        for delta, group in q0.groupby("delta_G"):
            ax.errorbar(group["epsilon_fraction"], group["Gamma_q"], yerr=group["Gamma_q_se"], fmt="o-", label=f"delta_G={delta:.4g}")
        ax.legend(fontsize=7); ax.set(xlabel="epsilon fraction", ylabel="Gamma0")
    save(fig, "02_epsilon_linearity.png")

    fig, ax = plt.subplots()
    if not mode.empty:
        for condition, group in mode[mode["set_type"].isin(["q0", "calibration"])].groupby("condition"):
            q0_gamma = float(group[group["set_type"] == "q0"]["Gamma_q"].iloc[0])
            ax.errorbar(group["q"]**2, group["Gamma_q"]-q0_gamma, yerr=group["Gamma_q_se"], fmt="o", label=condition)
        ax.legend(fontsize=6); ax.set(xlabel="q^2", ylabel="Gamma(q)-Gamma0")
    save(fig, "03_calibration_dispersion.png")

    fig, ax = plt.subplots()
    if not mode.empty:
        markers = {"calibration": "o", "validation": "s"}
        for set_type, marker in markers.items():
            group = mode[(mode["set_type"] == set_type) & np.isfinite(mode["X"])]
            ax.errorbar(group["X"], group["Y"], yerr=group["Y_se"], fmt=marker, ls="none", label=f"{set_type} ({'fit' if set_type=='calibration' else 'held out'})")
        xmax = max(0.55, float(mode["X"].max()) if np.any(np.isfinite(mode["X"])) else 0.55)
        x = np.linspace(0, xmax, 300); ax.plot(x, 1/(1+x*x), "k-", label="1/(1+X^2)")
        ax.legend(fontsize=7); ax.set(
            xlabel="X=q xi_cal", ylabel="Gamma0/Gamma(q)",
            title="D, xi: qR<=0.15 only; squares are held-out validation",
        )
    save(fig, "04_largeX_scaling_validation.png")

    fig, ax = plt.subplots()
    if not mode.empty:
        group = mode[mode["set_type"] == "validation"]
        ax.errorbar(group["X"], group["residual"], yerr=group["residual_se"], fmt="s")
        ax.axhline(0, color="k", lw=1); ax.set(xlabel="X", ylabel="Y-1/(1+X^2)")
    save(fig, "05_largeX_residuals.png")

    fig, ax = plt.subplots()
    if not q0.empty:
        ax.plot(q0["s"], q0["Gamma_q"], "o-", label="unconditional")
        ax.plot(q0["s"], q0["Gamma_q_current_survivor"], "s--", label="current survivor")
        ax.plot(q0["s"], q0["Gamma_q_fixed_survivor"], "^--", label="fixed survivor cohort")
        ax.legend(fontsize=7); ax.set(xlabel="s", ylabel="Gamma0")
    save(fig, "06_survival_conditioning_comparison.png")

    fig, ax = plt.subplots()
    if not M_convergence.empty:
        for condition, group in M_convergence.groupby("condition"):
            ax.errorbar(group["M_used"], group["Y_at_max_validation_q"], yerr=group["Y_SE"], fmt="o-", label=condition)
        ax.legend(fontsize=6); ax.set(xscale="log", xlabel="M", ylabel="Y at maximum validation q")
    save(fig, "07_M_convergence.png")

    fig, ax = plt.subplots()
    if not mode.empty:
        unique = mode.sort_values("mode_index").drop_duplicates(["R", "N", "mode_index"])
        ax.plot(unique["qR"], unique["kernel_minus_log"], "o-", label="-ln|K_R|")
        ax.plot(unique["qR"], unique["kappa_q2"], "s--", label="kappa q^2")
        ax.legend(); ax.set(xlabel="qR", ylabel="kernel relaxation contribution")
    save(fig, "08_kernel_q4_diagnostic.png")

    fig, ax = plt.subplots()
    if correlations:
        correlation_modes, matrix = next(iter(correlations.values()))
        image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        fig.colorbar(image, ax=ax)
        ax.set_xticks(range(len(correlation_modes)), correlation_modes)
        ax.set_yticks(range(len(correlation_modes)), correlation_modes)
        ax.set(title="common-block Gamma correlation", xlabel="mode", ylabel="mode")
    save(fig, "09_common_random_covariance.png")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase5 large-X checkpoint analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    baseline = sub.add_parser("baseline")
    baseline.add_argument("--poster-dir", type=Path, default=Path("results/runs/poster_ABCD"))
    baseline.add_argument("--output-dir", type=Path, default=Path("results/runs/phase5_largeX"))
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--input-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, default=None)
    analyze.add_argument("--delta-ps", type=float, required=True)
    analyze.add_argument("--bootstrap-replicates", type=int, default=2000)
    analyze.add_argument("--bootstrap-seed", type=int, default=20260909)
    analyze.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True)
    memory = sub.add_parser("memory")
    memory.add_argument("--N", type=int, default=4096)
    memory.add_argument("--block-size", type=int, default=32)
    memory.add_argument("--T", type=int, default=50)
    memory.add_argument("--ranks", type=int, default=76)
    memory.add_argument("--output", type=Path, default=None)
    grid = sub.add_parser("q-grid")
    grid.add_argument("--N", type=int, default=4096)
    grid.add_argument("--R", type=int, default=24)
    grid.add_argument("--a", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "baseline":
        payload = build_baseline(args.poster_dir, args.output_dir)
    elif args.command == "analyze":
        output = args.output_dir or args.input_dir / "largeX_analysis"
        _, _, payload = analyze_run(
            args.input_dir, output, delta_ps=args.delta_ps,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed, make_figures=args.figures,
        )
    elif args.command == "memory":
        payload = estimate_memory(args.N, args.block_size, args.T, args.ranks)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        rows = long_wavelength_q_grid(args.N, args.R, args.a)
        payload = {"rows": rows, "signature": q_grid_signature(
            rows, N=args.N, R=args.R, lattice_spacing=args.a,
            calibration_qR_max=CALIBRATION_QR_MAX,
            validation_qR_max=VALIDATION_QR_MAX,
        )}
    print(json.dumps(_json_safe(payload), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
