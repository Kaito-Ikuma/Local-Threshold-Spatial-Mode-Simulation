#!/usr/bin/env python3
"""Rank-0 post-processing for microscopic Spinodal Phase5 checkpoints."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spatial_mode_ensemble_validation import Config, kernel_hat
from spinodal_phase34 import fit_power_law
from spinodal_phase5_core import (
    SCRIPT_VERSION,
    Phase5BlockResult,
    Phase5Task,
    Phase5WorkUnit,
    load_block_checkpoint,
)


M_CONVERGENCE_COLUMNS = [
    "task_id",
    "delta",
    "mode_index",
    "epsilon_fraction",
    "M_requested",
    "M_used",
    "n_blocks",
    "Gamma_micro",
    "fit_r2",
]

SCALING_SUMMARY_COLUMNS = [
    "quantity",
    "amplitude",
    "exponent",
    "exponent_regression_se",
    "r2",
    "n_points",
    "interpretation",
]


@dataclass(frozen=True)
class Phase5Analysis:
    mode_results: pd.DataFrame
    dispersion_fits: pd.DataFrame
    scaling_summary: pd.DataFrame
    epsilon_convergence: pd.DataFrame
    M_convergence: pd.DataFrame
    fit_window_diagnostics: pd.DataFrame
    validation_summary: dict[str, Any]


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    sse = float(np.sum((observed - predicted) ** 2))
    sst = float(np.sum((observed - np.mean(observed)) ** 2))
    return 1.0 - sse / sst if sst > np.finfo(float).tiny else math.nan


def aggregate_block_series(
    blocks: Sequence[Phase5BlockResult],
) -> dict[str, np.ndarray | float | int]:
    """Combine checkpoints in deterministic block-ID order."""
    if not blocks:
        raise ValueError("at least one block is required")
    ordered = sorted(blocks, key=lambda block: block.block_id)
    lengths = {len(block.A_q) for block in ordered}
    if len(lengths) != 1:
        raise ValueError("block time-series lengths differ")
    weights = np.asarray([block.block_n for block in ordered], dtype=float)
    total = int(np.sum(weights))
    if total <= 0:
        raise ValueError("total trial count must be positive")

    def average(name: str) -> np.ndarray:
        values = np.stack(
            [np.asarray(getattr(block, name), dtype=float) for block in ordered]
        )
        return np.average(values, axis=0, weights=weights)

    return {
        "A_q": average("A_q"),
        "mean_m_plus": average("mean_m_plus"),
        "mean_m_minus": average("mean_m_minus"),
        "baseline_m": average("baseline_m"),
        "escape_fraction": average("escape_fraction"),
        "preparation_magnetization": average("preparation_magnetization"),
        "M_total": total,
        "wall_seconds": float(sum(block.wall_seconds for block in ordered)),
        "epsilon_achieved": float(
            np.average([block.epsilon_achieved for block in ordered], weights=weights)
        ),
        "initial_amplitude": float(
            np.average([block.initial_amplitude for block in ordered], weights=weights)
        ),
    }


def fit_microscopic_relaxation(
    amplitude: Sequence[float], fit_start: int, fit_end: int
) -> dict[str, float | int]:
    """Return Method B origin regression and Method C log-amplitude fit."""
    values = np.asarray(amplitude, dtype=float)
    if not (0 <= fit_start < fit_end < len(values)):
        raise ValueError("fit window lies outside amplitude series")
    x = values[fit_start:fit_end]
    y = values[fit_start + 1 : fit_end + 1]
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(float).tiny:
        lambda_B = r2_B = Gamma_B = math.nan
    else:
        lambda_B = float(np.dot(x, y) / denominator)
        r2_B = _r_squared(y, lambda_B * x)
        Gamma_B = (
            -math.log(abs(lambda_B))
            if math.isfinite(lambda_B) and lambda_B != 0.0
            else math.nan
        )

    times = np.arange(fit_start, fit_end + 1, dtype=float)
    selected = values[fit_start : fit_end + 1]
    floor = max(abs(values[0]) * 1e-6, 1e-14)
    valid = np.isfinite(selected) & (np.abs(selected) > floor)
    if int(np.sum(valid)) < 2 or abs(values[0]) <= floor:
        Gamma_C = r2_C = intercept_C = math.nan
        n_C = int(np.sum(valid))
    else:
        design = np.column_stack((np.ones(int(np.sum(valid))), times[valid]))
        log_y = np.log(np.abs(selected[valid] / values[0]))
        beta, _, _, _ = np.linalg.lstsq(design, log_y, rcond=None)
        predicted = design @ beta
        intercept_C = float(beta[0])
        Gamma_C = -float(beta[1])
        r2_C = _r_squared(log_y, predicted)
        n_C = int(np.sum(valid))
    return {
        "lambda_micro": lambda_B,
        "Gamma_micro": Gamma_B,
        "fit_r2": r2_B,
        "Gamma_logfit": Gamma_C,
        "logfit_r2": r2_C,
        "logfit_intercept": intercept_C,
        "logfit_n_points": n_C,
        "method_B_C_relative_difference": (
            abs(Gamma_B - Gamma_C) / max(abs(Gamma_B), 1e-15)
            if math.isfinite(Gamma_B) and math.isfinite(Gamma_C)
            else math.nan
        ),
    }


def bootstrap_gamma(
    blocks: Sequence[Phase5BlockResult],
    fit_start: int,
    fit_end: int,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    if replicates < 1:
        return np.empty(0, dtype=float)
    ordered = sorted(blocks, key=lambda block: block.block_id)
    rng = np.random.Generator(np.random.Philox(seed))
    estimates = []
    for _ in range(replicates):
        selected = [ordered[index] for index in rng.integers(0, len(ordered), len(ordered))]
        amplitude = np.asarray(aggregate_block_series(selected)["A_q"])
        gamma = float(
            fit_microscopic_relaxation(amplitude, fit_start, fit_end)["Gamma_micro"]
        )
        if math.isfinite(gamma):
            estimates.append(gamma)
    return np.asarray(estimates, dtype=float)


def _bootstrap_summary(samples: np.ndarray) -> tuple[float, float, float]:
    if len(samples) < 2:
        return math.nan, math.nan, math.nan
    return (
        float(np.std(samples, ddof=1)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _fit_window_rows(
    task: Phase5Task, amplitude: np.ndarray
) -> list[dict[str, Any]]:
    rows = []
    primary_width = task.fit_end - task.fit_start
    candidates = sorted(
        {
            (task.fit_start, task.fit_end),
            (task.fit_start, min(task.T, task.fit_end + max(2, primary_width // 2))),
            (min(2, task.fit_end - 1), task.fit_end),
        }
    )
    for start, end in candidates:
        if end <= start:
            continue
        fit = fit_microscopic_relaxation(amplitude, start, end)
        rows.append(
            {
                "task_id": task.task_id,
                "delta": task.delta,
                "mode_index": task.mode_index,
                "epsilon_fraction": task.epsilon_fraction,
                "fit_start": start,
                "fit_end": end,
                "Gamma_micro": fit["Gamma_micro"],
                "Gamma_logfit": fit["Gamma_logfit"],
                "fit_r2": fit["fit_r2"],
                "primary_window": start == task.fit_start and end == task.fit_end,
            }
        )
    return rows


def analyze_mode_tasks(
    tasks: Sequence[Phase5Task],
    blocks_by_task: dict[str, list[Phase5BlockResult]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    window_rows = []
    bootstrap_by_task: dict[str, np.ndarray] = {}
    for index, task in enumerate(sorted(tasks, key=lambda item: item.task_id)):
        blocks = blocks_by_task.get(task.task_id, [])
        if not blocks:
            raise ValueError(f"no checkpoints found for task {task.task_id}")
        aggregate = aggregate_block_series(blocks)
        amplitude = np.asarray(aggregate["A_q"], dtype=float)
        fit = fit_microscopic_relaxation(amplitude, task.fit_start, task.fit_end)
        samples = bootstrap_gamma(
            blocks,
            task.fit_start,
            task.fit_end,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed + index,
        )
        bootstrap_by_task[task.task_id] = samples
        se, ci_low, ci_high = _bootstrap_summary(samples)
        gamma = float(fit["Gamma_micro"])
        baseline = np.asarray(aggregate["baseline_m"], dtype=float)
        escape = np.asarray(aggregate["escape_fraction"], dtype=float)
        preparation = np.asarray(
            aggregate["preparation_magnetization"], dtype=float
        )
        reasons = []
        if not math.isfinite(gamma) or gamma <= 0.0:
            reasons.append("nonpositive_or_nonfinite_Gamma")
        if math.isfinite(float(fit["fit_r2"])) and float(fit["fit_r2"]) < 0.8:
            reasons.append("fit_r2_below_0.8")
        if abs(float(aggregate["initial_amplitude"])) < 1e-8:
            reasons.append("initial_amplitude_too_small")
        if math.isfinite(float(fit["method_B_C_relative_difference"])) and float(
            fit["method_B_C_relative_difference"]
        ) > 0.25:
            reasons.append("method_B_C_difference_above_25pct")
        if float(np.max(escape)) > 0.1:
            reasons.append("escape_fraction_above_10pct")
        rows.append(
            {
                "task_id": task.task_id,
                "task_group": task.task_group,
                "delta": task.delta,
                "Delta": task.Delta,
                "mode_index": task.mode_index,
                "q": blocks[0].q,
                "qR": blocks[0].qR,
                "N": task.N,
                "R": task.R,
                "M_total": int(aggregate["M_total"]),
                "n_blocks": len(blocks),
                "block_size": task.block_size,
                "epsilon_fraction": task.epsilon_fraction,
                "epsilon_achieved": float(aggregate["epsilon_achieved"]),
                "epsilon_target": blocks[0].epsilon_target,
                "initialization_mode": task.initialization_mode,
                "microscopic_kernel": task.microscopic_kernel,
                "Gamma_micro": gamma,
                "Gamma_micro_se": se,
                "Gamma_micro_ci_low": ci_low,
                "Gamma_micro_ci_high": ci_high,
                "lambda_micro": fit["lambda_micro"],
                "Gamma_logfit": fit["Gamma_logfit"],
                "method_B_C_relative_difference": fit[
                    "method_B_C_relative_difference"
                ],
                "Gamma_closure": task.Gamma_closure,
                "Gamma_difference": gamma - task.Gamma_closure,
                "Gamma_ratio": gamma / task.Gamma_closure,
                "fit_start": task.fit_start,
                "fit_end": task.fit_end,
                "fit_r2": fit["fit_r2"],
                "logfit_r2": fit["logfit_r2"],
                "baseline_drift": float(np.max(np.abs(baseline - baseline[0]))),
                "escape_fraction": float(np.max(escape)),
                "preparation_initial_m": (
                    float(preparation[0]) if preparation.size else math.nan
                ),
                "preparation_final_m": (
                    float(preparation[-1]) if preparation.size else math.nan
                ),
                "preparation_drift": (
                    float(preparation[-1] - preparation[0])
                    if preparation.size
                    else math.nan
                ),
                "reliable": not reasons,
                "reliability_reason": ";".join(reasons) if reasons else "ok",
                "wall_seconds": float(aggregate["wall_seconds"]),
            }
        )
        window_rows.extend(_fit_window_rows(task, amplitude))
    return pd.DataFrame(rows), pd.DataFrame(window_rows), bootstrap_by_task


def fit_microscopic_dispersion(
    mode_results: pd.DataFrame,
    tasks_by_id: dict[str, Phase5Task],
    blocks_by_task: dict[str, list[Phase5BlockResult]],
    *,
    primary_epsilon_fraction: float,
    qR_max: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    closure_dispersion: pd.DataFrame,
    kappa_R: float,
) -> pd.DataFrame:
    primary = mode_results[
        np.isclose(
            mode_results["epsilon_fraction"],
            primary_epsilon_fraction,
            rtol=1e-12,
            atol=1e-15,
        )
    ].copy()
    rows = []
    closure_by_delta = closure_dispersion.set_index("delta")
    for delta, group in primary.groupby("delta"):
        eligible = group[
            (group["qR"] <= qR_max)
            & group["reliable"]
            & np.isfinite(group["Gamma_micro"])
        ].sort_values("mode_index")
        q0 = eligible[eligible["mode_index"] == 0]
        if len(eligible) < 2 or q0.empty:
            continue
        x = eligible["q"].to_numpy(dtype=float) ** 2
        y = eligible["Gamma_micro"].to_numpy(dtype=float)
        design = np.column_stack((np.ones_like(x), x))
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        predicted = design @ beta
        bootstrap_D = []
        rng = np.random.Generator(
            np.random.Philox([bootstrap_seed, int(round(-math.log10(delta)))])
        )
        for _ in range(bootstrap_replicates):
            gamma_values = []
            valid = True
            for task_id in eligible["task_id"]:
                task = tasks_by_id[str(task_id)]
                blocks = blocks_by_task[str(task_id)]
                selected = [
                    blocks[index]
                    for index in rng.integers(0, len(blocks), len(blocks))
                ]
                gamma = float(
                    fit_microscopic_relaxation(
                        np.asarray(aggregate_block_series(selected)["A_q"]),
                        task.fit_start,
                        task.fit_end,
                    )["Gamma_micro"]
                )
                if not math.isfinite(gamma):
                    valid = False
                    break
                gamma_values.append(gamma)
            if valid:
                boot_beta, _, _, _ = np.linalg.lstsq(
                    design, np.asarray(gamma_values), rcond=None
                )
                bootstrap_D.append(float(boot_beta[1]))
        D_samples = np.asarray(bootstrap_D)
        D_se, D_low, D_high = _bootstrap_summary(D_samples)
        closure = closure_by_delta.loc[delta]
        rows.append(
            {
                "delta": float(delta),
                "Gamma0_micro": float(q0["Gamma_micro"].iloc[0]),
                "Gamma0_micro_se": float(q0["Gamma_micro_se"].iloc[0]),
                "D_micro": float(beta[1]),
                "D_micro_se": D_se,
                "D_micro_ci_low": D_low,
                "D_micro_ci_high": D_high,
                "Gamma0_closure": float(q0["Gamma_closure"].iloc[0]),
                "D_closure": float(closure["D_fit"]),
                "kappa_R": kappa_R,
                "D_ratio_to_kappa": float(beta[1]) / kappa_R,
                "dispersion_r2": _r_squared(y, predicted),
                "n_modes_used": len(eligible),
                "qR_max_used": float(eligible["qR"].max()),
            }
        )
    columns = [
        "delta",
        "Gamma0_micro",
        "Gamma0_micro_se",
        "D_micro",
        "D_micro_se",
        "D_micro_ci_low",
        "D_micro_ci_high",
        "Gamma0_closure",
        "D_closure",
        "kappa_R",
        "D_ratio_to_kappa",
        "dispersion_r2",
        "n_modes_used",
        "qR_max_used",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("delta", ascending=False)
        .reset_index(drop=True)
    )


def build_epsilon_convergence(mode_results: pd.DataFrame) -> pd.DataFrame:
    if mode_results["epsilon_fraction"].nunique() < 2:
        return pd.DataFrame(
            columns=[
                "delta",
                "mode_index",
                "epsilon_fraction",
                "epsilon_achieved",
                "Gamma_micro",
                "Gamma_micro_se",
                "Gamma_relative_change_from_previous",
                "escape_fraction",
                "reliable",
            ]
        )
    frames = []
    for (_, _), group in mode_results.groupby(["delta", "mode_index"]):
        group = group.sort_values("epsilon_fraction").copy()
        group["Gamma_relative_change_from_previous"] = (
            group["Gamma_micro"].pct_change().abs()
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True)[
        [
            "delta",
            "mode_index",
            "epsilon_fraction",
            "epsilon_achieved",
            "Gamma_micro",
            "Gamma_micro_se",
            "Gamma_relative_change_from_previous",
            "escape_fraction",
            "reliable",
        ]
    ]


def build_M_convergence(
    tasks: Sequence[Phase5Task],
    blocks_by_task: dict[str, list[Phase5BlockResult]],
    candidates: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        ordered = sorted(blocks_by_task[task.task_id], key=lambda block: block.block_id)
        for target_M in sorted(set(int(value) for value in candidates)):
            selected = []
            count = 0
            for block in ordered:
                if count >= target_M:
                    break
                selected.append(block)
                count += block.block_n
            if count < target_M or not selected:
                continue
            aggregate = aggregate_block_series(selected)
            fit = fit_microscopic_relaxation(
                np.asarray(aggregate["A_q"]), task.fit_start, task.fit_end
            )
            rows.append(
                {
                    "task_id": task.task_id,
                    "delta": task.delta,
                    "mode_index": task.mode_index,
                    "epsilon_fraction": task.epsilon_fraction,
                    "M_requested": target_M,
                    "M_used": int(aggregate["M_total"]),
                    "n_blocks": len(selected),
                    "Gamma_micro": fit["Gamma_micro"],
                    "fit_r2": fit["fit_r2"],
                }
            )
    return pd.DataFrame(rows, columns=M_CONVERGENCE_COLUMNS)


def build_scaling_summary(
    mode_results: pd.DataFrame,
    dispersion: pd.DataFrame,
    primary_epsilon_fraction: float,
) -> pd.DataFrame:
    primary = mode_results[
        np.isclose(mode_results["epsilon_fraction"], primary_epsilon_fraction)
        & (mode_results["mode_index"] == 0)
        & mode_results["reliable"]
    ].sort_values("delta")
    if len(primary) < 2:
        return pd.DataFrame(columns=SCALING_SUMMARY_COLUMNS)
    fit = fit_power_law(primary["delta"], primary["Gamma_micro"])
    rows = [
        {
            "quantity": "Gamma0_micro_exploratory_Gaussian_centered",
            "amplitude": fit["amplitude"],
            "exponent": fit["exponent"],
            "exponent_regression_se": fit["exponent_regression_se"],
            "r2": fit["r2"],
            "n_points": fit["n_points"],
            "interpretation": "exploratory; Gaussian-centered delta is fixed",
        }
    ]
    if not dispersion.empty and np.all(
        (dispersion["D_micro"] > 0.0) & (dispersion["Gamma0_micro"] > 0.0)
    ):
        xi = np.sqrt(dispersion["D_micro"] / dispersion["Gamma0_micro"])
        xi_fit = fit_power_law(dispersion["delta"], xi)
        rows.append(
            {
                "quantity": "xi_micro_derived",
                "amplitude": xi_fit["amplitude"],
                "exponent": xi_fit["exponent"],
                "exponent_regression_se": xi_fit["exponent_regression_se"],
                "r2": xi_fit["r2"],
                "n_points": xi_fit["n_points"],
                "interpretation": "derived from D_micro/Gamma0_micro; not real-space xi",
            }
        )
    return pd.DataFrame(rows, columns=SCALING_SUMMARY_COLUMNS)


def load_all_checkpoints(
    work_units: Sequence[Phase5WorkUnit], blocks_dir: Path
) -> dict[str, list[Phase5BlockResult]]:
    grouped: dict[str, list[Phase5BlockResult]] = {}
    for unit in sorted(work_units, key=lambda item: item.unit_id):
        path = Path(blocks_dir) / f"{unit.unit_id}.npz"
        block = load_block_checkpoint(path, unit)
        grouped.setdefault(unit.task.task_id, []).append(block)
    return grouped


def run_phase5_analysis(
    work_units: Sequence[Phase5WorkUnit],
    *,
    blocks_dir: Path,
    closure_dispersion: pd.DataFrame,
    primary_epsilon_fraction: float,
    qR_max: float,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    M_candidates: Sequence[int],
    kappa_R: float,
    performance: dict[str, Any],
    reproducibility: dict[str, Any],
) -> Phase5Analysis:
    tasks_by_id = {unit.task.task_id: unit.task for unit in work_units}
    tasks = list(tasks_by_id.values())
    blocks = load_all_checkpoints(work_units, blocks_dir)
    mode, windows, _ = analyze_mode_tasks(
        tasks,
        blocks,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    dispersion = fit_microscopic_dispersion(
        mode,
        tasks_by_id,
        blocks,
        primary_epsilon_fraction=primary_epsilon_fraction,
        qR_max=qR_max,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        closure_dispersion=closure_dispersion,
        kappa_R=kappa_R,
    )
    epsilon = build_epsilon_convergence(mode)
    M_table = build_M_convergence(tasks, blocks, M_candidates)
    scaling = build_scaling_summary(mode, dispersion, primary_epsilon_fraction)
    primary = mode[np.isclose(mode["epsilon_fraction"], primary_epsilon_fraction)]
    reliable_primary = primary[primary["reliable"]]
    q0 = reliable_primary[reliable_primary["mode_index"] == 0]
    kernel_deviations = []
    for delta, group in reliable_primary.groupby("delta"):
        q0_group = group[group["mode_index"] == 0]
        if q0_group.empty:
            continue
        gamma0 = float(q0_group["Gamma_micro"].iloc[0])
        config = Config(
            N=int(group["N"].iloc[0]),
            R=int(group["R"].iloc[0]),
        )
        for _, row in group.iterrows():
            exact = -math.log(abs(kernel_hat(int(row["mode_index"]), config)))
            kernel_deviations.append(
                abs((float(row["Gamma_micro"]) - gamma0) - exact)
            )

    q0_deviation = (
        float(np.max(np.abs(q0["Gamma_ratio"] - 1.0))) if not q0.empty else None
    )
    D_deviation = (
        float(np.max(np.abs(dispersion["D_micro"] / dispersion["D_closure"] - 1.0)))
        if not dispersion.empty
        else None
    )
    warnings = []
    if epsilon.empty:
        warnings.append("epsilon convergence scan has not been performed")
    if M_table.empty or M_table["M_used"].nunique() < 2:
        warnings.append("M convergence scan has not been performed")
    if float(primary["escape_fraction"].max()) > 0.1:
        warnings.append(
            "substantial metastable escape/rounding observed; do not claim a true microscopic spinodal"
        )
    if q0.empty:
        warnings.append("no reliable microscopic q=0 estimate is available")
    finite_preparation_drift = primary["preparation_drift"].to_numpy(dtype=float)
    finite_preparation_drift = finite_preparation_drift[
        np.isfinite(finite_preparation_drift)
    ]
    summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": (
            "existing annealed-Gaussian-J, quenched-threshold microscopic binary dynamics"
        ),
        "microscopic_kernel": sorted(set(primary["microscopic_kernel"])),
        "initialization": sorted(set(primary["initialization_mode"])),
        "performance": performance,
        "statistics": {
            "primary_epsilon_fraction": float(primary_epsilon_fraction),
            "M_total": sorted(set(int(value) for value in primary["M_total"])),
            "n_blocks": int(primary["n_blocks"].sum()),
            "n_blocks_primary": int(primary["n_blocks"].sum()),
            "n_blocks_all_epsilon": int(mode["n_blocks"].sum()),
            "bootstrap_replicates": int(bootstrap_replicates),
            "bootstrap_seed": int(bootstrap_seed),
        },
        "convergence": {
            "epsilon_scan_performed": not epsilon.empty,
            "M_scan_performed": (
                not M_table.empty and M_table["M_used"].nunique() >= 2
            ),
        },
        "metastability": {
            "max_baseline_drift": float(primary["baseline_drift"].max()),
            "max_escape_fraction": float(primary["escape_fraction"].max()),
            "max_absolute_preparation_drift": (
                float(np.max(np.abs(finite_preparation_drift)))
                if finite_preparation_drift.size
                else None
            ),
            "wording": (
                "Gaussian-centered comparison; any shift is an effective "
                "microscopic transition or pseudospinodal-like crossover"
            ),
        },
        "comparison_to_closure": {
            "max_q0_relative_deviation": q0_deviation,
            "max_D_relative_deviation": D_deviation,
            "max_kernel_relation_absolute_deviation": (
                float(max(kernel_deviations)) if kernel_deviations else None
            ),
            "exact_kernel_relation_assumed_for_microscopic": False,
            "q0_relative_deviation": q0_deviation,
            "D_relative_deviation": D_deviation,
            "kernel_relation_deviation": (
                float(max(kernel_deviations)) if kernel_deviations else None
            ),
        },
        "reproducibility": reproducibility,
        "warnings": warnings,
    }
    return Phase5Analysis(
        mode_results=mode.sort_values(
            ["delta", "mode_index", "epsilon_fraction"],
            ascending=[False, True, True],
        ),
        dispersion_fits=dispersion,
        scaling_summary=scaling,
        epsilon_convergence=epsilon,
        M_convergence=M_table,
        fit_window_diagnostics=windows,
        validation_summary=summary,
    )


def _make_figures(analysis: Phase5Analysis, output_dir: Path) -> dict[str, Path]:
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    mode = analysis.mode_results
    primary_fraction = float(
        analysis.validation_summary["statistics"]["primary_epsilon_fraction"]
    )
    primary = mode[np.isclose(mode["epsilon_fraction"], primary_fraction)]
    paths: dict[str, Path] = {}

    path = figures / "phase5_q0_gamma_vs_delta.png"
    q0 = primary[
        (primary["mode_index"] == 0)
        & primary["reliable"]
        & np.isfinite(primary["Gamma_micro"])
        & (primary["Gamma_micro"] > 0.0)
    ].sort_values("delta")
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    if not q0.empty:
        ax.errorbar(q0["delta"], q0["Gamma_micro"], yerr=q0["Gamma_micro_se"], fmt="o-", capsize=3, label="microscopic")
        ax.loglog(q0["delta"], q0["Gamma_closure"], "s--", label="closure")
        ax.set(xscale="log", yscale="log")
    else:
        ax.text(0.5, 0.5, "No reliable positive q=0 estimate", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="Gaussian-centered delta", ylabel="Gamma0", title="Phase5: q=0 microscopic robustness test")
    ax.grid(True, which="both", alpha=0.25)
    if not q0.empty:
        ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths[path.stem] = path

    path = figures / "phase5_gamma_vs_q2.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for delta, group in primary.groupby("delta"):
        ax.errorbar(group["q"] ** 2, group["Gamma_micro"], yerr=group["Gamma_micro_se"], fmt="o", label=f"micro {delta:g}")
        ax.plot(group["q"] ** 2, group["Gamma_closure"], "--", alpha=0.7)
    ax.set(xlabel="q^2", ylabel="Gamma(q)", title="Phase5: microscopic dispersion (dashed: closure)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths[path.stem] = path

    path = figures / "phase5_gamma_ratio_to_closure.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for mode_index, group in primary.groupby("mode_index"):
        ax.semilogx(group["delta"], group["Gamma_ratio"], "o-", label=f"mode {mode_index}")
    ax.axhline(1.0, color="k", linestyle="--")
    ax.set(xlabel="delta", ylabel="Gamma_micro/Gamma_closure", title="Phase5: closure deviation")
    ax.grid(True, which="both", alpha=0.25)
    if not primary.empty:
        ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths[path.stem] = path

    path = figures / "phase5_D_vs_delta.png"
    dispersion = analysis.dispersion_fits.sort_values("delta")
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    if not dispersion.empty:
        ax.errorbar(dispersion["delta"], dispersion["D_micro"], yerr=dispersion["D_micro_se"], fmt="o-", capsize=3, label="microscopic")
        ax.semilogx(dispersion["delta"], dispersion["D_closure"], "s--", label="closure")
        ax.axhline(float(dispersion["kappa_R"].iloc[0]), color="k", linestyle=":", label="kappa_R")
    ax.set(xlabel="delta", ylabel="D", title="Phase5: long-wave coefficient")
    ax.grid(True, which="both", alpha=0.25)
    if not dispersion.empty:
        ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths[path.stem] = path

    path = figures / "phase5_kernel_relation_deviation.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for delta, group in primary.groupby("delta"):
        q0_group = group[group["mode_index"] == 0]
        if q0_group.empty:
            continue
        gamma0 = float(q0_group["Gamma_micro"].iloc[0])
        config = Config(N=int(group["N"].iloc[0]), R=int(group["R"].iloc[0]))
        deviation = [
            (float(row["Gamma_micro"]) - gamma0)
            + math.log(abs(kernel_hat(int(row["mode_index"]), config)))
            for _, row in group.iterrows()
        ]
        ax.plot(group["qR"], deviation, "o-", label=f"{delta:g}")
    ax.axhline(0.0, color="k", linestyle="--")
    ax.set(xlabel="qR", ylabel="microscopic minus closure-kernel relation", title="Phase5: exact-kernel relation is tested, not assumed")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths[path.stem] = path

    path = figures / "phase5_metastable_survival.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    q0 = primary[primary["mode_index"] == 0]
    ax.semilogx(q0["delta"], 1.0 - q0["escape_fraction"], "o-")
    ax.set(xlabel="Gaussian-centered delta", ylabel="minimum survival fraction", title="Phase5: metastable survival diagnostic")
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths[path.stem] = path

    path = figures / "phase5_epsilon_convergence.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for (delta, mode_index), group in analysis.epsilon_convergence.groupby(["delta", "mode_index"]):
        ax.errorbar(group["epsilon_fraction"], group["Gamma_micro"], yerr=group["Gamma_micro_se"], fmt="o-", label=f"d={delta:g},m={mode_index}")
    ax.set(xlabel="epsilon fraction", ylabel="Gamma_micro", title="Phase5 pilot: epsilon convergence")
    ax.grid(True, alpha=0.25)
    if not analysis.epsilon_convergence.empty:
        ax.legend(fontsize=6, ncol=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths[path.stem] = path

    path = figures / "phase5_M_convergence.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    for task_id, group in analysis.M_convergence.groupby("task_id"):
        ax.plot(group["M_used"], group["Gamma_micro"], "o-", label=task_id)
    ax.set(xlabel="cumulative M", ylabel="Gamma_micro", title="Phase5 pilot: ensemble-size convergence")
    ax.grid(True, alpha=0.25)
    if not analysis.M_convergence.empty and analysis.M_convergence["task_id"].nunique() <= 12:
        ax.legend(fontsize=6)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths[path.stem] = path
    return paths


def write_phase5_analysis(
    analysis: Phase5Analysis, output_dir: Path
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "mode_results": output_dir / "phase5_mode_results.csv",
        "dispersion_fits": output_dir / "phase5_dispersion_fits.csv",
        "scaling_summary": output_dir / "phase5_scaling_summary.csv",
        "epsilon_convergence": output_dir / "phase5_epsilon_convergence.csv",
        "M_convergence": output_dir / "phase5_M_convergence.csv",
        "fit_window_diagnostics": output_dir / "phase5_fit_window_diagnostics.csv",
        "validation_summary": output_dir / "phase5_validation_summary.json",
    }
    analysis.mode_results.to_csv(paths["mode_results"], index=False)
    analysis.dispersion_fits.to_csv(paths["dispersion_fits"], index=False)
    analysis.scaling_summary.to_csv(paths["scaling_summary"], index=False)
    analysis.epsilon_convergence.to_csv(paths["epsilon_convergence"], index=False)
    analysis.M_convergence.to_csv(paths["M_convergence"], index=False)
    analysis.fit_window_diagnostics.to_csv(
        paths["fit_window_diagnostics"], index=False
    )
    paths["validation_summary"].write_text(
        json.dumps(analysis.validation_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    paths.update(_make_figures(analysis, output_dir))
    return paths
