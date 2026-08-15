#!/usr/bin/env python3
"""Numerical core and analysis for spinodal Phase1-2.

Phase1 measures the q=0 relaxation rate and Phase2 measures finite-q
relaxation in the deterministic continuous-valued Gaussian closure map.  This
module contains no MPI state; each Phase12Task is independently executable.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import ndtr

from spatial_mode_ensemble_validation import Config, kernel_hat, q_from_mode
from spinodal_phase0 import (
    DEFAULT_DELTAS,
    Phase0Task,
    run_phase0_case,
    write_phase0_outputs,
)


SCRIPT_VERSION = "2026.08.15-phase12-v1"
DEFAULT_MODES = (0, 1, 2, 3, 4, 5, 6)


@dataclass(frozen=True)
class Phase0Reference:
    phase0_dir: Path
    summary: dict[str, Any]
    delta_table: pd.DataFrame
    regenerated: bool


@dataclass(frozen=True)
class Phase12Task:
    task_index: int
    task_group: str
    delta: float
    Delta: float
    m_star: float
    m_spinodal: float
    Lambda_star: float
    Gamma0_theory: float
    tau0_theory: float
    kappa_R_theory: float
    xi_theory: float
    sigma_eff: float
    mu: float
    N: int
    R: int
    lattice_spacing: float
    mode_index: int
    epsilon_fraction: float
    T: int
    fit_start: int
    fit_end: int
    qR_max_fit: float
    debug_profiles: bool = False


@dataclass
class Phase12ModeResult:
    task: Phase12Task
    metrics: dict[str, Any]
    t: np.ndarray
    amplitude: np.ndarray
    theory_amplitude: np.ndarray
    profile_history_plus: np.ndarray | None = None
    profile_history_minus: np.ndarray | None = None


def parse_number_list(text: str, value_type: type = float) -> tuple[Any, ...]:
    values = tuple(value_type(token.strip()) for token in text.split(",") if token.strip())
    if not values:
        raise ValueError("list must contain at least one value")
    return values


def _phase0_task_from_summary(
    summary: dict[str, Any],
    delta_list: Sequence[float],
) -> Phase0Task:
    inputs = summary.get("inputs", {})
    return Phase0Task(
        B=float(inputs.get("B", 2.0)),
        R=int(inputs.get("R", 12)),
        sigma_J=float(inputs.get("sigma_J", 1.0)),
        sigma_phi=float(inputs.get("sigma_phi", 0.06)),
        phi_bar=float(inputs.get("phi_bar", 0.0)),
        lattice_spacing=float(inputs.get("a", inputs.get("lattice_spacing", 1.0))),
        branch=str(inputs.get("branch", "stay_to_evacuate")),
        delta_list=tuple(float(value) for value in delta_list),
    )


def ensure_phase0_reference(
    phase0_dir: Path,
    required_deltas: Sequence[float] = DEFAULT_DELTAS,
    fallback_task: Phase0Task | None = None,
) -> Phase0Reference:
    """Load high-precision Phase0 outputs, regenerating them when absent."""
    summary_path = phase0_dir / "phase0_summary.json"
    table_path = phase0_dir / "phase0_delta_table.csv"
    regenerated = False

    if not summary_path.exists() or not table_path.exists():
        task = fallback_task or Phase0Task(delta_list=tuple(required_deltas))
        phase0_result = run_phase0_case(task)
        write_phase0_outputs(phase0_result, phase0_dir)
        regenerated = True

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    table = pd.read_csv(table_path)
    required_summary = {
        "sigma_eff",
        "mu",
        "m_spinodal",
        "Delta_spinodal",
        "kappa_R_theory",
        "inputs",
    }
    missing_summary = required_summary - set(summary)
    if missing_summary:
        raise ValueError(f"Phase0 summary is missing keys: {sorted(missing_summary)}")
    required_columns = {
        "delta",
        "Delta",
        "m_star",
        "Lambda_star",
        "Gamma0_theory",
        "tau0_theory",
        "kappa_R_theory",
        "xi_theory",
    }
    missing_columns = required_columns - set(table.columns)
    if missing_columns:
        raise ValueError(f"Phase0 delta table is missing columns: {sorted(missing_columns)}")

    missing_deltas = [
        float(delta)
        for delta in required_deltas
        if not np.isclose(table["delta"].to_numpy(dtype=float), delta, rtol=1e-12, atol=1e-15).any()
    ]
    if missing_deltas:
        extra_task = _phase0_task_from_summary(summary, missing_deltas)
        extra = run_phase0_case(extra_task).delta_table
        table = pd.concat([table, extra], ignore_index=True)
        table = table.drop_duplicates(subset=["delta"], keep="first")

    return Phase0Reference(
        phase0_dir=phase0_dir,
        summary=summary,
        delta_table=table.sort_values("delta", ascending=False).reset_index(drop=True),
        regenerated=regenerated,
    )


def select_phase0_row(reference: Phase0Reference, delta: float) -> pd.Series:
    values = reference.delta_table["delta"].to_numpy(dtype=float)
    matches = np.flatnonzero(np.isclose(values, delta, rtol=1e-12, atol=1e-15))
    if len(matches) != 1:
        raise ValueError(f"Expected one Phase0 row for delta={delta:g}, found {len(matches)}")
    return reference.delta_table.iloc[int(matches[0])]


def adaptive_time_length(
    tau0_theory: float,
    T_min: int = 50,
    tau_multiplier: float = 6.0,
    T_fixed: int | None = None,
) -> int:
    if T_fixed is not None:
        if T_fixed < 2:
            raise ValueError("T_fixed must be >= 2")
        return int(T_fixed)
    if T_min < 2 or tau_multiplier <= 0.0:
        raise ValueError("T_min must be >= 2 and tau_multiplier must be positive")
    return max(int(T_min), int(math.ceil(tau_multiplier * tau0_theory)))


def adaptive_fit_end(
    tau0_theory: float,
    T: int,
    fit_start: int = 0,
    fit_end_fixed: int | None = None,
) -> int:
    if fit_start < 0 or fit_start >= T:
        raise ValueError("fit_start must satisfy 0 <= fit_start < T")
    suggested = min(50, max(10, int(math.ceil(0.5 * tau0_theory))))
    fit_end = min(T, fit_end_fixed if fit_end_fixed is not None else suggested)
    if fit_end <= fit_start:
        raise ValueError("fit_end must be greater than fit_start")
    return int(fit_end)


def build_phase12_tasks(
    reference: Phase0Reference,
    deltas: Sequence[float],
    modes: Sequence[int],
    N: int = 1024,
    epsilon_fraction: float = 0.05,
    tau_multiplier: float = 6.0,
    T_min: int = 50,
    T_fixed: int | None = None,
    fit_start: int = 0,
    fit_end_fixed: int | None = None,
    qR_max_fit: float = 0.35,
    task_group: str = "main",
    start_index: int = 0,
    debug_profiles: bool = False,
) -> list[Phase12Task]:
    inputs = reference.summary["inputs"]
    R = int(inputs["R"])
    lattice_spacing = float(inputs.get("a", inputs.get("lattice_spacing", 1.0)))
    if N <= 2 * R:
        raise ValueError("N must be greater than 2R")
    if epsilon_fraction <= 0.0:
        raise ValueError("epsilon_fraction must be positive")
    if qR_max_fit <= 0.0:
        raise ValueError("qR_max_fit must be positive")

    modes_unique = tuple(sorted(set(int(mode) for mode in modes)))
    if not modes_unique or modes_unique[0] < 0 or modes_unique[-1] > N // 2:
        raise ValueError("modes must lie in [0, N/2]")

    tasks: list[Phase12Task] = []
    task_index = int(start_index)
    for delta in deltas:
        row = select_phase0_row(reference, float(delta))
        T = adaptive_time_length(
            float(row["tau0_theory"]),
            T_min=T_min,
            tau_multiplier=tau_multiplier,
            T_fixed=T_fixed,
        )
        fit_end = adaptive_fit_end(
            float(row["tau0_theory"]),
            T=T,
            fit_start=fit_start,
            fit_end_fixed=fit_end_fixed,
        )
        for mode in modes_unique:
            tasks.append(
                Phase12Task(
                    task_index=task_index,
                    task_group=task_group,
                    delta=float(delta),
                    Delta=float(row["Delta"]),
                    m_star=float(row["m_star"]),
                    m_spinodal=float(reference.summary["m_spinodal"]),
                    Lambda_star=float(row["Lambda_star"]),
                    Gamma0_theory=float(row["Gamma0_theory"]),
                    tau0_theory=float(row["tau0_theory"]),
                    kappa_R_theory=float(row["kappa_R_theory"]),
                    xi_theory=float(row["xi_theory"]),
                    sigma_eff=float(reference.summary["sigma_eff"]),
                    mu=float(reference.summary["mu"]),
                    N=int(N),
                    R=R,
                    lattice_spacing=lattice_spacing,
                    mode_index=mode,
                    epsilon_fraction=float(epsilon_fraction),
                    T=T,
                    fit_start=int(fit_start),
                    fit_end=fit_end,
                    qR_max_fit=float(qR_max_fit),
                    debug_profiles=bool(debug_profiles),
                )
            )
            task_index += 1
    return tasks


def periodic_local_average(values: np.ndarray, R: int) -> np.ndarray:
    """Average the 2R periodic neighbours, excluding the central site."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")
    N = len(values)
    if R < 1 or 2 * R >= N:
        raise ValueError("periodic_local_average requires 1 <= R < N/2")
    extended = np.concatenate((values[-R:], values, values[:R]))
    cumulative = np.concatenate(([0.0], np.cumsum(extended, dtype=float)))
    window_sum = cumulative[2 * R + 1 : 2 * R + 1 + N] - cumulative[:N]
    return (window_sum - values) / (2.0 * R)


def deterministic_closure_step(
    values: np.ndarray,
    R: int,
    mu: float,
    Delta: float,
    sigma_eff: float,
) -> np.ndarray:
    """Apply one synchronous continuous-valued Gaussian closure update."""
    if sigma_eff <= 0.0:
        raise ValueError("sigma_eff must be positive")
    local_mean = periodic_local_average(values, R)
    updated = 2.0 * ndtr((mu * local_mean + Delta) / sigma_eff) - 1.0
    return np.asarray(updated, dtype=float)


def cosine_mode_amplitude(response: np.ndarray, mode_index: int, a: float) -> float:
    N = len(response)
    if mode_index < 0 or mode_index > N // 2:
        raise ValueError("mode_index must lie in [0, N/2]")
    x = np.arange(N, dtype=float) * a
    q = 2.0 * math.pi * mode_index / (N * a)
    factor = 1.0 if mode_index == 0 or (N % 2 == 0 and mode_index == N // 2) else 2.0
    return float(factor * np.dot(response, np.cos(q * x)) / N)


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float:
    residual_sum = float(np.sum((observed - predicted) ** 2))
    centered_sum = float(np.sum((observed - np.mean(observed)) ** 2))
    if centered_sum <= np.finfo(float).tiny:
        return math.nan
    return 1.0 - residual_sum / centered_sum


def fit_lambda_origin(
    amplitude: np.ndarray,
    fit_start: int,
    fit_end: int,
) -> tuple[float, float, int]:
    """Fit A(t+1)=lambda A(t) through the origin on [start, end]."""
    x = np.asarray(amplitude[fit_start:fit_end], dtype=float)
    y = np.asarray(amplitude[fit_start + 1 : fit_end + 1], dtype=float)
    denominator = float(np.dot(x, x))
    if denominator <= np.finfo(float).tiny:
        return math.nan, math.nan, len(x)
    fitted = float(np.dot(x, y) / denominator)
    return fitted, _r_squared(y, fitted * x), len(x)


def fit_log_amplitude(
    amplitude: np.ndarray,
    fit_start: int,
    fit_end: int,
) -> tuple[float, float, float, int]:
    """Fit log|A(t)/A(0)| = intercept - Gamma*t."""
    times = np.arange(fit_start, fit_end + 1, dtype=float)
    values = np.asarray(amplitude[fit_start : fit_end + 1], dtype=float)
    scale = abs(float(amplitude[0]))
    floor = max(np.finfo(float).eps * max(scale, 1.0) * 16.0, 1e-300)
    valid = np.isfinite(values) & (np.abs(values) > floor) & (scale > 0.0)
    if int(np.sum(valid)) < 2:
        return math.nan, math.nan, math.nan, int(np.sum(valid))
    x = times[valid]
    y = np.log(np.abs(values[valid]) / scale)
    design = np.column_stack((np.ones_like(x), x))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ beta
    Gamma = -float(beta[1])
    return Gamma, _r_squared(y, predicted), float(beta[0]), len(x)


def simulate_deterministic_mode(task: Phase12Task) -> Phase12ModeResult:
    """Run one independent (delta, q, epsilon_fraction) closure task."""
    config = Config(
        N=task.N,
        R=task.R,
        lattice_spacing=task.lattice_spacing,
    )
    q = q_from_mode(task.mode_index, config)
    khat = kernel_hat(task.mode_index, config)
    qR = q * task.R * task.lattice_spacing
    lambda_theory = task.Lambda_star * khat
    Gamma_theory = -math.log(abs(lambda_theory)) if lambda_theory != 0.0 else math.inf

    distance_to_spinodal = abs(task.m_star - task.m_spinodal)
    epsilon = task.epsilon_fraction * distance_to_spinodal
    x = np.arange(task.N, dtype=float) * task.lattice_spacing
    perturbation = epsilon * np.cos(q * x)
    u_plus = task.m_star + perturbation
    u_minus = task.m_star - perturbation
    if np.min(u_plus) < -1.0 or np.max(u_plus) > 1.0 or np.min(u_minus) < -1.0 or np.max(u_minus) > 1.0:
        raise ValueError(
            f"Initial perturbation leaves [-1,1] for delta={task.delta:g}, "
            f"mode={task.mode_index}, epsilon_fraction={task.epsilon_fraction:g}."
        )

    amplitude = np.empty(task.T + 1, dtype=float)
    amplitude[0] = cosine_mode_amplitude(0.5 * (u_plus - u_minus), task.mode_index, task.lattice_spacing)
    plus_history = [u_plus.copy()] if task.debug_profiles else None
    minus_history = [u_minus.copy()] if task.debug_profiles else None

    for time_index in range(task.T):
        u_plus = deterministic_closure_step(
            u_plus, task.R, task.mu, task.Delta, task.sigma_eff
        )
        u_minus = deterministic_closure_step(
            u_minus, task.R, task.mu, task.Delta, task.sigma_eff
        )
        response = 0.5 * (u_plus - u_minus)
        amplitude[time_index + 1] = cosine_mode_amplitude(
            response, task.mode_index, task.lattice_spacing
        )
        if task.debug_profiles:
            assert plus_history is not None and minus_history is not None
            plus_history.append(u_plus.copy())
            minus_history.append(u_minus.copy())

    lambda_1step = float(amplitude[1] / amplitude[0])
    lambda_fit, lambda_fit_r2, fit_n_pairs = fit_lambda_origin(
        amplitude, task.fit_start, task.fit_end
    )
    Gamma_1step = -math.log(abs(lambda_1step))
    Gamma_from_lambda = -math.log(abs(lambda_fit))
    Gamma_logfit, Gamma_logfit_r2, logfit_intercept, logfit_n_points = fit_log_amplitude(
        amplitude, task.fit_start, task.fit_end
    )

    method_A_B_relative_difference = abs(lambda_1step - lambda_fit) / max(
        abs(lambda_theory), 1e-15
    )
    method_B_C_relative_difference = abs(Gamma_from_lambda - Gamma_logfit) / max(
        abs(Gamma_theory), 1e-15
    )
    fit_amplitudes = np.abs(amplitude[task.fit_start : task.fit_end + 1])
    amplitude_floor = max(abs(amplitude[0]) * 1e-10, 1e-14)
    minimum_fit_amplitude = float(np.min(fit_amplitudes))
    basin_margin = distance_to_spinodal - epsilon

    reliability_reasons: list[str] = []
    if not math.isfinite(lambda_fit) or not math.isfinite(Gamma_logfit):
        reliability_reasons.append("non_finite_fit")
    if math.isfinite(lambda_fit_r2) and lambda_fit_r2 < 0.999:
        reliability_reasons.append("lambda_fit_r2_below_0.999")
    if math.isfinite(Gamma_logfit_r2) and Gamma_logfit_r2 < 0.999:
        reliability_reasons.append("logfit_r2_below_0.999")
    if method_A_B_relative_difference > 0.02:
        reliability_reasons.append("method_A_B_difference_above_2pct")
    if method_B_C_relative_difference > 0.02:
        reliability_reasons.append("method_B_C_difference_above_2pct")
    if minimum_fit_amplitude <= amplitude_floor:
        reliability_reasons.append("amplitude_near_machine_precision")
    if epsilon >= 0.25 * distance_to_spinodal:
        reliability_reasons.append("initial_perturbation_near_basin_boundary")

    theory_amplitude = amplitude[0] * np.power(
        lambda_theory, np.arange(task.T + 1, dtype=int)
    )
    q_min = 2.0 * math.pi / (task.N * task.lattice_spacing)
    metrics: dict[str, Any] = {
        "task_index": task.task_index,
        "task_group": task.task_group,
        "delta": task.delta,
        "Delta": task.Delta,
        "m_star": task.m_star,
        "m_spinodal": task.m_spinodal,
        "distance_to_spinodal": distance_to_spinodal,
        "epsilon": epsilon,
        "epsilon_fraction": task.epsilon_fraction,
        "basin_margin": basin_margin,
        "N": task.N,
        "R": task.R,
        "a": task.lattice_spacing,
        "T": task.T,
        "mode_index": task.mode_index,
        "q": q,
        "q_over_pi": q / math.pi,
        "qR": qR,
        "kernel_hat": khat,
        "Lambda_star": task.Lambda_star,
        "lambda_theory": lambda_theory,
        "lambda_1step": lambda_1step,
        "lambda_fit": lambda_fit,
        "lambda_fit_r2": lambda_fit_r2,
        "Gamma_theory": Gamma_theory,
        "Gamma_1step": Gamma_1step,
        "Gamma_from_lambda": Gamma_from_lambda,
        "Gamma_logfit": Gamma_logfit,
        "Gamma_logfit_r2": Gamma_logfit_r2,
        "Gamma_logfit_intercept": logfit_intercept,
        "Gamma0_theory": task.Gamma0_theory,
        "tau0_theory": task.tau0_theory,
        "kappa_R_theory": task.kappa_R_theory,
        "xi_theory": task.xi_theory,
        "N_over_xi_theory": task.N / task.xi_theory,
        "qmin_xi_theory": q_min * task.xi_theory,
        "fit_start": task.fit_start,
        "fit_end": task.fit_end,
        "fit_n_pairs": fit_n_pairs,
        "logfit_n_points": logfit_n_points,
        "minimum_fit_amplitude": minimum_fit_amplitude,
        "amplitude_floor": amplitude_floor,
        "method_A_B_relative_difference": method_A_B_relative_difference,
        "method_B_C_relative_difference": method_B_C_relative_difference,
        "lambda_relative_error": abs(lambda_fit - lambda_theory) / max(abs(lambda_theory), 1e-15),
        "Gamma_relative_error": abs(Gamma_from_lambda - Gamma_theory) / max(abs(Gamma_theory), 1e-15),
        "longwave_eligible": bool(qR <= task.qR_max_fit and khat > 0.0),
        "reliable": not reliability_reasons,
        "reliability_reason": ";".join(reliability_reasons) if reliability_reasons else "ok",
    }
    return Phase12ModeResult(
        task=task,
        metrics=metrics,
        t=np.arange(task.T + 1, dtype=int),
        amplitude=amplitude,
        theory_amplitude=theory_amplitude,
        profile_history_plus=np.stack(plus_history) if plus_history is not None else None,
        profile_history_minus=np.stack(minus_history) if minus_history is not None else None,
    )


def mode_results_dataframe(results: Sequence[Phase12ModeResult]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame()
    return pd.DataFrame([result.metrics for result in results]).sort_values(
        ["task_group", "delta", "epsilon_fraction", "mode_index"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)


def fit_dispersion_relations(
    mode_df: pd.DataFrame,
    qR_max_fit: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for delta, group in mode_df.groupby("delta", sort=False):
        group = group.sort_values("mode_index")
        q0 = group[group["mode_index"] == 0]
        if q0.empty:
            raise ValueError(f"q=0 result is required for delta={delta:g}")
        eligible = group[
            (group["qR"] <= qR_max_fit)
            & (group["kernel_hat"] > 0.0)
            & np.isfinite(group["Gamma_from_lambda"])
        ].copy()
        x = eligible["q"].to_numpy(dtype=float) ** 2
        y = eligible["Gamma_from_lambda"].to_numpy(dtype=float)
        intercept = slope = slope_se = r2 = math.nan
        if len(eligible) >= 2 and np.ptp(x) > 0.0:
            design = np.column_stack((np.ones_like(x), x))
            beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
            intercept, slope = (float(beta[0]), float(beta[1]))
            predicted = design @ beta
            r2 = _r_squared(y, predicted)
            dof = len(y) - 2
            if dof > 0:
                residual_variance = float(np.sum((y - predicted) ** 2) / dof)
                covariance = residual_variance * np.linalg.inv(design.T @ design)
                slope_se = float(math.sqrt(max(covariance[1, 1], 0.0)))

        kappa = float(group["kappa_R_theory"].iloc[0])
        gamma0_q0 = float(q0["Gamma_from_lambda"].iloc[0])
        rows.append(
            {
                "delta": float(delta),
                "Delta": float(group["Delta"].iloc[0]),
                "Gamma0_theory": float(group["Gamma0_theory"].iloc[0]),
                "Gamma0_q0_sim": gamma0_q0,
                "Gamma0_dispersion_fit": intercept,
                "Gamma0_connectivity_absolute_error": abs(intercept - gamma0_q0),
                "D_fit": slope,
                "D_fit_se": slope_se,
                "kappa_R_theory": kappa,
                "D_absolute_error": abs(slope - kappa),
                "D_relative_error": abs(slope - kappa) / kappa,
                "dispersion_r2": r2,
                "n_modes_used": int(len(eligible)),
                "qR_max_fit": float(qR_max_fit),
                "qR_max_used": float(eligible["qR"].max()) if len(eligible) else math.nan,
                "mode_indices_used": ",".join(str(int(value)) for value in eligible["mode_index"]),
            }
        )
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def build_kernel_relation(mode_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for delta, group in mode_df.groupby("delta", sort=False):
        group = group.sort_values("mode_index")
        q0 = group[group["mode_index"] == 0]
        if q0.empty:
            raise ValueError(f"q=0 result is required for delta={delta:g}")
        gamma0 = float(q0["Gamma_from_lambda"].iloc[0])
        for _, row in group.iterrows():
            kernel_term = -math.log(abs(float(row["kernel_hat"])))
            simulated = float(row["Gamma_from_lambda"]) - gamma0
            absolute_error = abs(simulated - kernel_term)
            relative_error = absolute_error / abs(kernel_term) if abs(kernel_term) > 1e-15 else 0.0
            rows.append(
                {
                    "delta": float(delta),
                    "mode_index": int(row["mode_index"]),
                    "q": float(row["q"]),
                    "qR": float(row["qR"]),
                    "Gamma_minus_Gamma0_sim": simulated,
                    "minus_log_kernel": kernel_term,
                    "absolute_error": absolute_error,
                    "relative_error": relative_error,
                }
            )
    return pd.DataFrame(rows).sort_values(["delta", "mode_index"], ascending=[False, True])


def build_epsilon_convergence(scan_df: pd.DataFrame) -> pd.DataFrame:
    if scan_df.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for (_, _), group in scan_df.groupby(["delta", "mode_index"]):
        group = group.sort_values("epsilon_fraction", ascending=False).copy()
        next_gamma = group["Gamma_from_lambda"].shift(-1)
        group["Gamma_relative_change_to_next_smaller_epsilon"] = (
            (group["Gamma_from_lambda"] - next_gamma).abs()
            / next_gamma.abs().clip(lower=1e-15)
        )
        frames.append(group)
    return pd.concat(frames, ignore_index=True).sort_values(
        ["delta", "mode_index", "epsilon_fraction"], ascending=[False, True, False]
    )


def _delta_label(delta: float) -> str:
    return f"{delta:.0e}"


def _save_timeseries(results: Sequence[Phase12ModeResult], output_dir: Path) -> None:
    timeseries_dir = output_dir / "timeseries"
    timeseries_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        if result.task.task_group != "main":
            continue
        amplitude0 = float(result.amplitude[0])
        frame = pd.DataFrame(
            {
                "t": result.t,
                "A_q": result.amplitude,
                "A_q_over_A0": result.amplitude / amplitude0,
                "theory_A_q": result.theory_amplitude,
                "theory_A_q_over_A0": result.theory_amplitude / amplitude0,
            }
        )
        filename = f"delta_{_delta_label(result.task.delta)}_mode_{result.task.mode_index}.csv"
        frame.to_csv(timeseries_dir / filename, index=False)


def _save_debug_profiles(results: Sequence[Phase12ModeResult], output_dir: Path) -> None:
    profile_dir = output_dir / "debug_profiles"
    for result in results:
        if result.profile_history_plus is None or result.profile_history_minus is None:
            continue
        profile_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"delta_{_delta_label(result.task.delta)}_mode_{result.task.mode_index}_"
            f"eps_{result.task.epsilon_fraction:g}.npz"
        )
        np.savez_compressed(
            profile_dir / filename,
            u_plus=result.profile_history_plus,
            u_minus=result.profile_history_minus,
        )


def make_phase1_q0_relaxation_figure(
    results: Sequence[Phase12ModeResult],
    output_path: Path,
) -> None:
    q0_results = sorted(
        [r for r in results if r.task.task_group == "main" and r.task.mode_index == 0],
        key=lambda item: item.task.delta,
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(len(q0_results), 1)))
    for color, result in zip(colors, q0_results):
        normalized = result.amplitude / result.amplitude[0]
        theory = result.theory_amplitude / result.theory_amplitude[0]
        stride = max(1, len(result.t) // 45)
        ax.plot(result.t, theory, color=color, lw=1.5)
        ax.plot(
            result.t[::stride],
            normalized[::stride],
            "o",
            color=color,
            ms=3,
            label=rf"$\delta={result.task.delta:g}$",
        )
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$A_0(t)/A_0(0)$")
    ax.set_yscale("log")
    ax.set_title("Phase1: q=0 deterministic-closure relaxation (lines: Phase0 theory)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_phase1_gamma_comparison_figure(mode_df: pd.DataFrame, output_path: Path) -> None:
    q0 = mode_df[mode_df["mode_index"] == 0].sort_values("delta")
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.loglog(q0["delta"], q0["Gamma0_theory"], "k--", lw=2, label="Phase0 theory")
    ax.loglog(q0["delta"], q0["Gamma_from_lambda"], "o-", label="Phase1 deterministic closure")
    ax.set_xlabel(r"distance to spinodal $\delta$")
    ax.set_ylabel(r"$\Gamma_0$")
    ax.set_title("Phase1: q=0 theory vs numerical relaxation")
    ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_phase2_dispersion_figure(
    mode_df: pd.DataFrame,
    dispersion_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.5), constrained_layout=True)
    deltas = list(dict.fromkeys(mode_df["delta"].tolist()))
    colors = plt.cm.plasma(np.linspace(0.05, 0.9, max(len(deltas), 1)))
    for color, delta in zip(colors, deltas):
        group = mode_df[mode_df["delta"] == delta].sort_values("q")
        eligible = group["longwave_eligible"].astype(bool)
        ax.scatter(
            group.loc[eligible, "q"] ** 2,
            group.loc[eligible, "Gamma_from_lambda"],
            color=color,
            marker="o",
            s=35,
            label=rf"$\delta={delta:g}$",
        )
        if (~eligible).any():
            ax.scatter(
                group.loc[~eligible, "q"] ** 2,
                group.loc[~eligible, "Gamma_from_lambda"],
                facecolors="none",
                edgecolors=color,
                marker="o",
                s=35,
            )
        q2 = np.linspace(0.0, float((group["q"] ** 2).max()), 120)
        gamma0 = float(group["Gamma0_theory"].iloc[0])
        kappa = float(group["kappa_R_theory"].iloc[0])
        ax.plot(q2, gamma0 + kappa * q2, color=color, lw=1.1, alpha=0.8)
    ax.set_xlabel(r"$q^2$")
    ax.set_ylabel(r"$\Gamma(q)$")
    ax.set_title("Phase2: finite-q dispersion (lines: Phase0 long-wave theory)")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_exact_kernel_figure(kernel_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4), constrained_layout=True)
    for delta, group in kernel_df.groupby("delta", sort=False):
        ax.scatter(
            group["minus_log_kernel"],
            group["Gamma_minus_Gamma0_sim"],
            s=30,
            label=rf"$\delta={delta:g}$",
        )
    upper = max(
        float(kernel_df["minus_log_kernel"].max()),
        float(kernel_df["Gamma_minus_Gamma0_sim"].max()),
        1e-6,
    )
    ax.plot([0.0, upper], [0.0, upper], "k--", lw=1.8, label="exact theory y=x")
    ax.set_xlabel(r"$-\ln|\widehat K_R(q)|$")
    ax.set_ylabel(r"$\Gamma_{sim}(q)-\Gamma_{sim}(0)$")
    ax.set_title("Phase2: exact kernel relation")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_D_comparison_figure(dispersion_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    valid = np.isfinite(dispersion_df["D_fit"])
    data = dispersion_df[valid].sort_values("delta")
    if not data.empty:
        ax.errorbar(
            data["delta"],
            data["D_fit"],
            yerr=data["D_fit_se"],
            fmt="o-",
            capsize=3,
            label="Phase2 dispersion fit",
        )
        kappa = float(data["kappa_R_theory"].iloc[0])
        ax.axhline(kappa, color="k", linestyle="--", label=rf"theory $\kappa_R={kappa:.6g}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"distance to spinodal $\delta$")
    ax.set_ylabel(r"$D_{fit}$")
    ax.set_title("Phase2: long-wave coefficient (not a Phase4 xi measurement)")
    if not data.empty:
        ax.legend()
    ax.grid(True, which="both", alpha=0.25)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _finite_max(values: pd.Series) -> float | None:
    array = values.to_numpy(dtype=float)
    finite = array[np.isfinite(array)]
    return float(np.max(finite)) if len(finite) else None


def build_validation_summary(
    mode_df: pd.DataFrame,
    dispersion_df: pd.DataFrame,
    kernel_df: pd.DataFrame,
    epsilon_df: pd.DataFrame,
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    q0 = mode_df[mode_df["mode_index"] == 0].copy()
    q0["relative_error"] = (
        (q0["Gamma_from_lambda"] - q0["Gamma0_theory"]).abs()
        / q0["Gamma0_theory"].abs().clip(lower=1e-15)
    )
    kernel_nonzero = kernel_df[kernel_df["mode_index"] != 0]
    epsilon_change = (
        _finite_max(epsilon_df["Gamma_relative_change_to_next_smaller_epsilon"])
        if not epsilon_df.empty
        else None
    )
    max_q0_error = _finite_max(q0["relative_error"])
    max_kernel_error = _finite_max(kernel_nonzero["absolute_error"])
    max_D_error = _finite_max(dispersion_df["D_relative_error"])
    max_connectivity = _finite_max(dispersion_df["Gamma0_connectivity_absolute_error"])
    return {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calculation_type": "deterministic Gaussian closure Phase1-2; no critical-exponent fit",
        "runtime": runtime_metadata,
        "phase1": {
            "per_delta": q0[
                ["delta", "Gamma0_theory", "Gamma_from_lambda", "relative_error"]
            ].to_dict(orient="records"),
            "max_relative_error": max_q0_error,
        },
        "phase2_exact_kernel": {
            "max_absolute_error_excluding_q0": max_kernel_error,
            "max_relative_error_excluding_q0": _finite_max(kernel_nonzero["relative_error"]),
        },
        "phase2_longwave": {
            "per_delta": dispersion_df[
                ["delta", "D_fit", "kappa_R_theory", "D_relative_error", "n_modes_used"]
            ].to_dict(orient="records"),
            "max_D_relative_error": max_D_error,
        },
        "q0_dispersion_connectivity": {
            "max_absolute_error": max_connectivity,
        },
        "epsilon_convergence": {
            "performed": not epsilon_df.empty,
            "max_relative_change_on_halving": epsilon_change,
        },
        "soft_checks": {
            "phase1_q0_relative_error_below_1e-3": max_q0_error is not None and max_q0_error < 1e-3,
            "exact_kernel_absolute_error_below_1e-3": max_kernel_error is not None and max_kernel_error < 1e-3,
            "D_relative_error_below_10pct": max_D_error is not None and max_D_error < 0.10,
            "q0_intercept_absolute_error_below_1e-3": max_connectivity is not None and max_connectivity < 1e-3,
        },
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_phase12_outputs(
    results: Sequence[Phase12ModeResult],
    output_dir: Path,
    qR_max_fit: float,
    runtime_metadata: dict[str, Any],
    save_timeseries: bool = True,
) -> dict[str, Path]:
    """Aggregate compact task results and write Phase1-2 CSV/JSON/figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    all_df = mode_results_dataframe(results)
    main_df = all_df[all_df["task_group"] == "main"].copy()
    scan_df = all_df[all_df["task_group"] == "epsilon_scan"].copy()
    dispersion_df = fit_dispersion_relations(main_df, qR_max_fit=qR_max_fit)
    kernel_df = build_kernel_relation(main_df)
    epsilon_df = build_epsilon_convergence(scan_df)

    mode_path = output_dir / "phase12_mode_results.csv"
    dispersion_path = output_dir / "phase12_dispersion_fits.csv"
    kernel_path = output_dir / "phase12_kernel_relation.csv"
    epsilon_path = output_dir / "phase12_epsilon_scan.csv"
    summary_path = output_dir / "phase12_validation_summary.json"
    main_df.to_csv(mode_path, index=False)
    dispersion_df.to_csv(dispersion_path, index=False)
    kernel_df.to_csv(kernel_path, index=False)
    if not epsilon_df.empty:
        epsilon_df.to_csv(epsilon_path, index=False)

    if save_timeseries:
        _save_timeseries(results, output_dir)
    _save_debug_profiles(results, output_dir)

    figure_paths = {
        "phase1_q0_relaxation": output_dir / "phase1_q0_relaxation.png",
        "phase1_gamma0_comparison": output_dir / "phase1_gamma0_theory_vs_numeric.png",
        "phase2_dispersion": output_dir / "phase2_gamma_vs_q2.png",
        "phase2_exact_kernel": output_dir / "phase2_exact_kernel_relation.png",
        "phase2_D": output_dir / "phase2_D_vs_delta.png",
    }
    make_phase1_q0_relaxation_figure(results, figure_paths["phase1_q0_relaxation"])
    make_phase1_gamma_comparison_figure(main_df, figure_paths["phase1_gamma0_comparison"])
    make_phase2_dispersion_figure(main_df, dispersion_df, figure_paths["phase2_dispersion"])
    make_exact_kernel_figure(kernel_df, figure_paths["phase2_exact_kernel"])
    make_D_comparison_figure(dispersion_df, figure_paths["phase2_D"])

    summary = build_validation_summary(
        main_df, dispersion_df, kernel_df, epsilon_df, runtime_metadata
    )
    summary_path.write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths = {
        "mode_results": mode_path,
        "dispersion_fits": dispersion_path,
        "kernel_relation": kernel_path,
        "validation_summary": summary_path,
        **figure_paths,
    }
    if not epsilon_df.empty:
        paths["epsilon_scan"] = epsilon_path
    return paths
