#!/usr/bin/env python3
"""Post-process deterministic-closure Phase1-2 data for Spinodal Phase3-4.

Phase3 tests the saddle-node scaling of the measured q=0 relaxation rate.
Phase4 derives a dynamic length from the long-wave dispersion and checks its
internal dynamic-scaling consistency.  No simulation and no MPI are used here.

The numerical helpers are pure functions so that independent B/R/N runs can
later be mapped over by an external driver.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spinodal_phase0 import kappa_R_theory


SCRIPT_VERSION = "2026.08.15-phase34-v1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MISSING_INPUT_MESSAGE = (
    "Phase1-2 outputs are missing. Run spinodal_phase12_mpi.py first."
)
EPSILON_WARNING = (
    "WARNING:\n"
    "Phase1-2 epsilon convergence scan has not been performed.\n"
    "Scaling results currently rely on the adaptive small-epsilon design."
)

PHASE0_SUMMARY_FILE = "phase0_summary.json"
PHASE0_TABLE_FILE = "phase0_delta_table.csv"
PHASE12_MODE_FILE = "phase12_mode_results.csv"
PHASE12_DISPERSION_FILE = "phase12_dispersion_fits.csv"
PHASE12_KERNEL_FILE = "phase12_kernel_relation.csv"
PHASE12_SUMMARY_FILE = "phase12_validation_summary.json"


@dataclass(frozen=True)
class Phase34Inputs:
    """Validated Phase0 and Phase1-2 inputs with no output-side state."""

    phase0_dir: Path
    phase12_dir: Path
    phase0_summary: dict[str, Any]
    phase0_table: pd.DataFrame
    mode_table: pd.DataFrame
    dispersion_table: pd.DataFrame
    kernel_table: pd.DataFrame
    phase12_summary: dict[str, Any]


@dataclass(frozen=True)
class Phase34Analysis:
    """All numerical Phase3-4 products before file or figure output."""

    phase3_scaling: pd.DataFrame
    phase3_powerlaw_fits: pd.DataFrame
    phase3_window_stability: pd.DataFrame
    phase3_effective_exponents: pd.DataFrame
    phase4_lengths: pd.DataFrame
    phase4_scaling_fits: pd.DataFrame
    phase4_systematics: pd.DataFrame
    phase4_collapse: pd.DataFrame
    validation_summary: dict[str, Any]


POWERLAW_COLUMNS = [
    "quantity",
    "window_name",
    "delta_min",
    "delta_max",
    "n_points",
    "amplitude",
    "exponent",
    "exponent_regression_se",
    "r2",
    "expected_exponent",
    "exponent_minus_expected",
    "primary_window",
]


def _require_columns(frame: pd.DataFrame, required: set[str], source_name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {missing}")


def _coerce_boolean(series: pd.Series, source_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(
            f"{source_name} contains invalid boolean values: {invalid}"
        )
    return normalized.map(mapping).astype(bool)


def load_phase34_inputs(phase0_dir: Path, phase12_dir: Path) -> Phase34Inputs:
    """Load and validate existing outputs without regenerating any simulation."""
    phase0_dir = Path(phase0_dir)
    phase12_dir = Path(phase12_dir)
    phase12_paths = [
        phase12_dir / PHASE12_MODE_FILE,
        phase12_dir / PHASE12_DISPERSION_FILE,
        phase12_dir / PHASE12_KERNEL_FILE,
        phase12_dir / PHASE12_SUMMARY_FILE,
    ]
    if any(not path.is_file() for path in phase12_paths):
        raise FileNotFoundError(MISSING_INPUT_MESSAGE)

    phase0_paths = [
        phase0_dir / PHASE0_SUMMARY_FILE,
        phase0_dir / PHASE0_TABLE_FILE,
    ]
    if any(not path.is_file() for path in phase0_paths):
        raise FileNotFoundError(
            "Phase0 outputs are missing. Run spinodal_phase0.py first."
        )

    phase0_summary = json.loads(phase0_paths[0].read_text(encoding="utf-8"))
    phase0_table = pd.read_csv(phase0_paths[1])
    mode_table = pd.read_csv(phase12_paths[0])
    dispersion_table = pd.read_csv(phase12_paths[1])
    kernel_table = pd.read_csv(phase12_paths[2])
    phase12_summary = json.loads(phase12_paths[3].read_text(encoding="utf-8"))

    required_phase0_keys = {"z_spinodal", "sigma_eff", "kappa_R_theory", "inputs"}
    missing_keys = sorted(required_phase0_keys - set(phase0_summary))
    if missing_keys:
        raise ValueError(f"{PHASE0_SUMMARY_FILE} is missing required keys: {missing_keys}")
    _require_columns(
        phase0_table,
        {"delta", "Delta", "Gamma0_theory", "xi_theory"},
        PHASE0_TABLE_FILE,
    )
    _require_columns(
        mode_table,
        {
            "task_group",
            "delta",
            "Delta",
            "N",
            "R",
            "a",
            "mode_index",
            "q",
            "qR",
            "kernel_hat",
            "Gamma_from_lambda",
            "Gamma0_theory",
            "reliable",
        },
        PHASE12_MODE_FILE,
    )
    _require_columns(
        dispersion_table,
        {
            "delta",
            "D_fit",
            "kappa_R_theory",
            "qR_max_fit",
            "n_modes_used",
        },
        PHASE12_DISPERSION_FILE,
    )
    _require_columns(
        kernel_table,
        {
            "delta",
            "mode_index",
            "q",
            "qR",
            "minus_log_kernel",
        },
        PHASE12_KERNEL_FILE,
    )
    if "epsilon_convergence" not in phase12_summary:
        raise ValueError(
            f"{PHASE12_SUMMARY_FILE} is missing required key: epsilon_convergence"
        )

    mode_table = mode_table.copy()
    mode_table["reliable"] = _coerce_boolean(
        mode_table["reliable"], PHASE12_MODE_FILE
    )
    return Phase34Inputs(
        phase0_dir=phase0_dir,
        phase12_dir=phase12_dir,
        phase0_summary=phase0_summary,
        phase0_table=phase0_table,
        mode_table=mode_table,
        dispersion_table=dispersion_table,
        kernel_table=kernel_table,
        phase12_summary=phase12_summary,
    )


def coefficient_of_determination(
    observed: Sequence[float], predicted: Sequence[float]
) -> float:
    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    residual_sum = float(np.sum((observed_array - predicted_array) ** 2))
    centered_sum = float(np.sum((observed_array - np.mean(observed_array)) ** 2))
    if centered_sum <= np.finfo(float).tiny:
        return math.nan
    return 1.0 - residual_sum / centered_sum


def fit_power_law(x: Sequence[float], y: Sequence[float]) -> dict[str, float | int]:
    """Fit y=A*x**p by OLS in log space.

    The reported slope standard error is a regression residual diagnostic.  It
    is not a stochastic measurement error or an experimental confidence bound.
    """
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if x_array.ndim != 1 or y_array.ndim != 1 or len(x_array) != len(y_array):
        raise ValueError("x and y must be one-dimensional arrays of equal length")
    if len(x_array) < 2:
        raise ValueError("power-law fit requires at least two points")
    if (
        not np.all(np.isfinite(x_array))
        or not np.all(np.isfinite(y_array))
        or np.any(x_array <= 0.0)
        or np.any(y_array <= 0.0)
    ):
        raise ValueError("power-law fit requires positive finite x and y")

    log_x = np.log(x_array)
    log_y = np.log(y_array)
    design = np.column_stack((np.ones_like(log_x), log_x))
    beta, _, _, _ = np.linalg.lstsq(design, log_y, rcond=None)
    predicted = design @ beta
    exponent_se = math.nan
    if len(log_x) > 2:
        dof = len(log_x) - 2
        residual_variance = float(np.sum((log_y - predicted) ** 2) / dof)
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        exponent_se = math.sqrt(max(float(covariance[1, 1]), 0.0))
    return {
        "amplitude": float(math.exp(beta[0])),
        "exponent": float(beta[1]),
        "exponent_regression_se": float(exponent_se),
        "r2": float(coefficient_of_determination(log_y, predicted)),
        "n_points": int(len(x_array)),
    }


def build_nested_windows(
    delta: Sequence[float],
    primary_delta_max: float,
    min_window_points: int = 3,
) -> list[dict[str, Any]]:
    """Return pre-specified nested windows, marking the fixed primary window."""
    values = np.asarray(delta, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("delta must contain at least two points")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("delta values must be positive and finite")
    if len(np.unique(values)) != len(values):
        raise ValueError("delta values must be unique")
    if primary_delta_max <= 0.0:
        raise ValueError("primary_delta_max must be positive")
    if min_window_points < 2:
        raise ValueError("min_window_points must be at least 2")

    order = np.argsort(values)
    primary_count = int(
        np.sum(values[order] <= primary_delta_max * (1.0 + 1e-12))
    )
    if primary_count < 2:
        raise ValueError(
            "The fixed primary delta window contains fewer than two points"
        )

    counts = list(range(min_window_points, len(values) + 1))
    if primary_count not in counts:
        counts.append(primary_count)
        counts.sort()
    windows = []
    for count in counts:
        indices = order[:count]
        windows.append(
            {
                "window_name": f"nearest_{count}",
                "indices": indices,
                "delta_min": float(np.min(values[indices])),
                "delta_max": float(np.max(values[indices])),
                "n_points": int(count),
                "primary_window": bool(count == primary_count),
            }
        )
    return windows


def fit_nested_powerlaw_windows(
    delta: Sequence[float],
    values: Sequence[float],
    *,
    quantity: str,
    expected_exponent: float,
    primary_delta_max: float,
    min_window_points: int = 3,
) -> pd.DataFrame:
    """Fit every nested nearest-spinodal window without optimizing the exponent."""
    delta_array = np.asarray(delta, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if len(delta_array) != len(value_array):
        raise ValueError("delta and values must have equal length")
    rows: list[dict[str, Any]] = []
    for window in build_nested_windows(
        delta_array, primary_delta_max, min_window_points
    ):
        fit = fit_power_law(
            delta_array[window["indices"]], value_array[window["indices"]]
        )
        rows.append(
            {
                "quantity": quantity,
                "window_name": window["window_name"],
                "delta_min": window["delta_min"],
                "delta_max": window["delta_max"],
                "n_points": window["n_points"],
                "amplitude": fit["amplitude"],
                "exponent": fit["exponent"],
                "exponent_regression_se": fit["exponent_regression_se"],
                "r2": fit["r2"],
                "expected_exponent": float(expected_exponent),
                "exponent_minus_expected": (
                    float(fit["exponent"]) - expected_exponent
                ),
                "primary_window": window["primary_window"],
            }
        )
    return pd.DataFrame(rows, columns=POWERLAW_COLUMNS)


def compute_effective_exponents(
    delta: Sequence[float],
    values: Sequence[float],
    *,
    quantity: str,
    expected_exponent: float,
) -> pd.DataFrame:
    """Compute adjacent-point effective exponents from nearest to farthest."""
    delta_array = np.asarray(delta, dtype=float)
    value_array = np.asarray(values, dtype=float)
    if (
        len(delta_array) != len(value_array)
        or len(delta_array) < 2
        or np.any(delta_array <= 0.0)
        or np.any(value_array <= 0.0)
    ):
        raise ValueError("effective exponents require equal positive arrays")
    order = np.argsort(delta_array)
    rows = []
    for near_index, far_index in zip(order[:-1], order[1:]):
        exponent = (
            math.log(value_array[far_index]) - math.log(value_array[near_index])
        ) / (
            math.log(delta_array[far_index]) - math.log(delta_array[near_index])
        )
        rows.append(
            {
                "quantity": quantity,
                "delta_near": float(delta_array[near_index]),
                "delta_far": float(delta_array[far_index]),
                "delta_geometric_mean": float(
                    math.sqrt(delta_array[near_index] * delta_array[far_index])
                ),
                "effective_exponent": float(exponent),
                "expected_exponent": float(expected_exponent),
                "exponent_minus_expected": float(exponent - expected_exponent),
            }
        )
    return pd.DataFrame(rows)


def top_hat_kernel_expansion(
    R: int, lattice_spacing: float
) -> dict[str, float]:
    """Return model-specific q² and q⁴ coefficients for -log(Khat_R)."""
    if R < 1 or not math.isfinite(lattice_spacing) or lattice_spacing <= 0.0:
        raise ValueError("R must be >=1 and lattice spacing must be positive")
    kappa = kappa_R_theory(R, lattice_spacing)
    eta = (
        lattice_spacing**4
        * (R + 1)
        * (2 * R + 1)
        * (3 * R**2 + 3 * R - 1)
        / 720.0
    )
    return {
        "kappa_R": float(kappa),
        "eta_R": float(eta),
        "c4_R": float(kappa**2 / 2.0 - eta),
    }


def fit_free_intercept(
    x: Sequence[float], y: Sequence[float]
) -> dict[str, float | int]:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    if len(x_array) != len(y_array) or len(x_array) < 2 or np.ptp(x_array) <= 0.0:
        raise ValueError("free-intercept fit requires at least two distinct x points")
    design = np.column_stack((np.ones_like(x_array), x_array))
    beta, _, _, _ = np.linalg.lstsq(design, y_array, rcond=None)
    predicted = design @ beta
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r2": float(coefficient_of_determination(y_array, predicted)),
        "n_points": int(len(x_array)),
    }


def fit_q2_q4_dispersion(
    q: Sequence[float],
    gamma_q: Sequence[float],
    gamma0: float,
) -> dict[str, float | int]:
    """Fit Gamma(q)-Gamma0=D2*q²+D4*q⁴ with fixed measured intercept."""
    q_array = np.asarray(q, dtype=float)
    gamma_array = np.asarray(gamma_q, dtype=float)
    if (
        len(q_array) != len(gamma_array)
        or len(q_array) < 3
        or not np.all(np.isfinite(q_array))
        or not np.all(np.isfinite(gamma_array))
        or not math.isfinite(gamma0)
    ):
        raise ValueError("q²+q⁴ fit requires at least three finite points")
    q2 = q_array**2
    design = np.column_stack((q2, q2**2))
    if np.linalg.matrix_rank(design) < 2:
        raise ValueError("q²+q⁴ fit requires at least two distinct nonzero q values")
    beta, _, _, _ = np.linalg.lstsq(design, gamma_array - gamma0, rcond=None)
    predicted = gamma0 + design @ beta
    return {
        "D2": float(beta[0]),
        "D4": float(beta[1]),
        "r2": float(coefficient_of_determination(gamma_array, predicted)),
        "n_points": int(len(q_array)),
    }


def compute_dynamic_lengths(
    gamma0: Sequence[float],
    D_raw: Sequence[float],
    D_q4: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    gamma_array = np.asarray(gamma0, dtype=float)
    raw_array = np.asarray(D_raw, dtype=float)
    if (
        gamma_array.shape != raw_array.shape
        or np.any(gamma_array <= 0.0)
        or np.any(raw_array <= 0.0)
    ):
        raise ValueError("Gamma0 and D_raw must be equal-shape positive arrays")
    xi_raw = np.sqrt(raw_array / gamma_array)
    if D_q4 is None:
        return xi_raw, np.full_like(xi_raw, np.nan)
    q4_array = np.asarray(D_q4, dtype=float)
    if q4_array.shape != gamma_array.shape:
        raise ValueError("D_q4 must have the same shape as Gamma0")
    xi_q4 = np.where(q4_array > 0.0, np.sqrt(q4_array / gamma_array), np.nan)
    return xi_raw, xi_q4


def build_collapse_values(
    q: Sequence[float],
    gamma_q: Sequence[float],
    gamma0: float,
    D: float,
) -> pd.DataFrame:
    """Build the long-wave scaling variables for one delta."""
    q_array = np.asarray(q, dtype=float)
    gamma_array = np.asarray(gamma_q, dtype=float)
    if (
        len(q_array) != len(gamma_array)
        or gamma0 <= 0.0
        or D <= 0.0
        or np.any(gamma_array <= 0.0)
    ):
        raise ValueError("collapse inputs require positive rates and D")
    xi = math.sqrt(D / gamma0)
    collapse_x = q_array * xi
    collapse_y = gamma0 / gamma_array
    theory = 1.0 / (1.0 + collapse_x**2)
    return pd.DataFrame(
        {
            "collapse_x_qxi": collapse_x,
            "collapse_y_tau_ratio": collapse_y,
            "collapse_theory": theory,
            "collapse_residual": collapse_y - theory,
        }
    )


def _select_phase3_primary_rows(
    inputs: Phase34Inputs, include_unreliable: bool
) -> pd.DataFrame:
    mode = inputs.mode_table
    selected = mode[
        (mode["task_group"] == "main") & (mode["mode_index"] == 0)
    ].copy()
    if not include_unreliable:
        selected = selected[selected["reliable"]]
    if selected.empty:
        raise ValueError("No main, reliable q=0 Phase1 rows were found")
    duplicates = selected["delta"].duplicated(keep=False)
    if duplicates.any():
        values = sorted(selected.loc[duplicates, "delta"].unique())
        raise ValueError(f"q=0 primary input is not unique for delta values: {values}")
    return selected.sort_values("delta", ascending=False).reset_index(drop=True)


def build_phase3_scaling_table(
    inputs: Phase34Inputs, include_unreliable: bool = False
) -> pd.DataFrame:
    """Construct Phase3 observables from the single designated q=0 input."""
    q0 = _select_phase3_primary_rows(inputs, include_unreliable)
    phase0 = inputs.phase0_table[
        ["delta", "Delta", "Gamma0_theory", "xi_theory"]
    ].copy()
    merged = q0.merge(
        phase0,
        on="delta",
        how="left",
        validate="one_to_one",
        suffixes=("", "_phase0"),
    )
    if merged[["Gamma0_theory_phase0", "xi_theory"]].isna().any().any():
        raise ValueError("Phase0 and Phase1-2 delta grids do not match")
    gamma = merged["Gamma_from_lambda"].to_numpy(dtype=float)
    if np.any(gamma <= 0.0) or not np.all(np.isfinite(gamma)):
        raise ValueError("Gamma_from_lambda(q=0) must be positive and finite")
    delta = merged["delta"].to_numpy(dtype=float)
    theory = merged["Gamma0_theory_phase0"].to_numpy(dtype=float)
    table = pd.DataFrame(
        {
            "delta": delta,
            "Delta": merged["Delta"].to_numpy(dtype=float),
            "Gamma0_num": gamma,
            "Gamma0_theory": theory,
            "tau0_num": 1.0 / gamma,
            "Gamma0_over_sqrt_delta": gamma / np.sqrt(delta),
            "tau0_times_sqrt_delta": np.sqrt(delta) / gamma,
            "Gamma0_relative_error_to_phase0": np.abs(gamma - theory)
            / np.abs(theory),
            "reliable": merged["reliable"].to_numpy(dtype=bool),
        }
    )
    return table.sort_values("delta", ascending=False).reset_index(drop=True)


def build_dispersion_systematics(
    inputs: Phase34Inputs,
    phase3_scaling: pd.DataFrame,
    qR_max: float,
    *,
    include_unreliable: bool = False,
    disable_q4: bool = False,
) -> pd.DataFrame:
    """Quantify finite-q-window and q⁴ systematics for every delta."""
    if qR_max <= 0.0:
        raise ValueError("qR_max must be positive")
    mode = inputs.mode_table[inputs.mode_table["task_group"] == "main"].copy()
    if not include_unreliable:
        mode = mode[mode["reliable"]]
    scaling_by_delta = phase3_scaling.set_index("delta")
    dispersion = inputs.dispersion_table.set_index("delta")
    inputs0 = inputs.phase0_summary["inputs"]
    R = int(inputs0["R"])
    a = float(inputs0.get("a", inputs0.get("lattice_spacing", 1.0)))
    coefficients = top_hat_kernel_expansion(R, a)

    rows: list[dict[str, Any]] = []
    for delta, group in mode.groupby("delta"):
        if delta not in scaling_by_delta.index or delta not in dispersion.index:
            continue
        eligible = group[
            (group["kernel_hat"] > 0.0) & (group["qR"] <= qR_max)
        ].sort_values("mode_index")
        q0 = eligible[eligible["mode_index"] == 0]
        if q0.empty:
            raise ValueError(f"q=0 mode is missing from dispersion window at delta={delta:g}")
        if len(eligible) < 3:
            raise ValueError(
                f"At least three low-q modes are required at delta={delta:g}"
            )
        q = eligible["q"].to_numpy(dtype=float)
        kernel_y = -np.log(np.abs(eligible["kernel_hat"].to_numpy(dtype=float)))
        kernel_fit = fit_free_intercept(q**2, kernel_y)
        gamma0 = float(scaling_by_delta.loc[delta, "Gamma0_num"])

        if disable_q4:
            q4_fit = {"D2": math.nan, "D4": math.nan, "r2": math.nan}
        else:
            q4_fit = fit_q2_q4_dispersion(
                q,
                eligible["Gamma_from_lambda"].to_numpy(dtype=float),
                gamma0,
            )
        D_fit = float(dispersion.loc[delta, "D_fit"])
        D_kernel = float(kernel_fit["slope"])
        kappa = coefficients["kappa_R"]
        D2 = float(q4_fit["D2"])
        D4 = float(q4_fit["D4"])
        rows.append(
            {
                "delta": float(delta),
                "Gamma0_num": gamma0,
                "D_fit_raw": D_fit,
                "D_kernel_window": D_kernel,
                "kappa_R_theory": kappa,
                "D_fit_minus_D_kernel_window": D_fit - D_kernel,
                "D_fit_minus_kappa": D_fit - kappa,
                "D2_q4_fit": D2,
                "D4_q4_fit": D4,
                "c4_theory": coefficients["c4_R"],
                "D2_relative_error_to_kappa": (
                    abs(D2 - kappa) / kappa if math.isfinite(D2) else math.nan
                ),
                "D4_relative_error_to_c4": (
                    abs(D4 - coefficients["c4_R"]) / abs(coefficients["c4_R"])
                    if math.isfinite(D4)
                    else math.nan
                ),
                "qR_max_used": float(eligible["qR"].max()),
                "n_modes_used": int(len(eligible)),
                "q4_fit_r2": float(q4_fit["r2"]),
            }
        )
    if not rows:
        raise ValueError("No matching Phase2 dispersion rows were found")
    return pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)


def build_phase4_length_table(
    inputs: Phase34Inputs,
    phase3_scaling: pd.DataFrame,
    systematics: pd.DataFrame,
) -> pd.DataFrame:
    merged = phase3_scaling.merge(
        systematics, on=["delta", "Gamma0_num"], how="inner", validate="one_to_one"
    )
    phase0 = inputs.phase0_table[["delta", "xi_theory"]]
    merged = merged.merge(phase0, on="delta", how="left", validate="one_to_one")
    q0 = _select_phase3_primary_rows(inputs, include_unreliable=True)[
        ["delta", "N", "a"]
    ]
    merged = merged.merge(q0, on="delta", how="left", validate="one_to_one")
    xi_raw, xi_q4 = compute_dynamic_lengths(
        merged["Gamma0_num"],
        merged["D_fit_raw"],
        merged["D2_q4_fit"],
    )
    qmin = 2.0 * math.pi / (
        merged["N"].to_numpy(dtype=float) * merged["a"].to_numpy(dtype=float)
    )
    table = pd.DataFrame(
        {
            "delta": merged["delta"].to_numpy(dtype=float),
            "Delta": merged["Delta"].to_numpy(dtype=float),
            "Gamma0_num": merged["Gamma0_num"].to_numpy(dtype=float),
            "D_fit_raw": merged["D_fit_raw"].to_numpy(dtype=float),
            "D_kernel_window": merged["D_kernel_window"].to_numpy(dtype=float),
            "D2_q4_fit": merged["D2_q4_fit"].to_numpy(dtype=float),
            "kappa_R_theory": merged["kappa_R_theory"].to_numpy(dtype=float),
            "xi_dyn_raw": xi_raw,
            "xi_dyn_q4": xi_q4,
            "xi_theory_phase0": merged["xi_theory"].to_numpy(dtype=float),
            "N": merged["N"].to_numpy(dtype=int),
            "qmin": qmin,
            "N_over_xi_raw": merged["N"].to_numpy(dtype=float) / xi_raw,
            "qmin_xi_raw": qmin * xi_raw,
            "xi_times_delta_quarter": xi_raw
            * merged["delta"].to_numpy(dtype=float) ** 0.25,
            "reliable": merged["reliable"].to_numpy(dtype=bool)
            & np.isfinite(xi_raw),
        }
    )
    return table.sort_values("delta", ascending=False).reset_index(drop=True)


def build_phase4_collapse_table(
    inputs: Phase34Inputs,
    phase4_lengths: pd.DataFrame,
    qR_max: float,
) -> pd.DataFrame:
    lengths = phase4_lengths.set_index("delta")
    mode = inputs.mode_table[inputs.mode_table["task_group"] == "main"].copy()
    rows: list[pd.DataFrame] = []
    for delta, group in mode.groupby("delta"):
        if delta not in lengths.index:
            continue
        group = group[group["kernel_hat"] > 0.0].sort_values("mode_index").copy()
        gamma0 = float(lengths.loc[delta, "Gamma0_num"])
        D = float(lengths.loc[delta, "D_fit_raw"])
        collapse = build_collapse_values(
            group["q"], group["Gamma_from_lambda"], gamma0, D
        )
        reliable = group["reliable"].to_numpy(dtype=bool)
        longwave = reliable & (group["qR"].to_numpy(dtype=float) <= qR_max)
        frame = pd.DataFrame(
            {
                "delta": float(delta),
                "mode_index": group["mode_index"].to_numpy(dtype=int),
                "q": group["q"].to_numpy(dtype=float),
                "qR": group["qR"].to_numpy(dtype=float),
                "Gamma_q": group["Gamma_from_lambda"].to_numpy(dtype=float),
                "Gamma0": gamma0,
                "D_used": D,
                "xi_dyn": float(lengths.loc[delta, "xi_dyn_raw"]),
                "collapse_x_qxi": collapse["collapse_x_qxi"].to_numpy(),
                "collapse_y_tau_ratio": collapse[
                    "collapse_y_tau_ratio"
                ].to_numpy(),
                "collapse_theory": collapse["collapse_theory"].to_numpy(),
                "collapse_residual": collapse["collapse_residual"].to_numpy(),
                "longwave_used": longwave,
                "reliable": reliable,
            }
        )
        rows.append(frame)
    if not rows:
        raise ValueError("No finite-q data are available for collapse")
    return pd.concat(rows, ignore_index=True).sort_values(
        ["delta", "mode_index"], ascending=[False, True]
    )


def _primary_fit(frame: pd.DataFrame, quantity: str) -> pd.Series:
    selected = frame[
        (frame["quantity"] == quantity) & frame["primary_window"]
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one primary fit for {quantity}, found {len(selected)}")
    return selected.iloc[0]


def _all_points_fit(frame: pd.DataFrame, quantity: str) -> pd.Series:
    selected = frame[frame["quantity"] == quantity]
    maximum = int(selected["n_points"].max())
    rows = selected[selected["n_points"] == maximum]
    if len(rows) != 1:
        raise ValueError(f"Expected one all-points fit for {quantity}")
    return rows.iloc[0]


def _build_phase4_scaling_fits(
    phase3: pd.DataFrame,
    lengths: pd.DataFrame,
    primary_delta_max: float,
    min_window_points: int,
) -> pd.DataFrame:
    delta = lengths["delta"].to_numpy(dtype=float)
    frames = [
        fit_nested_powerlaw_windows(
            delta,
            lengths["xi_dyn_raw"],
            quantity="xi_dyn_raw",
            expected_exponent=-0.25,
            primary_delta_max=primary_delta_max,
            min_window_points=min_window_points,
        )
    ]
    q4_valid = np.isfinite(lengths["xi_dyn_q4"]) & (lengths["xi_dyn_q4"] > 0.0)
    if int(q4_valid.sum()) >= 2:
        frames.append(
            fit_nested_powerlaw_windows(
                delta[q4_valid],
                lengths.loc[q4_valid, "xi_dyn_q4"],
                quantity="xi_dyn_q4",
                expected_exponent=-0.25,
                primary_delta_max=primary_delta_max,
                min_window_points=min(min_window_points, int(q4_valid.sum())),
            )
        )

    tau_by_delta = phase3.set_index("delta")["tau0_num"]
    tau = np.array([float(tau_by_delta.loc[value]) for value in delta])
    z_rows = []
    for window in build_nested_windows(delta, primary_delta_max, min_window_points):
        indices = window["indices"]
        fit = fit_power_law(
            lengths["xi_dyn_raw"].to_numpy(dtype=float)[indices], tau[indices]
        )
        z_rows.append(
            {
                "quantity": "tau0_vs_xi_raw",
                "window_name": window["window_name"],
                "delta_min": window["delta_min"],
                "delta_max": window["delta_max"],
                "n_points": window["n_points"],
                "amplitude": fit["amplitude"],
                "exponent": fit["exponent"],
                "exponent_regression_se": fit["exponent_regression_se"],
                "r2": fit["r2"],
                "expected_exponent": 2.0,
                "exponent_minus_expected": float(fit["exponent"]) - 2.0,
                "primary_window": window["primary_window"],
            }
        )
    frames.append(pd.DataFrame(z_rows, columns=POWERLAW_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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


def run_phase34_analysis(
    inputs: Phase34Inputs,
    *,
    primary_delta_max: float = 3e-4,
    qR_max_collapse: float = 0.35,
    min_window_points: int = 3,
    include_unreliable: bool = False,
    disable_q4_systematics: bool = False,
) -> Phase34Analysis:
    """Run all Phase3-4 numerical post-processing without file output."""
    phase3 = build_phase3_scaling_table(inputs, include_unreliable)
    delta = phase3["delta"].to_numpy(dtype=float)
    gamma = phase3["Gamma0_num"].to_numpy(dtype=float)
    tau = phase3["tau0_num"].to_numpy(dtype=float)

    gamma_fits = fit_nested_powerlaw_windows(
        delta,
        gamma,
        quantity="Gamma0_num",
        expected_exponent=0.5,
        primary_delta_max=primary_delta_max,
        min_window_points=min_window_points,
    )
    tau_fits = fit_nested_powerlaw_windows(
        delta,
        tau,
        quantity="tau0_num_derived",
        expected_exponent=-0.5,
        primary_delta_max=primary_delta_max,
        min_window_points=min_window_points,
    )
    phase3_fits = pd.concat([gamma_fits, tau_fits], ignore_index=True)
    stability = gamma_fits[
        [
            "window_name",
            "delta_min",
            "delta_max",
            "n_points",
            "amplitude",
            "exponent",
            "exponent_regression_se",
            "r2",
            "primary_window",
        ]
    ].rename(
        columns={
            "amplitude": "C_Gamma",
            "exponent": "p_Gamma",
            "exponent_regression_se": "p_Gamma_regression_se",
            "r2": "Gamma_r2",
        }
    )
    tau_stability = tau_fits[
        ["window_name", "exponent", "exponent_regression_se", "r2"]
    ].rename(
        columns={
            "exponent": "p_tau",
            "exponent_regression_se": "p_tau_regression_se",
            "r2": "tau_r2",
        }
    )
    stability = stability.merge(tau_stability, on="window_name", validate="one_to_one")

    systematics = build_dispersion_systematics(
        inputs,
        phase3,
        qR_max_collapse,
        include_unreliable=include_unreliable,
        disable_q4=disable_q4_systematics,
    )
    lengths = build_phase4_length_table(inputs, phase3, systematics)
    phase4_fits = _build_phase4_scaling_fits(
        phase3, lengths, primary_delta_max, min_window_points
    )
    collapse = build_phase4_collapse_table(inputs, lengths, qR_max_collapse)

    effective_frames = [
        compute_effective_exponents(
            delta, gamma, quantity="Gamma0_num", expected_exponent=0.5
        ),
        compute_effective_exponents(
            delta, tau, quantity="tau0_num_derived", expected_exponent=-0.5
        ),
        compute_effective_exponents(
            lengths["delta"],
            lengths["xi_dyn_raw"],
            quantity="xi_dyn_raw",
            expected_exponent=-0.25,
        ),
    ]
    q4_valid = np.isfinite(lengths["xi_dyn_q4"]) & (lengths["xi_dyn_q4"] > 0.0)
    if int(q4_valid.sum()) >= 2:
        effective_frames.append(
            compute_effective_exponents(
                lengths.loc[q4_valid, "delta"],
                lengths.loc[q4_valid, "xi_dyn_q4"],
                quantity="xi_dyn_q4",
                expected_exponent=-0.25,
            )
        )
    effective = pd.concat(effective_frames, ignore_index=True)

    gamma_primary = _primary_fit(phase3_fits, "Gamma0_num")
    gamma_all = _all_points_fit(phase3_fits, "Gamma0_num")
    tau_primary = _primary_fit(phase3_fits, "tau0_num_derived")
    xi_primary = _primary_fit(phase4_fits, "xi_dyn_raw")
    xi_q4_rows = phase4_fits[phase4_fits["quantity"] == "xi_dyn_q4"]
    xi_q4_primary = (
        _primary_fit(phase4_fits, "xi_dyn_q4")
        if not xi_q4_rows.empty
        else None
    )
    z_primary = _primary_fit(phase4_fits, "tau0_vs_xi_raw")

    phase0_summary = inputs.phase0_summary
    C_gamma_theory = math.sqrt(
        2.0
        * abs(float(phase0_summary["z_spinodal"]))
        / float(phase0_summary["sigma_eff"])
    )
    kappa = float(phase0_summary["kappa_R_theory"])
    C_xi_theory = math.sqrt(kappa / C_gamma_theory)
    nearest_phase3 = phase3.loc[phase3["delta"].idxmin()]
    nearest_systematics = systematics.loc[systematics["delta"].idxmin()]
    collapse_primary = collapse[collapse["longwave_used"]]
    collapse_residual = collapse_primary["collapse_residual"].to_numpy(dtype=float)
    D_kappa_error = np.abs(systematics["D_fit_raw"] - kappa) / kappa
    D_kernel_error = np.abs(
        systematics["D_fit_raw"] - systematics["D_kernel_window"]
    ) / np.abs(systematics["D_kernel_window"])
    epsilon_performed = bool(
        inputs.phase12_summary.get("epsilon_convergence", {}).get("performed", False)
    )

    validation = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calculation_type": (
            "serial post-processing of deterministic Gaussian-closure Phase1-2 data"
        ),
        "inputs": {
            "source_phase12_dir": str(inputs.phase12_dir),
            "source_phase0_dir": str(inputs.phase0_dir),
            "primary_delta_max": float(primary_delta_max),
            "qR_max_collapse": float(qR_max_collapse),
            "min_window_points": int(min_window_points),
            "include_unreliable": bool(include_unreliable),
            "q4_systematics_enabled": not disable_q4_systematics,
        },
        "source_phase12_dir": str(inputs.phase12_dir),
        "source_phase0_dir": str(inputs.phase0_dir),
        "interpretation": {
            "phase3_primary": "Gamma0_num is measured q=0 Gamma_from_lambda",
            "tau0": "derived as 1/Gamma0_num; not an independent exponent measurement",
            "phase4_xi": (
                "derived as sqrt(D/Gamma0); an internal space-time scaling "
                "consistency check, not an independent real-space measurement"
            ),
            "dynamic_z": (
                "derived from the same Gamma0 and D data; a dynamic-scaling "
                "consistency check, not an independent z measurement"
            ),
            "future_independent_test": "Phase6 fixed-boundary response xi_boundary",
            "regression_se": (
                "OLS residual diagnostic in log space, not stochastic sampling error"
            ),
        },
        "phase3": {
            "primary_delta_max": float(primary_delta_max),
            "gamma_exponent_primary": float(gamma_primary["exponent"]),
            "gamma_exponent_expected": 0.5,
            "gamma_exponent_difference": float(
                gamma_primary["exponent_minus_expected"]
            ),
            "gamma_primary_regression_se": float(
                gamma_primary["exponent_regression_se"]
            ),
            "gamma_primary_r2": float(gamma_primary["r2"]),
            "gamma_exponent_all_points": float(gamma_all["exponent"]),
            "tau_exponent_primary": float(tau_primary["exponent"]),
            "C_gamma_fit": float(gamma_primary["amplitude"]),
            "C_gamma_theory": C_gamma_theory,
            "C_tau_theory": 1.0 / C_gamma_theory,
            "scaled_amplitude_nearest_spinodal": float(
                nearest_phase3["Gamma0_over_sqrt_delta"]
            ),
        },
        "phase3_window_stability": stability[
            ["window_name", "delta_max", "n_points", "p_Gamma", "p_tau"]
        ].to_dict(orient="records"),
        "phase4": {
            "xi_exponent_raw_primary": float(xi_primary["exponent"]),
            "xi_exponent_expected": -0.25,
            "xi_exponent_q4_primary": (
                float(xi_q4_primary["exponent"])
                if xi_q4_primary is not None
                else None
            ),
            "z_raw_primary": float(z_primary["exponent"]),
            "z_expected": 2.0,
            "C_xi_theory": C_xi_theory,
        },
        "dispersion_systematics": {
            "eta_R_theory": top_hat_kernel_expansion(
                int(phase0_summary["inputs"]["R"]),
                float(
                    phase0_summary["inputs"].get(
                        "a", phase0_summary["inputs"].get("lattice_spacing", 1.0)
                    )
                ),
            )["eta_R"],
            "c4_R_theory": float(nearest_systematics["c4_theory"]),
            "max_Dfit_relative_error_to_kappa": float(np.max(D_kappa_error)),
            "max_Dfit_relative_error_to_kernel_window": float(
                np.max(D_kernel_error)
            ),
            "nearest_spinodal_D2_relative_error_to_kappa": float(
                nearest_systematics["D2_relative_error_to_kappa"]
            ),
            "nearest_spinodal_D4_relative_error_to_c4": float(
                nearest_systematics["D4_relative_error_to_c4"]
            ),
        },
        "collapse": {
            "number_of_points": int(len(collapse_primary)),
            "rms_error": float(math.sqrt(np.mean(collapse_residual**2))),
            "max_absolute_error": float(np.max(np.abs(collapse_residual))),
        },
        "finite_size": {
            "min_N_over_xi": float(lengths["N_over_xi_raw"].min()),
            "max_qmin_xi": float(lengths["qmin_xi_raw"].max()),
            "interpretation": "practical diagnostics, not universal hard thresholds",
        },
        "linearity_warning": (
            None
            if epsilon_performed
            else "epsilon convergence not yet explicitly checked"
        ),
        "soft_checks": {
            "primary_gamma_exponent_within_0.02_of_half": abs(
                float(gamma_primary["exponent"]) - 0.5
            )
            < 0.02,
            "primary_xi_exponent_within_0.02_of_minus_quarter": abs(
                float(xi_primary["exponent"]) + 0.25
            )
            < 0.02,
            "z_within_0.05_of_2": abs(float(z_primary["exponent"]) - 2.0) < 0.05,
            "collapse_rms_below_1e-3": math.sqrt(
                float(np.mean(collapse_residual**2))
            )
            < 1e-3,
            "q4_corrected_D_near_kappa": (
                math.isfinite(
                    float(nearest_systematics["D2_relative_error_to_kappa"])
                )
                and float(nearest_systematics["D2_relative_error_to_kappa"]) < 0.01
            ),
        },
    }
    return Phase34Analysis(
        phase3_scaling=phase3,
        phase3_powerlaw_fits=phase3_fits,
        phase3_window_stability=stability,
        phase3_effective_exponents=effective,
        phase4_lengths=lengths,
        phase4_scaling_fits=phase4_fits,
        phase4_systematics=systematics,
        phase4_collapse=collapse,
        validation_summary=_json_safe(validation),
    )


def _reference_powerlaw(
    x: np.ndarray, anchor_x: float, anchor_y: float, exponent: float
) -> np.ndarray:
    return anchor_y * (x / anchor_x) ** exponent


def make_phase34_figures(analysis: Phase34Analysis, output_dir: Path) -> dict[str, Path]:
    """Write the ten required diagnostic figures from already computed tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p3 = analysis.phase3_scaling.sort_values("delta")
    p3fits = analysis.phase3_powerlaw_fits
    stability = analysis.phase3_window_stability.sort_values("n_points")
    lengths = analysis.phase4_lengths.sort_values("delta")
    systematics = analysis.phase4_systematics.sort_values("delta")
    collapse = analysis.phase4_collapse
    summary = analysis.validation_summary
    primary_max = float(summary["phase3"]["primary_delta_max"])
    primary_mask = p3["delta"] <= primary_max * (1.0 + 1e-12)
    gamma_fit = _primary_fit(p3fits, "Gamma0_num")
    tau_fit = _primary_fit(p3fits, "tau0_num_derived")
    xi_fit = _primary_fit(analysis.phase4_scaling_fits, "xi_dyn_raw")
    z_fit = _primary_fit(analysis.phase4_scaling_fits, "tau0_vs_xi_raw")

    paths: dict[str, Path] = {}

    path = output_dir / "phase3_gamma0_scaling.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.loglog(p3.loc[~primary_mask, "delta"], p3.loc[~primary_mask, "Gamma0_num"], "o", mfc="none", label="numerical (outside primary)")
    ax.loglog(p3.loc[primary_mask, "delta"], p3.loc[primary_mask, "Gamma0_num"], "o", label="numerical (primary)")
    ax.loglog(p3["delta"], p3["Gamma0_theory"], "-", label="Phase0 Gaussian-map theory")
    xline = np.logspace(np.log10(p3["delta"].min()), np.log10(primary_max), 200)
    ax.loglog(xline, float(gamma_fit["amplitude"]) * xline ** float(gamma_fit["exponent"]), "--", label=rf"primary fit $p={float(gamma_fit['exponent']):.5f}$")
    anchor = p3.loc[primary_mask].iloc[-1]
    ax.loglog(xline, _reference_powerlaw(xline, float(anchor["delta"]), float(anchor["Gamma0_num"]), 0.5), ":", label="slope 1/2")
    ax.set(xlabel=r"$\delta$", ylabel=r"$\Gamma_0$", title="Phase3: q=0 spinodal slowing down")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase3_gamma0_scaling"] = path

    path = output_dir / "phase3_tau0_scaling.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.loglog(p3.loc[~primary_mask, "delta"], p3.loc[~primary_mask, "tau0_num"], "o", mfc="none", label="derived tau (outside primary)")
    ax.loglog(p3.loc[primary_mask, "delta"], p3.loc[primary_mask, "tau0_num"], "o", label="derived tau (primary)")
    ax.loglog(xline, float(tau_fit["amplitude"]) * xline ** float(tau_fit["exponent"]), "--", label=rf"primary fit $p={float(tau_fit['exponent']):.5f}$")
    anchor = p3.loc[primary_mask].iloc[-1]
    ax.loglog(xline, _reference_powerlaw(xline, float(anchor["delta"]), float(anchor["tau0_num"]), -0.5), ":", label="slope -1/2")
    ax.set(xlabel=r"$\delta$", ylabel=r"$\tau_0=1/\Gamma_0$", title="Phase3: derived relaxation time")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase3_tau0_scaling"] = path

    path = output_dir / "phase3_gamma_scaled_amplitude.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.semilogx(p3["delta"], p3["Gamma0_over_sqrt_delta"], "o-", label=r"$\Gamma_0/\sqrt{\delta}$")
    ax.axhline(float(summary["phase3"]["C_gamma_theory"]), color="k", linestyle="--", label=r"$C_\Gamma^{th}$")
    ax.set(xlabel=r"$\delta$", ylabel=r"$\Gamma_0/\sqrt{\delta}$", title="Phase3: Gaussian-map saddle-node amplitude")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase3_gamma_scaled_amplitude"] = path

    path = output_dir / "phase3_exponent_window_stability.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.errorbar(stability["n_points"], stability["p_Gamma"], yerr=stability["p_Gamma_regression_se"], fmt="o-", capsize=3, label=r"$p_\Gamma$")
    primary = stability[stability["primary_window"]]
    ax.scatter(primary["n_points"], primary["p_Gamma"], s=110, facecolors="none", edgecolors="tab:red", linewidths=1.8, label="fixed primary window")
    ax.axhline(0.5, color="k", linestyle="--", label="expected 1/2")
    ax.set(xlabel="number of nearest-spinodal points", ylabel=r"$p_\Gamma$", title="Phase3: fit-window systematic diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase3_exponent_window_stability"] = path

    path = output_dir / "phase4_D_systematics.png"
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    ax.semilogx(systematics["delta"], systematics["D_fit_raw"], "o-", label=r"raw $D_{fit}$")
    ax.semilogx(systematics["delta"], systematics["D_kernel_window"], "s-", label=r"$D_{kernel,window}$")
    ax.semilogx(systematics["delta"], systematics["D2_q4_fit"], "^-", label=r"$D_2$ from $q^2+q^4$")
    ax.axhline(float(systematics["kappa_R_theory"].iloc[0]), color="k", linestyle="--", label=r"$\kappa_R$")
    ax.set(xlabel=r"$\delta$", ylabel="dispersion coefficient", title="Phase4: finite-q dispersion systematics")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase4_D_systematics"] = path

    path = output_dir / "phase4_xi_scaling.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    ax.loglog(lengths["delta"], lengths["xi_dyn_raw"], "o-", label=r"$\xi_{dyn,raw}$")
    ax.loglog(lengths["delta"], lengths["xi_dyn_q4"], "s-", label=r"$\xi_{dyn,q4}$")
    ax.loglog(lengths["delta"], lengths["xi_theory_phase0"], "-", label="Phase0 theory")
    xline = np.logspace(np.log10(lengths["delta"].min()), np.log10(primary_max), 200)
    ax.loglog(xline, float(xi_fit["amplitude"]) * xline ** float(xi_fit["exponent"]), "--", label=rf"raw primary fit $p={float(xi_fit['exponent']):.5f}$")
    anchor = lengths[lengths["delta"] <= primary_max * (1.0 + 1e-12)].iloc[-1]
    ax.loglog(xline, _reference_powerlaw(xline, float(anchor["delta"]), float(anchor["xi_dyn_raw"]), -0.25), ":", label="slope -1/4")
    ax.set(xlabel=r"$\delta$", ylabel=r"$\xi_{dyn}$", title="Phase4: derived dynamic length")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase4_xi_scaling"] = path

    path = output_dir / "phase4_xi_scaled_amplitude.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    ax.semilogx(lengths["delta"], lengths["xi_times_delta_quarter"], "o-", label=r"$\xi_{dyn,raw}\delta^{1/4}$")
    ax.axhline(float(summary["phase4"]["C_xi_theory"]), color="k", linestyle="--", label=r"$C_\xi^{th}$")
    ax.set(xlabel=r"$\delta$", ylabel=r"$\xi\delta^{1/4}$", title="Phase4: dynamic-length amplitude")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase4_xi_scaled_amplitude"] = path

    path = output_dir / "phase4_tau_vs_xi.png"
    fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
    tau_by_delta = p3.set_index("delta")["tau0_num"]
    tau_values = np.array([float(tau_by_delta.loc[value]) for value in lengths["delta"]])
    ax.loglog(lengths["xi_dyn_raw"], tau_values, "o", label="derived data")
    xi_line = np.logspace(np.log10(lengths["xi_dyn_raw"].min()), np.log10(lengths["xi_dyn_raw"].max()), 200)
    ax.loglog(xi_line, float(z_fit["amplitude"]) * xi_line ** float(z_fit["exponent"]), "--", label=rf"primary fit $z={float(z_fit['exponent']):.5f}$")
    anchor_index = int(np.argmin(lengths["delta"].to_numpy()))
    ax.loglog(xi_line, _reference_powerlaw(xi_line, float(lengths["xi_dyn_raw"].iloc[anchor_index]), float(tau_values[anchor_index]), 2.0), ":", label="slope z=2")
    ax.set(xlabel=r"$\xi_{dyn,raw}$", ylabel=r"$\tau_0$", title="Phase4: dynamic-scaling consistency")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase4_tau_vs_xi"] = path

    path = output_dir / "phase4_data_collapse.png"
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    for delta_value, group in collapse.groupby("delta", sort=False):
        used = group[group["longwave_used"]]
        outside = group[~group["longwave_used"]]
        line = ax.plot(used["collapse_x_qxi"], used["collapse_y_tau_ratio"], "o", label=rf"$\delta={delta_value:g}$")[0]
        if not outside.empty:
            ax.plot(outside["collapse_x_qxi"], outside["collapse_y_tau_ratio"], "o", mfc="none", mec=line.get_color(), alpha=0.65)
    xmax = max(float(collapse["collapse_x_qxi"].max()), 1e-6)
    theory_x = np.linspace(0.0, xmax, 500)
    ax.plot(theory_x, 1.0 / (1.0 + theory_x**2), "k--", lw=1.8, label=r"$1/(1+x^2)$")
    ax.set(xlabel=r"$q\xi_{dyn}$", ylabel=r"$\tau(q)/\tau_0$", title="Phase4: finite-q data collapse")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase4_data_collapse"] = path

    path = output_dir / "phase4_collapse_residual.png"
    fig, ax = plt.subplots(figsize=(7.4, 5.0), constrained_layout=True)
    for delta_value, group in collapse.groupby("delta", sort=False):
        used = group[group["longwave_used"]]
        ax.plot(used["collapse_x_qxi"], used["collapse_residual"], "o", label=rf"$\delta={delta_value:g}$")
    ax.axhline(0.0, color="k", linestyle="--")
    ax.set(xlabel=r"$q\xi_{dyn}$", ylabel="collapse residual", title="Phase4: long-wave collapse residual")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=220)
    plt.close(fig)
    paths["phase4_collapse_residual"] = path
    return paths


def write_phase34_outputs(
    analysis: Phase34Analysis, output_dir: Path, runtime: dict[str, float] | None = None
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "phase3_scaling_table": output_dir / "phase3_scaling_table.csv",
        "phase3_powerlaw_fits": output_dir / "phase3_powerlaw_fits.csv",
        "phase3_window_stability": output_dir / "phase3_window_stability.csv",
        "phase3_effective_exponents": output_dir / "phase3_effective_exponents.csv",
        "phase4_length_table": output_dir / "phase4_length_table.csv",
        "phase4_scaling_fits": output_dir / "phase4_scaling_fits.csv",
        "phase4_dispersion_systematics": output_dir
        / "phase4_dispersion_systematics.csv",
        "phase4_collapse": output_dir / "phase4_collapse.csv",
        "validation_summary": output_dir / "phase34_validation_summary.json",
    }
    analysis.phase3_scaling.to_csv(paths["phase3_scaling_table"], index=False)
    analysis.phase3_powerlaw_fits.to_csv(paths["phase3_powerlaw_fits"], index=False)
    analysis.phase3_window_stability.to_csv(
        paths["phase3_window_stability"], index=False
    )
    analysis.phase3_effective_exponents.to_csv(
        paths["phase3_effective_exponents"], index=False
    )
    analysis.phase4_lengths.to_csv(paths["phase4_length_table"], index=False)
    analysis.phase4_scaling_fits.to_csv(paths["phase4_scaling_fits"], index=False)
    analysis.phase4_systematics.to_csv(
        paths["phase4_dispersion_systematics"], index=False
    )
    analysis.phase4_collapse.to_csv(paths["phase4_collapse"], index=False)

    figure_paths = make_phase34_figures(analysis, output_dir)
    summary = dict(analysis.validation_summary)
    if runtime is not None:
        summary["runtime"] = runtime
    paths["validation_summary"].write_text(
        json.dumps(_json_safe(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths.update(figure_paths)
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Serial post-processing of existing deterministic Phase1-2 outputs "
            "for spinodal Phase3-4 scaling. No simulation or MPI is performed."
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
        "--phase12-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "phase12_B2_R12",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "phase34_B2_R12",
    )
    parser.add_argument("--primary-delta-max", type=float, default=3e-4)
    parser.add_argument("--qR-max-collapse", type=float, default=0.35)
    parser.add_argument("--min-window-points", type=int, default=3)
    parser.add_argument("--include-unreliable", action="store_true")
    parser.add_argument("--disable-q4-systematics", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    total_start = time.perf_counter()
    try:
        inputs = load_phase34_inputs(args.phase0_dir, args.phase12_dir)
        analysis_start = time.perf_counter()
        analysis = run_phase34_analysis(
            inputs,
            primary_delta_max=args.primary_delta_max,
            qR_max_collapse=args.qR_max_collapse,
            min_window_points=args.min_window_points,
            include_unreliable=args.include_unreliable,
            disable_q4_systematics=args.disable_q4_systematics,
        )
        analysis_seconds = time.perf_counter() - analysis_start
        output_start = time.perf_counter()
        paths = write_phase34_outputs(
            analysis,
            args.output_dir,
            runtime={
                "analysis_seconds": analysis_seconds,
                "output_seconds": 0.0,
                "total_wall_seconds": 0.0,
                "mpi_used": False,
            },
        )
        output_seconds = time.perf_counter() - output_start
        total_seconds = time.perf_counter() - total_start
        summary_path = paths["validation_summary"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["runtime"]["output_seconds"] = output_seconds
        summary["runtime"]["total_wall_seconds"] = total_seconds
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    if analysis.validation_summary["linearity_warning"] is not None:
        print(EPSILON_WARNING)
    phase3 = analysis.validation_summary["phase3"]
    phase4 = analysis.validation_summary["phase4"]
    collapse = analysis.validation_summary["collapse"]
    print("=== Spinodal Phase3-4 serial post-processing ===")
    print(
        f"Gamma primary exponent={phase3['gamma_exponent_primary']:.8f}, "
        f"all-points={phase3['gamma_exponent_all_points']:.8f}"
    )
    print(
        f"xi_raw primary exponent={phase4['xi_exponent_raw_primary']:.8f}, "
        f"z consistency={phase4['z_raw_primary']:.8f}"
    )
    print(
        f"collapse RMS={collapse['rms_error']:.6e}, "
        f"max_abs={collapse['max_absolute_error']:.6e}"
    )
    print(f"completed in {total_seconds:.3f} s (MPI not used)")
    print("=== Outputs ===")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
