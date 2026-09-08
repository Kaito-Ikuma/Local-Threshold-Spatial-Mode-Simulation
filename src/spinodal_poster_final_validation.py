#!/usr/bin/env python3
"""Poster-final validation from existing Phase5/Phase6 CSV products.

This is post-processing only.  It diagnoses whether a dedicated microscopic
Result-B refinement is needed, tests Phase6 epsilon and fit-window robustness,
combines the independently fitted real-space length with q=0 relaxation, and
separates boundary injection from bulk decay.  It never launches a simulation.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import LogNorm
from matplotlib.ticker import ScalarFormatter
from scipy.stats import t as student_t

from spinodal_poster_abcd import fit_measured_dispersion, load_pseudospinodal
from spinodal_phase0 import kappa_R_theory
from spinodal_phase34 import coefficient_of_determination, fit_power_law


SCRIPT_VERSION = "2026.09.08-poster-final-validation-v3"
BOUNDARY_TYPES = (
    "ghost_dirichlet",
    "open_fixed_denominator",
    "open_renormalized",
)
PRIMARY_EPSILON = 0.05
PRIMARY_DELTA_MAX = 3e-4
PRIMARY_QR_MAX = 0.35
FULL_NUMERIC_R_VALUES = (6, 12, 24, 48, 96)
FULL_NUMERIC_DELTAS = (1e-5, 3e-5, 1e-4, 3e-4)
FULL_NUMERIC_GAMMA_SOURCE = "deterministic q=0 numerical"


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    invalid = sorted(set(normalized) - set(mapping))
    if invalid:
        raise ValueError(f"invalid boolean values: {invalid}")
    return normalized.map(mapping).astype(bool)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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


def fit_dynamic_exponent(
    xi: Sequence[float], tau: Sequence[float]
) -> dict[str, float | int]:
    """Fit tau=A*xi**z and report a regression uncertainty interval."""
    fit = fit_power_law(xi, tau)
    z = float(fit["exponent"])
    se = float(fit["exponent_regression_se"])
    n = int(fit["n_points"])
    if n > 2 and math.isfinite(se):
        half_width = float(student_t.ppf(0.975, n - 2)) * se
        low, high = z - half_width, z + half_width
    else:
        low = high = math.nan
    return {
        "amplitude": float(fit["amplitude"]),
        "z": z,
        "z_se": se,
        "z_ci_low": low,
        "z_ci_high": high,
        "r2": float(fit["r2"]),
        "n_points": n,
    }


def build_epsilon_robustness(
    xi_table: pd.DataFrame,
    *,
    boundary_type: str = "ghost_dirichlet",
    reference_epsilon: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    """Normalize independently fitted boundary lengths by epsilon=0.05."""
    required = [
        "R", "delta", "epsilon_fraction", "boundary_type", "xi_bnd_cosh",
        "fit_reliable", "converged",
    ]
    _require_columns(xi_table, required, "resultC_xi_boundary.csv")
    frame = xi_table.copy()
    frame["fit_reliable"] = _coerce_bool(frame["fit_reliable"])
    frame["converged"] = _coerce_bool(frame["converged"])
    frame = frame[frame["boundary_type"] == boundary_type].copy()
    reference = frame[
        np.isclose(frame["epsilon_fraction"], reference_epsilon, rtol=0.0, atol=1e-12)
    ][["R", "delta", "xi_bnd_cosh"]].rename(
        columns={"xi_bnd_cosh": "xi_reference_eps005"}
    )
    result = frame.merge(reference, on=["R", "delta"], how="left", validate="many_to_one")
    result["relative_ratio"] = result["xi_bnd_cosh"] / result["xi_reference_eps005"]
    result["relative_difference"] = result["relative_ratio"] - 1.0
    return result.rename(columns={"xi_bnd_cosh": "xi_bnd"})[
        [
            "R", "delta", "epsilon_fraction", "xi_bnd",
            "xi_reference_eps005", "relative_ratio", "relative_difference",
            "fit_reliable", "converged",
        ]
    ].sort_values(["R", "delta", "epsilon_fraction"]).reset_index(drop=True)


def build_fitwindow_robustness(
    systematics: pd.DataFrame,
    xi_table: pd.DataFrame,
    *,
    boundary_type: str = "ghost_dirichlet",
    epsilon_fraction: float = PRIMARY_EPSILON,
) -> pd.DataFrame:
    """Normalize x_min=R,2R,3R fits by the fixed primary x_min=2R fit."""
    _require_columns(
        systematics,
        ["R", "delta", "epsilon_fraction", "boundary_type", "fit_x_min_factor_R", "xi_cosh", "r2", "rmse"],
        "resultC_boundary_fit_systematics.csv",
    )
    frame = systematics[
        (systematics["boundary_type"] == boundary_type)
        & np.isclose(systematics["epsilon_fraction"], epsilon_fraction, rtol=0.0, atol=1e-12)
    ].copy()
    reference = frame[np.isclose(frame["fit_x_min_factor_R"], 2.0)][
        ["R", "delta", "xi_cosh"]
    ].rename(columns={"xi_cosh": "xi_reference_2R"})
    frame = frame.merge(reference, on=["R", "delta"], how="left", validate="many_to_one")

    reliability = xi_table.copy()
    reliability["fit_reliable"] = _coerce_bool(reliability["fit_reliable"])
    reliability["converged"] = _coerce_bool(reliability["converged"])
    reliability = reliability[
        (reliability["boundary_type"] == boundary_type)
        & np.isclose(reliability["epsilon_fraction"], epsilon_fraction, rtol=0.0, atol=1e-12)
    ][["R", "delta", "fit_reliable", "converged"]]
    frame = frame.merge(reliability, on=["R", "delta"], how="left", validate="many_to_one")
    frame["relative_ratio"] = frame["xi_cosh"] / frame["xi_reference_2R"]
    return frame.rename(columns={"xi_cosh": "xi_bnd"})[
        [
            "R", "delta", "fit_x_min_factor_R", "xi_bnd",
            "xi_reference_2R", "relative_ratio", "r2", "rmse",
            "fit_reliable", "converged",
        ]
    ].sort_values(["R", "delta", "fit_x_min_factor_R"]).reset_index(drop=True)


def bulk_normalized_profile(
    x: Sequence[float], response: Sequence[float], *, R: int, xi_bulk: float
) -> pd.DataFrame:
    """Return X=(x-2R)/xi and response/response(2R) for x>=2R."""
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(response, dtype=float)
    if x_array.shape != y_array.shape or x_array.ndim != 1:
        raise ValueError("x and response must be equal-length one-dimensional arrays")
    if R < 1 or not math.isfinite(xi_bulk) or xi_bulk <= 0.0:
        raise ValueError("R and xi_bulk must be positive")
    index = int(np.argmin(np.abs(x_array - 2.0 * R)))
    if not math.isclose(float(x_array[index]), 2.0 * R, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("profile does not contain x=2R")
    amplitude = float(y_array[index])
    if not math.isfinite(amplitude) or abs(amplitude) <= np.finfo(float).tiny:
        raise ValueError("response at x=2R is zero or non-finite")
    mask = x_array >= 2.0 * R - 1e-12
    X = (x_array[mask] - 2.0 * R) / xi_bulk
    normalized = y_array[mask] / amplitude
    return pd.DataFrame(
        {"x": x_array[mask], "X_bulk": X, "normalized_response": normalized,
         "exp_reference": np.exp(-X)}
    )


def _load_combined_micro_modes(
    base_path: Path,
    refinement_root: Path,
    pseudospinodal_root: Path,
    *,
    T_obs: int,
    epsilon_fraction: float,
) -> pd.DataFrame:
    paths: list[tuple[Path, str]] = []
    if base_path.is_file():
        paths.append((base_path, "poster_AB_existing"))
    for path in sorted(refinement_root.glob("R*/phase5_mode_results.csv")):
        paths.append((path, "poster_B_refinement"))
    if not paths:
        raise FileNotFoundError("no Result B microscopic mode table was found")

    frames: list[pd.DataFrame] = []
    for path, source in paths:
        frame = pd.read_csv(path)
        _require_columns(
            frame,
            ["R", "delta", "mode_index", "q", "qR", "M_total", "epsilon_fraction",
             "Gamma_micro", "Gamma_micro_se", "reliable", "escape_fraction",
             "preparation_drift", "baseline_drift", "method_B_C_relative_difference"],
            str(path),
        )
        frame = frame[np.isclose(frame["epsilon_fraction"], epsilon_fraction, rtol=0.0, atol=1e-12)].copy()
        frame["reliable"] = _coerce_bool(frame["reliable"])
        frame["data_source"] = source
        for R in frame["R"].unique():
            time_path = (
                pseudospinodal_root / f"R{int(R):03d}" / "pseudospinodal_fine"
                / "analysis" / "phase5_pseudospinodal_time_dependence.csv"
            )
            pseudo = load_pseudospinodal(time_path, T_obs=T_obs)
            mask = frame["R"] == R
            frame.loc[mask, "delta_ps_T50"] = pseudo["delta_ps"]
            frame.loc[mask, "matched_distance"] = frame.loc[mask, "delta"] - pseudo["delta_ps"]
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    combined["matched_key"] = combined["matched_distance"].round(10)
    combined = combined.sort_values(["M_total", "data_source"])
    combined = combined.drop_duplicates(
        ["R", "matched_key", "mode_index", "epsilon_fraction"], keep="last"
    )
    return combined.drop(columns="matched_key").sort_values(
        ["R", "matched_distance", "mode_index"]
    ).reset_index(drop=True)


def build_result_b_precision(
    modes: pd.DataFrame, *, qR_max: float = PRIMARY_QR_MAX
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute measured dispersion and precision diagnostics per (R,s)."""
    rows: list[dict[str, Any]] = []
    collapse_rows: list[dict[str, Any]] = []
    for (R, delta), group in modes.groupby(["R", "delta"]):
        in_window = group[group["qR"] <= qR_max * (1.0 + 1e-12)].sort_values("mode_index")
        eligible = in_window[
            in_window["reliable"] & np.isfinite(in_window["Gamma_micro"])
            & (in_window["Gamma_micro"] > 0.0)
        ]
        q0 = eligible[eligible["mode_index"] == 0]
        if len(eligible) < 2 or len(q0) != 1:
            continue
        fit = fit_measured_dispersion(
            eligible["q"], eligible["Gamma_micro"], eligible["Gamma_micro_se"]
        )
        gamma0 = float(q0["Gamma_micro"].iloc[0])
        D = float(fit["D"])
        D_se = float(fit["D_se"])
        xi = math.sqrt(D / gamma0) if D > 0.0 and gamma0 > 0.0 else math.nan
        X = eligible["q"].to_numpy(float) * xi if math.isfinite(xi) else np.full(len(eligible), math.nan)
        Y = gamma0 / eligible["Gamma_micro"].to_numpy(float)
        theory = 1.0 / (1.0 + X**2)
        residual = Y - theory
        condition_reliable = bool(
            in_window["reliable"].all() and math.isfinite(xi) and xi > 0.0
        )
        D_over_se = D / D_se if math.isfinite(D_se) and D_se > 0.0 else math.nan
        x_max = float(np.nanmax(X)) if np.any(np.isfinite(X)) else math.nan
        escape = float(in_window["escape_fraction"].max())
        rows.append(
            {
                "R": int(R), "N": int(q0["N"].iloc[0]), "delta": float(delta),
                "matched_distance": float(q0["matched_distance"].iloc[0]),
                "M": int(in_window["M_total"].max()), "X_max": x_max,
                "D_micro": D, "sigma_D": D_se, "D_over_sigma_D": D_over_se,
                "dispersion_r2": float(fit["r2"]), "number_of_modes": int(len(eligible)),
                "qR_max": float(eligible["qR"].max()),
                "collapse_RMSE": float(np.sqrt(np.mean(residual**2))),
                "maximum_residual": float(np.max(np.abs(residual))),
                "cumulative_escape_fraction": escape, "survival_fraction": 1.0 - escape,
                "max_absolute_preparation_drift": float(in_window["preparation_drift"].abs().max()),
                "max_baseline_drift": float(in_window["baseline_drift"].max()),
                "max_method_B_C_relative_difference": float(in_window["method_B_C_relative_difference"].max()),
                "all_primary_modes_reliable": bool(in_window["reliable"].all()),
                "condition_reliable": condition_reliable,
                "near_pseudospinodal_rounded_regime": escape > 0.2,
                "precision_target_met": bool(condition_reliable and x_max >= 0.4 and D_over_se >= 2.0),
                "data_source": "+".join(sorted(set(in_window["data_source"].astype(str)))),
            }
        )
        for (_, mode), x_value, y_value, theory_value, residual_value in zip(
            eligible.iterrows(), X, Y, theory, residual
        ):
            collapse_rows.append(
                {
                    "R": int(R), "delta": float(delta),
                    "matched_distance": float(mode["matched_distance"]),
                    "M": int(mode["M_total"]), "mode_index": int(mode["mode_index"]),
                    "q": float(mode["q"]), "qR": float(mode["qR"]),
                    "Gamma_micro": float(mode["Gamma_micro"]),
                    "Gamma_micro_se": float(mode["Gamma_micro_se"]),
                    "Gamma0_micro": gamma0, "D_micro": D, "xi_dyn_micro": xi,
                    "X_q_xi": float(x_value), "Y_tau_ratio": float(y_value),
                    "Y_theory": float(theory_value), "residual": float(residual_value),
                    "condition_reliable": condition_reliable,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(collapse_rows)


def _numerical_gamma_lookup(phase12_dir: Path) -> pd.DataFrame:
    path = phase12_dir / "phase12_mode_results.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["R", "delta", "Gamma0", "Gamma0_source"])
    frame = pd.read_csv(path)
    _require_columns(frame, ["R", "delta", "mode_index", "task_group", "Gamma_from_lambda", "reliable"], str(path))
    frame["reliable"] = _coerce_bool(frame["reliable"])
    frame = frame[(frame["mode_index"] == 0) & (frame["task_group"] == "main") & frame["reliable"]].copy()
    frame["Gamma0_source"] = "deterministic q=0 numerical Gamma_from_lambda"
    return frame.rename(columns={"Gamma_from_lambda": "Gamma0"})[
        ["R", "delta", "Gamma0", "Gamma0_source"]
    ]


def build_dynamic_z_tables(
    xi_table: pd.DataFrame,
    phase12_dir: Path,
    *,
    primary_delta_max: float = PRIMARY_DELTA_MAX,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine independent xi_bnd with q=0 rate, preferring numerical rates."""
    frame = xi_table.copy()
    for column in ("primary_epsilon", "fit_reliable", "converged"):
        frame[column] = _coerce_bool(frame[column])
    frame = frame[
        frame["primary_epsilon"] & frame["fit_reliable"] & frame["converged"]
        & (frame["boundary_type"] == "ghost_dirichlet") & (frame["xi_bnd_cosh"] > 0.0)
    ].copy()
    numerical = _numerical_gamma_lookup(phase12_dir)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        match = numerical[
            (numerical["R"] == int(row["R"]))
            & np.isclose(numerical["delta"], float(row["delta"]), rtol=1e-10, atol=1e-15)
        ]
        if len(match) == 1:
            gamma0 = float(match["Gamma0"].iloc[0])
            source = str(match["Gamma0_source"].iloc[0])
        else:
            R = int(row["R"])
            kappa = (R + 1) * (2 * R + 1) / 12.0
            gamma0 = kappa / float(row["xi_theory"]) ** 2
            source = "Phase0 theory reconstructed from stored xi_theory and kappa_R"
        rows.append(
            {
                "R": int(row["R"]), "delta": float(row["delta"]),
                "Gamma0": gamma0, "Gamma0_source": source, "tau0": 1.0 / gamma0,
                "xi_bnd": float(row["xi_bnd_cosh"]), "fit_reliable": True,
            }
        )
    points = pd.DataFrame(rows).sort_values(["R", "delta"]).reset_index(drop=True)
    summaries: list[dict[str, Any]] = []
    for R, group in points.groupby("R"):
        primary = group[group["delta"] <= primary_delta_max * (1.0 + 1e-12)]
        if len(primary) < 2:
            continue
        fit = fit_dynamic_exponent(primary["xi_bnd"], primary["tau0"])
        summaries.append(
            {
                "R": int(R), "delta_window": f"delta<={primary_delta_max:g} (fixed)",
                "n_points": int(fit["n_points"]), "z": float(fit["z"]),
                "z_se": float(fit["z_se"]), "z_ci_low": float(fit["z_ci_low"]),
                "z_ci_high": float(fit["z_ci_high"]), "r2": float(fit["r2"]),
                "expected_z": 2.0, "z_minus_2": float(fit["z"]) - 2.0,
            }
        )
    return points, pd.DataFrame(summaries)


def build_fully_numeric_dynamic_z_tables(
    xi_table: pd.DataFrame,
    q0_numeric: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join numerical q=0 rates to independent boundary lengths without fallback."""
    q0_required = [
        "R", "N", "delta", "epsilon_fraction", "Gamma0_num",
        "Gamma0_source", "reliable",
    ]
    _require_columns(q0_numeric, q0_required, "q0_numeric_all_5R.csv")
    q0 = q0_numeric.copy()
    q0["reliable"] = _coerce_bool(q0["reliable"])
    invalid_sources = q0[q0["Gamma0_source"] != FULL_NUMERIC_GAMMA_SOURCE]
    if not invalid_sources.empty:
        values = sorted(set(invalid_sources["Gamma0_source"].astype(str)))
        raise ValueError(
            "fully numerical z analysis rejects non-numerical Gamma0 sources: "
            f"{values}"
        )
    q0 = q0[
        q0["R"].isin(FULL_NUMERIC_R_VALUES)
        & (q0["delta"] <= PRIMARY_DELTA_MAX * (1.0 + 1e-12))
        & np.isclose(q0["epsilon_fraction"], PRIMARY_EPSILON, rtol=0.0, atol=1e-12)
    ].copy()
    if q0.duplicated(["R", "delta"]).any():
        raise ValueError("q0_numeric_all_5R.csv has duplicate (R, delta) rows")

    xi_required = [
        "R", "N", "delta", "epsilon_fraction", "boundary_type", "xi_bnd_cosh",
        "fit_x_min", "fit_reliable", "converged",
    ]
    _require_columns(xi_table, xi_required, "resultC_xi_boundary.csv")
    xi = xi_table.copy()
    xi["fit_reliable"] = _coerce_bool(xi["fit_reliable"])
    xi["converged"] = _coerce_bool(xi["converged"])
    xi = xi[
        xi["R"].isin(FULL_NUMERIC_R_VALUES)
        & (xi["boundary_type"] == "ghost_dirichlet")
        & np.isclose(xi["epsilon_fraction"], PRIMARY_EPSILON, rtol=0.0, atol=1e-12)
        & (xi["delta"] <= PRIMARY_DELTA_MAX * (1.0 + 1e-12))
        & np.isclose(xi["fit_x_min"], 2.0 * xi["R"], rtol=0.0, atol=1e-9)
    ][
        ["R", "N", "delta", "epsilon_fraction", "xi_bnd_cosh", "fit_x_min",
         "fit_reliable", "converged"]
    ].copy()
    if xi.duplicated(["R", "delta"]).any():
        raise ValueError("primary Phase6 boundary data has duplicate (R, delta) rows")

    joined = q0.merge(
        xi, on=["R", "N", "delta", "epsilon_fraction"],
        how="left", validate="one_to_one"
    )
    joined["tau0_num"] = 1.0 / joined["Gamma0_num"]
    joined["Gamma0_reliable"] = joined["reliable"]
    joined["xi_bnd"] = joined["xi_bnd_cosh"]
    joined["fit_reliable"] = joined["fit_reliable"].fillna(False).astype(bool)
    joined["converged"] = joined["converged"].fillna(False).astype(bool)
    joined["included"] = (
        joined["Gamma0_reliable"]
        & joined["fit_reliable"]
        & joined["converged"]
        & np.isfinite(joined["Gamma0_num"])
        & (joined["Gamma0_num"] > 0.0)
        & np.isfinite(joined["xi_bnd"])
        & (joined["xi_bnd"] > 0.0)
    )

    def exclusion_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if not bool(row["Gamma0_reliable"]):
            reasons.append("Gamma0_unreliable")
        if not bool(row["fit_reliable"]):
            reasons.append("boundary_fit_unreliable_or_missing")
        if not bool(row["converged"]):
            reasons.append("boundary_not_converged_or_missing")
        if not math.isfinite(float(row["Gamma0_num"])) or float(row["Gamma0_num"]) <= 0.0:
            reasons.append("Gamma0_nonpositive_or_nonfinite")
        xi_value = float(row["xi_bnd"]) if pd.notna(row["xi_bnd"]) else math.nan
        if not math.isfinite(xi_value) or xi_value <= 0.0:
            reasons.append("xi_bnd_nonpositive_or_missing")
        return "ok" if not reasons else ";".join(reasons)

    joined["exclusion_reason"] = joined.apply(exclusion_reason, axis=1)
    points_columns = [
        "R", "N", "delta", "Gamma0_num", "Gamma0_source", "tau0_num", "xi_bnd",
        "epsilon_fraction", "fit_x_min", "fit_reliable", "converged", "Gamma0_reliable",
        "included", "exclusion_reason",
    ]
    points = joined[points_columns].sort_values(["R", "delta"]).reset_index(drop=True)

    summaries: list[dict[str, Any]] = []
    for R in FULL_NUMERIC_R_VALUES:
        group = points[(points["R"] == R) & points["included"]]
        if len(group) < 2:
            continue
        fit = fit_dynamic_exponent(group["xi_bnd"], group["tau0_num"])
        summaries.append(
            {
                "R": R,
                "n_points": int(fit["n_points"]),
                "delta_min": float(group["delta"].min()),
                "delta_max": float(group["delta"].max()),
                "z": float(fit["z"]),
                "z_se": float(fit["z_se"]),
                "z_ci_low": float(fit["z_ci_low"]),
                "z_ci_high": float(fit["z_ci_high"]),
                "r2": float(fit["r2"]),
                "expected_z": 2.0,
                "z_minus_2": float(fit["z"]) - 2.0,
            }
        )
    summary_columns = [
        "R", "n_points", "delta_min", "delta_max", "z", "z_se",
        "z_ci_low", "z_ci_high", "r2", "expected_z", "z_minus_2",
    ]
    return points, pd.DataFrame(summaries, columns=summary_columns)


def infer_q0_lattice_spacing(q0_numeric: pd.DataFrame) -> float:
    """Read ``a`` from the Phase12 numerical sources and require one value."""
    if "a" in q0_numeric.columns:
        values = q0_numeric["a"].dropna().to_numpy(dtype=float)
    else:
        _require_columns(q0_numeric, ["source_file"], "q0_numeric_all_5R.csv")
        collected: list[float] = []
        for source_text in sorted(set(q0_numeric["source_file"].astype(str))):
            source = Path(source_text)
            if not source.is_absolute():
                source = Path.cwd() / source
            if not source.is_file():
                raise FileNotFoundError(f"cannot read lattice spacing from {source}")
            source_table = pd.read_csv(source, usecols=["a"])
            collected.extend(source_table["a"].dropna().to_numpy(dtype=float).tolist())
        values = np.asarray(collected, dtype=float)
    unique = np.unique(np.round(values, decimals=14))
    if len(unique) != 1 or not math.isfinite(float(unique[0])) or float(unique[0]) <= 0.0:
        raise ValueError(f"expected one positive finite lattice spacing, found {unique.tolist()}")
    return float(unique[0])


def _dynamic_fit_record(frame: pd.DataFrame) -> dict[str, float | int]:
    fit = fit_dynamic_exponent(frame["xi_scaled"], frame["tau0_num"])
    return {
        "n_points": int(fit["n_points"]),
        "A": float(fit["amplitude"]),
        "z": float(fit["z"]),
        "z_standard_error": float(fit["z_se"]),
        "z_ci_low": float(fit["z_ci_low"]),
        "z_ci_high": float(fit["z_ci_high"]),
        "r2": float(fit["r2"]),
    }


def build_fully_numeric_combined_analysis(
    points: pd.DataFrame,
    *,
    lattice_spacing: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove the known kappa_R prefactor and fit all reliable points together."""
    required = [
        "R", "N", "delta", "Gamma0_num", "Gamma0_source", "tau0_num", "xi_bnd",
        "epsilon_fraction", "fit_reliable", "converged", "Gamma0_reliable", "included",
    ]
    _require_columns(points, required, "resultC_dynamic_z_points_fully_numeric.csv")
    if not math.isfinite(lattice_spacing) or lattice_spacing <= 0.0:
        raise ValueError("lattice spacing must be positive and finite")
    frame = points.copy()
    for column in ("fit_reliable", "converged", "Gamma0_reliable", "included"):
        frame[column] = _coerce_bool(frame[column])
    invalid_sources = frame[frame["Gamma0_source"] != FULL_NUMERIC_GAMMA_SOURCE]
    if not invalid_sources.empty:
        raise ValueError("combined collapse rejects non-numerical Gamma0 sources")
    frame = frame[
        frame["R"].isin(FULL_NUMERIC_R_VALUES)
        & frame["included"]
        & (frame["delta"] <= PRIMARY_DELTA_MAX * (1.0 + 1e-12))
    ].copy()
    frame["kappa_R"] = [
        kappa_R_theory(int(R), lattice_spacing) for R in frame["R"]
    ]
    frame["xi_scaled"] = frame["xi_bnd"] / np.sqrt(frame["kappa_R"])
    frame["collapse_ratio"] = (
        frame["kappa_R"] * frame["tau0_num"] / frame["xi_bnd"] ** 2
    )
    positive_columns = ["Gamma0_num", "tau0_num", "xi_bnd", "kappa_R", "xi_scaled", "collapse_ratio"]
    valid = np.ones(len(frame), dtype=bool)
    for column in positive_columns:
        valid &= np.isfinite(frame[column]) & (frame[column] > 0.0)
    frame = frame[valid].copy()
    combined = frame.rename(
        columns={
            "epsilon_fraction": "epsilon",
            "Gamma0_reliable": "gamma_reliable",
            "fit_reliable": "boundary_fit_reliable",
        }
    )[
        [
            "R", "N", "delta", "epsilon", "Gamma0_num", "tau0_num", "xi_bnd",
            "kappa_R", "xi_scaled", "collapse_ratio", "Gamma0_source",
            "gamma_reliable", "boundary_fit_reliable", "converged",
        ]
    ].sort_values(["R", "delta"]).reset_index(drop=True)
    if len(combined) < 2:
        raise ValueError("combined collapse requires at least two reliable finite points")

    expected_pairs = {
        (R, f"{delta:.12g}") for R in FULL_NUMERIC_R_VALUES for delta in FULL_NUMERIC_DELTAS
    }
    actual_pairs = {
        (int(row.R), f"{float(row.delta):.12g}") for row in combined.itertuples()
    }
    all_fit = _dynamic_fit_record(combined)
    per_R_smallest_3: list[dict[str, Any]] = []
    per_R_smallest_2: list[dict[str, Any]] = []
    smallest_2_frames: list[pd.DataFrame] = []
    smallest_3_frames: list[pd.DataFrame] = []
    for R in FULL_NUMERIC_R_VALUES:
        group = combined[combined["R"] == R].nsmallest(3, "delta")
        if len(group) >= 3:
            per_R_smallest_3.append({"R": R, **_dynamic_fit_record(group)})
            smallest_3_frames.append(group)
        pair = group.nsmallest(2, "delta")
        if len(pair) >= 2:
            per_R_smallest_2.append({"R": R, **_dynamic_fit_record(pair)})
            smallest_2_frames.append(pair)
    smallest_2 = pd.concat(smallest_2_frames, ignore_index=True)
    smallest_3 = pd.concat(smallest_3_frames, ignore_index=True)
    ratios = combined["collapse_ratio"].to_numpy(dtype=float)
    summary: dict[str, Any] = {
        "git_sha": _git_sha(),
        "script_version": SCRIPT_VERSION,
        "selected_R_values": list(FULL_NUMERIC_R_VALUES),
        "selected_delta_values": list(FULL_NUMERIC_DELTAS),
        "number_of_points": int(len(combined)),
        "Gamma0_source": FULL_NUMERIC_GAMMA_SOURCE,
        "Gamma0_definition": "Gamma0_num = Gamma_from_lambda; tau0_num = 1/Gamma0_num",
        "spatial_observable": "boundary-induced xi_bnd from the same deterministic Gaussian closure",
        "lattice_spacing_a": float(lattice_spacing),
        "kappa_definition": "kappa_R = a^2 (R+1)(2R+1)/12",
        "xi_scaled_definition": "xi_scaled = xi_bnd/sqrt(kappa_R)",
        "combined_fit_all_20": all_fit,
        "per_R_smallest_3_delta_fits": per_R_smallest_3,
        "per_R_smallest_2_delta_fits": per_R_smallest_2,
        "combined_smallest_3_delta_fit": _dynamic_fit_record(smallest_3),
        "combined_smallest_2_delta_fit": _dynamic_fit_record(smallest_2),
        "collapse_ratio_definition": "kappa_R*tau0_num/xi_bnd^2",
        "collapse_ratio_statistics": {
            "mean": float(np.mean(ratios)),
            "min": float(np.min(ratios)),
            "max": float(np.max(ratios)),
            "standard_deviation_population": float(np.std(ratios, ddof=0)),
            "maximum_absolute_deviation_from_1": float(np.max(np.abs(ratios - 1.0))),
        },
        "all_20_points_present": actual_pairs == expected_pairs,
        "all_20_points_passed_reliability_criteria": bool(
            actual_pairs == expected_pairs
            and combined["gamma_reliable"].all()
            and combined["boundary_fit_reliable"].all()
            and combined["converged"].all()
        ),
        "regression_uncertainty_note": (
            "The 95% interval measures residual scatter about a pure power law in "
            "deterministic data; it is not stochastic sampling uncertainty."
        ),
        "scientific_wording": {
            "main": "Fully numerical normalized space-time collapse",
            "protocols": "q=0 temporal relaxation and boundary-induced spatial length",
            "normalization": "R-dependent kernel prefactor removed by xi_bnd/sqrt(kappa_R)",
            "interpretation": "z_eff approaches 2 toward the spinodal",
            "raw_caption": "Raw coordinates retain the R-dependent prefactor 1/kappa_R.",
        },
    }
    return combined, summary


def build_boundary_transmission(
    profiles: pd.DataFrame, xi_table: pd.DataFrame
) -> pd.DataFrame:
    frame = xi_table.copy()
    for column in ("primary_epsilon", "converged", "fit_reliable"):
        frame[column] = _coerce_bool(frame[column])
    frame = frame[frame["primary_epsilon"]]
    rows: list[dict[str, Any]] = []
    for _, fit in frame.iterrows():
        group = profiles[
            (profiles["R"] == int(fit["R"]))
            & np.isclose(profiles["delta"], float(fit["delta"]), rtol=0.0, atol=1e-15)
            & np.isclose(profiles["epsilon_fraction"], PRIMARY_EPSILON, rtol=0.0, atol=1e-12)
            & (profiles["boundary_type"] == fit["boundary_type"])
        ]
        at_zero = group[np.isclose(group["x"], 0.0)]
        at_bulk = group[np.isclose(group["x"], 2.0 * int(fit["R"]))]
        if len(at_zero) != 1 or len(at_bulk) != 1:
            continue
        denominator = float(at_zero["delta_u"].iloc[0])
        transmission = float(at_bulk["delta_u"].iloc[0]) / denominator if abs(denominator) > 0.0 else math.nan
        rows.append(
            {
                "boundary_type": str(fit["boundary_type"]), "R": int(fit["R"]),
                "delta": float(fit["delta"]), "T_bnd": transmission,
                "xi_bulk": float(fit["xi_bnd_exp"]), "fit_r2": float(fit["exp_fit_r2"]),
                "fit_rmse": float(fit["exp_fit_rmse"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["R", "delta", "boundary_type"]).reset_index(drop=True)


def _select_representative_conditions(table: pd.DataFrame) -> pd.DataFrame:
    candidates = table[
        table["R"].isin([12, 24, 48]) & table["delta"].isin([1e-4, 3e-4])
        & table["fit_reliable"] & table["converged"]
    ]
    selected = []
    for R in (12, 24, 48):
        group = candidates[candidates["R"] == R]
        for delta in (1e-4, 3e-4):
            row = group[np.isclose(group["delta"], delta, rtol=0.0, atol=1e-15)]
            if not row.empty:
                selected.append(row.iloc[0])
                break
    return pd.DataFrame(selected)


def _save_figure(fig: plt.Figure, path: Path, *, dpi: int = 240) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def make_result_b_figures(
    figures_dir: Path, diagnostics: pd.DataFrame, collapse: pd.DataFrame
) -> list[Path]:
    paths: list[Path] = []
    valid = collapse[collapse["condition_reliable"]].copy()
    fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
    markers = {24: "o", 48: "s"}
    if not valid.empty:
        norm = LogNorm(
            vmin=float(valid["matched_distance"].min()),
            vmax=float(valid["matched_distance"].max()),
        )
        for (R, s), group in valid.groupby(["R", "matched_distance"]):
            ax.scatter(
                group["X_q_xi"], group["Y_tau_ratio"],
                marker=markers.get(int(R), "^"), c=np.full(len(group), s),
                cmap="viridis", norm=norm, s=30,
            )
        xmax = max(0.45, float(valid["X_q_xi"].max()) * 1.1)
        X = np.linspace(0.0, xmax, 300)
        ax.plot(X, 1.0 / (1.0 + X**2), "k--", lw=1.5, label=r"$1/(1+X^2)$")
        for R in sorted(valid["R"].unique()):
            ax.scatter([], [], marker=markers.get(int(R), "^"), color="0.35", s=30,
                       label=f"R={int(R)}")
        scalar = plt.cm.ScalarMappable(norm=norm, cmap="viridis")
        colorbar = fig.colorbar(scalar, ax=ax, pad=0.02)
        colorbar.set_label(r"matched distance $s$")
        M_values = sorted(int(value) for value in valid["M"].unique())
        ax.text(0.03, 0.05, f"observed $X_{{max}}$={valid['X_q_xi'].max():.3f}\nM={M_values}; qR≤0.35",
                transform=ax.transAxes, fontsize=8, va="bottom")
    else:
        ax.text(0.5, 0.5, "No reliable microscopic finite-q conditions", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel=r"$X=q\xi_{dyn,micro}$", ylabel=r"$\tau(q)/\tau(0)=\Gamma_0/\Gamma(q)$",
           title="Result B: resolved microscopic finite-q range")
    ax.grid(alpha=0.25)
    if not valid.empty:
        ax.legend(fontsize=6, ncol=2)
    paths.append(_save_figure(fig, figures_dir / "01_ResultB_micro_finiteq.png"))

    eligible = diagnostics[diagnostics["condition_reliable"]].sort_values(
        ["D_over_sigma_D", "X_max"], ascending=False
    )
    representatives = eligible.groupby("R", as_index=False).head(1).head(3)
    if not representatives.empty:
        fig, ax = plt.subplots(figsize=(7.4, 5.2), constrained_layout=True)
        for _, condition in representatives.iterrows():
            group = valid[
                (valid["R"] == condition["R"])
                & np.isclose(valid["delta"], condition["delta"], rtol=0.0, atol=1e-15)
            ].sort_values("q")
            y = group["Gamma_micro"] - group["Gamma0_micro"]
            ax.errorbar(group["q"] ** 2, y, yerr=group["Gamma_micro_se"], fmt="o", capsize=3,
                        label=f"R={int(condition['R'])}, s={condition['matched_distance']:g}, D/SE={condition['D_over_sigma_D']:.2f}")
            xline = np.linspace(0.0, float((group["q"] ** 2).max()), 100)
            ax.plot(xline, float(condition["D_micro"]) * xline, "--", lw=1)
        ax.axhline(0.0, color="0.5", lw=0.8)
        ax.set(xlabel=r"$q^2$", ylabel=r"$\Gamma(q)-\Gamma_0$",
               title="Result B: dispersion precision (selected by D/SE only)")
        ax.grid(alpha=0.25); ax.legend(fontsize=7)
        paths.append(_save_figure(fig, figures_dir / "07_ResultB_dispersion_precision.png"))
    return paths


def make_result_c_figures(
    figures_dir: Path,
    epsilon: pd.DataFrame,
    windows: pd.DataFrame,
    z_points: pd.DataFrame,
    z_summary: pd.DataFrame,
) -> list[Path]:
    paths: list[Path] = []
    selected_eps = _select_representative_conditions(epsilon)
    fig, ax = plt.subplots(figsize=(7.0, 4.9), constrained_layout=True)
    for _, condition in selected_eps.iterrows():
        group = epsilon[(epsilon["R"] == condition["R"]) & np.isclose(epsilon["delta"], condition["delta"])].sort_values("epsilon_fraction")
        ax.plot(group["epsilon_fraction"], group["relative_ratio"], "o-",
                label=f"R={int(condition['R'])}, δ={condition['delta']:g}")
    ax.axhline(1.0, color="k", linestyle="--", lw=1)
    ax.set(xlabel="epsilon fraction", ylabel=r"$\xi_{bnd}(\epsilon)/\xi_{bnd}(0.05)$",
           title="Result C: linear-response amplitude robustness")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    paths.append(_save_figure(fig, figures_dir / "02_ResultC_epsilon_robustness.png"))

    selected_windows = _select_representative_conditions(windows)
    fig, ax = plt.subplots(figsize=(7.0, 4.9), constrained_layout=True)
    for _, condition in selected_windows.iterrows():
        group = windows[(windows["R"] == condition["R"]) & np.isclose(windows["delta"], condition["delta"])].sort_values("fit_x_min_factor_R")
        ax.plot(group["fit_x_min_factor_R"], group["relative_ratio"], "o-",
                label=f"R={int(condition['R'])}, δ={condition['delta']:g}")
    ax.axhline(1.0, color="k", linestyle="--", lw=1)
    ax.axvline(2.0, color="0.4", linestyle=":", lw=1, label="fixed primary 2R")
    ax.set(xticks=[1, 2, 3], xlabel=r"$x_{min}/R$", ylabel=r"$\xi_{bnd}(x_{min})/\xi_{bnd}(2R)$",
           title="Result C: fit-window robustness")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    paths.append(_save_figure(fig, figures_dir / "03_ResultC_fitwindow_robustness.png"))

    R_values = sorted(z_summary["R"].unique())
    ncols = 3
    nrows = max(1, int(math.ceil(len(R_values) / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 3.6 * nrows), constrained_layout=True, squeeze=False)
    for ax, R in zip(axes.flat, R_values):
        points = z_points[z_points["R"] == R].sort_values("xi_bnd")
        primary = points[points["delta"] <= PRIMARY_DELTA_MAX * (1.0 + 1e-12)]
        outside = points[points["delta"] > PRIMARY_DELTA_MAX * (1.0 + 1e-12)]
        summary = z_summary[z_summary["R"] == R].iloc[0]
        ax.loglog(outside["xi_bnd"], outside["tau0"], "o", mfc="none", color="tab:blue", label="outside")
        ax.loglog(primary["xi_bnd"], primary["tau0"], "o", color="tab:blue", label="δ≤3e-4")
        xline = np.logspace(np.log10(primary["xi_bnd"].min()), np.log10(primary["xi_bnd"].max()), 100)
        fit = fit_dynamic_exponent(primary["xi_bnd"], primary["tau0"])
        ax.loglog(xline, float(fit["amplitude"]) * xline ** float(fit["z"]), "-", lw=1.3, label=f"fit z={fit['z']:.3f}")
        anchor = primary.iloc[len(primary) // 2]
        ax.loglog(xline, float(anchor["tau0"]) * (xline / float(anchor["xi_bnd"])) ** 2,
                  "k--", lw=1, label="slope 2")
        ax.set(title=f"R={int(R)}", xlabel=r"$\xi_{bnd}$", ylabel=r"$\tau_0$")
        ax.grid(which="both", alpha=0.2); ax.legend(fontsize=7)
    for ax in axes.flat[len(R_values):]:
        ax.axis("off")
    fig.suptitle(r"Result C: independent dynamic test $\tau_0\sim\xi_{bnd}^{z}$")
    paths.append(_save_figure(fig, figures_dir / "04_ResultC_dynamic_z.png"))
    return paths


def make_fully_numeric_z_figures(
    figures_dir: Path,
    q0_numeric: pd.DataFrame,
    points: pd.DataFrame,
    summary: pd.DataFrame,
) -> list[Path]:
    """Create dedicated main and diagnostic figures for the no-fallback test."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = figures_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    q0 = q0_numeric.copy()
    q0["reliable"] = _coerce_bool(q0["reliable"])
    colors = dict(zip(FULL_NUMERIC_R_VALUES, plt.cm.viridis(np.linspace(0.05, 0.95, 5))))
    markers = {6: "o", 12: "s", 24: "^", 48: "D", 96: "v"}
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.2, 5.1), constrained_layout=True)
    for R in FULL_NUMERIC_R_VALUES:
        group = q0[q0["R"] == R].sort_values("delta")
        ax.loglog(
            group["delta"], group["Gamma0_num"], marker=markers[R],
            color=colors[R], linestyle="none", label=f"R={R} numerical",
        )
        ax.loglog(
            group["delta"], group["Gamma0_theory"], linestyle="--",
            color=colors[R], lw=1.2,
        )
    ax.plot([], [], "k--", lw=1.2, label="Phase0 theory (same R)")
    ax.set(xlabel=r"distance to spinodal $\delta$", ylabel=r"$\Gamma_{0}$",
           title="Direct numerical q=0 relaxation rate")
    ax.grid(which="both", alpha=0.25); ax.legend(fontsize=8, ncol=2)
    paths.append(_save_figure(fig, figures_dir / "01_q0_Gamma_numeric_vs_theory.png"))

    fig, ax = plt.subplots(figsize=(7.2, 4.9), constrained_layout=True)
    for R in FULL_NUMERIC_R_VALUES:
        group = q0[q0["R"] == R].sort_values("delta")
        ax.semilogx(
            group["delta"], group["Gamma0_num"] / group["Gamma0_theory"],
            marker=markers[R], color=colors[R], label=f"R={R}",
        )
    ax.axhline(1.0, color="k", linestyle="--", lw=1.2)
    ax.set(xlabel=r"$\delta$", ylabel=r"$\Gamma_{0,num}/\Gamma_{0,theory}$",
           title="q=0 numerical consistency")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
    paths.append(_save_figure(fig, figures_dir / "02_q0_Gamma_ratio.png"))

    ncols, nrows = 3, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.2, 7.2), constrained_layout=True, squeeze=False)
    for ax, R in zip(axes.flat, FULL_NUMERIC_R_VALUES):
        group = points[(points["R"] == R) & points["included"]].sort_values("xi_bnd")
        fit_row = summary[summary["R"] == R]
        ax.loglog(group["xi_bnd"], group["tau0_num"], "o", color=colors[R], label="numerical")
        if len(group) >= 2 and len(fit_row) == 1:
            fit = fit_dynamic_exponent(group["xi_bnd"], group["tau0_num"])
            xline = np.logspace(np.log10(group["xi_bnd"].min()), np.log10(group["xi_bnd"].max()), 100)
            ax.loglog(xline, float(fit["amplitude"]) * xline ** float(fit["z"]),
                      color=colors[R], lw=1.4, label=rf"fit $z={fit['z']:.3f}$")
            anchor = group.iloc[len(group) // 2]
            ax.loglog(
                xline, float(anchor["tau0_num"]) * (xline / float(anchor["xi_bnd"])) ** 2,
                "k--", lw=1.0, label="slope 2",
            )
            row = fit_row.iloc[0]
            ax.text(0.04, 0.96, rf"$R^2={row['r2']:.6f}$", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8)
        ax.set(title=f"R={R}", xlabel=r"$\xi_{bnd}$", ylabel=r"$\tau_{0,num}$")
        ax.grid(which="both", alpha=0.2); ax.legend(fontsize=7)
    for ax in axes.flat[len(FULL_NUMERIC_R_VALUES):]:
        ax.axis("off")
    fig.suptitle(r"Fully numerical independent dynamic scaling: $\tau_0\sim\xi_{bnd}^{z}$")
    paths.append(_save_figure(fig, figures_dir / "03_fully_numeric_dynamic_z.png"))

    fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    if not summary.empty:
        ax.errorbar(summary["R"], summary["z"], yerr=summary["z_se"], fmt="o-", capsize=4)
    ax.axhline(2.0, color="k", linestyle="--", label="z=2")
    ax.set(xscale="log", xticks=list(FULL_NUMERIC_R_VALUES), xlabel="R", ylabel=r"$z_R$",
           title="Fully numerical dynamic exponent by interaction range")
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.grid(alpha=0.25); ax.legend()
    paths.append(_save_figure(fig, figures_dir / "04_z_vs_R.png"))

    fig, ax = plt.subplots(figsize=(7.1, 4.8), constrained_layout=True)
    for R in FULL_NUMERIC_R_VALUES:
        group = q0[q0["R"] == R].sort_values("delta")
        ax.semilogx(group["delta"], group["lambda_fit_r2"], "o-", color=colors[R], label=f"R={R}")
    ax.axhline(0.999, color="k", linestyle="--", lw=1)
    ax.set(xlabel=r"$\delta$", ylabel=r"$R^2$ of $\lambda$ fit", title="q=0 lambda-fit quality")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
    paths.append(_save_figure(fig, diagnostics_dir / "q0_lambda_fit_r2.png"))

    fig, ax = plt.subplots(figsize=(7.1, 4.8), constrained_layout=True)
    for R in FULL_NUMERIC_R_VALUES:
        group = q0[q0["R"] == R].sort_values("delta")
        ax.loglog(group["delta"], group["method_B_C_relative_difference"], "o-",
                  color=colors[R], label=f"R={R}")
    ax.set(xlabel=r"$\delta$", ylabel="relative method B/C difference",
           title=r"$\Gamma_{from\ lambda}$ vs $\Gamma_{logfit}$")
    ax.grid(which="both", alpha=0.25); ax.legend(fontsize=8, ncol=2)
    paths.append(_save_figure(fig, diagnostics_dir / "q0_Gamma_method_difference.png"))

    fig, ax = plt.subplots(figsize=(7.1, 4.8), constrained_layout=True)
    for R in FULL_NUMERIC_R_VALUES:
        group = q0[q0["R"] == R].sort_values("delta")
        ax.loglog(group["delta"], group["Gamma_relative_error"], "o-",
                  color=colors[R], label=f"R={R}")
    ax.set(xlabel=r"$\delta$", ylabel="relative error vs Phase0 theory",
           title="Numerical q=0 validation")
    ax.grid(which="both", alpha=0.25); ax.legend(fontsize=8, ncol=2)
    paths.append(_save_figure(fig, diagnostics_dir / "q0_relative_error_vs_delta.png"))

    fig, ax = plt.subplots(figsize=(7.1, 4.8), constrained_layout=True)
    for R in FULL_NUMERIC_R_VALUES:
        source_rows = q0[q0["R"] == R]
        if source_rows.empty:
            continue
        source_file = Path(str(source_rows["source_file"].iloc[0]))
        if not source_file.is_absolute():
            source_file = Path.cwd() / source_file
        time_path = source_file.parent / "timeseries" / "delta_1e-05_mode_0.csv"
        if not time_path.is_file():
            continue
        time_series = pd.read_csv(time_path)
        ax.semilogy(time_series["t"], time_series["A_q_over_A0"].abs(),
                    color=colors[R], label=f"R={R}")
    ax.set(xlabel="time t", ylabel=r"$|A_0(t)/A_0(0)|$",
           title=r"Representative q=0 amplitude decay ($\delta=10^{-5}$)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
    paths.append(_save_figure(fig, diagnostics_dir / "q0_amplitude_decay_representative.png"))

    fig, ax = plt.subplots(figsize=(7.1, 4.8), constrained_layout=True)
    for _, row in summary.iterrows():
        R = int(row["R"])
        group = points[(points["R"] == R) & points["included"]]
        fit = fit_dynamic_exponent(group["xi_bnd"], group["tau0_num"])
        residual = np.log(group["tau0_num"]) - np.log(
            float(fit["amplitude"]) * group["xi_bnd"] ** float(fit["z"])
        )
        ax.semilogx(group["delta"], residual, "o-", color=colors[R], label=f"R={R}")
    ax.axhline(0.0, color="k", linestyle="--", lw=1)
    ax.set(xlabel=r"$\delta$", ylabel="log-fit residual", title="Fully numerical dynamic-z residual")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
    paths.append(_save_figure(fig, diagnostics_dir / "dynamic_z_residual.png"))
    return paths


def make_fully_numeric_combined_figures(
    figures_dir: Path,
    combined_points: pd.DataFrame,
    combined_summary: dict[str, Any],
) -> list[Path]:
    """Plot the normalized one-axis poster candidate and raw diagnostic."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    colors = dict(zip(FULL_NUMERIC_R_VALUES, plt.cm.viridis(np.linspace(0.05, 0.95, 5))))
    markers = {6: "o", 12: "s", 24: "^", 48: "D", 96: "v"}
    fit = combined_summary["combined_fit_all_20"]
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(7.6, 5.5), constrained_layout=True)
    for R in FULL_NUMERIC_R_VALUES:
        group = combined_points[combined_points["R"] == R].sort_values("xi_scaled")
        ax.plot(group["xi_scaled"], group["tau0_num"], color=colors[R], lw=0.8, alpha=0.35)
        ax.scatter(
            group["xi_scaled"], group["tau0_num"], marker=markers[R],
            color=colors[R], edgecolor="white", linewidth=0.7, s=68,
            zorder=3, label=f"R={R}",
        )
    x_min = float(combined_points["xi_scaled"].min())
    x_max = float(combined_points["xi_scaled"].max())
    x_line = np.logspace(np.log10(0.92 * x_min), np.log10(1.08 * x_max), 300)
    ax.loglog(x_line, x_line**2, color="black", lw=2.4, label="Theory: z = 2", zorder=2)
    ax.loglog(
        x_line, float(fit["A"]) * x_line ** float(fit["z"]),
        color="0.4", linestyle="--", lw=1.4, zorder=1,
    )
    ax.text(
        0.97, 0.05, rf"$z_{{\rm eff,combined}}={fit['z']:.3f}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=13,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
    )
    ax.set_xlabel(r"$\xi_{\rm bnd}/\sqrt{\kappa_R}$", fontsize=16)
    ax.set_ylabel(r"$\tau_{0,\rm num}=1/\Gamma_{0,\rm num}$", fontsize=16)
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.grid(which="both", alpha=0.18)
    ax.legend(fontsize=11, ncol=2, frameon=True)
    paths.append(
        _save_figure(
            fig, figures_dir / "03_fully_numeric_dynamic_z_combined.png", dpi=320
        )
    )

    fig, ax = plt.subplots(figsize=(7.6, 5.5), constrained_layout=True)
    for R in FULL_NUMERIC_R_VALUES:
        group = combined_points[combined_points["R"] == R].sort_values("xi_bnd")
        ax.plot(group["xi_bnd"], group["tau0_num"], color=colors[R], lw=0.9, alpha=0.45)
        ax.scatter(
            group["xi_bnd"], group["tau0_num"], marker=markers[R], color=colors[R],
            edgecolor="white", linewidth=0.6, s=62, zorder=3, label=f"R={R}",
        )
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$\xi_{\rm bnd}$", fontsize=14)
    ax.set_ylabel(r"$\tau_{0,\rm num}$", fontsize=14)
    ax.set_title(r"Raw coordinates retain the R-dependent prefactor $1/\kappa_R$", fontsize=13)
    ax.tick_params(axis="both", which="both", labelsize=11)
    ax.grid(which="both", alpha=0.2)
    ax.legend(fontsize=10, ncol=2)
    paths.append(
        _save_figure(
            fig, figures_dir / "03_fully_numeric_dynamic_z_combined_raw.png", dpi=320
        )
    )
    return paths


def _representative_phase6(xi_table: pd.DataFrame) -> tuple[int, float, pd.DataFrame]:
    frame = xi_table.copy()
    for column in ("primary_epsilon", "converged", "fit_reliable"):
        frame[column] = _coerce_bool(frame[column])
    frame = frame[frame["primary_epsilon"] & frame["converged"]]
    for delta in (1e-4, 3e-4):
        subset = frame[(frame["R"] == 24) & np.isclose(frame["delta"], delta, rtol=0.0, atol=1e-15)]
        if not subset.empty:
            return 24, delta, subset
    raise ValueError("Phase6 lacks the fixed representative R=24, delta=1e-4/3e-4 condition")


def make_result_d_figures(
    figures_dir: Path, profiles: pd.DataFrame, xi_table: pd.DataFrame
) -> tuple[list[Path], dict[str, Any]]:
    R, delta, fits = _representative_phase6(xi_table)
    reliable_fits = fits[fits["fit_reliable"]]
    excluded = sorted(set(fits["boundary_type"]) - set(reliable_fits["boundary_type"]))
    selected_profiles = profiles[
        (profiles["R"] == R) & np.isclose(profiles["delta"], delta, rtol=0.0, atol=1e-15)
        & np.isclose(profiles["epsilon_fraction"], PRIMARY_EPSILON, rtol=0.0, atol=1e-12)
    ]
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(7.1, 4.9), constrained_layout=True)
    for _, fit in reliable_fits.iterrows():
        group = selected_profiles[selected_profiles["boundary_type"] == fit["boundary_type"]].sort_values("x")
        group = group[group["x"] <= 3.0 * R]
        amplitude = float(group.loc[np.isclose(group["x"], 0.0), "delta_u"].iloc[0])
        ax.plot(group["x"] / R, group["delta_u"] / amplitude, label=fit["boundary_type"])
    ax.axvline(2.0, color="k", linestyle="--", lw=1, label="bulk fit starts at 2R")
    if excluded:
        ax.text(0.98, 0.96, "excluded by fixed reliability rule:\n" + ", ".join(excluded),
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set(xlim=(0.0, 3.0), xlabel=r"$x/R$", ylabel=r"$\Delta u(x)/\Delta u(0)$",
           title=f"Result D: boundary layer (R={R}, δ={delta:g})")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    paths.append(_save_figure(fig, figures_dir / "05_ResultD_boundary_layer.png"))

    fig, ax = plt.subplots(figsize=(7.1, 4.9), constrained_layout=True)
    maximum_X = 0.0
    for _, fit in reliable_fits.iterrows():
        group = selected_profiles[selected_profiles["boundary_type"] == fit["boundary_type"]].sort_values("x")
        bulk = bulk_normalized_profile(group["x"], group["delta_u"], R=R, xi_bulk=float(fit["xi_bnd_exp"]))
        bulk = bulk[bulk["X_bulk"] <= 6.0]
        maximum_X = max(maximum_X, float(bulk["X_bulk"].max()))
        ax.plot(bulk["X_bulk"], bulk["normalized_response"], label=fit["boundary_type"])
    X = np.linspace(0.0, min(6.0, maximum_X), 300)
    ax.plot(X, np.exp(-X), "k--", lw=1.5, label=r"$e^{-X}$")
    if excluded:
        ax.text(0.98, 0.96, "unreliable bulk fit excluded:\n" + ", ".join(excluded),
                transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.set(xlim=(0.0, 6.0), xlabel=r"$X=(x-2R)/\xi_{bulk}$",
           ylabel=r"$\Delta u(x)/\Delta u(2R)$",
           title=f"Result D: bulk exponential collapse (R={R}, δ={delta:g})")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    paths.append(_save_figure(fig, figures_dir / "06_ResultD_bulk_collapse.png"))
    return paths, {"R": R, "delta": delta, "included": sorted(reliable_fits["boundary_type"].tolist()), "excluded": excluded}


def make_diagnostics(
    diagnostics_dir: Path,
    b_precision: pd.DataFrame,
    epsilon: pd.DataFrame,
    windows: pd.DataFrame,
    z_points: pd.DataFrame,
    z_summary: pd.DataFrame,
    transmission: pd.DataFrame,
    profiles: pd.DataFrame,
    xi_table: pd.DataFrame,
) -> list[Path]:
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for R, group in b_precision.groupby("R"):
        ax.scatter(group["X_max"], group["D_over_sigma_D"], label=f"R={int(R)}")
    ax.axvline(0.4, color="k", linestyle="--"); ax.axhline(2.0, color="k", linestyle=":")
    ax.set(xlabel=r"$X_{max}$", ylabel=r"$D/SE(D)$", title="B precision target diagnostic")
    ax.grid(alpha=0.25); ax.legend()
    paths.append(_save_figure(fig, diagnostics_dir / "B_precision_all_conditions.png"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    reliable = epsilon[epsilon["fit_reliable"] & epsilon["converged"]]
    ax.scatter(reliable["epsilon_fraction"], reliable["relative_difference"], c=np.log10(reliable["delta"]), s=16)
    ax.axhline(0.0, color="k", linestyle="--")
    ax.set(xlabel="epsilon fraction", ylabel="relative xi change", title="C epsilon robustness: all reliable conditions")
    paths.append(_save_figure(fig, diagnostics_dir / "C_epsilon_all_conditions.png"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    reliable_w = windows[windows["fit_reliable"] & windows["converged"]]
    ax.scatter(reliable_w["fit_x_min_factor_R"], reliable_w["relative_ratio"] - 1.0,
               c=np.log10(reliable_w["delta"]), s=16)
    ax.axhline(0.0, color="k", linestyle="--")
    ax.set(xticks=[1, 2, 3], xlabel=r"$x_{min}/R$", ylabel="relative xi change", title="C fit-window robustness: all reliable conditions")
    paths.append(_save_figure(fig, diagnostics_dir / "C_fitwindow_all_conditions.png"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for _, summary in z_summary.iterrows():
        group = z_points[(z_points["R"] == summary["R"]) & (z_points["delta"] <= PRIMARY_DELTA_MAX * (1.0 + 1e-12))]
        fit = fit_dynamic_exponent(group["xi_bnd"], group["tau0"])
        residual = np.log(group["tau0"]) - np.log(float(fit["amplitude"]) * group["xi_bnd"] ** float(fit["z"]))
        ax.semilogx(group["delta"], residual, "o-", label=f"R={int(summary['R'])}")
    ax.axhline(0.0, color="k", linestyle="--")
    ax.set(xlabel="delta", ylabel="log-fit residual", title="Independent-z fit residual")
    ax.legend(fontsize=8)
    paths.append(_save_figure(fig, diagnostics_dir / "C_dynamic_z_residuals.png"))

    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    rep = transmission[(transmission["R"] == 24) & np.isclose(transmission["delta"], 1e-4)]
    ax.bar(rep["boundary_type"], rep["T_bnd"])
    ax.set(ylabel=r"$T_{bnd}=\Delta u(2R)/\Delta u(0)$", title="D boundary transmission")
    ax.tick_params(axis="x", rotation=15)
    paths.append(_save_figure(fig, diagnostics_dir / "D_boundary_transmission.png"))

    R, delta, _ = _representative_phase6(xi_table)
    subset = profiles[(profiles["R"] == R) & np.isclose(profiles["delta"], delta) & np.isclose(profiles["epsilon_fraction"], PRIMARY_EPSILON)]
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for boundary, group in subset.groupby("boundary_type"):
        valid = group["delta_u"].abs() > 1e-15
        ax.semilogy(group.loc[valid, "x"] / R, group.loc[valid, "delta_u"].abs(), label=boundary)
    ax.axvline(2.0, color="k", linestyle="--")
    ax.set(xlabel=r"$x/R$", ylabel=r"$|\Delta u|$", title="D semilog response (all boundary types; diagnostic)")
    ax.legend(fontsize=8)
    paths.append(_save_figure(fig, diagnostics_dir / "D_semilog_all_boundaries.png"))
    return paths


def run_analysis(args: argparse.Namespace) -> list[Path]:
    output_root = args.output_root
    figures_dir = output_root / "poster_final_figures"
    diagnostics_dir = figures_dir / "diagnostics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    phase6_dir = args.phase6_dir
    xi = pd.read_csv(phase6_dir / "resultC_xi_boundary.csv")
    systematics = pd.read_csv(phase6_dir / "resultC_boundary_fit_systematics.csv")
    profiles = pd.read_csv(phase6_dir / "resultC_boundary_profiles.csv")

    modes = _load_combined_micro_modes(
        args.micro_modes, args.refinement_root, args.pseudospinodal_root,
        T_obs=args.T_obs, epsilon_fraction=args.epsilon_fraction,
    )
    b_precision, b_collapse = build_result_b_precision(modes, qR_max=args.qR_max)
    epsilon = build_epsilon_robustness(xi)
    windows = build_fitwindow_robustness(systematics, xi)
    z_points, z_summary = build_dynamic_z_tables(
        xi, args.phase12_dir, primary_delta_max=args.primary_delta_max
    )
    fully_numeric_points = pd.DataFrame()
    fully_numeric_summary = pd.DataFrame()
    combined_points = pd.DataFrame()
    combined_summary: dict[str, Any] = {}
    combined_summary_path: Path | None = None
    q0_numeric = pd.DataFrame()
    fully_numeric_figure_paths: list[Path] = []
    fully_numeric_requested = args.fully_numeric_only or args.q0_numeric_master.is_file()
    if fully_numeric_requested:
        if not args.q0_numeric_master.is_file():
            raise FileNotFoundError(
                "incomplete: numerical Gamma0 missing; expected "
                f"{args.q0_numeric_master}"
            )
        q0_numeric = pd.read_csv(args.q0_numeric_master)
        fully_numeric_points, fully_numeric_summary = build_fully_numeric_dynamic_z_tables(
            xi, q0_numeric
        )
        lattice_spacing = infer_q0_lattice_spacing(q0_numeric)
        combined_points, combined_summary = build_fully_numeric_combined_analysis(
            fully_numeric_points, lattice_spacing=lattice_spacing
        )
    transmission = build_boundary_transmission(profiles, xi)

    table_paths = {
        "B_combined_modes": output_root / "resultB_final_combined_mode_results.csv",
        "B_combined_collapse": output_root / "resultB_final_combined_collapse.csv",
        "B_precision": output_root / "resultB_precision_diagnostic.csv",
        "C_epsilon": output_root / "resultC_epsilon_robustness.csv",
        "C_window": output_root / "resultC_fitwindow_robustness.csv",
        "C_z_points": output_root / "resultC_dynamic_z_points.csv",
        "C_z_summary": output_root / "resultC_dynamic_z_summary.csv",
        "D_transmission": output_root / "resultD_boundary_transmission.csv",
    }
    if fully_numeric_requested:
        table_paths.update(
            {
                "C_z_points_fully_numeric": output_root / "resultC_dynamic_z_points_fully_numeric.csv",
                "C_z_summary_fully_numeric": output_root / "resultC_dynamic_z_summary_fully_numeric.csv",
                "C_z_combined_points": output_root / "fully_numeric_dynamic_z_combined_points.csv",
            }
        )
    for frame, key in (
        (modes, "B_combined_modes"), (b_collapse, "B_combined_collapse"),
        (b_precision, "B_precision"), (epsilon, "C_epsilon"), (windows, "C_window"),
        (z_points, "C_z_points"), (z_summary, "C_z_summary"), (transmission, "D_transmission"),
    ):
        frame.to_csv(table_paths[key], index=False)
    if fully_numeric_requested:
        fully_numeric_points.to_csv(table_paths["C_z_points_fully_numeric"], index=False)
        fully_numeric_summary.to_csv(table_paths["C_z_summary_fully_numeric"], index=False)
        combined_points.to_csv(table_paths["C_z_combined_points"], index=False)

    figure_paths = []
    figure_paths.extend(make_result_b_figures(figures_dir, b_precision, b_collapse))
    figure_paths.extend(make_result_c_figures(figures_dir, epsilon, windows, z_points, z_summary))
    d_paths, d_selection = make_result_d_figures(figures_dir, profiles, xi)
    figure_paths.extend(d_paths)
    diagnostic_paths = make_diagnostics(
        diagnostics_dir, b_precision, epsilon, windows, z_points, z_summary,
        transmission, profiles, xi,
    )
    if fully_numeric_requested:
        fully_numeric_figure_paths.extend(make_fully_numeric_z_figures(
            figures_dir / "fully_numeric_z", q0_numeric,
            fully_numeric_points, fully_numeric_summary,
        ))
        fully_numeric_figure_paths.extend(make_fully_numeric_combined_figures(
            figures_dir / "fully_numeric_z", combined_points, combined_summary
        ))
        combined_summary_path = output_root / "fully_numeric_dynamic_z_combined_summary.json"
        combined_summary["output_files"] = {
            "points_csv": str(table_paths["C_z_combined_points"]),
            "normalized_poster_figure": str(
                figures_dir / "fully_numeric_z" / "03_fully_numeric_dynamic_z_combined.png"
            ),
            "raw_diagnostic_figure": str(
                figures_dir / "fully_numeric_z" / "03_fully_numeric_dynamic_z_combined_raw.png"
            ),
            "five_panel_backup": str(
                figures_dir / "fully_numeric_z" / "03_fully_numeric_dynamic_z.png"
            ),
        }
        combined_summary_path.write_text(
            json.dumps(
                _json_safe(combined_summary), indent=2, ensure_ascii=False,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

    reliable_epsilon = epsilon[epsilon["fit_reliable"] & epsilon["converged"]]
    critical_epsilon = reliable_epsilon[
        reliable_epsilon["delta"] <= args.primary_delta_max * (1.0 + 1e-12)
    ]
    representative_epsilon = _select_representative_conditions(epsilon)
    rep_eps_keys = set(zip(representative_epsilon["R"], representative_epsilon["delta"]))
    rep_eps_rows = reliable_epsilon[
        reliable_epsilon.apply(lambda row: (row["R"], row["delta"]) in rep_eps_keys, axis=1)
    ]
    representative_windows = _select_representative_conditions(windows)
    rep_win_keys = set(zip(representative_windows["R"], representative_windows["delta"]))
    rep_win_rows = windows[
        windows.apply(lambda row: (row["R"], row["delta"]) in rep_win_keys, axis=1)
    ]
    reliable_windows = windows[windows["fit_reliable"] & windows["converged"]]
    critical_windows = reliable_windows[
        reliable_windows["delta"] <= args.primary_delta_max * (1.0 + 1e-12)
    ]
    refinement_rows = modes[modes["data_source"] == "poster_B_refinement"]
    refinement_detected = not refinement_rows.empty
    precision_target_met = bool(b_precision["precision_target_met"].any())
    fully_numeric_validation: dict[str, Any] = {
        "status": "not requested",
        "fully_numeric": False,
    }
    if fully_numeric_requested:
        expected_pairs = {
            (R, delta) for R in FULL_NUMERIC_R_VALUES for delta in FULL_NUMERIC_DELTAS
        }
        actual_pairs = {
            (int(row.R), float(row.delta))
            for row in q0_numeric.itertuples()
            if int(row.R) in FULL_NUMERIC_R_VALUES
            and any(math.isclose(float(row.delta), value, rel_tol=1e-12, abs_tol=1e-15)
                    for value in FULL_NUMERIC_DELTAS)
        }
        missing_pairs = sorted(expected_pairs - actual_pairs)
        invalid_sources = sorted(
            set(q0_numeric["Gamma0_source"].astype(str)) - {FULL_NUMERIC_GAMMA_SOURCE}
        )
        excluded = fully_numeric_points[~fully_numeric_points["included"]]
        complete_fits = (
            set(fully_numeric_summary["R"].astype(int)) == set(FULL_NUMERIC_R_VALUES)
            and bool((fully_numeric_summary["n_points"] == 4).all())
        )
        completed = not missing_pairs and not invalid_sources and excluded.empty and complete_fits
        campaign_summary_path = args.q0_numeric_master.parent / "q0_full_numeric_validation_summary.json"
        campaign_summary = (
            json.loads(campaign_summary_path.read_text(encoding="utf-8"))
            if campaign_summary_path.is_file() else {}
        )
        z_consistency = {
            str(int(row.R)): bool(float(row.z_ci_low) <= 2.0 <= float(row.z_ci_high))
            for row in fully_numeric_summary.itertuples()
        }
        fully_numeric_validation = {
            "status": (
                "fully numerical independent validation completed"
                if completed else "incomplete: numerical Gamma0 missing or unreliable"
            ),
            "fully_numeric": completed,
            "Gamma0_definition": "Gamma_from_lambda = -ln(abs(lambda_fit))",
            "tau0_definition": "tau0_num = 1/Gamma0_num",
            "xi_definition": "independently fitted Phase6 real-space boundary response xi_bnd_cosh",
            "xi_dyn_used": False,
            "Phase0_Gamma_fallback_used": False,
            "Gamma0_source_required": FULL_NUMERIC_GAMMA_SOURCE,
            "missing_conditions": [f"R={R},delta={delta:g}" for R, delta in missing_pairs],
            "invalid_sources": invalid_sources,
            "excluded_conditions": excluded[
                ["R", "delta", "exclusion_reason"]
            ].to_dict(orient="records"),
            "per_R_z": fully_numeric_summary.to_dict(orient="records"),
            "z_95pct_CI_contains_2": z_consistency,
            "all_R_CI_contain_2": bool(z_consistency) and all(z_consistency.values()),
            "q0_campaign_summary": str(campaign_summary_path),
            "per_R_provenance": campaign_summary.get("per_R_provenance", {}),
            "max_Gamma0_relative_error": (
                float(q0_numeric["Gamma_relative_error"].abs().max())
                if "Gamma_relative_error" in q0_numeric else None
            ),
            "max_Gamma_from_lambda_minus_Gamma_logfit": (
                float((q0_numeric["Gamma0_num"] - q0_numeric["Gamma0_logfit"]).abs().max())
                if "Gamma0_logfit" in q0_numeric else None
            ),
            "normalized_combined_collapse": combined_summary,
        }
    validation = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_sha(), "command_line": sys.argv,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "pandas_version": pd.__version__, "scipy_version": scipy.__version__,
        "analysis_only": True, "phase6_new_simulation_performed": False,
        "B": {
            "integration_status": (
                "refinement integrated; fixed X/D precision target still unmet"
                if refinement_detected and not precision_target_met
                else "refinement integrated; fixed precision target met"
                if refinement_detected
                else "existing data only; refinement has not been transferred"
            ),
            "refinement_data_detected": refinement_detected,
            "refinement_R": sorted(int(value) for value in refinement_rows["R"].unique()),
            "refinement_matched_distances": sorted(float(value) for value in refinement_rows["matched_distance"].unique()),
            "refinement_M": sorted(int(value) for value in refinement_rows["M_total"].unique()),
            "combined_mode_rows": int(len(modes)),
            "combined_conditions": int(len(b_precision)),
            "precision_target_met_after_available_data": precision_target_met,
            "maximum_X": float(b_precision["X_max"].max()),
            "maximum_D_over_sigma_D": float(b_precision["D_over_sigma_D"].max()),
            "additional_simulation_needed_to_reach_fixed_target": not precision_target_met,
            "fixed_target": "X_max>=0.4 and D/SE(D)>=2; qR<=0.35",
            "recommended_first_refinement": (
                None if refinement_detected
                else "R=24; s=0.001,0.002,0.003,0.005; M=16384"
            ),
        },
        "C_epsilon": {
            "representative_max_absolute_relative_change": float(rep_eps_rows["relative_difference"].abs().max()),
            "critical_window_max_absolute_relative_change": float(critical_epsilon["relative_difference"].abs().max()),
            "all_reliable_max_absolute_relative_change": float(reliable_epsilon["relative_difference"].abs().max()),
        },
        "C_fitwindow": {
            "representative_max_absolute_relative_change": float((rep_win_rows["relative_ratio"] - 1.0).abs().max()),
            "critical_window_max_absolute_relative_change": float((critical_windows["relative_ratio"] - 1.0).abs().max()),
            "all_reliable_max_absolute_relative_change": float((reliable_windows["relative_ratio"] - 1.0).abs().max()),
        },
        "C_dynamic_z": z_summary.to_dict(orient="records"),
        "C_dynamic_z_fully_numeric": fully_numeric_validation,
        "D_representative": d_selection,
        "scientific_constraints": {
            "fit_windows_selected_from_result": False,
            "xi_dyn_used_in_independent_z_fit": False,
            "Phase0_Gamma_fallback_used_in_fully_numeric_z_fit": False,
            "finite_R_delta_ps_wording": "operational microscopic pseudospinodal-like crossover",
            "unreliable_D_curves_excluded_from_main_figures": True,
        },
    }
    validation_path = output_root / "poster_final_validation_summary.json"
    validation_path.write_text(
        json.dumps(
            _json_safe(validation), indent=2, ensure_ascii=False, allow_nan=False
        ) + "\n",
        encoding="utf-8",
    )
    output_paths = [
        *table_paths.values(), validation_path, *figure_paths, *diagnostic_paths,
        *fully_numeric_figure_paths,
    ]
    if combined_summary_path is not None:
        output_paths.append(combined_summary_path)
    return output_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("results/runs/poster_ABCD"))
    parser.add_argument("--micro-modes", type=Path, default=Path("results/runs/poster_ABCD/resultB_micro_mode_results.csv"))
    parser.add_argument("--refinement-root", type=Path, default=Path("results/runs/poster_ABCD/B_refinement"))
    parser.add_argument("--pseudospinodal-root", type=Path, default=Path("results/runs/phase5_R_sweep"))
    parser.add_argument("--phase6-dir", type=Path, default=Path("results/runs/poster_ABCD/phase6_boundary"))
    parser.add_argument("--phase12-dir", type=Path, default=Path("results/runs/phase12_B2_R12"))
    parser.add_argument(
        "--q0-numeric-master", type=Path,
        default=Path("results/runs/poster_ABCD/q0_full_numeric_validation/q0_numeric_all_5R.csv"),
    )
    parser.add_argument(
        "--fully-numeric-only", action="store_true",
        help="require the five-R numerical q=0 master and forbid theory fallback",
    )
    parser.add_argument("--epsilon-fraction", type=float, default=PRIMARY_EPSILON)
    parser.add_argument("--T-obs", type=int, default=50)
    parser.add_argument("--qR-max", type=float, default=PRIMARY_QR_MAX)
    parser.add_argument("--primary-delta-max", type=float, default=PRIMARY_DELTA_MAX)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        paths = run_analysis(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
