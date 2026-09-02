#!/usr/bin/env python3
"""Local post-processing for poster Results A and B.

The microscopic simulations remain owned by :mod:`spinodal_phase5_mpi`.  This
module reads its completed CSV products, changes coordinates from the Gaussian
distance ``delta`` to the operational matched distance
``s = delta - delta_ps(Tobs=50)``, and builds poster tables/figures without
changing Phase5 checkpoints or assuming the exact closure kernel relation.
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

from spinodal_phase34 import coefficient_of_determination, fit_power_law


SCRIPT_VERSION = "2026.08.31-poster-ab-v1"
DEFAULT_R = (6, 12, 24, 48, 96)
DEFAULT_B_R = (24, 48)
PRIMARY_S_MIN = 0.010
PRIMARY_S_MAX = 0.040
FIT_WINDOWS = (
    (0.005, 0.040, "s_0.005_0.040"),
    (0.0075, 0.040, "s_0.0075_0.040"),
    (0.010, 0.040, "s_0.010_0.040_primary"),
    (0.015, 0.040, "s_0.015_0.040"),
)


def parse_number_list(text: str, value_type: type = float) -> tuple[Any, ...]:
    values = tuple(value_type(token.strip()) for token in text.split(",") if token.strip())
    if not values:
        raise ValueError("list must contain at least one value")
    return values


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False}).fillna(False)


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_pseudospinodal(time_table: Path, T_obs: int = 50) -> dict[str, float]:
    frame = pd.read_csv(time_table)
    selected = frame[frame["T_obs"] == T_obs]
    if len(selected) != 1 or not np.isfinite(selected["delta_ps_estimate"].iloc[0]):
        raise ValueError(f"no bracketed operational pseudospinodal at T={T_obs}: {time_table}")
    row = selected.iloc[0]
    return {
        "delta_ps": float(row["delta_ps_estimate"]),
        "delta_ps_se": float(row.get("delta_ps_se", math.nan)),
        "delta_ps_ci_low": float(row.get("delta_ps_ci_low", math.nan)),
        "delta_ps_ci_high": float(row.get("delta_ps_ci_high", math.nan)),
    }


def fit_matched_power_law(
    matched_distance: Sequence[float], gamma: Sequence[float]
) -> dict[str, float | int]:
    """Fit ``Gamma=A*s**p``; kept public for synthetic validation."""
    return fit_power_law(matched_distance, gamma)


def bootstrap_matched_exponent(
    delta: np.ndarray,
    gamma: np.ndarray,
    gamma_se: np.ndarray,
    *,
    delta_ps: float,
    delta_ps_se: float,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Parametric propagation when raw block bootstrap samples are unavailable.

    The nominal fit window is selected before this function is called and is
    never changed according to its agreement with 1/2.
    """
    if replicates < 1:
        return np.empty(0, dtype=float)
    rng = np.random.Generator(np.random.Philox(seed))
    estimates: list[float] = []
    finite_se = np.where(np.isfinite(gamma_se) & (gamma_se > 0.0), gamma_se, 0.0)
    ps_scale = delta_ps_se if math.isfinite(delta_ps_se) and delta_ps_se > 0.0 else 0.0
    for _ in range(replicates):
        sampled_ps = delta_ps + ps_scale * float(rng.standard_normal())
        sampled_s = delta - sampled_ps
        sampled_gamma = gamma + finite_se * rng.standard_normal(len(gamma))
        if np.any(sampled_s <= 0.0) or np.any(sampled_gamma <= 0.0):
            continue
        estimates.append(float(fit_matched_power_law(sampled_s, sampled_gamma)["exponent"]))
    return np.asarray(estimates, dtype=float)


def _summary(samples: np.ndarray) -> tuple[float, float, float]:
    if len(samples) < 2:
        return math.nan, math.nan, math.nan
    return (
        float(np.std(samples, ddof=1)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
    )


def _micro_mode_path(micro_root: Path, R: int) -> Path:
    return micro_root / f"R{R:03d}" / "phase5_mode_results.csv"


def collect_result_a(
    micro_root: Path,
    pseudospinodal_root: Path,
    R_values: Sequence[int],
    *,
    primary_epsilon: float,
    T_obs: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    scaling_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for R in R_values:
        mode_path = _micro_mode_path(micro_root, R)
        time_path = (
            pseudospinodal_root
            / f"R{R:03d}"
            / "pseudospinodal_fine"
            / "analysis"
            / "phase5_pseudospinodal_time_dependence.csv"
        )
        if not mode_path.is_file() or not time_path.is_file():
            missing.append(f"R={R}: {mode_path if not mode_path.is_file() else time_path}")
            continue
        pseudo = load_pseudospinodal(time_path, T_obs=T_obs)
        frame = pd.read_csv(mode_path)
        frame["reliable"] = _coerce_bool(frame["reliable"])
        selected = frame[
            (frame["mode_index"] == 0)
            & np.isclose(frame["epsilon_fraction"], primary_epsilon, rtol=0.0, atol=1e-12)
        ].copy()
        selected["matched_distance"] = selected["delta"] - pseudo["delta_ps"]
        selected = selected[selected["matched_distance"] > 0.0].sort_values("matched_distance")
        for _, row in selected.iterrows():
            scaling_rows.append(
                {
                    "R": R,
                    "N": int(row["N"]),
                    "delta_ps_T50": pseudo["delta_ps"],
                    "delta_ps_T50_se": pseudo["delta_ps_se"],
                    "delta_ps_T50_ci_low": pseudo["delta_ps_ci_low"],
                    "delta_ps_T50_ci_high": pseudo["delta_ps_ci_high"],
                    "delta": float(row["delta"]),
                    "matched_distance": float(row["matched_distance"]),
                    "Gamma_micro": float(row["Gamma_micro"]),
                    "Gamma_micro_se": float(row.get("Gamma_micro_se", math.nan)),
                    "Gamma_micro_ci_low": float(row.get("Gamma_micro_ci_low", math.nan)),
                    "Gamma_micro_ci_high": float(row.get("Gamma_micro_ci_high", math.nan)),
                    "Gamma_closure": float(row.get("Gamma_closure", math.nan)),
                    "escape_fraction": float(row.get("escape_fraction", math.nan)),
                    "survival_fraction": 1.0 - float(row.get("escape_fraction", math.nan)),
                    "fit_r2": float(row.get("fit_r2", math.nan)),
                    "method_B_C_relative_difference": float(row.get("method_B_C_relative_difference", math.nan)),
                    "reliable": bool(row["reliable"]),
                }
            )

        for s_min, s_max, label in FIT_WINDOWS:
            window = selected[
                selected["reliable"]
                & (selected["matched_distance"] >= s_min * (1.0 - 1e-10))
                & (selected["matched_distance"] <= s_max * (1.0 + 1e-10))
                & (selected["Gamma_micro"] > 0.0)
            ]
            if len(window) < 2:
                continue
            fit = fit_matched_power_law(window["matched_distance"], window["Gamma_micro"])
            boot = bootstrap_matched_exponent(
                window["delta"].to_numpy(float),
                window["Gamma_micro"].to_numpy(float),
                window["Gamma_micro_se"].to_numpy(float),
                delta_ps=pseudo["delta_ps"],
                delta_ps_se=pseudo["delta_ps_se"],
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + R * 100 + int(round(10000 * s_min)),
            )
            se, low, high = _summary(boot)
            sensitivity = {}
            for key, ps_value in (
                ("ci_low", pseudo["delta_ps_ci_low"]),
                ("ci_high", pseudo["delta_ps_ci_high"]),
            ):
                shifted_s = window["delta"].to_numpy(float) - ps_value
                sensitivity[key] = (
                    float(fit_matched_power_law(shifted_s, window["Gamma_micro"])["exponent"])
                    if np.all(np.isfinite(shifted_s)) and np.all(shifted_s > 0.0)
                    else math.nan
                )
            fit_rows.append(
                {
                    "R": R,
                    "N": int(window["N"].iloc[0]),
                    "coordinate": "s=delta-delta_ps(10%,Tobs=50)",
                    "fit_window": label,
                    "s_min": s_min,
                    "s_max": s_max,
                    "n_points": len(window),
                    "p_R": float(fit["exponent"]),
                    "p_R_se": se,
                    "p_R_ci_low": low,
                    "p_R_ci_high": high,
                    "p_R_delta_ps_ci_low_sensitivity": sensitivity["ci_low"],
                    "p_R_delta_ps_ci_high_sensitivity": sensitivity["ci_high"],
                    "p_R_minus_half": float(fit["exponent"]) - 0.5,
                    "r2": float(fit["r2"]),
                    "primary_window": label.endswith("primary"),
                    "uncertainty_method": "parametric Gamma-SE plus delta_ps-SE; fixed nominal window",
                    "interpretation": "effective/matched microscopic exponent; not a true critical exponent",
                }
            )
    return pd.DataFrame(scaling_rows), pd.DataFrame(fit_rows), missing


def fit_measured_dispersion(
    q: Sequence[float], gamma: Sequence[float], gamma_se: Sequence[float] | None = None
) -> dict[str, float | int]:
    """Fit measured ``Gamma(q)=Gamma0+D*q^2`` with a free intercept."""
    q_array = np.asarray(q, dtype=float)
    y = np.asarray(gamma, dtype=float)
    if len(q_array) != len(y) or len(y) < 2 or not np.all(np.isfinite(q_array)) or not np.all(np.isfinite(y)):
        raise ValueError("dispersion fit requires at least two finite q/Gamma values")
    x = q_array**2
    if np.ptp(x) <= 0.0:
        raise ValueError("dispersion fit requires distinct q^2 values")
    design = np.column_stack((np.ones_like(x), x))
    if gamma_se is not None:
        se = np.asarray(gamma_se, dtype=float)
        valid = np.isfinite(se) & (se > 0.0)
    else:
        se = np.ones_like(y)
        valid = np.zeros_like(y, dtype=bool)
    weights = np.where(valid, 1.0 / se**2, 1.0)
    normal = design.T @ (weights[:, None] * design)
    beta = np.linalg.solve(normal, design.T @ (weights * y))
    predicted = design @ beta
    dof = len(y) - 2
    if np.any(valid):
        covariance = np.linalg.inv(normal)
    elif dof > 0:
        covariance = np.linalg.inv(normal) * float(np.sum((y - predicted) ** 2) / dof)
    else:
        covariance = np.full((2, 2), math.nan)
    return {
        "Gamma0": float(beta[0]),
        "D": float(beta[1]),
        "Gamma0_se": float(math.sqrt(max(covariance[0, 0], 0.0))) if np.isfinite(covariance[0, 0]) else math.nan,
        "D_se": float(math.sqrt(max(covariance[1, 1], 0.0))) if np.isfinite(covariance[1, 1]) else math.nan,
        "r2": float(coefficient_of_determination(y, predicted)),
        "n_points": int(len(y)),
    }


def fit_measured_q4(q: np.ndarray, gamma: np.ndarray, gamma0: float) -> dict[str, float]:
    q2 = np.asarray(q, dtype=float) ** 2
    design = np.column_stack((q2, q2**2))
    if len(q2) < 3 or np.linalg.matrix_rank(design) < 2:
        return {"D2": math.nan, "D4": math.nan, "r2_q4": math.nan}
    beta, _, _, _ = np.linalg.lstsq(design, np.asarray(gamma) - gamma0, rcond=None)
    predicted = gamma0 + design @ beta
    return {
        "D2": float(beta[0]),
        "D4": float(beta[1]),
        "r2_q4": float(coefficient_of_determination(gamma, predicted)),
    }


def collect_result_b(
    micro_root: Path,
    pseudospinodal_root: Path,
    R_values: Sequence[int],
    *,
    primary_epsilon: float,
    qR_max: float,
    T_obs: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    mode_rows: list[pd.DataFrame] = []
    dispersion_rows: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    collapse_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for R in R_values:
        mode_path = _micro_mode_path(micro_root, R)
        time_path = pseudospinodal_root / f"R{R:03d}" / "pseudospinodal_fine" / "analysis" / "phase5_pseudospinodal_time_dependence.csv"
        if not mode_path.is_file() or not time_path.is_file():
            missing.append(f"R={R}: Result B input missing")
            continue
        pseudo = load_pseudospinodal(time_path, T_obs=T_obs)
        frame = pd.read_csv(mode_path)
        frame["reliable"] = _coerce_bool(frame["reliable"])
        frame = frame[np.isclose(frame["epsilon_fraction"], primary_epsilon, rtol=0.0, atol=1e-12)].copy()
        frame["R"] = R
        frame["delta_ps_T50"] = pseudo["delta_ps"]
        frame["matched_distance"] = frame["delta"] - pseudo["delta_ps"]
        mode_rows.append(frame)
        for delta, group in frame.groupby("delta"):
            eligible = group[
                group["reliable"]
                & np.isfinite(group["Gamma_micro"])
                & (group["Gamma_micro"] > 0.0)
                & (group["qR"] <= qR_max * (1.0 + 1e-12))
            ].sort_values("mode_index")
            if len(eligible) < 2 or not (eligible["mode_index"] == 0).any():
                continue
            fit = fit_measured_dispersion(eligible["q"], eligible["Gamma_micro"], eligible["Gamma_micro_se"])
            q0 = eligible[eligible["mode_index"] == 0].iloc[0]
            q4 = fit_measured_q4(eligible["q"].to_numpy(float), eligible["Gamma_micro"].to_numpy(float), float(q0["Gamma_micro"]))
            kappa = float((R + 1) * (2 * R + 1) / 12.0)
            D = float(fit["D"])
            Gamma0 = float(q0["Gamma_micro"])
            Gamma0_se = float(q0.get("Gamma_micro_se", math.nan))
            xi = math.sqrt(D / Gamma0) if D > 0.0 and Gamma0 > 0.0 else math.nan
            xi_se = math.nan
            if xi > 0.0 and float(fit["D_se"]) >= 0.0 and Gamma0_se >= 0.0:
                xi_se = 0.5 * xi * math.sqrt((float(fit["D_se"]) / D) ** 2 + (Gamma0_se / Gamma0) ** 2)
            matched = float(delta) - pseudo["delta_ps"]
            dispersion_rows.append(
                {
                    "R": R, "N": int(q0["N"]), "delta_ps_T50": pseudo["delta_ps"],
                    "delta": float(delta), "matched_distance": matched,
                    "Gamma0_micro": Gamma0, "Gamma0_micro_se": Gamma0_se,
                    "Gamma0_free_intercept_fit": fit["Gamma0"],
                    "Gamma0_free_intercept_fit_se": fit["Gamma0_se"],
                    "D_micro": D, "D_micro_se": fit["D_se"],
                    "D_micro_ci_low": D - 1.96 * float(fit["D_se"]),
                    "D_micro_ci_high": D + 1.96 * float(fit["D_se"]),
                    "D2_q4_diagnostic": q4["D2"], "D4_diagnostic": q4["D4"],
                    "dispersion_r2": fit["r2"], "dispersion_q4_r2": q4["r2_q4"],
                    "n_modes_used": fit["n_points"], "qR_max_used": float(eligible["qR"].max()),
                    "kappa_R": kappa, "D_ratio_to_kappa": D / kappa,
                    "micro_exact_kernel_assumed": False,
                }
            )
            length_rows.append(
                {
                    "R": R, "N": int(q0["N"]), "delta": float(delta), "matched_distance": matched,
                    "Gamma0_micro": Gamma0, "D_micro": D, "xi_dyn_micro": xi,
                    "xi_dyn_micro_se": xi_se, "length_definition": "sqrt(D_micro/Gamma0_micro); internal consistency",
                }
            )
            local_residuals = []
            for _, row in eligible.iterrows():
                X = float(row["q"]) * xi
                Y = Gamma0 / float(row["Gamma_micro"])
                theory = 1.0 / (1.0 + X * X)
                residual = Y - theory
                local_residuals.append(residual)
                collapse_rows.append(
                    {
                        "R": R, "N": int(row["N"]), "delta": float(delta), "matched_distance": matched,
                        "mode_index": int(row["mode_index"]), "q": float(row["q"]), "qR": float(row["qR"]),
                        "xi_source": "xi_dyn_micro", "xi": xi, "X_q_xi": X,
                        "Gamma_micro": float(row["Gamma_micro"]), "Gamma0_micro": Gamma0,
                        "delta_Gamma": float(row["Gamma_micro"]) - Gamma0, "q2": float(row["q"]) ** 2,
                        "Y_tau_ratio": Y, "Y_theory": theory, "residual": residual,
                    }
                )
            residuals = np.asarray(local_residuals, dtype=float)
            summary_rows.append(
                {
                    "R": R, "delta": float(delta), "matched_distance": matched,
                    "xi_source": "xi_dyn_micro", "rmse": float(np.sqrt(np.mean(residuals**2))),
                    "maximum_absolute_residual": float(np.max(np.abs(residuals))),
                    "reduced_residual": float(np.sum(residuals**2) / max(len(residuals) - 2, 1)),
                    "n_points": len(residuals), "qR_max": float(eligible["qR"].max()),
                }
            )
    modes = pd.concat(mode_rows, ignore_index=True) if mode_rows else pd.DataFrame()
    return modes, pd.DataFrame(dispersion_rows), pd.DataFrame(length_rows), pd.DataFrame(collapse_rows), pd.DataFrame(summary_rows), missing


def build_independent_collapse(
    modes: pd.DataFrame, xi_boundary_path: Path, *, qR_max: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["R", "delta", "mode_index", "q", "qR", "xi_source", "xi", "X_q_xi", "Y_tau_ratio", "Y_theory", "residual"]
    if modes.empty or not xi_boundary_path.is_file():
        return pd.DataFrame(columns=columns), pd.DataFrame()
    xi = pd.read_csv(xi_boundary_path)
    if "primary_epsilon" in xi:
        xi = xi[_coerce_bool(xi["primary_epsilon"])]
    if "boundary_type" in xi:
        xi = xi[xi["boundary_type"] == "ghost_dirichlet"]
    rows: list[dict[str, Any]] = []
    for _, length in xi.iterrows():
        group = modes[
            (modes["R"] == int(length["R"]))
            & np.isclose(modes["delta"], float(length["delta"]), rtol=0.0, atol=5e-10)
            & modes["reliable"]
            & (modes["qR"] <= qR_max * (1.0 + 1e-12))
        ].sort_values("mode_index")
        q0 = group[group["mode_index"] == 0]
        if q0.empty or len(group) < 2:
            continue
        gamma0 = float(q0["Gamma_micro"].iloc[0])
        length_value = float(length["xi_bnd_cosh"])
        for _, mode in group.iterrows():
            X = float(mode["q"]) * length_value
            Y = gamma0 / float(mode["Gamma_micro"])
            theory = 1.0 / (1.0 + X * X)
            rows.append(
                {"R": int(length["R"]), "delta": float(length["delta"]), "mode_index": int(mode["mode_index"]),
                 "q": float(mode["q"]), "qR": float(mode["qR"]), "xi_source": "xi_bnd_independent",
                 "xi": length_value, "X_q_xi": X, "Y_tau_ratio": Y, "Y_theory": theory, "residual": Y - theory}
            )
    table = pd.DataFrame(rows, columns=columns)
    summaries = []
    if not table.empty:
        for (R, delta), group in table.groupby(["R", "delta"]):
            residual = group["residual"].to_numpy(float)
            summaries.append({"R": R, "delta": delta, "xi_source": "xi_bnd_independent", "rmse": float(np.sqrt(np.mean(residual**2))), "maximum_absolute_residual": float(np.max(np.abs(residual))), "n_points": len(group)})
    return table, pd.DataFrame(summaries)


def build_direct_J_comparison(
    micro_root: Path,
    direct_J_root: Path,
    R_values: Sequence[int],
    *,
    primary_epsilon: float,
) -> pd.DataFrame:
    """Compare optional small direct-J references to production aggregates."""
    frames: list[pd.DataFrame] = []
    for R in R_values:
        aggregate_path = _micro_mode_path(micro_root, R)
        direct_path = direct_J_root / f"R{R:03d}" / "phase5_mode_results.csv"
        if not aggregate_path.is_file() or not direct_path.is_file():
            continue
        aggregate = pd.read_csv(aggregate_path)
        direct = pd.read_csv(direct_path)
        aggregate = aggregate[np.isclose(aggregate["epsilon_fraction"], primary_epsilon, rtol=0.0, atol=1e-12)]
        direct = direct[np.isclose(direct["epsilon_fraction"], primary_epsilon, rtol=0.0, atol=1e-12)]
        merged = aggregate.merge(direct, on=["delta", "mode_index", "epsilon_fraction"], suffixes=("_aggregated", "_direct_J"))
        if merged.empty:
            continue
        merged.insert(0, "R", R)
        merged["Gamma_relative_difference"] = np.abs(merged["Gamma_micro_direct_J"] - merged["Gamma_micro_aggregated"]) / np.maximum(np.abs(merged["Gamma_micro_aggregated"]), 1e-15)
        frames.append(merged[["R", "delta", "mode_index", "epsilon_fraction", "M_total_aggregated", "M_total_direct_J", "Gamma_micro_aggregated", "Gamma_micro_se_aggregated", "Gamma_micro_direct_J", "Gamma_micro_se_direct_J", "Gamma_relative_difference"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["R", "delta", "mode_index", "epsilon_fraction", "M_total_aggregated", "M_total_direct_J", "Gamma_micro_aggregated", "Gamma_micro_se_aggregated", "Gamma_micro_direct_J", "Gamma_micro_se_direct_J", "Gamma_relative_difference"])


def _make_figures(
    output_dir: Path,
    result_a: pd.DataFrame,
    p_summary: pd.DataFrame,
    dispersion: pd.DataFrame,
    collapse: pd.DataFrame,
    independent: pd.DataFrame,
) -> list[Path]:
    diagnostics = output_dir / "figures" / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    if not result_a.empty:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for R, group in result_a[result_a["reliable"]].groupby("R"):
            ax.errorbar(group["matched_distance"], group["Gamma_micro"], yerr=group["Gamma_micro_se"], marker="o", linestyle="-", label=f"R={R}")
        reference_x = np.array([PRIMARY_S_MIN, PRIMARY_S_MAX])
        if not result_a.empty:
            anchor = float(np.nanmedian(result_a["Gamma_micro"] / np.sqrt(result_a["matched_distance"])))
            ax.plot(reference_x, anchor * np.sqrt(reference_x), "k--", label="slope 1/2")
        ax.set(xscale="log", yscale="log", xlabel=r"matched distance $s=\delta-\delta_{ps}$", ylabel=r"$\Gamma_{0,\mathrm{micro}}$", title="Result A: matched microscopic slowing down")
        ax.legend(fontsize=8)
        path = diagnostics / "A1_micro_gamma_scaling_by_R.png"
        fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    primary = p_summary[p_summary.get("primary_window", False) == True] if not p_summary.empty else pd.DataFrame()
    if not primary.empty:
        primary = primary.sort_values("R")
        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        yerr = np.vstack((primary["p_R"] - primary["p_R_ci_low"], primary["p_R_ci_high"] - primary["p_R"]))
        ax.errorbar(primary["R"], primary["p_R"], yerr=yerr, marker="o", capsize=3)
        ax.axhline(0.5, color="black", linestyle="--", label="Gaussian 1/2")
        ax.set(xlabel="R", ylabel=r"effective matched exponent $p_R$", title="Result A: large-R exponent convergence")
        ax.legend()
        path = output_dir / "01_ResultA_micro_exponent.png"
        fig.savefig(path, dpi=220)
        diagnostic = diagnostics / "A2_pR_vs_R.png"
        fig.savefig(diagnostic, dpi=220); plt.close(fig); paths.extend((path, diagnostic))
        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        ax.plot(1.0 / primary["R"], np.abs(primary["p_R_minus_half"]), "o-")
        ax.set(xlabel="1/R", ylabel=r"$|p_R-1/2|$", title="Convergence diagnostic (no crossover fit)")
        path = diagnostics / "A3_pR_deviation.png"
        fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    if not dispersion.empty:
        if not collapse.empty:
            fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
            for (R, s), group in collapse.groupby(["R", "matched_distance"]):
                ax.plot(group["q2"], group["delta_Gamma"], "o-", label=f"R={R}, s={s:g}")
            ax.axhline(0.0, color="0.7", linewidth=0.8)
            ax.set(xlabel=r"$q^2$", ylabel=r"$\Gamma(q)-\Gamma_0$", title="Result B: measured microscopic dispersion")
            ax.legend(fontsize=6, ncol=2)
            path = diagnostics / "B1_micro_dispersion.png"
            fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
        fig, ax = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
        for R, group in dispersion.groupby("R"):
            ax.errorbar(group["matched_distance"], group["D_ratio_to_kappa"], yerr=group["D_micro_se"] / group["kappa_R"], marker="o", label=f"R={R}")
        ax.axhline(1.0, color="black", linestyle="--")
        ax.set(xlabel="matched distance s", ylabel=r"$D_{micro}/\kappa_R$", title="Result B: measured microscopic dispersion")
        ax.legend()
        path = diagnostics / "B2_D_micro_vs_distance.png"
        fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    if not collapse.empty:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for (R, s), group in collapse.groupby(["R", "matched_distance"]):
            ax.plot(group["X_q_xi"], group["Y_tau_ratio"], "o", ms=4, label=f"R={R}, s={s:g}")
        x = np.linspace(0.0, max(1.0, float(collapse["X_q_xi"].max()) * 1.05), 300)
        ax.plot(x, 1.0 / (1.0 + x*x), "k--", label=r"$1/(1+x^2)$")
        ax.set(xlabel=r"$q\xi_{dyn,micro}$", ylabel=r"$\tau(q)/\tau(0)$", title="Result B: microscopic finite-q collapse")
        ax.legend(fontsize=6, ncol=2)
        path = output_dir / "02_ResultB_micro_collapse.png"
        fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
        diagnostic = diagnostics / "B3_micro_data_collapse.png"
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        ax.scatter(collapse["X_q_xi"], collapse["Y_tau_ratio"], c=collapse["R"], s=20)
        ax.plot(x, 1.0/(1.0+x*x), "k--")
        ax.set(xlabel=r"$q\xi_{dyn,micro}$", ylabel=r"$\tau(q)/\tau(0)$")
        fig.savefig(diagnostic, dpi=220); plt.close(fig); paths.append(diagnostic)
    if not independent.empty:
        fig, ax = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        for (R, delta), group in independent.groupby(["R", "delta"]):
            ax.plot(group["X_q_xi"], group["Y_tau_ratio"], "o", label=f"R={R}, delta={delta:g}")
        x = np.linspace(0.0, max(1.0, float(independent["X_q_xi"].max()) * 1.05), 300)
        ax.plot(x, 1.0/(1.0+x*x), "k--")
        ax.set(xlabel=r"$q\xi_{bnd}$", ylabel=r"$\tau(q)/\tau(0)$", title="Independent real-space-length collapse")
        ax.legend(fontsize=7)
        path = diagnostics / "B4_micro_data_collapse_independent_xi.png"
        fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
    return paths


def _build_abcd_summary(a: pd.DataFrame, p: pd.DataFrame, b_dispersion: pd.DataFrame, b_length: pd.DataFrame, collapse_summary: pd.DataFrame, phase6_dir: Path) -> pd.DataFrame:
    primary_p = p[p["primary_window"]][["R", "p_R"]] if not p.empty else pd.DataFrame(columns=["R", "p_R"])
    base = b_dispersion.merge(b_length[["R", "delta", "xi_dyn_micro"]], on=["R", "delta"], how="outer") if not b_dispersion.empty else pd.DataFrame()
    if base.empty and not a.empty:
        base = a[["R", "N", "delta", "matched_distance", "Gamma_micro"]].rename(columns={"Gamma_micro": "Gamma0_micro"})
    if not base.empty:
        base = base.merge(primary_p, on="R", how="left")
        if not collapse_summary.empty:
            base = base.merge(collapse_summary[["R", "delta", "rmse"]].rename(columns={"rmse": "collapse_rmse"}), on=["R", "delta"], how="left")
    xi_path = phase6_dir / "resultC_xi_boundary.csv"
    d_path = phase6_dir / "resultD_boundary_fit_summary.csv"
    if not base.empty and xi_path.is_file():
        xi = pd.read_csv(xi_path)
        xi = xi[_coerce_bool(xi["primary_epsilon"])] if "primary_epsilon" in xi else xi
        xi = xi[xi["boundary_type"] == "ghost_dirichlet"] if "boundary_type" in xi else xi
        base = base.merge(xi[["R", "delta", "xi_bnd_cosh", "xi_bnd_over_xi_dyn"]], on=["R", "delta"], how="outer")
    if not base.empty and d_path.is_file():
        d = pd.read_csv(d_path)
        base = base.merge(d[["R", "delta", "boundary_type", "rmse"]].rename(columns={"rmse": "boundary_fit_rmse"}), on=["R", "delta"], how="outer")
    return base


def run_analysis(args: argparse.Namespace) -> list[Path]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    a, p, missing_a = collect_result_a(
        args.micro_root, args.pseudospinodal_root, args.R_list,
        primary_epsilon=args.epsilon_fraction, T_obs=args.T_obs,
        bootstrap_replicates=args.bootstrap_replicates, bootstrap_seed=args.bootstrap_seed,
    )
    modes, dispersion, length, collapse, collapse_summary, missing_b = collect_result_b(
        args.micro_root, args.pseudospinodal_root, args.B_R_list,
        primary_epsilon=args.epsilon_fraction, qR_max=args.qR_max, T_obs=args.T_obs,
    )
    independent, independent_summary = build_independent_collapse(
        modes, args.phase6_dir / "resultC_xi_boundary.csv", qR_max=args.qR_max
    )
    direct_J = build_direct_J_comparison(
        args.micro_root, args.direct_J_root, args.R_list,
        primary_epsilon=args.epsilon_fraction,
    )
    paths = {
        "A_scaling": output_dir / "resultA_microscopic_gamma_scaling.csv",
        "A_summary": output_dir / "resultA_pR_summary.csv",
        "B_modes": output_dir / "resultB_micro_mode_results.csv",
        "B_dispersion": output_dir / "resultB_micro_dispersion.csv",
        "B_length": output_dir / "resultB_micro_length.csv",
        "B_collapse": output_dir / "resultB_micro_collapse.csv",
        "B_collapse_summary": output_dir / "resultB_micro_collapse_summary.csv",
        "B_independent": output_dir / "resultB_micro_collapse_independent_xi.csv",
        "direct_J": output_dir / "resultAB_direct_J_correctness.csv",
    }
    for frame, key in ((a, "A_scaling"), (p, "A_summary"), (modes, "B_modes"), (dispersion, "B_dispersion"), (length, "B_length"), (collapse, "B_collapse"), (pd.concat([collapse_summary, independent_summary], ignore_index=True), "B_collapse_summary"), (independent, "B_independent"), (direct_J, "direct_J")):
        frame.to_csv(paths[key], index=False)
    summary = _build_abcd_summary(a, p, dispersion, length, collapse_summary, args.phase6_dir)
    summary_path = output_dir / "result_ABCD_summary.csv"
    summary.to_csv(summary_path, index=False)
    figure_paths = _make_figures(output_dir, a, p, dispersion, collapse, independent) if args.figures else []
    primary_p = p[p["primary_window"]] if not p.empty else pd.DataFrame()
    completed_A_R = sorted(int(value) for value in primary_p["R"].unique()) if not primary_p.empty else []
    completed_B_R = sorted(int(value) for value in dispersion["R"].unique()) if not dispersion.empty else []
    requested_A_R = sorted(int(value) for value in args.R_list)
    requested_B_R = sorted(int(value) for value in args.B_R_list)
    valid_D = bool(not dispersion.empty and (dispersion["D_micro"] > 0.0).all())
    valid_q_window = bool(not dispersion.empty and (dispersion["qR_max_used"] <= args.qR_max * (1.0 + 1e-12)).all())
    validation = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_sha(),
        "command_line": sys.argv,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "pandas_version": pd.__version__,
        "bootstrap_seed": args.bootstrap_seed,
        "pseudospinodal_source_root": str(args.pseudospinodal_root),
        "T_obs": args.T_obs,
        "primary_matched_window": [PRIMARY_S_MIN, PRIMARY_S_MAX],
        "fit_window_selected_from_result": False,
        "qR_max_primary": args.qR_max,
        "micro_exact_kernel_assumed": False,
        "finite_R_wording": "operational microscopic pseudospinodal-like crossover; not a true microscopic spinodal",
        "xi_dyn_interpretation": "derived from the same Gamma(q) dispersion; internal consistency",
        "xi_bnd_interpretation": "independent only when a matching Phase6 real-space row is present",
        "missing_result_A_inputs": missing_a,
        "missing_result_B_inputs": missing_b,
        "completed_result_A_primary_R": completed_A_R,
        "completed_result_B_dispersion_R": completed_B_R,
        "result_A_fit_points_at_least_4": bool(not primary_p.empty and (primary_p["n_points"] >= 4).all()),
        "result_A_bootstrap_ci_available": bool(not primary_p.empty and primary_p[["p_R_ci_low", "p_R_ci_high"]].notna().all().all()),
        "result_B_D_positive": valid_D,
        "result_B_qR_window_valid": valid_q_window,
        "result_B_collapse_rows": len(collapse),
        "direct_J_comparison_rows": len(direct_J),
        "direct_J_max_relative_difference": float(direct_J["Gamma_relative_difference"].max()) if not direct_J.empty else None,
        "independent_collapse_rows": len(independent),
        "complete_A": not missing_a and completed_A_R == requested_A_R,
        "complete_B": not missing_b and completed_B_R == requested_B_R and valid_D and valid_q_window,
    }
    validation_path = output_dir / "poster_AB_validation_summary.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    return [*paths.values(), summary_path, validation_path, *figure_paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--micro-root", type=Path, default=Path("results/runs/poster_ABCD/microscopic"))
    parser.add_argument("--pseudospinodal-root", type=Path, default=Path("results/runs/phase5_R_sweep"))
    parser.add_argument("--phase6-dir", type=Path, default=Path("results/runs/poster_ABCD/phase6_boundary"))
    parser.add_argument("--direct-J-root", type=Path, default=Path("results/runs/poster_ABCD/direct_J_reference"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/poster_ABCD"))
    parser.add_argument("--R-list", type=lambda value: parse_number_list(value, int), default=DEFAULT_R)
    parser.add_argument("--B-R-list", type=lambda value: parse_number_list(value, int), default=DEFAULT_B_R)
    parser.add_argument("--epsilon-fraction", type=float, default=0.05)
    parser.add_argument("--T-obs", type=int, default=50)
    parser.add_argument("--qR-max", type=float, default=0.35)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    for path in run_analysis(args):
        print(path)


if __name__ == "__main__":
    main()
