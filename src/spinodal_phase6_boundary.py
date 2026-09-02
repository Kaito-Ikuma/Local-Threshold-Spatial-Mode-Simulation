#!/usr/bin/env python3
"""Phase6 deterministic boundary response for independent real-space lengths.

Periodic Phase1-2 functions are intentionally left untouched.  This module
adds boundary-aware local averages, subtracts a same-boundary baseline, waits
for a steady state, and fits the response profile without fixing the length to
``sqrt(D/Gamma0)``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import curve_fit
from scipy.special import ndtr

from spinodal_phase0 import Phase0Task, run_phase0_case
from spinodal_phase34 import fit_power_law


SCRIPT_VERSION = "2026.08.31-phase6-boundary-v1"
BOUNDARY_TYPES = ("ghost_dirichlet", "open_fixed_denominator", "open_renormalized")
RIGHT_BOUNDARIES = ("reflecting",)
DEFAULT_DELTAS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
DEFAULT_EPSILON = (0.025, 0.05, 0.10)


@dataclass(frozen=True)
class BoundaryRun:
    R: int
    N: int
    delta: float
    epsilon_fraction: float
    boundary_type: str
    right_boundary: str
    m_star: float
    m_spinodal: float
    Delta: float
    mu: float
    sigma_eff: float
    Gamma0_theory: float
    xi_dyn: float
    lattice_spacing: float = 1.0


def parse_number_list(text: str, value_type: type = float) -> tuple[Any, ...]:
    values = tuple(value_type(token.strip()) for token in text.split(",") if token.strip())
    if not values:
        raise ValueError("list must contain at least one value")
    return values


def parse_N_map(text: str) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for token in text.split(","):
        if not token.strip():
            continue
        left, right = token.split(":", maxsplit=1)
        mapping[int(left)] = int(right)
    if not mapping:
        raise ValueError("N-map must contain R:N pairs")
    return mapping


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def boundary_local_average(
    values: np.ndarray,
    R: int,
    *,
    left_value: float,
    boundary_type: str,
    right_boundary: str = "reflecting",
) -> tuple[np.ndarray, np.ndarray]:
    """Return boundary-aware average and effective coordination count.

    ``ghost_dirichlet`` substitutes ``left_value`` for every index below zero.
    The two open variants omit those neighbours; the fixed-denominator variant
    retains the bulk ``2R`` denominator while the renormalized diagnostic uses
    the available coordination count.
    """
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("values must be one-dimensional")
    N = len(array)
    if R < 1 or 2 * R >= N:
        raise ValueError("boundary local average requires 1 <= R < N/2")
    if boundary_type not in BOUNDARY_TYPES:
        raise ValueError(f"unknown boundary type: {boundary_type}")
    if right_boundary not in RIGHT_BOUNDARIES:
        raise ValueError(f"unknown right boundary: {right_boundary}")
    total = np.zeros(N, dtype=float)
    count = np.zeros(N, dtype=int)
    sites = np.arange(N)
    for distance in range(1, R + 1):
        right = sites + distance
        beyond = right >= N
        if np.any(beyond):
            right = right.copy()
            right[beyond] = 2 * N - 1 - right[beyond]
        total += array[right]
        count += 1

        left = sites - distance
        present = left >= 0
        total[present] += array[left[present]]
        count[present] += 1
        if boundary_type == "ghost_dirichlet":
            total[~present] += left_value
            count[~present] += 1
    denominator = count.astype(float) if boundary_type == "open_renormalized" else np.full(N, 2.0 * R)
    if np.any(denominator <= 0.0):
        raise RuntimeError("boundary average encountered zero coordination")
    return total / denominator, count


def boundary_closure_step(
    values: np.ndarray,
    *,
    R: int,
    mu: float,
    Delta: float,
    sigma_eff: float,
    left_value: float,
    boundary_type: str,
    right_boundary: str = "reflecting",
) -> tuple[np.ndarray, np.ndarray]:
    if sigma_eff <= 0.0:
        raise ValueError("sigma_eff must be positive")
    local, coordination = boundary_local_average(
        values, R, left_value=left_value, boundary_type=boundary_type, right_boundary=right_boundary
    )
    updated = np.asarray(2.0 * ndtr((mu * local + Delta) / sigma_eff) - 1.0, dtype=float)
    if boundary_type.startswith("open_"):
        updated[0] = left_value
    return updated, coordination


def _finite_L_shape(x: np.ndarray, xi: float, L: float) -> np.ndarray:
    """Stable cosh((L-x)/xi)/cosh(L/xi) ratio."""
    xi = max(float(xi), np.finfo(float).tiny)
    return np.exp(-x / xi) * (1.0 + np.exp(-2.0 * np.maximum(L - x, 0.0) / xi)) / (1.0 + math.exp(-2.0 * L / xi))


def finite_L_profile(x: Sequence[float], amplitude: float, xi: float, L: float) -> np.ndarray:
    return float(amplitude) * _finite_L_shape(np.asarray(x, dtype=float), float(xi), float(L))


def exponential_profile(x: Sequence[float], amplitude: float, xi: float) -> np.ndarray:
    return float(amplitude) * np.exp(-np.asarray(x, dtype=float) / float(xi))


def _fit_metrics(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    residual = observed - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    centered = float(np.sum((observed - np.mean(observed)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / centered if centered > np.finfo(float).tiny else math.nan
    return rmse, r2


def fit_cosh_profile(
    x: Sequence[float], response: Sequence[float], *, L: float
) -> dict[str, float | int]:
    x_array = np.asarray(x, dtype=float)
    y = np.asarray(response, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y)
    x_array, y = x_array[valid], y[valid]
    if len(x_array) < 4 or np.ptp(x_array) <= 0.0:
        raise ValueError("cosh profile fit requires at least four points")
    amplitude0 = float(y[0]) if abs(float(y[0])) > np.finfo(float).tiny else float(np.max(np.abs(y)))
    xi0 = max(float(np.ptp(x_array)) / 5.0, 1.0)
    params, covariance = curve_fit(
        lambda coordinate, amplitude, xi: finite_L_profile(coordinate, amplitude, xi, L),
        x_array, y, p0=(amplitude0, xi0),
        bounds=([-np.inf, 1e-8], [np.inf, max(100.0 * L, 1.0)]), maxfev=50000,
    )
    predicted = finite_L_profile(x_array, params[0], params[1], L)
    rmse, r2 = _fit_metrics(y, predicted)
    return {
        "amplitude": float(params[0]), "xi": float(params[1]),
        "xi_se": float(math.sqrt(max(covariance[1, 1], 0.0))) if np.isfinite(covariance[1, 1]) else math.nan,
        "rmse": rmse, "r2": r2, "n_points": len(x_array),
    }


def fit_exponential_profile(x: Sequence[float], response: Sequence[float]) -> dict[str, float | int]:
    x_array = np.asarray(x, dtype=float)
    y = np.asarray(response, dtype=float)
    valid = np.isfinite(x_array) & np.isfinite(y)
    x_array, y = x_array[valid], y[valid]
    if len(x_array) < 4 or np.ptp(x_array) <= 0.0:
        raise ValueError("exponential profile fit requires at least four points")
    amplitude0 = float(y[0]) if abs(float(y[0])) > np.finfo(float).tiny else float(np.max(np.abs(y)))
    xi0 = max(float(np.ptp(x_array)) / 5.0, 1.0)
    params, covariance = curve_fit(
        exponential_profile, x_array, y, p0=(amplitude0, xi0),
        bounds=([-np.inf, 1e-8], [np.inf, max(100.0 * np.ptp(x_array), 1.0)]), maxfev=50000,
    )
    predicted = exponential_profile(x_array, params[0], params[1])
    rmse, r2 = _fit_metrics(y, predicted)
    return {
        "amplitude": float(params[0]), "xi": float(params[1]),
        "xi_se": float(math.sqrt(max(covariance[1, 1], 0.0))) if np.isfinite(covariance[1, 1]) else math.nan,
        "rmse": rmse, "r2": r2, "n_points": len(x_array),
    }


def simulate_boundary_pair(
    run: BoundaryRun,
    *,
    steady_tol: float,
    max_steps: int,
    minimum_tau_multiplier: float,
) -> dict[str, Any]:
    if steady_tol <= 0.0 or max_steps < 1 or minimum_tau_multiplier <= 0.0:
        raise ValueError("invalid steady-state controls")
    direction = run.m_spinodal - run.m_star
    epsilon = run.epsilon_fraction * abs(direction)
    left_forced = run.m_star + math.copysign(epsilon, direction)
    baseline = np.full(run.N, run.m_star, dtype=float)
    forced = np.full(run.N, run.m_star, dtype=float)
    if run.boundary_type.startswith("open_"):
        baseline[0] = run.m_star
        forced[0] = left_forced
    minimum_steps = max(1, int(math.ceil(minimum_tau_multiplier / run.Gamma0_theory)))
    step_budget = max(max_steps, minimum_steps)
    converged = False
    baseline_change = forced_change = math.inf
    coordination = np.full(run.N, 2 * run.R, dtype=int)
    for step in range(1, step_budget + 1):
        next_baseline, coordination = boundary_closure_step(
            baseline, R=run.R, mu=run.mu, Delta=run.Delta, sigma_eff=run.sigma_eff,
            left_value=run.m_star, boundary_type=run.boundary_type, right_boundary=run.right_boundary,
        )
        next_forced, _ = boundary_closure_step(
            forced, R=run.R, mu=run.mu, Delta=run.Delta, sigma_eff=run.sigma_eff,
            left_value=left_forced, boundary_type=run.boundary_type, right_boundary=run.right_boundary,
        )
        baseline_change = float(np.max(np.abs(next_baseline - baseline)))
        forced_change = float(np.max(np.abs(next_forced - forced)))
        baseline, forced = next_baseline, next_forced
        if step >= minimum_steps and max(baseline_change, forced_change) <= steady_tol:
            converged = True
            break
    response = forced - baseline
    return {
        "baseline": baseline, "forced": forced, "response": response,
        "coordination": coordination, "left_forced": left_forced,
        "epsilon_achieved": left_forced - run.m_star, "steps": step,
        "minimum_steps": minimum_steps, "step_budget": step_budget,
        "baseline_change": baseline_change, "forced_change": forced_change,
        "converged": converged,
    }


def analyze_boundary_run(
    run: BoundaryRun,
    *,
    steady_tol: float,
    max_steps: int,
    minimum_tau_multiplier: float,
    fit_x_min_factor: float,
) -> tuple[pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    simulation = simulate_boundary_pair(
        run, steady_tol=steady_tol, max_steps=max_steps, minimum_tau_multiplier=minimum_tau_multiplier
    )
    x = np.arange(run.N, dtype=float) * run.lattice_spacing
    L = float(x[-1])
    primary_x_min = fit_x_min_factor * run.R * run.lattice_spacing
    fit_x_max = max(primary_x_min + 4.0 * run.lattice_spacing, L - 2.0 * run.R * run.lattice_spacing)
    primary_mask = (x >= primary_x_min) & (x <= fit_x_max)
    response = np.asarray(simulation["response"], dtype=float)
    failed_fit = {"amplitude": math.nan, "xi": math.nan, "xi_se": math.nan, "rmse": math.nan, "r2": math.nan, "n_points": int(np.sum(primary_mask))}
    fit_errors: list[str] = []
    try:
        cosh_fit = fit_cosh_profile(x[primary_mask], response[primary_mask], L=L)
    except (ValueError, RuntimeError) as exc:
        cosh_fit = failed_fit.copy()
        fit_errors.append(f"cosh_fit_failed:{exc}")
    try:
        exp_fit = fit_exponential_profile(x[primary_mask], response[primary_mask])
    except (ValueError, RuntimeError) as exc:
        exp_fit = failed_fit.copy()
        fit_errors.append(f"exp_fit_failed:{exc}")
    amplitude = float(cosh_fit["amplitude"])
    if not math.isfinite(amplitude):
        selected_response = response[primary_mask]
        amplitude = float(selected_response[0]) if len(selected_response) else math.nan
    normalized = response / amplitude if abs(amplitude) > np.finfo(float).tiny else np.full_like(response, math.nan)
    predicted = (
        finite_L_profile(x, amplitude, float(cosh_fit["xi"]), L)
        if math.isfinite(float(cosh_fit["xi"])) and float(cosh_fit["xi"]) > 0.0
        else np.full_like(response, math.nan)
    )
    xi_value = float(cosh_fit["xi"])
    N_over_xi = run.N * run.lattice_spacing / xi_value if math.isfinite(xi_value) and xi_value > 0.0 else math.nan
    bulk_signal = float(np.max(np.abs(response[primary_mask]))) if np.any(primary_mask) else 0.0
    fit_reliable = bool(
        simulation["converged"]
        and math.isfinite(xi_value)
        and xi_value > 0.0
        and math.isfinite(N_over_xi)
        and N_over_xi >= 10.0
        and bulk_signal > 1e-14
    )
    reliability_reasons = list(fit_errors)
    if not simulation["converged"]:
        reliability_reasons.append("steady_state_not_converged")
    if not math.isfinite(xi_value) or xi_value <= 0.0:
        reliability_reasons.append("nonpositive_or_nonfinite_xi")
    if math.isfinite(N_over_xi) and N_over_xi < 10.0:
        reliability_reasons.append("N_over_xi_below_10")
    if bulk_signal <= 1e-14:
        reliability_reasons.append("bulk_response_below_numerical_floor")
    profile = pd.DataFrame(
        {
            "delta": run.delta, "R": run.R, "N": run.N, "x": x,
            "u_baseline": simulation["baseline"], "u_forced": simulation["forced"],
            "delta_u": response, "normalized_delta_u": normalized,
            "boundary_type": run.boundary_type, "right_boundary": run.right_boundary,
            "epsilon_fraction": run.epsilon_fraction,
            "coordination_count": simulation["coordination"],
            "coordination_ratio": np.asarray(simulation["coordination"]) / (2.0 * run.R),
            "cosh_fit_prediction": predicted, "cosh_fit_residual": response - predicted,
        }
    )
    fit_summary = {
        "delta": run.delta, "R": run.R, "N": run.N, "epsilon_fraction": run.epsilon_fraction,
        "boundary_type": run.boundary_type, "right_boundary": run.right_boundary,
        "xi_bnd_cosh": cosh_fit["xi"], "xi_bnd_cosh_se": cosh_fit["xi_se"],
        "xi_bnd_exp": exp_fit["xi"], "xi_dyn": run.xi_dyn, "xi_theory": run.xi_dyn,
        "xi_bnd_over_xi_dyn": xi_value / run.xi_dyn,
        "fit_r2": cosh_fit["r2"], "fit_rmse": cosh_fit["rmse"],
        "exp_fit_r2": exp_fit["r2"], "exp_fit_rmse": exp_fit["rmse"],
        "fit_x_min": primary_x_min, "fit_x_max": fit_x_max,
        "fit_range_selected_from_result": False,
        "converged": simulation["converged"], "steady_steps": simulation["steps"],
        "minimum_steps": simulation["minimum_steps"], "step_budget": simulation["step_budget"],
        "final_max_update": max(simulation["baseline_change"], simulation["forced_change"]),
        "N_over_xi_bnd": N_over_xi,
        "right_edge_relative_response": abs(float(response[-1])) / max(float(np.max(np.abs(response))), 1e-300),
        "primary_epsilon": math.isclose(run.epsilon_fraction, 0.05, rel_tol=0.0, abs_tol=1e-12),
        "fit_reliable": fit_reliable,
        "reliability_reason": "ok" if fit_reliable else ";".join(reliability_reasons),
    }
    systematics: list[dict[str, Any]] = []
    for factor in (1.0, 2.0, 3.0):
        x_min = factor * run.R * run.lattice_spacing
        mask = (x >= x_min) & (x <= fit_x_max)
        if int(np.sum(mask)) < 4:
            continue
        try:
            fit = fit_cosh_profile(x[mask], response[mask], L=L)
            fit_error = ""
        except (ValueError, RuntimeError) as exc:
            fit = failed_fit
            fit_error = str(exc)
        systematics.append(
            {"delta": run.delta, "R": run.R, "N": run.N, "epsilon_fraction": run.epsilon_fraction,
             "boundary_type": run.boundary_type, "fit_x_min_factor_R": factor,
             "fit_x_min": x_min, "fit_x_max": fit_x_max, "xi_cosh": fit["xi"],
             "rmse": fit["rmse"], "r2": fit["r2"], "primary_window": factor == fit_x_min_factor,
             "fit_error": fit_error}
        )
    return profile, fit_summary, systematics


def _make_figures(output_dir: Path, profiles: pd.DataFrame, xi: pd.DataFrame, scaling: pd.DataFrame) -> list[Path]:
    diagnostics = output_dir / "figures" / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    primary = xi[xi["primary_epsilon"] & xi["converged"]].copy()
    ghost = primary[(primary["boundary_type"] == "ghost_dirichlet") & primary["fit_reliable"]]
    profile_primary = profiles[np.isclose(profiles["epsilon_fraction"], 0.05, rtol=0.0, atol=1e-12)]
    if not ghost.empty:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        for _, row in ghost.iterrows():
            group = profile_primary[(profile_primary["R"] == row["R"]) & np.isclose(profile_primary["delta"], row["delta"]) & (profile_primary["boundary_type"] == "ghost_dirichlet")]
            ax.plot(group["x"], group["normalized_delta_u"], label=f"R={int(row['R'])}, delta={row['delta']:g}")
        ax.set(xlabel="x", ylabel=r"$\Delta u/A$", title="Result C: boundary response profiles")
        ax.legend(fontsize=7)
        path = diagnostics / "C1_boundary_profiles.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        for _, row in ghost.iterrows():
            group = profile_primary[(profile_primary["R"] == row["R"]) & np.isclose(profile_primary["delta"], row["delta"]) & (profile_primary["boundary_type"] == "ghost_dirichlet")]
            ax.plot(group["x"] / row["xi_bnd_cosh"], group["normalized_delta_u"], label=f"delta={row['delta']:g}")
        X = np.linspace(0.0, 6.0, 300); ax.plot(X, np.exp(-X), "k--", label="exp(-X)")
        ax.set(xlabel=r"$x/\xi_{bnd}$", ylabel=r"$\Delta u/A$", title="Boundary profile collapse")
        ax.set_xlim(0.0, 6.0); ax.legend(fontsize=7)
        path = diagnostics / "C2_boundary_profile_collapse.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        for R, group in ghost.groupby("R"):
            group = group.sort_values("delta"); ax.loglog(group["delta"], group["xi_bnd_cosh"], "o-", label=f"R={R}")
        primary_fit = scaling[(scaling["quantity"] == "xi_bnd_cosh") & scaling["primary_window"]] if not scaling.empty else pd.DataFrame()
        if not primary_fit.empty:
            fit = primary_fit.iloc[0]; xref = np.array([ghost["delta"].min(), min(3e-4, ghost["delta"].max())]); ax.loglog(xref, fit["amplitude"] * xref ** fit["exponent"], "k--", label=f"fit p={fit['exponent']:.3f}; ref -1/4")
        ax.set(xlabel=r"Gaussian distance $\delta$", ylabel=r"$\xi_{bnd}$", title="Result C: independent boundary length")
        ax.legend(fontsize=8)
        path = output_dir / "03_ResultC_boundary_length.png"; fig.savefig(path, dpi=220)
        diagnostic = diagnostics / "C3_xi_boundary_scaling.png"; fig.savefig(diagnostic, dpi=220)
        plt.close(fig); paths.extend((path, diagnostic))

        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        for R, group in ghost.groupby("R"):
            ax.semilogx(group["delta"], group["xi_bnd_over_xi_dyn"], "o-", label=f"R={R}")
        ax.axhline(1.0, color="black", linestyle="--")
        ax.set(xlabel=r"$\delta$", ylabel=r"$\xi_{bnd}/\xi_{dyn}$", title="Independent vs Fourier-derived length")
        ax.legend(); path = diagnostics / "C4_xi_boundary_vs_dynamic.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

    if not primary.empty:
        representative_delta = float(primary["delta"].max())
        subset = primary[np.isclose(primary["delta"], representative_delta)]
        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        for _, row in subset.iterrows():
            group = profile_primary[(profile_primary["R"] == row["R"]) & np.isclose(profile_primary["delta"], row["delta"]) & (profile_primary["boundary_type"] == row["boundary_type"])]
            group = group.sort_values("x")
            edge_amplitude = float(group["delta_u"].iloc[0])
            edge_normalized = group["delta_u"] / edge_amplitude if abs(edge_amplitude) > 1e-300 else np.full(len(group), math.nan)
            ax.plot(group["x"], edge_normalized, label=row["boundary_type"])
        ax.set(xlabel="x", ylabel=r"$\Delta u(x)/\Delta u(0)$", title="Boundary-condition profiles and boundary layer")
        ax.legend(fontsize=8)
        path = diagnostics / "D1_boundary_condition_profiles.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        for _, row in subset.iterrows():
            group = profile_primary[(profile_primary["R"] == row["R"]) & np.isclose(profile_primary["delta"], row["delta"]) & (profile_primary["boundary_type"] == row["boundary_type"])]
            bulk = group[(group["x"] >= row["fit_x_min"]) & (group["x"] <= row["fit_x_max"])].sort_values("x")
            if bulk.empty:
                continue
            bulk_amplitude = float(bulk["delta_u"].iloc[0])
            bulk_normalized = bulk["delta_u"] / bulk_amplitude if abs(bulk_amplitude) > 1e-300 else np.full(len(bulk), math.nan)
            bulk_coordinate = (bulk["x"] - float(row["fit_x_min"])) / float(row["xi_bnd_cosh"])
            ax.plot(bulk_coordinate, bulk_normalized, label=row["boundary_type"])
        X = np.linspace(0.0, 6.0, 300); ax.plot(X, np.exp(-X), "k--", label="exp(-X)")
        ax.set(xlim=(0.0, 6.0), xlabel=r"$(x-2R)/\xi_{bulk}$", ylabel=r"$\Delta u(x)/\Delta u(2R)$", title="Result D: bulk boundary-response comparison")
        ax.legend(fontsize=8); path = output_dir / "04_ResultD_boundary_response.png"; fig.savefig(path, dpi=220)
        diagnostic = diagnostics / "D3_boundary_scaled_profiles.png"; fig.savefig(diagnostic, dpi=220)
        plt.close(fig); paths.extend((path, diagnostic))

        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        for _, row in subset.iterrows():
            group = profile_primary[(profile_primary["R"] == row["R"]) & np.isclose(profile_primary["delta"], row["delta"]) & (profile_primary["boundary_type"] == row["boundary_type"])]
            positive = np.abs(group["delta_u"]) > 1e-15
            ax.semilogy(group.loc[positive, "x"], np.abs(group.loc[positive, "delta_u"]), label=row["boundary_type"])
        ax.set(xlabel="x", ylabel=r"$|\Delta u|$", title="Bulk exponential test"); ax.legend(fontsize=8)
        path = diagnostics / "D2_boundary_exponential_test.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)

        fig, ax1 = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        open_rows = profile_primary[(profile_primary["boundary_type"] == "open_fixed_denominator") & np.isclose(profile_primary["delta"], representative_delta)]
        if not open_rows.empty:
            ax1.plot(open_rows["x"], open_rows["coordination_ratio"], color="tab:blue", label="coordination ratio")
            ax2 = ax1.twinx(); ax2.plot(open_rows["x"], open_rows["cosh_fit_residual"], color="tab:red", label="response residual")
            ax1.set(xlabel="x", ylabel=r"$c_i/(2R)$", title="Boundary-layer diagnostic"); ax2.set_ylabel("cosh-fit residual")
            path = diagnostics / "D4_boundary_layer_diagnostic.png"; fig.savefig(path, dpi=220); plt.close(fig); paths.append(path)
        else:
            plt.close(fig)
    return paths


def run_phase6(args: argparse.Namespace) -> list[Path]:
    if args.N is not None and len(args.R_list) != 1:
        raise ValueError("--N can only be used with one R; use --N-map for an R sweep")
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_frames: list[pd.DataFrame] = []
    xi_rows: list[dict[str, Any]] = []
    systematics: list[dict[str, Any]] = []
    for R in args.R_list:
        N = args.N if args.N is not None else args.N_map.get(R)
        if N is None:
            raise ValueError(f"N is not specified for R={R}")
        phase0 = run_phase0_case(
            Phase0Task(B=args.B, R=R, sigma_J=args.sigma_J, sigma_phi=args.sigma_phi,
                       phi_bar=args.phi_bar, lattice_spacing=args.lattice_spacing,
                       branch=args.branch, delta_list=tuple(args.deltas))
        )
        for _, theory in phase0.delta_table.iterrows():
            for epsilon_fraction in args.epsilon_fractions:
                for boundary_type in args.boundary_types:
                    run = BoundaryRun(
                        R=R, N=N, delta=float(theory["delta"]), epsilon_fraction=float(epsilon_fraction),
                        boundary_type=boundary_type, right_boundary=args.right_boundary,
                        m_star=float(theory["m_star"]), m_spinodal=phase0.spinodal.m_spinodal,
                        Delta=float(theory["Delta"]), mu=phase0.spinodal.mu,
                        sigma_eff=phase0.spinodal.sigma_eff, Gamma0_theory=float(theory["Gamma0_theory"]),
                        xi_dyn=float(theory["xi_theory"]), lattice_spacing=args.lattice_spacing,
                    )
                    profile, fit, diagnostic = analyze_boundary_run(
                        run, steady_tol=args.steady_tol, max_steps=args.max_steps,
                        minimum_tau_multiplier=args.minimum_tau_multiplier,
                        fit_x_min_factor=args.fit_x_min_factor,
                    )
                    profile_frames.append(profile); xi_rows.append(fit); systematics.extend(diagnostic)
    profiles = pd.concat(profile_frames, ignore_index=True)
    xi = pd.DataFrame(xi_rows)
    systematics_table = pd.DataFrame(systematics)
    scaling_rows: list[dict[str, Any]] = []
    primary = xi[
        xi["primary_epsilon"] & xi["converged"] & xi["fit_reliable"] & (xi["boundary_type"] == "ghost_dirichlet")
        & (xi["delta"] <= args.primary_delta_max * (1.0 + 1e-12)) & (xi["xi_bnd_cosh"] > 0.0)
    ]
    for R, group in primary.groupby("R"):
        if len(group) >= 2:
            fit = fit_power_law(group["delta"], group["xi_bnd_cosh"])
            scaling_rows.append(
                {"R": R, "quantity": "xi_bnd_cosh", "window": f"delta<={args.primary_delta_max:g}",
                 "n_points": len(group), "amplitude": fit["amplitude"], "exponent": fit["exponent"],
                 "exponent_se": fit["exponent_regression_se"], "r2": fit["r2"],
                 "expected_exponent": -0.25, "difference": float(fit["exponent"]) + 0.25,
                 "primary_window": True, "window_selected_from_result": False}
            )
    scaling = pd.DataFrame(scaling_rows)
    c_profiles = output_dir / "resultC_boundary_profiles.csv"
    c_xi = output_dir / "resultC_xi_boundary.csv"
    c_scaling = output_dir / "resultC_xi_scaling.csv"
    d_comparison = output_dir / "resultD_boundary_comparison.csv"
    d_fit = output_dir / "resultD_boundary_fit_summary.csv"
    systematics_path = output_dir / "resultC_boundary_fit_systematics.csv"
    profiles.to_csv(c_profiles, index=False)
    xi.to_csv(c_xi, index=False)
    scaling.to_csv(c_scaling, index=False)
    profiles[["boundary_type", "right_boundary", "delta", "R", "N", "x", "delta_u", "normalized_delta_u", "coordination_count", "coordination_ratio", "cosh_fit_residual"]].to_csv(d_comparison, index=False)
    xi.rename(columns={"xi_bnd_cosh": "xi_cosh", "xi_bnd_exp": "xi_exp_bulk", "fit_rmse": "rmse", "fit_r2": "r2"}).to_csv(d_fit, index=False)
    systematics_table.to_csv(systematics_path, index=False)
    validation = {
        "script_version": SCRIPT_VERSION, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit_sha": _git_sha(), "command_line": sys.argv,
        "python_version": platform.python_version(), "numpy_version": np.__version__,
        "scipy_version": scipy.__version__, "seed": None, "deterministic": True,
        "R_N": {str(R): int(args.N if args.N is not None else args.N_map[R]) for R in args.R_list},
        "delta_definition": "Gaussian spinodal distance for deterministic closure",
        "boundary_types": list(args.boundary_types), "right_boundary": args.right_boundary,
        "primary_fit_window": f"x>={args.fit_x_min_factor:g}R; fixed before results",
        "steady_tol": args.steady_tol, "minimum_tau_multiplier": args.minimum_tau_multiplier,
        "all_converged": bool(xi["converged"].all()),
        "minimum_N_over_xi_bnd": float(xi["N_over_xi_bnd"].min()),
        "all_xi_positive": bool((xi["xi_bnd_cosh"] > 0.0).all()),
        "all_primary_ghost_fits_reliable": bool(xi[(xi["primary_epsilon"]) & (xi["boundary_type"] == "ghost_dirichlet")]["fit_reliable"].all()),
        "max_right_edge_relative_response": float(xi["right_edge_relative_response"].max()),
        "periodic_implementation_modified": False,
        "xi_dyn_fixed_in_profile_fit": False,
        "finite_R_wording": "Phase6 C/D is deterministic Gaussian closure; microscopic finite-R results remain pseudospinodal-like",
    }
    validation_path = output_dir / "phase6_validation_summary.json"
    validation_path.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")
    figures = _make_figures(output_dir, profiles, xi, scaling) if args.figures else []
    return [c_profiles, c_xi, c_scaling, d_comparison, d_fit, systematics_path, validation_path, *figures]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--B", type=float, default=2.0)
    parser.add_argument("--R-list", type=lambda value: parse_number_list(value, int), default=(12,))
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--N-map", type=parse_N_map, default={6: 512, 12: 1024, 24: 2048, 48: 4096, 96: 8192})
    parser.add_argument("--sigma-J", type=float, default=1.0, dest="sigma_J")
    parser.add_argument("--sigma-phi", type=float, default=0.06, dest="sigma_phi")
    parser.add_argument("--phi-bar", type=float, default=0.0, dest="phi_bar")
    parser.add_argument("--a", type=float, default=1.0, dest="lattice_spacing")
    parser.add_argument("--branch", choices=("stay_to_evacuate", "evacuate_to_stay"), default="stay_to_evacuate")
    parser.add_argument("--deltas", type=parse_number_list, default=DEFAULT_DELTAS)
    parser.add_argument("--epsilon-fractions", type=parse_number_list, default=DEFAULT_EPSILON)
    parser.add_argument("--boundary-types", type=lambda value: parse_number_list(value, str), choices=None, default=BOUNDARY_TYPES)
    parser.add_argument("--right-boundary", choices=RIGHT_BOUNDARIES, default="reflecting")
    parser.add_argument("--steady-tol", type=float, default=1e-12)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--minimum-tau-multiplier", type=float, default=10.0)
    parser.add_argument("--fit-x-min-factor", type=float, default=2.0)
    parser.add_argument("--primary-delta-max", type=float, default=3e-4)
    parser.add_argument("--figures", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/poster_ABCD/phase6_boundary"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    invalid = sorted(set(args.boundary_types) - set(BOUNDARY_TYPES))
    if invalid:
        raise SystemExit(f"unknown boundary types: {invalid}")
    for path in run_phase6(args):
        print(path)


if __name__ == "__main__":
    main()
