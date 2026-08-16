#!/usr/bin/env python3
"""Plan and combine finite-range microscopic/Gaussian R-sweep outputs."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spinodal_gaussian_R_sweep import (
    DEFAULT_R_VALUES,
    build_R_N_mapping,
    parse_number_list,
)
from spinodal_phase5_followup_analysis import interpolate_escape_crossing


SCRIPT_VERSION = "2026.08.16-R-sweep-analysis-v1"
DEFAULT_FIXED_DELTAS = (0.08, 0.10, 0.12)
DEFAULT_MATCHED_OFFSETS = (0.005, 0.010, 0.020)


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


def fine_grid_from_bracket(lower: float, upper: float, max_step: float = 0.002) -> tuple[float, ...]:
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError("fine bracket must satisfy 0 < lower < upper")
    if not math.isfinite(max_step) or max_step <= 0.0:
        raise ValueError("max_step must be positive")
    intervals = max(1, int(math.ceil((upper - lower) / max_step)))
    return tuple(float(value) for value in np.linspace(lower, upper, intervals + 1))


def plan_fine_scan(
    deltas: Sequence[float],
    probabilities: Sequence[float],
    *,
    criterion: float = 0.10,
    max_step: float = 0.002,
    extension_factor: float = 1.5,
) -> dict[str, Any]:
    if extension_factor <= 1.0:
        raise ValueError("extension_factor must exceed 1")
    delta = np.asarray(deltas, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    if len(delta) < 1 or len(delta) != len(probability):
        raise ValueError("delta and probability arrays must have equal nonzero length")
    crossing = interpolate_escape_crossing(delta, probability, criterion)
    if crossing["estimate"] is not None:
        grid = fine_grid_from_bracket(
            float(crossing["lower"]), float(crossing["upper"]), max_step=max_step
        )
        return {
            "status": "bracketed",
            "delta_lower": crossing["lower"],
            "delta_upper": crossing["upper"],
            "coarse_estimate": crossing["estimate"],
            "fine_deltas": grid,
            "extension_direction": None,
            "extension_delta": None,
            "monotonicity_ok": crossing["monotonicity_ok"],
        }
    order = np.argsort(delta)
    delta = delta[order]
    probability = probability[order]
    if np.all(probability < criterion):
        direction = "smaller_delta"
        extension = float(delta[0] / extension_factor)
        status = "extension_required"
    elif np.all(probability > criterion):
        direction = "larger_delta"
        extension = float(delta[-1] * extension_factor)
        status = "extension_required"
    else:
        direction = "manual_nonmonotonic_review"
        extension = math.nan
        status = "ambiguous"
    return {
        "status": status,
        "delta_lower": None,
        "delta_upper": None,
        "coarse_estimate": None,
        "fine_deltas": (),
        "extension_direction": direction,
        "extension_delta": extension,
        "monotonicity_ok": crossing["monotonicity_ok"],
    }


def _format_values(values: Sequence[float]) -> str:
    return ",".join(f"{float(value):.12g}" for value in values)


def _load_coarse_table(micro_root: Path, R: int, primary_T: int) -> pd.DataFrame:
    paths = sorted(
        (micro_root / f"R{R:03d}").glob(
            "pseudospinodal_coarse*/analysis/phase5_pseudospinodal_fine_scan.csv"
        )
    )
    if not paths:
        raise FileNotFoundError(f"no coarse analysis table found for R={R}")
    tables = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"delta", "T", "escape_fraction_cumulative_T"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        tables.append(frame[frame["T"] == primary_T])
    combined = pd.concat(tables, ignore_index=True)
    combined = combined.groupby("delta", as_index=False)["escape_fraction_cumulative_T"].mean()
    return combined.sort_values("delta")


def write_fine_plan(args: argparse.Namespace) -> Path:
    mapping = build_R_N_mapping(
        args.R_list, reference_R=args.reference_R, reference_N=args.reference_N
    )
    rows = []
    for R, N in mapping.items():
        table = _load_coarse_table(args.micro_root, R, args.primary_T)
        plan = plan_fine_scan(
            table["delta"],
            table["escape_fraction_cumulative_T"],
            criterion=args.criterion_probability,
            max_step=args.max_step,
            extension_factor=args.extension_factor,
        )
        rows.append(
            {
                "R": R,
                "N": N,
                "N_over_R": N / R,
                "status": plan["status"],
                "delta_lower": plan["delta_lower"],
                "delta_upper": plan["delta_upper"],
                "coarse_estimate": plan["coarse_estimate"],
                "fine_deltas": _format_values(plan["fine_deltas"]),
                "extension_direction": plan["extension_direction"],
                "extension_delta": plan["extension_delta"],
                "monotonicity_ok": plan["monotonicity_ok"],
                "criterion_probability": args.criterion_probability,
                "primary_T": args.primary_T,
                "max_fine_step": args.max_step,
                "extension_factor": args.extension_factor,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    return args.output


def value_from_plan(plan_path: Path, R: int, column: str) -> str:
    frame = pd.read_csv(plan_path, dtype={"fine_deltas": str})
    row = frame[frame["R"] == R]
    if len(row) != 1:
        raise ValueError(f"expected one plan row for R={R}")
    value = row.iloc[0][column]
    if pd.isna(value) or str(value).strip() == "":
        raise ValueError(f"fine scan is not bracketed for R={R}; inspect {plan_path}")
    return str(value)


def pseudospinodal_at_T(time_table: Path, T: int = 50) -> float:
    frame = pd.read_csv(time_table)
    row = frame[frame["T_obs"] == T]
    if len(row) != 1 or pd.isna(row["delta_ps_estimate"].iloc[0]):
        raise ValueError(f"no bracketed delta_ps at T={T} in {time_table}")
    return float(row["delta_ps_estimate"].iloc[0])


def matched_deltas(time_table: Path, offsets: Sequence[float], T: int = 50) -> tuple[float, ...]:
    estimate = pseudospinodal_at_T(time_table, T=T)
    return tuple(estimate + float(offset) for offset in offsets)


def classify_response_coordinate(
    observed: Sequence[float],
    *,
    fixed_deltas: Sequence[float] = DEFAULT_FIXED_DELTAS,
    delta_ps: float | None = None,
    matched_offsets: Sequence[float] = DEFAULT_MATCHED_OFFSETS,
    atol: float = 5e-10,
) -> str:
    values = np.sort(np.asarray(observed, dtype=float))
    fixed = np.sort(np.asarray(fixed_deltas, dtype=float))
    if len(values) == len(fixed) and np.allclose(values, fixed, rtol=0.0, atol=atol):
        return "fixed_gaussian_delta"
    if delta_ps is not None:
        matched = np.sort(float(delta_ps) + np.asarray(matched_offsets, dtype=float))
        if len(values) == len(matched) and np.allclose(values, matched, rtol=0.0, atol=atol):
            return "operational_pseudospinodal_matched"
    raise ValueError("response delta grid matches neither fixed nor operational-matched coordinates")


def collect_pseudospinodal(
    gaussian: pd.DataFrame, micro_root: Path, R_values: Sequence[int]
) -> pd.DataFrame:
    rows = []
    for R in R_values:
        path = (
            micro_root
            / f"R{R:03d}"
            / "pseudospinodal_fine"
            / "analysis"
            / "phase5_pseudospinodal_time_dependence.csv"
        )
        if not path.is_file():
            continue
        time = pd.read_csv(path).set_index("T_obs")
        gaussian_row = gaussian[gaussian["R"] == R]
        if len(gaussian_row) != 1:
            raise ValueError(f"Gaussian summary missing R={R}")

        def get(T: int, column: str = "delta_ps_estimate") -> float:
            return float(time.loc[T, column]) if T in time.index else math.nan

        delta50 = get(50)
        delta30 = get(30)
        rows.append(
            {
                "R": R,
                "N": int(gaussian_row["N"].iloc[0]),
                "Delta_spinodal_Gaussian": float(gaussian_row["Delta_spinodal"].iloc[0]),
                "delta_ps_T20": get(20),
                "delta_ps_T30": delta30,
                "delta_ps_T40": get(40),
                "delta_ps_T50": delta50,
                "delta_ps_T50_se": get(50, "delta_ps_se"),
                "delta_ps_T50_ci_low": get(50, "delta_ps_ci_low"),
                "delta_ps_T50_ci_high": get(50, "delta_ps_ci_high"),
                "delta_ps_shift_from_gaussian": delta50,
                "time_shift_30_to_50": delta50 - delta30,
                "rounding_diagnostic": abs(delta50 - delta30),
            }
        )
    return pd.DataFrame(rows)


def _comparison_lookup(summary: dict[str, Any]) -> dict[tuple[float, int], dict[str, Any]]:
    comparisons = summary.get("survival", {}).get("comparisons", [])
    return {
        (round(float(row["delta"]), 12), int(row["mode_index"])): row
        for row in comparisons
    }


def _gamma_eff_ratio(table: pd.DataFrame, delta: float, mode: int) -> float:
    selected = table[
        np.isclose(table["delta"], delta, rtol=0.0, atol=5e-10)
        & (table["mode_index"] == mode)
    ]
    t0 = selected[selected["t"] == 0]
    t3 = selected[selected["t"] == 3]
    if len(t0) != 1 or len(t3) != 1:
        return math.nan
    denominator = float(t0["Gamma_eff"].iloc[0])
    return float(t3["Gamma_eff"].iloc[0]) / denominator if denominator != 0.0 else math.nan


def _late_amplitude_ratios(table: pd.DataFrame, delta: float, mode: int) -> tuple[float, float, float]:
    selected = table[
        np.isclose(table["delta"], delta, rtol=0.0, atol=5e-10)
        & (table["mode_index"] == mode)
    ].sort_values("t")
    if selected.empty:
        return math.nan, math.nan, math.nan
    first = selected.iloc[0]
    last = selected.iloc[-1]
    uncond = float(last["A_unconditional"]) / float(first["A_unconditional"])
    survive = float(last["A_survive_to_T"]) / float(first["A_survive_to_T"])
    escape = float(last["escape_fraction_cumulative"])
    return uncond, survive, escape


def collect_responses(
    micro_root: Path,
    pseudo: pd.DataFrame,
    R_values: Sequence[int],
) -> pd.DataFrame:
    rows = []
    pseudo_lookup = pseudo.set_index("R") if not pseudo.empty else pd.DataFrame()
    for R in R_values:
        delta_ps = (
            float(pseudo_lookup.loc[R, "delta_ps_T50"])
            if not pseudo.empty and R in pseudo_lookup.index
            else None
        )
        for directory_name, expected_coordinate in (
            ("response_fixed_delta", "fixed_gaussian_delta"),
            ("response_matched", "operational_pseudospinodal_matched"),
        ):
            root = micro_root / f"R{R:03d}" / directory_name
            mode_path = root / "phase5_mode_results.csv"
            analysis_dir = root / "analysis"
            summary_path = analysis_dir / "phase5_followup_validation_summary.json"
            gamma_path = analysis_dir / "phase5_gamma_eff.csv"
            survival_path = analysis_dir / "phase5_survival_conditioned.csv"
            if not all(path.is_file() for path in (mode_path, summary_path, gamma_path, survival_path)):
                continue
            mode = pd.read_csv(mode_path)
            coordinate = classify_response_coordinate(
                mode["delta"].unique(), delta_ps=delta_ps
            )
            if coordinate != expected_coordinate:
                raise ValueError(f"coordinate mismatch in {root}: {coordinate}")
            followup = json.loads(summary_path.read_text(encoding="utf-8"))
            comparisons = _comparison_lookup(followup)
            gamma_eff = pd.read_csv(gamma_path)
            survival = pd.read_csv(survival_path)
            for _, item in mode.iterrows():
                delta = float(item["delta"])
                mode_index = int(item["mode_index"])
                comparison = comparisons.get((round(delta, 12), mode_index), {})
                gamma_u = float(comparison.get("Gamma_unconditional", item["Gamma_micro"]))
                gamma_s = float(comparison.get("Gamma_survive_to_T", math.nan))
                closure = float(item["Gamma_closure"])
                late_u, late_s, escape = _late_amplitude_ratios(
                    survival, delta, mode_index
                )
                rows.append(
                    {
                        "R": R,
                        "N": int(item["N"]),
                        "coordinate": coordinate,
                        "delta": delta,
                        "delta_offset_from_ps": delta - delta_ps if delta_ps is not None else math.nan,
                        "mode_index": mode_index,
                        "qR": float(item["qR"]),
                        "Gamma_unconditional": gamma_u,
                        "Gamma_survive_to_T": gamma_s,
                        "Gamma_closure": closure,
                        "Gamma_unconditional_over_closure": gamma_u / closure,
                        "Gamma_survive_to_T_over_closure": gamma_s / closure,
                        "Gamma_survive_to_T_over_unconditional": gamma_s / gamma_u,
                        "escape_correction_fraction": abs(gamma_s - gamma_u) / abs(gamma_u),
                        "Gamma_eff_t3_over_t0": _gamma_eff_ratio(gamma_eff, delta, mode_index),
                        "method_B_C_relative_difference": float(item["method_B_C_relative_difference"]),
                        "late_A_unconditional_ratio": late_u,
                        "late_A_survive_to_T_ratio": late_s,
                        "escape_fraction_cumulative_T": escape,
                    }
                )
    return pd.DataFrame(rows)


def collect_dispersion(micro_root: Path, R_values: Sequence[int]) -> pd.DataFrame:
    rows = []
    for R in R_values:
        path = micro_root / f"R{R:03d}" / "dispersion" / "phase5_dispersion_fits.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        for _, row in frame.iterrows():
            rows.append(
                {
                    "R": R,
                    "delta": float(row["delta"]),
                    "D_micro": float(row["D_micro"]),
                    "D_micro_se": float(row["D_micro_se"]),
                    "D_closure": float(row["D_closure"]),
                    "kappa_R": float(row["kappa_R"]),
                    "D_micro_over_kappa": float(row["D_micro"]) / float(row["kappa_R"]),
                    "D_closure_over_kappa": float(row["D_closure"]) / float(row["kappa_R"]),
                    "n_modes_used": int(row["n_modes_used"]),
                }
            )
    return pd.DataFrame(rows)


def collect_benchmarks(micro_root: Path, R_values: Sequence[int]) -> pd.DataFrame:
    rows = []
    for R in R_values:
        path = micro_root / f"R{R:03d}" / "benchmark" / "benchmarks" / "phase5_block_size_benchmark.csv"
        if not path.is_file():
            continue
        frame = pd.read_csv(path)
        selected = frame[frame["block_size"] == 64]
        if selected.empty:
            selected = frame.iloc[[0]]
        row = selected.iloc[0].to_dict()
        row["R"] = R
        rows.append(row)
    return pd.DataFrame(rows)


def _first_or_nan(frame: pd.DataFrame, column: str) -> float:
    return float(frame[column].iloc[0]) if not frame.empty else math.nan


def build_combined_summary(
    gaussian: pd.DataFrame,
    pseudo: pd.DataFrame,
    responses: pd.DataFrame,
    dispersion: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, g in gaussian.sort_values("R").iterrows():
        R = int(g["R"])
        p = pseudo[pseudo["R"] == R] if not pseudo.empty else pd.DataFrame()
        fixed = (
            responses[
                (responses["R"] == R)
                & (responses["coordinate"] == "fixed_gaussian_delta")
                & (responses["mode_index"] == 0)
            ]
            if not responses.empty
            else pd.DataFrame()
        )
        matched = (
            responses[
                (responses["R"] == R)
                & (responses["coordinate"] == "operational_pseudospinodal_matched")
                & (responses["mode_index"] == 0)
            ].copy()
            if not responses.empty
            else pd.DataFrame()
        )
        if not matched.empty:
            matched["offset_distance"] = abs(matched["delta_offset_from_ps"] - 0.010)
            matched = matched.sort_values("offset_distance").iloc[[0]]
        d = dispersion[dispersion["R"] == R] if not dispersion.empty else pd.DataFrame()

        def fixed_ratio(delta: float) -> float:
            if fixed.empty:
                return math.nan
            row = fixed[np.isclose(fixed["delta"], delta, rtol=0.0, atol=5e-10)]
            return _first_or_nan(row, "Gamma_unconditional_over_closure")

        rows.append(
            {
                "R": R,
                "N": int(g["N"]),
                "N_over_R": float(g["N_over_R"]),
                "sigma_eff": float(g["sigma_eff"]),
                "mu": float(g["mu"]),
                "kappa_R": float(g["kappa_R"]),
                "gaussian_pGamma": float(g["p_Gamma"]),
                "gaussian_pXi": float(g["p_xi"]),
                "gaussian_z": float(g["z"]),
                "gaussian_D_over_kappa": float(g["D_over_kappa_nearest"]),
                "micro_delta_ps_T50": _first_or_nan(p, "delta_ps_T50"),
                "micro_delta_ps_time_shift": _first_or_nan(p, "time_shift_30_to_50"),
                "micro_Gamma_ratio_fixed_delta_008": fixed_ratio(0.08),
                "micro_Gamma_ratio_fixed_delta_010": fixed_ratio(0.10),
                "micro_Gamma_ratio_fixed_delta_012": fixed_ratio(0.12),
                "micro_survival_Gamma_ratio": _first_or_nan(matched, "Gamma_survive_to_T_over_closure"),
                "micro_escape_correction": _first_or_nan(matched, "escape_correction_fraction"),
                "micro_Gamma_eff_ratio_t3_t0": _first_or_nan(matched, "Gamma_eff_t3_over_t0"),
                "micro_method_B_C_difference": _first_or_nan(matched, "method_B_C_relative_difference"),
                "micro_D_over_kappa": _first_or_nan(d, "D_micro_over_kappa"),
            }
        )
    return pd.DataFrame(rows)


def _empty_figure(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.text(0.5, 0.5, "SQUID data not available", ha="center", va="center")
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_micro_figures(
    pseudo: pd.DataFrame, responses: pd.DataFrame, dispersion: pd.DataFrame, output_dir: Path
) -> list[Path]:
    paths = [
        output_dir / "micro_R_gamma_ratio.png",
        output_dir / "micro_R_gamma_ratio_matched.png",
        output_dir / "micro_R_pseudospinodal_shift.png",
        output_dir / "micro_R_pseudospinodal_time_dependence.png",
        output_dir / "micro_R_escape_correction.png",
        output_dir / "micro_R_nonexponentiality.png",
        output_dir / "micro_R_D_over_kappa.png",
    ]
    if responses.empty:
        _empty_figure(paths[0], "Fixed Gaussian-delta response")
        _empty_figure(paths[1], "Operational-matched response")
        _empty_figure(paths[4], "Survival correction")
        _empty_figure(paths[5], "Early-time nonexponentiality")
    else:
        q0 = responses[responses["mode_index"] == 0]
        fixed = q0[q0["coordinate"] == "fixed_gaussian_delta"]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for delta, group in fixed.groupby("delta"):
            ax.plot(group["R"], group["Gamma_unconditional_over_closure"], "o-", label=f"delta={delta:g}")
        ax.axhline(1.0, color="black", linestyle="--")
        ax.set(xlabel="R", ylabel=r"$\Gamma_{micro}/\Gamma_{closure}$", title="Fixed Gaussian distance")
        ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(paths[0], dpi=220); plt.close(fig)

        matched = q0[q0["coordinate"] == "operational_pseudospinodal_matched"]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for offset, group in matched.groupby(matched["delta_offset_from_ps"].round(6)):
            ax.plot(group["R"], group["Gamma_survive_to_T_over_closure"], "o-", label=f"offset={offset:g}")
        ax.axhline(1.0, color="black", linestyle="--")
        ax.set(xlabel="R", ylabel=r"$\Gamma_{survive}/\Gamma_{closure}$", title="Operational-matched distance")
        ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(paths[1], dpi=220); plt.close(fig)

        primary = matched[np.isclose(matched["delta_offset_from_ps"], 0.010, atol=5e-5)]
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(primary["R"], primary["Gamma_survive_to_T_over_unconditional"], "o-")
        ax.axhline(1.0, color="black", linestyle="--")
        ax.set(xlabel="R", ylabel=r"$\Gamma_{survive}/\Gamma_{unconditional}$", title="Survival correction")
        ax.grid(True, alpha=0.25); fig.tight_layout(); fig.savefig(paths[4], dpi=220); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
        axes[0].plot(primary["R"], primary["Gamma_eff_t3_over_t0"], "o-")
        axes[0].axhline(1.0, color="black", linestyle="--")
        axes[0].set(xlabel="R", ylabel=r"$\Gamma_{eff}(3)/\Gamma_{eff}(0)$")
        axes[1].plot(primary["R"], primary["method_B_C_relative_difference"], "o-")
        axes[1].set(xlabel="R", ylabel="Method B/C relative difference")
        for ax in axes: ax.grid(True, alpha=0.25)
        fig.suptitle("Early-time nonexponentiality diagnostics"); fig.tight_layout(); fig.savefig(paths[5], dpi=220); plt.close(fig)

    if pseudo.empty:
        _empty_figure(paths[2], "Operational pseudospinodal-like shift")
        _empty_figure(paths[3], "Observation-time dependence")
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.errorbar(pseudo["R"], pseudo["delta_ps_T50"], yerr=pseudo["delta_ps_T50_se"], fmt="o-")
        ax.axhline(0.0, color="black", linestyle="--", label="Gaussian spinodal")
        ax.set(xlabel="R", ylabel=r"$\delta_{ps}^{10\%,T=50}$", title="Operational pseudospinodal-like shift")
        ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(paths[2], dpi=220); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        for _, row in pseudo.iterrows():
            times = np.array([20, 30, 40, 50])
            values = np.array([row[f"delta_ps_T{T}"] for T in times], dtype=float)
            ax.plot(times, values, "o-", label=f"R={int(row['R'])}")
        ax.set(xlabel=r"$T_{obs}$", ylabel=r"$\delta_{ps}$", title="Observation-time dependence")
        ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(paths[3], dpi=220); plt.close(fig)

    if dispersion.empty:
        _empty_figure(paths[6], "Microscopic normalized dispersion")
    else:
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.errorbar(
            dispersion["R"],
            dispersion["D_micro_over_kappa"],
            yerr=dispersion["D_micro_se"] / dispersion["kappa_R"],
            fmt="o-",
            label="microscopic",
        )
        ax.plot(dispersion["R"], dispersion["D_closure_over_kappa"], "s--", label="closure")
        ax.axhline(1.0, color="black", linestyle=":")
        ax.set(xlabel="R", ylabel=r"$D/\kappa_R$", title="Normalized finite-q coefficient")
        ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout(); fig.savefig(paths[6], dpi=220); plt.close(fig)
    return paths


def analyze_command(args: argparse.Namespace) -> list[Path]:
    gaussian_path = args.gaussian_dir / "gaussian_R_sweep_summary.csv"
    if not gaussian_path.is_file():
        raise FileNotFoundError(f"missing Gaussian R-sweep summary: {gaussian_path}")
    gaussian = pd.read_csv(gaussian_path)
    R_values = tuple(int(value) for value in args.R_list)
    pseudo = collect_pseudospinodal(gaussian, args.micro_root, R_values)
    responses = collect_responses(args.micro_root, pseudo, R_values)
    dispersion = collect_dispersion(args.micro_root, R_values)
    benchmarks = collect_benchmarks(args.micro_root, R_values)
    combined = build_combined_summary(gaussian, pseudo, responses, dispersion)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_path = args.output_dir / "microscopic_R_pseudospinodal.csv"
    response_path = args.output_dir / "microscopic_R_response.csv"
    dispersion_path = args.output_dir / "microscopic_R_dispersion.csv"
    benchmark_path = args.output_dir / "microscopic_R_benchmark.csv"
    combined_path = args.output_dir / "R_sweep_combined_summary.csv"
    validation_path = args.output_dir / "R_sweep_combined_validation_summary.json"
    pseudo.to_csv(pseudo_path, index=False)
    responses.to_csv(response_path, index=False)
    dispersion.to_csv(dispersion_path, index=False)
    benchmarks.to_csv(benchmark_path, index=False)
    combined.to_csv(combined_path, index=False)
    figure_paths = make_micro_figures(pseudo, responses, dispersion, args.output_dir)
    missing_stages = []
    if len(benchmarks) < len(R_values): missing_stages.append("M1 benchmark")
    if len(pseudo) < len(R_values): missing_stages.append("M3 fine pseudospinodal")
    if responses.empty: missing_stages.append("M4/M5 response")
    if dispersion.empty: missing_stages.append("M6 dispersion")
    validation = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "R_values": list(R_values),
        "fixed_delta_coordinate": list(DEFAULT_FIXED_DELTAS),
        "matched_coordinate_offsets": list(DEFAULT_MATCHED_OFFSETS),
        "coordinates_kept_separate": True,
        "pseudospinodal_wording": "operational observation-time-dependent 10%-escape crossover; not a critical point",
        "rounding_diagnostic": "abs(delta_ps_T50 - delta_ps_T30)",
        "power_law_R_fit_primary": False,
        "missing_stages": missing_stages,
        "complete": not missing_stages,
    }
    validation_path.write_text(json.dumps(_json_safe(validation), indent=2) + "\n", encoding="utf-8")
    return [pseudo_path, response_path, dispersion_path, benchmark_path, combined_path, validation_path, *figure_paths]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan-fine", help="build deterministic fine grids from coarse scans")
    plan.add_argument("--micro-root", type=Path, default=Path("results/runs/phase5_R_sweep"))
    plan.add_argument("--R-list", type=lambda value: parse_number_list(value, int), default=DEFAULT_R_VALUES)
    plan.add_argument("--reference-R", type=int, default=12)
    plan.add_argument("--reference-N", type=int, default=1024)
    plan.add_argument("--criterion-probability", type=float, default=0.10)
    plan.add_argument("--primary-T", type=int, default=50)
    plan.add_argument("--max-step", type=float, default=0.002)
    plan.add_argument("--extension-factor", type=float, default=1.5)
    plan.add_argument("--output", type=Path, default=Path("results/runs/phase5_R_sweep/fine_scan_plan.csv"))

    show = sub.add_parser("print-plan-value", help="print one field for shell scripts")
    show.add_argument("--plan", type=Path, required=True)
    show.add_argument("--R", type=int, required=True)
    show.add_argument("--column", default="fine_deltas")

    matched = sub.add_parser("print-matched-deltas", help="print delta_ps(T)+offsets")
    matched.add_argument("--time-table", type=Path, required=True)
    matched.add_argument("--offsets", type=parse_number_list, default=DEFAULT_MATCHED_OFFSETS)
    matched.add_argument("--T", type=int, default=50)

    analyze = sub.add_parser("analyze", help="combine completed Gaussian and SQUID R sweeps")
    analyze.add_argument("--gaussian-dir", type=Path, default=Path("results/runs/gaussian_R_sweep"))
    analyze.add_argument("--micro-root", type=Path, default=Path("results/runs/phase5_R_sweep"))
    analyze.add_argument("--R-list", type=lambda value: parse_number_list(value, int), default=DEFAULT_R_VALUES)
    analyze.add_argument("--output-dir", type=Path, default=Path("results/runs/R_sweep_combined"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan-fine":
        print(write_fine_plan(args))
    elif args.command == "print-plan-value":
        print(value_from_plan(args.plan, args.R, args.column))
    elif args.command == "print-matched-deltas":
        print(_format_values(matched_deltas(args.time_table, args.offsets, T=args.T)))
    elif args.command == "analyze":
        for path in analyze_command(args):
            print(path)


if __name__ == "__main__":
    main()
