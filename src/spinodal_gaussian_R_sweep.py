#!/usr/bin/env python3
"""Systematic interaction-range sweep for Gaussian Spinodal Phase0--4.

The driver deliberately reuses the existing serial Phase0, Phase1--2, and
Phase3--4 APIs.  Keeping N/R fixed gives every mode index the same qR at every
range, so finite-q systematics can be compared without changing resolution.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spinodal_phase0 import Phase0Task, run_phase0_case, write_phase0_outputs
from spinodal_phase12 import (
    Phase0Reference,
    build_phase12_tasks,
    simulate_deterministic_mode,
    write_phase12_outputs,
)
from spinodal_phase34 import (
    load_phase34_inputs,
    run_phase34_analysis,
    write_phase34_outputs,
)


SCRIPT_VERSION = "2026.08.16-gaussian-R-sweep-v1"
DEFAULT_R_VALUES = (6, 12, 24, 48)
DEFAULT_DELTAS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
DEFAULT_MODES = (0, 1, 2, 3, 4, 5, 6)


def parse_number_list(text: str, value_type: type = float) -> tuple[Any, ...]:
    try:
        values = tuple(value_type(token.strip()) for token in text.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected a comma-separated number list") from exc
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def lattice_size_for_range(
    R: int, *, reference_R: int = 12, reference_N: int = 1024
) -> int:
    """Return round(N_ref R/R_ref), rejecting invalid lattice geometries."""
    if R < 1 or reference_R < 1 or reference_N < 1:
        raise ValueError("R, reference_R, and reference_N must be positive")
    N = int(round(reference_N * R / reference_R))
    if N <= 2 * R:
        raise ValueError(f"mapped N={N} must exceed 2R={2 * R}")
    return N


def build_R_N_mapping(
    R_values: Sequence[int], *, reference_R: int = 12, reference_N: int = 1024
) -> dict[int, int]:
    ranges = tuple(int(value) for value in R_values)
    if not ranges or len(set(ranges)) != len(ranges):
        raise ValueError("R values must be non-empty and unique")
    return {
        R: lattice_size_for_range(R, reference_R=reference_R, reference_N=reference_N)
        for R in ranges
    }


def qR_for_mode(mode_index: int, N: int, R: int) -> float:
    return 2.0 * math.pi * int(mode_index) * int(R) / int(N)


def validate_common_qR(
    mapping: dict[int, int], modes: Sequence[int], *, atol: float = 1e-13
) -> dict[int, list[float]]:
    values = {
        int(R): [qR_for_mode(mode, N, R) for mode in modes]
        for R, N in mapping.items()
    }
    reference = np.asarray(next(iter(values.values())), dtype=float)
    for R, row in values.items():
        if not np.allclose(row, reference, rtol=0.0, atol=atol):
            raise ValueError(f"qR grid differs at R={R}")
    return values


def _reference_from_phase0(result: Any, directory: Path) -> Phase0Reference:
    return Phase0Reference(
        phase0_dir=directory,
        summary={
            "inputs": {
                "B": result.task.B,
                "R": result.task.R,
                "sigma_J": result.task.sigma_J,
                "sigma_phi": result.task.sigma_phi,
                "phi_bar": result.task.phi_bar,
                "a": result.task.lattice_spacing,
                "branch": result.task.branch,
            },
            **asdict(result.spinodal),
        },
        delta_table=result.delta_table,
        regenerated=False,
    )


def _primary_row(frame: pd.DataFrame, quantity: str) -> pd.Series:
    selected = frame[(frame["quantity"] == quantity) & frame["primary_window"]]
    if len(selected) != 1:
        raise ValueError(f"expected one primary {quantity} fit, found {len(selected)}")
    return selected.iloc[0]


def _all_true(mapping: dict[str, Any]) -> bool:
    return bool(mapping) and all(bool(value) for value in mapping.values())


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


def run_gaussian_R_case(
    *,
    R: int,
    N: int,
    B: float,
    sigma_J: float,
    sigma_phi: float,
    phi_bar: float,
    lattice_spacing: float,
    branch: str,
    deltas: Sequence[float],
    modes: Sequence[int],
    epsilon_fraction: float,
    qR_max_fit: float,
    primary_delta_max: float,
    qR_max_collapse: float,
    output_root: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    case_dir = output_root / f"R{R:03d}"
    phase0_dir = case_dir / "phase0"
    phase12_dir = case_dir / "phase12"
    phase34_dir = case_dir / "phase34"
    total_start = time.perf_counter()

    phase0_result = run_phase0_case(
        Phase0Task(
            B=B,
            R=R,
            sigma_J=sigma_J,
            sigma_phi=sigma_phi,
            phi_bar=phi_bar,
            lattice_spacing=lattice_spacing,
            branch=branch,
            delta_list=tuple(float(value) for value in deltas),
        )
    )
    write_phase0_outputs(phase0_result, phase0_dir)

    reference = _reference_from_phase0(phase0_result, phase0_dir)
    tasks = build_phase12_tasks(
        reference,
        deltas=deltas,
        modes=modes,
        N=N,
        epsilon_fraction=epsilon_fraction,
        qR_max_fit=qR_max_fit,
        task_group="main",
    )
    phase12_start = time.perf_counter()
    mode_results = [simulate_deterministic_mode(task) for task in tasks]
    phase12_seconds = time.perf_counter() - phase12_start
    write_phase12_outputs(
        mode_results,
        phase12_dir,
        qR_max_fit=qR_max_fit,
        runtime_metadata={
            "mode_simulation_seconds": phase12_seconds,
            "mpi_used": False,
            "N": N,
            "R": R,
        },
    )

    phase34_inputs = load_phase34_inputs(phase0_dir, phase12_dir)
    phase34_start = time.perf_counter()
    phase34 = run_phase34_analysis(
        phase34_inputs,
        primary_delta_max=primary_delta_max,
        qR_max_collapse=qR_max_collapse,
    )
    phase34_seconds = time.perf_counter() - phase34_start
    write_phase34_outputs(
        phase34,
        phase34_dir,
        runtime={
            "analysis_seconds": phase34_seconds,
            "mpi_used": False,
        },
    )

    phase12_summary = json.loads(
        (phase12_dir / "phase12_validation_summary.json").read_text(encoding="utf-8")
    )
    phase34_summary = phase34.validation_summary
    gamma_fit = _primary_row(phase34.phase3_powerlaw_fits, "Gamma0_num")
    xi_fit = _primary_row(phase34.phase4_scaling_fits, "xi_dyn_raw")
    z_fit = _primary_row(phase34.phase4_scaling_fits, "tau0_vs_xi_raw")
    nearest_delta = float(phase34.phase3_scaling["delta"].min())
    nearest_systematics = phase34.phase4_systematics.loc[
        phase34.phase4_systematics["delta"].idxmin()
    ]
    nearest_length = phase34.phase4_lengths.loc[
        phase34.phase4_lengths["delta"].idxmin()
    ]
    kappa = float(phase0_result.spinodal.kappa_R_theory)
    C_gamma_theory = float(phase34_summary["phase3"]["C_gamma_theory"])
    C_gamma_fit = float(gamma_fit["amplitude"])
    D_ratio = phase34.phase4_systematics["D_fit_raw"].to_numpy(dtype=float) / kappa

    summary_row = {
        "R": R,
        "N": N,
        "N_over_R": N / R,
        "sigma_eff": phase0_result.spinodal.sigma_eff,
        "mu": phase0_result.spinodal.mu,
        "Delta_spinodal": phase0_result.spinodal.Delta_spinodal,
        "m_spinodal": phase0_result.spinodal.m_spinodal,
        "kappa_R": kappa,
        "C_Gamma_theory": C_gamma_theory,
        "C_Gamma_fit": C_gamma_fit,
        "C_Gamma_ratio": C_gamma_fit / C_gamma_theory,
        "C_xi_theory": float(phase34_summary["phase4"]["C_xi_theory"]),
        "p_Gamma": float(gamma_fit["exponent"]),
        "p_Gamma_se": float(gamma_fit["exponent_regression_se"]),
        "p_tau": float(phase34_summary["phase3"]["tau_exponent_primary"]),
        "p_xi": float(xi_fit["exponent"]),
        "p_xi_se": float(xi_fit["exponent_regression_se"]),
        "z": float(z_fit["exponent"]),
        "D_nearest_spinodal": float(nearest_systematics["D_fit_raw"]),
        "D_over_kappa_nearest": float(nearest_systematics["D_fit_raw"]) / kappa,
        "max_D_over_kappa_deviation": float(np.max(np.abs(D_ratio - 1.0))),
        "xi_nearest": float(nearest_length["xi_dyn_raw"]),
        "xi_over_R_nearest": float(nearest_length["xi_dyn_raw"]) / R,
        "collapse_rms": float(phase34_summary["collapse"]["rms_error"]),
        "collapse_max_abs": float(phase34_summary["collapse"]["max_absolute_error"]),
        "max_exact_kernel_absolute_error": float(
            phase12_summary["phase2_exact_kernel"]["max_absolute_error_excluding_q0"]
        ),
        "nearest_q4_D2_over_kappa": float(nearest_systematics["D2_q4_fit"]) / kappa,
        "nearest_q4_D4_relative_error": float(
            nearest_systematics["D4_relative_error_to_c4"]
        ),
        "min_N_over_xi": float(phase34.phase4_lengths["N_over_xi_raw"].min()),
        "max_qmin_xi": float(phase34.phase4_lengths["qmin_xi_raw"].max()),
        "phase12_all_checks_passed": _all_true(phase12_summary["soft_checks"]),
        "phase34_all_checks_passed": _all_true(phase34_summary["soft_checks"]),
    }

    delta_table = phase34.phase3_scaling.merge(
        phase34.phase4_systematics[
            ["delta", "D_fit_raw", "kappa_R_theory"]
        ],
        on="delta",
        validate="one_to_one",
    ).merge(
        phase34.phase4_lengths[
            [
                "delta",
                "xi_dyn_raw",
                "xi_theory_phase0",
                "N_over_xi_raw",
                "qmin_xi_raw",
            ]
        ],
        on="delta",
        validate="one_to_one",
    )
    delta_table = pd.DataFrame(
        {
            "R": R,
            "N": N,
            "delta": delta_table["delta"],
            "Gamma0_num": delta_table["Gamma0_num"],
            "Gamma0_theory": delta_table["Gamma0_theory"],
            "Gamma_ratio": delta_table["Gamma0_num"] / delta_table["Gamma0_theory"],
            "D_fit": delta_table["D_fit_raw"],
            "kappa_R": delta_table["kappa_R_theory"],
            "D_over_kappa": delta_table["D_fit_raw"] / delta_table["kappa_R_theory"],
            "xi_dyn": delta_table["xi_dyn_raw"],
            "xi_theory": delta_table["xi_theory_phase0"],
            "xi_ratio": delta_table["xi_dyn_raw"] / delta_table["xi_theory_phase0"],
            "xi_over_R": delta_table["xi_dyn_raw"] / R,
            "N_over_xi": delta_table["N_over_xi_raw"],
            "qmin_xi": delta_table["qmin_xi_raw"],
        }
    )

    mode_table = pd.read_csv(phase12_dir / "phase12_mode_results.csv")
    mode_table.insert(0, "R_sweep", R)
    case_summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "fixed dimensionless Gaussian control B; mu varies with R",
        "summary": summary_row,
        "qR_by_mode": {
            str(int(mode)): qR_for_mode(int(mode), N, R) for mode in modes
        },
        "nearest_spinodal_delta": nearest_delta,
        "C_xi_theory": phase34_summary["phase4"]["C_xi_theory"],
        "phase12_soft_checks": phase12_summary["soft_checks"],
        "phase34_soft_checks": phase34_summary["soft_checks"],
        "total_wall_seconds": time.perf_counter() - total_start,
    }
    (case_dir / "gaussian_R_case_summary.json").write_text(
        json.dumps(_json_safe(case_summary), indent=2) + "\n", encoding="utf-8"
    )
    return summary_row, delta_table, mode_table


def make_combined_figures(
    summary: pd.DataFrame,
    delta_table: pd.DataFrame,
    mode_table: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    def save(name: str) -> Path:
        path = output_dir / name
        paths[name.removesuffix(".png")] = path
        return path

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for R, group in delta_table.groupby("R"):
        ax.loglog(group["delta"], group["Gamma0_num"], "o-", label=f"R={R}")
    ax.set(xlabel=r"$\delta$", ylabel=r"$\Gamma_0$", title="Gaussian closure: range sweep")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save("gaussian_R_gamma_scaling.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.errorbar(summary["R"], summary["p_Gamma"], yerr=summary["p_Gamma_se"], fmt="o-")
    ax.axhline(0.5, color="black", linestyle="--", label="1/2")
    ax.set(xlabel="R", ylabel=r"$p_\Gamma$", xscale="log", title="Critical exponent versus range")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save("gaussian_R_exponent_vs_R.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for R, group in delta_table.groupby("R"):
        ax.semilogx(group["delta"], group["D_over_kappa"], "o-", label=f"R={R}")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set(xlabel=r"$\delta$", ylabel=r"$D/\kappa_R$", title="Normalized spatial coefficient")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save("gaussian_R_D_over_kappa.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.loglog(summary["R"], summary["kappa_R"], "o-", label=r"exact discrete $\kappa_R$")
    ax.loglog(summary["R"], summary["R"] ** 2 / 6.0, "--", label=r"$R^2/6$")
    ax.set(xlabel="R", ylabel=r"$\kappa_R$", title="Top-hat long-wave coefficient")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save("gaussian_R_kappa_vs_R.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for R, group in delta_table.groupby("R"):
        ax.loglog(group["delta"], group["xi_over_R"], "o-", label=f"R={R}")
    ax.set(xlabel=r"$\delta$", ylabel=r"$\xi_{dyn}/R$", title="Range-normalized dynamic length")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save("gaussian_R_xi_over_R.png"), dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.loglog(summary["R"], summary["collapse_rms"], "o-")
    ax.set(xlabel="R", ylabel="collapse RMS", title="Long-wave collapse quality")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(save("gaussian_R_collapse_quality.png"), dpi=220)
    plt.close(fig)

    nearest = float(mode_table["delta"].min())
    selected = mode_table[np.isclose(mode_table["delta"], nearest)].copy()
    q_nonzero = selected["q"].to_numpy(dtype=float) != 0.0
    selected = selected[q_nonzero]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for R, group in selected.groupby("R_sweep"):
        gamma0 = float(
            mode_table[
                (mode_table["R_sweep"] == R)
                & np.isclose(mode_table["delta"], nearest)
                & (mode_table["mode_index"] == 0)
            ]["Gamma_from_lambda"].iloc[0]
        )
        normalized = (group["Gamma_from_lambda"] - gamma0) / (
            group["kappa_R_theory"] * group["q"] ** 2
        )
        ax.plot(group["qR"], normalized, "o-", label=f"R={int(R)}")
    ax.axhline(1.0, color="black", linestyle="--", label="long-wave limit")
    ax.set(
        xlabel=r"$qR$",
        ylabel=r"$[\Gamma(q)-\Gamma_0]/(\kappa_Rq^2)$",
        title=rf"Normalized dispersion at $\delta={nearest:g}$",
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save("gaussian_R_normalized_dispersion.png"), dpi=220)
    plt.close(fig)
    return paths


def run_sweep(args: argparse.Namespace) -> dict[str, Path]:
    mapping = build_R_N_mapping(
        args.R_list, reference_R=args.reference_R, reference_N=args.reference_N
    )
    qR_grid = validate_common_qR(mapping, args.modes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    delta_frames = []
    mode_frames = []
    for R, N in mapping.items():
        print(f"[Gaussian R sweep] R={R}, N={N}, N/R={N / R:g}", flush=True)
        summary, delta_table, mode_table = run_gaussian_R_case(
            R=R,
            N=N,
            B=args.B,
            sigma_J=args.sigma_J,
            sigma_phi=args.sigma_phi,
            phi_bar=args.phi_bar,
            lattice_spacing=args.lattice_spacing,
            branch=args.branch,
            deltas=args.delta_list,
            modes=args.modes,
            epsilon_fraction=args.epsilon_fraction,
            qR_max_fit=args.qR_max_fit,
            primary_delta_max=args.primary_delta_max,
            qR_max_collapse=args.qR_max_collapse,
            output_root=args.output_dir,
        )
        summary_rows.append(summary)
        delta_frames.append(delta_table)
        mode_frames.append(mode_table)

    summary_table = pd.DataFrame(summary_rows).sort_values("R")
    delta_table = pd.concat(delta_frames, ignore_index=True).sort_values(["R", "delta"])
    mode_table = pd.concat(mode_frames, ignore_index=True).sort_values(
        ["R_sweep", "delta", "mode_index"]
    )
    summary_path = args.output_dir / "gaussian_R_sweep_summary.csv"
    delta_path = args.output_dir / "gaussian_R_sweep_delta_table.csv"
    validation_path = args.output_dir / "gaussian_R_sweep_validation_summary.json"
    summary_table.to_csv(summary_path, index=False)
    delta_table.to_csv(delta_path, index=False)
    figure_paths = make_combined_figures(summary_table, delta_table, mode_table, args.output_dir)

    qR_array = np.asarray(list(qR_grid.values()), dtype=float)
    validation = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "R_values": list(mapping),
            "N_mapping": {str(R): N for R, N in mapping.items()},
            "N_over_R": {str(R): N / R for R, N in mapping.items()},
            "B_fixed": args.B,
            "mu_fixed": False,
            "interpretation": "same dimensionless Gaussian spinodal control B; mu and sigma_eff vary with R",
        },
        "qR_validation": {
            "same_mode_same_qR": bool(np.max(np.ptp(qR_array, axis=0)) < 1e-13),
            "qR_by_mode": {
                str(mode): float(qR_array[0, index]) for index, mode in enumerate(args.modes)
            },
        },
        "critical_scaling": summary_table[
            ["R", "p_Gamma", "p_Gamma_se", "p_tau", "p_xi", "p_xi_se", "z"]
        ].to_dict(orient="records"),
        "spatial_coefficient": summary_table[
            ["R", "kappa_R", "D_over_kappa_nearest", "max_D_over_kappa_deviation"]
        ].to_dict(orient="records"),
        "finite_size": delta_table[
            ["R", "delta", "N_over_xi", "qmin_xi"]
        ].to_dict(orient="records"),
        "exact_kernel_relation": summary_table[
            ["R", "max_exact_kernel_absolute_error"]
        ].to_dict(orient="records"),
        "finite_q_q4_systematics": summary_table[
            ["R", "nearest_q4_D2_over_kappa", "nearest_q4_D4_relative_error"]
        ].to_dict(orient="records"),
        "all_phase12_checks_passed": bool(summary_table["phase12_all_checks_passed"].all()),
        "all_phase34_checks_passed": bool(summary_table["phase34_all_checks_passed"].all()),
        "wording": "Gaussian closure range dependence; no microscopic mean-field convergence is inferred",
    }
    validation_path.write_text(
        json.dumps(_json_safe(validation), indent=2) + "\n", encoding="utf-8"
    )
    return {
        "summary": summary_path,
        "delta_table": delta_path,
        "validation": validation_path,
        **figure_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the serial Gaussian deterministic Phase0--4 interaction-range sweep.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--R-list", type=lambda value: parse_number_list(value, int), default=DEFAULT_R_VALUES)
    parser.add_argument("--reference-R", type=int, default=12)
    parser.add_argument("--reference-N", type=int, default=1024)
    parser.add_argument("--B", type=float, default=2.0)
    parser.add_argument("--sigma-J", type=float, default=1.0, dest="sigma_J")
    parser.add_argument("--sigma-phi", type=float, default=0.06, dest="sigma_phi")
    parser.add_argument("--phi-bar", type=float, default=0.0, dest="phi_bar")
    parser.add_argument("--a", type=float, default=1.0, dest="lattice_spacing")
    parser.add_argument("--branch", choices=("stay_to_evacuate", "evacuate_to_stay"), default="stay_to_evacuate")
    parser.add_argument("--delta-list", type=parse_number_list, default=DEFAULT_DELTAS)
    parser.add_argument("--modes", type=lambda value: parse_number_list(value, int), default=DEFAULT_MODES)
    parser.add_argument("--epsilon-fraction", type=float, default=0.05)
    parser.add_argument("--qR-max-fit", type=float, default=0.35)
    parser.add_argument("--primary-delta-max", type=float, default=3e-4)
    parser.add_argument("--qR-max-collapse", type=float, default=0.35)
    parser.add_argument("--output-dir", type=Path, default=Path("results/runs/gaussian_R_sweep"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = run_sweep(args)
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
