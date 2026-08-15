#!/usr/bin/env python3
"""Phase0 theory setup for spinodal-near spatial-mode simulations.

This module locates the analytic spinodal of the Gaussian mean-field map,
continues the stable metastable fixed point on either side of the hysteresis
loop, and reports theory-only time and length scales for later simulation
phases.  The numerical core is serial and independent of MPI and file I/O.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import root_scalar

from spatial_mode_ensemble_validation import (
    Config,
    lambda_star,
    mean_field_map,
    mu_from_B,
    normal_cdf_scalar,
    sigma_eff,
)


SCRIPT_VERSION = "2026.08.15-phase0-v1"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DELTAS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)
BRANCH_SIGNS = {
    "stay_to_evacuate": -1,
    "evacuate_to_stay": 1,
}


@dataclass(frozen=True)
class Phase0Task:
    """One independent Phase0 parameter task, suitable for future mapping."""

    B: float = 2.0
    R: int = 12
    sigma_J: float = 1.0
    sigma_phi: float = 0.06
    phi_bar: float = 0.0
    lattice_spacing: float = 1.0
    branch: str = "stay_to_evacuate"
    delta_list: tuple[float, ...] = DEFAULT_DELTAS


@dataclass(frozen=True)
class SpinodalResult:
    B: float
    branch: str
    branch_sign: int
    sigma_eff: float
    mu: float
    z_spinodal: float
    m_spinodal: float
    Delta_spinodal: float
    h_spinodal: float
    fixed_point_residual: float
    tangency_residual: float
    kappa_R_theory: float


@dataclass(frozen=True)
class Phase0Result:
    task: Phase0Task
    config: Config
    spinodal: SpinodalResult
    delta_table: pd.DataFrame


def _validate_task(task: Phase0Task) -> None:
    if task.branch not in BRANCH_SIGNS:
        raise ValueError(
            f"Unknown branch {task.branch!r}; choose one of {sorted(BRANCH_SIGNS)}."
        )
    if not math.isfinite(task.B) or task.B <= 1.0:
        raise ValueError("No spinodal exists for B <= 1 in this mean-field map.")
    if task.R < 1:
        raise ValueError("R must be >= 1.")
    if (
        not math.isfinite(task.sigma_J)
        or not math.isfinite(task.sigma_phi)
        or task.sigma_J < 0.0
        or task.sigma_phi < 0.0
    ):
        raise ValueError("sigma_J and sigma_phi must be non-negative and finite.")
    if not math.isfinite(task.phi_bar):
        raise ValueError("phi_bar must be finite.")
    if task.sigma_J == 0.0 and task.sigma_phi == 0.0:
        raise ValueError("sigma_eff must be positive.")
    if not math.isfinite(task.lattice_spacing) or task.lattice_spacing <= 0.0:
        raise ValueError("lattice spacing a must be positive and finite.")
    if not task.delta_list:
        raise ValueError("delta_list must contain at least one value.")
    if any((not math.isfinite(delta) or delta <= 0.0) for delta in task.delta_list):
        raise ValueError("Every delta must be positive and finite.")


def config_from_task(task: Phase0Task) -> Config:
    """Build the existing model Config without changing its default behavior."""
    _validate_task(task)
    return Config(
        R=task.R,
        lattice_spacing=task.lattice_spacing,
        B=task.B,
        sigma_J=task.sigma_J,
        sigma_phi=task.sigma_phi,
        phi_bar=task.phi_bar,
        h=task.phi_bar,
        dynamics="gaussian_map",
    )


def kappa_R_theory(R: int, lattice_spacing: float) -> float:
    """Long-wavelength top-hat-kernel coefficient (theory, not a fit)."""
    if R < 1:
        raise ValueError("R must be >= 1.")
    if lattice_spacing <= 0.0:
        raise ValueError("lattice spacing a must be positive.")
    return lattice_spacing**2 * (R + 1) * (2 * R + 1) / 12.0


def compute_spinodal(config: Config, branch: str = "stay_to_evacuate") -> SpinodalResult:
    """Evaluate the analytic Gaussian-map spinodal and verify both conditions."""
    if branch not in BRANCH_SIGNS:
        raise ValueError(
            f"Unknown branch {branch!r}; choose one of {sorted(BRANCH_SIGNS)}."
        )
    if not math.isfinite(config.B) or config.B <= 1.0:
        raise ValueError("No spinodal exists for B <= 1 in this mean-field map.")

    sign = BRANCH_SIGNS[branch]
    sig = sigma_eff(config)
    mu = mu_from_B(config)
    z_spinodal = sign * math.sqrt(2.0 * math.log(config.B))
    m_spinodal = 2.0 * normal_cdf_scalar(z_spinodal) - 1.0
    Delta_spinodal = sig * z_spinodal - mu * m_spinodal
    h_spinodal = config.phi_bar + Delta_spinodal

    spinodal_config = replace(config, h=h_spinodal)
    fixed_point_residual = mean_field_map(m_spinodal, spinodal_config) - m_spinodal
    tangency_residual = lambda_star(m_spinodal, spinodal_config) - 1.0

    if abs(fixed_point_residual) > 1e-10 or abs(tangency_residual) > 1e-10:
        raise RuntimeError(
            "Analytic spinodal verification failed: "
            f"fixed-point residual={fixed_point_residual:.3e}, "
            f"tangency residual={tangency_residual:.3e}."
        )

    return SpinodalResult(
        B=config.B,
        branch=branch,
        branch_sign=sign,
        sigma_eff=sig,
        mu=mu,
        z_spinodal=z_spinodal,
        m_spinodal=m_spinodal,
        Delta_spinodal=Delta_spinodal,
        h_spinodal=h_spinodal,
        fixed_point_residual=fixed_point_residual,
        tangency_residual=tangency_residual,
        kappa_R_theory=kappa_R_theory(config.R, config.lattice_spacing),
    )


def solve_metastable_fixed_point(
    config: Config,
    spinodal: SpinodalResult,
    delta: float,
) -> dict[str, float | bool | str]:
    """Solve the explicitly selected stable metastable branch at ``delta > 0``.

    For ``stay_to_evacuate`` the root is bracketed on [-1, m_spinodal].
    For ``evacuate_to_stay`` it is bracketed on [m_spinodal, 1].  This avoids
    selecting a nearby unstable root through a fixed-point guess.
    """
    if not math.isfinite(delta) or delta <= 0.0:
        raise ValueError("delta must be positive and finite.")

    Delta = spinodal.Delta_spinodal + spinodal.branch_sign * delta
    h = config.phi_bar + Delta
    delta_config = replace(config, h=h)

    if spinodal.branch == "stay_to_evacuate":
        bracket = (-1.0, spinodal.m_spinodal)
    else:
        bracket = (spinodal.m_spinodal, 1.0)

    def residual(m: float) -> float:
        return mean_field_map(m, delta_config) - m

    endpoint_values = (residual(bracket[0]), residual(bracket[1]))
    if endpoint_values[0] * endpoint_values[1] > 0.0:
        raise RuntimeError(
            f"Could not bracket the {spinodal.branch} metastable root at "
            f"delta={delta:g}; endpoint residuals={endpoint_values}."
        )

    root = root_scalar(
        residual,
        bracket=bracket,
        method="brentq",
        xtol=1e-14,
        rtol=1e-14,
        maxiter=200,
    )
    if not root.converged:
        raise RuntimeError(
            f"Metastable fixed-point solver did not converge at delta={delta:g}."
        )

    m_star = float(root.root)
    sig = sigma_eff(delta_config)
    mu = mu_from_B(delta_config)
    z_star = (mu * m_star + Delta) / sig
    Lambda = lambda_star(m_star, delta_config)
    fixed_point_residual = residual(m_star)
    stable = 0.0 < Lambda < 1.0
    if not stable:
        raise RuntimeError(
            f"Selected root is not a stable metastable fixed point at delta={delta:g}: "
            f"m_star={m_star:.16g}, Lambda_star={Lambda:.16g}."
        )

    Gamma0 = -math.log(Lambda)
    tau0 = 1.0 / Gamma0
    xi = math.sqrt(spinodal.kappa_R_theory / Gamma0)

    return {
        "branch": spinodal.branch,
        "delta": float(delta),
        "Delta": Delta,
        "h": h,
        "m_star": m_star,
        "z_star": z_star,
        "Lambda_star": Lambda,
        "Gamma0_theory": Gamma0,
        "tau0_theory": tau0,
        "kappa_R_theory": spinodal.kappa_R_theory,
        "xi_theory": xi,
        "fixed_point_residual": fixed_point_residual,
        "stable": stable,
    }


def build_phase0_table(
    config: Config,
    spinodal: SpinodalResult,
    delta_list: Sequence[float],
) -> pd.DataFrame:
    """Return one independent theory row per requested distance to spinodal."""
    rows = [
        solve_metastable_fixed_point(config, spinodal, float(delta))
        for delta in delta_list
    ]
    columns = [
        "branch",
        "delta",
        "Delta",
        "h",
        "m_star",
        "z_star",
        "Lambda_star",
        "Gamma0_theory",
        "tau0_theory",
        "kappa_R_theory",
        "xi_theory",
        "fixed_point_residual",
        "stable",
    ]
    return pd.DataFrame(rows, columns=columns)


def run_phase0_case(task: Phase0Task) -> Phase0Result:
    """Compute one Phase0 task without MPI, plotting, or file I/O."""
    config = config_from_task(task)
    spinodal = compute_spinodal(config, task.branch)
    table = build_phase0_table(config, spinodal, task.delta_list)
    return Phase0Result(task=task, config=config, spinodal=spinodal, delta_table=table)


def make_fixed_point_branch_figure(result: Phase0Result, output_path: Path) -> None:
    """Plot the analytic fixed-point manifold and selected metastable points."""
    spinodal = result.spinodal
    z_extent = max(4.0, abs(spinodal.z_spinodal) + 1.5)
    z_values = np.linspace(-z_extent, z_extent, 5001)
    m_values = np.array(
        [2.0 * normal_cdf_scalar(float(z)) - 1.0 for z in z_values]
    )
    Delta_values = spinodal.sigma_eff * z_values - spinodal.mu * m_values
    Lambda_values = result.task.B * np.exp(-0.5 * z_values**2)
    stable_values = np.ma.masked_where(Lambda_values >= 1.0, m_values)
    unstable_values = np.ma.masked_where(Lambda_values <= 1.0, m_values)

    table = result.delta_table.sort_values("Delta")
    fig, ax = plt.subplots(figsize=(8.0, 5.5), constrained_layout=True)
    ax.plot(Delta_values, stable_values, color="0.45", lw=1.8, label="stable fixed points")
    ax.plot(
        Delta_values,
        unstable_values,
        color="0.55",
        lw=1.5,
        linestyle="--",
        label="unstable fixed points",
    )
    ax.plot(
        table["Delta"],
        table["m_star"],
        "o-",
        color="tab:blue",
        lw=2.0,
        label=f"target metastable branch ({spinodal.branch})",
    )
    ax.scatter(
        [spinodal.Delta_spinodal],
        [spinodal.m_spinodal],
        marker="*",
        s=190,
        color="tab:red",
        edgecolor="black",
        linewidth=0.5,
        zorder=5,
        label="spinodal",
    )
    ax.axvline(0.0, color="0.8", lw=0.8)
    ax.axhline(0.0, color="0.8", lw=0.8)
    ax.set_xlabel(r"$\Delta=h-\bar\phi$")
    ax.set_ylabel(r"fixed point $m$")
    ax.set_title("Gaussian mean-field fixed-point branches (Phase0 theory)")
    ax.legend(fontsize=8, loc="best")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def make_theory_scales_figure(result: Phase0Result, output_path: Path) -> None:
    """Plot theory predictions used to size later simulations."""
    table = result.delta_table.sort_values("delta")
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), constrained_layout=True)
    for ax, column, ylabel in [
        (axes[0], "Gamma0_theory", r"$\Gamma_0^{\rm th}=-\ln\Lambda_*$"),
        (axes[1], "tau0_theory", r"$\tau_0^{\rm th}$"),
        (axes[2], "xi_theory", r"$\xi_{\rm th}$"),
    ]:
        ax.loglog(table["delta"], table[column], "o-")
        ax.set_xlabel(r"distance to spinodal $\delta$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)
    fig.suptitle("Gaussian-map theory predictions for later simulation phases")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def write_phase0_outputs(result: Phase0Result, output_dir: Path) -> dict[str, Path]:
    """Write Phase0 metadata, table, and figures; numerical work stays separate."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "phase0_summary.json"
    table_path = output_dir / "phase0_delta_table.csv"
    branch_path = output_dir / "phase0_fixed_point_branch.png"
    scales_path = output_dir / "phase0_theory_scales.png"

    summary = {
        "script_version": SCRIPT_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "calculation_type": "Gaussian-map Phase0 theory prediction; not a simulation measurement",
        "inputs": {
            "B": result.task.B,
            "R": result.task.R,
            "sigma_J": result.task.sigma_J,
            "sigma_phi": result.task.sigma_phi,
            "phi_bar": result.task.phi_bar,
            "a": result.task.lattice_spacing,
            "branch": result.task.branch,
            "delta_list": result.task.delta_list,
        },
        **asdict(result.spinodal),
        "definitions": {
            "delta": "positive distance from the selected spinodal on its metastable side",
            "Delta_from_delta": "Delta = Delta_spinodal + branch_sign * delta",
            "Gamma0_theory": "-ln(Lambda_star)",
            "tau0_theory": "1/Gamma0_theory",
            "kappa_R_theory": "a^2 (R+1)(2R+1)/12",
            "xi_theory": "sqrt(kappa_R_theory/Gamma0_theory)",
        },
        "reused_model_functions": [
            "Config",
            "sigma_eff",
            "mu_from_B",
            "mean_field_map",
            "lambda_star",
        ],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    result.delta_table.to_csv(table_path, index=False)
    make_fixed_point_branch_figure(result, branch_path)
    make_theory_scales_figure(result, scales_path)
    return {
        "summary": summary_path,
        "delta_table": table_path,
        "fixed_point_branch": branch_path,
        "theory_scales": scales_path,
    }


def parse_delta_list(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(token.strip()) for token in text.split(",") if token.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("delta-list must be comma-separated numbers") from exc
    if not values:
        raise argparse.ArgumentTypeError("delta-list must not be empty")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise argparse.ArgumentTypeError("every delta must be positive and finite")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute analytic Gaussian-map spinodals, metastable fixed points, "
            "and Phase1+ theory scale predictions. Phase0 performs no simulation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    parser.add_argument("--B", type=float, default=2.0, help="mean-field slope parameter; must exceed 1")
    parser.add_argument("--R", type=int, default=12, help="interaction range")
    parser.add_argument("--sigma-J", type=float, default=1.0, dest="sigma_J")
    parser.add_argument("--sigma-phi", type=float, default=0.06, dest="sigma_phi")
    parser.add_argument("--phi-bar", type=float, default=0.0, dest="phi_bar")
    parser.add_argument("--a", type=float, default=1.0, dest="lattice_spacing")
    parser.add_argument(
        "--branch",
        choices=sorted(BRANCH_SIGNS),
        default="stay_to_evacuate",
        help="metastable transition direction",
    )
    parser.add_argument(
        "--delta-list",
        type=parse_delta_list,
        default=DEFAULT_DELTAS,
        help="comma-separated positive distances from the selected spinodal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "runs" / "phase0_B2_R12",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    task = Phase0Task(
        B=args.B,
        R=args.R,
        sigma_J=args.sigma_J,
        sigma_phi=args.sigma_phi,
        phi_bar=args.phi_bar,
        lattice_spacing=args.lattice_spacing,
        branch=args.branch,
        delta_list=tuple(args.delta_list),
    )
    try:
        result = run_phase0_case(task)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    paths = write_phase0_outputs(result, args.output_dir)
    spinodal = result.spinodal
    print("=== Spinodal Phase0 (Gaussian-map theory) ===")
    print(f"branch={spinodal.branch}")
    print(f"sigma_eff={spinodal.sigma_eff:.16g}")
    print(f"mu={spinodal.mu:.16g}")
    print(f"z_spinodal={spinodal.z_spinodal:.16g}")
    print(f"m_spinodal={spinodal.m_spinodal:.16g}")
    print(f"Delta_spinodal={spinodal.Delta_spinodal:.16g}")
    print(f"h_spinodal={spinodal.h_spinodal:.16g}")
    print(f"fixed_point_residual={spinodal.fixed_point_residual:.3e}")
    print(f"tangency_residual={spinodal.tangency_residual:.3e}")
    print("These scales are theory predictions, not simulation measurements.")
    print("=== Outputs ===")
    for path in paths.values():
        print(path)


if __name__ == "__main__":
    main()
