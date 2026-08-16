#!/usr/bin/env python3
"""Create local-only figures for Phase5 V1--V5 final validation."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from spinodal_phase5_analysis import kernel_hat


def read_optional(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_or_empty(fig: plt.Figure, axes, path: Path, has_data: bool, title: str) -> None:
    if not has_data:
        axis = np.asarray(axes).flat[0]
        axis.text(0.5, 0.5, "required data not yet complete", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/runs/phase5_final_validation"))
    parser.add_argument("--r-sweep-dir", type=Path, default=Path("results/runs/phase5_R_sweep"))
    parser.add_argument("--combined-summary", type=Path, default=Path("results/runs/R_sweep_combined/R_sweep_combined_summary.csv"))
    args = parser.parse_args()
    args.input_dir.mkdir(parents=True, exist_ok=True)

    ps = read_optional(args.input_dir / "finite_size_pseudospinodal.csv")
    response = read_optional(args.input_dir / "finite_size_response.csv")
    D = read_optional(args.input_dir / "high_precision_D_over_kappa.csv")
    if not D.empty and "production_M_sufficient" in D:
        D = D[D["production_M_sufficient"]]
    time = read_optional(args.input_dir / "completed_pseudospinodal_time_dependence.csv")
    r96 = read_optional(args.input_dir / "R96_validation.csv")
    seeds = read_optional(args.input_dir / "seed_reproducibility.csv")
    combined = read_optional(args.combined_summary)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for R, group in response.groupby("R") if not response.empty else []:
        ax.errorbar(group["N"], group["Gamma_survive_over_closure"], yerr=group["Gamma_ratio_se"], fmt="o-", capsize=3, label=f"R={int(R)}")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set(xlabel="N", ylabel=r"$\Gamma_{survive}/\Gamma_{closure}$", title="Finite-size matched response")
    ax.grid(True, alpha=0.25)
    if not response.empty: ax.legend()
    save_or_empty(fig, ax, args.input_dir / "final_finite_size_gamma.png", not response.empty, "Finite-size matched response")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for R, group in ps.groupby("R") if not ps.empty else []:
        ax.errorbar(group["N"], group["delta_ps_T50"], yerr=group["delta_ps_se"], fmt="o-", capsize=3, label=f"R={int(R)}")
    ax.set(xlabel="N", ylabel=r"$\delta_{ps}(T=50)$", title="Finite-size pseudospinodal control")
    ax.grid(True, alpha=0.25)
    if not ps.empty: ax.legend()
    save_or_empty(fig, ax, args.input_dir / "final_finite_size_ps.png", not ps.empty, "Finite-size pseudospinodal control")

    for filename in ("final_D_over_kappa_high_precision.png", "high_precision_D_over_kappa.png"):
        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        if not D.empty:
            lower = D["D_over_kappa"] - D["D_over_kappa_CI_low"]
            upper = D["D_over_kappa_CI_high"] - D["D_over_kappa"]
            ax.errorbar(D["R"], D["D_over_kappa"], yerr=np.vstack((lower, upper)), fmt="o-", capsize=3)
        ax.axhline(1.0, color="black", linestyle="--", label="closure")
        ax.set(xlabel="R", ylabel=r"$D_{micro}/\kappa_R$", title="High-precision microscopic dispersion")
        ax.grid(True, alpha=0.25)
        ax.legend()
        save_or_empty(fig, ax, args.input_dir / filename, not D.empty, "High-precision microscopic dispersion")

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), sharey=False)
    has_panels = False
    for axis, R in zip(axes, (12, 24, 48)):
        mode = read_optional(args.r_sweep_dir / f"R{R:03d}/dispersion/phase5_mode_results.csv")
        if mode.empty:
            axis.set_title(f"R={R}: missing")
            continue
        has_panels = True
        axis.errorbar(mode["q"] ** 2, mode["Gamma_micro"], yerr=mode["Gamma_micro_se"], fmt="o", capsize=2)
        axis.set(title=f"R={R}", xlabel=r"$q^2$", ylabel=r"$\Gamma(q)$")
        axis.grid(True, alpha=0.25)
    fig.suptitle("High-precision dispersion panels")
    save_or_empty(fig, axes, args.input_dir / "high_precision_dispersion_panels.png", has_panels, "High-precision dispersion panels")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    has_kernel = False
    for R in (12, 24, 48):
        mode = read_optional(args.r_sweep_dir / f"R{R:03d}/dispersion/phase5_mode_results.csv")
        if mode.empty: continue
        mode = mode.sort_values("mode_index")
        q0 = mode[mode["mode_index"] == 0]
        if len(q0) != 1: continue
        has_kernel = True
        N = int(mode["N"].iloc[0])
        x = [-math.log(abs(kernel_hat(int(value), N, R))) for value in mode["mode_index"]]
        y = mode["Gamma_micro"] - float(q0["Gamma_micro"].iloc[0])
        ax.plot(x, y, "o-", label=f"R={R}")
    ax.plot([0, 0.02], [0, 0.02], "k--", label="unit slope")
    ax.set(xlabel=r"$-\ln \hat K_R(q)$", ylabel=r"$\Gamma(q)-\Gamma(0)$", title="Secondary exact-kernel diagnostic")
    ax.grid(True, alpha=0.25)
    if has_kernel: ax.legend()
    save_or_empty(fig, ax, args.input_dir / "kernel_relation_high_precision.png", has_kernel, "Secondary exact-kernel diagnostic")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for R, group in time.groupby("R") if not time.empty else []:
        ax.errorbar(group["T_obs"], group["delta_ps"], yerr=group["SE"], fmt="o-", capsize=2, label=f"R={int(R)}")
    ax.set(xlabel=r"$T_{obs}$", ylabel=r"$\delta_{ps}$", title="Observation-time dependence")
    ax.grid(True, alpha=0.25)
    if not time.empty: ax.legend()
    save_or_empty(fig, ax, args.input_dir / "final_observation_time_all_R.png", not time.empty, "Observation-time dependence")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    primary_time = time.drop_duplicates("R") if not time.empty else time
    if not primary_time.empty: ax.plot(primary_time["R"], primary_time["relative_time_rounding"], "o-")
    ax.set(xlabel="R", ylabel=r"$[\delta_{ps}(50)-\delta_{ps}(30)]/\delta_{ps}(50)$", title="Relative observation-time rounding")
    ax.grid(True, alpha=0.25)
    save_or_empty(fig, ax, args.input_dir / "final_relative_rounding_vs_R.png", not primary_time.empty, "Relative observation-time rounding")

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0))
    has_r96 = not r96.empty and not combined.empty
    if has_r96:
        base = combined[["R", "micro_delta_ps_T50", "micro_survival_Gamma_ratio", "micro_Gamma_eff_ratio_t3_t0"]].copy()
        extra = pd.DataFrame({"R": r96["R"], "micro_delta_ps_T50": r96["delta_ps"], "micro_survival_Gamma_ratio": r96["Gamma_ratio"], "micro_Gamma_eff_ratio_t3_t0": r96["Gamma_eff_ratio"]})
        values = pd.concat((base, extra), ignore_index=True).sort_values("R")
        for axis, column, label in zip(axes, ("micro_survival_Gamma_ratio", "micro_Gamma_eff_ratio_t3_t0", "micro_delta_ps_T50"), ("Gamma ratio", "Gamma_eff ratio", "delta_ps")):
            axis.plot(values["R"], values[column], "o-")
            axis.set(xlabel="R", ylabel=label)
            axis.grid(True, alpha=0.25)
    fig.suptitle("R=96 large-range confirmation")
    save_or_empty(fig, axes, args.input_dir / "final_R96_extension.png", has_r96, "R=96 large-range confirmation")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for R, group in seeds.groupby("R") if not seeds.empty else []:
        ax.errorbar(group["seed"].astype(str), group["Gamma_ratio"], yerr=group["Gamma_ratio_se"], fmt="o", capsize=3, label=f"R={int(R)}")
    ax.axhline(1.0, color="black", linestyle="--")
    ax.set(xlabel="base seed", ylabel=r"$\Gamma_{survive}/\Gamma_{closure}$", title="Independent-seed reproducibility")
    ax.grid(True, alpha=0.25)
    if not seeds.empty: ax.legend()
    save_or_empty(fig, ax, args.input_dir / "final_seed_reproducibility.png", not seeds.empty, "Independent-seed reproducibility")


if __name__ == "__main__":
    main()
