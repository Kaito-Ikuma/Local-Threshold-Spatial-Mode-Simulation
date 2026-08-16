#!/usr/bin/env python3
"""Create local-only Phase5 follow-up figures from CSV outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _load(directory: Path, name: str) -> pd.DataFrame | None:
    path = directory / name
    return pd.read_csv(path) if path.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=Path("results/runs/phase5_B2_R12_followup")
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = args.output_dir or args.input_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    gamma = _load(args.input_dir, "phase5_gamma_eff.csv")
    windows = _load(args.input_dir, "phase5_fit_window_extended.csv")
    preparation = _load(args.input_dir, "phase5_preparation_scan.csv")
    survival = _load(args.input_dir, "phase5_survival_conditioned.csv")
    fine = _load(args.input_dir, "phase5_pseudospinodal_fine_scan.csv")
    time_dep = _load(args.input_dir, "phase5_pseudospinodal_time_dependence.csv")

    if gamma is not None:
        selected = gamma[
            np.isclose(gamma["epsilon_fraction"], 0.05)
            & gamma["mode_index"].isin([0, 1, 4])
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for axis, mode in zip(axes, [0, 1, 4]):
            for delta, group in selected[selected["mode_index"] == mode].groupby("delta"):
                first = float(group["A_q"].iloc[0])
                axis.plot(group["t"], group["A_q"] / first, "o-", ms=3, label=f"δ={delta:g}")
            axis.set(title=f"mode {mode}", xlabel="t", ylabel="A(t)/A(0)")
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.savefig(output / "phase5_Aq_time_diagnostics.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
        q0 = selected[selected["mode_index"] == 0]
        for delta, group in q0.groupby("delta"):
            reliable = group["snr_reliable"].astype(bool)
            ax.plot(group["t"], group["Gamma_eff"], "o-", ms=3, label=f"δ={delta:g}")
            ax.fill_between(group["t"], group["Gamma_eff_ci_low"], group["Gamma_eff_ci_high"], alpha=0.15)
            ax.plot(group.loc[~reliable, "t"], group.loc[~reliable, "Gamma_eff"], "x", color="gray")
        ax.set(xlabel="t", ylabel="Gamma_eff(t)", title="Phase5 effective relaxation rate (mode 0)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output / "phase5_gamma_eff_time.png", dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True, sharey=True)
        for axis, mode in zip(axes, [0, 1, 4]):
            for delta, group in selected[selected["mode_index"] == mode].groupby("delta"):
                axis.plot(group["t"], group["Gamma_eff"], "o-", ms=3, label=f"δ={delta:g}")
            axis.set(title=f"mode {mode}", xlabel="t")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Gamma_eff")
        axes[0].legend(fontsize=7)
        fig.savefig(output / "phase5_gamma_eff_mode_panel.png", dpi=220)
        plt.close(fig)

    if windows is not None:
        primary = windows[windows["primary_window"].astype(bool)]
        fig, ax = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
        x = np.arange(len(primary))
        ax.plot(x, primary["Gamma_B"], "o", label="Method B")
        ax.plot(x, primary["Gamma_C"], "s", label="Method C")
        plateau = windows[windows["plateau_found"].astype(bool)]
        if not plateau.empty:
            key_to_x = {task: index for index, task in enumerate(primary["task_id"])}
            ax.plot([key_to_x[task] for task in plateau["task_id"] if task in key_to_x], [row.plateau_mean_gamma for row in plateau.itertuples() if row.task_id in key_to_x], "^", label="plateau")
        ax.set(xlabel="task index", ylabel="Gamma", title="Method B/C and plateau diagnostics")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output / "phase5_gamma_method_comparison.png", dpi=220)
        plt.close(fig)

    if preparation is not None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        for (delta, mode), group in preparation.groupby(["delta", "mode_index"]):
            label = f"δ={delta:g},m={mode}"
            axes[0].plot(group["burn_steps_per_stage"], group["Gamma_B"], "o-", label=label)
            axes[1].plot(group["burn_steps_per_stage"], group["max_baseline_drift"], "o-", label=label)
        axes[0].set(xlabel="burn steps/stage", ylabel="Gamma_B")
        axes[1].set(xlabel="burn steps/stage", ylabel="max baseline drift")
        for axis in axes:
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=6)
        fig.savefig(output / "phase5_preparation_dependence.png", dpi=220)
        plt.close(fig)

    if survival is not None:
        representative = survival[
            (survival["mode_index"] == 0)
            & np.isclose(survival["epsilon_fraction"], 0.05)
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        for axis, (delta, group) in zip(axes, representative.groupby("delta")):
            axis.plot(group["t"], group["A_unconditional"], label="unconditional")
            axis.plot(group["t"], group["A_surviving_current"], label="surviving-current")
            axis.plot(group["t"], group["A_survive_to_T"], label="survive-to-T")
            axis.set(title=f"δ={delta:g}", xlabel="t", ylabel="A_q")
            axis.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.savefig(output / "phase5_survival_conditioned_Aq.png", dpi=220)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
        for delta, group in representative.groupby("delta"):
            ax.plot(group["t"], 1.0 - group["escape_fraction_cumulative"], label=f"δ={delta:g}")
        ax.set(xlabel="t", ylabel="P_surv(t)", title="First-passage survival probability")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.savefig(output / "phase5_survival_probability.png", dpi=220)
        plt.close(fig)

    if fine is not None:
        fig, ax = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
        yerr = np.vstack((fine["escape_fraction_cumulative_T"] - fine["escape_cumulative_ci_low"], fine["escape_cumulative_ci_high"] - fine["escape_fraction_cumulative_T"]))
        ax.errorbar(fine["delta"], fine["escape_fraction_cumulative_T"], yerr=yerr, fmt="o-", capsize=3)
        ax.axhline(float(fine["criterion_probability"].iloc[0]), color="k", linestyle="--")
        ax.set(xlabel="Gaussian-centered delta", ylabel="P_esc cumulative", title="Operational 10%-escape crossover")
        ax.grid(alpha=0.25)
        fig.savefig(output / "phase5_pseudospinodal_escape_curve.png", dpi=220)
        plt.close(fig)

    if time_dep is not None:
        fig, ax = plt.subplots(figsize=(7.2, 5), constrained_layout=True)
        valid = time_dep[time_dep["delta_ps_estimate"].notna()]
        if not valid.empty:
            yerr = np.vstack((valid["delta_ps_estimate"] - valid["delta_ps_ci_low"], valid["delta_ps_ci_high"] - valid["delta_ps_estimate"]))
            ax.errorbar(valid["T_obs"], valid["delta_ps_estimate"], yerr=yerr, fmt="o-", capsize=3)
        ax.set(xlabel="T_obs", ylabel="delta_ps^(10%,T_obs)", title="Observation-time dependence")
        ax.grid(alpha=0.25)
        fig.savefig(output / "phase5_pseudospinodal_time_dependence.png", dpi=220)
        plt.close(fig)

    print(output)


if __name__ == "__main__":
    main()
