#!/usr/bin/env python3
"""
Replot sweep lambda-summary figure in a vertical 2x1 layout
from an existing *_lambda_aggregate.csv file.

Example
-------
python3 scripts/replot_lambda_summary_vertical.py \
  --input results/presentation_materials/delta_h_minus_phi_sweep_lambda_aggregate.csv \
  --output results/presentation_materials/delta_h_minus_phi_sweep_lambda_summary_vertical.png \
  --selected-mode 2
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

def infer_parameter(df: pd.DataFrame) -> str:
    if "sweep_parameter" in df.columns and df["sweep_parameter"].notna().any():
        return str(df["sweep_parameter"].dropna().iloc[0])
    raise ValueError("Could not infer sweep parameter. The CSV must contain 'sweep_parameter'.")


def xlabel_for_parameter(parameter: str) -> str:
    if parameter == "R":
        return "interaction range R"
    if parameter == "B":
        return "B"
    return r"$h-\bar\phi$"


def title_for_mode(mode: int, selected_mode: int) -> str:
    if mode == 0:
        return r"uniform mode $q=0$"
    return rf"selected mode $n={selected_mode}$"


def plot_R_sweep_normalized_overlay(
    df: pd.DataFrame,
    output_path: Path,
    show_gaussian: bool = True,
) -> None:
    """
    Plot normalized spectra for all R values on a single axis.

    theory      : line
    microscopic : square marker
    gaussian    : x marker
    """
    fig, ax = plt.subplots(
        figsize=(10.0, 6.2),
        constrained_layout=True,
    )

    R_values = sorted(df["sweep_value"].unique())

    for R in R_values:
        sub_R = df[df["sweep_value"] == R]

        print(f"--- R = {R} ---")
        print(sub_R["dynamics"].value_counts())

        gauss = (
            sub_R[sub_R["dynamics"] == "gaussian_map"]
            .sort_values("mode_index")
        )
        print("gaussian rows:")
        print(gauss[["mode_index", "q_over_pi", "lambda_ratio_mean"]])

        theory = (
    sub_R[sub_R["dynamics"] == "gaussian_map"]
    .sort_values("mode_index")
    .drop_duplicates("mode_index")
)

        # ---- theory ----
        line, = ax.plot(
            theory["q_over_pi"],
            theory["kernel_hat"],
            "-",
            linewidth=2.2,
            label=rf"$R={int(R)}$",
            zorder=2,
        )
        color = line.get_color()


        # ---- microscopic : square ----
        micro = (
            sub_R[sub_R["dynamics"] == "microscopic"]
            .sort_values("mode_index")
        )
        ax.errorbar(
            micro["q_over_pi"],
            micro["lambda_ratio_mean"],
            yerr=micro["lambda_ratio_se"],
            fmt="s",
            color=color,
            markerfacecolor="white",
            markeredgewidth=1.6,
            markersize=8,
            capsize=3,
            linestyle="none",
            zorder=4,
        )

        # ---- gaussian_map : x ----
        if show_gaussian:
            gauss = (
                sub_R[sub_R["dynamics"] == "gaussian_map"]
                .sort_values("mode_index")
            )

            # 必要ならごく小さく横にずらす
            delta_q = 0.0015

            # エラーバーだけ
            ax.errorbar(
                gauss["q_over_pi"] + delta_q,
                gauss["lambda_ratio_mean"],
                yerr=gauss["lambda_ratio_se"],
                fmt="none",
                ecolor=color,
                elinewidth=1.2,
                capsize=3,
                zorder=6,
            )

            # ×マーカー本体
            ax.scatter(
                gauss["q_over_pi"] + delta_q,
                gauss["lambda_ratio_mean"],
                marker="x",
                s=90,
                c=color,
                linewidths=2.2,
                zorder=7,
                label=None,
            )
            
    ax.axhline(0.0, color="black", linewidth=0.9)

    ax.set_xlabel(r"$q/\pi$", fontsize=16)
    ax.set_ylabel(r"$\lambda(q)/\lambda(0)$", fontsize=16)
    ax.set_title(
        r"Effect of interaction range $R$ on spatial-mode relaxation",
        fontsize=20,
    )
    ax.tick_params(labelsize=13)

    # 凡例1：色 = R
    legend1 = ax.legend(
        title="interaction range",
        fontsize=12,
        title_fontsize=13,
        ncol=2,
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(legend1)

    # 凡例2：記号の意味
    from matplotlib.lines import Line2D
    marker_handles = [
        Line2D([0], [0], color="black", lw=2, label="theory"),
        Line2D([0], [0], marker="s", color="black", linestyle="None",
               markerfacecolor="white", markeredgewidth=1.6, markersize=8,
               label="microscopic"),
        Line2D([0], [0], marker="x", color="black", linestyle="None",
               markeredgewidth=2.0, markersize=8,
               label="gaussian_map"),
    ]
    ax.legend(
        handles=marker_handles,
        fontsize=11,
        loc="lower left",
        frameon=False,
    )

    fig.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

def plot_lambda_summary_vertical(
    df: pd.DataFrame,
    parameter: str,
    selected_mode: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 11.0), constrained_layout=True)

    plot_specs = [
        (axes[0], 0, title_for_mode(0, selected_mode)),
        (axes[1], selected_mode, title_for_mode(selected_mode, selected_mode)),
    ]

    for ax, mode, title in plot_specs:
        theory = (
            df[(df["mode_index"] == mode) & (df["dynamics"] == "gaussian_map")]
            .sort_values("sweep_value")
        )
        if theory.empty:
            raise ValueError(
                f"No gaussian_map rows found for mode_index={mode}. "
                "The aggregate CSV does not have the expected structure."
            )

        ax.plot(
            theory["sweep_value"],
            theory["lambda_theory"],
            "k--",
            lw=2.0,
            label="theory",
        )

        for dynamics, marker in [("gaussian_map", "o"), ("microscopic", "s")]:
            sub = df[(df["mode_index"] == mode) & (df["dynamics"] == dynamics)].sort_values("sweep_value")
            if sub.empty:
                continue
            yerr = sub["lambda_fit_se"] if "lambda_fit_se" in sub.columns else None
            ax.errorbar(
                sub["sweep_value"],
                sub["lambda_fit_mean"],
                yerr=yerr,
                fmt=f"{marker}-",
                capsize=3,
                ms=6,
                lw=1.3,
                label=dynamics,
            )

        ax.axhline(0.0, color="k", lw=0.9)
        ax.set_xlabel(xlabel_for_parameter(parameter), fontsize=13)
        ax.set_ylabel(r"$\lambda(q)$", fontsize=13)
        ax.set_title(title, fontsize=18)
        ax.tick_params(labelsize=12)
        ax.legend(fontsize=11)

    fig.suptitle(f"{parameter}-sweep: relaxation-eigenvalue summary", fontsize=22)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Path to *_lambda_aggregate.csv")
    p.add_argument("--output", type=Path, default=None, help="Output PNG path")
    p.add_argument("--selected-mode", type=int, default=2, help="Selected nonzero mode index")
    p.add_argument(
        "--parameter",
        type=str,
        default=None,
        choices=["R", "B", "delta_h_minus_phi"],
        help="Override sweep parameter if needed",
    )
    args = p.parse_args()

    df = pd.read_csv(args.input)
    parameter = args.parameter or infer_parameter(df)

    if args.output is None:
        stem = args.input.stem
        if stem.endswith("_lambda_aggregate"):
            out_name = stem.replace("_lambda_aggregate", "_lambda_summary_vertical") + ".png"
        else:
            out_name = stem + "_vertical.png"
        output_path = args.input.with_name(out_name)
    else:
        output_path = args.output

    plot_lambda_summary_vertical(df=df, parameter=parameter, selected_mode=args.selected_mode, output_path=output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
