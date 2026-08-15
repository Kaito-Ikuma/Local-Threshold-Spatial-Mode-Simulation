#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Replot R-sweep selected-mode decay on a single axis.

Input
-----
results/presentation_materials/R_sweep_selected_mode_decay.csv

Expected columns
----------------
sweep_parameter, sweep_value, dynamics, t,
theory_norm_mean, amp_norm_mean, amp_norm_se

Output style
------------
color          : interaction range R
solid line     : theory
open square    : microscopic
x marker       : gaussian_map

Example
-------
python3 scripts/replot_R_sweep_selected_mode_decay_overlay.py \
  --input results/presentation_materials/R_sweep_selected_mode_decay.csv \
  --output results/presentation_materials/R_sweep_selected_mode_decay_overlay.png \
  --selected-mode 2 \
  --tmax 4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESENTATION_RESULTS = PROJECT_ROOT / "results" / "presentation_materials"


def load_decay_data(csv_path: Path) -> pd.DataFrame:
    """Load and validate the R-sweep selected-mode decay CSV."""
    df = pd.read_csv(csv_path)

    required = {
        "sweep_value",
        "dynamics",
        "t",
        "theory_norm_mean",
        "amp_norm_mean",
        "amp_norm_se",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Input CSV does not have the required columns: "
            f"{sorted(missing)}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Keep only the R sweep if a sweep-parameter column is present.
    if "sweep_parameter" in df.columns:
        df = df[df["sweep_parameter"] == "R"].copy()

    if df.empty:
        raise ValueError("No R-sweep rows were found in the input CSV.")

    # Standardize the interaction-range column.
    if "R" not in df.columns:
        df["R"] = df["sweep_value"].round().astype(int)
    else:
        df["R"] = df["R"].round().astype(int)

    # Normalize textual values in case of accidental whitespace.
    df["dynamics"] = df["dynamics"].astype(str).str.strip()

    expected_dynamics = {"gaussian_map", "microscopic"}
    found = set(df["dynamics"].unique())
    missing_dynamics = expected_dynamics - found
    if missing_dynamics:
        raise ValueError(
            f"Missing dynamics rows: {sorted(missing_dynamics)}. "
            f"Found: {sorted(found)}"
        )

    return df.sort_values(["R", "dynamics", "t"]).reset_index(drop=True)


def plot_R_sweep_selected_mode_decay_overlay(
    df: pd.DataFrame,
    output_path: Path,
    selected_mode: int = 2,
    tmax: float | None = 4.0,
    marker_shift: float = 0.045,
) -> None:
    """
    Plot all R values on one axis.

    The numerical markers are shifted only in the horizontal display position:
      microscopic : t - marker_shift
      gaussian_map: t + marker_shift
    The underlying data values are not changed.
    """
    fig, ax = plt.subplots(figsize=(10.0, 6.2), constrained_layout=True)

    R_values = sorted(df["R"].unique())

    for R in R_values:
        sub_R = df[df["R"] == R]

        gauss = (
            sub_R[sub_R["dynamics"] == "gaussian_map"]
            .sort_values("t")
            .copy()
        )
        micro = (
            sub_R[sub_R["dynamics"] == "microscopic"]
            .sort_values("t")
            .copy()
        )

        if gauss.empty or micro.empty:
            raise ValueError(
                f"R={R}: gaussian_map or microscopic rows are missing."
            )

        # Theory is common to both dynamics. Use the gaussian_map rows once.
        theory_line, = ax.plot(
            gauss["t"],
            gauss["theory_norm_mean"],
            "-",
            linewidth=2.2,
            label=rf"$R={int(R)}$",
            zorder=2,
        )
        color = theory_line.get_color()

        # Microscopic: open squares, shifted slightly to the left.
        ax.errorbar(
            micro["t"] - marker_shift,
            micro["amp_norm_mean"],
            yerr=micro["amp_norm_se"],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            alpha=0.95,
            zorder=4,
        )
        ax.scatter(
            micro["t"] - marker_shift,
            micro["amp_norm_mean"],
            marker="s",
            s=72,
            facecolors="white",
            edgecolors=color,
            linewidths=1.7,
            zorder=5,
        )

        # Gaussian map: x markers, shifted slightly to the right.
        ax.errorbar(
            gauss["t"] + marker_shift,
            gauss["amp_norm_mean"],
            yerr=gauss["amp_norm_se"],
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            alpha=0.95,
            zorder=6,
        )
        ax.scatter(
            gauss["t"] + marker_shift,
            gauss["amp_norm_mean"],
            marker="x",
            s=82,
            c=color,
            linewidths=2.0,
            zorder=7,
        )

    ax.axhline(0.0, color="black", linewidth=0.9, zorder=0)

    if tmax is not None:
        ax.set_xlim(-0.15, float(tmax) + 0.15)

    ax.set_xlabel("time $t$", fontsize=16)
    ax.set_ylabel(r"$A_q(t)/A_q(0)$", fontsize=16)
    ax.set_title(
        rf"Effect of interaction range $R$ on selected-mode decay ($n={selected_mode}$)",
        fontsize=20,
    )
    ax.tick_params(labelsize=13)

    # Legend 1: colors identify R.
    legend_R = ax.legend(
        title="interaction range",
        fontsize=12,
        title_fontsize=13,
        ncol=2,
        loc="upper right",
        frameon=True,
    )
    ax.add_artist(legend_R)

    # Legend 2: line/marker style identifies theory and dynamics.
    style_handles = [
        Line2D([0], [0], color="black", lw=2.2, label="theory"),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=1.7,
            markersize=7,
            label="microscopic",
        ),
        Line2D(
            [0], [0],
            marker="x",
            linestyle="None",
            color="black",
            markeredgewidth=2.0,
            markersize=8,
            label="gaussian_map",
        ),
    ]
    ax.legend(
        handles=style_handles,
        fontsize=11,
        loc="lower left",
        frameon=False,
    )

    # ax.text(
    #     0.02,
    #     0.02,
    #     "markers are shifted slightly in time for visibility",
    #     transform=ax.transAxes,
    #     fontsize=9,
    #     va="bottom",
    #     ha="left",
    # )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay R-sweep selected-mode decay curves on one axis."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=PRESENTATION_RESULTS / "R_sweep_selected_mode_decay.csv",
        help="Input R_sweep_selected_mode_decay.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PRESENTATION_RESULTS / "R_sweep_selected_mode_decay_overlay.png",
        help="Output PNG path",
    )
    parser.add_argument(
        "--selected-mode",
        type=int,
        default=2,
        help="Selected mode index shown in the title",
    )
    parser.add_argument(
        "--tmax",
        type=float,
        default=4.0,
        help="Maximum displayed time. Use a negative value to disable clipping.",
    )
    parser.add_argument(
        "--marker-shift",
        type=float,
        default=0.045,
        help="Horizontal display shift for microscopic/gaussian markers",
    )
    args = parser.parse_args()

    df = load_decay_data(args.input)
    tmax = None if args.tmax < 0 else args.tmax

    plot_R_sweep_selected_mode_decay_overlay(
        df=df,
        output_path=args.output,
        selected_mode=args.selected_mode,
        tmax=tmax,
        marker_shift=args.marker_shift,
    )
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
