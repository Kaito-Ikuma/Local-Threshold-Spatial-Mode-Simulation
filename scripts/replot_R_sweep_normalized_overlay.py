#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PRESENTATION_RESULTS = PROJECT_ROOT / "results" / "presentation_materials"


def load_and_prepare(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required = {"sweep_value", "dynamics", "mode_index", "q_over_pi", "kernel_hat", "lambda_fit_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # R sweep のみ使用
    if "sweep_parameter" in df.columns:
        df = df[df["sweep_parameter"] == "R"].copy()
    if df.empty:
        raise ValueError("No R-sweep rows found in input CSV.")

    # R 列を標準化
    if "R" not in df.columns:
        df["R"] = df["sweep_value"].astype(int)
    else:
        df["R"] = df["R"].astype(int)

    # 各 (R, dynamics) ごとに lambda(q)/lambda(0) を再構成
    out_frames = []
    for (R, dynamics), sub in df.groupby(["R", "dynamics"], sort=True):
        sub = sub.sort_values("mode_index").copy()
        if sub.empty:
            continue

        # q=0 を規格化に使う
        q0_rows = sub[sub["mode_index"] == 0]
        if q0_rows.empty:
            q0_rows = sub.sort_values("q_over_pi").head(1)

        lambda0 = float(q0_rows["lambda_fit_mean"].iloc[0])
        if abs(lambda0) < 1e-14:
            raise ValueError(f"lambda(q=0) is too small for R={R}, dynamics={dynamics}")

        sub["lambda_ratio_plot"] = sub["lambda_fit_mean"] / lambda0
        out_frames.append(sub)

    if not out_frames:
        raise ValueError("No usable grouped data found.")

    return pd.concat(out_frames, ignore_index=True)


def plot_overlay(df: pd.DataFrame, output_path: Path, shift: float = 0.0015) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 7.6), constrained_layout=True)

    R_values = sorted(df["R"].unique())

    for R in R_values:
        sub_R = df[df["R"] == R].copy()

        # theory は dynamics に依らず同じなので mode ごとに 1 本だけ取る
        theory = sub_R.sort_values(["mode_index", "dynamics"]).drop_duplicates(subset=["mode_index"])

        line, = ax.plot(
            theory["q_over_pi"],
            theory["kernel_hat"],
            "-",
            linewidth=2.6,
            label=rf"$R={int(R)}$",
            zorder=1,
        )
        color = line.get_color()

        # microscopic: 白抜き四角
        micro = sub_R[sub_R["dynamics"] == "microscopic"].sort_values("mode_index")
        if not micro.empty:
            ax.scatter(
                micro["q_over_pi"],
                micro["lambda_ratio_plot"],
                marker="s",
                s=130,
                facecolors="white",
                edgecolors=color,
                linewidths=2.4,
                zorder=4,
            )

        # gaussian_map: ×
        # 理論線と重なって見えなくならないように少しだけ右へずらす
        gauss = sub_R[sub_R["dynamics"] == "gaussian_map"].sort_values("mode_index")
        if not gauss.empty:
            ax.scatter(
                gauss["q_over_pi"] + shift,
                gauss["lambda_ratio_plot"],
                marker="x",
                s=140,
                c=color,
                linewidths=2.6,
                zorder=6,
            )

    ax.axhline(0.0, color="black", linewidth=1.0, zorder=0)
    ax.set_xlim(-0.008, 0.175 + shift + 0.003)
    ax.set_xlabel(r"$q/\pi$", fontsize=20)
    ax.set_ylabel(r"$\lambda(q)/\lambda(0)$", fontsize=20)
    ax.set_title(r"Effect of interaction range $R$ on spatial-mode relaxation", fontsize=28)
    ax.tick_params(axis="both", labelsize=17, width=1.0, length=6)

    # 凡例1: 色 = R
    legend1 = ax.legend(
        title="interaction range",
        fontsize=18,
        title_fontsize=19,
        loc="upper right",
        ncol=2,
        frameon=True,
    )
    ax.add_artist(legend1)

    # 凡例2: 記号の意味
    style_handles = [
        Line2D([0], [0], color="black", lw=2.6, label="theory"),
        Line2D(
            [0], [0],
            marker="s", linestyle="None",
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=2.4,
            markersize=11,
            label="microscopic",
        ),
        Line2D(
            [0], [0],
            marker="x", linestyle="None",
            color="black",
            markeredgewidth=2.6,
            markersize=12,
            label="gaussian_map",
        ),
    ]
    ax.legend(handles=style_handles, fontsize=16, loc="lower left", frameon=False)

    ax.text(
        0.02,
        0.02,
        "gaussian_map markers are shifted slightly in q for visibility",
        transform=ax.transAxes,
        fontsize=12,
        va="bottom",
        ha="left",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Replot R-sweep normalized overlay with visible gaussian_map markers."
    )
    p.add_argument(
        "--input",
        type=Path,
        default=PRESENTATION_RESULTS / "R_sweep_lambda_aggregate.csv",
        help="Input CSV from the R sweep",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=PRESENTATION_RESULTS / "R_sweep_normalized_overlay.png",
        help="Output figure path",
    )
    p.add_argument(
        "--shift",
        type=float,
        default=0.0015,
        help="Horizontal shift applied only to gaussian_map markers",
    )
    args = p.parse_args()

    df = load_and_prepare(args.input)
    plot_overlay(df, args.output, shift=args.shift)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
