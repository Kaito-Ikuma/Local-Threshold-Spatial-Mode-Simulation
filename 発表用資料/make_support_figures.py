"""Regenerate the finite-R lattice-length comparison used in the support PDF."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT = ROOT / "results/runs/poster_ABCD/fully_numeric_dynamic_z_combined_points.csv"
OUTPUT_DATA = HERE / "data/xi_lat_vs_xi_bnd.csv"
OUTPUT_FIGURE = HERE / "figures/xi_lat_vs_xi_bnd.png"


def lattice_residual(xi_lat: float, *, R: int, gamma0: float, a: float = 1.0) -> float:
    r = np.arange(1, R + 1, dtype=float)
    return np.exp(-gamma0) * np.mean(np.cosh(r * a / xi_lat)) - 1.0


def solve_xi_lat(*, R: int, gamma0: float, a: float = 1.0) -> float:
    lower = max(R * a / 700.0, np.finfo(float).eps)
    upper = max(R * a, 1.0)
    while lattice_residual(upper, R=R, gamma0=gamma0, a=a) > 0.0:
        upper *= 2.0
    return float(
        brentq(
            lambda xi: lattice_residual(xi, R=R, gamma0=gamma0, a=a),
            lower,
            upper,
            xtol=1.0e-13,
            rtol=1.0e-13,
        )
    )


def main() -> None:
    points = pd.read_csv(INPUT)
    selected = points.loc[
        points["R"].isin([6, 12, 24, 48, 96])
        & points["delta"].isin([1.0e-5, 3.0e-5, 1.0e-4, 3.0e-4])
    ].copy()
    if len(selected) != 20:
        raise RuntimeError(f"expected 20 Result-2 rows, found {len(selected)}")

    selected["xi_lat"] = [
        solve_xi_lat(R=int(row.R), gamma0=float(row.Gamma0_num))
        for row in selected.itertuples()
    ]
    selected["xi_Gamma"] = np.sqrt(selected["kappa_R"] / selected["Gamma0_num"])
    selected["relative_difference_bnd_vs_lat"] = (
        selected["xi_bnd"] / selected["xi_lat"] - 1.0
    )
    selected["relative_difference_Gamma_vs_lat"] = (
        selected["xi_Gamma"] / selected["xi_lat"] - 1.0
    )
    selected = selected.sort_values(["R", "delta"]).reset_index(drop=True)

    OUTPUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURE.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT_DATA, index=False)

    colors = {6: "#1f77b4", 12: "#ff7f0e", 24: "#2ca02c", 48: "#d62728", 96: "#9467bd"}
    markers = {6: "o", 12: "s", 24: "^", 48: "D", 96: "P"}
    fig, ax = plt.subplots(figsize=(6.8, 5.3), constrained_layout=True)
    for R, group in selected.groupby("R", sort=True):
        ax.scatter(
            group["xi_lat"],
            group["xi_bnd"],
            s=55,
            marker=markers[int(R)],
            color=colors[int(R)],
            edgecolor="white",
            linewidth=0.6,
            label=f"R={int(R)}",
            zorder=3,
        )
    lower = 0.92 * min(selected["xi_lat"].min(), selected["xi_bnd"].min())
    upper = 1.08 * max(selected["xi_lat"].max(), selected["xi_bnd"].max())
    ax.plot([lower, upper], [lower, upper], "k--", lw=1.5, label=r"$y=x$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_xlabel(r"finite-$R$ characteristic length $\xi_{\rm lat}$")
    ax.set_ylabel(r"boundary-fit length $\xi_{\rm bnd}$")
    ax.set_title("Independent boundary fit vs finite-R lattice prediction")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    mean_abs = selected["relative_difference_bnd_vs_lat"].abs().mean() * 100.0
    max_abs = selected["relative_difference_bnd_vs_lat"].abs().max() * 100.0
    ax.text(
        0.04,
        0.96,
        f"20 conditions\nmean |rel. diff.| = {mean_abs:.3f}%\nmax |rel. diff.| = {max_abs:.3f}%",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.65", "alpha": 0.92},
    )
    fig.savefig(OUTPUT_FIGURE, dpi=240)
    plt.close(fig)

    print(f"wrote {OUTPUT_DATA}")
    print(f"wrote {OUTPUT_FIGURE}")
    print(f"mean_abs_percent={mean_abs:.9f}")
    print(f"max_abs_percent={max_abs:.9f}")


if __name__ == "__main__":
    main()
