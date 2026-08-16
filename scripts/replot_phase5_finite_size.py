#!/usr/bin/env python3
"""Replot the V1 finite-size figures from finalized CSV files."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("results/runs/phase5_final_validation"))
    args = parser.parse_args()
    response = pd.read_csv(args.input_dir / "finite_size_response.csv")
    pseudospinodal = pd.read_csv(args.input_dir / "finite_size_pseudospinodal.csv")

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for R, group in response.groupby("R"):
        axis.errorbar(group["N"], group["Gamma_survive_over_closure"], yerr=group["Gamma_ratio_se"], fmt="o-", capsize=3, label=f"R={int(R)}")
    axis.axhline(1.0, color="black", linestyle="--")
    axis.set(xlabel="N", ylabel=r"$\Gamma_{survive}/\Gamma_{closure}$", title="Finite-size matched response")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.input_dir / "final_finite_size_gamma.png", dpi=220)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.4))
    for R, group in pseudospinodal.groupby("R"):
        axis.errorbar(group["N"], group["delta_ps_T50"], yerr=group["delta_ps_se"], fmt="o-", capsize=3, label=f"R={int(R)}")
    axis.set(xlabel="N", ylabel=r"$\delta_{ps}(T=50)$", title="Finite-size pseudospinodal control")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(args.input_dir / "final_finite_size_ps.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
