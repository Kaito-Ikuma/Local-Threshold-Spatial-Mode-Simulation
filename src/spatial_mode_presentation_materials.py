#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial-mode presentation materials builder
===========================================

目的
----
既存の `spatial_mode_ensemble_validation.py` を利用して、
発表（スライド・ポスター）向けの資料を自動生成する。

本スクリプトで作る主資料
------------------------
1. 理論・gaussian_map・microscopic を同一条件で比較した固有値スペクトル図
2. 規格化スペクトル lambda(q)/lambda(0) と Khat_R(q) の比較図
3. 選択モード A_q(t)/A_q(0) の減衰比較図（平均±標準誤差）
4. （任意）epsilon 依存性による線形領域確認図
5. （任意）R 依存性によるカーネル形状確認図
6. （任意）負固有値モードの符号反転減衰図

基本方針
--------
- gaussian_map と microscopic を**同一条件**で実行する。
- 複数 seed の独立反復を回し、誤差棒を付ける。
- 選択モードは A_q(t) だけでなく A_q(t)/A_q(0) も保存する。
- 低信号モードは "unreliable" フラグを付ける。

使い方の例
----------
python3 src/spatial_mode_presentation_materials.py \
  --base-script src/spatial_mode_ensemble_validation.py \
  --output-dir results/presentation_materials \
  --N 96 --R 12 --T 40 --ensemble 2000 \
  --B 0.72 --sigma-J 1.0 --sigma-phi 0.06 --phi-bar 0.0 --h 0.0 \
  --epsilon 0.05 --fit-steps 5 \
  --modes 0,1,2,4,6,8 --selected-mode 2 \
  --n-seeds 12 --seed0 20260801 \
  --epsilon-scan 0.02,0.05,0.08 \
  --R-scan 6,12,18 \
  --negative-mode 6
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# -----------------------------------------------------------------------------
# utility
# -----------------------------------------------------------------------------


def configure_font() -> None:
    candidates = [
        "Hiragino Sans",
        "Yu Gothic",
        "YuGothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAGothic",
        "TakaoGothic",
    ]
    installed = {f.name for f in mpl.font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            mpl.rcParams["font.family"] = name
            break
    mpl.rcParams["axes.unicode_minus"] = False


configure_font()


def parse_int_list(text: str) -> list[int]:
    vals = []
    for token in text.split(","):
        token = token.strip()
        if token:
            vals.append(int(token))
    return sorted(set(vals))


def parse_float_list(text: str) -> list[float]:
    vals = []
    for token in text.split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    return vals


def load_base_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("spatial_mode_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sem(x: np.ndarray, axis: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    if n <= 1:
        return np.zeros_like(np.mean(x, axis=axis))
    return np.std(x, axis=axis, ddof=1) / math.sqrt(n)


# -----------------------------------------------------------------------------
# simulation wrappers
# -----------------------------------------------------------------------------


def run_seed_suite(
    base: ModuleType,
    cfg: Any,
    modes: list[int],
    selected_mode: int,
    n_seeds: int,
    seed0: int,
    dynamics: str,
) -> dict[str, Any]:
    """Run the original simulator for one dynamics type over many seeds."""
    cfg_dyn = replace(cfg, dynamics=dynamics)
    base.validate_config(cfg_dyn, modes, selected_mode)

    m_star, roots = base.choose_fixed_point(cfg_dyn)
    Lambda = base.lambda_star(m_star, cfg_dyn)
    mu = base.mu_from_B(cfg_dyn)
    sig = base.sigma_eff(cfg_dyn)

    results_by_seed: list[dict[int, Any]] = []
    mode_rows: list[dict[str, Any]] = []
    selected_amp_runs = []
    selected_amp_norm_runs = []
    selected_theory_runs = []

    for s in range(n_seeds):
        cfg_seed = replace(cfg_dyn, rng_seed=seed0 + s)
        per_mode: dict[int, Any] = {}
        for mode in modes:
            result = base.simulate_mode(
                mode_index=mode,
                m_star=m_star,
                Lambda=Lambda,
                config=cfg_seed,
                seed_offset=100_000 * mode,
            )
            per_mode[mode] = result
            mode_rows.append(
                {
                    "dynamics": dynamics,
                    "seed_index": s,
                    "seed": seed0 + s,
                    "mode_index": mode,
                    "q": result.q,
                    "q_over_pi": result.q / math.pi,
                    "kernel_hat": result.khat,
                    "lambda_theory": result.lambda_theory,
                    "lambda_fit": result.lambda_fit,
                    "fit_r2": result.fit_r2,
                    "A0": result.cosine_amplitude[0],
                    "A1": result.cosine_amplitude[1] if len(result.cosine_amplitude) > 1 else np.nan,
                }
            )
            if mode == selected_mode:
                amp = np.asarray(result.cosine_amplitude, dtype=float)
                selected_amp_runs.append(amp)
                denom = amp[0] if abs(amp[0]) > 1e-14 else np.nan
                selected_amp_norm_runs.append(amp / denom)
                selected_theory_runs.append(np.asarray(result.theory_cosine_amplitude, dtype=float) / cfg_seed.epsilon)
        results_by_seed.append(per_mode)

    per_seed_df = pd.DataFrame(mode_rows)

    # normalized ratios computed seed-by-seed using the same seed's n=0 estimate
    ratio_rows = []
    for s in range(n_seeds):
        sub = per_seed_df[per_seed_df["seed_index"] == s].copy()
        row0 = sub[sub["mode_index"] == 0]
        if len(row0) == 0:
            continue
        lam0 = float(row0.iloc[0]["lambda_fit"])
        for _, row in sub.iterrows():
            ratio_rows.append(
                {
                    "dynamics": dynamics,
                    "seed_index": s,
                    "mode_index": int(row["mode_index"]),
                    "q": float(row["q"]),
                    "q_over_pi": float(row["q_over_pi"]),
                    "kernel_hat": float(row["kernel_hat"]),
                    "lambda_theory": float(row["lambda_theory"]),
                    "lambda_fit": float(row["lambda_fit"]),
                    "lambda_ratio": float(row["lambda_fit"] / lam0) if abs(lam0) > 1e-14 else np.nan,
                }
            )
    ratio_df = pd.DataFrame(ratio_rows)

    # aggregate over seeds
    agg = (
        per_seed_df.groupby(["dynamics", "mode_index", "q", "q_over_pi", "kernel_hat", "lambda_theory"], as_index=False)
        .agg(
            lambda_fit_mean=("lambda_fit", "mean"),
            lambda_fit_std=("lambda_fit", "std"),
            fit_r2_mean=("fit_r2", "mean"),
            fit_r2_std=("fit_r2", "std"),
            A0_mean=("A0", "mean"),
            A1_mean=("A1", "mean"),
        )
    )
    counts = per_seed_df.groupby(["dynamics", "mode_index"], as_index=False).size().rename(columns={"size": "n_seed"})
    agg = agg.merge(counts, on=["dynamics", "mode_index"], how="left")
    agg["lambda_fit_se"] = agg["lambda_fit_std"] / np.sqrt(agg["n_seed"].clip(lower=1))
    agg["relative_error"] = np.where(
        np.abs(agg["lambda_theory"]) > 1e-12,
        np.abs(agg["lambda_fit_mean"] - agg["lambda_theory"]) / np.abs(agg["lambda_theory"]),
        np.nan,
    )

    ratio_agg = (
        ratio_df.groupby(["dynamics", "mode_index", "q", "q_over_pi", "kernel_hat", "lambda_theory"], as_index=False)
        .agg(
            lambda_ratio_mean=("lambda_ratio", "mean"),
            lambda_ratio_std=("lambda_ratio", "std"),
        )
    )
    ratio_counts = ratio_df.groupby(["dynamics", "mode_index"], as_index=False).size().rename(columns={"size": "n_seed"})
    ratio_agg = ratio_agg.merge(ratio_counts, on=["dynamics", "mode_index"], how="left")
    ratio_agg["lambda_ratio_se"] = ratio_agg["lambda_ratio_std"] / np.sqrt(ratio_agg["n_seed"].clip(lower=1))

    # reliability flag using both theory amplitude and empirical one-step signal
    # This avoids over-interpreting modes that die within one step.
    noise_scale = float(np.nanmedian(np.abs(per_seed_df["A1"].to_numpy())))
    agg["unreliable"] = (
        (np.abs(agg["lambda_theory"]) < 0.03)
        | (np.abs(agg["A1_mean"]) < max(3.0 * noise_scale / max(math.sqrt(n_seeds), 1.0), 2e-3))
        | (agg["fit_r2_mean"] < 0.7)
    )

    selected_amp_runs_arr = np.vstack(selected_amp_runs)
    selected_amp_norm_runs_arr = np.vstack(selected_amp_norm_runs)
    selected_theory_arr = np.vstack(selected_theory_runs)

    t = np.arange(selected_amp_runs_arr.shape[1])
    selected_decay_df = pd.DataFrame(
        {
            "t": t,
            "theory_norm_mean": np.mean(selected_theory_arr, axis=0),
            "theory_norm_se": sem(selected_theory_arr, axis=0),
            "amp_mean": np.mean(selected_amp_runs_arr, axis=0),
            "amp_se": sem(selected_amp_runs_arr, axis=0),
            "amp_norm_mean": np.mean(selected_amp_norm_runs_arr, axis=0),
            "amp_norm_se": sem(selected_amp_norm_runs_arr, axis=0),
        }
    )
    selected_decay_df.insert(0, "dynamics", dynamics)

    return {
        "config": asdict(cfg_dyn),
        "m_star": m_star,
        "roots": roots,
        "Lambda_star": Lambda,
        "mu": mu,
        "sigma_eff": sig,
        "per_seed": per_seed_df,
        "aggregate": agg,
        "ratio_aggregate": ratio_agg,
        "selected_decay": selected_decay_df,
        "results_by_seed": results_by_seed,
    }


# -----------------------------------------------------------------------------
# figure builders
# -----------------------------------------------------------------------------


def plot_combined_lambda_spectrum(
    theory_df: pd.DataFrame,
    gauss_df: pd.DataFrame,
    micro_df: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(theory_df["q_over_pi"], theory_df["lambda_theory"], "-o", label="theory", lw=2)

    for df, label, color, marker in [
        (gauss_df, "gaussian_map", "tab:blue", "o"),
        (micro_df, "microscopic", "tab:orange", "s"),
    ]:
        reliable = ~df["unreliable"].astype(bool)
        unreliable = df["unreliable"].astype(bool)
        if reliable.any():
            sub = df[reliable]
            ax.errorbar(
                sub["q_over_pi"], sub["lambda_fit_mean"], yerr=sub["lambda_fit_se"],
                fmt=marker, color=color, label=label, capsize=3, ms=6, lw=1.2
            )
        if unreliable.any():
            sub = df[unreliable]
            ax.errorbar(
                sub["q_over_pi"], sub["lambda_fit_mean"], yerr=sub["lambda_fit_se"],
                fmt=marker, mfc="white", mec=color, ecolor=color,
                label=f"{label} (low-SNR)", capsize=3, ms=6, lw=1.0, alpha=0.8
            )

    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$q/\pi$")
    ax.set_ylabel(r"$\lambda(q)$")
    ax.set_title("Spatial-mode relaxation spectrum")
    ax.legend(fontsize=9)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)



def plot_normalized_spectrum(
    ratio_theory_df: pd.DataFrame,
    gauss_ratio: pd.DataFrame,
    micro_ratio: pd.DataFrame,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    ax.plot(ratio_theory_df["q_over_pi"], ratio_theory_df["kernel_hat"], "-o", label=r"$\widehat K_R(q)$", lw=2)

    for df, label, color, marker in [
        (gauss_ratio, "gaussian_map", "tab:blue", "o"),
        (micro_ratio, "microscopic", "tab:orange", "s"),
    ]:
        ax.errorbar(
            df["q_over_pi"], df["lambda_ratio_mean"], yerr=df["lambda_ratio_se"],
            fmt=marker, color=color, label=rf"{label}: $\lambda(q)/\lambda(0)$",
            capsize=3, ms=6, lw=1.2
        )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel(r"$q/\pi$")
    ax.set_ylabel("normalized eigenvalue")
    ax.set_title("Kernel-shape validation")
    ax.legend(fontsize=9)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)



def plot_selected_decay(
    gauss_decay: pd.DataFrame,
    micro_decay: pd.DataFrame,
    lambda_th: float,
    selected_mode: int,
    q: float,
    output_path: Path,
    tmax_zoom: int | None = 6,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)

    for df, label, color in [
        (gauss_decay, "gaussian_map", "tab:blue"),
        (micro_decay, "microscopic", "tab:orange"),
    ]:
        ax.errorbar(
            df["t"], df["amp_norm_mean"], yerr=df["amp_norm_se"],
            fmt="o-", color=color, capsize=3, ms=4, lw=1.2, label=label
        )
    ax.plot(gauss_decay["t"], gauss_decay["theory_norm_mean"], "k--", lw=2, label=r"theory: $\lambda(q)^t$")
    ax.axhline(0.0, color="k", lw=0.8)
    if tmax_zoom is not None:
        ax.set_xlim(0, tmax_zoom)
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$A_q(t)/A_q(0)$")
    ax.set_title(rf"Selected mode decay: n={selected_mode}, $q={q:.4f}$, $\lambda_{{th}}={lambda_th:.4f}$")
    ax.legend(fontsize=9)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)



def plot_negative_mode(
    neg_gauss: pd.DataFrame,
    neg_micro: pd.DataFrame,
    lambda_th: float,
    negative_mode: int,
    q: float,
    output_path: Path,
    tmax_zoom: int | None = 8,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    for df, label, color in [
        (neg_gauss, "gaussian_map", "tab:blue"),
        (neg_micro, "microscopic", "tab:orange"),
    ]:
        ax.errorbar(df["t"], df["amp_norm_mean"], yerr=df["amp_norm_se"], fmt="o-", color=color,
                    ms=4, lw=1.2, capsize=3, label=label)
    ax.plot(neg_gauss["t"], neg_gauss["theory_norm_mean"], "k--", lw=2, label=r"theory: $\lambda(q)^t$")
    ax.axhline(0.0, color="k", lw=0.8)
    if tmax_zoom is not None:
        ax.set_xlim(0, tmax_zoom)
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$A_q(t)/A_q(0)$")
    ax.set_title(rf"Negative-eigenvalue mode: n={negative_mode}, $q={q:.4f}$, $\lambda_{{th}}={lambda_th:.4f}$")
    ax.legend(fontsize=9)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# scans
# -----------------------------------------------------------------------------


def run_linearity_scan(
    base: ModuleType,
    cfg: Any,
    eps_list: list[float],
    modes: list[int],
    selected_mode: int,
    n_seeds: int,
    seed0: int,
) -> pd.DataFrame:
    rows = []
    for eps in eps_list:
        cfg_eps = replace(cfg, epsilon=eps)
        for dynamics in ["gaussian_map", "microscopic"]:
            suite = run_seed_suite(base, cfg_eps, modes, selected_mode, n_seeds, seed0, dynamics)
            agg = suite["aggregate"]
            row = agg[agg["mode_index"] == selected_mode].iloc[0]
            rows.append(
                {
                    "epsilon": eps,
                    "dynamics": dynamics,
                    "mode_index": selected_mode,
                    "lambda_theory": float(row["lambda_theory"]),
                    "lambda_fit_mean": float(row["lambda_fit_mean"]),
                    "lambda_fit_se": float(row["lambda_fit_se"]),
                    "fit_r2_mean": float(row["fit_r2_mean"]),
                }
            )
    return pd.DataFrame(rows)



def plot_linearity_scan(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    theory = df["lambda_theory"].iloc[0]
    ax.axhline(theory, color="k", linestyle="--", lw=1.8, label="theory")
    for dynamics, color, marker in [("gaussian_map", "tab:blue", "o"), ("microscopic", "tab:orange", "s")]:
        sub = df[df["dynamics"] == dynamics].sort_values("epsilon")
        ax.errorbar(sub["epsilon"], sub["lambda_fit_mean"], yerr=sub["lambda_fit_se"],
                    fmt=f"{marker}-", color=color, capsize=3, ms=6, lw=1.2, label=dynamics)
    ax.set_xlabel(r"perturbation amplitude $\epsilon$")
    ax.set_ylabel(r"$\lambda_{fit}(q_{sel})$")
    ax.set_title("Linearity check for selected mode")
    ax.legend(fontsize=9)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)



def run_R_scan(
    base: ModuleType,
    cfg: Any,
    R_list: list[int],
    modes: list[int],
    selected_mode: int,
    n_seeds: int,
    seed0: int,
) -> pd.DataFrame:
    rows = []
    for R in R_list:
        cfg_R = replace(cfg, R=R)
        for dynamics in ["gaussian_map", "microscopic"]:
            suite = run_seed_suite(base, cfg_R, modes, selected_mode, n_seeds, seed0, dynamics)
            ratio = suite["ratio_aggregate"]
            agg = suite["aggregate"]
            theory_map = {int(r["mode_index"]): float(r["kernel_hat"]) for _, r in ratio.iterrows()}
            for _, row in ratio.iterrows():
                rows.append(
                    {
                        "R": R,
                        "dynamics": dynamics,
                        "mode_index": int(row["mode_index"]),
                        "q": float(row["q"]),
                        "q_over_pi": float(row["q_over_pi"]),
                        "kernel_hat": float(row["kernel_hat"]),
                        "lambda_ratio_mean": float(row["lambda_ratio_mean"]),
                        "lambda_ratio_se": float(row["lambda_ratio_se"]),
                    }
                )
    return pd.DataFrame(rows)



def plot_R_scan(df: pd.DataFrame, output_path: Path) -> None:
    R_values = sorted(df["R"].unique())
    fig, axes = plt.subplots(len(R_values), 1, figsize=(7.8, 4.0 * len(R_values)), constrained_layout=True, sharex=True)
    if len(R_values) == 1:
        axes = [axes]
    for ax, R in zip(axes, R_values):
        subR = df[df["R"] == R]
        theory = subR.drop_duplicates(subset=["mode_index"]).sort_values("mode_index")
        ax.plot(theory["q_over_pi"], theory["kernel_hat"], "k-o", lw=1.8, label=rf"theory $\widehat K_R(q)$, R={R}")
        for dynamics, color, marker in [("gaussian_map", "tab:blue", "o"), ("microscopic", "tab:orange", "s")]:
            sub = subR[subR["dynamics"] == dynamics].sort_values("mode_index")
            ax.errorbar(sub["q_over_pi"], sub["lambda_ratio_mean"], yerr=sub["lambda_ratio_se"],
                        fmt=marker, color=color, capsize=3, ms=5, lw=1.2, label=dynamics)
        ax.axhline(0.0, color="k", lw=0.8)
        ax.set_ylabel(r"$\lambda(q)/\lambda(0)$")
        ax.legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel(r"$q/\pi$")
    fig.suptitle("R-dependence of normalized spatial spectrum")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build presentation-ready figures for lambda(q) analysis.")
    p.add_argument("--base-script", type=Path, default=SCRIPT_DIR / "spatial_mode_ensemble_validation.py")
    p.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "presentation_materials")

    p.add_argument("--N", type=int, default=96)
    p.add_argument("--R", type=int, default=12)
    p.add_argument("--T", type=int, default=40)
    p.add_argument("--ensemble", type=int, default=2000)
    p.add_argument("--a", type=float, default=1.0, dest="lattice_spacing")

    p.add_argument("--B", type=float, default=0.72)
    p.add_argument("--sigma-J", type=float, default=1.0, dest="sigma_J")
    p.add_argument("--sigma-phi", type=float, default=0.06, dest="sigma_phi")
    p.add_argument("--phi-bar", type=float, default=0.0, dest="phi_bar")
    p.add_argument("--h", type=float, default=0.0)

    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--fixed-point-guess", type=float, default=0.0)
    p.add_argument("--fit-steps", type=int, default=5)

    p.add_argument("--modes", type=parse_int_list, default=parse_int_list("0,1,2,4,6,8"))
    p.add_argument("--selected-mode", type=int, default=2)
    p.add_argument("--negative-mode", type=int, default=6)

    p.add_argument("--n-seeds", type=int, default=12)
    p.add_argument("--seed0", type=int, default=20260801)

    p.add_argument("--epsilon-scan", type=parse_float_list, default=[])
    p.add_argument("--R-scan", type=parse_int_list, default=[])

    return p



def main() -> None:
    args = build_parser().parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    base = load_base_module(args.base_script)
    cfg = base.Config(
        N=args.N,
        R=args.R,
        T=args.T,
        ensemble=args.ensemble,
        lattice_spacing=args.lattice_spacing,
        B=args.B,
        sigma_J=args.sigma_J,
        sigma_phi=args.sigma_phi,
        phi_bar=args.phi_bar,
        h=args.h,
        epsilon=args.epsilon,
        fixed_point_guess=args.fixed_point_guess,
        rng_seed=args.seed0,
        fit_steps=args.fit_steps,
        dynamics="gaussian_map",
    )

    modes = args.modes
    selected_mode = args.selected_mode
    negative_mode = args.negative_mode

    # main matched-condition comparison
    gauss = run_seed_suite(base, cfg, modes, selected_mode, args.n_seeds, args.seed0, "gaussian_map")
    micro = run_seed_suite(base, cfg, modes, selected_mode, args.n_seeds, args.seed0, "microscopic")

    # save raw tables
    gauss["per_seed"].to_csv(out / "gaussian_per_seed_lambda.csv", index=False)
    micro["per_seed"].to_csv(out / "microscopic_per_seed_lambda.csv", index=False)
    gauss["aggregate"].to_csv(out / "gaussian_lambda_aggregate.csv", index=False)
    micro["aggregate"].to_csv(out / "microscopic_lambda_aggregate.csv", index=False)
    gauss["ratio_aggregate"].to_csv(out / "gaussian_ratio_aggregate.csv", index=False)
    micro["ratio_aggregate"].to_csv(out / "microscopic_ratio_aggregate.csv", index=False)
    gauss["selected_decay"].to_csv(out / f"gaussian_selected_mode_n{selected_mode}_decay.csv", index=False)
    micro["selected_decay"].to_csv(out / f"microscopic_selected_mode_n{selected_mode}_decay.csv", index=False)

    # integrated figures
    theory_df = gauss["aggregate"].sort_values("mode_index")
    plot_combined_lambda_spectrum(theory_df, gauss["aggregate"].sort_values("mode_index"), micro["aggregate"].sort_values("mode_index"), out / "integrated_lambda_spectrum.png")
    plot_normalized_spectrum(theory_df, gauss["ratio_aggregate"].sort_values("mode_index"), micro["ratio_aggregate"].sort_values("mode_index"), out / "integrated_normalized_spectrum.png")

    q_sel = float(theory_df.loc[theory_df["mode_index"] == selected_mode, "q"].iloc[0])
    lam_sel = float(theory_df.loc[theory_df["mode_index"] == selected_mode, "lambda_theory"].iloc[0])
    plot_selected_decay(
        gauss["selected_decay"],
        micro["selected_decay"],
        lambda_th=lam_sel,
        selected_mode=selected_mode,
        q=q_sel,
        output_path=out / f"selected_mode_n{selected_mode}_decay_comparison.png",
        tmax_zoom=min(6, args.T),
    )

    # negative mode supplementary figure
    if negative_mode in modes:
        gauss_neg = run_seed_suite(base, cfg, modes, negative_mode, args.n_seeds, args.seed0, "gaussian_map")["selected_decay"]
        micro_neg = run_seed_suite(base, cfg, modes, negative_mode, args.n_seeds, args.seed0, "microscopic")["selected_decay"]
        q_neg = float(theory_df.loc[theory_df["mode_index"] == negative_mode, "q"].iloc[0])
        lam_neg = float(theory_df.loc[theory_df["mode_index"] == negative_mode, "lambda_theory"].iloc[0])
        gauss_neg.to_csv(out / f"gaussian_negative_mode_n{negative_mode}_decay.csv", index=False)
        micro_neg.to_csv(out / f"microscopic_negative_mode_n{negative_mode}_decay.csv", index=False)
        plot_negative_mode(
            gauss_neg, micro_neg,
            lambda_th=lam_neg,
            negative_mode=negative_mode,
            q=q_neg,
            output_path=out / f"negative_mode_n{negative_mode}_signflip.png",
            tmax_zoom=min(8, args.T),
        )

    # optional scans
    if len(args.epsilon_scan) > 0:
        eps_df = run_linearity_scan(base, cfg, args.epsilon_scan, modes, selected_mode, max(4, min(args.n_seeds, 8)), args.seed0)
        eps_df.to_csv(out / "linearity_epsilon_scan.csv", index=False)
        plot_linearity_scan(eps_df, out / "linearity_epsilon_scan.png")

    if len(args.R_scan) > 0:
        R_df = run_R_scan(base, cfg, args.R_scan, modes, selected_mode, max(4, min(args.n_seeds, 8)), args.seed0)
        R_df.to_csv(out / "R_scan_normalized_spectrum.csv", index=False)
        plot_R_scan(R_df, out / "R_scan_normalized_spectrum.png")

    summary = {
        "base_script": str(args.base_script),
        "matched_config": asdict(cfg),
        "modes": modes,
        "selected_mode": selected_mode,
        "negative_mode": negative_mode,
        "n_seeds": args.n_seeds,
        "seed0": args.seed0,
        "gaussian": {
            "m_star": gauss["m_star"],
            "Lambda_star": gauss["Lambda_star"],
            "mu": gauss["mu"],
            "sigma_eff": gauss["sigma_eff"],
        },
        "microscopic": {
            "m_star": micro["m_star"],
            "Lambda_star": micro["Lambda_star"],
            "mu": micro["mu"],
            "sigma_eff": micro["sigma_eff"],
        },
        "note": "gaussian_map and microscopic are compared under matched N,R,T,ensemble,epsilon,fit_steps, and seed set.",
    }
    with open(out / "presentation_materials_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # brief README
    readme = out / "README_presentation_materials.txt"
    readme.write_text(
        """作成された主資料
----------------
1. integrated_lambda_spectrum.png
   理論・gaussian_map・microscopic を同一図上で比較した固有値スペクトル。
   中抜きマーカーは low-SNR / unreliable 判定。

2. integrated_normalized_spectrum.png
   lambda(q)/lambda(0) と Khat_R(q) の比較。
   空間カーネル形状の頑健性を示す主図。

3. selected_mode_n*_decay_comparison.png
   選択モードの A_q(t)/A_q(0) の時間発展。
   理論線 lambda(q)^t と、gaussian_map / microscopic の平均±SE を比較。

4. negative_mode_n*_signflip.png
   負の固有値モードの符号反転しながらの減衰。

5. linearity_epsilon_scan.png（指定時のみ）
   epsilon 依存性。lambda_fit が epsilon に依らないことを確認する資料。

6. R_scan_normalized_spectrum.png（指定時のみ）
   R 依存性。lambda(q)/lambda(0) が Khat_R(q) に従うことを確認する資料。

主要CSV
-------
- gaussian_lambda_aggregate.csv
- microscopic_lambda_aggregate.csv
- gaussian_ratio_aggregate.csv
- microscopic_ratio_aggregate.csv
- gaussian_selected_mode_n*_decay.csv
- microscopic_selected_mode_n*_decay.csv

発表での推奨使用法
------------------
- 口頭発表本編：1,2,3
- 補足またはバックアップ：4,5,6
- ポスター：1 を主図、2 と 3 を右側に添える
""",
        encoding="utf-8",
    )

    print(f"Saved materials to: {out}")
    for p in sorted(out.iterdir()):
        print(p)


if __name__ == "__main__":
    main()
