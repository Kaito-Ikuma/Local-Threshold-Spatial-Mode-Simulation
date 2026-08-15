#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一次元局所二値しきい値モデル：空間モードのアンサンブル検証・可視化

目的
----
周期境界条件下で、線形化理論

    A_q(t+1) = lambda(q) A_q(t),
    lambda(q) = Lambda_* Khat_R(q)

を数値的に検証する。

本スクリプトは、以下を同時に出力する。

1. 代表的な二値実現 S_i(t)=±1
2. アンサンブル平均 ubar_i(t)=M^{-1} sum_alpha S_i^(alpha)(t)
3. コヒーレント平均 Abar_q(t)
4. 構造因子 S(k,t)=N^{-1}<|sum_i delta S_i exp(-ikx_i)|^2>
5. Abar_q(t) の理論減衰 A_q(0) lambda(q)^t との比較
6. 複数波数 q_n に対する lambda_sim(q_n) と lambda_th(q_n) の比較

モデル
------
    H_i(t) = (1/(2R)) sum_{r=1}^R [
                 J_{i,+r}(t) S_{i+r}(t)
               + J_{i,-r}(t) S_{i-r}(t)
             ] - phi_i + h

    S_i(t+1) = sign(H_i(t))

    J_{i,r}(t) ~ Normal(mu, sigma_J^2)       （時間ごとに更新：annealed）
    phi_i       ~ Normal(phi_bar, sigma_phi^2)（各試行で固定：quenched）

局所平均場近似では

    sigma_eff^2 = sigma_J^2/(2R) + sigma_phi^2
    B           = 2 mu / (sqrt(2pi) sigma_eff)

    F(u) = 2 Phi((mu u + h - phi_bar)/sigma_eff) - 1

一様固定点 m_* は m_*=F(m_*) を満たし、

    Lambda_* = F'(m_*)
             = B exp[-z_*^2/2],
    z_*      = (mu m_* + h - phi_bar)/sigma_eff

である。一様 top-hat カーネルでは

    Khat_R(q) = (1/R) sum_{r=1}^R cos(q r a)

である。

実行例
------
軽い動作確認：
    python spatial_mode_ensemble_validation.py \
        --N 48 --R 3 --T 30 --ensemble 120 \
        --modes 0,1,2,4 --animate-mode 2 --format gif

本計算例：
    python spatial_mode_ensemble_validation.py \
        --N 128 --R 6 --T 60 --ensemble 1000 \
        --modes 0,1,2,3,4,6,8,12,16 \
        --animate-mode 4 --epsilon 0.04 --fit-steps 12 --format mp4

必要ライブラリ
------------
    numpy, pandas, matplotlib, scipy, pillow
MP4出力には ffmpeg が必要。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter, writers
import numpy as np
import pandas as pd
from scipy.special import ndtr


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    N: int = 64
    R: int = 4
    T: int = 45
    ensemble: int = 400
    lattice_spacing: float = 1.0

    B: float = 0.72
    sigma_J: float = 1.0
    sigma_phi: float = 0.06
    phi_bar: float = 0.0
    h: float = 0.0

    epsilon: float = 0.05
    fixed_point_guess: float = 0.0
    rng_seed: int = 20260731
    fit_steps: int = 10
    dynamics: str = "gaussian_map"


@dataclass
class ModeResult:
    mode_index: int
    q: float
    khat: float
    lambda_theory: float
    lambda_fit: float
    fit_r2: float

    representative_states: np.ndarray       # (T+1, N)
    ensemble_mean: np.ndarray               # plus-perturbed ensemble mean, (T+1, N)
    response_mean: np.ndarray               # symmetric response (u_plus-u_minus)/2
    coherent_spectrum: np.ndarray            # symmetric-response coherent spectrum
    coherent_plus_spectrum: np.ndarray       # direct coherent average of plus ensemble
    structure_factor: np.ndarray             # (T+1, N//2+1), real
    mean_magnetization: np.ndarray            # (T+1,)
    cosine_amplitude: np.ndarray              # (T+1,)
    theory_cosine_amplitude: np.ndarray       # (T+1,)


# -----------------------------------------------------------------------------
# 数学・モデル定義
# -----------------------------------------------------------------------------


def configure_font() -> None:
    """利用可能なら日本語フォントを選ぶ。"""
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


def normal_cdf_scalar(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def sigma_eff(config: Config) -> float:
    c = 2 * config.R
    if c <= 0:
        raise ValueError("R must be >= 1")
    return math.sqrt(config.sigma_J**2 / c + config.sigma_phi**2)


def mu_from_B(config: Config) -> float:
    sig = sigma_eff(config)
    return config.B * math.sqrt(2.0 * math.pi) * sig / 2.0


def mean_field_map(u: float, config: Config) -> float:
    sig = sigma_eff(config)
    mu = mu_from_B(config)
    z = (mu * u + config.h - config.phi_bar) / sig
    return 2.0 * normal_cdf_scalar(z) - 1.0


def find_fixed_points(config: Config, grid_size: int = 4001) -> list[float]:
    """[-1,1] 上の F(m)-m=0 の根を走査＋二分法で求める。"""
    xs = np.linspace(-1.0, 1.0, grid_size)
    vals = np.array([mean_field_map(float(x), config) - float(x) for x in xs])
    roots: list[float] = []

    def bisect(a: float, b: float, fa: float, fb: float) -> float:
        if abs(fa) < 1e-14:
            return a
        if abs(fb) < 1e-14:
            return b
        for _ in range(100):
            c = 0.5 * (a + b)
            fc = mean_field_map(c, config) - c
            if abs(fc) < 1e-13 or abs(b - a) < 1e-12:
                return c
            if fa * fc <= 0.0:
                b, fb = c, fc
            else:
                a, fa = c, fc
        return 0.5 * (a + b)

    for j in range(grid_size - 1):
        a, b = float(xs[j]), float(xs[j + 1])
        fa, fb = float(vals[j]), float(vals[j + 1])
        if abs(fa) < 1e-10:
            roots.append(a)
        if fa * fb < 0.0:
            roots.append(bisect(a, b, fa, fb))

    if abs(vals[-1]) < 1e-10:
        roots.append(float(xs[-1]))

    roots_sorted: list[float] = []
    for root in sorted(roots):
        if not roots_sorted or abs(root - roots_sorted[-1]) > 1e-6:
            roots_sorted.append(root)
    return roots_sorted


def choose_fixed_point(config: Config) -> tuple[float, list[float]]:
    roots = find_fixed_points(config)
    if not roots:
        raise RuntimeError("一様固定点を [-1,1] で検出できませんでした。")
    chosen = min(roots, key=lambda x: abs(x - config.fixed_point_guess))
    return float(chosen), roots


def lambda_star(m_star: float, config: Config) -> float:
    sig = sigma_eff(config)
    mu = mu_from_B(config)
    z = (mu * m_star + config.h - config.phi_bar) / sig
    return config.B * math.exp(-0.5 * z * z)


def q_from_mode(n: int, config: Config) -> float:
    return 2.0 * math.pi * n / (config.N * config.lattice_spacing)


def kernel_hat(mode_index: int, config: Config) -> float:
    q = q_from_mode(mode_index, config)
    r = np.arange(1, config.R + 1, dtype=float)
    return float(np.mean(np.cos(q * r * config.lattice_spacing)))


def cosine_amplitude_from_coherent(
    coherent: np.ndarray,
    mode_index: int,
    N: int,
) -> np.ndarray:
    """
    u_i = m_* + eps cos(q_n x_i) の eps に対応する実余弦振幅。

    通常の非零・非Nyquistモードでは DFT の ±q に eps/2 ずつ入るため
        amplitude = 2 Re(Abar_q)
    n=0 および Nyquist では一つの実モードなので factor=1。
    """
    if mode_index == 0 or (N % 2 == 0 and mode_index == N // 2):
        return coherent[:, mode_index].real.copy()
    return 2.0 * coherent[:, mode_index].real


def estimate_lambda(amplitude: np.ndarray, fit_steps: int) -> tuple[float, float]:
    """A(t+1)=lambda A(t) を原点通過回帰で推定する。"""
    n_pairs = min(max(fit_steps, 1), len(amplitude) - 1)
    x = np.asarray(amplitude[:n_pairs], dtype=float)
    y = np.asarray(amplitude[1 : n_pairs + 1], dtype=float)

    denom = float(np.dot(x, x))
    if denom <= 1e-20:
        return math.nan, math.nan

    lam = float(np.dot(x, y) / denom)
    y_pred = lam * x
    sse = float(np.sum((y - y_pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 1e-20 else math.nan
    return lam, r2


# -----------------------------------------------------------------------------
# シミュレーション
# -----------------------------------------------------------------------------


def validate_config(config: Config, modes: Sequence[int], animate_mode: int) -> None:
    if config.dynamics not in {"gaussian_map", "microscopic"}:
        raise ValueError("dynamics must be gaussian_map or microscopic")
    if config.N < 8:
        raise ValueError("N must be >= 8")
    if config.R < 1 or 2 * config.R >= config.N:
        raise ValueError("1 <= R < N/2 が必要です。")
    if config.T < 2:
        raise ValueError("T must be >= 2")
    if config.ensemble < 2:
        raise ValueError("ensemble must be >= 2")
    if not (0.0 < config.epsilon < 1.0):
        raise ValueError("epsilon must be in (0,1)")

    max_mode = config.N // 2
    for n in modes:
        if n < 0 or n > max_mode:
            raise ValueError(f"mode {n} is outside [0, N/2]")
    if animate_mode not in modes:
        raise ValueError("animate-mode must be included in modes")


def initial_probabilities(
    mode_index: int,
    m_star: float,
    config: Config,
) -> np.ndarray:
    x = np.arange(config.N, dtype=float) * config.lattice_spacing
    q = q_from_mode(mode_index, config)
    u0 = m_star + config.epsilon * np.cos(q * x)
    if np.min(u0) < -1.0 or np.max(u0) > 1.0:
        raise ValueError(
            "初期平均 m_* + epsilon cos(qx) が [-1,1] を外れます。"
            " epsilon を小さくしてください。"
        )
    return 0.5 * (1.0 + u0)


def simulate_mode(
    mode_index: int,
    m_star: float,
    Lambda: float,
    config: Config,
    seed_offset: int,
) -> ModeResult:
    """一つの波数モードを同じ位相で M 試行初期化して同期更新する。"""
    rng = np.random.default_rng(config.rng_seed + seed_offset)
    mu = mu_from_B(config)

    # 対称な ±epsilon 摂動を、同じ一様乱数で初期化する。
    # response=(<S_plus>-<S_minus>)/2 とすると、初期期待値は epsilon cos(qx)。
    x_sites = np.arange(config.N, dtype=float) * config.lattice_spacing
    q = q_from_mode(mode_index, config)
    u_plus0 = m_star + config.epsilon * np.cos(q * x_sites)
    u_minus0 = m_star - config.epsilon * np.cos(q * x_sites)
    p_plus = 0.5 * (1.0 + u_plus0)
    p_minus = 0.5 * (1.0 + u_minus0)
    initial_uniform = rng.random((config.ensemble, config.N))
    states_plus = np.where(initial_uniform < p_plus[None, :], 1, -1).astype(np.int8)
    states_minus = np.where(initial_uniform < p_minus[None, :], 1, -1).astype(np.int8)

    thresholds = None
    if config.dynamics == "microscopic":
        thresholds = rng.normal(
            loc=config.phi_bar,
            scale=config.sigma_phi,
            size=(config.ensemble, config.N),
        )

    n_k = config.N // 2 + 1
    representative = np.empty((config.T + 1, config.N), dtype=np.int8)
    ensemble_mean = np.empty((config.T + 1, config.N), dtype=float)
    response_mean = np.empty((config.T + 1, config.N), dtype=float)
    coherent = np.empty((config.T + 1, n_k), dtype=np.complex128)
    coherent_plus = np.empty((config.T + 1, n_k), dtype=np.complex128)
    structure = np.empty((config.T + 1, n_k), dtype=float)
    mean_m = np.empty(config.T + 1, dtype=float)

    def record(t: int) -> None:
        representative[t] = states_plus[0]
        ubar_plus = states_plus.mean(axis=0)
        ubar_minus = states_minus.mean(axis=0)
        response = 0.5 * (ubar_plus - ubar_minus)
        ensemble_mean[t] = ubar_plus
        response_mean[t] = response
        mean_m[t] = float(np.mean(ubar_plus))

        # コヒーレント平均は対称差分応答から計算する。
        # これにより背景の有限 M 揺らぎと偶数次非線形項を抑える。
        coherent[t] = np.fft.rfft(response) / config.N
        coherent_plus[t] = np.fft.rfft(ubar_plus - m_star) / config.N

        # 構造因子は plus 側の各二値実現から計算する。
        trial_centered = states_plus - states_plus.mean(axis=1, keepdims=True)
        trial_fft = np.fft.rfft(trial_centered, axis=1)
        structure[t] = np.mean(np.abs(trial_fft) ** 2, axis=0) / config.N

    record(0)

    for t in range(config.T):
        if config.dynamics == "gaussian_map":
            # ガウス閉包で得た局所平均写像を確率的に直接実装する。
            # E[S_i(t+1)|S(t)] = F(sum_r K_r S_{i+r}) となる。
            local_plus = np.zeros((config.ensemble, config.N), dtype=float)
            local_minus = np.zeros((config.ensemble, config.N), dtype=float)
            for r in range(1, config.R + 1):
                local_plus += np.roll(states_plus, shift=-r, axis=1)
                local_plus += np.roll(states_plus, shift=+r, axis=1)
                local_minus += np.roll(states_minus, shift=-r, axis=1)
                local_minus += np.roll(states_minus, shift=+r, axis=1)
            local_plus /= (2 * config.R)
            local_minus /= (2 * config.R)

            sig = sigma_eff(config)
            p_next_plus = ndtr(
                (mu * local_plus + config.h - config.phi_bar) / sig
            )
            p_next_minus = ndtr(
                (mu * local_minus + config.h - config.phi_bar) / sig
            )
            common_uniform = rng.random((config.ensemble, config.N))
            states_plus = np.where(
                common_uniform < p_next_plus, 1, -1
            ).astype(np.int8)
            states_minus = np.where(
                common_uniform < p_next_minus, 1, -1
            ).astype(np.int8)

        else:
            # 元のミクロモデル：時間依存 J と quenched threshold を明示生成。
            assert thresholds is not None
            interaction_plus = np.zeros((config.ensemble, config.N), dtype=float)
            interaction_minus = np.zeros((config.ensemble, config.N), dtype=float)
            for r in range(1, config.R + 1):
                for signed_r in (-r, r):
                    z = rng.standard_normal((config.ensemble, config.N))
                    J = mu + config.sigma_J * z
                    interaction_plus += J * np.roll(
                        states_plus, shift=-signed_r, axis=1
                    )
                    interaction_minus += J * np.roll(
                        states_minus, shift=-signed_r, axis=1
                    )

            interaction_plus /= (2 * config.R)
            interaction_minus /= (2 * config.R)
            field_plus = interaction_plus - thresholds + config.h
            field_minus = interaction_minus - thresholds + config.h
            states_plus = np.where(field_plus >= 0.0, 1, -1).astype(np.int8)
            states_minus = np.where(field_minus >= 0.0, 1, -1).astype(np.int8)

        record(t + 1)

    q = q_from_mode(mode_index, config)
    khat = kernel_hat(mode_index, config)
    lam_th = Lambda * khat
    amp = cosine_amplitude_from_coherent(coherent, mode_index, config.N)
    theory_amp = config.epsilon * np.power(lam_th, np.arange(config.T + 1))
    lam_fit, r2 = estimate_lambda(amp, config.fit_steps)

    return ModeResult(
        mode_index=mode_index,
        q=q,
        khat=khat,
        lambda_theory=lam_th,
        lambda_fit=lam_fit,
        fit_r2=r2,
        representative_states=representative,
        ensemble_mean=ensemble_mean,
        response_mean=response_mean,
        coherent_spectrum=coherent,
        coherent_plus_spectrum=coherent_plus,
        structure_factor=structure,
        mean_magnetization=mean_m,
        cosine_amplitude=amp,
        theory_cosine_amplitude=theory_amp,
    )


# -----------------------------------------------------------------------------
# 保存・可視化
# -----------------------------------------------------------------------------


def save_mode_timeseries(result: ModeResult, output_dir: Path) -> None:
    t = np.arange(len(result.cosine_amplitude))
    q_index = result.mode_index
    coh = result.coherent_spectrum[:, q_index]
    coh_plus = result.coherent_plus_spectrum[:, q_index]
    df = pd.DataFrame(
        {
            "t": t,
            "mode_index": q_index,
            "q": result.q,
            "mean_magnetization": result.mean_magnetization,
            "coherent_A_real": coh.real,
            "coherent_A_imag": coh.imag,
            "coherent_A_abs": np.abs(coh),
            "direct_coherent_A_plus_real": coh_plus.real,
            "direct_coherent_A_plus_imag": coh_plus.imag,
            "direct_coherent_A_plus_abs": np.abs(coh_plus),
            "cosine_amplitude": result.cosine_amplitude,
            "theory_cosine_amplitude": result.theory_cosine_amplitude,
        }
    )
    df.to_csv(output_dir / f"mode_timeseries_n{q_index}.csv", index=False)


def save_selected_spatiotemporal_data(result: ModeResult, output_dir: Path) -> None:
    T_plus_1, N = result.ensemble_mean.shape
    rows = []
    x = np.arange(N)
    q = result.q
    for t in range(T_plus_1):
        theory_profile = (
            result.ensemble_mean[0].mean() * 0.0
            + result.theory_cosine_amplitude[t] * np.cos(q * x)
        )
        for i in range(N):
            rows.append(
                {
                    "t": t,
                    "i": i,
                    "representative_spin": int(result.representative_states[t, i]),
                    "ensemble_mean_plus": float(result.ensemble_mean[t, i]),
                    "symmetric_response": float(result.response_mean[t, i]),
                    "theory_perturbation": float(theory_profile[i]),
                }
            )
    pd.DataFrame(rows).to_csv(
        output_dir / f"spatiotemporal_selected_n{result.mode_index}.csv",
        index=False,
    )

    spec_rows = []
    q_values = 2.0 * np.pi * np.arange(result.coherent_spectrum.shape[1]) / N
    for t in range(T_plus_1):
        for k_index, qk in enumerate(q_values):
            spec_rows.append(
                {
                    "t": t,
                    "mode_index": k_index,
                    "q": qk,
                    "coherent_response_abs": float(abs(result.coherent_spectrum[t, k_index])),
                    "coherent_plus_abs": float(abs(result.coherent_plus_spectrum[t, k_index])),
                    "coherent_response_real": float(result.coherent_spectrum[t, k_index].real),
                    "coherent_response_imag": float(result.coherent_spectrum[t, k_index].imag),
                    "coherent_plus_real": float(result.coherent_plus_spectrum[t, k_index].real),
                    "coherent_plus_imag": float(result.coherent_plus_spectrum[t, k_index].imag),
                    "structure_factor": float(result.structure_factor[t, k_index]),
                }
            )
    pd.DataFrame(spec_rows).to_csv(
        output_dir / f"spectra_selected_n{result.mode_index}.csv",
        index=False,
    )


def make_lambda_summary(
    results: Sequence[ModeResult],
    Lambda: float,
    output_dir: Path,
) -> pd.DataFrame:
    rows = []
    lambda0_sim = math.nan
    for result in results:
        if result.mode_index == 0:
            lambda0_sim = result.lambda_fit
            break

    for result in results:
        rel_error = (
            abs(result.lambda_fit - result.lambda_theory)
            / max(abs(result.lambda_theory), 1e-12)
            if math.isfinite(result.lambda_fit)
            else math.nan
        )
        rows.append(
            {
                "mode_index": result.mode_index,
                "q": result.q,
                "q_over_pi": result.q / math.pi,
                "kernel_hat": result.khat,
                "Lambda_star": Lambda,
                "lambda_theory": result.lambda_theory,
                "lambda_fit": result.lambda_fit,
                "fit_r2": result.fit_r2,
                "relative_error": rel_error,
                "lambda_fit_over_lambda0_fit": (
                    result.lambda_fit / lambda0_sim
                    if math.isfinite(lambda0_sim) and abs(lambda0_sim) > 1e-12
                    else math.nan
                ),
            }
        )

    df = pd.DataFrame(rows).sort_values("mode_index")
    df.to_csv(output_dir / "lambda_spectrum.csv", index=False)
    return df


def make_lambda_validation_figure(df: pd.DataFrame, output_path: Path) -> None:
    configure_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    ax.plot(df["q_over_pi"], df["lambda_theory"], marker="o", label="theory")
    ax.scatter(df["q_over_pi"], df["lambda_fit"], marker="x", s=70, label="simulation")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel(r"$q/\pi$")
    ax.set_ylabel(r"$\lambda(q)$")
    ax.set_title("Spatial-mode eigenvalue")
    ax.legend()

    ax = axes[1]
    valid = np.isfinite(df["lambda_fit_over_lambda0_fit"].to_numpy())
    ax.plot(df["q_over_pi"], df["kernel_hat"], marker="o", label=r"$\widehat K_R(q)$")
    ax.scatter(
        df.loc[valid, "q_over_pi"],
        df.loc[valid, "lambda_fit_over_lambda0_fit"],
        marker="x",
        s=70,
        label=r"$\lambda_{sim}(q)/\lambda_{sim}(0)$",
    )
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel(r"$q/\pi$")
    ax.set_ylabel("normalized eigenvalue")
    ax.set_title("Kernel-shape validation")
    ax.legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_animation(
    result: ModeResult,
    config: Config,
    m_star: float,
    output_path: Path,
    fps: int,
) -> None:
    configure_font()
    T_plus_1, N = result.ensemble_mean.shape
    n_k = result.coherent_spectrum.shape[1]
    x = np.arange(N, dtype=float) * config.lattice_spacing
    k_indices = np.arange(n_k)
    q_values = 2.0 * np.pi * k_indices / (N * config.lattice_spacing)

    fig = plt.figure(figsize=(15.5, 10.0), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[0.75, 1.1, 1.1])
    ax_binary = fig.add_subplot(grid[0, 0])
    ax_arrows = fig.add_subplot(grid[0, 1])
    ax_profile = fig.add_subplot(grid[1, 0])
    ax_amp = fig.add_subplot(grid[1, 1])
    ax_coh = fig.add_subplot(grid[2, 0])
    ax_sf = fig.add_subplot(grid[2, 1])

    # --- 代表的二値実現 ------------------------------------------------------
    ax_binary.axhline(0.0, linewidth=0.8)
    ax_binary.set_xlim(-1, N)
    ax_binary.set_ylim(-0.4, 0.7)
    ax_binary.set_yticks([])
    ax_binary.set_xlabel("site i")
    ax_binary.set_title("Representative binary realization")
    up = ax_binary.scatter([], [], marker=r"$\uparrow$", s=125)
    down = ax_binary.scatter([], [], marker=r"$\downarrow$", s=125)
    binary_status = ax_binary.text(0.01, 0.05, "", transform=ax_binary.transAxes)

    # --- アンサンブル平均の角度矢印 ----------------------------------------
    ax_arrows.axhline(0.0, linewidth=0.8)
    ax_arrows.set_xlim(-1, N)
    ax_arrows.set_ylim(-0.8, 0.8)
    ax_arrows.set_yticks([])
    ax_arrows.set_xlabel("site i")
    ax_arrows.set_title(r"Ensemble mean arrows: $\theta_i=(\pi/2)\,\bar u_i$")
    u0 = result.ensemble_mean[0]
    theta0 = 0.5 * np.pi * u0
    quiver = ax_arrows.quiver(
        x,
        np.zeros(N),
        np.cos(theta0),
        np.sin(theta0),
        angles="xy",
        scale_units="xy",
        scale=2.6,
        width=0.003,
        pivot="mid",
    )

    # --- 実空間プロファイル --------------------------------------------------
    ax_profile.set_xlim(x[0], x[-1])
    ax_profile.set_ylim(-1.05, 1.05)
    ax_profile.set_xlabel("position x")
    ax_profile.set_ylabel(r"$m_*+\delta\bar u_i(t)$")
    ax_profile.set_title("Symmetric ensemble response vs. linear theory")
    profile_line, = ax_profile.plot(x, m_star + result.response_mean[0], marker=".", label="simulation")
    theory_line, = ax_profile.plot(
        x,
        m_star + result.theory_cosine_amplitude[0] * np.cos(result.q * x),
        linestyle="--",
        label="linear theory",
    )
    ax_profile.axhline(m_star, linewidth=0.8, linestyle=":", label=r"$m_*$")
    ax_profile.legend(loc="upper right")

    # --- A_q(t) ---------------------------------------------------------------
    t_all = np.arange(T_plus_1)
    amp_min = min(np.min(result.cosine_amplitude), np.min(result.theory_cosine_amplitude), -0.02)
    amp_max = max(np.max(result.cosine_amplitude), np.max(result.theory_cosine_amplitude), 0.02)
    padding = 0.12 * max(amp_max - amp_min, 0.05)
    ax_amp.set_xlim(0, config.T)
    ax_amp.set_ylim(amp_min - padding, amp_max + padding)
    ax_amp.set_xlabel("time t")
    ax_amp.set_ylabel(r"cosine amplitude $2\,\mathrm{Re}\,\bar A_q$")
    ax_amp.set_title(r"Coherent decay: $A_q(t)$")
    amp_line, = ax_amp.plot([], [], marker="o", markersize=3, label="ensemble")
    amp_theory_line, = ax_amp.plot([], [], linestyle="--", label=r"$\epsilon\lambda(q)^t$")
    amp_cursor = ax_amp.axvline(0, linewidth=0.8, linestyle=":")
    ax_amp.axhline(0.0, linewidth=0.8)
    ax_amp.legend(loc="best")
    amp_status = ax_amp.text(0.02, 0.04, "", transform=ax_amp.transAxes)

    # --- コヒーレントスペクトル ---------------------------------------------
    coh_values = np.abs(result.coherent_plus_spectrum)
    response_coh_values = np.abs(result.coherent_spectrum)
    coh_max = max(float(np.max(coh_values)), float(np.max(response_coh_values)), 1e-5)
    coh_bars = ax_coh.bar(k_indices, coh_values[0], label=r"$|\bar A_k^{(+)}|$")
    response_coh_line, = ax_coh.plot(
        k_indices, response_coh_values[0], marker="o", markersize=3,
        label=r"$|\delta\bar A_k|$"
    )
    ax_coh.axvline(result.mode_index, linestyle="--", linewidth=1.0)
    ax_coh.set_xlim(-0.5, n_k - 0.5)
    ax_coh.set_ylim(0.0, 1.1 * coh_max)
    ax_coh.set_xlabel("mode index k")
    ax_coh.set_ylabel(r"$|\bar A_k(t)|$")
    ax_coh.set_title("Coherent spectrum")
    ax_coh.legend(loc="upper right")

    # --- 構造因子 -------------------------------------------------------------
    sf_values = result.structure_factor
    sf_max = max(float(np.max(sf_values)), 1e-5)
    sf_line, = ax_sf.plot(k_indices, sf_values[0], marker="o", markersize=3)
    ax_sf.axvline(result.mode_index, linestyle="--", linewidth=1.0)
    ax_sf.set_xlim(0, n_k - 1)
    ax_sf.set_ylim(0.0, 1.1 * sf_max)
    ax_sf.set_xlabel("mode index k")
    ax_sf.set_ylabel(r"$\mathcal{S}(k,t)$")
    ax_sf.set_title("Structure factor")

    fig.suptitle(
        rf"Spatial mode validation: n={result.mode_index}, "
        rf"$q={result.q:.4f}$, $\lambda_{{th}}={result.lambda_theory:.4f}$"
    )

    def set_scatter_offsets(collection: mpl.collections.PathCollection, indices: np.ndarray) -> None:
        if len(indices) == 0:
            collection.set_offsets(np.empty((0, 2)))
        else:
            collection.set_offsets(np.column_stack([indices, np.full(len(indices), 0.12)]))

    def update(frame: int):
        S = result.representative_states[frame]
        up_idx = np.flatnonzero(S == 1)
        down_idx = np.flatnonzero(S == -1)
        set_scatter_offsets(up, up_idx)
        set_scatter_offsets(down, down_idx)
        binary_status.set_text(
            f"t={frame}   m_rep={np.mean(S):+.3f}   "
            f"m_ensemble={result.mean_magnetization[frame]:+.3f}"
        )

        ubar = result.ensemble_mean[frame]
        theta = 0.5 * np.pi * ubar
        quiver.set_UVC(np.cos(theta), np.sin(theta))

        profile_line.set_ydata(m_star + result.response_mean[frame])
        theory_line.set_ydata(
            m_star
            + result.theory_cosine_amplitude[frame] * np.cos(result.q * x)
        )

        amp_line.set_data(t_all[: frame + 1], result.cosine_amplitude[: frame + 1])
        amp_theory_line.set_data(
            t_all[: frame + 1], result.theory_cosine_amplitude[: frame + 1]
        )
        amp_cursor.set_xdata([frame, frame])
        amp_status.set_text(
            rf"$\lambda_{{fit}}={result.lambda_fit:.4f}$, "
            rf"$R^2={result.fit_r2:.3f}$"
        )

        current_coh = coh_values[frame]
        for rectangle, height in zip(coh_bars, current_coh):
            rectangle.set_height(float(height))
        response_coh_line.set_ydata(response_coh_values[frame])

        sf_line.set_ydata(sf_values[frame])

        return (
            up,
            down,
            binary_status,
            quiver,
            profile_line,
            theory_line,
            amp_line,
            amp_theory_line,
            amp_cursor,
            amp_status,
            *coh_bars,
            response_coh_line,
            sf_line,
        )

    animation = FuncAnimation(
        fig,
        update,
        frames=T_plus_1,
        interval=1000 / max(fps, 1),
        blit=False,
    )

    suffix = output_path.suffix.lower()
    if suffix == ".gif":
        animation.save(output_path, writer=PillowWriter(fps=fps), dpi=120)
    elif suffix == ".mp4":
        if not writers.is_available("ffmpeg"):
            raise RuntimeError("MP4保存には ffmpeg が必要です。GIFを指定してください。")
        animation.save(output_path, writer=FFMpegWriter(fps=fps, bitrate=2400), dpi=120)
    else:
        raise ValueError("output_path must end with .gif or .mp4")

    plt.close(fig)


def make_selected_static_figure(
    result: ModeResult,
    config: Config,
    m_star: float,
    output_path: Path,
) -> None:
    """学会スライド用の代表的な静止診断図。"""
    configure_font()
    x = np.arange(config.N) * config.lattice_spacing
    times = sorted(set([0, min(2, config.T), min(5, config.T), config.T]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)

    ax = axes[0]
    for t in times:
        ax.plot(x, m_star + result.response_mean[t], label=f"t={t}")
    ax.set_xlabel("position x")
    ax.set_ylabel(r"$m_*+\delta\bar u_i(t)$")
    ax.set_title("Symmetric ensemble-response profiles")
    ax.legend()

    ax = axes[1]
    t = np.arange(config.T + 1)
    ax.plot(t, result.cosine_amplitude, marker="o", markersize=3, label="simulation")
    ax.plot(t, result.theory_cosine_amplitude, linestyle="--", label="linear theory")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("time t")
    ax.set_ylabel(r"$2\,\mathrm{Re}\,\bar A_q(t)$")
    ax.set_title(
        rf"$\lambda_{{th}}={result.lambda_theory:.4f}$, "
        rf"$\lambda_{{fit}}={result.lambda_fit:.4f}$"
    )
    ax.legend()

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_modes(text: str) -> list[int]:
    values: list[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise argparse.ArgumentTypeError("modes is empty")
    return sorted(set(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="一次元局所二値モデルの空間モードをアンサンブル平均で検証する。"
    )
    parser.add_argument("--N", type=int, default=64)
    parser.add_argument("--R", type=int, default=4)
    parser.add_argument("--T", type=int, default=45)
    parser.add_argument("--ensemble", type=int, default=400)
    parser.add_argument("--a", type=float, default=1.0, dest="lattice_spacing")

    parser.add_argument("--B", type=float, default=0.72)
    parser.add_argument("--sigma-J", type=float, default=1.0, dest="sigma_J")
    parser.add_argument("--sigma-phi", type=float, default=0.06, dest="sigma_phi")
    parser.add_argument("--phi-bar", type=float, default=0.0, dest="phi_bar")
    parser.add_argument("--h", type=float, default=0.0)

    parser.add_argument("--epsilon", type=float, default=0.05)
    parser.add_argument("--fixed-point-guess", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260731, dest="rng_seed")
    parser.add_argument("--fit-steps", type=int, default=10)
    parser.add_argument(
        "--dynamics",
        choices=["gaussian_map", "microscopic"],
        default="gaussian_map",
        help="gaussian_map: 理論写像を直接検証、microscopic: 元の J・phi モデルを検証",
    )

    parser.add_argument("--modes", type=parse_modes, default=parse_modes("0,1,2,4,8"))
    parser.add_argument("--animate-mode", type=int, default=2)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--format", choices=["gif", "mp4"], default="gif")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "spatial_mode_ensemble_output",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = Config(
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
        rng_seed=args.rng_seed,
        fit_steps=args.fit_steps,
        dynamics=args.dynamics,
    )

    modes = args.modes
    validate_config(config, modes, args.animate_mode)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    m_star, roots = choose_fixed_point(config)
    Lambda = lambda_star(m_star, config)
    mu = mu_from_B(config)
    sig = sigma_eff(config)

    print("=== Model parameters ===")
    print(f"N={config.N}, R={config.R}, T={config.T}, M={config.ensemble}")
    print(f"dynamics={config.dynamics}")
    print(f"B={config.B:.6g}, mu={mu:.6g}, sigma_eff={sig:.6g}")
    print(f"fixed points={roots}")
    print(f"chosen m*={m_star:.8f}, Lambda*={Lambda:.8f}")

    results: list[ModeResult] = []
    for index, mode in enumerate(modes):
        print(f"simulate mode n={mode} ...")
        result = simulate_mode(
            mode_index=mode,
            m_star=m_star,
            Lambda=Lambda,
            config=config,
            seed_offset=100_000 * index,
        )
        results.append(result)
        save_mode_timeseries(result, output_dir)
        print(
            f"  q={result.q:.5f}, Khat={result.khat:.5f}, "
            f"lambda_th={result.lambda_theory:.5f}, "
            f"lambda_fit={result.lambda_fit:.5f}, R2={result.fit_r2:.4f}"
        )

    selected = next(r for r in results if r.mode_index == args.animate_mode)
    save_selected_spatiotemporal_data(selected, output_dir)

    lambda_df = make_lambda_summary(results, Lambda, output_dir)
    make_lambda_validation_figure(
        lambda_df,
        output_dir / "lambda_validation.png",
    )
    make_selected_static_figure(
        selected,
        config,
        m_star,
        output_dir / f"selected_mode_n{selected.mode_index}_validation.png",
    )

    animation_path = output_dir / f"spatial_mode_n{selected.mode_index}.{args.format}"
    make_animation(selected, config, m_star, animation_path, args.fps)

    metadata = {
        "config": asdict(config),
        "modes": modes,
        "animate_mode": args.animate_mode,
        "mu": mu,
        "sigma_eff": sig,
        "fixed_points": roots,
        "chosen_fixed_point": m_star,
        "Lambda_star": Lambda,
        "dynamics": config.dynamics,
        "definitions": {
            "coherent_response_spectrum": "FFT((ensemble_mean_plus-ensemble_mean_minus)/2) / N",
            "direct_coherent_plus_spectrum": "FFT(ensemble_mean_plus-m_star) / N",
            "cosine_amplitude": "2 Re(Abar_q), except n=0/Nyquist where factor=1",
            "structure_factor": "mean(|FFT(S_alpha - mean_i S_alpha)|^2) / N",
        },
    }
    with open(output_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("=== Outputs ===")
    for path in sorted(output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
