#!/usr/bin/env python3
"""MPI-free microscopic simulation core for Spinodal Phase5.

direct_J preserves the existing microscopic update literally: one independent
annealed Gaussian coupling is generated for every directed neighbour and is
shared by the paired plus/minus simulations.

aggregated_exact analytically integrates those Gaussian couplings conditional
on the current pair of spin configurations. It is exact for the existing
Gaussian-J model, not a central-limit approximation. The joint plus/minus
noise is reconstructed from the local spin overlap.

Random streams are keyed by stable work-unit identifiers and use NumPy Philox,
following Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3"
(SC11, 2011).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import resource
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_VERSION = "2026.08.16-phase5-v3-followup"
CHECKPOINT_SCHEMA_VERSION = 2
MICROSCOPIC_KERNELS = ("aggregated_exact", "direct_J")
INITIALIZATION_MODES = ("prepared_metastable", "bernoulli_meanfield")


@dataclass(frozen=True)
class Phase5Task:
    task_id: str
    task_group: str
    delta_index: int
    epsilon_index: int
    delta: float
    Delta: float
    m_star: float
    m_spinodal: float
    Gamma_closure: float
    N: int
    R: int
    lattice_spacing: float
    mode_index: int
    epsilon_fraction: float
    M_total: int
    block_size: int
    T: int
    fit_start: int
    fit_end: int
    mu: float
    sigma_J: float
    sigma_phi: float
    phi_bar: float
    branch: str
    microscopic_kernel: str = "aggregated_exact"
    initialization_mode: str = "prepared_metastable"
    preparation_width: float = 0.02
    preparation_steps: int = 6
    burn_steps_per_stage: int = 8
    float_dtype: str = "float64"
    base_seed: int = 20260815
    save_structure_factor: bool = False
    track_survival: bool = False
    unperturbed: bool = False


@dataclass(frozen=True)
class Phase5WorkUnit:
    unit_id: str
    task: Phase5Task
    block_id: int
    start_trial: int
    end_trial: int
    estimated_cost: int

    @property
    def block_n(self) -> int:
        return self.end_trial - self.start_trial


@dataclass
class Phase5BlockResult:
    unit_id: str
    task_id: str
    task_group: str
    block_id: int
    start_trial: int
    end_trial: int
    block_n: int
    delta: float
    Delta: float
    mode_index: int
    epsilon_fraction: float
    epsilon_target: float
    epsilon_achieved: float
    initial_amplitude: float
    q: float
    qR: float
    A_q: np.ndarray
    mean_m_plus: np.ndarray
    mean_m_minus: np.ndarray
    baseline_m: np.ndarray
    escape_fraction: np.ndarray
    preparation_magnetization: np.ndarray
    structure_factor: np.ndarray
    escape_fraction_cumulative: np.ndarray
    survival_fraction_cumulative: np.ndarray
    survival_count: np.ndarray
    survivor_amplitude_sum_current: np.ndarray
    A_q_surviving_current: np.ndarray
    survive_to_T_amplitude_sum: np.ndarray
    A_q_survive_to_T: np.ndarray
    baseline_m_surviving_current: np.ndarray
    survive_to_T_count: int
    checkpoint_schema_version: int
    survival_tracking_enabled: bool
    threshold_checksum: str
    rng_identifier: dict[str, Any]
    microscopic_kernel: str
    initialization_mode: str
    task_fingerprint: str
    wall_seconds: float


def validate_task(task: Phase5Task) -> None:
    if task.microscopic_kernel not in MICROSCOPIC_KERNELS:
        raise ValueError(f"unknown microscopic kernel: {task.microscopic_kernel}")
    if task.initialization_mode not in INITIALIZATION_MODES:
        raise ValueError(f"unknown initialization mode: {task.initialization_mode}")
    if task.N < 8 or task.R < 1 or 2 * task.R >= task.N:
        raise ValueError("Phase5 requires N>=8 and 1<=R<N/2")
    if task.M_total < 1 or task.block_size < 1:
        raise ValueError("M_total and block_size must be positive")
    if task.T < 2 or not (0 <= task.fit_start < task.fit_end <= task.T):
        raise ValueError("fit window must satisfy 0 <= start < end <= T")
    if task.mode_index < 0 or task.mode_index > task.N // 2:
        raise ValueError("mode_index must lie in [0,N/2]")
    if task.epsilon_fraction < 0.0 or (
        task.epsilon_fraction == 0.0 and not task.unperturbed
    ):
        raise ValueError(
            "epsilon_fraction must be positive unless unperturbed=True"
        )
    if task.sigma_J < 0.0 or task.sigma_phi < 0.0:
        raise ValueError("noise scales must be non-negative")
    if task.float_dtype not in {"float64", "float32"}:
        raise ValueError("float_dtype must be float64 or float32")
    if task.preparation_steps < 1 or task.burn_steps_per_stage < 0:
        raise ValueError("invalid preparation protocol")


def build_work_units(task: Phase5Task) -> list[Phase5WorkUnit]:
    """Split stable trial IDs into rank-independent checkpoint blocks."""
    validate_task(task)
    prep = (
        task.preparation_steps * task.burn_steps_per_stage
        if task.initialization_mode == "prepared_metastable"
        else 0
    )
    units = []
    for block_id, start in enumerate(range(0, task.M_total, task.block_size)):
        end = min(start + task.block_size, task.M_total)
        block_n = end - start
        unit_id = f"{task.task_id}_b{block_id:04d}"
        units.append(
            Phase5WorkUnit(
                unit_id=unit_id,
                task=task,
                block_id=block_id,
                start_trial=start,
                end_trial=end,
                estimated_cost=int(block_n * task.N * (prep + task.T)),
            )
        )
    return units


def make_work_unit_rng(work_unit: Phase5WorkUnit) -> np.random.Generator:
    """Return a Philox stream keyed only by stable physical work-unit IDs."""
    task = work_unit.task
    entropy = [
        int(task.base_seed),
        int(task.delta_index),
        int(task.mode_index),
        int(task.epsilon_index),
        int(work_unit.block_id),
    ]
    return np.random.Generator(np.random.Philox(np.random.SeedSequence(entropy)))


def rng_identifier(work_unit: Phase5WorkUnit) -> dict[str, Any]:
    task = work_unit.task
    return {
        "generator": "numpy.random.Philox",
        "base_seed": int(task.base_seed),
        "seed_sequence_entropy": [
            int(task.base_seed),
            int(task.delta_index),
            int(task.mode_index),
            int(task.epsilon_index),
            int(work_unit.block_id),
        ],
        "delta_index": int(task.delta_index),
        "mode_index": int(task.mode_index),
        "epsilon_index": int(task.epsilon_index),
        "block_id": int(work_unit.block_id),
        "start_trial": int(work_unit.start_trial),
        "end_trial": int(work_unit.end_trial),
        "mpi_rank_used_in_seed": False,
    }


def task_fingerprint(task: Phase5Task) -> str:
    """Hash every simulation-setting field used to validate resume files."""
    serialized = json.dumps(
        asdict(task), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def legacy_task_fingerprint(task: Phase5Task) -> str:
    """Return the pre-follow-up fingerprint used by schema-v1 checkpoints."""
    payload = asdict(task)
    payload.pop("track_survival", None)
    payload.pop("unperturbed", None)
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def periodic_neighbor_sum_batch(states: np.ndarray, R: int) -> np.ndarray:
    """Return the 2R-neighbour periodic sum in O(block*N), excluding self."""
    array = np.asarray(states)
    if array.ndim != 2:
        raise ValueError("states must have shape (block,N)")
    block_n, N = array.shape
    if block_n < 1 or R < 1 or 2 * R >= N:
        raise ValueError("periodic neighbour sum requires block>=1 and 1<=R<N/2")
    extended = np.concatenate((array[:, -R:], array, array[:, :R]), axis=1)
    cumulative = np.concatenate(
        (
            np.zeros((block_n, 1), dtype=np.int64),
            np.cumsum(extended, axis=1, dtype=np.int64),
        ),
        axis=1,
    )
    window = cumulative[:, 2 * R + 1 : 2 * R + 1 + N] - cumulative[:, :N]
    return window - array


def periodic_neighbor_sum_roll_reference(states: np.ndarray, R: int) -> np.ndarray:
    array = np.asarray(states)
    result = np.zeros_like(array, dtype=np.int64)
    for distance in range(1, R + 1):
        result += np.roll(array, shift=-distance, axis=1)
        result += np.roll(array, shift=distance, axis=1)
    return result


def direct_J_interactions(
    states_plus: np.ndarray,
    states_minus: np.ndarray,
    *,
    R: int,
    mu: float,
    sigma_J: float,
    rng: np.random.Generator,
    float_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Existing explicit-J interaction kernel, preserved as reference."""
    plus = np.asarray(states_plus, dtype=np.int8)
    minus = np.asarray(states_minus, dtype=np.int8)
    if plus.shape != minus.shape or plus.ndim != 2:
        raise ValueError("paired states must have equal (block,N) shape")
    interaction_plus = np.zeros(plus.shape, dtype=float_dtype)
    interaction_minus = np.zeros(minus.shape, dtype=float_dtype)
    for distance in range(1, R + 1):
        for signed_distance in (-distance, distance):
            z = rng.standard_normal(plus.shape, dtype=float_dtype)
            coupling = mu + sigma_J * z
            interaction_plus += coupling * np.roll(
                plus, shift=-signed_distance, axis=1
            )
            interaction_minus += coupling * np.roll(
                minus, shift=-signed_distance, axis=1
            )
    interaction_plus /= 2 * R
    interaction_minus /= 2 * R
    return interaction_plus, interaction_minus


def aggregated_exact_interactions(
    states_plus: np.ndarray,
    states_minus: np.ndarray,
    *,
    R: int,
    mu: float,
    sigma_J: float,
    rng: np.random.Generator,
    float_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact conditional bivariate Gaussian interaction kernel."""
    plus = np.asarray(states_plus, dtype=np.int8)
    minus = np.asarray(states_minus, dtype=np.int8)
    if plus.shape != minus.shape or plus.ndim != 2:
        raise ValueError("paired states must have equal (block,N) shape")
    c = 2 * R
    sum_plus = periodic_neighbor_sum_batch(plus, R).astype(float_dtype)
    sum_minus = periodic_neighbor_sum_batch(minus, R).astype(float_dtype)
    overlap = periodic_neighbor_sum_batch(plus * minus, R).astype(float_dtype)
    m_plus = sum_plus / c
    m_minus = sum_minus / c
    rho = np.clip(overlap / c, -1.0, 1.0)
    z1 = rng.standard_normal(plus.shape, dtype=float_dtype)
    z2 = rng.standard_normal(plus.shape, dtype=float_dtype)
    scale = sigma_J / math.sqrt(c)
    noise_plus = scale * z1
    noise_minus = scale * (
        rho * z1 + np.sqrt(np.maximum(1.0 - rho * rho, 0.0)) * z2
    )
    return mu * m_plus + noise_plus, mu * m_minus + noise_minus


def direct_J_step_reference(
    states_plus: np.ndarray,
    states_minus: np.ndarray,
    thresholds: np.ndarray,
    *,
    R: int,
    mu: float,
    sigma_J: float,
    h: float,
    rng: np.random.Generator,
    float_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    interaction_plus, interaction_minus = direct_J_interactions(
        states_plus,
        states_minus,
        R=R,
        mu=mu,
        sigma_J=sigma_J,
        rng=rng,
        float_dtype=float_dtype,
    )
    return (
        np.where(interaction_plus - thresholds + h >= 0.0, 1, -1).astype(np.int8),
        np.where(interaction_minus - thresholds + h >= 0.0, 1, -1).astype(np.int8),
    )


def aggregated_exact_step(
    states_plus: np.ndarray,
    states_minus: np.ndarray,
    thresholds: np.ndarray,
    *,
    R: int,
    mu: float,
    sigma_J: float,
    h: float,
    rng: np.random.Generator,
    float_dtype: np.dtype | type = np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    interaction_plus, interaction_minus = aggregated_exact_interactions(
        states_plus,
        states_minus,
        R=R,
        mu=mu,
        sigma_J=sigma_J,
        rng=rng,
        float_dtype=float_dtype,
    )
    return (
        np.where(interaction_plus - thresholds + h >= 0.0, 1, -1).astype(np.int8),
        np.where(interaction_minus - thresholds + h >= 0.0, 1, -1).astype(np.int8),
    )


def _paired_step(
    kernel: str,
    states_plus: np.ndarray,
    states_minus: np.ndarray,
    thresholds: np.ndarray,
    *,
    task: Phase5Task,
    h: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    dtype = np.float64 if task.float_dtype == "float64" else np.float32
    kwargs = {
        "R": task.R,
        "mu": task.mu,
        "sigma_J": task.sigma_J,
        "h": h,
        "rng": rng,
        "float_dtype": dtype,
    }
    if kernel == "direct_J":
        return direct_J_step_reference(states_plus, states_minus, thresholds, **kwargs)
    if kernel == "aggregated_exact":
        return aggregated_exact_step(states_plus, states_minus, thresholds, **kwargs)
    raise ValueError(f"unknown kernel: {kernel}")


def prepare_metastable_block(
    block_n: int,
    thresholds: np.ndarray,
    task: Phase5Task,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Prepare a stay-side metastable background using a recorded Delta ramp."""
    if block_n != thresholds.shape[0]:
        raise ValueError("threshold block size mismatch")
    if task.branch == "stay_to_evacuate":
        states = np.full((block_n, task.N), -1, dtype=np.int8)
        Delta_start = task.Delta - task.preparation_width
    else:
        states = np.full((block_n, task.N), 1, dtype=np.int8)
        Delta_start = task.Delta + task.preparation_width
    magnetization = [float(np.mean(states))]
    deltas = np.linspace(Delta_start, task.Delta, task.preparation_steps)
    for Delta_stage in deltas:
        h_stage = task.phi_bar + float(Delta_stage)
        paired = states.copy()
        for _ in range(task.burn_steps_per_stage):
            states, paired = _paired_step(
                task.microscopic_kernel,
                states,
                paired,
                thresholds,
                task=task,
                h=h_stage,
                rng=rng,
            )
        if not np.array_equal(states, paired):
            raise RuntimeError("identical preparation pair diverged under common noise")
        magnetization.append(float(np.mean(states)))
    return states, np.asarray(magnetization, dtype=float)


def _apply_mean_shift(
    base_states: np.ndarray,
    desired_shift: np.ndarray,
    common_uniform: np.ndarray,
) -> np.ndarray:
    site_mean = base_states.mean(axis=0, dtype=float)
    positive_probability = np.divide(
        desired_shift,
        np.maximum(1.0 - site_mean, np.finfo(float).eps),
        out=np.zeros_like(desired_shift, dtype=float),
        where=desired_shift > 0.0,
    )
    negative_probability = np.divide(
        -desired_shift,
        np.maximum(1.0 + site_mean, np.finfo(float).eps),
        out=np.zeros_like(desired_shift, dtype=float),
        where=desired_shift < 0.0,
    )
    positive_probability = np.clip(positive_probability, 0.0, 1.0)
    negative_probability = np.clip(negative_probability, 0.0, 1.0)
    result = base_states.copy()
    flip_up = (base_states == -1) & (
        common_uniform < positive_probability[None, :]
    )
    flip_down = (base_states == 1) & (
        common_uniform < negative_probability[None, :]
    )
    result[flip_up] = 1
    result[flip_down] = -1
    return result


def cosine_amplitude(response: np.ndarray, mode_index: int, a: float) -> float:
    response = np.asarray(response, dtype=float)
    N = len(response)
    q = 2.0 * math.pi * mode_index / (N * a)
    factor = 1.0 if mode_index == 0 or (N % 2 == 0 and mode_index == N // 2) else 2.0
    return float(
        factor
        * np.dot(response, np.cos(q * np.arange(N, dtype=float) * a))
        / N
    )


def cosine_amplitude_batch(
    responses: np.ndarray, mode_index: int, a: float
) -> np.ndarray:
    """Return one cosine-mode amplitude per trial."""
    array = np.asarray(responses, dtype=float)
    if array.ndim != 2:
        raise ValueError("responses must have shape (trials,N)")
    N = array.shape[1]
    q = 2.0 * math.pi * mode_index / (N * a)
    factor = (
        1.0
        if mode_index == 0 or (N % 2 == 0 and mode_index == N // 2)
        else 2.0
    )
    cosine = np.cos(q * np.arange(N, dtype=float) * a)
    return factor * (array @ cosine) / N


def apply_paired_mode_perturbation(
    base_states: np.ndarray,
    *,
    mode_index: int,
    epsilon: float,
    lattice_spacing: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Perturb a prepared binary background while retaining its correlations."""
    block_n, N = base_states.shape
    q = 2.0 * math.pi * mode_index / (N * lattice_spacing)
    desired = epsilon * np.cos(q * np.arange(N) * lattice_spacing)
    common_uniform = rng.random((block_n, N))
    plus = _apply_mean_shift(base_states, desired, common_uniform)
    minus = _apply_mean_shift(base_states, -desired, common_uniform)
    response = 0.5 * (plus.mean(axis=0) - minus.mean(axis=0))
    return plus, minus, cosine_amplitude(response, mode_index, lattice_spacing)


def bernoulli_meanfield_initialization(
    block_n: int,
    task: Phase5Task,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, float]:
    x = np.arange(task.N, dtype=float) * task.lattice_spacing
    q = 2.0 * math.pi * task.mode_index / (task.N * task.lattice_spacing)
    perturbation = epsilon * np.cos(q * x)
    u_plus = task.m_star + perturbation
    u_minus = task.m_star - perturbation
    if np.min(u_minus) < -1.0 or np.max(u_plus) > 1.0:
        raise ValueError("Bernoulli initialization leaves the physical mean range")
    common_uniform = rng.random((block_n, task.N))
    plus = np.where(common_uniform < 0.5 * (1.0 + u_plus), 1, -1).astype(np.int8)
    minus = np.where(common_uniform < 0.5 * (1.0 + u_minus), 1, -1).astype(np.int8)
    response = 0.5 * (plus.mean(axis=0) - minus.mean(axis=0))
    return plus, minus, cosine_amplitude(
        response, task.mode_index, task.lattice_spacing
    )


def _threshold_checksum(thresholds: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(thresholds).view(np.uint8)).hexdigest()


def _escape_mask(
    states_plus: np.ndarray, states_minus: np.ndarray, task: Phase5Task
) -> np.ndarray:
    """Return the instantaneous per-trial Gaussian-threshold escape mask."""
    pair_m = 0.5 * (
        states_plus.mean(axis=1, dtype=float)
        + states_minus.mean(axis=1, dtype=float)
    )
    if task.branch == "stay_to_evacuate":
        return pair_m > task.m_spinodal
    return pair_m < task.m_spinodal


def _escape_fraction(
    states_plus: np.ndarray, states_minus: np.ndarray, task: Phase5Task
) -> float:
    """Preserve the original instantaneous escape-fraction definition."""
    return float(np.mean(_escape_mask(states_plus, states_minus, task)))


def simulate_microscopic_block(work_unit: Phase5WorkUnit) -> Phase5BlockResult:
    """Run one independent block, retaining only online sufficient statistics."""
    task = work_unit.task
    validate_task(task)
    start = time.perf_counter()
    rng = make_work_unit_rng(work_unit)
    dtype = np.float64 if task.float_dtype == "float64" else np.float32
    thresholds = rng.normal(
        loc=task.phi_bar,
        scale=task.sigma_phi,
        size=(work_unit.block_n, task.N),
    ).astype(dtype, copy=False)
    threshold_checksum = _threshold_checksum(thresholds)
    epsilon = (
        0.0
        if task.unperturbed
        else task.epsilon_fraction * abs(task.m_star - task.m_spinodal)
    )

    if task.initialization_mode == "prepared_metastable":
        base, preparation_m = prepare_metastable_block(
            work_unit.block_n, thresholds, task, rng
        )
        states_plus, states_minus, epsilon_achieved = apply_paired_mode_perturbation(
            base,
            mode_index=task.mode_index,
            epsilon=epsilon,
            lattice_spacing=task.lattice_spacing,
            rng=rng,
        )
    else:
        preparation_m = np.empty(0, dtype=float)
        states_plus, states_minus, epsilon_achieved = bernoulli_meanfield_initialization(
            work_unit.block_n, task, epsilon, rng
        )

    A_q = np.empty(task.T + 1, dtype=float)
    mean_plus = np.empty(task.T + 1, dtype=float)
    mean_minus = np.empty(task.T + 1, dtype=float)
    baseline = np.empty(task.T + 1, dtype=float)
    escape = np.empty(task.T + 1, dtype=float)
    structure_factor = (
        np.empty((task.T + 1, task.N // 2 + 1), dtype=float)
        if task.save_structure_factor
        else np.empty((0, 0), dtype=float)
    )
    empty = np.empty(0, dtype=float)
    escape_cumulative = np.empty(task.T + 1, dtype=float) if task.track_survival else empty.copy()
    survival_fraction = np.empty(task.T + 1, dtype=float) if task.track_survival else empty.copy()
    survival_count = np.empty(task.T + 1, dtype=np.int64) if task.track_survival else np.empty(0, dtype=np.int64)
    survivor_sum_current = np.empty(task.T + 1, dtype=float) if task.track_survival else empty.copy()
    A_surviving_current = np.empty(task.T + 1, dtype=float) if task.track_survival else empty.copy()
    baseline_surviving_current = np.empty(task.T + 1, dtype=float) if task.track_survival else empty.copy()
    trial_amplitude_history = (
        np.empty((work_unit.block_n, task.T + 1), dtype=float)
        if task.track_survival
        else np.empty((0, 0), dtype=float)
    )
    alive = np.ones(work_unit.block_n, dtype=bool)

    def record(index: int) -> None:
        plus_profile = states_plus.mean(axis=0, dtype=float)
        minus_profile = states_minus.mean(axis=0, dtype=float)
        response = 0.5 * (plus_profile - minus_profile)
        A_q[index] = cosine_amplitude(response, task.mode_index, task.lattice_spacing)
        mean_plus[index] = float(np.mean(plus_profile))
        mean_minus[index] = float(np.mean(minus_profile))
        baseline[index] = 0.5 * (mean_plus[index] + mean_minus[index])
        escape[index] = _escape_fraction(states_plus, states_minus, task)
        if task.track_survival:
            escaped_now = _escape_mask(states_plus, states_minus, task)
            alive[:] = alive & ~escaped_now
            trial_response = 0.5 * (
                states_plus.astype(float) - states_minus.astype(float)
            )
            trial_amplitudes = cosine_amplitude_batch(
                trial_response, task.mode_index, task.lattice_spacing
            )
            trial_amplitude_history[:, index] = trial_amplitudes
            trial_baseline = 0.5 * (
                states_plus.mean(axis=1, dtype=float)
                + states_minus.mean(axis=1, dtype=float)
            )
            count = int(np.sum(alive))
            survival_count[index] = count
            survival_fraction[index] = count / work_unit.block_n
            escape_cumulative[index] = 1.0 - survival_fraction[index]
            survivor_sum_current[index] = float(np.sum(trial_amplitudes[alive]))
            A_surviving_current[index] = (
                survivor_sum_current[index] / count if count else math.nan
            )
            baseline_surviving_current[index] = (
                float(np.sum(trial_baseline[alive])) / count if count else math.nan
            )
        if task.save_structure_factor:
            centered = states_plus - states_plus.mean(axis=1, keepdims=True)
            spectrum = np.fft.rfft(centered, axis=1)
            structure_factor[index] = (
                np.mean(np.abs(spectrum) ** 2, axis=0) / task.N
            )

    record(0)
    h_target = task.phi_bar + task.Delta
    for index in range(task.T):
        states_plus, states_minus = _paired_step(
            task.microscopic_kernel,
            states_plus,
            states_minus,
            thresholds,
            task=task,
            h=h_target,
            rng=rng,
        )
        record(index + 1)
    if threshold_checksum != _threshold_checksum(thresholds):
        raise RuntimeError("quenched thresholds changed during simulation")

    if task.track_survival:
        survive_to_T_count = int(np.sum(alive))
        survive_to_T_sum = np.sum(
            trial_amplitude_history[alive], axis=0, dtype=float
        )
        A_survive_to_T = (
            survive_to_T_sum / survive_to_T_count
            if survive_to_T_count
            else np.full(task.T + 1, math.nan, dtype=float)
        )
    else:
        survive_to_T_count = 0
        survive_to_T_sum = empty.copy()
        A_survive_to_T = empty.copy()

    q = 2.0 * math.pi * task.mode_index / (task.N * task.lattice_spacing)
    return Phase5BlockResult(
        unit_id=work_unit.unit_id,
        task_id=task.task_id,
        task_group=task.task_group,
        block_id=work_unit.block_id,
        start_trial=work_unit.start_trial,
        end_trial=work_unit.end_trial,
        block_n=work_unit.block_n,
        delta=task.delta,
        Delta=task.Delta,
        mode_index=task.mode_index,
        epsilon_fraction=task.epsilon_fraction,
        epsilon_target=epsilon,
        epsilon_achieved=epsilon_achieved,
        initial_amplitude=float(A_q[0]),
        q=q,
        qR=q * task.R * task.lattice_spacing,
        A_q=A_q,
        mean_m_plus=mean_plus,
        mean_m_minus=mean_minus,
        baseline_m=baseline,
        escape_fraction=escape,
        preparation_magnetization=preparation_m,
        structure_factor=structure_factor,
        escape_fraction_cumulative=escape_cumulative,
        survival_fraction_cumulative=survival_fraction,
        survival_count=survival_count,
        survivor_amplitude_sum_current=survivor_sum_current,
        A_q_surviving_current=A_surviving_current,
        survive_to_T_amplitude_sum=survive_to_T_sum,
        A_q_survive_to_T=A_survive_to_T,
        baseline_m_surviving_current=baseline_surviving_current,
        survive_to_T_count=survive_to_T_count,
        checkpoint_schema_version=CHECKPOINT_SCHEMA_VERSION,
        survival_tracking_enabled=task.track_survival,
        threshold_checksum=threshold_checksum,
        rng_identifier=rng_identifier(work_unit),
        microscopic_kernel=task.microscopic_kernel,
        initialization_mode=task.initialization_mode,
        task_fingerprint=task_fingerprint(task),
        wall_seconds=time.perf_counter() - start,
    )


def checkpoint_path(blocks_dir: Path, work_unit: Phase5WorkUnit) -> Path:
    return Path(blocks_dir) / f"{work_unit.unit_id}.npz"


def save_block_checkpoint(result: Phase5BlockResult, path: Path) -> None:
    """Atomically save one compact completed work unit."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    metadata = {
        key: value
        for key, value in asdict(result).items()
        if not isinstance(value, np.ndarray)
    }
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            A_q=result.A_q,
            mean_m_plus=result.mean_m_plus,
            mean_m_minus=result.mean_m_minus,
            baseline_m=result.baseline_m,
            escape_fraction=result.escape_fraction,
            preparation_magnetization=result.preparation_magnetization,
            structure_factor=result.structure_factor,
            escape_fraction_cumulative=result.escape_fraction_cumulative,
            survival_fraction_cumulative=result.survival_fraction_cumulative,
            survival_count=result.survival_count,
            survivor_amplitude_sum_current=result.survivor_amplitude_sum_current,
            A_q_surviving_current=result.A_q_surviving_current,
            survive_to_T_amplitude_sum=result.survive_to_T_amplitude_sum,
            A_q_survive_to_T=result.A_q_survive_to_T,
            baseline_m_surviving_current=result.baseline_m_surviving_current,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_block_checkpoint(
    path: Path, expected_unit: Phase5WorkUnit | None = None
) -> Phase5BlockResult:
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {
                "metadata_json",
                "A_q",
                "mean_m_plus",
                "mean_m_minus",
                "baseline_m",
                "escape_fraction",
                "preparation_magnetization",
                "structure_factor",
            }
            missing = required - set(archive.files)
            if missing:
                raise ValueError(f"checkpoint missing arrays: {sorted(missing)}")
            metadata = json.loads(str(archive["metadata_json"].item()))
            arrays = {
                name: np.asarray(archive[name]).copy()
                for name in required
                if name != "metadata_json"
            }
            optional = {
                "escape_fraction_cumulative",
                "survival_fraction_cumulative",
                "survival_count",
                "survivor_amplitude_sum_current",
                "A_q_surviving_current",
                "survive_to_T_amplitude_sum",
                "A_q_survive_to_T",
                "baseline_m_surviving_current",
            }
            for name in optional & set(archive.files):
                arrays[name] = np.asarray(archive[name]).copy()
    except Exception as exc:
        raise ValueError(f"corrupt Phase5 checkpoint {path}: {exc}") from exc
    if expected_unit is not None:
        if metadata.get("unit_id") != expected_unit.unit_id:
            raise ValueError(
                f"checkpoint unit mismatch: {metadata.get('unit_id')} != {expected_unit.unit_id}"
            )
        if int(metadata.get("block_n", -1)) != expected_unit.block_n:
            raise ValueError("checkpoint block size mismatch")
        if int(metadata.get("start_trial", -1)) != expected_unit.start_trial:
            raise ValueError("checkpoint start_trial mismatch")
        if int(metadata.get("end_trial", -1)) != expected_unit.end_trial:
            raise ValueError("checkpoint end_trial mismatch")
        schema_version = int(metadata.get("checkpoint_schema_version", 1))
        expected_fingerprint = (
            legacy_task_fingerprint(expected_unit.task)
            if schema_version == 1
            else task_fingerprint(expected_unit.task)
        )
        fingerprint_matches = metadata.get("task_fingerprint") == expected_fingerprint
        if not fingerprint_matches and schema_version == 1:
            # Schema-v1 hashes included platform-evaluated analytic floats.
            # SQUID and macOS libm can differ at the last bit, so transferred
            # legacy checkpoints are validated against all stored physical
            # metadata and array shapes instead of being declared corrupt.
            task = expected_unit.task
            expected_q = 2.0 * math.pi * task.mode_index / (
                task.N * task.lattice_spacing
            )
            scalar_pairs = (
                (metadata.get("delta"), task.delta),
                (metadata.get("Delta"), task.Delta),
                (metadata.get("epsilon_fraction"), task.epsilon_fraction),
                (metadata.get("q"), expected_q),
                (metadata.get("qR"), expected_q * task.R * task.lattice_spacing),
            )
            legacy_compatible = (
                metadata.get("task_id") == task.task_id
                and metadata.get("task_group") == task.task_group
                and int(metadata.get("mode_index", -1)) == task.mode_index
                and metadata.get("microscopic_kernel") == task.microscopic_kernel
                and metadata.get("initialization_mode") == task.initialization_mode
                and all(
                    value is not None
                    and math.isclose(
                        float(value), float(expected), rel_tol=1e-12, abs_tol=1e-15
                    )
                    for value, expected in scalar_pairs
                )
            )
            if not legacy_compatible:
                raise ValueError("checkpoint task configuration mismatch")
        elif not fingerprint_matches:
            raise ValueError("checkpoint task configuration mismatch")
        expected_time_shape = (expected_unit.task.T + 1,)
        for name in (
            "A_q",
            "mean_m_plus",
            "mean_m_minus",
            "baseline_m",
            "escape_fraction",
        ):
            if arrays[name].shape != expected_time_shape:
                raise ValueError(f"checkpoint {name} shape mismatch")
        if expected_unit.task.track_survival:
            survival_time_arrays = (
                "escape_fraction_cumulative",
                "survival_fraction_cumulative",
                "survival_count",
                "survivor_amplitude_sum_current",
                "A_q_surviving_current",
                "survive_to_T_amplitude_sum",
                "A_q_survive_to_T",
                "baseline_m_surviving_current",
            )
            if not bool(metadata.get("survival_tracking_enabled", False)):
                raise ValueError("checkpoint lacks requested survival tracking")
            for name in survival_time_arrays:
                if name not in arrays or arrays[name].shape != expected_time_shape:
                    raise ValueError(f"checkpoint {name} shape mismatch")
    core_array_names = {
        "A_q",
        "mean_m_plus",
        "mean_m_minus",
        "baseline_m",
        "escape_fraction",
        "preparation_magnetization",
        "structure_factor",
    }
    if not all(
        np.all(np.isfinite(arrays[name]))
        for name in core_array_names
        if name in arrays
    ):
        raise ValueError(f"checkpoint contains non-finite values: {path}")
    metadata.setdefault("checkpoint_schema_version", 1)
    metadata.setdefault("survival_tracking_enabled", False)
    metadata.setdefault("survive_to_T_count", 0)
    arrays.setdefault("escape_fraction_cumulative", np.empty(0, dtype=float))
    arrays.setdefault("survival_fraction_cumulative", np.empty(0, dtype=float))
    arrays.setdefault("survival_count", np.empty(0, dtype=np.int64))
    arrays.setdefault("survivor_amplitude_sum_current", np.empty(0, dtype=float))
    arrays.setdefault("A_q_surviving_current", np.empty(0, dtype=float))
    arrays.setdefault("survive_to_T_amplitude_sum", np.empty(0, dtype=float))
    arrays.setdefault("A_q_survive_to_T", np.empty(0, dtype=float))
    arrays.setdefault("baseline_m_surviving_current", np.empty(0, dtype=float))
    try:
        return Phase5BlockResult(**metadata, **arrays)
    except TypeError as exc:
        raise ValueError(f"checkpoint metadata schema mismatch: {path}") from exc


def checkpoint_is_valid(path: Path, work_unit: Phase5WorkUnit) -> bool:
    if not Path(path).is_file():
        return False
    try:
        load_block_checkpoint(path, work_unit)
    except ValueError:
        return False
    return True


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def benchmark_microscopic_kernels(
    *,
    N: int = 256,
    R: int = 12,
    block_size: int = 32,
    steps: int = 20,
    mu: float = 0.5,
    sigma_J: float = 1.0,
    sigma_phi: float = 0.06,
    h: float = 0.15,
    seed: int = 12345,
) -> dict[str, Any]:
    """Single-core kernel benchmark; performance assertions belong on SQUID."""
    setup_rng = np.random.Generator(np.random.Philox(seed))
    initial_plus = np.where(
        setup_rng.random((block_size, N)) < 0.2, 1, -1
    ).astype(np.int8)
    initial_minus = np.where(
        setup_rng.random((block_size, N)) < 0.2, 1, -1
    ).astype(np.int8)
    thresholds = setup_rng.normal(0.0, sigma_phi, size=(block_size, N))
    results: dict[str, Any] = {}
    for kernel in ("direct_J", "aggregated_exact"):
        plus = initial_plus.copy()
        minus = initial_minus.copy()
        rng = np.random.Generator(
            np.random.Philox(seed + (1 if kernel == "direct_J" else 2))
        )
        start = time.perf_counter()
        for _ in range(steps):
            if kernel == "direct_J":
                plus, minus = direct_J_step_reference(
                    plus, minus, thresholds, R=R, mu=mu, sigma_J=sigma_J, h=h, rng=rng
                )
            else:
                plus, minus = aggregated_exact_step(
                    plus, minus, thresholds, R=R, mu=mu, sigma_J=sigma_J, h=h, rng=rng
                )
        seconds = time.perf_counter() - start
        trial_site_steps = block_size * N * steps
        results[kernel] = {
            "seconds": seconds,
            "trial_site_steps_per_sec": trial_site_steps / seconds,
        }
    results["speedup"] = results["direct_J"]["seconds"] / results["aggregated_exact"]["seconds"]
    results["peak_rss_mb"] = peak_rss_mb()
    results["parameters"] = {
        "N": N,
        "R": R,
        "block_size": block_size,
        "steps": steps,
        "float_dtype": "float64",
    }
    results["performance_warning"] = (
        "aggregated_exact was slower in this benchmark; measure on SQUID"
        if results["speedup"] < 1.0
        else None
    )
    return results
