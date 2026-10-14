"""Build PPT-ready tables and figures from the three-mode GPU benchmark."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import (B_DAMP, D_AMP, D_FREQ, DOB_ALPHA, DT, HORIZON, LAMBDA,
                    MPPI_MEAN_UPDATE_ALPHA, MPPI_W_POS, MPPI_W_TERM,
                    NOISE_VAR, N_SAMPLES, TAU_MAX, TRAJ_CENTER, TRAJ_RADIUS,
                    W_ACT, W_SMOOTH, W_VEL)
from experiment_runner import waypoints


SOURCE_DIR = Path("results/model_only_rollout_gpu/c1477a74e82b")
OUTPUT_DIR = SOURCE_DIR / "ppt_summary"
PROFILES = ("constant", "step", "slow_sinusoidal", "fast_sinusoidal", "large_step")
MODES = ("post_dob", "nominal_shift", "model_only")
MODE_LABELS = {
    "post_dob": "Post-DOB",
    "nominal_shift": "Rollout Nominal Shift",
    "model_only": "Model-only Rollout",
}
MODE_COLORS = {
    "post_dob": "#1f77b4",
    "nominal_shift": "#ff7f0e",
    "model_only": "#2ca02c",
}
SEEDS = (7, 17, 27)
STEPS = 250
DEADLINE_MS = DT * 1000.0


METRIC_FIELDS = (
    "ee_rmse_mm",
    "ee_mean_mm",
    "ee_max_mm",
    "joint_rmse_rad",
    "applied_torque_peak_nm",
    "torque_limit_reached_steps",
    "torque_limit_reached_ratio_pct",
    "torque_rate_rms_nm_per_step",
    "solve_mean_ms",
    "solve_p95_ms",
    "solve_max_ms",
    "deadline_miss_count",
    "deadline_miss_ratio_pct",
    "dob_error_rms_per_joint_nm",
    "dob_error_vector_rms_nm",
    "dob_error_vector_max_nm",
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()


def _scalar(array) -> object:
    value = np.asarray(array)
    return value.item() if value.ndim == 0 else value


def _prefix(profile: str, mode: str, seed: int) -> str:
    return f"{profile}__{mode}__seed_{seed}__"


def _run_data(archive, profile: str, mode: str, seed: int) -> dict[str, np.ndarray]:
    prefix = _prefix(profile, mode, seed)
    return {
        name[len(prefix):]: archive[name]
        for name in archive.files if name.startswith(prefix)
    }


def _metrics(data: dict[str, np.ndarray]) -> dict[str, float | int | str]:
    error = np.asarray(data["err_mm"], dtype=float)
    q = np.asarray(data["q"], dtype=float)[1:]
    q_reference = np.asarray(data["q_reference"], dtype=float)[1:len(q) + 1]
    joint_error = q - q_reference
    torque = np.asarray(data["u_applied"], dtype=float)
    torque_delta = np.diff(torque, axis=0)
    saturated = np.asarray(data["actuator_saturated"], dtype=bool)
    solve_ms = np.asarray(data["solve_t"], dtype=float) * 1000.0
    solve_ms = solve_ms[np.isfinite(solve_ms)]
    d_tilde = np.asarray(data["d_tilde"], dtype=float)
    d_error_norm = np.linalg.norm(d_tilde, axis=1)
    deadline_misses = int(np.sum(solve_ms > DEADLINE_MS))
    return {
        "ee_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "ee_mean_mm": float(np.mean(error)),
        "ee_max_mm": float(np.max(error)),
        "joint_rmse_rad": float(np.sqrt(np.mean(joint_error ** 2))),
        "applied_torque_peak_nm": float(np.max(np.abs(torque))),
        "torque_limit_reached_steps": int(np.sum(saturated)),
        "torque_limit_reached_ratio_pct": float(100.0 * np.mean(saturated)),
        "torque_rate_rms_nm_per_step": float(np.sqrt(np.mean(
            np.sum(torque_delta ** 2, axis=1)
        ))),
        "solve_mean_ms": float(np.mean(solve_ms)),
        "solve_p95_ms": float(np.percentile(solve_ms, 95)),
        "solve_max_ms": float(np.max(solve_ms)),
        "deadline_miss_count": deadline_misses,
        "deadline_miss_ratio_pct": float(100.0 * deadline_misses / len(solve_ms)),
        "dob_error_rms_per_joint_nm": float(np.sqrt(np.mean(d_tilde ** 2))),
        "dob_error_vector_rms_nm": float(np.sqrt(np.mean(d_error_norm ** 2))),
        "dob_error_vector_max_nm": float(np.max(d_error_norm)),
        "execution_failure_count": 0,
        "fallback_count": "미측정",
        "joint_position_violation_count": "미측정",
        "joint_velocity_violation_count": "미측정",
        "workspace_violation_count": "미측정",
        "self_collision_count": "미측정",
    }


def _metric_rows(archive) -> tuple[list[dict], dict[tuple[str, str, int], dict]]:
    rows: list[dict] = []
    runs: dict[tuple[str, str, int], dict] = {}
    for profile in PROFILES:
        for seed in SEEDS:
            for mode in MODES:
                data = _run_data(archive, profile, mode, seed)
                metrics = _metrics(data)
                runs[(profile, mode, seed)] = data
                rows.append({
                    "record_type": "seed",
                    "disturbance_profile": profile,
                    "seed": seed,
                    "mode": mode,
                    "mode_label": MODE_LABELS[mode],
                    **metrics,
                    **{f"{field}_std": "" for field in METRIC_FIELDS},
                })
    for profile in PROFILES:
        for mode in MODES:
            selected = [row for row in rows
                        if row["disturbance_profile"] == profile
                        and row["mode"] == mode]
            aggregate = {
                "record_type": "aggregate",
                "disturbance_profile": profile,
                "seed": "mean±std (n=3)",
                "mode": mode,
                "mode_label": MODE_LABELS[mode],
                "execution_failure_count": 0,
                "fallback_count": "미측정",
                "joint_position_violation_count": "미측정",
                "joint_velocity_violation_count": "미측정",
                "workspace_violation_count": "미측정",
                "self_collision_count": "미측정",
            }
            for field in METRIC_FIELDS:
                values = np.asarray([row[field] for row in selected], dtype=float)
                aggregate[field] = float(np.mean(values))
                aggregate[f"{field}_std"] = float(np.std(values, ddof=1))
            rows.append(aggregate)
    return rows, runs


def _write_metrics(rows: list[dict]) -> None:
    fieldnames = list(rows[0])
    with (OUTPUT_DIR / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _time_history_rows(runs: dict[tuple[str, str, int], dict]):
    reference = waypoints("circle", n_steps=STEPS)
    reference_at_result = np.vstack((reference[1:], reference[-1:]))
    for profile in PROFILES:
        for seed in SEEDS:
            for mode in MODES:
                data = runs[(profile, mode, seed)]
                q = np.asarray(data["q"])[1:]
                q_reference = np.asarray(data["q_reference"])[1:STEPS + 1]
                ee = np.asarray(data["ee"])[1:]
                d_true = np.asarray(data["d_true"])[1:]
                d_hat_used = np.asarray(data["d_hat"])[0:STEPS]
                d_hat_updated = np.asarray(data["d_hat"])[1:]
                u_ff = np.asarray(data["u_ff"])
                u_center = np.asarray(data["u_sampling_center"])
                u_mppi = np.asarray(data["u_mppi"])
                u_applied = np.asarray(data["u_applied"])
                residual = np.asarray(data["r_star"])
                solve_ms = np.asarray(data["solve_t"]) * 1000.0
                error = np.asarray(data["err_mm"])
                saturated = np.asarray(data["actuator_saturated"], dtype=bool)
                for step in range(STEPS):
                    row = {
                        "disturbance_profile": profile,
                        "seed": seed,
                        "mode": mode,
                        "step": step,
                        "time_s": step * DT,
                        "ee_error_mm": error[step],
                        "solve_ms": solve_ms[step],
                        "deadline_miss": bool(np.isfinite(solve_ms[step])
                                              and solve_ms[step] > DEADLINE_MS),
                        "saturation": bool(saturated[step]),
                    }
                    for joint in range(6):
                        index = joint + 1
                        row[f"q{index}_rad"] = q[step, joint]
                        row[f"q{index}_ref_rad"] = q_reference[step, joint]
                        row[f"d{index}_nm"] = d_true[step, joint]
                        row[f"d_hat_used{index}_nm"] = d_hat_used[step, joint]
                        row[f"d_hat_updated{index}_nm"] = d_hat_updated[step, joint]
                        row[f"u_ff{index}_nm"] = u_ff[step, joint]
                        row[f"u_center{index}_nm"] = u_center[step, joint]
                        row[f"u_mppi{index}_nm"] = u_mppi[step, joint]
                        row[f"dob_output_comp{index}_nm"] = (
                            u_applied[step, joint] - u_mppi[step, joint]
                        )
                        row[f"u_applied{index}_nm"] = u_applied[step, joint]
                        row[f"residual{index}_nm"] = residual[step, joint]
                    for axis, name in enumerate("xyz"):
                        row[f"ee_{name}_m"] = ee[step, axis]
                        row[f"ee_ref_{name}_m"] = reference_at_result[step, axis]
                    yield row


def _write_time_history(runs: dict[tuple[str, str, int], dict]) -> None:
    iterator = iter(_time_history_rows(runs))
    first = next(iterator)
    with (OUTPUT_DIR / "time_history.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(iterator)


def _segment_rows(runs: dict[tuple[str, str, int], dict]) -> list[dict]:
    rows = []
    for profile in PROFILES:
        for seed in SEEDS:
            post = runs[(profile, "post_dob", seed)]
            nominal = runs[(profile, "nominal_shift", seed)]
            saturation = (
                np.asarray(post["actuator_saturated"], dtype=bool)
                | np.asarray(nominal["actuator_saturated"], dtype=bool)
            )
            for segment, mask in (("non_saturated", ~saturation),
                                  ("saturated", saturation)):
                if not np.any(mask):
                    rows.append({
                        "disturbance_profile": profile,
                        "seed": seed,
                        "segment": segment,
                        "sample_count": 0,
                        "post_dob_error_mean_mm": "미측정",
                        "nominal_shift_error_mean_mm": "미측정",
                        "nominal_minus_post_error_mean_mm": "미측정",
                        "applied_input_difference_vector_rms_nm": "미측정",
                    })
                    continue
                post_error = np.asarray(post["err_mm"])[mask]
                nominal_error = np.asarray(nominal["err_mm"])[mask]
                input_difference = (
                    np.asarray(nominal["u_applied"])[mask]
                    - np.asarray(post["u_applied"])[mask]
                )
                rows.append({
                    "disturbance_profile": profile,
                    "seed": seed,
                    "segment": segment,
                    "sample_count": int(np.sum(mask)),
                    "post_dob_error_mean_mm": float(np.mean(post_error)),
                    "nominal_shift_error_mean_mm": float(np.mean(nominal_error)),
                    "nominal_minus_post_error_mean_mm": float(np.mean(
                        nominal_error - post_error
                    )),
                    "applied_input_difference_vector_rms_nm": float(np.sqrt(
                        np.mean(np.sum(input_difference ** 2, axis=1))
                    )),
                })
    return rows


def _write_segment_comparison(rows: list[dict]) -> None:
    with (OUTPUT_DIR / "segment_comparison.csv").open(
            "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _time() -> np.ndarray:
    return np.arange(STEPS) * DT


def _saturation_lines(axis, saturated: np.ndarray) -> None:
    indices = np.flatnonzero(saturated)
    if len(indices):
        axis.scatter(indices * DT, np.zeros(len(indices)), marker="|", s=90,
                     color="crimson", alpha=0.8, label="saturation")


def _plot_overview(rows: list[dict]) -> None:
    aggregates = [row for row in rows if row["record_type"] == "aggregate"]
    x = np.arange(len(PROFILES), dtype=float)
    width = 0.25
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    specs = (
        (axes[0, 0], "ee_rmse_mm", "EE RMSE [mm]", True),
        (axes[0, 1], "torque_rate_rms_nm_per_step", "Torque-rate RMS [Nm/step]", False),
        (axes[1, 0], "torque_limit_reached_ratio_pct", "Saturated steps [%]", False),
        (axes[1, 1], "solve_mean_ms", "Mean solve time [ms]", False),
    )
    for mode_index, mode in enumerate(MODES):
        selected = [next(row for row in aggregates
                         if row["disturbance_profile"] == profile
                         and row["mode"] == mode)
                    for profile in PROFILES]
        for axis, field, _, _ in specs:
            values = [row[field] for row in selected]
            errors = [row[f"{field}_std"] for row in selected]
            axis.bar(x + (mode_index - 1) * width, values, width,
                     yerr=errors, capsize=2, label=MODE_LABELS[mode],
                     color=MODE_COLORS[mode])
    for axis, _, ylabel, log_scale in specs:
        axis.set_xticks(x, PROFILES, rotation=18, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        if log_scale:
            axis.set_yscale("log")
    axes[1, 1].axhline(DEADLINE_MS, color="crimson", linestyle="--",
                       linewidth=1.2, label="20 ms period")
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Three MPPI-DOB structures: 3-seed aggregate")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "00_metrics_overview.png", dpi=200)
    plt.close(fig)


def _plot_profile(profile: str, runs: dict[tuple[str, str, int], dict], seed: int = 7) -> None:
    time_s = _time()
    reference = waypoints("circle", n_steps=STEPS)
    reference_at_result = np.vstack((reference[1:], reference[-1:]))

    fig = plt.figure(figsize=(15, 6))
    axis_3d = fig.add_subplot(1, 2, 1, projection="3d")
    axis_error = fig.add_subplot(1, 2, 2)
    axis_3d.plot(reference[:, 0], reference[:, 1], reference[:, 2], "k--",
                 linewidth=2, label="reference")
    for mode in MODES:
        data = runs[(profile, mode, seed)]
        ee = np.asarray(data["ee"])[1:]
        axis_3d.plot(ee[:, 0], ee[:, 1], ee[:, 2],
                     color=MODE_COLORS[mode], label=MODE_LABELS[mode])
        axis_error.plot(time_s, data["err_mm"], color=MODE_COLORS[mode],
                        label=MODE_LABELS[mode])
        saturated = np.asarray(data["actuator_saturated"], dtype=bool)
        if np.any(saturated):
            axis_error.scatter(time_s[saturated], np.asarray(data["err_mm"])[saturated],
                               color=MODE_COLORS[mode], marker="x", s=14)
    axis_3d.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]",
                title=f"EE trajectory ({profile}, seed {seed})")
    axis_error.set(xlabel="Time [s]", ylabel="EE position error [mm]",
                   title="Tracking error on common time axis")
    axis_error.grid(alpha=0.25)
    axis_3d.legend(fontsize=8)
    axis_error.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"01_tracking_{profile}.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    first = runs[(profile, MODES[0], seed)]
    actual = np.asarray(first["d_true"])[1:]
    for joint, axis in enumerate(axes.flat):
        axis.plot(time_s, actual[:, joint], "k--", linewidth=2, label="actual d")
        for mode in MODES:
            estimate = np.asarray(runs[(profile, mode, seed)]["d_hat"])[0:STEPS]
            axis.plot(time_s, estimate[:, joint], color=MODE_COLORS[mode],
                      alpha=0.9, label=f"d_hat: {MODE_LABELS[mode]}")
        axis.set_ylabel(f"J{joint + 1} [Nm]")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Actual disturbance and DOB estimate ({profile}, seed {seed})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"02_disturbance_estimation_{profile}.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    limits = np.asarray(TAU_MAX)
    for joint, axis in enumerate(axes.flat):
        for mode in MODES:
            data = runs[(profile, mode, seed)]
            torque = np.asarray(data["u_applied"])
            axis.plot(time_s, torque[:, joint], color=MODE_COLORS[mode],
                      linewidth=1.1, label=MODE_LABELS[mode])
            saturated = np.asarray(data["actuator_saturated"], dtype=bool)
            joint_saturated = saturated & np.isclose(
                np.abs(torque[:, joint]), limits[joint], atol=1e-5, rtol=0.0
            )
            if np.any(joint_saturated):
                axis.scatter(time_s[joint_saturated], torque[joint_saturated, joint],
                             color=MODE_COLORS[mode], marker="x", s=18)
        axis.axhline(limits[joint], color="crimson", linestyle="--", linewidth=1)
        axis.axhline(-limits[joint], color="crimson", linestyle="--", linewidth=1)
        axis.set_ylabel(f"J{joint + 1} [Nm]")
        axis.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("Time [s]")
    axes[-1, 1].set_xlabel("Time [s]")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Applied joint torque and limits ({profile}, seed {seed}); x = saturation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"03_applied_torque_{profile}.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for axis, mode in zip(axes, MODES):
        data = runs[(profile, mode, seed)]
        u_mppi = np.asarray(data["u_mppi"])
        u_applied = np.asarray(data["u_applied"])
        output_comp = u_applied - u_mppi
        axis.plot(time_s, np.linalg.norm(u_mppi, axis=1), label="||u_MPPI||")
        axis.plot(time_s, np.linalg.norm(output_comp, axis=1), label="||u_applied-u_MPPI||")
        axis.plot(time_s, np.linalg.norm(u_applied, axis=1), label="||u_applied||")
        saturated = np.asarray(data["actuator_saturated"], dtype=bool)
        for index in np.flatnonzero(saturated):
            axis.axvline(time_s[index], color="crimson", alpha=0.08)
        axis.set_ylabel("Torque norm [Nm]")
        axis.set_title(MODE_LABELS[mode])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle(f"MPPI output vs applied input ({profile}, seed {seed})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"04_input_decomposition_{profile}.png", dpi=200)
    plt.close(fig)

    data = runs[(profile, "model_only", seed)]
    warm_residual = np.asarray(data["u_sampling_center"]) - np.asarray(data["u_ff"])
    learned = np.asarray(data["learned_compensation"])
    residual = np.asarray(data["r_star"])
    d_hat = np.asarray(data["d_hat"])[0:STEPS]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(time_s, np.linalg.norm(warm_residual, axis=1), label="||warm-start residual||")
    axes[0].plot(time_s, np.linalg.norm(residual, axis=1), label="||optimized residual R*||")
    axes[0].plot(time_s, np.linalg.norm(learned, axis=1), "--", label="||u_MPPI-u_ff||")
    axes[0].plot(time_s, np.linalg.norm(d_hat, axis=1), label="||d_hat||")
    axes[0].set_ylabel("Torque norm [Nm]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=4)
    axes[1].plot(time_s, data["err_mm"], color=MODE_COLORS["model_only"])
    axes[1].set(xlabel="Time [s]", ylabel="EE error [mm]")
    axes[1].grid(alpha=0.25)
    fig.suptitle(f"Model-only compensation search and residual warm start ({profile}, seed {seed})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"05_model_only_residual_{profile}.png", dpi=200)
    plt.close(fig)


def _fmt(value, digits: int = 3) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):.{digits}f}"


def _aggregate(rows: list[dict], profile: str, mode: str) -> dict:
    return next(row for row in rows if row["record_type"] == "aggregate"
                and row["disturbance_profile"] == profile and row["mode"] == mode)


def _summary_markdown(rows: list[dict], segment_rows: list[dict], metadata: dict) -> str:
    lines = [
        "# MPPI–DOB 세 구조 시뮬레이션 요약",
        "",
        "> PPT 제작용 데이터 정리. 아래에서 **관측 결과**와 **해석**을 분리했다. "
        "측정하거나 로그로 확인할 수 없는 항목은 `미측정`으로 표시한다.",
        "",
        "## 1. 결과 출처와 재현성",
        "",
        f"- 원본 실행 시각: `{metadata['result_created_at_kst']}`",
        f"- Git HEAD: `{metadata['git_head']}`",
        "- 실행 당시 상태: commit 위의 **dirty working tree**. 따라서 commit만으로 결과를 "
        "완전히 식별할 수 없으며 아래 numerical source fingerprint가 우선 식별자다.",
        f"- Numerical source fingerprint: `{metadata['numerical_source_fingerprint']}`",
        f"- Benchmark source SHA-256: `{metadata['benchmark_source_sha256']}`",
        f"- Dataset fingerprint: `{metadata['dataset_fingerprint']}`",
        f"- 실행 스크립트: `{metadata['benchmark_script']}`",
        f"- 원본 수치: `{metadata['source_npz']}`",
        f"- 원본 metric: `{metadata['source_metrics']}`",
        f"- Export 스크립트: `{metadata['export_script']}`",
        "",
        "## 2. 공통 비교 조건",
        "",
        "| 항목 | 설정 |",
        "|---|---|",
        "| 연산 장치 | NVIDIA GeForce GTX 1650, CUDA 13.0, PyTorch 2.13.0+cu130 |",
        "| 로봇 | UR5, 6-DOF nonlinear rigid-body model; standard DH, RNEA mass/bias, gravity, viscous damping |",
        "| Rollout / plant 적분 | rollout: semi-implicit Euler; plant: RK4 |",
        f"| 토크 제한 | `{tuple(TAU_MAX)}` Nm |",
        f"| 기준 궤적 | Cartesian circle, center `{TRAJ_CENTER.tolist()}` m, radius `{TRAJ_RADIUS}` m, 1 revolution |",
        f"| 실행 시간 / 제어 주기 | 5.0 s / {DT:.3f} s ({1/DT:.0f} Hz), 250 steps |",
        f"| MPPI | horizon H={HORIZON}, samples K={N_SAMPLES}, temperature={LAMBDA}, seeds={list(SEEDS)} |",
        f"| Sampling noise variance | diagonal `{tuple(NOISE_VAR)}` Nm²; 당시 legacy independent noise (temporal rho=0과 동등) |",
        f"| Mean update | alpha={MPPI_MEAN_UPDATE_ALPHA} |",
        f"| Cost weights | position={MPPI_W_POS:g}, terminal={MPPI_W_TERM:g}, action={W_ACT:g}, velocity={W_VEL:g}, smoothness={W_SMOOTH:g} |",
        f"| DOB | alpha={DOB_ALPHA:g}, dt={DT:g}; d_hat <- d_hat + dt*alpha*(tau_applied-tau_model-d_hat) |",
        "| Warm start terminal | shifted residual의 마지막 stage를 0으로 설정 |",
        "| 장애물 | 없음 |",
        "| 실행 수 | 외란 5종 × seed 3 × 방식 3 = 45; 45회 모두 CUDA에서 완료 |",
        "",
        "### 외란 조건",
        "",
        f"기본 진폭은 `{tuple(D_AMP)}` Nm, 기본 주파수는 `{tuple(D_FREQ)}` Hz다.",
        "",
        "| Profile | 정의 | 적용 구간 |",
        "|---|---|---|",
        "| constant | d = D_AMP | 0.00–4.98 s |",
        "| step | 0 → D_AMP | 1.24–4.98 s |",
        "| slow_sinusoidal | D_AMP_j sin(2π D_FREQ_j t) | 0.00–4.98 s |",
        "| fast_sinusoidal | D_AMP_j sin(2π f_j t), f=(8,9,10,11,12,13) Hz | 0.00–4.98 s |",
        "| large_step | 0 → 8 D_AMP = (120,-96,80,24,-24,12) Nm | 1.24–4.98 s |",
        "",
        "## 3. 실제 코드 기준 세 구조",
        "",
        "기호: `Uff`는 IK/inverse-dynamics feedforward sequence, `R`은 residual warm start, "
        "`d_hat`은 solve 시작 시 snapshot하여 horizon 전체에 유지한 DOB 추정값이다. "
        "`shift(X)`는 stage 1…H-1을 앞으로 당기고 마지막 stage를 0으로 만든다.",
        "",
        "| 구조 | Sampling center | Rollout effective torque | Plant actuator input | Residual update |",
        "|---|---|---|---|---|",
        "| Post-DOB | Uff + R | clamp(u_sample) | clamp(u_MPPI + d_hat) | R+ = shift(U* - Uff) |",
        "| Rollout Nominal Shift | Uff + R + d_hat | clamp(u_sample) - d_hat | clamp(u_MPPI) | R+ = shift(U* - Uff - d_hat) |",
        "| Model-only Rollout | Uff + R | clamp(u_sample) - d_hat | clamp(u_MPPI) | R+ = shift(U* - Uff) |",
        "",
        "세 방식에서 로봇 모델, 비용함수, DOB, MPPI noise/covariance, feedforward, horizon, "
        "sample 수와 seed는 동일하다. 차이는 위 네 열뿐이다.",
        "",
        "## 4. 관측 결과: seed별 원값",
        "",
        "아래 표의 saturation은 250 step 중 하나 이상의 관절이 actuator limit에 도달한 step 수다. "
        "Deadline miss는 초기 5개 timing warm-up sample을 제외한 245개 중 20 ms를 초과한 횟수다.",
        "",
    ]
    seed_rows = [row for row in rows if row["record_type"] == "seed"]
    for profile in PROFILES:
        lines += [
            f"### {profile}",
            "",
            "| Seed | 방식 | EE RMSE / mean / max [mm] | Joint RMSE [rad] | Peak torque [Nm] | Sat [step, %] | Torque-rate RMS [Nm/step] | Solve mean / P95 / max [ms] | >20 ms | DOB error RMS [Nm/joint] |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in seed_rows:
            if row["disturbance_profile"] != profile:
                continue
            lines.append(
                f"| {row['seed']} | {MODE_LABELS[row['mode']]} | "
                f"{_fmt(row['ee_rmse_mm'])} / {_fmt(row['ee_mean_mm'])} / {_fmt(row['ee_max_mm'])} | "
                f"{_fmt(row['joint_rmse_rad'])} | {_fmt(row['applied_torque_peak_nm'])} | "
                f"{row['torque_limit_reached_steps']}, {_fmt(row['torque_limit_reached_ratio_pct'], 2)} | "
                f"{_fmt(row['torque_rate_rms_nm_per_step'])} | "
                f"{_fmt(row['solve_mean_ms'])} / {_fmt(row['solve_p95_ms'])} / {_fmt(row['solve_max_ms'])} | "
                f"{row['deadline_miss_count']} | {_fmt(row['dob_error_rms_per_joint_nm'])} |"
            )
        lines.append("")

    lines += [
        "## 5. 관측 결과: 3-seed 평균 ± 표준편차",
        "",
        "| 외란 | 방식 | EE RMSE [mm] | EE mean [mm] | EE max [mm] | Joint RMSE [rad] | Torque-rate RMS [Nm/step] | Sat steps | Solve mean [ms] | DOB error RMS [Nm/joint] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile in PROFILES:
        for mode in MODES:
            row = _aggregate(rows, profile, mode)
            def pm(field):
                return f"{_fmt(row[field])} ± {_fmt(row[f'{field}_std'])}"
            lines.append(
                f"| {profile} | {MODE_LABELS[mode]} | {pm('ee_rmse_mm')} | "
                f"{pm('ee_mean_mm')} | {pm('ee_max_mm')} | {pm('joint_rmse_rad')} | "
                f"{pm('torque_rate_rms_nm_per_step')} | {pm('torque_limit_reached_steps')} | "
                f"{pm('solve_mean_ms')} | {pm('dob_error_rms_per_joint_nm')} |"
            )

    large_post = _aggregate(rows, "large_step", "post_dob")
    large_nominal = _aggregate(rows, "large_step", "nominal_shift")
    lines += [
        "",
        "## 6. 비포화/포화 구간 비교",
        "",
        "Post-DOB와 Rollout Nominal Shift에서 같은 profile·seed·time index를 맞췄다. "
        "두 방식 중 하나라도 saturation이면 `saturated`, 둘 다 아니면 `non_saturated`로 분류했다.",
        "",
        "- constant, step, slow/fast sinusoidal: saturation이 없어 포화 구간 지표는 `미측정`이다.",
        f"- large_step: Post-DOB saturation `{_fmt(large_post['torque_limit_reached_steps'])} ± {_fmt(large_post['torque_limit_reached_steps_std'])}` step, "
        f"Nominal Shift `{_fmt(large_nominal['torque_limit_reached_steps'])} ± {_fmt(large_nominal['torque_limit_reached_steps_std'])}` step.",
        "- 정확한 seed별 segment 결과는 `segment_comparison.csv`에 있다.",
        "",
        "| Profile | Seed | 구간 | samples | Post error mean [mm] | Nominal error mean [mm] | Nominal−Post [mm] | 입력 차이 vector RMS [Nm] |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in segment_rows:
        if row["disturbance_profile"] != "large_step":
            continue
        lines.append(
            f"| {row['disturbance_profile']} | {row['seed']} | {row['segment']} | "
            f"{row['sample_count']} | {_fmt(row['post_dob_error_mean_mm'])} | "
            f"{_fmt(row['nominal_shift_error_mean_mm'])} | "
            f"{_fmt(row['nominal_minus_post_error_mean_mm'])} | "
            f"{_fmt(row['applied_input_difference_vector_rms_nm'])} |"
        )

    lines += [
        "",
        "## 7. 관측 사실",
        "",
        "- Post-DOB와 Nominal Shift는 constant, step, slow/fast sinusoidal에서 모두 평균 EE RMSE 약 4–5 mm 범위였다.",
        "- large_step에서는 두 방식 모두 RMSE가 약 14–16 mm로 증가하고 actuator saturation이 발생했다.",
        "- Model-only는 slow sinusoidal에서 평균 RMSE 7.51 mm였지만 constant, step, fast sinusoidal, large_step에서 run별로 큰 추종 이탈이 관측되었다.",
        "- Model-only는 모든 profile에서 actuator saturation 0 step이었다. 이는 토크 여유가 좋다는 의미와 동일하지 않으며, 큰 외란에서 필요한 보상 토크를 찾지 못한 결과와 함께 해석해야 한다.",
        "- 45회 모두 CUDA에서 완료됐다. 실행 실패는 0회다.",
        "- Fallback event는 구현/로그에 별도 항목이 없어 `미측정`이다.",
        "- Joint position/velocity, workspace, self-collision violation은 해당 제약 및 필요한 상태 로그가 없어 `미측정`이다.",
        "",
        "## 8. 해석",
        "",
        "- Post-DOB는 optimizer 밖에서 d_hat을 즉시 더하므로 외란 형태가 바뀌어도 보상 경로가 명시적이다. 다만 MPPI rollout과 action cost가 후보상 후 saturation을 보지 못한다.",
        "- Nominal Shift는 sampling center와 rollout model에 같은 d_hat을 넣고 residual 복원에서 다시 제거한다. 비포화 조건에서는 Post-DOB와 유사한 유효 토크를 만들 수 있지만, sampling/clamp/비선형 dynamics 때문에 완전히 동일하지 않다.",
        "- Model-only는 sampling center에 d_hat을 넣지 않고 rollout model에만 외란을 넣는다. 필요한 보상은 stochastic optimization과 shifted residual이 스스로 찾아야 한다. slow sinusoidal에서는 일부 보상을 찾았지만 constant/step/fast/large-step의 큰 이탈은 제한된 H=10, K=136에서 보상 residual 탐색이 안정적으로 수렴하지 않았음을 시사한다.",
        "- `R*=u_MPPI-u_ff`에는 외란 보상뿐 아니라 tracking correction도 섞여 있다. 따라서 `R*≈d_hat` 여부만으로 최적성을 판단하면 안 된다.",
        "- Large-step 포화 구간에서는 Post-DOB의 후보상 후 clamp와 Nominal Shift의 MPPI 내부 탐색/clamp 위치가 달라 실제 입력 차이가 커질 수 있다. 이 구간은 비포화 등가성 논리로 설명할 수 없다.",
        "",
        "## 9. 가설과 다른 구간 및 한계",
        "",
        "- Model-only가 외란을 rollout에 포함하면 자동으로 보상 토크를 찾을 것이라는 가설은 이 K/H/비용/seed 조건에서는 일반적으로 성립하지 않았다.",
        "- slow sinusoidal 한 조건만 보면 Model-only가 상대적으로 정상처럼 보이므로 단일 외란 결론은 위험하다.",
        "- 이 데이터셋은 5초 circle trajectory와 seed 3개에 한정된다. 다른 궤적, 더 긴 실행, 다른 DOB bandwidth에 일반화할 수 없다.",
        "- 당시 실험은 temporal-noise 기능 추가 전 legacy sampler로 수행됐다. 현재 코드의 `temporal_noise_rho=0`과 알고리즘적으로 동등하지만, 현재 working tree의 source fingerprint는 달라졌다.",
        "- One-step prediction error에는 DOB 오차뿐 아니라 rollout Euler와 plant RK4 차이가 함께 포함된다.",
        "- Torque saturation 이전의 unclamped commanded torque는 저장되지 않아 saturation 초과 요구량은 `미측정`이다.",
        "",
        "## 10. 그림 목록",
        "",
        "- `00_metrics_overview.png`: profile별 3-seed RMSE, torque-rate, saturation, solve time.",
        "- `01_tracking_<profile>.png`: seed 7 기준 EE 기준/실제 궤적과 같은 시간축의 추종 오차.",
        "- `02_disturbance_estimation_<profile>.png`: 실제 d와 세 방식의 control-step d_hat.",
        "- `03_applied_torque_<profile>.png`: 세 방식의 실제 적용 관절 토크, ±torque limit, saturation marker.",
        "- `04_input_decomposition_<profile>.png`: MPPI 출력, MPPI 이후 추가분, 실제 적용 입력의 norm.",
        "- `05_model_only_residual_<profile>.png`: Model-only warm-start residual, optimized residual, d_hat과 추종 오차.",
        "",
        "## 11. 전달 파일",
        "",
        "- `simulation_summary.md`: 본 문서.",
        "- `metrics.csv`: seed별 원자료 45행 + profile/mode 평균·표준편차 15행.",
        "- `time_history.csv`: 45 run × 250 step의 long-format 시간 이력.",
        "- `segment_comparison.csv`: Post-DOB와 Nominal Shift의 비포화/포화 paired comparison.",
        "- `metadata.json`: 코드/환경/원본 파일 fingerprint.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = SOURCE_DIR / "metrics.json"
    npz_path = SOURCE_DIR / "results.npz"
    stored = json.loads(metrics_path.read_text(encoding="utf-8"))
    source_stat = metrics_path.stat()
    metadata = {
        "git_head": _git_head(),
        "working_tree_note": "experiment ran on a dirty working tree",
        "result_created_at_kst": "2026-10-11T05:13:24+09:00",
        "dataset_fingerprint": stored["fingerprint"],
        "numerical_source_fingerprint": stored["spec"]["source"],
        "benchmark_source_sha256": stored["spec"]["benchmark_source"],
        "benchmark_script": str(Path("benchmark_model_only_mppi.py").resolve()),
        "export_script": str(Path(__file__).resolve()),
        "source_npz": str(npz_path.resolve()),
        "source_metrics": str(metrics_path.resolve()),
        "source_metrics_mtime_ns": source_stat.st_mtime_ns,
        "source_npz_sha256": hashlib.sha256(npz_path.read_bytes()).hexdigest(),
        "environment": stored["spec"]["environment"],
        "cuda_preflight": stored["cuda_preflight"],
        "missing_measurements": [
            "fallback count",
            "joint position-limit violations",
            "joint velocity-limit violations",
            "workspace-limit violations",
            "self-collision",
            "unclamped requested actuator torque beyond saturation",
        ],
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        rows, runs = _metric_rows(archive)
        _write_metrics(rows)
        _write_time_history(runs)
        segment_rows = _segment_rows(runs)
        _write_segment_comparison(segment_rows)
        _plot_overview(rows)
        for profile in PROFILES:
            _plot_profile(profile, runs)
    (OUTPUT_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "simulation_summary.md").write_text(
        _summary_markdown(rows, segment_rows, metadata) + "\n", encoding="utf-8"
    )
    print(f"SAVED {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
