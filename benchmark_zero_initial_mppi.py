"""GPU-only comparison of feedforward-centered and zero-initial MPPI."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_dob_aware_nominal_shift import cuda_preflight
from benchmark_hardware_mppi import environment_metadata
from config import DT
from experiment_runner import run_mppi, waypoints


MODES = {
    "feedforward": "feedforward_residual",
    "zero_initial": "zero_initial_shift",
}
SCENARIOS = {
    "nominal": {"has_disturbance": False, "use_dob": False},
    "post_dob": {"has_disturbance": True, "use_dob": True},
}
EARLY_STEPS = 25


def metrics(data: dict[str, Any]) -> dict[str, float | int]:
    error = np.asarray(data["err_mm"], dtype=float)
    torque = np.asarray(data["u_applied"], dtype=float)
    delta_norm = np.linalg.norm(np.diff(torque, axis=0), axis=1)
    solve_ms = np.asarray(data["solve_t"], dtype=float) * 1e3
    solve_ms = solve_ms[np.isfinite(solve_ms)]
    early_count = min(EARLY_STEPS, len(error))
    late = error[early_count:] if early_count < len(error) else error
    return {
        "tracking_mean_mm": float(error.mean()),
        "tracking_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "tracking_max_mm": float(error.max()),
        "early_rmse_mm": float(np.sqrt(np.mean(error[:early_count] ** 2))),
        "late_rmse_mm": float(np.sqrt(np.mean(late ** 2))),
        "actuator_torque_rms_nm": float(np.sqrt(np.mean(torque ** 2))),
        "torque_rate_rms_nm_per_step": float(np.sqrt(np.mean(delta_norm ** 2))),
        "saturated_steps": int(np.asarray(
            data["actuator_saturated"], dtype=bool
        ).sum()),
        "solve_mean_ms": float(solve_ms.mean()),
        "solve_p95_ms": float(np.percentile(solve_ms, 95)),
        "first_sampling_center_norm_nm": float(np.linalg.norm(
            np.asarray(data["u_sampling_center"], dtype=float)[0]
        )),
        "first_feedforward_norm_nm": float(np.linalg.norm(
            np.asarray(data["u_ff"], dtype=float)[0]
        )),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "tracking_rmse_mm", "tracking_max_mm", "early_rmse_mm",
        "late_rmse_mm", "actuator_torque_rms_nm",
        "torque_rate_rms_nm_per_step", "saturated_steps",
        "solve_mean_ms", "solve_p95_ms",
        "first_sampling_center_norm_nm",
    )
    summary: dict[str, Any] = {}
    for scenario in SCENARIOS:
        summary[scenario] = {}
        for mode in MODES:
            selected = [row["metrics"] for row in rows
                        if row["scenario"] == scenario and row["mode"] == mode]
            entry: dict[str, Any] = {"runs": len(selected)}
            for field in fields:
                values = np.asarray([item[field] for item in selected], dtype=float)
                entry[f"{field}_mean"] = float(values.mean())
                entry[f"{field}_std"] = float(
                    values.std(ddof=1 if len(values) > 1 else 0)
                )
            summary[scenario][mode] = entry
    return summary


def save_plots(representative: dict[tuple[str, str], dict[str, Any]],
               summary: dict[str, Any], output_dir: Path) -> None:
    colors = {"feedforward": "#276FBF", "zero_initial": "#D1495B"}
    labels = {"feedforward": "IK feedforward center",
              "zero_initial": "Zero initial + shift"}

    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for column, scenario in enumerate(SCENARIOS):
        for mode in MODES:
            data = representative[(scenario, mode)]
            time_s = np.arange(len(data["err_mm"])) * DT
            axes[0, column].plot(time_s, data["err_mm"], color=colors[mode],
                                 label=labels[mode], linewidth=1.4)
            torque_norm = np.linalg.norm(data["u_applied"], axis=1)
            axes[1, column].plot(time_s, torque_norm, color=colors[mode],
                                 label=labels[mode], linewidth=1.1)
        axes[0, column].set_title(scenario.replace("_", " ").title())
        axes[0, column].set_ylabel("EE error [mm]")
        axes[1, column].set_ylabel("Applied torque norm [N m]")
        axes[1, column].set_xlabel("Time [s]")
        axes[0, column].grid(alpha=0.25)
        axes[1, column].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    figure.suptitle("MPPI sampling-center comparison (representative seed)")
    figure.tight_layout()
    figure.savefig(output_dir / "timeseries_comparison.png", dpi=180)
    plt.close(figure)

    fields = ("early_rmse_mm", "late_rmse_mm", "tracking_rmse_mm")
    field_labels = ("First 0.5 s", "After 0.5 s", "Whole episode")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    x = np.arange(len(fields))
    width = 0.36
    for axis, scenario in zip(axes, SCENARIOS):
        for index, mode in enumerate(MODES):
            means = [summary[scenario][mode][f"{field}_mean"] for field in fields]
            stds = [summary[scenario][mode][f"{field}_std"] for field in fields]
            axis.bar(x + (index - 0.5) * width, means, width, yerr=stds,
                     capsize=3, label=labels[mode], color=colors[mode])
        axis.set_title(scenario.replace("_", " ").title())
        axis.set_xticks(x, field_labels)
        axis.set_ylabel("Tracking RMSE [mm]")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    figure.suptitle("Tracking error: mean and seed standard deviation")
    figure.tight_layout()
    figure.savefig(output_dir / "rmse_comparison.png", dpi=180)
    plt.close(figure)


def report_text(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# MPPI 초기 input sequence 비교 보고서",
        "",
        f"작성일: {report['created_at'][:10]}",
        "",
        "## 결론",
        "",
    ]
    for scenario in SCENARIOS:
        base = summary[scenario]["feedforward"]["tracking_rmse_mm_mean"]
        zero = summary[scenario]["zero_initial"]["tracking_rmse_mm_mean"]
        change = 100.0 * (zero / base - 1.0)
        lines.append(
            f"- **{scenario}**: zero-initial RMSE {zero:.3f} mm, "
            f"기존 feedforward 중심 {base:.3f} mm 대비 {change:+.1f}%."
        )
    lines += [
        "",
        "`zero_initial_shift`는 교과서적인 absolute-control warm start 예시로 "
        "첫 MPPI solve의 전체 input sequence를 0으로 설정한다. 이후에는 직전 "
        "최적 sequence를 한 칸 shift하고 마지막 입력을 0으로 채운다. IK로 계산한 "
        "`U_ff`는 joint reference와 진단 로그에만 사용하며 sampling 중심에는 "
        "더하지 않았다. GUI와 일반 실행의 기본값은 기존 "
        "`feedforward_residual`이다.",
        "사용 중인 `pytorch_mppi`의 표준 shift 역시 sequence를 한 칸 roll한 뒤 "
        "마지막 항을 기본 `u_init=0`으로 채우므로 동일한 규칙이다.",
        "",
        "## 구성 차이",
        "",
        "```text",
        "기존: U_center(k) = U_ff(k) + shifted residual",
        "추가: U_center(0) = 0",
        "      U_center(k+1) = shift(U_star(k)), terminal = 0",
        "공통: U_sample = clamp(U_center + epsilon)",
        "      동일 dynamics, cost, noise, sample 수, horizon, seed",
        "```",
        "",
        "외란 비교는 rollout 변경 없이 기존 `dob_compensation_mode=output`, "
        "즉 Post-DOB 후보상을 사용했다.",
        "",
        "## GPU 및 실험 설정",
        "",
        f"- GPU: {report['cuda_preflight']['gpu_name']}",
        f"- PyTorch/CUDA: {report['cuda_preflight']['torch']} / "
        f"{report['cuda_preflight']['cuda_runtime']}",
        f"- trajectory: circle, {report['steps']} steps "
        f"({report['steps'] * DT:.1f} s)",
        f"- seeds: {', '.join(map(str, report['seeds']))}",
        "- 초기 구간: 첫 25 steps (0.5 s)",
        "- CUDA preflight와 각 결과의 `compute_device=cuda`를 통과한 결과만 저장",
        "",
        "## 정량 결과 (3-seed 평균 ± 표준편차)",
        "",
        "| 조건 | 중심 | 전체 RMSE [mm] | 초기 RMSE [mm] | 후기 RMSE [mm] | "
        "Torque-rate RMS [N·m/step] | Mean solve [ms] |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in SCENARIOS:
        for mode in MODES:
            item = summary[scenario][mode]
            def value(field: str) -> str:
                return (f"{item[field + '_mean']:.3f} ± "
                        f"{item[field + '_std']:.3f}")
            lines.append(
                f"| {scenario} | {mode} | {value('tracking_rmse_mm')} | "
                f"{value('early_rmse_mm')} | {value('late_rmse_mm')} | "
                f"{value('torque_rate_rms_nm_per_step')} | "
                f"{value('solve_mean_ms')} |"
            )
    lines += [
        "",
        "## 초기화 검증",
        "",
    ]
    for scenario in SCENARIOS:
        ff = summary[scenario]["feedforward"][
            "first_sampling_center_norm_nm_mean"
        ]
        zero = summary[scenario]["zero_initial"][
            "first_sampling_center_norm_nm_mean"
        ]
        lines.append(
            f"- {scenario}: 첫 center norm은 기존 {ff:.6f} N·m, "
            f"zero-initial {zero:.6f} N·m."
        )
    lines += [
        "",
        "## 해석",
        "",
        "현재 구현은 torque 자체를 sampling하며 한 제어 주기마다 MPPI update를 "
        "한 번 수행한다. 중력·궤적 가속도를 상쇄할 입력이 0에서 멀기 때문에, "
        "zero-initial 방식은 첫 구간에서 필요한 torque를 noise sampling만으로 "
        "찾아야 한다. 따라서 차이가 크다면 알고리즘 오류라기보다 `U_ff`가 "
        "현재 문제에서 강한 control prior 역할을 한다는 의미다. 반대로 충분한 "
        "반복 update, 더 큰/적응형 covariance, 또는 residual-control formulation을 "
        "사용하면 zero 초기화의 불리함은 줄어들 수 있다.",
        "",
        "## 산출물",
        "",
        "- `metrics.json`: run별 지표와 집계",
        "- `results.npz`: 모든 비교 시계열",
        "- `timeseries_comparison.png`: 대표 seed의 오차/토크",
        "- `rmse_comparison.png`: 초기·후기·전체 RMSE",
        "",
    ]
    return "\n".join(lines)


def run(seeds: list[int], steps: int, output_dir: Path) -> None:
    preflight = cuda_preflight()
    print("[CUDA PREFLIGHT] " + json.dumps(preflight))
    reference = waypoints("circle", n_steps=steps)
    rows: list[dict[str, Any]] = []
    payload: dict[str, np.ndarray] = {}
    representative: dict[tuple[str, str], dict[str, Any]] = {}

    for scenario_index, (scenario, scenario_options) in enumerate(
            SCENARIOS.items()):
        for seed_index, seed in enumerate(seeds):
            mode_names = list(MODES)
            if (scenario_index + seed_index) % 2:
                mode_names.reverse()
            for mode in mode_names:
                print(f"\n[RUN] scenario={scenario}, seed={seed}, mode={mode}")
                data = run_mppi(
                    reference,
                    **scenario_options,
                    dob_compensation_mode="output",
                    sampling_center_mode=MODES[mode],
                    require_cuda=True,
                    seed=seed,
                )
                if str(data["compute_device"]) != "cuda":
                    raise RuntimeError("non-CUDA result rejected before saving")
                row = {"scenario": scenario, "seed": seed, "mode": mode,
                       "metrics": metrics(data)}
                rows.append(row)
                print(json.dumps(row, indent=2))
                prefix = f"{scenario}__{mode}__seed_{seed}"
                for name, value in data.items():
                    if not name.startswith("_"):
                        payload[f"{prefix}__{name}"] = np.asarray(value)
                if seed == seeds[0]:
                    representative[(scenario, mode)] = data
                else:
                    del data
                gc.collect()
                torch.cuda.empty_cache()

    report = {
        "benchmark": "mppi_zero_initial_sequence",
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": environment_metadata(),
        "cuda_preflight": preflight,
        "trajectory": "circle",
        "steps": steps,
        "seeds": seeds,
        "modes": MODES,
        "scenarios": SCENARIOS,
        "rows": rows,
        "summary": aggregate(rows),
    }
    if not report["environment"].get("cuda_available"):
        raise RuntimeError("CUDA metadata validation failed; refusing to save")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "results.npz", **payload)
    save_plots(representative, report["summary"], output_dir)
    (output_dir / "REPORT.md").write_text(
        report_text(report), encoding="utf-8"
    )
    print(f"[SAVE] {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/mppi_zero_initial_gpu"),
    )
    args = parser.parse_args()
    run(args.seeds, args.steps, args.output_dir)


if __name__ == "__main__":
    main()
