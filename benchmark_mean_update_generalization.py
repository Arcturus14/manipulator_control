"""Validate the tuned alpha=0.9 MPPI across DOB and obstacle conditions."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_hardware_mppi import environment_metadata
from benchmark_mean_update_mppi import torque_smoothness_metrics
from config import DT, HORIZON_OBS, TAU_MAX
from experiment_runner import run_mppi, waypoints
from utils.trajectory import chasing_obstacle_paths, static_obstacle_centers
from visualization.plotters import (
    assess_dynamic_obstacle,
    assess_static_obstacles,
    compute_metrics,
    filtered_solve_times,
    plot_3d_trajectory,
    plot_joint_torques,
    plot_timeseries,
)


VARIANTS: dict[str, dict[str, Any]] = {
    "original_alpha_1": {
        "mean_update_alpha": 1.0,
        "n_samples": 136,
        "horizon": 10,
        "temperature": 0.20,
        "noise_variance": [16.0, 28.0, 16.0, 3.2, 3.2, 1.0],
        "position_weight": 960_000.0,
        "terminal_weight": 2_400_000.0,
        "smooth_weight": 0.25,
    },
    "tuned_alpha_0.9": {
        "mean_update_alpha": 0.9,
        "n_samples": 136,
        "horizon": 10,
        "temperature": 0.24,
        "noise_variance": [13.6, 23.8, 13.6, 2.72, 2.72, 0.85],
        "position_weight": 1_200_000.0,
        "terminal_weight": 3_600_000.0,
        "smooth_weight": 0.25,
    },
}

SCENARIOS = {
    "dist_off_dob_off": {"has_disturbance": False, "use_dob": False},
    "dist_off_dob_on": {"has_disturbance": False, "use_dob": True},
    "dist_on_dob_on": {"has_disturbance": True, "use_dob": True},
}
OBSTACLE_MODES = ("none", "static", "dynamic")
SCALAR_METRICS = (
    "mean_err_mm", "rmse_err_mm", "std_err_mm", "max_err_mm",
    "final_err_mm", "mean_solve_ms", "p95_solve_ms", "worst_solve_ms",
    "deadline_miss_pct", "dob_rms_nm", "mean_delta_tau_l2_nm",
    "rms_delta_tau_l2_nm", "p95_delta_tau_l2_nm",
    "max_delta_tau_l2_nm", "saturated_step_pct", "contact_count",
    "min_clearance_mm",
)
VECTOR_METRICS = (
    "torque_mean_per_joint_nm", "torque_mean_abs_per_joint_nm",
    "torque_rms_per_joint_nm", "torque_std_per_joint_nm",
    "torque_peak_abs_per_joint_nm", "mean_abs_delta_tau_per_joint_nm",
    "rms_delta_tau_per_joint_nm",
)


def _metrics(data: dict, obstacle_mode: str, obstacle_data: np.ndarray | None
             ) -> dict[str, Any]:
    torque = np.asarray(data["u"], dtype=float)
    solve_time = filtered_solve_times(data, preserve_length=False)
    saturated = np.isclose(
        np.abs(torque), np.asarray(TAU_MAX), rtol=0.0, atol=1e-5
    ).any(axis=1)
    result = {
        **compute_metrics(data),
        **torque_smoothness_metrics(torque),
        "p95_solve_ms": float(np.percentile(solve_time, 95) * 1e3),
        "deadline_miss_pct": float(np.mean(solve_time > DT) * 100.0),
        "saturated_step_pct": float(np.mean(saturated) * 100.0),
        "torque_mean_per_joint_nm": np.mean(torque, axis=0).tolist(),
        "torque_mean_abs_per_joint_nm": np.mean(np.abs(torque), axis=0).tolist(),
        "torque_rms_per_joint_nm": np.sqrt(np.mean(torque ** 2, axis=0)).tolist(),
        "torque_std_per_joint_nm": np.std(torque, axis=0).tolist(),
        "torque_peak_abs_per_joint_nm": np.max(np.abs(torque), axis=0).tolist(),
        "contact_count": 0,
        "min_clearance_mm": None,
    }
    if obstacle_mode == "static":
        assessments = assess_static_obstacles(data["ee"], obstacle_data)
        result["contact_count"] = int(sum(row["contact"] for row in assessments))
        result["min_clearance_mm"] = float(min(
            row["clearance_mm"] for row in assessments
        ))
        result["static_clearance_mm"] = [
            float(row["clearance_mm"]) for row in assessments
        ]
    elif obstacle_mode == "dynamic":
        assessment = assess_dynamic_obstacle(data["ee"], obstacle_data[0])
        result["contact_count"] = int(bool(assessment["contact"]))
        result["min_clearance_mm"] = float(assessment["clearance_mm"])
    return result


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for obstacle_mode in OBSTACLE_MODES:
        for scenario in SCENARIOS:
            for variant in VARIANTS:
                selected = [row for row in rows
                            if row["obstacle_mode"] == obstacle_mode
                            and row["scenario"] == scenario
                            and row["variant"] == variant]
                summary: dict[str, Any] = {"runs": len(selected)}
                for field in SCALAR_METRICS:
                    values = [row["metrics"][field] for row in selected
                              if row["metrics"][field] is not None]
                    if not values:
                        summary[f"{field}_mean"] = None
                        summary[f"{field}_std"] = None
                        continue
                    array = np.asarray(values, dtype=float)
                    summary[f"{field}_mean"] = float(array.mean())
                    summary[f"{field}_std"] = float(
                        array.std(ddof=1 if len(array) > 1 else 0)
                    )
                for field in VECTOR_METRICS:
                    array = np.asarray([
                        row["metrics"][field] for row in selected
                    ], dtype=float)
                    summary[f"{field}_mean"] = array.mean(axis=0).tolist()
                    summary[f"{field}_std"] = array.std(
                        axis=0, ddof=1 if len(array) > 1 else 0
                    ).tolist()
                key = f"{obstacle_mode}__{scenario}__{variant}"
                result[key] = summary
    return result


def _comparisons(summary: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for obstacle_mode in OBSTACLE_MODES:
        for scenario in SCENARIOS:
            prefix = f"{obstacle_mode}__{scenario}"
            original = summary[f"{prefix}__original_alpha_1"]
            tuned = summary[f"{prefix}__tuned_alpha_0.9"]

            def change(field: str) -> float:
                baseline = original[field]
                return 100.0 * (tuned[field] / baseline - 1.0)

            result[prefix] = {
                "tracking_rmse_change_pct": change("rmse_err_mm_mean"),
                "torque_delta_rms_change_pct": change(
                    "rms_delta_tau_l2_nm_mean"
                ),
                "torque_delta_p95_change_pct": change(
                    "p95_delta_tau_l2_nm_mean"
                ),
                "mean_solve_time_change_pct": change("mean_solve_ms_mean"),
                "original_contact_count_mean": original["contact_count_mean"],
                "tuned_contact_count_mean": tuned["contact_count_mean"],
                "original_min_clearance_mm_mean": original[
                    "min_clearance_mm_mean"
                ],
                "tuned_min_clearance_mm_mean": tuned[
                    "min_clearance_mm_mean"
                ],
            }
    return result


def _save_plots(representative: dict[tuple[str, str, str], dict],
                reference: np.ndarray, obstacles: dict[str, np.ndarray | None],
                output_dir: Path) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for obstacle_mode in OBSTACLE_MODES:
        for scenario in SCENARIOS:
            results = {
                "CURRENT": representative[
                    (obstacle_mode, scenario, "original_alpha_1")
                ],
                "MEAN_UPDATE": representative[
                    (obstacle_mode, scenario, "tuned_alpha_0.9")
                ],
            }
            stem = f"{obstacle_mode}__{scenario}"
            title = f"Applied Joint Torque — {obstacle_mode} — {scenario}"
            figure = plot_joint_torques(results, title=title)
            figure.savefig(plot_dir / f"torque__{stem}.png", dpi=180,
                           bbox_inches="tight")
            plt.close(figure)
            figure = plot_timeseries(results)
            figure.savefig(plot_dir / f"tracking_timing__{stem}.png", dpi=180,
                           bbox_inches="tight")
            plt.close(figure)
            figure = plot_3d_trajectory(
                results, reference,
                dynamic_paths=(obstacles["dynamic"]
                               if obstacle_mode == "dynamic" else None),
                static_centers=(obstacles["static"]
                                if obstacle_mode == "static" else None),
            )
            figure.savefig(plot_dir / f"trajectory__{stem}.png", dpi=180,
                           bbox_inches="tight")
            plt.close(figure)


def run(output_dir: Path, seeds: list[int], *, require_cuda: bool = False) -> None:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but unavailable; refusing to record CPU results"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    reference = waypoints("infinity")
    obstacles: dict[str, np.ndarray | None] = {
        "none": None,
        "static": static_obstacle_centers(reference),
        "dynamic": chasing_obstacle_paths(reference, HORIZON_OBS + 2),
    }
    rows: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    representative: dict[tuple[str, str, str], dict] = {}
    total = len(seeds) * len(OBSTACLE_MODES) * len(SCENARIOS) * len(VARIANTS)
    index = 0
    for seed_index, seed in enumerate(seeds):
        variant_items = list(VARIANTS.items())
        if seed_index % 2:
            variant_items.reverse()
        for obstacle_mode in OBSTACLE_MODES:
            obstacle_data = obstacles[obstacle_mode]
            obstacle_options = {
                "static_centers": (obstacle_data
                                   if obstacle_mode == "static" else None),
                "dynamic_paths": (obstacle_data
                                  if obstacle_mode == "dynamic" else None),
            }
            for scenario, condition in SCENARIOS.items():
                for variant, parameters in variant_items:
                    index += 1
                    print(
                        f"\n[VALIDATE {index}/{total}] seed={seed}, "
                        f"obstacle={obstacle_mode}, scenario={scenario}, "
                        f"variant={variant}"
                    )
                    data = run_mppi(
                        reference, seed=seed, **condition, **obstacle_options,
                        **parameters,
                    )
                    metrics = _metrics(data, obstacle_mode, obstacle_data)
                    rows.append({
                        "seed": seed,
                        "obstacle_mode": obstacle_mode,
                        "scenario": scenario,
                        "variant": variant,
                        "metrics": metrics,
                    })
                    print(json.dumps(metrics, indent=2))
                    prefix = f"{obstacle_mode}__{scenario}__{variant}__seed_{seed}"
                    for name, value in data.items():
                        arrays[f"{prefix}__{name}"] = np.asarray(value)
                    if seed == seeds[0]:
                        representative[(obstacle_mode, scenario, variant)] = data
                    else:
                        del data
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

    summary = _aggregate(rows)
    report = {
        "benchmark": "mppi_mean_update_generalization",
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": environment_metadata(),
        "require_cuda": require_cuda,
        "trajectory": "infinity",
        "duration_s": len(reference) * DT,
        "seeds": seeds,
        "obstacle_modes": list(OBSTACLE_MODES),
        "scenarios": SCENARIOS,
        "variants": VARIANTS,
        "obstacle_parameters": {
            "static_count": int(len(obstacles["static"])),
            "dynamic_path_count": int(len(obstacles["dynamic"])),
        },
        "rows": rows,
        "summary": summary,
        "comparisons": _comparisons(summary),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "results.npz", **arrays)
    _save_plots(representative, reference, obstacles, output_dir)
    print(json.dumps(report["comparisons"], indent=2))
    print(f"[SAVE] {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("results/mean_update_generalization"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--require-cuda", action="store_true",
                        help="fail before running unless CUDA is available")
    args = parser.parse_args()
    run(args.output, args.seeds, require_cuda=args.require_cuda)


if __name__ == "__main__":
    main()
