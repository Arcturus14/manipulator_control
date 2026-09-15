"""Benchmark obstacle-free MPPI input-smoothness weights."""
from __future__ import annotations

import argparse
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from benchmark_hardware_mppi import environment_metadata
from benchmark_mean_update_mppi import _metrics
from experiment_runner import run_mppi, waypoints


SCENARIOS = {
    "nominal": {"has_disturbance": False, "use_dob": False},
    "disturbance_dob": {"has_disturbance": True, "use_dob": True},
}


def _trajectory_jitter(data: dict, reference: np.ndarray) -> dict[str, float]:
    """Return high-frequency Cartesian motion/error indicators."""
    ee = np.asarray(data["ee"], dtype=float)[1:]
    target = np.concatenate((reference[1:], reference[-1:]), axis=0)
    count = min(len(ee), len(target))
    residual = ee[:count] - target[:count]

    def second_difference_rms(values: np.ndarray) -> float:
        second = np.diff(values, n=2, axis=0)
        return float(np.sqrt(np.mean(np.sum(second * second, axis=1))) * 1e3)

    return {
        "ee_second_diff_rms_mm": second_difference_rms(ee[:count]),
        "tracking_residual_second_diff_rms_mm": second_difference_rms(residual),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "rmse_err_mm", "max_err_mm", "mean_solve_ms", "worst_solve_ms",
        "rms_delta_tau_l2_nm", "p95_delta_tau_l2_nm",
        "ee_second_diff_rms_mm", "tracking_residual_second_diff_rms_mm",
    )
    result: dict[str, Any] = {}
    weights = sorted({float(row["smooth_weight"]) for row in rows})
    for scenario in SCENARIOS:
        for weight in weights:
            selected = [
                row["metrics"] for row in rows
                if row["scenario"] == scenario
                and float(row["smooth_weight"]) == weight
            ]
            summary: dict[str, Any] = {"runs": len(selected)}
            for field in fields:
                values = np.asarray([metrics[field] for metrics in selected])
                summary[f"{field}_mean"] = float(values.mean())
                summary[f"{field}_std"] = float(
                    values.std(ddof=1 if len(values) > 1 else 0)
                )
            result[f"{scenario}__smooth_{weight:g}"] = summary
    return result


def run(*, weights: list[float], seeds: list[int], output: Path,
        require_cuda: bool) -> None:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError("weights must contain positive values")
    if not seeds:
        raise ValueError("at least one seed is required")

    reference = waypoints("infinity")
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        ordered_weights = weights if seed_index % 2 == 0 else list(reversed(weights))
        for scenario, condition in SCENARIOS.items():
            for weight in ordered_weights:
                print(
                    f"\n[SMOOTHNESS] scenario={scenario}, seed={seed}, "
                    f"weight={weight:g}"
                )
                data = run_mppi(
                    reference, seed=seed, smooth_weight=weight,
                    dynamic_paths=None, static_centers=None, **condition,
                )
                metrics = {**_metrics(data), **_trajectory_jitter(data, reference)}
                row = {
                    "scenario": scenario,
                    "seed": seed,
                    "smooth_weight": weight,
                    "metrics": metrics,
                }
                rows.append(row)
                print(json.dumps(row, indent=2))
                del data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    report = {
        "benchmark": "mppi_obstacle_free_smoothness_weight",
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": environment_metadata(),
        "trajectory": "infinity",
        "obstacle": "none",
        "scenarios": SCENARIOS,
        "weights": weights,
        "seeds": seeds,
        "rows": rows,
        "summary": _aggregate(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n[SAVE] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=float,
                        default=[0.25, 0.35, 0.5, 0.75, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--output", type=Path,
                        default=Path("results/smoothness_screen/metrics.json"))
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    run(weights=args.weights, seeds=args.seeds, output=args.output,
        require_cuda=args.require_cuda)


if __name__ == "__main__":
    main()
