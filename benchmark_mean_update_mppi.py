"""Compare current and CuRobo-style MPPI mean updates with DOB off and on."""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from experiment_runner import run_mppi, waypoints
from visualization.plotters import compute_metrics, plot_joint_torques


def torque_smoothness_metrics(control: np.ndarray) -> dict[str, Any]:
    """Measure changes between consecutive applied joint-torque commands."""
    torque = np.asarray(control, dtype=float)
    if torque.ndim != 2 or torque.shape[1] != 6 or len(torque) < 2:
        raise ValueError("control history must have shape (steps >= 2, 6)")
    delta = np.diff(torque, axis=0)
    delta_l2 = np.linalg.norm(delta, axis=1)
    return {
        "mean_delta_tau_l2_nm": float(np.mean(delta_l2)),
        "rms_delta_tau_l2_nm": float(np.sqrt(np.mean(delta_l2 ** 2))),
        "p95_delta_tau_l2_nm": float(np.percentile(delta_l2, 95)),
        "max_delta_tau_l2_nm": float(np.max(delta_l2)),
        "mean_abs_delta_tau_per_joint_nm": np.mean(np.abs(delta), axis=0).tolist(),
        "rms_delta_tau_per_joint_nm": np.sqrt(np.mean(delta ** 2, axis=0)).tolist(),
    }


def _metrics(data: dict) -> dict[str, Any]:
    return {**compute_metrics(data), **torque_smoothness_metrics(data["u"])}


def _summary(rows: list[dict]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    scalar_fields = (
        "rmse_err_mm", "mean_solve_ms", "worst_solve_ms",
        "mean_delta_tau_l2_nm", "rms_delta_tau_l2_nm",
        "p95_delta_tau_l2_nm", "max_delta_tau_l2_nm",
    )
    vector_fields = (
        "mean_abs_delta_tau_per_joint_nm",
        "rms_delta_tau_per_joint_nm",
    )
    for use_dob in (False, True):
        for alpha in sorted({row["alpha"] for row in rows}, reverse=True):
            selected = [
                row for row in rows
                if row["use_dob"] == use_dob and row["alpha"] == alpha
            ]
            key = f"dob_{'on' if use_dob else 'off'}_alpha_{alpha:g}"
            summary: dict[str, Any] = {"runs": len(selected)}
            for field in scalar_fields:
                values = np.asarray([row["metrics"][field] for row in selected])
                summary[f"{field}_mean"] = float(values.mean())
                summary[f"{field}_std"] = float(
                    values.std(ddof=1 if len(values) > 1 else 0)
                )
            for field in vector_fields:
                values = np.asarray([row["metrics"][field] for row in selected])
                summary[f"{field}_mean"] = values.mean(axis=0).tolist()
            result[key] = summary
    return result


def _comparisons(summary: dict[str, Any], baseline_alpha: float,
                 candidate_alpha: float) -> dict[str, Any]:
    comparisons = {}
    for use_dob in (False, True):
        mode = f"dob_{'on' if use_dob else 'off'}"
        baseline = summary[f"{mode}_alpha_{baseline_alpha:g}"]
        candidate = summary[f"{mode}_alpha_{candidate_alpha:g}"]

        def reduction(field: str) -> float:
            return 100.0 * (1.0 - candidate[field] / baseline[field])

        comparisons[mode] = {
            "baseline_alpha": baseline_alpha,
            "candidate_alpha": candidate_alpha,
            "rms_delta_tau_reduction_pct": reduction(
                "rms_delta_tau_l2_nm_mean"
            ),
            "mean_delta_tau_reduction_pct": reduction(
                "mean_delta_tau_l2_nm_mean"
            ),
            "p95_delta_tau_reduction_pct": reduction(
                "p95_delta_tau_l2_nm_mean"
            ),
            "tracking_rmse_change_pct": -reduction("rmse_err_mm_mean"),
            "solve_time_change_pct": -reduction("mean_solve_ms_mean"),
        }
    return comparisons


def _save_torque_plots(representative: dict[tuple[bool, float], dict], *,
                       alphas: list[float], output_dir: Path) -> None:
    for use_dob in (False, True):
        results = {
            "CURRENT": representative[(use_dob, alphas[0])],
            "MEAN_UPDATE": representative[(use_dob, alphas[1])],
        }
        mode = "ON" if use_dob else "OFF"
        figure = plot_joint_torques(
            results,
            title=(f"Applied Joint Torque — DOB {mode} "
                   f"(alpha={alphas[0]:g} vs {alphas[1]:g})"),
        )
        destination = output_dir / f"joint_torques_dob_{mode.lower()}.png"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"[PLOT] {destination}")


def run_benchmark(*, seeds: list[int], trajectory: str, alphas: list[float],
                  output_dir: Path, save_result: bool, include_torque: bool,
                  n_samples: int, horizon: int, temperature: float,
                  noise_variance: list[float],
                  position_weight: float, terminal_weight: float,
                  smooth_weight: float) -> None:
    if len(alphas) != 2 or any(not 0 < alpha <= 1 for alpha in alphas):
        raise ValueError("alphas must contain two values in (0, 1]")
    if not seeds:
        raise ValueError("at least one seed is required")
    reference = waypoints(trajectory)
    rows: list[dict] = []
    representative: dict[tuple[bool, float], dict] = {}
    result_payload: dict[str, np.ndarray] = {}

    for seed_index, seed in enumerate(seeds):
        alpha_order = alphas if seed_index % 2 == 0 else list(reversed(alphas))
        for use_dob in (False, True):
            for alpha in alpha_order:
                print(
                    f"\n[MEAN UPDATE] seed={seed}, DOB={'ON' if use_dob else 'OFF'}, "
                    f"disturbance={'ON' if use_dob else 'OFF'}, alpha={alpha:g}"
                )
                data = run_mppi(
                    reference, has_disturbance=use_dob, use_dob=use_dob,
                    dynamic_paths=None, static_centers=None, seed=seed,
                    n_samples=n_samples, horizon=horizon,
                    temperature=temperature, mean_update_alpha=alpha,
                    noise_variance=noise_variance,
                    position_weight=position_weight,
                    terminal_weight=terminal_weight,
                    smooth_weight=smooth_weight,
                )
                row = {
                    "seed": seed,
                    "use_dob": use_dob,
                    "has_disturbance": use_dob,
                    "alpha": alpha,
                    "metrics": _metrics(data),
                }
                rows.append(row)
                print(json.dumps(row, indent=2))
                if seed == seeds[0]:
                    representative[(use_dob, alpha)] = data
                if save_result:
                    prefix = f"dob_{int(use_dob)}_alpha_{alpha:g}_seed_{seed}"
                    for name, value in data.items():
                        result_payload[f"{prefix}__{name}"] = np.asarray(value)
                if seed != seeds[0]:
                    del data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = _summary(rows)
    report = {
        "benchmark": "mppi_mean_update",
        "trajectory": trajectory,
        "conditions": {
            "dob_off": {"disturbance": False, "dob": False},
            "dob_on": {"disturbance": True, "dob": True},
        },
        "obstacle": "none",
        "seeds": seeds,
        "alphas": alphas,
        "rows": rows,
        "summary": summary,
        "comparisons": _comparisons(summary, alphas[0], alphas[1]),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n[MEAN UPDATE SUMMARY]")
    print(json.dumps(report["comparisons"], indent=2))
    print(f"[SAVE] {metrics_path}")
    if save_result:
        destination = output_dir / "results.npz"
        np.savez_compressed(destination, **result_payload)
        print(f"[SAVE] {destination}")
    if include_torque:
        _save_torque_plots(representative, alphas=alphas,
                           output_dir=output_dir)
