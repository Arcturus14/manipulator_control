"""Compare external DOB output compensation with DOB-aware MPPI rollouts."""
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
from config import (HORIZON, LAMBDA, MPPI_MEAN_UPDATE_ALPHA, MPPI_W_POS,
                    MPPI_W_TERM, NOISE_VAR, N_SAMPLES, TAU_MAX, W_SMOOTH)
from experiment_runner import run_mppi, waypoints


MODES = ("output", "rollout")


def _run_metrics(data: dict) -> dict[str, Any]:
    metrics = _metrics(data)
    torque = np.asarray(data["u"], dtype=float)
    limits = np.asarray(TAU_MAX, dtype=float)
    saturated = np.isclose(
        np.abs(torque), limits, rtol=0.0, atol=1e-5
    ).any(axis=1)
    metrics.update({
        "rms_tau_l2_nm": float(np.sqrt(np.mean(np.sum(torque ** 2, axis=1)))),
        "saturated_step_pct": float(np.mean(saturated) * 100.0),
    })
    return metrics


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "mean_err_mm", "rmse_err_mm", "max_err_mm", "final_err_mm",
        "dob_rms_nm", "mean_delta_tau_l2_nm", "rms_delta_tau_l2_nm",
        "p95_delta_tau_l2_nm", "max_delta_tau_l2_nm", "rms_tau_l2_nm",
        "saturated_step_pct", "mean_solve_ms", "worst_solve_ms",
    )
    summary: dict[str, Any] = {}
    for mode in MODES:
        selected = [row["metrics"] for row in rows if row["mode"] == mode]
        mode_summary: dict[str, Any] = {"runs": len(selected)}
        for field in fields:
            values = np.asarray([entry[field] for entry in selected], dtype=float)
            mode_summary[f"{field}_mean"] = float(values.mean())
            mode_summary[f"{field}_std"] = float(
                values.std(ddof=1 if len(values) > 1 else 0)
            )
        summary[mode] = mode_summary
    return summary


def _paired_comparison(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(row["seed"], row["mode"]): row["metrics"] for row in rows}
    fields = (
        "rmse_err_mm", "mean_err_mm", "max_err_mm",
        "rms_delta_tau_l2_nm", "p95_delta_tau_l2_nm",
        "rms_tau_l2_nm", "saturated_step_pct", "mean_solve_ms",
    )
    seeds = sorted({row["seed"] for row in rows})
    result: dict[str, Any] = {}
    for field in fields:
        output = np.asarray([by_key[(seed, "output")][field] for seed in seeds])
        rollout = np.asarray([by_key[(seed, "rollout")][field] for seed in seeds])
        difference = rollout - output
        valid = output != 0.0
        percent = np.full_like(output, np.nan, dtype=float)
        percent[valid] = 100.0 * difference[valid] / output[valid]
        result[field] = {
            "rollout_minus_output_mean": float(difference.mean()),
            "rollout_vs_output_pct_mean": (
                float(percent[valid].mean()) if np.any(valid) else None
            ),
            "per_seed_rollout_vs_output_pct": [
                float(value) if np.isfinite(value) else None
                for value in percent
            ],
        }
    return result


def run(*, seeds: list[int], steps: int, output: Path) -> None:
    if not seeds:
        raise ValueError("at least one seed is required")
    if steps < 3:
        raise ValueError("steps must be at least 3")

    reference = waypoints("circle", n_steps=steps)
    rows: list[dict[str, Any]] = []
    for seed_index, seed in enumerate(seeds):
        mode_order = MODES if seed_index % 2 == 0 else tuple(reversed(MODES))
        for mode in mode_order:
            print(f"\n[DOB MODE] trajectory=circle, seed={seed}, mode={mode}")
            data = run_mppi(
                reference,
                has_disturbance=True,
                use_dob=True,
                dob_compensation_mode=mode,
                dynamic_paths=None,
                static_centers=None,
                seed=seed,
            )
            row = {"seed": seed, "mode": mode, "metrics": _run_metrics(data)}
            rows.append(row)
            print(json.dumps(row, indent=2))
            del data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    report = {
        "benchmark": "mppi_dob_compensation_mode",
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": environment_metadata(),
        "trajectory": "circle",
        "steps": steps,
        "duration_s": steps * 0.02,
        "disturbance": "identical deterministic sinusoidal disturbance",
        "dob": True,
        "seeds": seeds,
        "modes": {
            "output": "original post-optimizer +d_hat compensation",
            "rollout": "tau_effective=tau_sample-d_hat in prediction; no post compensation",
        },
        "mppi": {
            "samples": N_SAMPLES,
            "horizon": HORIZON,
            "temperature": LAMBDA,
            "mean_update_alpha": MPPI_MEAN_UPDATE_ALPHA,
            "noise_variance": list(NOISE_VAR),
            "position_weight": MPPI_W_POS,
            "terminal_weight": MPPI_W_TERM,
            "smooth_weight": W_SMOOTH,
        },
        "rows": rows,
        "summary": _aggregate(rows),
        "paired_rollout_vs_output": _paired_comparison(rows),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n[DOB MODE SUMMARY]")
    print(json.dumps(report["summary"], indent=2))
    print("\n[PAIRED ROLLOUT VS OUTPUT]")
    print(json.dumps(report["paired_rollout_vs_output"], indent=2))
    print(f"[SAVE] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/dob_rollout_comparison/metrics.json"),
    )
    args = parser.parse_args()
    run(seeds=args.seeds, steps=args.steps, output=args.output)


if __name__ == "__main__":
    main()
