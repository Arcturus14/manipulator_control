"""Evaluate MPPI parameter candidates on reference-indexed static cubes."""
from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import torch

from config import N_SAMPLES, TAU_MAX
from experiment_runner import run_mppi, waypoints
from utils.trajectory import static_obstacle_centers
from visualization.plotters import assess_static_obstacles, compute_metrics


CASES = {
    "baseline_original": {
        "n_samples": 136,
        "horizon": 10,
        "temperature": 0.2,
        "static_exp_weight": 5_000.0,
        "static_exp_gain": 1.5,
        "static_margin": 0.050,
        "static_zone": 0.030,
    },
    "tuned_quality": {
        "n_samples": 128,
        "horizon": 13,
        "temperature": 0.2,
        "static_exp_weight": 5_000.0,
        "static_exp_gain": 500.0,
        "static_margin": 0.002,
        "static_zone": 0.004,
    },
    "tuned_realtime": {
        "n_samples": 112,
        "horizon": 9,
        "temperature": 0.2,
        "static_exp_weight": 5_000.0,
        "static_exp_gain": 500.0,
        "static_margin": 0.002,
        "static_zone": 0.004,
    },
    "horizon_10": {
        "n_samples": 112,
        "horizon": 10,
        "temperature": 0.2,
        "static_exp_weight": 5_000.0,
        "static_exp_gain": 500.0,
        "static_margin": 0.002,
        "static_zone": 0.004,
    },
    "horizon_11": {
        "n_samples": 112,
        "horizon": 11,
        "temperature": 0.2,
        "static_exp_weight": 5_000.0,
        "static_exp_gain": 500.0,
        "static_margin": 0.002,
        "static_zone": 0.004,
    },
}


def summarize(data: dict, centers: np.ndarray) -> dict:
    metrics = compute_metrics(data)
    solve_time = np.asarray(data["solve_t"], dtype=float)
    solve_time = solve_time[np.isfinite(solve_time)]
    contacts = assess_static_obstacles(data["ee"], centers)
    torque = np.asarray(data["u"])
    saturated = np.isclose(
        np.abs(torque), np.asarray(TAU_MAX), rtol=0.0, atol=1e-5
    ).any(axis=1)
    return {
        **metrics,
        "p95_solve_ms": float(np.percentile(solve_time, 95) * 1e3),
        "deadline_miss_pct": float(np.mean(solve_time > 0.020) * 100.0),
        "contact_count": int(sum(item["contact"] for item in contacts)),
        "clearance_mm": [float(item["clearance_mm"]) for item in contacts],
        "saturated_step_pct": float(np.mean(saturated) * 100.0),
    }


def run_benchmark(*, cases: list[str], trajectories: list[str], seeds: list[int],
                  output: Path, save_result: Path | None = None) -> None:
    unknown = set(cases) - CASES.keys()
    if unknown:
        raise ValueError(f"Unknown static benchmark cases: {sorted(unknown)}")
    if save_result and (len(cases) != 1 or len(seeds) != 1 or len(trajectories) != 1):
        raise ValueError("Saving benchmark NPZ requires one trajectory, case, and seed")

    rows = []
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}, "
          f"default_samples={N_SAMPLES}")
    for trajectory in trajectories:
        reference = waypoints(trajectory)
        centers = static_obstacle_centers(reference)
        for seed in seeds:
            for name in cases:
                print(f"\n[TUNE] {trajectory}, {name}, seed={seed}")
                parameters = CASES[name]
                data = run_mppi(
                    reference, has_disturbance=True, use_dob=True,
                    static_centers=centers, seed=seed, **parameters,
                )
                row = {
                    "trajectory": trajectory,
                    "case": name,
                    "seed": seed,
                    "parameters": parameters,
                    "metrics": summarize(data, centers),
                }
                rows.append(row)
                print(json.dumps(row, indent=2))
                if save_result:
                    save_result.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(save_result, **data)
                    print(f"[SAVE] {save_result}")
                del data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    report = {
        "trajectories": trajectories,
        "seeds": seeds,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[SAVE] {output}")
