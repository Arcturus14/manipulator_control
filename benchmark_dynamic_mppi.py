"""Benchmark MPPI against the fast reference-chasing dynamic cube."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from config import HORIZON_OBS, TAU_MAX
from run_comparison import run_mppi, waypoints
from utils.trajectory import chasing_obstacle_paths
from visualization.plotters import assess_dynamic_obstacle, compute_metrics


def summarize(data: dict, path: np.ndarray) -> dict:
    metrics = compute_metrics(data)
    assessment = assess_dynamic_obstacle(data["ee"], path)
    solve_time = np.asarray(data["solve_t"], dtype=float)
    solve_time = solve_time[np.isfinite(solve_time)]
    saturated = np.isclose(
        np.abs(np.asarray(data["u"])), np.asarray(TAU_MAX),
        rtol=0.0, atol=1e-5,
    ).any(axis=1)
    return {
        **metrics,
        **assessment,
        "p95_solve_ms": float(np.percentile(solve_time, 95) * 1e3),
        "deadline_miss_pct": float(np.mean(solve_time > 0.020) * 100.0),
        "saturated_step_pct": float(np.mean(saturated) * 100.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", type=float, required=True)
    parser.add_argument("--trajectories", nargs="+", default=["infinity"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}")
    for trajectory in args.trajectories:
        reference = waypoints(trajectory)
        paths = chasing_obstacle_paths(reference, HORIZON_OBS + 2)
        for seed in args.seeds:
            for weight in args.weights:
                print(f"\n[TUNE] {trajectory}, weight={weight:g}, seed={seed}")
                data = run_mppi(
                    reference, has_disturbance=True, use_dob=True,
                    dynamic_paths=paths, seed=seed,
                    dynamic_exp_weight=weight,
                )
                row = {
                    "trajectory": trajectory,
                    "weight": weight,
                    "seed": seed,
                    "metrics": summarize(data, paths[0]),
                }
                rows.append(row)
                print("DYNAMIC_MPPI_ROW " + json.dumps(row))
                del data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    report = {
        "trajectories": args.trajectories,
        "weights": args.weights,
        "seeds": args.seeds,
        "rows": rows,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[SAVE] {args.output}")


if __name__ == "__main__":
    main()
