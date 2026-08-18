"""Evaluate MPPI parameter candidates on reference-indexed static cubes."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from config import N_SAMPLES, TAU_MAX
from run_comparison import run_mppi, waypoints
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=CASES,
                        default=list(CASES))
    parser.add_argument("--trajectories", nargs="+", default=["infinity"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--save-result", type=Path,
                        help="save NPZ data; requires exactly one case and seed")
    args = parser.parse_args()
    if args.save_result and (len(args.cases) != 1 or len(args.seeds) != 1
                             or len(args.trajectories) != 1):
        parser.error(
            "--save-result requires exactly one trajectory, case, and seed"
        )

    rows = []
    print(f"device={'cuda' if torch.cuda.is_available() else 'cpu'}, "
          f"default_samples={N_SAMPLES}")
    for trajectory in args.trajectories:
        reference = waypoints(trajectory)
        centers = static_obstacle_centers(reference)
        for seed in args.seeds:
            for name in args.cases:
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
                if args.save_result:
                    args.save_result.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(args.save_result, **data)
                    print(f"[SAVE] {args.save_result}")
                del data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    report = {
        "trajectories": args.trajectories,
        "seeds": args.seeds,
        "rows": rows,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"[SAVE] {args.output}")


if __name__ == "__main__":
    main()
