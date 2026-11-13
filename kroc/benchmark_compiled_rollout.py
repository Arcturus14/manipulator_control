"""Closed-loop speed/quality A/B for baseline and complete-loop compiled MPPI."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT, TIMING_WARMUP_STEPS
from experiment_runner import run_mppi, waypoints


CONDITIONS = {"baseline": False, "compiled_rollout": True}


def metrics(data):
    times = np.asarray(data["solve_t_raw"])[TIMING_WARMUP_STEPS:] * 1e3
    error = np.asarray(data["err_mm"])
    return {
        "tracking_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "tracking_max_mm": float(error.max()),
        "torque_rate_rms_nm_s": float(np.sqrt(np.mean(
            (np.diff(data["u_applied"], axis=0) / DT) ** 2
        ))),
        "solve_mean_ms": float(times.mean()),
        "solve_p50_ms": float(np.percentile(times, 50)),
        "solve_p95_ms": float(np.percentile(times, 95)),
        "solve_max_ms": float(times.max()),
        "deadline_miss_rate": float((times > DT * 1e3).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--steps", type=int, default=250)
    args = parser.parse_args()
    preflight = cuda_preflight()
    root = Path(__file__).resolve().parent
    output = root / "results" / (
        "compiled_rollout_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": vars(args), "cuda": preflight,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True),
        "conditions": CONDITIONS, "trajectory": "circle",
        "controller": "Post-DOB, disturbance off, dynamic margin off",
        "settings": {"samples": 136, "horizon": 10, "temperature": 1.,
                     "temporal_noise_rho": .8},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for seed_index, seed in enumerate(args.seeds):
        names = list(CONDITIONS)
        if seed_index % 2:
            names.reverse()
        for condition in names:
            print(f"RUN {seed=} {condition=}", flush=True)
            data = run_mppi(
                waypoints("circle", n_steps=args.steps),
                has_disturbance=False, use_dob=True,
                dob_compensation_mode="output", require_cuda=True,
                seed=seed, temperature=1., temporal_noise_rho=.8,
                joint_position_limit_weight=50000.,
                joint_velocity_limit_weight=50000.,
                compile_rollout_loop=CONDITIONS[condition],
            )
            row = {"seed": seed, "condition": condition,
                   "metrics": metrics(data)}
            rows.append(row)
            print(json.dumps(row), flush=True)
            np.savez_compressed(
                output / f"seed{seed}_{condition}.npz",
                **{key: value for key, value in data.items()
                   if not key.startswith("_")},
            )
            (output / "metrics.json").write_text(
                json.dumps(rows, indent=2) + "\n"
            )
            del data
            import gc
            gc.collect()
            torch.cuda.empty_cache()
    summary = {}
    for condition in CONDITIONS:
        selected = [row["metrics"] for row in rows
                    if row["condition"] == condition]
        summary[condition] = {
            field: {"mean": float(np.mean([item[field] for item in selected])),
                    "std": (float(np.std([item[field] for item in selected], ddof=1))
                            if len(selected) > 1 else 0.0)}
            for field in selected[0]
        }
    baseline = summary["baseline"]["solve_mean_ms"]["mean"]
    compiled = summary["compiled_rollout"]["solve_mean_ms"]["mean"]
    summary["speedup"] = baseline / compiled
    summary["time_reduction_percent"] = (1 - compiled / baseline) * 100
    (output / "aggregate.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    lines = [
        "# Complete rollout-loop compile A/B", "",
        "| Condition | RMSE [mm] | solve mean [ms] | p95 [ms] | miss rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        entry = summary[condition]
        lines.append(
            f"| {condition} | {entry['tracking_rmse_mm']['mean']:.3f} | "
            f"{entry['solve_mean_ms']['mean']:.3f} | "
            f"{entry['solve_p95_ms']['mean']:.3f} | "
            f"{entry['deadline_miss_rate']['mean']:.3f} |"
        )
    lines += ["", f"Speedup: {summary['speedup']:.3f}x; time reduction: "
              f"{summary['time_reduction_percent']:.2f}%"]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
