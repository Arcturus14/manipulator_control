"""Paired multi-seed benchmark for baseline and tuned MPPI parameters."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from config import DT, MPPI_W_POS, MPPI_W_TERM, N_SAMPLES, TAU_MAX
from run_comparison import run_mppi, waypoints


CASES = {
    "baseline": {
        "temperature": 0.45,
        "position_weight": 120_000.0,
        "terminal_weight": 300_000.0,
    },
    "tuned": {
        "temperature": 0.2,
        "position_weight": MPPI_W_POS,
        "terminal_weight": MPPI_W_TERM,
    },
}


def metrics(data: dict) -> dict[str, float]:
    error = np.asarray(data["err_mm"], dtype=float)
    solve_time = np.asarray(data["solve_t"], dtype=float)
    solve_time = solve_time[np.isfinite(solve_time)]
    torque = np.asarray(data["u"], dtype=float)
    limits = np.asarray(TAU_MAX, dtype=float)
    saturated_steps = np.isclose(
        np.abs(torque), limits, rtol=0.0, atol=1e-5
    ).any(axis=1)
    return {
        "mean_mm": float(error.mean()),
        "rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "p95_mm": float(np.percentile(error, 95)),
        "max_mm": float(error.max()),
        "mean_solve_ms": float(solve_time.mean() * 1e3),
        "worst_solve_ms": float(solve_time.max() * 1e3),
        "deadline_miss_pct": float(np.mean(solve_time > DT) * 100),
        "saturated_step_pct": float(saturated_steps.mean() * 100),
    }


def summarize(rows: list[dict], cases: list[str]) -> dict[str, dict[str, float]]:
    result = {}
    for case in cases:
        selected = [row for row in rows if row["case"] == case]
        fields = selected[0]["metrics"]
        result[case] = {}
        for field in fields:
            values = np.asarray([row["metrics"][field] for row in selected])
            result[case][f"{field}_mean"] = float(values.mean())
            result[case][f"{field}_std"] = float(
                values.std(ddof=1 if len(values) > 1 else 0)
            )
    return result


def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[7, 17, 27, 37, 47])
    parser.add_argument("--trajectory", default="infinity")
    parser.add_argument("--cases", nargs="+", choices=CASES,
                        default=["baseline", "tuned"])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = command_line()
    reference = waypoints(args.trajectory)
    rows = []
    for seed in args.seeds:
        for case in args.cases:
            parameters = CASES[case]
            print(f"\n[BENCHMARK] case={case}, seed={seed}")
            data = run_mppi(
                reference, has_disturbance=False, use_dob=False,
                seed=seed, n_samples=N_SAMPLES, **parameters,
            )
            row = {"case": case, "seed": seed, "metrics": metrics(data)}
            rows.append(row)
            print(json.dumps(row, indent=2))
            del data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary = summarize(rows, args.cases)
    report = {
        "trajectory": args.trajectory,
        "seeds": args.seeds,
        "cases": {case: CASES[case] for case in args.cases},
        "rows": rows,
        "summary": summary,
    }
    if {"baseline", "tuned"}.issubset(summary):
        baseline_rmse = summary["baseline"]["rmse_mm_mean"]
        tuned_rmse = summary["tuned"]["rmse_mm_mean"]
        report["rmse_improvement_pct"] = 100 * (
            1 - tuned_rmse / baseline_rmse
        )
    print("\n[MULTI-SEED SUMMARY]")
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n",
                               encoding="utf-8")
        print(f"[SAVE] {args.output}")


if __name__ == "__main__":
    main()
