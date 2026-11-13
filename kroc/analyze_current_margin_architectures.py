"""Aggregate paired Post-DOB/model-only dynamic-margin benchmark results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


FIELDS = (
    "elbow_upper_distance_rad",
    "wrist3_peak_velocity_rad_s",
    "wrist3_velocity_excess_area",
    "velocity_violation_steps",
    "tracking_rmse_mm",
    "compute_mean_ms",
    "compute_p95_ms",
    "dob_estimation_rmse_nm",
)


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.result_dir / "metrics.json").read_text())
    scenarios = sorted({row["scenario"] for row in rows})
    architectures = ("post_dob", "model_only")
    modes = ("fixed", "directional")
    result = {"n_seeds": len({row["seed"] for row in rows}), "groups": {}}

    for scenario in scenarios:
        result["groups"][scenario] = {}
        for architecture in architectures:
            result["groups"][scenario][architecture] = {}
            groups = {}
            for mode in modes:
                selected = [row for row in rows
                            if row["scenario"] == scenario
                            and row["architecture"] == architecture
                            and row["mode"] == mode]
                groups[mode] = {row["seed"]: row["metrics"] for row in selected}
                result["groups"][scenario][architecture][mode] = {
                    field: stats([row["metrics"][field] for row in selected])
                    for field in FIELDS
                }
            seeds = sorted(set(groups["fixed"]) & set(groups["directional"]))
            result["groups"][scenario][architecture]["paired_directional_minus_fixed"] = {
                field: stats([
                    groups["directional"][seed][field]
                    - groups["fixed"][seed][field]
                    for seed in seeds
                ]) for field in FIELDS
            }

        for mode in modes:
            grouped = {
                architecture: {
                    row["seed"]: row["metrics"] for row in rows
                    if row["scenario"] == scenario
                    and row["architecture"] == architecture
                    and row["mode"] == mode
                } for architecture in architectures
            }
            seeds = sorted(set(grouped["post_dob"]) & set(grouped["model_only"]))
            result["groups"][scenario][f"paired_model_only_minus_post_dob_{mode}"] = {
                field: stats([
                    grouped["model_only"][seed][field]
                    - grouped["post_dob"][seed][field]
                    for seed in seeds
                ]) for field in FIELDS
            }

    destination = args.result_dir / "analysis.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
