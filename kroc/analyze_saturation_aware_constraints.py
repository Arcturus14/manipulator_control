"""Aggregate and plot the final saturation-aware constraint comparison."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DT, Q_MAX, TAU_MAX


FIELDS = (
    "minimum_upper_distance_rad", "hard_violation_steps",
    "hard_violation_integral_rad_s", "test_joint_saturation_steps",
    "requested_torque_excess_max_nm",
    "requested_torque_excess_integral_nm_s", "headroom_gain_max",
    "margin_max_rad", "tracking_rmse_mm", "compute_mean_ms",
)


def stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)),
        "min": float(values.min()), "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    rows = json.loads((args.result_dir / "metrics.json").read_text())
    cases = sorted({row["case"] for row in rows})
    grouped = {
        case: {row["seed"]: row for row in rows if row["case"] == case}
        for case in cases
    }
    result = {"seed_count": len({row["seed"] for row in rows}), "groups": {}}
    joint, onset = 5, 10
    arrays = {}
    for case in cases:
        selected = list(grouped[case].values())
        files = [args.result_dir / f"seed{row['seed']}_{case}.npz"
                 for row in selected]
        loaded = [np.load(path) for path in files]
        arrays[case] = loaded
        result["groups"][case] = {
            field: stats([row["metrics"][field] for row in selected])
            for field in FIELDS
        }
        upper_margin = [data["kroc_margin"][:, 0, 1, joint]
                        for data in loaded]
        q = [data["q"][:, joint] for data in loaded]
        result["groups"][case]["diagnostics"] = {
            "pre_disturbance_margin_mean_rad": float(np.mean(
                [value[:onset].mean() for value in upper_margin]
            )),
            "risk_window_margin_mean_rad": float(np.mean(
                [value[onset:46].mean() for value in upper_margin]
            )),
            "final_margin_mean_rad": float(np.mean(
                [value[-1] for value in upper_margin]
            )),
            "position_at_onset_mean_rad": float(np.mean(
                [value[onset] for value in q]
            )),
            "first_saturation_step_mean": float(np.mean([
                row["metrics"]["first_test_joint_saturation_step"]
                for row in selected
            ])),
            "peak_step_mean": float(np.mean([
                row["metrics"]["peak_step"] for row in selected
            ])),
        }

    comparisons = (
        ("dynamic_minus_basic", "dynamic_headroom_0_to_10pct", "basic_soft"),
        ("headroom_minus_directional", "dynamic_headroom_0_to_10pct",
         "directional_ablation_0_to_10pct"),
        ("fixed5_minus_dynamic", "fixed_small_5pct",
         "dynamic_headroom_0_to_10pct"),
    )
    result["paired"] = {}
    for label, first, second in comparisons:
        seeds = sorted(set(grouped[first]) & set(grouped[second]))
        result["paired"][label] = {
            field: stats([
                grouped[first][seed]["metrics"][field]
                - grouped[second][seed]["metrics"][field]
                for seed in seeds
            ]) for field in FIELDS
        }
    (args.result_dir / "analysis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )

    main_cases = (
        "basic_soft", "fixed_small_5pct", "fixed_conservative_10pct",
        "dynamic_headroom_0_to_10pct",
    )
    labels = {
        "basic_soft": "basic soft (0%)",
        "fixed_small_5pct": "fixed 5%",
        "fixed_conservative_10pct": "fixed 10%",
        "dynamic_headroom_0_to_10pct": "dynamic 0-10%",
    }
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for case in main_cases:
        q = np.stack([data["q"][:, joint] for data in arrays[case]])
        margin = np.stack([
            data["kroc_margin"][:, 0, 1, joint] for data in arrays[case]
        ])
        command = np.stack([
            data["u_mppi"][:, joint] + data["d_hat_used"][:, joint]
            for data in arrays[case]
        ])
        time_q = np.arange(q.shape[1]) * DT
        time_u = np.arange(command.shape[1]) * DT
        axes[0].plot(time_q, q.mean(0), label=labels[case])
        axes[0].fill_between(time_q, q.mean(0) - q.std(0),
                             q.mean(0) + q.std(0), alpha=.12)
        axes[1].plot(time_u, margin.mean(0), label=labels[case])
        axes[2].plot(time_u, command.mean(0), label=labels[case])
    axes[0].axhline(Q_MAX[joint], color="black", linestyle="--",
                    label="physical upper limit")
    axes[2].axhline(-TAU_MAX[joint], color="black", linestyle="--",
                    label="negative torque limit")
    for axis in axes:
        axis.axvline(onset * DT, color="grey", linestyle=":")
        axis.grid(alpha=.25)
        axis.legend(ncol=2, fontsize=8)
    axes[0].set_ylabel("wrist-3 q [rad]")
    axes[1].set_ylabel("upper margin [rad]")
    axes[2].set_ylabel("requested torque [Nm]")
    axes[2].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(args.result_dir / "comparison.png", dpi=180)
    print(args.result_dir / "analysis.json")
    print(args.result_dir / "comparison.png")


if __name__ == "__main__":
    main()
