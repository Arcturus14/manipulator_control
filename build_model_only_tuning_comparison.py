"""Combine validated tuned model-only metrics with existing paired baselines."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


VALIDATION = Path("results/model_only_validation_gpu/dbd36d07613f/metrics.json")
BASELINE = Path("results/model_only_rollout_gpu/c1477a74e82b/metrics.json")
OUTPUT = VALIDATION.parent
PROFILES = ("constant", "step", "slow_sinusoidal")
SEEDS = (7, 17, 27)


def main() -> None:
    validation = json.loads(VALIDATION.read_text())
    baseline = json.loads(BASELINE.read_text())
    rows = []
    for row in validation["rows"]:
        if row["case"] != "stable_r08_lambda1":
            continue
        rows.append({
            "profile": row["profile"], "seed": row["seed"],
            "method": "model_only_tuned",
            **row["metrics"],
        })
    for row in baseline["rows"]:
        if row["profile"] not in PROFILES or row["mode"] not in {
                "model_only", "post_dob"}:
            continue
        rows.append({
            "profile": row["profile"], "seed": row["seed"],
            "method": ("model_only_original" if row["mode"] == "model_only"
                       else "post_dob_default"),
            **row["metrics"],
        })
    fields = sorted({key for row in rows for key in row})
    with (OUTPUT / "post_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {}
    methods = ("model_only_original", "model_only_tuned", "post_dob_default")
    for profile in PROFILES:
        summary[profile] = {}
        for method in methods:
            selected = [row for row in rows
                        if row["profile"] == profile and row["method"] == method]
            values = np.asarray([row["tracking_rmse_mm"] for row in selected])
            torque_rate = np.asarray([
                row["torque_rate_rms_nm_per_step"] for row in selected
            ])
            summary[profile][method] = {
                "tracking_rmse_mm_mean": float(values.mean()),
                "tracking_rmse_mm_std": float(values.std(ddof=1)),
                "torque_rate_rms_nm_per_step_mean": float(torque_rate.mean()),
                "runs": len(selected),
            }
        original = summary[profile]["model_only_original"]["tracking_rmse_mm_mean"]
        tuned = summary[profile]["model_only_tuned"]["tracking_rmse_mm_mean"]
        post = summary[profile]["post_dob_default"]["tracking_rmse_mm_mean"]
        summary[profile]["comparison"] = {
            "tuned_vs_original_reduction_pct": 100.0 * (1.0 - tuned / original),
            "tuned_vs_post_increase_pct": 100.0 * (tuned / post - 1.0),
        }
    (OUTPUT / "post_comparison.json").write_text(json.dumps({
        "validation_source": str(VALIDATION),
        "baseline_source": str(BASELINE),
        "note": ("Post-DOB/default and original model-only are reused from the "
                 "existing same-profile, same-seed K=136 H=10 GPU experiment."),
        "summary": summary,
    }, indent=2))

    x = np.arange(len(PROFILES))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for index, method in enumerate(methods):
        means = [summary[p][method]["tracking_rmse_mm_mean"] for p in PROFILES]
        stds = [summary[p][method]["tracking_rmse_mm_std"] for p in PROFILES]
        axes[0].bar(x + (index-1)*width, means, width, yerr=stds,
                    capsize=3, label=method)
        torque = [summary[p][method]["torque_rate_rms_nm_per_step_mean"]
                  for p in PROFILES]
        axes[1].bar(x + (index-1)*width, torque, width, label=method)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Tracking RMSE [mm, log scale]")
    axes[1].set_ylabel("Torque-rate RMS [N m/step]")
    for axis in axes:
        axis.set_xticks(x, PROFILES, rotation=15)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT / "post_comparison.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print("SAVED", OUTPUT)


if __name__ == "__main__":
    main()
