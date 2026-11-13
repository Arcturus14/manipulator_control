"""Plot the already-validated noisy legacy vs filtered DOB circle runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config import DT
from experiment_runner import waypoints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--scenario", default="nominal")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    conditions = {
        "noisy_legacy": "Legacy: accel LPF off, Euler Q-filter",
        "noisy_exact_8hz": "Changed: accel LPF 10 Hz, exact-ZOH Q 8 Hz",
    }
    loaded = {}
    for key in conditions:
        path = args.result_dir / f"{args.scenario}_seed{args.seed}_{key}.npz"
        loaded[key] = np.load(path)

    reference = waypoints("circle", n_steps=len(loaded["noisy_legacy"]["err_mm"]))
    fig = plt.figure(figsize=(13, 4.3))
    ax0 = fig.add_subplot(131, projection="3d")
    ax0.plot(reference[:, 0], reference[:, 1], reference[:, 2], "k--",
             linewidth=2, label="reference")
    colors = ("tab:orange", "tab:blue")
    for color, (key, label) in zip(colors, conditions.items()):
        ee = loaded[key]["ee"]
        ax0.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=color, label=label)
    ax0.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]",
            title=f"Noisy circle tracking (seed {args.seed})")
    ax0.legend(fontsize=7)

    ax1 = fig.add_subplot(132)
    for color, (key, label) in zip(colors, conditions.items()):
        err = loaded[key]["err_mm"]
        ax1.plot(np.arange(len(err)) * DT, err, color=color, label=label)
    ax1.set(xlabel="time [s]", ylabel="EE error [mm]",
            title="Tracking error")
    ax1.grid(alpha=.25)

    ax2 = fig.add_subplot(133)
    for color, (key, label) in zip(colors, conditions.items()):
        d_hat = loaded[key]["d_hat"][1:]
        rms = np.sqrt(np.mean(d_hat ** 2, axis=1))
        ax2.plot(np.arange(len(rms)) * DT, rms, color=color, label=label)
    ax2.set(xlabel="time [s]", ylabel=r"joint RMS $\hat d$ [Nm]",
            title="False DOB estimate (no disturbance)")
    ax2.grid(alpha=.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    metrics = json.loads((args.result_dir / "metrics.json").read_text())
    rows = [row for row in metrics if row["scenario"] == args.scenario
            and row["condition"] in conditions]
    summary = {}
    for key in conditions:
        selected = [row["metrics"] for row in rows if row["condition"] == key]
        summary[key] = {
            field: {"mean": float(np.mean([row[field] for row in selected])),
                    "sample_std": float(np.std([row[field] for row in selected], ddof=1))}
            for field in ("tracking_rmse_mm", "torque_rate_rms_nm_s",
                          "dob_estimation_rmse_nm", "dob_estimate_rms_nm",
                          "solve_mean_ms", "solve_p95_ms")
        }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
