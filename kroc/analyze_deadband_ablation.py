"""Compare nominal false activation and disturbance retention across deadbands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import DQ_MAX, DT


def _aggregate(rows: list[dict]) -> dict:
    keys = [key for key, value in rows[0].items()
            if key != "seed" and isinstance(value, (int, float))]
    return {
        key: {
            "mean": float(np.mean([row[key] for row in rows])),
            "min": float(np.min([row[key] for row in rows])),
            "max": float(np.max([row[key] for row in rows])),
        }
        for key in keys
    }


def analyze_directory(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    profile = manifest["acceleration_deadband"]["profile"]
    rows = {"nominal": [], "velocity_limit": []}
    for scenario in rows:
        for path in sorted(root.glob(f"{scenario}_*_directional.npz")):
            seed = int(path.stem.split("_seed", 1)[1].split("_", 1)[0])
            with np.load(path) as data:
                filtered = np.asarray(data["kroc_a_d_hat"])
                raw = np.asarray(data["kroc_a_d_hat_raw"])
                deadband = np.asarray(data["kroc_acceleration_deadband"])
                margin = np.asarray(data["kroc_margin"])
                qd = np.asarray(data["qd"])
                err = np.asarray(data["err_mm"])
            expected = np.sign(raw) * np.maximum(np.abs(raw) - deadband, 0.0)
            np.testing.assert_allclose(filtered, expected, atol=1e-6, rtol=1e-6)
            active = np.abs(filtered) > 1e-6
            velocity_excess = np.maximum(
                margin[:, 1] - 0.05 * (2.0 * DQ_MAX), 0.0
            )
            item = {
                "seed": seed,
                "a_channel_activation_rate": float(active.mean()),
                "a_step_activation_rate": float(np.any(active, axis=1).mean()),
                "velocity_margin_excess_mean_rad_s": float(velocity_excess.mean()),
                "velocity_margin_excess_max_rad_s": float(velocity_excess.max()),
                "margin_variation_l1": float(np.abs(np.diff(margin, axis=0)).sum()),
                "tracking_rmse_mm": float(np.sqrt(np.mean(err ** 2))),
            }
            if scenario == "velocity_limit":
                onset = len(qd) // 4
                wrist_active = active[:, 5]
                detected = np.flatnonzero(wrist_active[onset:])
                wrist_speed = np.abs(qd[:, 5])
                after = wrist_speed[onset:]
                item.update({
                    "wrist_pre_onset_activation_rate": float(
                        wrist_active[:onset].mean()
                    ),
                    "wrist_detection_latency_s": float(detected[0] * DT)
                    if len(detected) else float("inf"),
                    "wrist_filtered_peak_acceleration_rad_s2": float(
                        np.abs(filtered[:, 5]).max()
                    ),
                    "wrist_raw_peak_acceleration_rad_s2": float(
                        np.abs(raw[:, 5]).max()
                    ),
                    "wrist_speed_peak_rad_s": float(wrist_speed.max()),
                    "wrist_soft_zone_steps": int(np.sum(after > 0.9 * DQ_MAX)),
                    "wrist_hard_violation_steps": int(np.sum(after > DQ_MAX)),
                    "wrist_excess_integral": float(
                        np.maximum(after - 0.9 * DQ_MAX, 0).sum() * DT
                    ),
                })
            rows[scenario].append(item)
    return {
        "profile": profile,
        "directory": str(root),
        "runs": rows,
        "aggregate": {name: _aggregate(values) for name, values in rows.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [analyze_directory(path) for path in args.directories]
    payload = {
        "calibration_seeds": [17, 27, 37],
        "validation_seeds": [47, 57, 67],
        "profiles": results,
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "ablation.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        "# Acceleration-deadband ablation", "",
        "Calibration seeds: 17, 27, 37; validation seeds: 47, 57, 67.", "",
        "| Profile | Nominal a-channel active | Nominal velocity-margin excess "
        "[rad/s] | Margin variation L1 | Ramp pre-onset wrist active | "
        "Detection latency [ms] | Wrist peak [rad/s] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        nominal = result["aggregate"]["nominal"]
        ramp = result["aggregate"]["velocity_limit"]
        lines.append(
            f"| {result['profile']} | "
            f"{100 * nominal['a_channel_activation_rate']['mean']:.2f}% | "
            f"{nominal['velocity_margin_excess_mean_rad_s']['mean']:.4f} | "
            f"{nominal['margin_variation_l1']['mean']:.3f} | "
            f"{100 * ramp['wrist_pre_onset_activation_rate']['mean']:.2f}% | "
            f"{1000 * ramp['wrist_detection_latency_s']['mean']:.1f} | "
            f"{ramp['wrist_speed_peak_rad_s']['mean']:.3f} |"
        )
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))
    print("SAVED", args.output)


if __name__ == "__main__":
    main()
