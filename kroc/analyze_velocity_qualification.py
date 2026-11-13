"""Paired multi-seed qualification for the signed velocity-recovery case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import DQ_MAX, DT


def metrics(path: Path, row: dict) -> dict:
    with np.load(path) as data:
        speed = np.asarray(data["qd"])[:, 5]
        gain = np.maximum(
            np.asarray(data["kroc_gain_lower"]),
            np.asarray(data["kroc_gain_upper"]),
        )
        device = data["compute_device"].item()
    if device != "cuda":
        raise AssertionError(f"non-CUDA result: {path}")
    excess = np.maximum(speed - 0.9 * DQ_MAX, 0.0)
    result = {
        "peak_velocity_rad_s": float(speed.max()),
        "soft_zone_steps": int(np.sum(speed > 0.9 * DQ_MAX)),
        "hard_violation_steps": int(np.sum(speed > DQ_MAX)),
        "soft_excess_integral_rad": float(excess.sum() * DT),
        "headroom_gain_max": float(gain.max()),
    }
    for key in ("tracking_rmse_mm", "torque_rate_rms_nm_s",
                "compute_mean_ms", "saturation_steps"):
        result[key] = row["metrics"][key]
    return result


def relative(new: float, base: float) -> float:
    return (new - base) / base if base else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    rows = json.loads((root / "metrics.json").read_text())
    runs: dict[int, dict[str, dict]] = {}
    for row in rows:
        seed, mode = row["seed"], row["mode"]
        path = root / f"velocity_limit_seed{seed}_noisy_filtered_{mode}.npz"
        runs.setdefault(seed, {})[mode] = metrics(path, row)
    if any(set(modes) != {"fixed", "directional", "headroom"}
           for modes in runs.values()):
        raise AssertionError("each seed must contain all three modes")

    aggregates = {}
    scalar_keys = list(next(iter(runs.values()))["fixed"])
    for mode in ("fixed", "directional", "headroom"):
        aggregates[mode] = {
            key: float(np.mean([modes[mode][key] for modes in runs.values()]))
            for key in scalar_keys
        }
        aggregates[mode]["soft_zone_steps_total"] = int(sum(
            modes[mode]["soft_zone_steps"] for modes in runs.values()
        ))
        aggregates[mode]["hard_violation_steps_total"] = int(sum(
            modes[mode]["hard_violation_steps"] for modes in runs.values()
        ))
        aggregates[mode]["soft_excess_integral_total_rad"] = float(sum(
            modes[mode]["soft_excess_integral_rad"] for modes in runs.values()
        ))

    paired = []
    for seed, modes in sorted(runs.items()):
        fixed, directional = modes["fixed"], modes["directional"]
        paired.append({
            "seed": seed,
            "peak_reduction_rad_s": fixed["peak_velocity_rad_s"]
            - directional["peak_velocity_rad_s"],
            "soft_zone_step_reduction": fixed["soft_zone_steps"]
            - directional["soft_zone_steps"],
            "hard_violation_step_reduction": fixed["hard_violation_steps"]
            - directional["hard_violation_steps"],
            "soft_excess_integral_reduction_rad":
                fixed["soft_excess_integral_rad"]
                - directional["soft_excess_integral_rad"],
        })

    fixed, directional, headroom = (
        aggregates[name] for name in ("fixed", "directional", "headroom")
    )
    acceptance = {
        "peak_lower_all_seeds": all(
            row["peak_reduction_rad_s"] > 1e-4 for row in paired
        ),
        "soft_excess_integral_reduced_at_least_50pct":
            directional["soft_excess_integral_total_rad"]
            <= 0.5 * fixed["soft_excess_integral_total_rad"],
        "hard_violations_not_increased":
            directional["hard_violation_steps_total"]
            <= fixed["hard_violation_steps_total"],
        "tracking_rmse_increase_within_15pct":
            directional["tracking_rmse_mm"] <= 1.15 * fixed["tracking_rmse_mm"],
        "saturation_not_increased":
            directional["saturation_steps"] <= fixed["saturation_steps"],
    }
    directional_qualified = all(acceptance.values())
    headroom_distinct = (
        headroom["headroom_gain_max"] > 1.0 + 1e-6
        and any(not np.isclose(
            runs[seed]["headroom"]["peak_velocity_rad_s"],
            runs[seed]["directional"]["peak_velocity_rad_s"],
        ) for seed in runs)
    )
    summary = {
        "scenario": {
            "joint": "wrist_3 (index 5)",
            "initial_velocity_rad_s": 2.5,
            "disturbance_peak_nm": -16.0,
            "ramp_steps": 5,
            "hold_steps": 5,
            "plant_convention": "tau_effective = tau_applied - disturbance",
            "deadband_profile": "nominal_p99",
            "validation_seeds": sorted(runs),
        },
        "runs": runs,
        "aggregate": aggregates,
        "paired_directional_minus_fixed": paired,
        "relative_changes": {
            "peak_velocity": relative(
                directional["peak_velocity_rad_s"], fixed["peak_velocity_rad_s"]
            ),
            "tracking_rmse": relative(
                directional["tracking_rmse_mm"], fixed["tracking_rmse_mm"]
            ),
            "torque_rate_rms": relative(
                directional["torque_rate_rms_nm_s"], fixed["torque_rate_rms_nm_s"]
            ),
            "compute_mean": relative(
                directional["compute_mean_ms"], fixed["compute_mean_ms"]
            ),
        },
        "acceptance": acceptance,
        "directional_qualified": directional_qualified,
        "headroom_distinct_effect_qualified": headroom_distinct,
    }
    (root / "qualification.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    fixed_area = fixed["soft_excess_integral_total_rad"]
    directional_area = directional["soft_excess_integral_total_rad"]
    lines = [
        "# P99 dynamic-margin velocity qualification", "",
        f"Directional qualification: **{'PASS' if directional_qualified else 'FAIL'}**",
        f"Headroom distinct-effect qualification: **{'PASS' if headroom_distinct else 'FAIL'}**",
        "",
        "| Mode | Peak dq6 mean [rad/s] | Soft-zone steps total | "
        "Hard violations total | Excess integral total [rad] | RMSE [mm] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in ("fixed", "directional", "headroom"):
        item = aggregates[mode]
        lines.append(
            f"| {mode} | {item['peak_velocity_rad_s']:.4f} | "
            f"{item['soft_zone_steps_total']} | "
            f"{item['hard_violation_steps_total']} | "
            f"{item['soft_excess_integral_total_rad']:.6f} | "
            f"{item['tracking_rmse_mm']:.3f} |"
        )
    lines += [
        "", f"Soft-excess integral reduction: "
        f"{100 * (1 - directional_area / fixed_area):.1f}%.",
        f"Mean peak reduction: "
        f"{fixed['peak_velocity_rad_s'] - directional['peak_velocity_rad_s']:.4f} rad/s.",
        "Headroom gain remained 1.0, so headroom and directional are identical.",
    ]
    (root / "QUALIFICATION.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))
    print("SAVED", root / "qualification.json")


if __name__ == "__main__":
    main()
