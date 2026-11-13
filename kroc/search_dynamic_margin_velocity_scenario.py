"""Search a recoverable wrist disturbance that exposes dynamic-margin action."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DQ_MAX, DT
from experiment_runner import run_mppi
from kroc.benchmark_noise_aware_state_margin import (
    ACCELERATION_DEADBANDS,
    build_scenario,
)
from kroc.benchmark_state_margin import measure
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker


def joint_ramp_disturbance(steps, amplitude_nm, ramp_steps, hold_steps,
                           joint_index, onset_step=None):
    onset = steps // 4 if onset_step is None else onset_step

    def disturbance(step, device, dtype):
        elapsed = step - onset
        value = 0.0
        if 0 <= elapsed < ramp_steps + hold_steps:
            value = amplitude_nm * min((elapsed + 1) / ramp_steps, 1.0)
        values = torch.zeros(6, device=device, dtype=dtype)
        values[joint_index] = value
        return values

    return disturbance, onset


def extra_metrics(data, onset, joint_index):
    speed = np.abs(np.asarray(data["qd"])[:, joint_index])
    base_margin = 0.05 * (2.0 * DQ_MAX)
    margins = np.asarray(data["kroc_margin"])[:, 1, :, joint_index]
    excess = np.maximum(margins - base_margin, 0.0)
    active = np.any(excess > 1e-5, axis=1)
    peak_index = int(np.argmax(speed))
    after = speed[onset:]
    return {
        "test_joint_speed_peak_rad_s": float(speed[peak_index]),
        "test_joint_speed_peak_step": peak_index,
        "fixed_soft_zone_steps": int(np.sum(after > 0.9 * DQ_MAX)),
        "hard_violation_steps": int(np.sum(after > DQ_MAX)),
        "speed_excess_integral_over_0p9_limit": float(
            np.maximum(after - 0.9 * DQ_MAX, 0.0).sum() * DT
        ),
        "margin_activation_steps": int(active.sum()),
        "first_margin_activation_step": int(np.argmax(active))
        if active.any() else None,
        "maximum_velocity_margin_rad_s": float(margins.max()),
        "margin_active_before_peak": bool(active[:peak_index + 1].any()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitudes", nargs="+", type=float,
                        default=[26.0, 30.0, 34.0, 38.0])
    parser.add_argument("--ramp-steps", nargs="+", type=int,
                        default=[3, 5, 8])
    parser.add_argument("--hold-steps", type=int, default=5)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--joint-index", type=int, default=5,
                        choices=range(6))
    parser.add_argument("--initial-speed", type=float, default=0.0)
    parser.add_argument("--onset-step", type=int)
    parser.add_argument("--acceleration-deadband-profile",
                        choices=ACCELERATION_DEADBANDS,
                        default="nominal_p99")
    args = parser.parse_args()
    if any(value == 0 for value in args.amplitudes):
        parser.error("amplitudes must be nonzero")
    if any(value <= 0 for value in args.ramp_steps) or args.hold_steps < 0:
        parser.error("ramp steps must be positive and hold steps nonnegative")
    if args.onset_step is not None and not 0 <= args.onset_step < args.steps:
        parser.error("onset step must be within the episode")
    preflight = cuda_preflight()

    root = Path(__file__).resolve().parent
    output = root / "results" / (
        "dynamic_margin_velocity_search_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    reference, q0, _, _ = build_scenario("velocity_limit", args.steps)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": vars(args), "cuda": preflight,
        "environment": {
            "sensor_position_std_rad": 1e-3,
            "sensor_velocity_std_rad_s": 1e-2,
            "acceleration_lpf_hz": 10.0,
            "dob_exact_zoh_hz": 8.0,
            "controller": "Post-DOB output compensation",
        },
        "margin": {"fixed_and_min_fraction": 0.05,
                   "max_fraction": 0.10, "reaction_time_s": 0.10,
                   "acceleration_deadband_profile":
                       args.acceleration_deadband_profile,
                   "acceleration_deadband_rad_s2": list(
                       ACCELERATION_DEADBANDS[
                           args.acceleration_deadband_profile])},
        "purpose": "scenario search only; not final performance evidence",
        "disturbed_joint_index": args.joint_index,
        "initial_speed_rad_s": args.initial_speed,
        "onset_step": args.onset_step,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    for amplitude in args.amplitudes:
        for ramp_steps in args.ramp_steps:
            disturbance, onset = joint_ramp_disturbance(
                args.steps, amplitude, ramp_steps, args.hold_steps,
                args.joint_index, args.onset_step,
            )
            for mode in ("fixed", "directional"):
                margin_cfg = MarginConfig(
                    mode=mode, min_fraction=.05, fixed_fraction=.05,
                    max_fraction=.10, reaction_time_s=.10,
                    acceleration_deadband_rad_s2=ACCELERATION_DEADBANDS[
                        args.acceleration_deadband_profile],
                )
                holder = []

                def factory(*factory_args, **factory_kwargs):
                    tracker = NoiseAwareKrocTracker(
                        *factory_args, **factory_kwargs
                    )
                    holder.append(tracker)
                    return tracker

                print(f"RUN {amplitude=} {ramp_steps=} {mode=}", flush=True)
                initial_qd = np.zeros(6)
                initial_qd[args.joint_index] = args.initial_speed
                data = run_mppi(
                    reference, initial_q=q0, initial_qd=initial_qd,
                    has_disturbance=True,
                    use_dob=True, dob_compensation_mode="output",
                    disturbance_function=disturbance, require_cuda=True,
                    seed=args.seed, temperature=1.0, temporal_noise_rho=.8,
                    joint_position_limit_weight=50000.0,
                    joint_velocity_limit_weight=50000.0,
                    tracker_class=factory,
                    tracker_options={
                        "margin_config": asdict(margin_cfg),
                        "sensor_noise": {
                            "position_std_rad": 1e-3,
                            "velocity_std_rad_s": 1e-2,
                            "seed": 30000 + args.seed,
                        },
                        "dob_acceleration_lpf_cutoff_hz": 10.0,
                        "dob_cutoff_hz": 8.0,
                    },
                )
                data.update(holder[0].export_margin_data())
                data.update(holder[0].export_sensor_data())
                data["qd_raw"] = data["qd"].copy()
                metrics = measure(data, 1.0)
                metrics.update(extra_metrics(data, onset, args.joint_index))
                row = {"amplitude_nm": amplitude,
                       "ramp_steps": ramp_steps,
                       "hold_steps": args.hold_steps,
                       "mode": mode, "metrics": metrics}
                rows.append(row)
                stem = (f"joint{args.joint_index}_amp{amplitude:g}_"
                        f"ramp{ramp_steps}_{mode}")
                np.savez_compressed(
                    output / f"{stem}.npz",
                    **{key: value for key, value in data.items()
                       if not key.startswith("_")},
                )
                (output / "metrics.json").write_text(
                    json.dumps(rows, indent=2) + "\n"
                )
                print(json.dumps(row), flush=True)
                del data, holder
                gc.collect()
                torch.cuda.empty_cache()

    pairs = []
    for amplitude in args.amplitudes:
        for ramp_steps in args.ramp_steps:
            selected = {
                row["mode"]: row["metrics"] for row in rows
                if row["amplitude_nm"] == amplitude
                and row["ramp_steps"] == ramp_steps
            }
            fixed, directional = selected["fixed"], selected["directional"]
            pairs.append({
                "amplitude_nm": amplitude, "ramp_steps": ramp_steps,
                "fixed_peak": fixed["test_joint_speed_peak_rad_s"],
                "directional_peak": directional["test_joint_speed_peak_rad_s"],
                "peak_reduction": fixed["test_joint_speed_peak_rad_s"]
                - directional["test_joint_speed_peak_rad_s"],
                "fixed_soft_zone_steps": fixed["fixed_soft_zone_steps"],
                "directional_soft_zone_steps": directional["fixed_soft_zone_steps"],
                "exposure_reduction": fixed["speed_excess_integral_over_0p9_limit"]
                - directional["speed_excess_integral_over_0p9_limit"],
                "fixed_hard_violations": fixed["hard_violation_steps"],
                "directional_hard_violations": directional["hard_violation_steps"],
                "directional_active_before_peak":
                    directional["margin_active_before_peak"],
            })
    eligible = [row for row in pairs
                if row["fixed_soft_zone_steps"] > 0
                and row["directional_active_before_peak"]
                and row["directional_hard_violations"] == 0]
    selected = max(
        eligible,
        key=lambda row: (row["exposure_reduction"], row["peak_reduction"]),
        default=None,
    )
    payload = {"candidates": pairs, "selected": selected,
               "selection_note": "search result requires new-seed validation"}
    (output / "selection.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
