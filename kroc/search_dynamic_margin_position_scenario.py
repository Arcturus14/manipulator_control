"""Search a signed wrist position-limit scenario for dynamic-margin action."""
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
from config import DT, Q_MAX, Q_MIN
from experiment_runner import run_mppi
from kinematics.ur5_kinematics import fk_joints_np
from kroc.benchmark_noise_aware_state_margin import ACCELERATION_DEADBANDS
from kroc.benchmark_state_margin import measure
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker


def joint_ramp_disturbance(steps, amplitude_nm, ramp_steps, hold_steps,
                           joint_index, onset_step):
    def disturbance(step, device, dtype):
        elapsed = step - onset_step
        value = 0.0
        if 0 <= elapsed < ramp_steps + hold_steps:
            value = amplitude_nm * min((elapsed + 1) / ramp_steps, 1.0)
        values = torch.zeros(6, device=device, dtype=dtype)
        values[joint_index] = value
        return values

    return disturbance


def static_reference(steps, joint_index, initial_position):
    q0 = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0[joint_index] = initial_position
    point = fk_joints_np(q0)[-1]
    return np.repeat(point[None, :], steps, axis=0), q0


def extra_metrics(data, onset, joint_index):
    q = np.asarray(data["q"])[:, joint_index]
    upper = Q_MAX[joint_index]
    span = Q_MAX[joint_index] - Q_MIN[joint_index]
    fixed_margin = 0.05 * span
    fixed_soft_upper = upper - fixed_margin
    margins = np.asarray(data["kroc_margin"])[:, 0, 1, joint_index]
    excess_margin = np.maximum(margins - fixed_margin, 0.0)
    active = excess_margin > 1e-5
    peak_index = int(np.argmax(q))
    after = q[onset:]
    return {
        "test_joint_position_peak_rad": float(q[peak_index]),
        "test_joint_position_peak_step": peak_index,
        "physical_upper_distance_min_rad": float(upper - q.max()),
        "fixed_soft_zone_steps": int(np.sum(after > fixed_soft_upper)),
        "hard_violation_steps": int(np.sum(after > upper)),
        "position_excess_integral_over_fixed_soft_limit_rad_s": float(
            np.maximum(after - fixed_soft_upper, 0.0).sum() * DT
        ),
        "margin_activation_steps": int(active.sum()),
        "first_margin_activation_step": int(np.argmax(active))
        if active.any() else None,
        "maximum_upper_position_margin_rad": float(margins.max()),
        "margin_active_before_peak": bool(active[:peak_index + 1].any()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-fractions", nargs="+", type=float,
                        default=[0.84, 0.86, 0.88])
    parser.add_argument("--initial-speeds", nargs="+", type=float,
                        default=[0.0, 0.4])
    parser.add_argument("--amplitudes", nargs="+", type=float,
                        default=[-16.0, -20.0, -24.0])
    parser.add_argument("--ramp-steps", nargs="+", type=int,
                        default=[3, 5])
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--joint-index", type=int, default=5,
                        choices=range(6))
    parser.add_argument("--onset-step", type=int, default=5)
    parser.add_argument("--acceleration-deadband-profile",
                        choices=ACCELERATION_DEADBANDS,
                        default="nominal_p99")
    parser.add_argument("--position-gain", type=float, default=1.0)
    args = parser.parse_args()
    if any(not 0.0 < value < 1.0 for value in args.initial_fractions):
        parser.error("initial fractions must be in (0, 1)")
    if any(value >= 0 for value in args.amplitudes):
        parser.error("upper-limit search needs negative disturbance values")
    if args.position_gain <= 0:
        parser.error("position gain must be positive")
    if any(value <= 0 for value in args.ramp_steps) or args.hold_steps < 0:
        parser.error("ramp steps must be positive and hold steps nonnegative")
    if not 0 <= args.onset_step < args.steps:
        parser.error("onset step must be within the episode")
    preflight = cuda_preflight()

    root = Path(__file__).resolve().parent
    output = root / "results" / (
        "dynamic_margin_position_search_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
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
        "margin": {
            "fixed_and_min_fraction": 0.05,
            "max_fraction": 0.10,
            "reaction_time_s": 0.10,
            "position_gain": args.position_gain,
            "acceleration_deadband_profile": args.acceleration_deadband_profile,
            "acceleration_deadband_rad_s2": list(
                ACCELERATION_DEADBANDS[args.acceleration_deadband_profile]),
        },
        "plant_convention": "tau_effective = tau_applied - disturbance",
        "purpose": "scenario search only; final evidence requires new seeds",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("OUTPUT", output, flush=True)

    rows = []
    for initial_fraction in args.initial_fractions:
        initial_position = initial_fraction * Q_MAX[args.joint_index]
        reference, q0 = static_reference(
            args.steps, args.joint_index, initial_position
        )
        for initial_speed in args.initial_speeds:
            initial_qd = np.zeros(6)
            initial_qd[args.joint_index] = initial_speed
            for amplitude in args.amplitudes:
                for ramp_steps in args.ramp_steps:
                    disturbance = joint_ramp_disturbance(
                        args.steps, amplitude, ramp_steps, args.hold_steps,
                        args.joint_index, args.onset_step,
                    )
                    for mode in ("fixed", "directional"):
                        margin_cfg = MarginConfig(
                            mode=mode, min_fraction=.05,
                            fixed_fraction=.05, max_fraction=.10,
                            reaction_time_s=.10,
                            position_gain=args.position_gain,
                            acceleration_deadband_rad_s2=
                                ACCELERATION_DEADBANDS[
                                    args.acceleration_deadband_profile],
                        )
                        holder = []

                        def factory(*factory_args, **factory_kwargs):
                            tracker = NoiseAwareKrocTracker(
                                *factory_args, **factory_kwargs
                            )
                            holder.append(tracker)
                            return tracker

                        print(
                            f"RUN {initial_fraction=} {initial_speed=} "
                            f"{amplitude=} {ramp_steps=} {mode=}", flush=True
                        )
                        data = run_mppi(
                            reference, initial_q=q0, initial_qd=initial_qd,
                            has_disturbance=True, use_dob=True,
                            dob_compensation_mode="output",
                            disturbance_function=disturbance,
                            require_cuda=True, seed=args.seed,
                            temperature=1.0, temporal_noise_rho=.8,
                            joint_position_limit_weight=50000.0,
                            joint_velocity_limit_weight=50000.0,
                            tracker_class=factory,
                            tracker_options={
                                "margin_config": asdict(margin_cfg),
                                "sensor_noise": {
                                    "position_std_rad": 1e-3,
                                    "velocity_std_rad_s": 1e-2,
                                    "seed": 40000 + args.seed,
                                },
                                "dob_acceleration_lpf_cutoff_hz": 10.0,
                                "dob_cutoff_hz": 8.0,
                            },
                        )
                        data.update(holder[0].export_margin_data())
                        data.update(holder[0].export_sensor_data())
                        data["qd_raw"] = data["qd"].copy()
                        metrics = measure(data, 1.0)
                        metrics.update(extra_metrics(
                            data, args.onset_step, args.joint_index
                        ))
                        row = {
                            "initial_fraction_of_positive_limit":
                                initial_fraction,
                            "initial_position_rad": initial_position,
                            "initial_speed_rad_s": initial_speed,
                            "amplitude_nm": amplitude,
                            "ramp_steps": ramp_steps,
                            "hold_steps": args.hold_steps,
                            "mode": mode,
                            "metrics": metrics,
                        }
                        rows.append(row)
                        stem = (
                            f"joint{args.joint_index}_q{initial_fraction:g}_"
                            f"dq{initial_speed:g}_amp{amplitude:g}_"
                            f"ramp{ramp_steps}_{mode}"
                        )
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

    keys = (
        "initial_fraction_of_positive_limit", "initial_speed_rad_s",
        "amplitude_nm", "ramp_steps",
    )
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, {})[row["mode"]] = row["metrics"]
    pairs = []
    for key, modes in groups.items():
        fixed, directional = modes["fixed"], modes["directional"]
        pairs.append({
            **dict(zip(keys, key)),
            "fixed_peak_rad": fixed["test_joint_position_peak_rad"],
            "directional_peak_rad":
                directional["test_joint_position_peak_rad"],
            "peak_reduction_rad":
                fixed["test_joint_position_peak_rad"]
                - directional["test_joint_position_peak_rad"],
            "fixed_soft_zone_steps": fixed["fixed_soft_zone_steps"],
            "directional_soft_zone_steps":
                directional["fixed_soft_zone_steps"],
            "fixed_hard_violations": fixed["hard_violation_steps"],
            "directional_hard_violations":
                directional["hard_violation_steps"],
            "exposure_reduction":
                fixed["position_excess_integral_over_fixed_soft_limit_rad_s"]
                - directional[
                    "position_excess_integral_over_fixed_soft_limit_rad_s"],
            "directional_active_before_peak":
                directional["margin_active_before_peak"],
            "directional_rmse_mm": directional["tracking_rmse_mm"],
        })
    eligible = [
        row for row in pairs
        if row["fixed_hard_violations"] > 0
        and row["directional_hard_violations"] == 0
        and row["directional_active_before_peak"]
    ]
    selected = max(
        eligible,
        key=lambda row: (row["fixed_hard_violations"],
                         row["exposure_reduction"],
                         row["peak_reduction_rad"]),
        default=None,
    )
    payload = {
        "candidates": pairs, "selected": selected,
        "selection_note": (
            "Candidate selection uses one search seed; validate with disjoint "
            "seeds before treating it as evidence."
        ),
    }
    (output / "selection.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    print(json.dumps(payload, indent=2), flush=True)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
