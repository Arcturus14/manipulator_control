"""Search a recoverable position-limit case for zero-floor headroom margins."""
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
from config import DT, Q_MAX, Q_MIN, TAU_MAX
from experiment_runner import run_mppi
from kinematics.ur5_kinematics import fk_joints_np
from kroc.benchmark_noise_aware_state_margin import ACCELERATION_DEADBANDS
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker


CASES = {
    "basic_soft": {"mode": "fixed", "fixed_fraction": 0.0},
    "dynamic_headroom": {"mode": "headroom", "fixed_fraction": 0.0},
}


def reference_and_initial(steps, joint, fraction):
    q0 = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0[joint] = fraction * Q_MAX[joint]
    target = fk_joints_np(q0)[-1]
    return np.repeat(target[None, :], steps, axis=0), q0


def ramp_pulse(amplitude, ramp_steps, hold_steps, onset, joint):
    def disturbance(step, device, dtype):
        elapsed = step - onset
        value = 0.0
        if 0 <= elapsed < ramp_steps + hold_steps:
            value = amplitude * min((elapsed + 1) / ramp_steps, 1.0)
        result = torch.zeros(6, device=device, dtype=dtype)
        result[joint] = value
        return result
    return disturbance


def metrics(data, joint, onset):
    q = np.asarray(data["q"])[:, joint]
    upper = Q_MAX[joint]
    commanded = np.asarray(data["u_mppi"]) + np.asarray(data["d_hat_used"])
    limit = np.asarray(data["physical_torque_limit"])
    requested_excess = np.maximum(np.abs(commanded) - limit, 0.0)
    margins = np.asarray(data["kroc_margin"])[:, 0, 1, joint]
    gain = np.asarray(data["kroc_gain_upper"])[:, joint]
    active = margins > 1e-5
    peak_step = int(np.argmax(q))
    return {
        "peak_position_rad": float(q[peak_step]),
        "peak_step": peak_step,
        "minimum_upper_distance_rad": float(upper - q.max()),
        "hard_violation_steps": int(np.sum(q > upper)),
        "hard_violation_integral_rad_s": float(
            np.maximum(q - upper, 0.0).sum() * DT
        ),
        "saturation_steps": int(np.asarray(data["actuator_saturated"]).sum()),
        "test_joint_saturation_steps": int(np.sum(
            np.isclose(np.abs(np.asarray(data["u_applied"])[:, joint]),
                       limit[joint], atol=1e-5, rtol=0.0)
        )),
        "requested_torque_excess_max_nm": float(requested_excess[:, joint].max()),
        "requested_torque_excess_integral_nm_s": float(
            requested_excess[:, joint].sum() * DT
        ),
        "minimum_directional_headroom_nm": float(
            np.asarray(data["kroc_headroom_negative"])[:, joint].min()
        ),
        "headroom_gain_max": float(gain.max()),
        "headroom_gain_active_steps": int(np.sum(gain > 1.0 + 1e-6)),
        "margin_max_rad": float(margins.max()),
        "margin_active_steps": int(active.sum()),
        "margin_first_active_step": int(np.argmax(active)) if active.any() else None,
        "margin_active_before_peak": bool(active[:max(peak_step, 1)].any()),
        "tracking_rmse_mm": float(np.sqrt(np.mean(np.asarray(data["err_mm"]) ** 2))),
        "post_onset_peak_rad": float(q[onset:].max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-fractions", nargs="+", type=float,
                        default=[0.82, 0.85, 0.88])
    parser.add_argument("--amplitudes", nargs="+", type=float,
                        default=[-32.0, -36.0, -40.0])
    parser.add_argument("--ramp-steps", nargs="+", type=int,
                        default=[10, 20])
    parser.add_argument("--hold-steps", type=int, default=15)
    parser.add_argument("--onset-step", type=int, default=10)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--joint", type=int, default=5)
    args = parser.parse_args()
    cuda = cuda_preflight()
    root = Path(__file__).resolve().parent
    output = root / "results" / (
        "saturation_aware_position_search_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True)
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": vars(args), "cuda": cuda, "cases": CASES,
        "purpose": "search only; final validation must use disjoint seeds",
        "plant_convention": "tau_effective = tau_applied - disturbance",
        "headroom_reference": "Post-DOB warm-start center + current d_hat",
        "penalty": "common softplus, sharpness 8, legacy 5%-span normalization",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows = []
    print("OUTPUT", output, flush=True)
    for fraction in args.initial_fractions:
        reference, q0 = reference_and_initial(args.steps, args.joint, fraction)
        for amplitude in args.amplitudes:
            for ramp_steps in args.ramp_steps:
                disturbance = ramp_pulse(
                    amplitude, ramp_steps, args.hold_steps,
                    args.onset_step, args.joint,
                )
                for case_name, case in CASES.items():
                    cfg = MarginConfig(
                        mode=case["mode"], min_fraction=0.0,
                        fixed_fraction=case["fixed_fraction"],
                        max_fraction=0.10, reaction_time_s=0.10,
                        acceleration_deadband_rad_s2=
                            ACCELERATION_DEADBANDS["nominal_p99"],
                        penalty_kind="softplus", softplus_sharpness=8.0,
                        output_compensated_headroom=True,
                    )
                    holder = []

                    def factory(*factory_args, **factory_kwargs):
                        tracker = NoiseAwareKrocTracker(
                            *factory_args, **factory_kwargs
                        )
                        holder.append(tracker)
                        return tracker

                    print(
                        f"RUN {fraction=} {amplitude=} {ramp_steps=} {case_name=}",
                        flush=True,
                    )
                    data = run_mppi(
                        reference, initial_q=q0, has_disturbance=True,
                        use_dob=True, dob_compensation_mode="output",
                        disturbance_function=disturbance, require_cuda=True,
                        seed=args.seed, temperature=1.0, temporal_noise_rho=0.8,
                        joint_position_limit_weight=50000.0,
                        joint_velocity_limit_weight=50000.0,
                        tracker_class=factory,
                        tracker_options={
                            "margin_config": asdict(cfg),
                            "sensor_noise": {
                                "position_std_rad": 1e-3,
                                "velocity_std_rad_s": 1e-2,
                                "seed": 50000 + args.seed,
                            },
                            "dob_acceleration_lpf_cutoff_hz": 10.0,
                            "dob_cutoff_hz": 8.0,
                        },
                    )
                    data.update(holder[0].export_margin_data())
                    data.update(holder[0].export_sensor_data())
                    data["physical_torque_limit"] = np.asarray(TAU_MAX)
                    row = {
                        "initial_fraction": fraction, "amplitude_nm": amplitude,
                        "ramp_steps": ramp_steps, "case": case_name,
                        "metrics": metrics(data, args.joint, args.onset_step),
                    }
                    rows.append(row)
                    (output / "metrics.json").write_text(
                        json.dumps(rows, indent=2) + "\n"
                    )
                    print(json.dumps(row), flush=True)
                    del data, holder
                    gc.collect()
                    torch.cuda.empty_cache()

    grouped = {}
    for row in rows:
        key = (row["initial_fraction"], row["amplitude_nm"], row["ramp_steps"])
        grouped.setdefault(key, {})[row["case"]] = row["metrics"]
    ranking = []
    for (fraction, amplitude, ramp_steps), cases in grouped.items():
        basic, dynamic = cases["basic_soft"], cases["dynamic_headroom"]
        ranking.append({
            "initial_fraction": fraction, "amplitude_nm": amplitude,
            "ramp_steps": ramp_steps,
            "basic_hard_violation_steps": basic["hard_violation_steps"],
            "dynamic_hard_violation_steps": dynamic["hard_violation_steps"],
            "violation_step_reduction": basic["hard_violation_steps"]
                - dynamic["hard_violation_steps"],
            "peak_reduction_rad": basic["peak_position_rad"]
                - dynamic["peak_position_rad"],
            "basic_saturation_steps": basic["test_joint_saturation_steps"],
            "dynamic_saturation_steps": dynamic["test_joint_saturation_steps"],
            "dynamic_headroom_gain_max": dynamic["headroom_gain_max"],
            "dynamic_margin_before_peak": dynamic["margin_active_before_peak"],
            "dynamic_tracking_rmse_mm": dynamic["tracking_rmse_mm"],
        })
    ranking.sort(key=lambda row: (
        row["violation_step_reduction"], row["peak_reduction_rad"]
    ), reverse=True)
    (output / "ranking.json").write_text(json.dumps(ranking, indent=2) + "\n")
    print("BEST", json.dumps(ranking[:5]), flush=True)


if __name__ == "__main__":
    main()
