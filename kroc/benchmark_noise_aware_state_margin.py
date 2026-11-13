"""Pilot for fair dynamic margins under clean and noise-aware Post-DOB."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DQ_MAX, DT, Q_MAX, Q_MIN, TAU_MAX
from experiment_runner import run_mppi, waypoints
from kinematics.ur5_kinematics import fk_joints_np
from kroc.benchmark_state_margin import measure
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker


ENVIRONMENTS = {
    "clean_legacy": {
        "sensor_noise": {"position_std_rad": 0.0,
                         "velocity_std_rad_s": 0.0},
        "dob_acceleration_lpf_cutoff_hz": 0.0,
        "dob_cutoff_hz": None,
    },
    "noisy_filtered": {
        "sensor_noise": {"position_std_rad": 1e-3,
                         "velocity_std_rad_s": 1e-2},
        "dob_acceleration_lpf_cutoff_hz": 10.0,
        "dob_cutoff_hz": 8.0,
    },
}

# Calibrated only from zero-disturbance noisy-filtered runs with seeds
# 17/27/37. Validation must use disjoint seeds.
ACCELERATION_DEADBANDS = {
    "none": (0.0,) * 6,
    "nominal_p95": (0.7491259, 0.64354855, 0.8437225,
                    1.1001786, 1.0568228, 2.3656714),
    "nominal_p99": (0.94216007, 0.8738429, 1.1924376,
                    1.4469504, 1.3889327, 2.9512007),
}


def _task_reference(joints: np.ndarray) -> np.ndarray:
    return np.asarray([fk_joints_np(q)[-1] for q in joints])


def build_scenario(name: str, steps: int, position_finish_fraction: float = 0.96,
                   position_disturbance_nm: float = -10.0):
    """Return task reference, matching q0 and a reproducible disturbance."""
    if steps < 80:
        raise ValueError("constraint-approach scenarios require at least 80 steps")
    base = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    joints = np.repeat(base[None, :], steps, axis=0)

    if name == "nominal":
        reference = waypoints("circle", n_steps=steps)
        q0 = None
        disturbance_values = np.zeros(6)
        disturbance_duration_steps = None
    elif name == "position_limit":
        # Elbow joint (index 2) approaches its +pi physical limit.  The fixed
        # 5% soft boundary begins at 0.9*pi, so the reference crosses it while
        # remaining below the physical limit.
        start, finish = 0.82 * math.pi, position_finish_fraction * math.pi
        phase = np.clip(np.arange(steps) / (0.65 * (steps - 1)), 0.0, 1.0)
        smooth = phase * phase * (3.0 - 2.0 * phase)
        joints[:, 2] = start + (finish - start) * smooth
        reference, q0 = _task_reference(joints), joints[0]
        # Plant convention is tau_effective = tau_applied - disturbance.
        # A negative elbow disturbance therefore pushes the joint toward its
        # positive position limit, matching the reference motion.
        disturbance_values = np.array(
            [0.0, 0.0, position_disturbance_nm, 0.0, 0.0, 0.0]
        )
        disturbance_duration_steps = None
    elif name == "velocity_limit":
        # Controlled recovery case: start wrist-3 below the fixed 5% soft
        # boundary and apply a signed disturbance that pushes it toward the
        # positive velocity limit under tau_effective = tau_applied - d.
        reference, q0 = _task_reference(joints), joints[0]
        disturbance_values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -16.0])
        disturbance_ramp_steps = 5
        disturbance_hold_steps = 5
        disturbance_duration_steps = disturbance_ramp_steps + disturbance_hold_steps
    else:
        raise ValueError(name)

    onset = 0 if name == "velocity_limit" else steps // 4

    def disturbance(step, device, dtype):
        active = step >= onset and (
            disturbance_duration_steps is None
            or step < onset + disturbance_duration_steps
        )
        values = disturbance_values if active else np.zeros(6)
        if active and name == "velocity_limit":
            elapsed = step - onset
            values = disturbance_values * min(
                (elapsed + 1) / disturbance_ramp_steps, 1.0
            )
        return torch.as_tensor(values, device=device, dtype=dtype)

    metadata = {
        "reference_joint_position_max": np.max(joints, axis=0).tolist()
        if q0 is not None else None,
        "reference_joint_position_min": np.min(joints, axis=0).tolist()
        if q0 is not None else None,
        "reference_joint_velocity_abs_max": float(
            np.abs(np.gradient(joints, DT, axis=0)).max()
        ) if q0 is not None else None,
        "physical_position_min": list(Q_MIN),
        "physical_position_max": list(Q_MAX),
        "physical_velocity_abs_max": DQ_MAX,
        "disturbance_nm": disturbance_values.tolist(),
        "disturbance_duration_steps": disturbance_duration_steps,
        "disturbance_ramp_steps": disturbance_ramp_steps
        if name == "velocity_limit" else None,
        "disturbance_hold_steps": disturbance_hold_steps
        if name == "velocity_limit" else None,
        "disturbance_onset_step": onset,
        "initial_joint_velocity": ([0.0] * 5 + [2.5])
        if name == "velocity_limit" else [0.0] * 6,
    }
    return reference, q0, disturbance, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--scenarios", nargs="+",
                        choices=("nominal", "position_limit", "velocity_limit"),
                        default=["nominal", "position_limit", "velocity_limit"])
    parser.add_argument("--environments", nargs="+", choices=ENVIRONMENTS,
                        default=list(ENVIRONMENTS))
    parser.add_argument("--modes", nargs="+",
                        choices=("fixed", "fixed_max", "directional", "headroom"),
                        default=["fixed", "directional", "headroom"])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--acceleration-deadband-profile",
                        choices=ACCELERATION_DEADBANDS, default="none")
    parser.add_argument("--joint-position-limit-weight", type=float,
                        default=50000.0)
    parser.add_argument("--position-finish-fraction", type=float, default=0.96)
    parser.add_argument("--position-disturbance-nm", type=float, default=-10.0)
    args = parser.parse_args()
    if args.joint_position_limit_weight < 0:
        parser.error("joint position limit weight must be nonnegative")
    if not 0.82 < args.position_finish_fraction < 1.0:
        parser.error("position finish fraction must be in (0.82, 1.0)")
    if args.position_disturbance_nm >= 0:
        parser.error("upper-limit position disturbance must be negative")
    preflight = cuda_preflight()

    root = Path(__file__).resolve().parent
    output = root / "results" / (
        "noise_aware_state_margin_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), root / "state_margin.py", root / "sensor_noise.py",
               Path("controllers/mppi_dob_controller.py"),
               Path("observers/dob.py"), Path("experiment_runner.py")]
    scenario_data = {
        name: build_scenario(
            name, args.steps, args.position_finish_fraction,
            args.position_disturbance_nm,
        )[3] for name in args.scenarios
    }
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": vars(args), "cuda": preflight,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True),
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "environments": ENVIRONMENTS,
        "scenarios": scenario_data,
        "controller": "Post-DOB output compensation",
        "margin_comparison": {
            "fixed_fraction": 0.05, "dynamic_min_fraction": 0.05,
            "dynamic_max_fraction": 0.10,
            "reaction_time_s": 0.10,
            "headroom_reference": "unclamped U_ff + R",
        },
        "dynamic_state_margin": True, "state_clamp": False,
        "acceleration_deadband": {
            "profile": args.acceleration_deadband_profile,
            "values_rad_s2": list(ACCELERATION_DEADBANDS[
                args.acceleration_deadband_profile]),
            "calibration_scenario": "zero-disturbance noisy-filtered nominal",
            "calibration_seeds": [17, 27, 37],
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("OUTPUT", output, flush=True)

    rows = []
    for scenario_index, scenario in enumerate(args.scenarios):
        reference, q0, disturbance, scenario_metadata = build_scenario(
            scenario, args.steps, args.position_finish_fraction,
            args.position_disturbance_nm,
        )
        for seed_index, seed in enumerate(args.seeds):
            cases = [(environment, mode) for environment in args.environments
                     for mode in args.modes]
            shift = (scenario_index + seed_index) % len(cases)
            cases = cases[shift:] + cases[:shift]
            for environment, mode in cases:
                environment_cfg = ENVIRONMENTS[environment]
                fixed_fraction = 0.10 if mode == "fixed_max" else 0.05
                internal_mode = "fixed" if mode == "fixed_max" else mode
                margin_cfg = MarginConfig(
                    mode=internal_mode, min_fraction=0.05,
                    fixed_fraction=fixed_fraction,
                    max_fraction=0.10, reaction_time_s=0.10,
                    acceleration_deadband_rad_s2=ACCELERATION_DEADBANDS[
                        args.acceleration_deadband_profile],
                )
                sensor_cfg = dict(
                    environment_cfg["sensor_noise"], seed=20000 + seed
                )
                holder = []

                def factory(*factory_args, **factory_kwargs):
                    tracker = NoiseAwareKrocTracker(
                        *factory_args, **factory_kwargs
                    )
                    holder.append(tracker)
                    return tracker

                print(f"RUN {scenario=} {seed=} {environment=} {mode=}", flush=True)
                data = run_mppi(
                    reference, initial_q=q0,
                    initial_qd=scenario_metadata["initial_joint_velocity"],
                    has_disturbance=scenario != "nominal", use_dob=True,
                    dob_compensation_mode="output",
                    disturbance_function=disturbance, require_cuda=True,
                    seed=seed, temperature=1.0, temporal_noise_rho=0.8,
                    joint_position_limit_weight=args.joint_position_limit_weight,
                    joint_velocity_limit_weight=50000.0,
                    tracker_class=factory,
                    tracker_options={
                        "margin_config": asdict(margin_cfg),
                        "sensor_noise": sensor_cfg,
                        "dob_acceleration_lpf_cutoff_hz":
                            environment_cfg["dob_acceleration_lpf_cutoff_hz"],
                        "dob_cutoff_hz": environment_cfg["dob_cutoff_hz"],
                    },
                )
                data.update(holder[0].export_margin_data())
                data.update(holder[0].export_sensor_data())
                data["true_state"] = np.concatenate(
                    (np.asarray(data["q"]), np.asarray(data["qd"])), axis=1
                )
                data["qd_raw"] = data["qd"].copy()
                data["state_clamp_enabled"] = np.asarray(False)
                data["dynamic_state_margin_enabled"] = np.asarray(True)
                data["physical_torque_limit"] = np.asarray(TAU_MAX)
                metrics = measure(data, 1.0)
                d_error = data["d_hat"][1:] - data["d_true"][1:]
                metrics["dob_estimation_rmse_nm"] = float(
                    np.sqrt(np.mean(d_error ** 2))
                )
                row = {
                    "scenario": scenario, "seed": seed,
                    "environment": environment, "mode": mode,
                    "margin_config": asdict(margin_cfg), "metrics": metrics,
                }
                rows.append(row)
                stem = f"{scenario}_seed{seed}_{environment}_{mode}"
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

    lines = [
        "# Noise-aware dynamic state-margin pilot", "",
        "Fixed and adaptive minimum margins are both 5% of full span.", "",
        "| Scenario | Environment | Mode | RMSE [mm] | q min dist [rad] | "
        "dq min dist [rad/s] | q violations | dq violations | saturation |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        item = row["metrics"]
        lines.append(
            f"| {row['scenario']} | {row['environment']} | {row['mode']} | "
            f"{item['tracking_rmse_mm']:.3f} | "
            f"{item['position_minimum_distance']:.4f} | "
            f"{item['velocity_minimum_distance']:.4f} | "
            f"{item['position_violation_steps']} | "
            f"{item['velocity_violation_steps']} | "
            f"{item['saturation_steps']} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
