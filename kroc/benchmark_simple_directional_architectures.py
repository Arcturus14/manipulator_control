"""Simple 2x3 comparison: DOB architecture by state-constraint strategy."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DQ_MAX, Q_MAX, TAU_MAX
from experiment_runner import run_mppi
from kroc.benchmark_noise_aware_state_margin import (
    ACCELERATION_DEADBANDS,
    ENVIRONMENTS,
    build_scenario,
)
from kroc.benchmark_state_margin import measure
from kroc.search_saturation_aware_position_scenario import (
    ramp_pulse,
    reference_and_initial,
)
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker


ARCHITECTURES = {
    "post_dob": "output",
    "model_only": "rollout_model_only",
}

ZERO_FLOOR_CASES = {
    "basic_soft_0pct": {"mode": "fixed", "fixed_fraction": 0.0},
    "static_conservative_10pct": {
        "mode": "fixed", "fixed_fraction": 0.10,
    },
    "dynamic_directional_0_to_10pct": {
        "mode": "directional", "fixed_fraction": 0.0,
    },
}

LEGACY_FLOOR_CASES = {
    "basic_native_soft_5pct": {
        "mode": "fixed", "min_fraction": 0.05,
        "fixed_fraction": 0.05, "penalty_kind": "hinge",
    },
    "static_conservative_10pct": {
        "mode": "fixed", "min_fraction": 0.05,
        "fixed_fraction": 0.10, "penalty_kind": "hinge",
    },
    "dynamic_directional_5_to_10pct": {
        "mode": "directional", "min_fraction": 0.05,
        "fixed_fraction": 0.05, "penalty_kind": "hinge",
    },
}

SCENARIOS = ("position_limit", "velocity_limit")


def scenario(name: str, steps: int):
    if name == "position_limit":
        reference, q0 = reference_and_initial(steps, joint=5, fraction=0.88)
        disturbance = ramp_pulse(
            amplitude=-30.0, ramp_steps=20, hold_steps=15,
            onset=10, joint=5,
        )
        return reference, q0, [0.0] * 6, disturbance, {
            "test_joint": 5,
            "initial_fraction_of_upper_limit": 0.88,
            "disturbance_nm": -30.0,
            "disturbance_onset_step": 10,
            "disturbance_ramp_steps": 20,
            "disturbance_hold_steps": 15,
        }
    reference, q0, disturbance, metadata = build_scenario(
        "velocity_limit", steps
    )
    return (
        reference, q0, metadata["initial_joint_velocity"], disturbance,
        {**metadata, "test_joint": 5},
    )


def extra_metrics(data, scenario_name: str, test_joint: int) -> dict:
    q = np.asarray(data["q"])[1:, test_joint]
    dq = np.asarray(data["qd"])[1:, test_joint]
    margin = np.asarray(data["kroc_margin"])
    upper_q_margin = margin[:, 0, 1, test_joint]
    upper_dq_margin = margin[:, 1, 1, test_joint]
    result = {
        "test_joint_position_upper_distance_rad": float(
            Q_MAX[test_joint] - q.max()
        ),
        "test_joint_velocity_upper_distance_rad_s": float(
            DQ_MAX - dq.max()
        ),
        "test_joint_peak_position_rad": float(q.max()),
        "test_joint_peak_velocity_rad_s": float(dq.max()),
        "position_upper_margin_mean_rad": float(upper_q_margin.mean()),
        "position_upper_margin_max_rad": float(upper_q_margin.max()),
        "velocity_upper_margin_mean_rad_s": float(upper_dq_margin.mean()),
        "velocity_upper_margin_max_rad_s": float(upper_dq_margin.max()),
    }
    if scenario_name == "position_limit":
        result["test_joint_violation_steps"] = int((q > Q_MAX[test_joint]).sum())
        result["test_joint_violation_max"] = float(
            np.maximum(q - Q_MAX[test_joint], 0.0).max()
        )
    else:
        result["test_joint_violation_steps"] = int((dq > DQ_MAX).sum())
        result["test_joint_violation_max"] = float(
            np.maximum(dq - DQ_MAX, 0.0).max()
        )
    return result


def stats(values) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sample_std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate(rows, cases, scenario_names) -> dict:
    fields = (
        "tracking_rmse_mm",
        "position_violation_steps", "position_violation_max",
        "position_violation_integral", "position_minimum_distance",
        "velocity_violation_steps", "velocity_violation_max",
        "velocity_violation_integral", "velocity_minimum_distance",
        "test_joint_position_upper_distance_rad",
        "test_joint_velocity_upper_distance_rad_s",
        "test_joint_peak_position_rad", "test_joint_peak_velocity_rad_s",
        "test_joint_violation_steps", "test_joint_violation_max",
        "saturation_steps", "torque_rate_rms_nm_s",
        "compute_mean_ms", "compute_p95_ms",
        "position_upper_margin_mean_rad", "position_upper_margin_max_rad",
        "velocity_upper_margin_mean_rad_s", "velocity_upper_margin_max_rad_s",
        "dob_estimation_rmse_nm",
    )
    result = {"n_seeds": len({row["seed"] for row in rows}), "groups": {}}
    for scenario_name in scenario_names:
        result["groups"][scenario_name] = {}
        for architecture in ARCHITECTURES:
            result["groups"][scenario_name][architecture] = {}
            for case_name in cases:
                selected = [
                    row["metrics"] for row in rows
                    if row["scenario"] == scenario_name
                    and row["architecture"] == architecture
                    and row["case"] == case_name
                ]
                result["groups"][scenario_name][architecture][case_name] = {
                    field: stats([item[field] for item in selected])
                    for field in fields
                }
    return result


def report(aggregate_data: dict, cases, scenario_names, margin_profile) -> str:
    lines = [
        "# Sensor-noise directional margin: Post-DOB vs Model-only", "",
        "Five paired seeds; values are mean ± sample standard deviation.", "",
    ]
    for scenario_name in scenario_names:
        lines.extend([
            f"## {scenario_name}", "",
            "| Architecture | Constraint | RMSE [mm] | q violation [step] | dq violation [step] | test-joint violation [step] | test-joint max violation | saturation [step] |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ])
        for architecture in ARCHITECTURES:
            for case_name in cases:
                group = aggregate_data["groups"][scenario_name][architecture][case_name]
                value = lambda field: group[field]
                fmt = lambda field, digits=3: (
                    f"{value(field)['mean']:.{digits}f} ± "
                    f"{value(field)['sample_std']:.{digits}f}"
                )
                lines.append(
                    f"| {architecture} | {case_name} | {fmt('tracking_rmse_mm')} | "
                    f"{fmt('position_violation_steps', 2)} | "
                    f"{fmt('velocity_violation_steps', 2)} | "
                    f"{fmt('test_joint_violation_steps', 2)} | "
                    f"{fmt('test_joint_violation_max', 4)} | "
                    f"{fmt('saturation_steps', 2)} |"
                )
        lines.append("")
    if margin_profile == "legacy_floor_hinge":
        design = [
            "- Dynamic: uploaded directional stopping-distance equations, m0=5%, Tr=0.1 s, cap=10%",
            "- Static conservative tightening: 10% of full physical span",
            "- Basic soft constraint: native 5%-of-full-span floor",
            "- Penalty: native ReLU-squared (hinge-squared)",
        ]
    else:
        design = [
            "- Dynamic: uploaded directional stopping-distance equations, m0=0, Tr=0.1 s, cap=10%",
            "- Static conservative tightening: 10% of full physical span",
            "- Basic soft constraint: zero tightening",
            "- Penalty: common softplus-squared, sharpness 8",
        ]
    lines.extend([
        "## Fixed design", "",
        "- Sensor noise: q 0.001 rad, dq 0.01 rad/s",
        "- DOB: acceleration LPF 10 Hz, exact-ZOH cutoff 8 Hz",
        *design,
        "- Dynamic acceleration: joint-wise P99 soft deadband",
        "- Headroom multiplier: disabled",
        "- State hard clamp: disabled",
    ])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17, 27, 37, 47, 57])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=list(SCENARIOS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--margin-profile",
        choices=("zero_floor_softplus", "legacy_floor_hinge"),
        default="zero_floor_softplus",
    )
    args = parser.parse_args()
    if args.steps < 80:
        parser.error("steps must be >= 80")

    cuda = cuda_preflight()
    if args.margin_profile == "legacy_floor_hinge":
        cases = LEGACY_FLOOR_CASES
    else:
        cases = {
            name: {
                **case, "min_fraction": 0.0, "penalty_kind": "softplus",
            }
            for name, case in ZERO_FLOOR_CASES.items()
        }
    root = Path(__file__).resolve().parent
    output = args.output or root / "results" / (
        "simple_directional_architectures_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    environment = ENVIRONMENTS["noisy_filtered"]
    sources = [
        Path(__file__), root / "state_margin.py", root / "sensor_noise.py",
        root / "benchmark_noise_aware_state_margin.py",
        Path("controllers/mppi_dob_controller.py"), Path("observers/dob.py"),
        Path("experiment_runner.py"), Path("costs/tracking_cost.py"),
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": {**vars(args), "output": str(args.output)},
        "cuda": cuda, "architectures": ARCHITECTURES, "cases": cases,
        "environment": environment,
        "scenarios": {
            name: scenario(name, args.steps)[4] for name in args.scenarios
        },
        "controlled_variables": {
            "sensor_position_std_rad": 1e-3,
            "sensor_velocity_std_rad_s": 1e-2,
            "acceleration_lpf_cutoff_hz": 10.0,
            "dob_exact_zoh_cutoff_hz": 8.0,
            "acceleration_deadband": "nominal_p99",
            "dynamic_min_fraction": (
                0.05 if args.margin_profile == "legacy_floor_hinge" else 0.0
            ),
            "dynamic_max_fraction": 0.10,
            "reaction_time_s": 0.10,
            "static_conservative_fraction": 0.10,
            "headroom_multiplier": False,
            "penalty": (
                "native ReLU-squared"
                if args.margin_profile == "legacy_floor_hinge"
                else "common softplus-squared, sharpness 8"
            ),
            "state_hard_clamp": False,
        },
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ),
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("OUTPUT", output, flush=True)

    rows = []
    for scenario_index, scenario_name in enumerate(args.scenarios):
        reference, q0, initial_qd, disturbance, metadata = scenario(
            scenario_name, args.steps
        )
        for seed_index, seed in enumerate(args.seeds):
            run_cases = [
                (architecture, case_name)
                for architecture in ARCHITECTURES for case_name in cases
            ]
            shift = (scenario_index + seed_index) % len(run_cases)
            run_cases = run_cases[shift:] + run_cases[:shift]
            for architecture, case_name in run_cases:
                case = cases[case_name]
                margin_cfg = MarginConfig(
                    mode=case["mode"], min_fraction=case["min_fraction"],
                    fixed_fraction=case["fixed_fraction"], max_fraction=0.10,
                    reaction_time_s=0.10,
                    acceleration_deadband_rad_s2=
                        ACCELERATION_DEADBANDS["nominal_p99"],
                    penalty_kind=case["penalty_kind"], softplus_sharpness=8.0,
                    output_compensated_headroom=False,
                )
                holder = []

                def factory(*factory_args, **factory_kwargs):
                    tracker = NoiseAwareKrocTracker(
                        *factory_args, **factory_kwargs
                    )
                    holder.append(tracker)
                    return tracker

                print(
                    f"RUN scenario={scenario_name} seed={seed} "
                    f"architecture={architecture} case={case_name}",
                    flush=True,
                )
                data = run_mppi(
                    reference, initial_q=q0, initial_qd=initial_qd,
                    has_disturbance=True, use_dob=True,
                    dob_compensation_mode=ARCHITECTURES[architecture],
                    disturbance_function=disturbance, require_cuda=True,
                    seed=seed, temperature=1.0, temporal_noise_rho=0.8,
                    joint_position_limit_weight=50000.0,
                    joint_velocity_limit_weight=50000.0,
                    tracker_class=factory,
                    tracker_options={
                        "margin_config": asdict(margin_cfg),
                        "sensor_noise": {
                            **environment["sensor_noise"], "seed": 70000 + seed,
                        },
                        "dob_acceleration_lpf_cutoff_hz":
                            environment["dob_acceleration_lpf_cutoff_hz"],
                        "dob_cutoff_hz": environment["dob_cutoff_hz"],
                    },
                )
                data.update(holder[0].export_margin_data())
                data.update(holder[0].export_sensor_data())
                data["true_state"] = np.concatenate(
                    (np.asarray(data["q"]), np.asarray(data["qd"])), axis=1
                )
                data["qd_raw"] = np.asarray(data["qd"]).copy()
                data["state_clamp_enabled"] = np.asarray(False)
                data["physical_torque_limit"] = np.asarray(TAU_MAX)
                metrics = measure(data, 1.0)
                metrics.update(extra_metrics(
                    data, scenario_name, metadata["test_joint"]
                ))
                d_error = np.asarray(data["d_hat"])[1:] - np.asarray(
                    data["d_true"]
                )[1:]
                metrics["dob_estimation_rmse_nm"] = float(
                    np.sqrt(np.mean(d_error ** 2))
                )
                row = {
                    "scenario": scenario_name, "seed": seed,
                    "architecture": architecture,
                    "dob_compensation_mode": ARCHITECTURES[architecture],
                    "case": case_name, "margin_config": asdict(margin_cfg),
                    "metrics": metrics,
                }
                rows.append(row)
                stem = f"{scenario_name}_seed{seed}_{architecture}_{case_name}"
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

    aggregate_data = aggregate(rows, cases, args.scenarios)
    (output / "aggregate.json").write_text(
        json.dumps(aggregate_data, indent=2) + "\n"
    )
    (output / "REPORT.md").write_text(
        report(aggregate_data, cases, args.scenarios, args.margin_profile)
    )
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
