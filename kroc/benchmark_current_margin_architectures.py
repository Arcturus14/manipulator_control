"""Compare current noise-aware dynamic margins in Post-DOB and model-only DOB."""
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
from config import DQ_MAX, Q_MAX, Q_MIN, TAU_MAX
from experiment_runner import run_mppi
from kroc.benchmark_noise_aware_state_margin import (
    ACCELERATION_DEADBANDS,
    ENVIRONMENTS,
    build_scenario,
)
from kroc.benchmark_state_margin import measure
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker


ARCHITECTURES = {
    "post_dob": "output",
    "model_only": "rollout_model_only",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[47, 57, 67, 77, 87])
    parser.add_argument("--scenarios", nargs="+",
                        choices=("position_limit", "velocity_limit"),
                        default=["position_limit", "velocity_limit"])
    parser.add_argument("--modes", nargs="+",
                        choices=("fixed", "directional"),
                        default=["fixed", "directional"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 80:
        parser.error("steps must be >= 80")

    preflight = cuda_preflight()
    root = Path(__file__).resolve().parent
    output = args.output or root / "results" / (
        "current_margin_architectures_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    sources = [
        Path(__file__), root / "state_margin.py",
        root / "sensor_noise.py",
        root / "benchmark_noise_aware_state_margin.py",
        Path("controllers/mppi_dob_controller.py"),
        Path("observers/dob.py"), Path("experiment_runner.py"),
    ]
    environment = ENVIRONMENTS["noisy_filtered"]
    scenario_data = {
        name: build_scenario(name, args.steps)[3]
        for name in args.scenarios
    }
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": {**vars(args), "output": str(args.output)},
        "cuda": preflight,
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True),
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
        "architectures": ARCHITECTURES,
        "scenarios": scenario_data,
        "environment": environment,
        "controlled_variables": {
            "sensor_position_std_rad": 1e-3,
            "sensor_velocity_std_rad_s": 1e-2,
            "dob_acceleration_lpf_cutoff_hz": 10.0,
            "dob_cutoff_hz": 8.0,
            "acceleration_deadband_profile": "nominal_p99",
            "acceleration_deadband_rad_s2": list(
                ACCELERATION_DEADBANDS["nominal_p99"]
            ),
            "fixed_margin_fraction": 0.05,
            "directional_margin_range_fraction": [0.05, 0.10],
            "reaction_time_s": 0.10,
            "joint_position_limit_weight": 50000.0,
            "joint_velocity_limit_weight": 50000.0,
            "state_clamp": False,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print("OUTPUT", output, flush=True)

    rows = []
    for scenario_index, scenario in enumerate(args.scenarios):
        reference, q0, disturbance, scenario_metadata = build_scenario(
            scenario, args.steps
        )
        for seed_index, seed in enumerate(args.seeds):
            cases = [(architecture, mode) for architecture in ARCHITECTURES
                     for mode in args.modes]
            shift = (scenario_index + seed_index) % len(cases)
            cases = cases[shift:] + cases[:shift]
            for architecture, mode in cases:
                margin_cfg = MarginConfig(
                    mode=mode, min_fraction=0.05, fixed_fraction=0.05,
                    max_fraction=0.10, reaction_time_s=0.10,
                    acceleration_deadband_rad_s2=
                        ACCELERATION_DEADBANDS["nominal_p99"],
                )
                holder = []

                def factory(*factory_args, **factory_kwargs):
                    tracker = NoiseAwareKrocTracker(
                        *factory_args, **factory_kwargs
                    )
                    holder.append(tracker)
                    return tracker

                print(
                    f"RUN {scenario=} {seed=} {architecture=} {mode=}",
                    flush=True,
                )
                data = run_mppi(
                    reference, initial_q=q0,
                    initial_qd=scenario_metadata["initial_joint_velocity"],
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
                            **environment["sensor_noise"],
                            "seed": 20000 + seed,
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
                data["qd_raw"] = data["qd"].copy()
                data["state_clamp_enabled"] = np.asarray(False)
                data["dynamic_state_margin_enabled"] = np.asarray(True)
                data["physical_torque_limit"] = np.asarray(TAU_MAX)
                metrics = measure(data, 1.0)
                d_error = data["d_hat"][1:] - data["d_true"][1:]
                metrics.update({
                    "dob_estimation_rmse_nm": float(
                        np.sqrt(np.mean(d_error ** 2))
                    ),
                    "elbow_upper_distance_rad": float(
                        Q_MAX[2] - np.max(data["q"][:, 2])
                    ),
                    "wrist3_peak_velocity_rad_s": float(
                        np.max(data["qd"][:, 5])
                    ),
                    "wrist3_velocity_excess_area": float(
                        np.maximum(data["qd"][:, 5] - DQ_MAX, 0).sum()
                    ),
                })
                row = {
                    "scenario": scenario, "seed": seed,
                    "architecture": architecture,
                    "dob_compensation_mode": ARCHITECTURES[architecture],
                    "mode": mode, "margin_config": asdict(margin_cfg),
                    "metrics": metrics,
                }
                rows.append(row)
                stem = f"{scenario}_seed{seed}_{architecture}_{mode}"
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

    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
