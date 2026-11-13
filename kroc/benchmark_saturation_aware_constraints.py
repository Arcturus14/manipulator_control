"""Final paired comparison of soft, static, and zero-floor dynamic constraints."""
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
from config import TAU_MAX
from experiment_runner import run_mppi
from kroc.benchmark_noise_aware_state_margin import ACCELERATION_DEADBANDS
from kroc.search_saturation_aware_position_scenario import (
    metrics,
    ramp_pulse,
    reference_and_initial,
)
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker


CASES = {
    "basic_soft": {
        "mode": "fixed", "fixed_fraction": 0.0, "main_comparison": True,
    },
    "fixed_small_5pct": {
        "mode": "fixed", "fixed_fraction": 0.05, "main_comparison": True,
    },
    "fixed_conservative_10pct": {
        "mode": "fixed", "fixed_fraction": 0.10, "main_comparison": True,
    },
    "dynamic_headroom_0_to_10pct": {
        "mode": "headroom", "fixed_fraction": 0.0, "main_comparison": True,
    },
    # Not a fifth proposal: this is an internal causal ablation that separates
    # disturbance direction from the torque-headroom multiplier.
    "directional_ablation_0_to_10pct": {
        "mode": "directional", "fixed_fraction": 0.0,
        "main_comparison": False,
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int,
                        default=[17, 27, 37, 47, 57])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--initial-fraction", type=float, default=0.88)
    parser.add_argument("--amplitude", type=float, default=-30.0)
    parser.add_argument("--ramp-steps", type=int, default=20)
    parser.add_argument("--hold-steps", type=int, default=15)
    parser.add_argument("--onset-step", type=int, default=10)
    parser.add_argument("--joint", type=int, default=5)
    args = parser.parse_args()
    cuda = cuda_preflight()
    root = Path(__file__).resolve().parent
    output = root / "results" / (
        "saturation_aware_constraints_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True)
    sources = [
        Path(__file__), root / "state_margin.py",
        root / "sensor_noise.py",
        root / "search_saturation_aware_position_scenario.py",
        Path("controllers/mppi_dob_controller.py"),
        Path("observers/dob.py"), Path("experiment_runner.py"),
    ]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": vars(args), "cuda": cuda, "cases": CASES,
        "selection": {
            "search_seed": 7,
            "search_result": "saturation_aware_position_search_20261109_030910_011298",
            "validation_seeds_are_disjoint": True,
        },
        "controller": "noise-aware Post-DOB output compensation",
        "sensor_noise": {"position_std_rad": 1e-3,
                         "velocity_std_rad_s": 1e-2},
        "dob": {"acceleration_lpf_hz": 10.0, "exact_zoh_cutoff_hz": 8.0},
        "deadband": {"profile": "nominal_p99", "values": list(
            ACCELERATION_DEADBANDS["nominal_p99"]
        )},
        "cost": {
            "kind": "softplus", "sharpness": 8.0,
            "normalization": "legacy 5%-of-full-span scale, common to all cases",
            "position_limit_weight": 50000.0,
            "velocity_limit_weight": 50000.0,
        },
        "headroom_reference": "unclamped Post-DOB center + current d_hat",
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True),
        "source_hashes": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sources
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("OUTPUT", output, flush=True)
    reference, q0 = reference_and_initial(
        args.steps, args.joint, args.initial_fraction
    )
    disturbance = ramp_pulse(
        args.amplitude, args.ramp_steps, args.hold_steps,
        args.onset_step, args.joint,
    )
    rows = []
    for seed_index, seed in enumerate(args.seeds):
        names = list(CASES)
        shift = seed_index % len(names)
        names = names[shift:] + names[:shift]
        for case_name in names:
            case = CASES[case_name]
            cfg = MarginConfig(
                mode=case["mode"], min_fraction=0.0,
                fixed_fraction=case["fixed_fraction"], max_fraction=0.10,
                reaction_time_s=0.10,
                acceleration_deadband_rad_s2=
                    ACCELERATION_DEADBANDS["nominal_p99"],
                penalty_kind="softplus", softplus_sharpness=8.0,
                output_compensated_headroom=True,
            )
            holder = []

            def factory(*factory_args, **factory_kwargs):
                tracker = NoiseAwareKrocTracker(*factory_args, **factory_kwargs)
                holder.append(tracker)
                return tracker

            print(f"RUN {seed=} {case_name=}", flush=True)
            data = run_mppi(
                reference, initial_q=q0, has_disturbance=True, use_dob=True,
                dob_compensation_mode="output",
                disturbance_function=disturbance, require_cuda=True,
                seed=seed, temperature=1.0, temporal_noise_rho=0.8,
                joint_position_limit_weight=50000.0,
                joint_velocity_limit_weight=50000.0,
                tracker_class=factory,
                tracker_options={
                    "margin_config": asdict(cfg),
                    "sensor_noise": {
                        "position_std_rad": 1e-3,
                        "velocity_std_rad_s": 1e-2,
                        "seed": 60000 + seed,
                    },
                    "dob_acceleration_lpf_cutoff_hz": 10.0,
                    "dob_cutoff_hz": 8.0,
                },
            )
            data.update(holder[0].export_margin_data())
            data.update(holder[0].export_sensor_data())
            data["physical_torque_limit"] = np.asarray(TAU_MAX)
            item = metrics(data, args.joint, args.onset_step)
            item.update({
                "compute_mean_ms": float(np.nanmean(data["solve_t"]) * 1e3),
                "compute_p95_ms": float(np.nanpercentile(data["solve_t"], 95) * 1e3),
            })
            saturation = np.isclose(
                np.abs(np.asarray(data["u_applied"])[:, args.joint]),
                TAU_MAX[args.joint], atol=1e-5, rtol=0.0,
            )
            item["first_test_joint_saturation_step"] = (
                int(np.argmax(saturation)) if saturation.any() else None
            )
            row = {
                "seed": seed, "case": case_name,
                "main_comparison": case["main_comparison"],
                "margin_config": asdict(cfg), "metrics": item,
            }
            rows.append(row)
            stem = f"seed{seed}_{case_name}"
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
