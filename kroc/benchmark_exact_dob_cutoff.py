"""GPU comparison of exact-ZOH DOB cutoffs with experimental sensor noise."""
from __future__ import annotations

import argparse
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
from config import D_AMP, D_FREQ, DT
from experiment_runner import run_mppi, waypoints
from kroc.benchmark_postdob_sensor_noise import aggregate, measure
from kroc.sensor_noise import SensorNoiseTracker


SCENARIOS = {
    "nominal": {"profile": "zero", "torque_scale": 1.0},
    "ramp": {"profile": "ramp_1x_0p5s", "torque_scale": 1.0},
    "sinusoidal": {"profile": "slow_sinusoidal_1x", "torque_scale": 1.0},
    "impulse": {"profile": "impulse_1x_0p1s", "torque_scale": 1.0},
    "strong_step": {"profile": "step_4x", "torque_scale": 1.0},
    "limited": {"profile": "step_2x", "torque_scale": 0.4},
}


def disturbance_function(name, steps):
    amplitude = np.asarray(D_AMP, dtype=float)
    onset = steps // 4

    def disturbance(step, device, dtype):
        if name == "nominal":
            values = np.zeros(6)
        elif name == "ramp":
            fraction = np.clip((step - onset) * DT / 0.5, 0.0, 1.0)
            values = fraction * amplitude
        elif name == "sinusoidal":
            time_s = step * DT
            values = np.asarray([
                a * math.sin(2.0 * math.pi * f * time_s)
                for a, f in zip(amplitude, D_FREQ)
            ])
        elif name == "impulse":
            values = amplitude if onset <= step < onset + 5 else np.zeros(6)
        elif name == "strong_step":
            values = 4.0 * amplitude if step >= onset else np.zeros(6)
        elif name == "limited":
            values = 2.0 * amplitude if step >= onset else np.zeros(6)
        else:
            raise ValueError(name)
        return torch.as_tensor(values, device=device, dtype=dtype)

    return disturbance


def select_cutoff(summary, cutoffs):
    # Respect --scenarios so focused smoke/validation runs do not attempt to
    # index scenarios that were deliberately omitted.
    disturbed = [name for name in summary if name != "nominal"]
    if not disturbed:
        raise ValueError("cutoff selection requires at least one disturbed scenario")
    labels = {value: f"noisy_exact_{value:g}hz" for value in cutoffs}
    best_rmse = {
        scenario: min(
            summary[scenario][labels[value]]["tracking_rmse_mm"]["mean"]
            for value in cutoffs
        )
        for scenario in disturbed
    }
    candidates = []
    for cutoff in cutoffs:
        label = labels[cutoff]
        ratios = {
            scenario: (
                summary[scenario][label]["tracking_rmse_mm"]["mean"]
                / best_rmse[scenario]
            )
            for scenario in disturbed
        }
        violations = sum(
            summary[scenario][label][field]["mean"]
            for scenario in SCENARIOS
            for field in ("position_violation_steps", "velocity_violation_steps")
        )
        average_degradation = float(np.mean(list(ratios.values())) - 1.0)
        worst_degradation = float(max(ratios.values()) - 1.0)
        eligible = (
            average_degradation <= 0.05
            and worst_degradation <= 0.10
            and violations == 0
        )
        nominal = summary["nominal"][label]
        candidates.append({
            "cutoff_hz": cutoff,
            "condition": label,
            "tracking_ratios": ratios,
            "average_tracking_degradation": average_degradation,
            "worst_tracking_degradation": worst_degradation,
            "state_violation_steps": violations,
            "eligible": eligible,
            "nominal_fake_disturbance_rms_nm":
                nominal["dob_estimate_rms_nm"]["mean"],
            "nominal_torque_rate_rms_nm_s":
                nominal["torque_rate_rms_nm_s"]["mean"],
        })
    eligible = [item for item in candidates if item["eligible"]]
    if eligible:
        min_fake = min(item["nominal_fake_disturbance_rms_nm"] for item in eligible)
        min_rate = min(item["nominal_torque_rate_rms_nm_s"] for item in eligible)
        for item in eligible:
            item["noise_score"] = 0.5 * (
                item["nominal_fake_disturbance_rms_nm"] / min_fake
                + item["nominal_torque_rate_rms_nm_s"] / min_rate
            )
        selected = min(eligible, key=lambda item: item["noise_score"])
        reason = "minimum nominal noise score among tracking-eligible candidates"
    else:
        selected = min(
            candidates,
            key=lambda item: (
                item["average_tracking_degradation"],
                item["worst_tracking_degradation"],
            ),
        )
        reason = "no candidate met the predeclared tracking gate"
    return {
        "tracking_gate": {
            "average_degradation_max": 0.05,
            "single_scenario_degradation_max": 0.10,
            "state_violation_steps_required": 0,
        },
        "noise_score": "equal normalized nominal d_hat RMS and torque-rate RMS",
        "candidates": candidates,
        "selected_cutoff_hz": selected["cutoff_hz"],
        "selection_reason": reason,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=list(SCENARIOS))
    parser.add_argument("--cutoffs", nargs="+", type=float,
                        default=[6.0, 8.0, 10.0, 12.0])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--position-std", type=float, default=1e-3)
    parser.add_argument("--velocity-std", type=float, default=1e-2)
    parser.add_argument("--acceleration-cutoff", type=float, default=10.0)
    args = parser.parse_args()
    if args.steps < 50:
        parser.error("steps must be >= 50")
    if any(not np.isfinite(value) or value <= 0 for value in args.cutoffs):
        parser.error("DOB cutoffs must be finite and positive")

    preflight = cuda_preflight()
    conditions = {
        "clean": {
            "sensor_noise": {"position_std_rad": 0.0,
                             "velocity_std_rad_s": 0.0},
            "acceleration_lpf_cutoff_hz": 0.0,
            "dob_cutoff_hz": None,
        },
        # Same-run control needed to attribute changes to filtering rather
        # than to a different experiment batch or MPPI random draw.
        "noisy_legacy": {
            "sensor_noise": {"position_std_rad": args.position_std,
                             "velocity_std_rad_s": args.velocity_std},
            "acceleration_lpf_cutoff_hz": 0.0,
            "dob_cutoff_hz": None,
        },
    }
    for cutoff in dict.fromkeys(args.cutoffs):
        conditions[f"noisy_exact_{cutoff:g}hz"] = {
            "sensor_noise": {"position_std_rad": args.position_std,
                             "velocity_std_rad_s": args.velocity_std},
            "acceleration_lpf_cutoff_hz": args.acceleration_cutoff,
            "dob_cutoff_hz": cutoff,
        }

    root = Path(__file__).resolve().parent
    output = root / "results" / (
        "exact_dob_cutoff_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), root / "sensor_noise.py",
               Path("observers/dob.py"),
               Path("controllers/mppi_dob_controller.py"),
               Path("experiment_runner.py")]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "arguments": vars(args), "cuda": preflight,
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
        "conditions": conditions,
        "scenarios": {name: SCENARIOS[name] for name in args.scenarios},
        "controller": "Post-DOB output compensation",
        "legacy_baseline_preserved": True,
        "same_run_noisy_legacy_control": True,
        "candidate_discretization": "exact_zoh",
        "dynamic_state_margin": False,
        "measurement_model": "independent zero-mean Gaussian q and dq noise",
        "noise_values_are_ur5_specifications": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("OUTPUT", output, flush=True)

    rows = []
    for scenario_index, scenario in enumerate(args.scenarios):
        scenario_cfg = SCENARIOS[scenario]
        disturbance = disturbance_function(scenario, args.steps)
        for seed_index, seed in enumerate(args.seeds):
            names = list(conditions)
            shift = (scenario_index + seed_index) % len(names)
            names = names[shift:] + names[:shift]
            for condition in names:
                cfg = conditions[condition]
                sensor_cfg = dict(cfg["sensor_noise"], seed=10000 + seed)
                holder = []

                def factory(*factory_args, **factory_kwargs):
                    tracker = SensorNoiseTracker(*factory_args, **factory_kwargs)
                    holder.append(tracker)
                    return tracker

                print(f"RUN {scenario=} {seed=} {condition=}", flush=True)
                data = run_mppi(
                    waypoints("circle", n_steps=args.steps),
                    has_disturbance=scenario != "nominal",
                    use_dob=True, dob_compensation_mode="output",
                    disturbance_function=disturbance, require_cuda=True,
                    seed=seed, temperature=1.0, temporal_noise_rho=0.8,
                    joint_position_limit_weight=50000.0,
                    joint_velocity_limit_weight=50000.0,
                    tracker_class=factory,
                    tracker_options={
                        "sensor_noise": sensor_cfg,
                        "torque_scale": scenario_cfg["torque_scale"],
                        "dob_acceleration_lpf_cutoff_hz":
                            cfg["acceleration_lpf_cutoff_hz"],
                        "dob_cutoff_hz": cfg["dob_cutoff_hz"],
                    },
                )
                data.update(holder[0].export_sensor_data())
                data["true_state"] = np.concatenate(
                    (np.asarray(data["q"]), np.asarray(data["qd"])), axis=1
                )
                data["state_clamp_enabled"] = np.asarray(False)
                data["dynamic_state_margin_enabled"] = np.asarray(False)
                row = {
                    "scenario": scenario, "seed": seed,
                    "condition": condition, "sensor_noise": sensor_cfg,
                    "dob_acceleration_lpf_cutoff_hz":
                        cfg["acceleration_lpf_cutoff_hz"],
                    "dob_cutoff_hz": cfg["dob_cutoff_hz"],
                    "metrics": measure(data, scenario_cfg["torque_scale"]),
                }
                rows.append(row)
                stem = f"{scenario}_seed{seed}_{condition}"
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

    summary = aggregate(rows)
    selection = select_cutoff(summary, list(dict.fromkeys(args.cutoffs)))
    (output / "aggregate.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    (output / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    lines = [
        "# Exact-ZOH DOB cutoff comparison", "",
        f"Selected cutoff: **{selection['selected_cutoff_hz']:g} Hz**", "",
        "| Scenario | Condition | RMSE [mm] | torque-rate [Nm/s] | "
        "DOB RMSE [Nm] | d_hat RMS [Nm] | saturation steps |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scenario in args.scenarios:
        for condition in conditions:
            item = summary[scenario][condition]
            value = lambda field: item[field]["mean"]
            lines.append(
                f"| {scenario} | {condition} | "
                f"{value('tracking_rmse_mm'):.3f} | "
                f"{value('torque_rate_rms_nm_s'):.3f} | "
                f"{value('dob_estimation_rmse_nm'):.3f} | "
                f"{value('dob_estimate_rms_nm'):.3f} | "
                f"{value('saturation_steps'):.3f} |"
            )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(selection, indent=2), flush=True)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
