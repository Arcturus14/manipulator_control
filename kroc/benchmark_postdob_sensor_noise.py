"""GPU A/B of clean-state and noisy-measurement Post-DOB without dynamic margins."""
from __future__ import annotations

import argparse
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight, disturbance_profiles
from config import (DOB_ALPHA, DQ_MAX, DT, Q_MAX, Q_MIN, TAU_MAX,
                    TIMING_WARMUP_STEPS)
from experiment_runner import run_mppi, waypoints
from kroc.benchmark_state_margin import SCENARIOS
from kroc.sensor_noise import SensorNoiseTracker


def measure(data, torque_scale):
    q, dq = np.asarray(data["q"])[1:], np.asarray(data["qd"])[1:]
    measured = np.asarray(data["measured_state"])
    true = np.concatenate((np.asarray(data["q"]), np.asarray(data["qd"])), axis=1)
    measurement_error = measured - true
    d_error = np.asarray(data["d_hat"])[1:] - np.asarray(data["d_true"])[1:]
    applied = np.asarray(data["u_applied"])
    limits = np.asarray(TAU_MAX) * torque_scale
    saturated = np.isclose(np.abs(applied), limits, rtol=0, atol=1e-5).any(1)
    solve_ms = np.asarray(data["solve_t_raw"])[TIMING_WARMUP_STEPS:] * 1e3
    q_violated = ((q < np.asarray(Q_MIN)) | (q > np.asarray(Q_MAX))).any(1)
    dq_violated = (np.abs(dq) > DQ_MAX).any(1)
    error = np.asarray(data["err_mm"])
    raw_acceleration = np.asarray(data["dob_acceleration_raw"])[1:]
    filtered_acceleration = np.asarray(
        data["dob_acceleration_filtered"]
    )[1:]
    return {
        "tracking_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "tracking_max_mm": float(error.max()),
        "position_violation_steps": int(q_violated.sum()),
        "velocity_violation_steps": int(dq_violated.sum()),
        "velocity_abs_max_rad_s": float(np.abs(dq).max()),
        "saturation_steps": int(saturated.sum()),
        "saturation_rate": float(saturated.mean()),
        "torque_rate_rms_nm_s": float(np.sqrt(np.mean(
            (np.diff(applied, axis=0) / DT) ** 2
        ))),
        "dob_estimation_rmse_nm": float(np.sqrt(np.mean(d_error ** 2))),
        "dob_estimation_peak_nm": float(np.abs(d_error).max()),
        "dob_estimate_rms_nm": float(np.sqrt(np.mean(
            np.asarray(data["d_hat"])[1:] ** 2
        ))),
        "dob_raw_acceleration_rms_rad_s2": float(np.sqrt(np.mean(
            raw_acceleration ** 2
        ))),
        "dob_filtered_acceleration_rms_rad_s2": float(np.sqrt(np.mean(
            filtered_acceleration ** 2
        ))),
        "realized_position_noise_rms_rad": float(np.sqrt(np.mean(
            measurement_error[:, :6] ** 2
        ))),
        "realized_velocity_noise_rms_rad_s": float(np.sqrt(np.mean(
            measurement_error[:, 6:] ** 2
        ))),
        "solve_mean_ms": float(solve_ms.mean()),
        "solve_p95_ms": float(np.percentile(solve_ms, 95)),
        "solve_deadline_miss_rate": float((solve_ms > DT * 1e3).mean()),
    }


def aggregate(rows):
    result = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        result[scenario] = {}
        conditions = sorted({
            row["condition"] for row in rows if row["scenario"] == scenario
        })
        for condition in conditions:
            selected = [row["metrics"] for row in rows
                        if row["scenario"] == scenario
                        and row["condition"] == condition]
            result[scenario][condition] = {}
            for field, value in selected[0].items():
                if not isinstance(value, (int, float)):
                    continue
                values = np.asarray([item[field] for item in selected], dtype=float)
                result[scenario][condition][field] = {
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.,
                    "max": float(values.max()),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=list(SCENARIOS))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--position-std", type=float, default=1e-3)
    parser.add_argument("--velocity-std", type=float, default=1e-2)
    parser.add_argument(
        "--acceleration-cutoffs", nargs="+", type=float,
        default=[0.0, 2.0, 5.0, 10.0],
        help="DOB acceleration LPF cutoffs in Hz; 0 keeps the raw derivative",
    )
    parser.add_argument(
        "--dob-alphas", nargs="+", type=float, default=[DOB_ALPHA],
        help="DOB Q-filter gains to compare",
    )
    args = parser.parse_args()
    if args.steps < 10:
        parser.error("steps must be >= 10")
    if args.position_std < 0 or args.velocity_std < 0:
        parser.error("noise standard deviations must be nonnegative")
    if any(not np.isfinite(value) or value < 0
           for value in args.acceleration_cutoffs):
        parser.error("acceleration cutoffs must be finite and nonnegative")
    if any(not np.isfinite(value) or value <= 0 for value in args.dob_alphas):
        parser.error("DOB alphas must be finite and positive")

    conditions = {
        "clean": {
            "sensor_noise": {
                "position_std_rad": 0., "velocity_std_rad_s": 0.
            },
            "acceleration_lpf_cutoff_hz": 0.0,
            "dob_alpha": DOB_ALPHA,
        },
    }
    for cutoff in dict.fromkeys(args.acceleration_cutoffs):
        for alpha in dict.fromkeys(args.dob_alphas):
            base = "noisy_raw" if cutoff == 0 else f"noisy_lpf_{cutoff:g}hz"
            if len(args.dob_alphas) == 1 and alpha == DOB_ALPHA:
                label = base
            else:
                label = f"{base}_alpha{alpha:g}"
            conditions[label] = {
                "sensor_noise": {
                    "position_std_rad": args.position_std,
                    "velocity_std_rad_s": args.velocity_std,
                },
                "acceleration_lpf_cutoff_hz": cutoff,
                "dob_alpha": alpha,
            }
    preflight = cuda_preflight()
    root = Path(__file__).resolve().parent
    output = root / "results" / (
        "postdob_sensor_noise_"
        + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), root / "sensor_noise.py",
               Path("controllers/mppi_dob_controller.py"),
               Path("observers/dob.py"), Path("experiment_runner.py")]
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
        "conditions": conditions, "scenarios": SCENARIOS,
        "controller": "Post-DOB output compensation",
        "dynamic_state_margin": False,
        "measurement_model": "independent zero-mean Gaussian q and dq noise",
        "noise_values_are_ur5_specifications": False,
        "settings": {
            "trajectory": "circle", "horizon": 10, "samples": 136,
            "temperature": 1., "temporal_noise_rho": .8,
            "static_soft_state_cost": True,
            "joint_position_limit_weight": 50000.,
            "joint_velocity_limit_weight": 50000.,
            "dob_alpha": DOB_ALPHA,
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("OUTPUT", output, flush=True)

    rows, representative = [], {}
    for scenario_index, scenario in enumerate(args.scenarios):
        scenario_cfg = SCENARIOS[scenario]
        disturbance = disturbance_profiles(
            args.steps, scenario_cfg["disturbance_scale"]
        )["large_step"]
        for seed_index, seed in enumerate(args.seeds):
            names = list(conditions)
            if (scenario_index + seed_index) % 2:
                names.reverse()
            for condition in names:
                condition_cfg = conditions[condition]
                sensor_cfg = dict(
                    condition_cfg["sensor_noise"], seed=10000 + seed
                )
                cutoff = condition_cfg["acceleration_lpf_cutoff_hz"]
                dob_alpha = condition_cfg["dob_alpha"]
                holder = []

                def factory(*factory_args, **factory_kwargs):
                    tracker = SensorNoiseTracker(*factory_args, **factory_kwargs)
                    holder.append(tracker)
                    return tracker

                print(
                    f"RUN {scenario=} {seed=} {condition=} {cutoff=} "
                    f"{dob_alpha=}",
                    flush=True,
                )
                data = run_mppi(
                    waypoints("circle", n_steps=args.steps),
                    has_disturbance=scenario_cfg["disturbance_scale"] > 0,
                    use_dob=True, dob_compensation_mode="output",
                    disturbance_function=disturbance, require_cuda=True,
                    seed=seed, temperature=1., temporal_noise_rho=.8,
                    joint_position_limit_weight=50000.,
                    joint_velocity_limit_weight=50000.,
                    tracker_class=factory,
                    tracker_options={"sensor_noise": sensor_cfg,
                                     "torque_scale": scenario_cfg["torque_scale"],
                                     "dob_acceleration_lpf_cutoff_hz": cutoff,
                                     "dob_alpha": dob_alpha},
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
                    "dob_acceleration_lpf_cutoff_hz": cutoff,
                    "dob_alpha": dob_alpha,
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
                if seed == args.seeds[0]:
                    representative[(scenario, condition)] = {
                        key: data[key] for key in
                        ("err_mm", "qd", "d_hat", "d_true")
                    }
                del data, holder
                gc.collect()
                torch.cuda.empty_cache()

    summary = aggregate(rows)
    (output / "aggregate.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    lines = [
        "# Post-DOB clean state vs sensor noise", "",
        "No dynamic state margin is used. Values are means across seeds.", "",
        "| Scenario | Condition | RMSE [mm] | torque-rate [Nm/s] | q violation | dq violation | saturation steps | DOB RMSE [Nm] | filtered accel RMS [rad/s^2] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in args.scenarios:
        for condition in conditions:
            metrics = [row["metrics"] for row in rows
                       if row["scenario"] == scenario
                       and row["condition"] == condition]
            mean = lambda field: np.mean([item[field] for item in metrics])
            lines.append(
                f"| {scenario} | {condition} | "
                f"{mean('tracking_rmse_mm'):.3f} | "
                f"{mean('torque_rate_rms_nm_s'):.3f} | "
                f"{mean('position_violation_steps'):.3f} | "
                f"{mean('velocity_violation_steps'):.3f} | "
                f"{mean('saturation_steps'):.3f} | "
                f"{mean('dob_estimation_rmse_nm'):.3f} | "
                f"{mean('dob_filtered_acceleration_rms_rad_s2'):.3f} |"
            )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")

    for scenario in args.scenarios:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for condition in conditions:
            data = representative[(scenario, condition)]
            time = np.arange(len(data["err_mm"])) * DT
            axes[0].plot(time, data["err_mm"], label=condition)
            axes[1].plot(time, np.abs(data["qd"][1:]).max(1), label=condition)
            d_error = data["d_hat"] - data["d_true"]
            axes[2].plot(np.arange(len(d_error)) * DT,
                         np.linalg.norm(d_error, axis=1), label=condition)
        axes[0].set_ylabel("EE error [mm]")
        axes[1].set_ylabel("max |dq| [rad/s]")
        axes[2].set_ylabel("||d_hat-d|| [Nm]")
        axes[2].set_xlabel("Time [s]")
        for axis in axes:
            axis.grid(alpha=.25)
            axis.legend()
        fig.suptitle(f"Post-DOB sensor noise: {scenario}, seed {args.seeds[0]}")
        fig.tight_layout()
        fig.savefig(output / f"{scenario}.png", dpi=160)
        plt.close(fig)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
