"""GPU A/B: post-DOB versus model-only with the same dynamic state margin."""
from __future__ import annotations

import argparse
from dataclasses import asdict
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
from config import DT, TAU_MAX
from experiment_runner import run_mppi, waypoints
from kroc.benchmark_state_margin import SCENARIOS, measure
from kroc.state_margin import KrocTracker, MarginConfig


ARCHITECTURES = {
    "post_dob": "output",
    "model_only": "rollout_model_only",
}


def aggregate(rows):
    fields = (
        "tracking_rmse_mm", "tracking_max_mm", "position_violation_steps",
        "velocity_violation_steps", "velocity_abs_max_rad_s",
        "saturation_steps", "saturation_rate", "torque_rate_rms_nm_s",
        "compute_mean_ms", "compute_p95_ms",
    )
    result = {}
    for scenario in sorted({row["scenario"] for row in rows}):
        result[scenario] = {}
        for architecture in ARCHITECTURES:
            selected = [row["metrics"] for row in rows
                        if row["scenario"] == scenario
                        and row["architecture"] == architecture]
            result[scenario][architecture] = {
                field: {
                    "mean": float(np.mean([item[field] for item in selected])),
                    "std": float(np.std([item[field] for item in selected], ddof=1))
                    if len(selected) > 1 else 0.0,
                    "max": float(np.max([item[field] for item in selected])),
                }
                for field in fields
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS,
                        default=list(SCENARIOS))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--margin-mode", choices=("directional", "headroom"),
                        default="headroom")
    parser.add_argument("--min-fraction", type=float, default=.025)
    parser.add_argument("--max-fraction", type=float, default=.10)
    parser.add_argument("--reaction-time", type=float, default=.10)
    args = parser.parse_args()
    if args.steps < 10:
        parser.error("steps must be >= 10")

    preflight = cuda_preflight()
    root = Path(__file__).resolve().parent
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output = root / "results" / f"postdob_vs_modelonly_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), root / "state_margin.py",
               Path("controllers/mppi_dob_controller.py"),
               Path("experiment_runner.py"), Path("config.py")]
    cfg = MarginConfig(
        mode=args.margin_mode, min_fraction=args.min_fraction,
        max_fraction=args.max_fraction, reaction_time_s=args.reaction_time,
    )
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
        "architectures": ARCHITECTURES, "scenarios": SCENARIOS,
        "margin_config": asdict(cfg),
        "controlled_variables": {
            "trajectory": "circle", "horizon": 10, "samples": 136,
            "temperature": 1.0, "temporal_noise_rho": .8,
            "joint_position_limit_weight": 50000.0,
            "joint_velocity_limit_weight": 50000.0,
            "torque_headroom_reference": "unclamped U_ff + R; d_hat not added",
            "margin_disturbance_source": "DOB d_hat in both architectures",
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("OUTPUT", output, flush=True)

    rows = []
    representative = {}
    for scenario_index, scenario in enumerate(args.scenarios):
        settings = SCENARIOS[scenario]
        disturbance = disturbance_profiles(
            args.steps, settings["disturbance_scale"]
        )["large_step"]
        for seed_index, seed in enumerate(args.seeds):
            names = list(ARCHITECTURES)
            if (scenario_index + seed_index) % 2:
                names.reverse()
            for architecture in names:
                holder = []

                def factory(*factory_args, **factory_kwargs):
                    tracker = KrocTracker(*factory_args, **factory_kwargs)
                    holder.append(tracker)
                    return tracker

                print(f"RUN {scenario=} {seed=} {architecture=}", flush=True)
                data = run_mppi(
                    waypoints("circle", n_steps=args.steps),
                    has_disturbance=settings["disturbance_scale"] > 0,
                    use_dob=True,
                    dob_compensation_mode=ARCHITECTURES[architecture],
                    disturbance_function=disturbance, require_cuda=True,
                    seed=seed, temperature=1., temporal_noise_rho=.8,
                    joint_position_limit_weight=50000.,
                    joint_velocity_limit_weight=50000.,
                    tracker_class=factory,
                    tracker_options={"margin_config": asdict(cfg),
                                     "torque_scale": settings["torque_scale"]},
                )
                data.update(holder[0].export_margin_data())
                data["qd_raw"] = data["qd"].copy()
                data["state_clamp_enabled"] = np.asarray(False)
                data["physical_torque_limit"] = (
                    np.asarray(TAU_MAX) * settings["torque_scale"]
                )
                metrics = measure(data, settings["torque_scale"])
                row = {"scenario": scenario, "seed": seed,
                       "architecture": architecture,
                       "dob_compensation_mode": ARCHITECTURES[architecture],
                       "margin_config": asdict(cfg), "metrics": metrics}
                rows.append(row)
                stem = f"{scenario}_seed{seed}_{architecture}_{args.margin_mode}"
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
                    representative[(scenario, architecture)] = {
                        key: data[key] for key in
                        ("qd", "err_mm", "u_applied", "kroc_margin")
                    }
                del data, holder
                gc.collect()
                torch.cuda.empty_cache()

    summary = aggregate(rows)
    (output / "aggregate.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    lines = [
        "# Post-DOB vs model-only with dynamic state margin", "",
        f"Margin mode: `{args.margin_mode}`. Values are means across seeds.", "",
        "| Scenario | Architecture | RMSE [mm] | q violation steps | dq violation steps | max speed [rad/s] | saturation steps | solve mean [ms] | solve p95 [ms] |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario in args.scenarios:
        for architecture in ARCHITECTURES:
            selected = [row["metrics"] for row in rows
                        if row["scenario"] == scenario
                        and row["architecture"] == architecture]
            mean = lambda field: np.mean([item[field] for item in selected])
            lines.append(
                f"| {scenario} | {architecture} | "
                f"{mean('tracking_rmse_mm'):.3f} | "
                f"{mean('position_violation_steps'):.3f} | "
                f"{mean('velocity_violation_steps'):.3f} | "
                f"{mean('velocity_abs_max_rad_s'):.3f} | "
                f"{mean('saturation_steps'):.3f} | "
                f"{mean('compute_mean_ms'):.3f} | "
                f"{mean('compute_p95_ms'):.3f} |"
            )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")

    for scenario in args.scenarios:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for architecture in ARCHITECTURES:
            data = representative[(scenario, architecture)]
            time = np.arange(len(data["err_mm"])) * DT
            axes[0].plot(time, data["err_mm"], label=architecture)
            axes[1].plot(time, np.abs(data["qd"][1:]).max(1),
                         label=architecture)
            limits = np.asarray(TAU_MAX) * SCENARIOS[scenario]["torque_scale"]
            axes[2].plot(time,
                         (np.abs(data["u_applied"]) / limits).max(1),
                         label=architecture)
        axes[0].set_ylabel("EE error [mm]")
        axes[1].set_ylabel("max |dq| [rad/s]")
        axes[2].set_ylabel("max torque / limit")
        axes[2].set_xlabel("Time [s]")
        axes[2].axhline(1, color="black", linestyle="--")
        for axis in axes:
            axis.grid(alpha=.25)
            axis.legend()
        fig.suptitle(
            f"{scenario}: same {args.margin_mode} margin, seed {args.seeds[0]}"
        )
        fig.tight_layout()
        fig.savefig(output / f"{scenario}.png", dpi=160)
        plt.close(fig)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
