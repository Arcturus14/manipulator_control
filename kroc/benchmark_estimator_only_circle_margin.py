"""CUDA comparison of fixed 5% and dynamic 5--10% margins.

The DOB estimates d_hat for margin generation, but d_hat is not applied at the
controller output and is not inserted into the MPPI rollout model.
"""
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

from benchmarks.benchmark_dob_aware_nominal_shift import (
    cuda_preflight, disturbance_profiles,
)
from config import DT, Q_MAX, Q_MIN, DQ_MAX, TAU_MAX, TIMING_WARMUP_STEPS
from experiment_runner import run_mppi, waypoints
from kroc.benchmark_noise_aware_state_margin import ACCELERATION_DEADBANDS
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker


CASES = {
    "fixed_5pct": MarginConfig(
        mode="fixed", min_fraction=.05, fixed_fraction=.05,
        max_fraction=.10, penalty_kind="hinge",
    ),
    "dynamic_5_to_10pct": MarginConfig(
        mode="directional", min_fraction=.05, fixed_fraction=.05,
        max_fraction=.10, reaction_time_s=.10,
        acceleration_deadband_rad_s2=ACCELERATION_DEADBANDS["nominal_p99"],
        penalty_kind="hinge", beta_up=1.0, beta_down=.10,
    ),
}


def metrics(data: dict) -> dict:
    q = np.asarray(data["q"])[1:]
    dq = np.asarray(data["qd"])[1:]
    err = np.asarray(data["err_mm"])
    u = np.asarray(data["u_applied"])
    sat = np.isclose(np.abs(u), np.asarray(TAU_MAX), rtol=0, atol=1e-5)
    solve_ms = np.asarray(data["solve_t_raw"])[TIMING_WARMUP_STEPS:] * 1e3
    q_distance = np.minimum(q - np.asarray(Q_MIN), np.asarray(Q_MAX) - q)
    dq_distance = DQ_MAX - np.abs(dq)
    return {
        "tracking_rmse_mm": float(np.sqrt(np.mean(err ** 2))),
        "tracking_mean_mm": float(err.mean()),
        "tracking_max_mm": float(err.max()),
        "position_minimum_distance_rad": float(q_distance.min()),
        "position_violation_steps": int((q_distance < 0).any(axis=1).sum()),
        "velocity_minimum_distance_rad_s": float(dq_distance.min()),
        "velocity_violation_steps": int((dq_distance < 0).any(axis=1).sum()),
        "saturation_steps": int(sat.any(axis=1).sum()),
        "saturation_joint_steps": sat.sum(axis=0).tolist(),
        "torque_rms_nm": float(np.sqrt(np.mean(u ** 2))),
        "torque_rate_rms_nm_s": float(np.sqrt(np.mean((np.diff(u, axis=0) / DT) ** 2))),
        "compute_mean_ms": float(solve_ms.mean()),
        "compute_p95_ms": float(np.percentile(solve_ms, 95)),
        "compute_max_ms": float(solve_ms.max()),
        "deadline_miss_rate": float((solve_ms > DT * 1e3).mean()),
        "dob_rmse_nm": float(np.sqrt(np.mean(
            (np.asarray(data["d_hat"])[1:] - np.asarray(data["d_true"])[1:]) ** 2
        ))),
    }


def stat(values) -> dict:
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max())}


def plot_results(output: Path, rows: list[dict], representative: dict,
                 reference: np.ndarray, disturbance_scale: float) -> None:
    colors = {"fixed_5pct": "tab:orange", "dynamic_5_to_10pct": "tab:blue"}
    labels = {"fixed_5pct": "Static 5%", "dynamic_5_to_10pct": "Dynamic 5–10%"}
    fig = plt.figure(figsize=(14, 8.5))
    ax = fig.add_subplot(231, projection="3d")
    ax.plot(reference[:, 0], reference[:, 1], reference[:, 2], "k--", label="reference")
    for case, data in representative.items():
        ee = data["ee"]
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=colors[case], label=labels[case])
    ax.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]", title="Final trajectory (seed 17)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(232)
    for case, data in representative.items():
        t = np.arange(len(data["err_mm"])) * DT
        ax.plot(t, data["err_mm"], color=colors[case], label=labels[case])
    ax.axvline(1.25, color="k", linestyle="--", linewidth=1, label="step disturbance")
    ax.set(xlabel="time [s]", ylabel="EE error [mm]", title="Tracking error (seed 17)")
    ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = fig.add_subplot(233)
    for case, data in representative.items():
        t = np.arange(len(data["u_applied"])) * DT
        ratio = np.max(np.abs(data["u_applied"]) / np.asarray(TAU_MAX), axis=1)
        ax.plot(t, ratio, color=colors[case], label=labels[case])
    ax.axhline(1, color="k", linestyle="--")
    ax.set(xlabel="time [s]", ylabel="max |torque| / limit", title="Actuator utilization (seed 17)")
    ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = fig.add_subplot(234)
    for case, data in representative.items():
        t = np.arange(len(data["kroc_margin"])) * DT
        max_q_fraction = np.max(data["kroc_margin"][:, 0] /
                                (np.asarray(Q_MAX) - np.asarray(Q_MIN))[None, None, :], axis=(1, 2))
        ax.plot(t, 100 * max_q_fraction, color=colors[case], label=labels[case])
    ax.set(xlabel="time [s]", ylabel="largest position margin [% span]", title="Applied position margin")
    ax.grid(alpha=.25); ax.legend(fontsize=8)

    ax = fig.add_subplot(235)
    positions = [1, 2]
    for pos, case in zip(positions, CASES):
        vals = [r["metrics"]["compute_mean_ms"] for r in rows if r["case"] == case]
        ax.scatter([pos] * len(vals), vals, color=colors[case])
        ax.hlines(np.mean(vals), pos-.2, pos+.2, color="k")
    ax.axhline(DT * 1000, color="k", linestyle="--", label="20 ms deadline")
    ax.set(xticks=positions, xticklabels=[labels[c] for c in CASES], ylabel="mean solve time [ms]", title="Compute time (5 seeds)")
    ax.grid(alpha=.25, axis="y"); ax.legend(fontsize=8)

    ax = fig.add_subplot(236)
    x = np.arange(2)
    sat_means = [np.mean([r["metrics"]["saturation_steps"] for r in rows if r["case"] == c]) for c in CASES]
    rmse_means = [np.mean([r["metrics"]["tracking_rmse_mm"] for r in rows if r["case"] == c]) for c in CASES]
    bars = ax.bar(x-.18, sat_means, .36, color=[colors[c] for c in CASES], alpha=.8, label="saturation steps")
    ax.set(xticks=x, xticklabels=[labels[c] for c in CASES], ylabel="saturation steps", title="Aggregate outcomes")
    ax.set_ylim(0, max(1.0, 1.2 * max(sat_means)))
    for index, value in enumerate(sat_means):
        ax.text(index - .18, .03, f"{value:.1f}", ha="center", va="bottom", fontsize=8)
    ax2 = ax.twinx(); ax2.bar(x+.18, rmse_means, .36, color="gray", alpha=.55, label="RMSE")
    ax2.set_ylabel("tracking RMSE [mm]")
    ax.grid(alpha=.2, axis="y")
    fig.suptitle("Estimator-only DOB: static 5% vs dynamic 5–10% margin\n"
                 f"(no DOB torque or rollout compensation; noisy sensors; {disturbance_scale:g}× step disturbance)")
    fig.tight_layout()
    fig.savefig(output / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 27, 37, 47, 57])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--disturbance-scale", type=float, default=4.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.steps < 80 or args.disturbance_scale <= 0:
        parser.error("steps must be >=80 and disturbance scale positive")
    cuda = cuda_preflight()
    root = Path(__file__).resolve().parent
    output = args.output or root / "results" / (
        "estimator_only_circle_margin_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    )
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), root / "state_margin.py", root / "sensor_noise.py",
               Path("controllers/mppi_dob_controller.py"), Path("observers/dob.py"),
               Path("costs/tracking_cost.py"), Path("experiment_runner.py")]
    manifest = {
        "created_at": datetime.now().astimezone().isoformat(), "arguments": {**vars(args), "output": str(args.output)},
        "cuda": cuda, "controller": "MPPI + estimator-only DOB", "dob_compensation": False,
        "rollout_disturbance_compensation": False, "trajectory": "circle",
        "disturbance": {"profile": "large_step", "scale": args.disturbance_scale,
                        "onset_step": args.steps // 4},
        "sensor_noise": {"q_std_rad": .001, "dq_std_rad_s": .01},
        "dob": {"acceleration_lpf_hz": 10., "exact_zoh_q_cutoff_hz": 8.},
        "cases": {name: asdict(cfg) for name, cfg in CASES.items()},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
        "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("OUTPUT", output, flush=True)
    reference = waypoints("circle", n_steps=args.steps)
    disturbance = disturbance_profiles(args.steps, args.disturbance_scale)["large_step"]
    rows, representative = [], {}
    for seed_index, seed in enumerate(args.seeds):
        names = list(CASES)
        names = names[seed_index % 2:] + names[:seed_index % 2]
        for case in names:
            holder = []
            def factory(*a, **kw):
                tracker = NoiseAwareKrocTracker(*a, **kw); holder.append(tracker); return tracker
            print(f"RUN seed={seed} case={case}", flush=True)
            data = run_mppi(
                reference, has_disturbance=True, use_dob=True,
                dob_compensation_mode="estimate_only", disturbance_function=disturbance,
                require_cuda=True, seed=seed, temperature=1.0, temporal_noise_rho=.8,
                joint_position_limit_weight=50000., joint_velocity_limit_weight=50000.,
                tracker_class=factory, tracker_options={
                    "margin_config": asdict(CASES[case]),
                    "sensor_noise": {"position_std_rad": .001, "velocity_std_rad_s": .01,
                                     "seed": 30000 + seed},
                    "dob_acceleration_lpf_cutoff_hz": 10., "dob_cutoff_hz": 8.,
                },
            )
            data.update(holder[0].export_margin_data()); data.update(holder[0].export_sensor_data())
            data["true_state"] = np.concatenate((data["q"], data["qd"]), axis=1)
            row = {"seed": seed, "case": case, "metrics": metrics(data)}
            rows.append(row)
            np.savez_compressed(output / f"seed{seed}_{case}.npz", **{k: v for k, v in data.items() if not k.startswith("_")})
            (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            if seed == args.seeds[0]:
                representative[case] = {k: np.asarray(data[k]) for k in ("ee", "err_mm", "u_applied", "kroc_margin")}
            del data, holder; gc.collect(); torch.cuda.empty_cache()
    fields = list(rows[0]["metrics"])
    aggregate = {case: {field: stat([r["metrics"][field] for r in rows if r["case"] == case])
                        for field in fields if field != "saturation_joint_steps"} for case in CASES}
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    plot_results(output, rows, representative, reference, args.disturbance_scale)
    lines = ["# Estimator-only DOB circle margin comparison", "",
             "DOB d_hat is used only for dynamic-margin generation; output and rollout compensation are disabled.", "",
             "| case | RMSE [mm] | max error [mm] | q violation [step] | dq violation [step] | saturation [step] | solve mean [ms] | solve p95 [ms] |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case in CASES:
        a = aggregate[case]
        f = lambda key: f"{a[key]['mean']:.3f} ± {a[key]['sample_std']:.3f}"
        lines.append(f"| {case} | {f('tracking_rmse_mm')} | {f('tracking_max_mm')} | {f('position_violation_steps')} | {f('velocity_violation_steps')} | {f('saturation_steps')} | {f('compute_mean_ms')} | {f('compute_p95_ms')} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
