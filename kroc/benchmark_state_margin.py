"""Run: python -m kroc.benchmark_state_margin --seeds 7 17 27 --caps .05 .10 .15"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
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
from config import DT, Q_MIN, Q_MAX, DQ_MAX, TAU_MAX, TIMING_WARMUP_STEPS
from experiment_runner import run_mppi, waypoints
from kroc.state_margin import KrocTracker, MarginConfig


SCENARIOS = {
    "nominal": {"disturbance_scale": 0., "torque_scale": 1.},
    "stress": {"disturbance_scale": 4., "torque_scale": 1.},
    "limited": {"disturbance_scale": 2., "torque_scale": .4},
}


def measure(data, torque_scale):
    q, dq = data["q"][1:], data["qd"][1:]
    lower, upper = np.asarray(Q_MIN), np.asarray(Q_MAX)
    q_distance = np.minimum(q - lower, upper - q)
    dq_distance = DQ_MAX - np.abs(dq)
    margin = data["kroc_margin"]
    physical = np.asarray(TAU_MAX) * torque_scale
    sat = np.isclose(np.abs(data["u_applied"]), physical, rtol=0, atol=1e-5)
    times = np.asarray(data["solve_t_raw"])[TIMING_WARMUP_STEPS:] * 1000
    cycles = np.asarray(data["control_cycle_t_raw"])[TIMING_WARMUP_STEPS:] * 1000
    error = np.asarray(data["err_mm"])
    if not all(np.isfinite(data[key]).all() for key in ("q", "qd", "err_mm", "kroc_margin")):
        raise RuntimeError("nonfinite closed-loop result")
    result = {
        "tracking_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "tracking_max_mm": float(error.max()),
        "velocity_abs_max_rad_s": float(np.abs(dq).max()),
        "torque_rate_rms_nm_s": float(np.sqrt(np.mean((np.diff(data["u_applied"], axis=0) / DT) ** 2))),
        "saturation_steps": int(sat.any(axis=1).sum()),
        "saturation_joint_steps": sat.sum(axis=0).tolist(),
        "saturation_duration_s": float(sat.any(axis=1).sum() * DT),
        "saturation_rate": float(sat.any(axis=1).mean()),
        "compute_mean_ms": float(times.mean()),
        "compute_p95_ms": float(np.percentile(times, 95)),
        "compute_max_ms": float(times.max()),
        "deadline_miss_rate": float((times > DT * 1000).mean()),
        "full_cycle_p95_ms": float(np.percentile(cycles, 95)),
        "full_cycle_deadline_miss_rate": float((cycles > DT * 1000).mean()),
        "margin_variation_l1": float(np.abs(np.diff(margin, axis=0)).sum()),
    }
    for name, distance in (("position", q_distance), ("velocity", dq_distance)):
        violation = np.maximum(-distance, 0)
        result.update({
            f"{name}_violation_steps": int((distance < 0).any(axis=1).sum()),
            f"{name}_violation_joint_steps": (distance < 0).sum(axis=0).tolist(),
            f"{name}_violation_max": float(violation.max()),
            f"{name}_violation_integral": float(violation.sum() * DT),
            f"{name}_minimum_distance": float(distance.min()),
            f"{name}_violation_max_by_joint": violation.max(axis=0).tolist(),
        })
    internal_q = np.minimum(q - (lower + margin[:, 0, 0]),
                            upper - margin[:, 0, 1] - q)
    internal_dq = np.minimum(dq + DQ_MAX - margin[:, 1, 0],
                             DQ_MAX - margin[:, 1, 1] - dq)
    result["internal_position_minimum_distance"] = float(internal_q.min())
    result["internal_velocity_minimum_distance"] = float(internal_dq.min())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--caps", nargs="+", type=float, default=[.10])
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--min-fraction", type=float, default=.025)
    parser.add_argument("--fixed-fraction", type=float, default=.05)
    parser.add_argument("--reaction-time", type=float, default=.10)
    args = parser.parse_args()
    if args.steps < 10:
        parser.error("steps must be >= 10")
    preflight = cuda_preflight()
    root = Path(__file__).resolve().parent
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output = root / "results" / stamp
    output.mkdir(parents=True, exist_ok=False)
    sources = [*root.glob("*.py"), Path("controllers/mppi_dob_controller.py"),
               Path("costs/tracking_cost.py"), Path("dynamics/ur5_dynamics.py"),
               Path("experiment_runner.py"), Path("config.py")]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}
    manifest = {"created_at": datetime.now().astimezone().isoformat(), "arguments": vars(args),
                "cuda": preflight, "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
                "source_hashes": source_hashes, "scenarios": SCENARIOS,
                "state_clamp": False, "margin_percent_basis": "full physical span",
                "settings": {"horizon": 10, "samples": 136, "temperature": 1., "temporal_noise_rho": .8,
                             "joint_position_limit_weight": 50000., "joint_velocity_limit_weight": 50000.},
                "timing": "prepare margin + diagnostic GPU clones + MPPI; excludes observer, plant and CPU logging"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("OUTPUT", output, flush=True)
    rows = []
    representative = {}
    for scenario in args.scenarios:
        settings = SCENARIOS[scenario]
        for cap in args.caps:
            for seed_index, seed in enumerate(args.seeds):
                modes = ["fixed", "directional", "headroom"]
                modes = modes[seed_index % 3:] + modes[:seed_index % 3]
                for mode in modes:
                    cfg = MarginConfig(mode=mode, max_fraction=cap,
                                       min_fraction=args.min_fraction,
                                       fixed_fraction=args.fixed_fraction,
                                       reaction_time_s=args.reaction_time)
                    holder = []
                    def factory(*a, **kw):
                        tracker = KrocTracker(*a, **kw)
                        holder.append(tracker)
                        return tracker
                    print(f"RUN {scenario=} {cap=} {seed=} {mode=}", flush=True)
                    disturbance = disturbance_profiles(args.steps, settings["disturbance_scale"])["large_step"]
                    data = run_mppi(
                        waypoints("circle", n_steps=args.steps), has_disturbance=settings["disturbance_scale"] > 0,
                        use_dob=True, dob_compensation_mode="rollout_model_only",
                        disturbance_function=disturbance, require_cuda=True, seed=seed,
                        temperature=1., temporal_noise_rho=.8,
                        tracker_class=factory, tracker_options={"margin_config": asdict(cfg),
                                                               "torque_scale": settings["torque_scale"]})
                    data.update(holder[0].export_margin_data())
                    # There is no state projection: qd is the raw RK4 state.
                    data["qd_raw"] = data["qd"].copy()
                    data["state_clamp_enabled"] = np.asarray(False)
                    data["physical_torque_limit"] = np.asarray(TAU_MAX) * settings["torque_scale"]
                    row = {"scenario": scenario, "cap": cap, "seed": seed, "mode": mode,
                           "margin_config": asdict(cfg), "metrics": measure(data, settings["torque_scale"])}
                    stem = f"{scenario}_cap{cap:.3f}_seed{seed}_{mode}"
                    np.savez_compressed(output / f"{stem}.npz", **{k: v for k, v in data.items() if not k.startswith("_")})
                    rows.append(row)
                    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                    if seed == args.seeds[0] and cap == args.caps[0]:
                        representative[(scenario, mode)] = {key: data[key] for key in ("qd", "err_mm", "u_applied", "kroc_margin")}
                    del data, holder
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
    lines = ["# KRoC dynamic margin pilot", "", "GPU: " + preflight["gpu_name"], "",
             "Each cell is mean across requested seeds. This is a pilot, not evidence of general safety.", "",
             "| Scenario | Cap/full span | Mode | RMSE mm | q violation steps | dq violation steps | max speed rad/s | saturation steps | p95 ms |",
             "|---|---:|---|---:|---:|---:|---:|---:|---:|"]
    for scenario in args.scenarios:
        for cap in args.caps:
            for mode in ("fixed", "directional", "headroom"):
                selected = [r["metrics"] for r in rows if r["scenario"] == scenario and r["cap"] == cap and r["mode"] == mode]
                fields = ("tracking_rmse_mm", "position_violation_steps", "velocity_violation_steps", "velocity_abs_max_rad_s", "saturation_steps", "compute_p95_ms")
                vals = [np.mean([r[f] for r in selected]) for f in fields]
                lines.append(f"| {scenario} | {cap:.0%} | {mode} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    for scenario in args.scenarios:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        for mode in ("fixed", "directional", "headroom"):
            data = representative[(scenario, mode)]
            t = np.arange(len(data["err_mm"])) * DT
            axes[0].plot(t, data["err_mm"], label=mode)
            axes[1].plot(t, np.abs(data["qd"][1:]).max(1), label=mode)
            limits = np.asarray(TAU_MAX) * SCENARIOS[scenario]["torque_scale"]
            axes[2].plot(t, (np.abs(data["u_applied"]) / limits).max(1), label=mode)
        axes[0].set_ylabel("EE error [mm]")
        axes[1].set_ylabel("max |dq| [rad/s]")
        axes[1].axhline(DQ_MAX, color="black", linestyle="--")
        axes[2].set_ylabel("max torque / limit")
        axes[2].axhline(1, color="black", linestyle="--")
        axes[2].set_xlabel("Time [s]")
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend()
        fig.suptitle(f"{scenario}, seed {args.seeds[0]}, cap {args.caps[0]:.0%} full span")
        fig.tight_layout()
        fig.savefig(output / f"{scenario}.png", dpi=160)
        plt.close(fig)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
