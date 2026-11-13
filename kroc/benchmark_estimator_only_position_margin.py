"""Search/validate fixed-5% vs dynamic-5--10% without DOB compensation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT, Q_MAX, Q_MIN, TAU_MAX, TIMING_WARMUP_STEPS
from experiment_runner import run_mppi
from kinematics.ur5_kinematics import fk_joints_np
from kroc.benchmark_noise_aware_state_margin import ACCELERATION_DEADBANDS
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker


CASES = {
    "fixed_5pct": MarginConfig(
        mode="fixed", min_fraction=.05, fixed_fraction=.05,
        max_fraction=.10, penalty_kind="hinge",
    ),
    "dynamic_5_to_10pct": MarginConfig(
        mode="directional", min_fraction=.05, fixed_fraction=.05,
        max_fraction=.10, reaction_time_s=.10, penalty_kind="hinge",
        acceleration_deadband_rad_s2=ACCELERATION_DEADBANDS["nominal_p99"],
        beta_up=1.0, beta_down=.10,
    ),
}


def reference_and_initial(steps: int, joint: int, fraction: float):
    q0 = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0[joint] = fraction * Q_MAX[joint]
    target = fk_joints_np(q0)[-1]
    return np.repeat(target[None, :], steps, axis=0), q0


def ramp_step(amplitude: float, ramp_steps: int, onset: int, joint: int):
    """Ramp once, then keep the final disturbance until episode end."""
    def disturbance(step, device, dtype):
        elapsed = step - onset
        value = 0.0 if elapsed < 0 else amplitude * min((elapsed + 1) / ramp_steps, 1.0)
        result = torch.zeros(6, device=device, dtype=dtype)
        result[joint] = value
        return result
    return disturbance


def metrics(data: dict, joint: int) -> dict:
    q = np.asarray(data["q"])[1:, joint]
    dq = np.asarray(data["qd"])[1:, joint]
    upper = Q_MAX[joint]
    margin = np.asarray(data["kroc_margin"])[:, 0, 1, joint]
    u = np.asarray(data["u_applied"])
    sat = np.isclose(np.abs(u), np.asarray(TAU_MAX), rtol=0, atol=1e-5)
    solve_ms = np.asarray(data["solve_t_raw"])[TIMING_WARMUP_STEPS:] * 1e3
    return {
        "peak_q_rad": float(q.max()),
        "upper_distance_min_rad": float(upper - q.max()),
        "hard_violation_steps": int((q > upper).sum()),
        "hard_violation_max_rad": float(np.maximum(q - upper, 0).max()),
        "hard_violation_integral_rad_s": float(np.maximum(q - upper, 0).sum() * DT),
        "peak_dq_rad_s": float(dq.max()),
        "margin_max_rad": float(margin.max()),
        "margin_max_fraction": float(margin.max() / (Q_MAX[joint] - Q_MIN[joint])),
        "saturation_steps": int(sat.any(axis=1).sum()),
        "test_joint_saturation_steps": int(sat[:, joint].sum()),
        "tracking_rmse_mm": float(np.sqrt(np.mean(np.asarray(data["err_mm"]) ** 2))),
        "compute_mean_ms": float(solve_ms.mean()),
        "compute_p95_ms": float(np.percentile(solve_ms, 95)),
    }


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max())}


def plot_pair(output: Path, representative: dict, joint: int, onset: int,
              ramp_steps: int) -> None:
    colors = {"fixed_5pct": "tab:orange", "dynamic_5_to_10pct": "tab:blue"}
    labels = {"fixed_5pct": "Static 5%", "dynamic_5_to_10pct": "Dynamic 5–10%"}
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for case, data in representative.items():
        t_state = np.arange(len(data["q"])) * DT
        t = np.arange(len(data["d_hat_used"])) * DT
        axes[0].plot(t_state, data["q"][:, joint], color=colors[case], label=labels[case])
        axes[1].plot(t, data["d_hat_used"][:, joint], color=colors[case], label=labels[case])
        axes[2].plot(t, 100 * data["kroc_margin"][:, 0, 1, joint] /
                     (Q_MAX[joint] - Q_MIN[joint]), color=colors[case], label=labels[case])
        axes[3].plot(t, data["u_applied"][:, joint], color=colors[case], label=labels[case])
    first = next(iter(representative.values()))
    t = np.arange(len(first["d_true"]) - 1) * DT
    axes[1].plot(t, first["d_true"][1:, joint], "k--", label="true disturbance")
    axes[0].axhline(Q_MAX[joint], color="red", linestyle="--", label="physical upper limit")
    axes[0].axhline(Q_MAX[joint] - .05 * (Q_MAX[joint] - Q_MIN[joint]),
                    color="gray", linestyle=":", label="static 5% boundary")
    axes[2].axhline(5, color="gray", linestyle=":")
    axes[2].axhline(10, color="red", linestyle="--")
    axes[3].axhline(TAU_MAX[joint], color="red", linestyle="--")
    axes[3].axhline(-TAU_MAX[joint], color="red", linestyle="--")
    axes[0].set_ylabel(f"q{joint + 1} [rad]")
    axes[1].set_ylabel(r"$d,\hat d$ [Nm]")
    axes[2].set_ylabel("upper margin [% span]")
    axes[3].set_ylabel("applied torque [Nm]"); axes[3].set_xlabel("time [s]")
    for ax in axes:
        ax.axvline(onset * DT, color="k", linestyle="--", linewidth=1)
        ax.axvline((onset + ramp_steps) * DT, color="k", linestyle=":", linewidth=1)
        ax.grid(alpha=.25); ax.legend(fontsize=8, loc="best")
    fig.suptitle("Estimator-only DOB position-limit comparison (representative seed)")
    fig.tight_layout()
    fig.savefig(output / "dhat_margin_timeseries.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--initial-fractions", nargs="+", type=float, default=[.88, .90, .92])
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[-8., -12., -16., -20.])
    parser.add_argument("--ramp-steps", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--onset-step", type=int, default=10)
    parser.add_argument("--joint", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cuda = cuda_preflight()
    root = Path(__file__).resolve().parent
    output = args.output or root / "results" / ("estimator_only_position_margin_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": datetime.now().astimezone().isoformat(), "arguments": {**vars(args), "output": str(args.output)},
                "cuda": cuda, "controller": "estimator-only DOB", "output_compensation": False,
                "rollout_compensation": False, "disturbance_shape": "one ramp then constant to end",
                "cases": {k: asdict(v) for k, v in CASES.items()},
                "sensor_noise": {"q_std_rad": .001, "dq_std_rad_s": .01},
                "dob": {"acceleration_lpf_hz": 10., "exact_zoh_q_hz": 8.}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, representative = [], {}
    single_combo = len(args.initial_fractions) * len(args.amplitudes) * len(args.ramp_steps) == 1
    for fraction in args.initial_fractions:
        reference, q0 = reference_and_initial(args.steps, args.joint, fraction)
        for amplitude in args.amplitudes:
            for ramp_steps in args.ramp_steps:
                disturbance = ramp_step(amplitude, ramp_steps, args.onset_step, args.joint)
                for seed_index, seed in enumerate(args.seeds):
                    names = list(CASES); names = names[seed_index % 2:] + names[:seed_index % 2]
                    for case in names:
                        holder = []
                        def factory(*a, **kw):
                            tracker = NoiseAwareKrocTracker(*a, **kw); holder.append(tracker); return tracker
                        print(f"RUN fraction={fraction} amp={amplitude} ramp={ramp_steps} seed={seed} case={case}", flush=True)
                        data = run_mppi(reference, initial_q=q0, has_disturbance=True, use_dob=True,
                            dob_compensation_mode="estimate_only", disturbance_function=disturbance,
                            require_cuda=True, seed=seed, temperature=1., temporal_noise_rho=.8,
                            joint_position_limit_weight=50000., joint_velocity_limit_weight=50000.,
                            tracker_class=factory, tracker_options={"margin_config": asdict(CASES[case]),
                                "sensor_noise": {"position_std_rad": .001, "velocity_std_rad_s": .01, "seed": 70000 + seed},
                                "dob_acceleration_lpf_cutoff_hz": 10., "dob_cutoff_hz": 8.})
                        data.update(holder[0].export_margin_data()); data.update(holder[0].export_sensor_data())
                        row = {"initial_fraction": fraction, "amplitude_nm": amplitude, "ramp_steps": ramp_steps,
                               "seed": seed, "case": case, "metrics": metrics(data, args.joint)}
                        rows.append(row); print(json.dumps(row), flush=True)
                        if single_combo:
                            np.savez_compressed(output / f"seed{seed}_{case}.npz", **{k: v for k, v in data.items() if not k.startswith("_")})
                            if seed == args.seeds[0]:
                                representative[case] = {k: np.asarray(data[k]) for k in ("q", "d_hat_used", "d_true", "kroc_margin", "u_applied")}
                        (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
                        del data, holder; gc.collect(); torch.cuda.empty_cache()
    groups = {}
    for row in rows:
        key = (row["initial_fraction"], row["amplitude_nm"], row["ramp_steps"], row["seed"])
        groups.setdefault(key, {})[row["case"]] = row["metrics"]
    candidates = []
    for key, pair in groups.items():
        fixed, dynamic = pair["fixed_5pct"], pair["dynamic_5_to_10pct"]
        candidates.append({"initial_fraction": key[0], "amplitude_nm": key[1], "ramp_steps": key[2], "seed": key[3],
                           "fixed_violation_steps": fixed["hard_violation_steps"],
                           "dynamic_violation_steps": dynamic["hard_violation_steps"],
                           "fixed_violation_max_rad": fixed["hard_violation_max_rad"],
                           "dynamic_violation_max_rad": dynamic["hard_violation_max_rad"],
                           "peak_reduction_rad": fixed["peak_q_rad"] - dynamic["peak_q_rad"]})
    eligible = [c for c in candidates if c["fixed_violation_steps"] > 0 and c["dynamic_violation_steps"] == 0]
    eligible.sort(key=lambda c: (c["fixed_violation_steps"], c["peak_reduction_rad"]), reverse=True)
    aggregate = {}
    if single_combo:
        for case in CASES:
            selected = [r["metrics"] for r in rows if r["case"] == case]
            aggregate[case] = {field: stats([m[field] for m in selected]) for field in selected[0]}
        (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
        plot_pair(output, representative, args.joint, args.onset_step, args.ramp_steps[0])
    (output / "selection.json").write_text(json.dumps({"eligible": eligible, "all_candidates": candidates}, indent=2) + "\n")
    print("ELIGIBLE", json.dumps(eligible[:10]), flush=True)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
