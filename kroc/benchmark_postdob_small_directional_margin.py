"""Post-DOB: fixed 1%, directional 1--5%, and fixed 5% near a torque limit."""
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
    "fixed_1pct": MarginConfig(
        mode="fixed", min_fraction=.01, fixed_fraction=.01,
        max_fraction=.05, penalty_kind="hinge",
    ),
    "dynamic_directional_1_to_5pct": MarginConfig(
        mode="directional", min_fraction=.01, fixed_fraction=.01,
        max_fraction=.05, reaction_time_s=.10, penalty_kind="hinge",
        acceleration_deadband_rad_s2=ACCELERATION_DEADBANDS["nominal_p99"],
        beta_up=1.0, beta_down=.10,
    ),
    "fixed_5pct": MarginConfig(
        mode="fixed", min_fraction=.01, fixed_fraction=.05,
        max_fraction=.05, penalty_kind="hinge",
    ),
}


def reference_and_initial(steps: int, joint: int, fraction: float):
    q0 = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0[joint] = fraction * Q_MAX[joint]
    target = fk_joints_np(q0)[-1]
    return np.repeat(target[None, :], steps, axis=0), q0


def ramp_step(amplitude: float, ramp_steps: int, onset: int, joint: int):
    def disturbance(step, device, dtype):
        elapsed = step - onset
        value = 0.0 if elapsed < 0 else amplitude * min((elapsed + 1) / ramp_steps, 1.0)
        result = torch.zeros(6, device=device, dtype=dtype)
        result[joint] = value
        return result
    return disturbance


def measure(data: dict, joint: int, torque_scale: float) -> dict:
    q = np.asarray(data["q"])[1:, joint]
    upper = Q_MAX[joint]
    limit = np.asarray(TAU_MAX) * torque_scale
    applied = np.asarray(data["u_applied"])
    requested = np.asarray(data["u_mppi"]) + np.asarray(data["d_hat_used"])
    sat = np.isclose(np.abs(applied), limit, rtol=0, atol=1e-5)
    requested_excess = np.maximum(np.abs(requested) - limit, 0)
    margin = np.asarray(data["kroc_margin"])[:, 0, 1, joint]
    solve_ms = np.asarray(data["solve_t_raw"])[TIMING_WARMUP_STEPS:] * 1e3
    return {
        "peak_q_rad": float(q.max()),
        "final_q_rad": float(q[-1]),
        "upper_distance_min_rad": float(upper - q.max()),
        "hard_violation_steps": int((q > upper).sum()),
        "hard_violation_max_rad": float(np.maximum(q - upper, 0).max()),
        "hard_violation_integral_rad_s": float(np.maximum(q - upper, 0).sum() * DT),
        "tracking_rmse_mm": float(np.sqrt(np.mean(np.asarray(data["err_mm"]) ** 2))),
        "saturation_steps": int(sat.any(axis=1).sum()),
        "test_joint_saturation_steps": int(sat[:, joint].sum()),
        "requested_excess_max_nm": float(requested_excess[:, joint].max()),
        "requested_excess_integral_nm_s": float(requested_excess[:, joint].sum() * DT),
        "margin_max_fraction": float(margin.max() / (Q_MAX[joint] - Q_MIN[joint])),
        "margin_mean_fraction": float(margin.mean() / (Q_MAX[joint] - Q_MIN[joint])),
        "torque_rate_rms_nm_s": float(np.sqrt(np.mean((np.diff(applied, axis=0) / DT) ** 2))),
        "compute_mean_ms": float(solve_ms.mean()),
        "compute_p95_ms": float(np.percentile(solve_ms, 95)),
    }


def stat(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max())}


def plot_validation(output: Path, representative: dict, joint: int,
                    torque_scale: float, onset: int, ramp_steps: int) -> None:
    colors = {"fixed_1pct": "tab:orange",
              "dynamic_directional_1_to_5pct": "tab:blue",
              "fixed_5pct": "tab:green"}
    labels = {"fixed_1pct": "Static 1%", "dynamic_directional_1_to_5pct": "Directional 1–5%", "fixed_5pct": "Static 5%"}
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    for case, data in representative.items():
        t_state = np.arange(len(data["q"])) * DT
        t = np.arange(len(data["u_applied"])) * DT
        axes[0].plot(t_state, data["q"][:, joint], color=colors[case], label=labels[case])
        axes[1].plot(t, 100 * data["kroc_margin"][:, 0, 1, joint] /
                     (Q_MAX[joint] - Q_MIN[joint]), color=colors[case], label=labels[case])
        axes[2].plot(t, data["u_mppi"][:, joint] + data["d_hat_used"][:, joint],
                     color=colors[case], label=labels[case])
        axes[3].plot(t, data["err_mm"], color=colors[case], label=labels[case])
    limit = TAU_MAX[joint] * torque_scale
    axes[0].axhline(Q_MAX[joint], color="red", linestyle="--", label="physical upper limit")
    axes[1].axhline(1, color="gray", linestyle=":"); axes[1].axhline(5, color="red", linestyle="--")
    axes[2].axhline(limit, color="red", linestyle="--"); axes[2].axhline(-limit, color="red", linestyle="--")
    axes[0].set_ylabel(f"q{joint + 1} [rad]")
    axes[1].set_ylabel("upper margin [% span]")
    axes[2].set_ylabel("requested torque [Nm]")
    axes[3].set_ylabel("EE error [mm]"); axes[3].set_xlabel("time [s]")
    for ax in axes:
        ax.axvline(onset * DT, color="k", linestyle="--", linewidth=1)
        ax.axvline((onset + ramp_steps) * DT, color="k", linestyle=":", linewidth=1)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.suptitle("Post-DOB directional margin before headroom (representative seed)")
    fig.tight_layout()
    fig.savefig(output / "directional_margin_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--initial-fractions", nargs="+", type=float, default=[.86, .88, .90])
    parser.add_argument("--amplitudes", nargs="+", type=float, default=[-8., -10., -12.])
    parser.add_argument("--torque-scales", nargs="+", type=float, default=[.35, .40])
    parser.add_argument("--ramp-steps", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--onset-step", type=int, default=10)
    parser.add_argument("--joint", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cuda = cuda_preflight()
    root = Path(__file__).resolve().parent
    output = args.output or root / "results" / ("postdob_small_directional_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": datetime.now().astimezone().isoformat(), "arguments": {**vars(args), "output": str(args.output)},
                "cuda": cuda, "controller": "Post-DOB output compensation", "headroom_enabled": False,
                "cases": {k: asdict(v) for k, v in CASES.items()},
                "sensor_noise": {"q_std_rad": .001, "dq_std_rad_s": .01},
                "dob": {"acceleration_lpf_hz": 10., "exact_zoh_q_hz": 8.}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    rows, representative = [], {}
    single_combo = len(args.initial_fractions) * len(args.amplitudes) * len(args.torque_scales) * len(args.ramp_steps) == 1
    for fraction in args.initial_fractions:
        reference, q0 = reference_and_initial(args.steps, args.joint, fraction)
        for amplitude in args.amplitudes:
            for torque_scale in args.torque_scales:
                for ramp_steps in args.ramp_steps:
                    disturbance = ramp_step(amplitude, ramp_steps, args.onset_step, args.joint)
                    combos = [(seed, case) for seed in args.seeds for case in CASES]
                    for index, (seed, case) in enumerate(combos):
                        holder = []
                        def factory(*a, **kw):
                            tracker = NoiseAwareKrocTracker(*a, **kw); holder.append(tracker); return tracker
                        print(f"RUN q={fraction} amp={amplitude} torque_scale={torque_scale} ramp={ramp_steps} seed={seed} case={case}", flush=True)
                        data = run_mppi(reference, initial_q=q0, has_disturbance=True, use_dob=True,
                            dob_compensation_mode="output", disturbance_function=disturbance,
                            require_cuda=True, seed=seed, temperature=1., temporal_noise_rho=.8,
                            joint_position_limit_weight=50000., joint_velocity_limit_weight=50000.,
                            tracker_class=factory, tracker_options={"margin_config": asdict(CASES[case]),
                                "torque_scale": torque_scale,
                                "sensor_noise": {"position_std_rad": .001, "velocity_std_rad_s": .01, "seed": 80000 + seed},
                                "dob_acceleration_lpf_cutoff_hz": 10., "dob_cutoff_hz": 8.})
                        data.update(holder[0].export_margin_data()); data.update(holder[0].export_sensor_data())
                        row = {"initial_fraction": fraction, "amplitude_nm": amplitude, "torque_scale": torque_scale,
                               "ramp_steps": ramp_steps, "seed": seed, "case": case,
                               "metrics": measure(data, args.joint, torque_scale)}
                        rows.append(row); print(json.dumps(row), flush=True)
                        if single_combo:
                            np.savez_compressed(output / f"seed{seed}_{case}.npz", **{k: v for k, v in data.items() if not k.startswith("_")})
                            if seed == args.seeds[0]:
                                representative[case] = {k: np.asarray(data[k]) for k in ("q", "kroc_margin", "u_mppi", "d_hat_used", "u_applied", "err_mm")}
                        (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
                        del data, holder; gc.collect(); torch.cuda.empty_cache()
    groups = {}
    for row in rows:
        key = (row["initial_fraction"], row["amplitude_nm"], row["torque_scale"], row["ramp_steps"], row["seed"])
        groups.setdefault(key, {})[row["case"]] = row["metrics"]
    candidates = []
    for key, pair in groups.items():
        small, dynamic, conservative = pair["fixed_1pct"], pair["dynamic_directional_1_to_5pct"], pair["fixed_5pct"]
        candidates.append({"initial_fraction": key[0], "amplitude_nm": key[1], "torque_scale": key[2], "ramp_steps": key[3], "seed": key[4],
            "fixed1_violation_steps": small["hard_violation_steps"], "dynamic_violation_steps": dynamic["hard_violation_steps"],
            "fixed5_violation_steps": conservative["hard_violation_steps"], "fixed1_max_violation": small["hard_violation_max_rad"],
            "dynamic_max_violation": dynamic["hard_violation_max_rad"], "fixed1_saturation": small["test_joint_saturation_steps"],
            "dynamic_saturation": dynamic["test_joint_saturation_steps"], "fixed5_saturation": conservative["test_joint_saturation_steps"],
            "dynamic_margin_max": dynamic["margin_max_fraction"], "fixed5_minus_dynamic_rmse": conservative["tracking_rmse_mm"] - dynamic["tracking_rmse_mm"]})
    eligible = [c for c in candidates if c["fixed1_violation_steps"] > 0 and c["dynamic_violation_steps"] == 0
                and c["fixed5_violation_steps"] == 0 and c["fixed1_saturation"] > 0]
    eligible.sort(key=lambda c: (c["fixed1_violation_steps"], c["fixed5_minus_dynamic_rmse"]), reverse=True)
    (output / "selection.json").write_text(json.dumps({"eligible": eligible, "all_candidates": candidates}, indent=2) + "\n")
    if single_combo:
        aggregate = {case: {field: stat([r["metrics"][field] for r in rows if r["case"] == case])
                            for field in rows[0]["metrics"]} for case in CASES}
        (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
        plot_validation(output, representative, args.joint, args.torque_scales[0], args.onset_step, args.ramp_steps[0])
    print("ELIGIBLE", json.dumps(eligible[:10]), flush=True)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
