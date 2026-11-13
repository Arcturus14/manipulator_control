"""Compare three DOB architectures by fixed and dynamic state margins."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT, Q_MAX, TAU_MAX, TIMING_WARMUP_STEPS
from experiment_runner import run_mppi
from kroc.benchmark_estimator_only_position_margin import (
    CASES, ramp_step, reference_and_initial,
)
from kroc.state_margin import NoiseAwareKrocTracker


ARCHITECTURES = {
    "estimate_only": "estimate_only",
    "post_dob": "output",
    "rollout_only": "rollout_model_only",
}


def measure(data: dict, joint: int) -> dict:
    q = np.asarray(data["q"])[1:, joint]
    upper = Q_MAX[joint]
    u = np.asarray(data["u_applied"])
    sat = np.isclose(np.abs(u), np.asarray(TAU_MAX), rtol=0, atol=1e-5)
    solve_ms = np.asarray(data["solve_t_raw"])[TIMING_WARMUP_STEPS:] * 1e3
    d_error = np.asarray(data["d_hat"])[1:] - np.asarray(data["d_true"])[1:]
    margin = np.asarray(data["kroc_margin"])[:, 0, 1, joint]
    return {
        "peak_q_rad": float(q.max()),
        "upper_distance_min_rad": float(upper - q.max()),
        "hard_violation_steps": int((q > upper).sum()),
        "hard_violation_max_rad": float(np.maximum(q - upper, 0).max()),
        "hard_violation_integral_rad_s": float(np.maximum(q - upper, 0).sum() * DT),
        "tracking_rmse_mm": float(np.sqrt(np.mean(np.asarray(data["err_mm"]) ** 2))),
        "saturation_steps": int(sat.any(axis=1).sum()),
        "test_joint_saturation_steps": int(sat[:, joint].sum()),
        "torque_rms_nm": float(np.sqrt(np.mean(u ** 2))),
        "torque_rate_rms_nm_s": float(np.sqrt(np.mean((np.diff(u, axis=0) / DT) ** 2))),
        "dob_rmse_nm": float(np.sqrt(np.mean(d_error ** 2))),
        "margin_max_fraction": float(margin.max() / (2 * Q_MAX[joint])),
        "compute_mean_ms": float(solve_ms.mean()),
        "compute_p95_ms": float(np.percentile(solve_ms, 95)),
    }


def stat(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()),
            "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "min": float(values.min()), "max": float(values.max())}


def plot(output: Path, rows: list[dict], representative: dict, joint: int) -> None:
    colors = {"estimate_only": "tab:gray", "post_dob": "tab:green", "rollout_only": "tab:purple"}
    labels = {"estimate_only": "Estimate only", "post_dob": "Post-DOB", "rollout_only": "Rollout only"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, case in zip(axes[0], CASES):
        for arch in ARCHITECTURES:
            data = representative[(arch, case)]
            t = np.arange(len(data["q"])) * DT
            ax.plot(t, data["q"][:, joint], color=colors[arch], label=labels[arch])
        ax.axhline(Q_MAX[joint], color="red", linestyle="--", label="physical upper limit")
        ax.set(title=("Static 5%" if case == "fixed_5pct" else "Dynamic 5–10%") + " (seed 17)",
               xlabel="time [s]", ylabel=f"q{joint + 1} [rad]")
        ax.grid(alpha=.25); ax.legend(fontsize=8)

    x = np.arange(len(ARCHITECTURES)); width = .34
    ax = axes[1, 0]
    for offset, case in zip((-.5, .5), CASES):
        means = [np.mean([r["metrics"]["hard_violation_steps"] for r in rows
                          if r["architecture"] == arch and r["case"] == case])
                 for arch in ARCHITECTURES]
        ax.bar(x + offset * width, means, width,
               label="Static 5%" if case == "fixed_5pct" else "Dynamic 5–10%")
    ax.set(xticks=x, xticklabels=[labels[a] for a in ARCHITECTURES],
           ylabel="position violation steps", title="Mean hard-limit violation (5 seeds)")
    ax.grid(alpha=.2, axis="y"); ax.legend()

    ax = axes[1, 1]
    for offset, case in zip((-.5, .5), CASES):
        means = [np.mean([r["metrics"]["tracking_rmse_mm"] for r in rows
                          if r["architecture"] == arch and r["case"] == case])
                 for arch in ARCHITECTURES]
        ax.bar(x + offset * width, means, width,
               label="Static 5%" if case == "fixed_5pct" else "Dynamic 5–10%")
    ax.set(xticks=x, xticklabels=[labels[a] for a in ARCHITECTURES],
           ylabel="tracking RMSE [mm]", title="Tracking error (5 seeds)")
    ax.grid(alpha=.2, axis="y"); ax.legend()
    fig.suptitle("Same position-limit disturbance: DOB architecture × margin")
    fig.tight_layout()
    fig.savefig(output / "architecture_margin_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 27, 37, 47, 57])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--initial-fraction", type=float, default=.92)
    parser.add_argument("--amplitude", type=float, default=-9.5)
    parser.add_argument("--ramp-steps", type=int, default=10)
    parser.add_argument("--onset-step", type=int, default=10)
    parser.add_argument("--joint", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cuda = cuda_preflight()
    root = Path(__file__).resolve().parent
    output = args.output or root / "results" / ("position_margin_architectures_" + datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f"))
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": datetime.now().astimezone().isoformat(),
                "arguments": {**vars(args), "output": str(args.output)}, "cuda": cuda,
                "architectures": ARCHITECTURES, "cases": {k: asdict(v) for k, v in CASES.items()},
                "disturbance": "single ramp then sustained constant", "sensor_noise": {"q_std_rad": .001, "dq_std_rad_s": .01},
                "dob": {"acceleration_lpf_hz": 10., "exact_zoh_q_hz": 8.}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    reference, q0 = reference_and_initial(args.steps, args.joint, args.initial_fraction)
    disturbance = ramp_step(args.amplitude, args.ramp_steps, args.onset_step, args.joint)
    rows, representative = [], {}
    combinations = [(a, c) for a in ARCHITECTURES for c in CASES]
    for seed_index, seed in enumerate(args.seeds):
        ordered = combinations[seed_index % len(combinations):] + combinations[:seed_index % len(combinations)]
        for architecture, case in ordered:
            holder = []
            def factory(*a, **kw):
                tracker = NoiseAwareKrocTracker(*a, **kw); holder.append(tracker); return tracker
            print(f"RUN seed={seed} architecture={architecture} case={case}", flush=True)
            data = run_mppi(reference, initial_q=q0, has_disturbance=True, use_dob=True,
                dob_compensation_mode=ARCHITECTURES[architecture], disturbance_function=disturbance,
                require_cuda=True, seed=seed, temperature=1., temporal_noise_rho=.8,
                joint_position_limit_weight=50000., joint_velocity_limit_weight=50000.,
                tracker_class=factory, tracker_options={"margin_config": asdict(CASES[case]),
                    "sensor_noise": {"position_std_rad": .001, "velocity_std_rad_s": .01, "seed": 70000 + seed},
                    "dob_acceleration_lpf_cutoff_hz": 10., "dob_cutoff_hz": 8.})
            data.update(holder[0].export_margin_data()); data.update(holder[0].export_sensor_data())
            item = {"seed": seed, "architecture": architecture, "case": case, "metrics": measure(data, args.joint)}
            rows.append(item); print(json.dumps(item), flush=True)
            np.savez_compressed(output / f"seed{seed}_{architecture}_{case}.npz", **{k: v for k, v in data.items() if not k.startswith("_")})
            if seed == args.seeds[0]:
                representative[(architecture, case)] = {"q": np.asarray(data["q"])}
            (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
            del data, holder; gc.collect(); torch.cuda.empty_cache()
    fields = list(rows[0]["metrics"])
    aggregate = {arch: {case: {field: stat([r["metrics"][field] for r in rows
                                            if r["architecture"] == arch and r["case"] == case])
                               for field in fields} for case in CASES} for arch in ARCHITECTURES}
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    plot(output, rows, representative, args.joint)
    lines = ["# Position margin architecture comparison", "",
             "| architecture | margin | violation steps | max violation [rad] | min upper distance [rad] | RMSE [mm] | saturation | solve mean [ms] |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for arch in ARCHITECTURES:
        for case in CASES:
            a = aggregate[arch][case]
            f = lambda key: f"{a[key]['mean']:.3f} ± {a[key]['sample_std']:.3f}"
            lines.append(f"| {arch} | {case} | {f('hard_violation_steps')} | {f('hard_violation_max_rad')} | {f('upper_distance_min_rad')} | {f('tracking_rmse_mm')} | {f('saturation_steps')} | {f('compute_mean_ms')} |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
