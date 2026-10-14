"""GPU-only paired comparison of MPPI horizon-noise correlation settings."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_dob_aware_nominal_shift import (
    cuda_preflight,
    disturbance_profiles,
    metrics,
)
from config import DT
from experiment_runner import run_mppi, waypoints
from utils.result_cache import environment_spec, source_fingerprint


MODES = {
    "post_dob": "output",
    "model_only": "rollout_model_only",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7])
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.0, 0.8])
    parser.add_argument(
        "--modes", nargs="+", choices=tuple(MODES), default=list(MODES)
    )
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument(
        "--profile",
        choices=("constant", "step", "slow_sinusoidal", "fast_sinusoidal", "large_step"),
        default="slow_sinusoidal",
    )
    parser.add_argument("--trajectory", default="circle")
    parser.add_argument(
        "--output-root", type=Path, default=Path("results/temporal_noise_gpu")
    )
    args = parser.parse_args()
    if args.steps < 3:
        parser.error("--steps must be at least 3")
    if any(not 0.0 <= rho <= 1.0 for rho in args.rhos):
        parser.error("every --rhos value must be in [0, 1]")
    return args


def _case_name(mode: str, rho: float) -> str:
    return f"{mode}__rho_{rho:g}".replace(".", "p")


def main() -> None:
    args = _arguments()
    gpu = cuda_preflight()
    print("GPU_PREFLIGHT", json.dumps(gpu, sort_keys=True), flush=True)

    cases = [
        {"name": _case_name(mode, rho), "mode": mode, "rho": rho}
        for mode in args.modes for rho in args.rhos
    ]
    spec = {
        "cases": cases,
        "seeds": args.seeds,
        "steps": args.steps,
        "profile": args.profile,
        "trajectory": args.trajectory,
        "source": source_fingerprint(Path.cwd(), "run", {}),
        "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "environment": environment_spec(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    output = args.output_root / fingerprint[:12]
    required = ("metrics.json", "results.npz", "tracking.png", "summary.png")
    if all((output / name).is_file() for name in required):
        print("REUSE", output, flush=True)
        return

    reference = waypoints(args.trajectory, n_steps=args.steps)
    disturbance = disturbance_profiles(args.steps)[args.profile]
    rows: list[dict] = []
    arrays: dict[str, np.ndarray] = {}
    representative: dict[str, dict] = {}
    for seed_index, seed in enumerate(args.seeds):
        ordered_cases = cases if seed_index % 2 == 0 else list(reversed(cases))
        for case in ordered_cases:
            cuda_preflight()
            print(
                "RUN",
                json.dumps({"seed": seed, **case}, sort_keys=True),
                flush=True,
            )
            data = run_mppi(
                reference,
                has_disturbance=True,
                use_dob=True,
                require_cuda=True,
                dob_compensation_mode=MODES[case["mode"]],
                temporal_noise_rho=case["rho"],
                seed=seed,
                disturbance_function=disturbance,
            )
            if str(data["compute_device"]) != "cuda":
                raise RuntimeError("CPU result rejected")
            if float(data["mppi_temporal_noise_rho"]) != case["rho"]:
                raise RuntimeError("temporal_noise_rho was not propagated")

            row = {
                "seed": seed,
                **case,
                "metrics": metrics(data),
                "compute_device": str(data["compute_device"]),
                "cuda_device_name": str(data["cuda_device_name"]),
            }
            rows.append(row)
            print("RESULT", json.dumps(row, sort_keys=True), flush=True)
            prefix = f"{case['name']}__seed_{seed}"
            arrays.update({
                f"{prefix}__{key}": value
                for key, value in data.items() if not key.startswith("_")
            })
            if seed == args.seeds[0]:
                representative[case["name"]] = data
            del data
            gc.collect()
            torch.cuda.empty_cache()

    summary = {}
    for case in cases:
        selected = [r["metrics"] for r in rows if r["name"] == case["name"]]
        summary[case["name"]] = {
            key: {
                "mean": float(np.mean([item[key] for item in selected])),
                "std": float(np.std([item[key] for item in selected], ddof=1))
                if len(selected) > 1 else 0.0,
            }
            for key in selected[0]
            if all(item[key] is not None for item in selected)
        }

    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "results.npz", **arrays)
    (output / "metrics.json").write_text(json.dumps({
        "fingerprint": fingerprint,
        "spec": spec,
        "rows": rows,
        "summary": summary,
    }, indent=2), encoding="utf-8")

    fig, axis = plt.subplots(figsize=(11, 5))
    time_s = np.arange(args.steps) * DT
    for case in cases:
        data = representative[case["name"]]
        axis.plot(time_s, data["err_mm"], label=case["name"])
    axis.set(xlabel="Time [s]", ylabel="EE tracking error [mm]",
             title=f"{args.profile}, seed {args.seeds[0]}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "tracking.png", dpi=180)
    plt.close(fig)

    names = [case["name"] for case in cases]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric_name, ylabel in (
        (axes[0], "tracking_rmse_mm", "Tracking RMSE [mm]"),
        (axes[1], "torque_rate_rms_nm_per_step", "Torque-rate RMS [Nm/step]"),
    ):
        means = [summary[name][metric_name]["mean"] for name in names]
        stds = [summary[name][metric_name]["std"] for name in names]
        axis.bar(np.arange(len(names)), means, yerr=stds, capsize=3)
        axis.set_xticks(np.arange(len(names)), names, rotation=30, ha="right")
        axis.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output / "summary.png", dpi=180)
    plt.close(fig)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
