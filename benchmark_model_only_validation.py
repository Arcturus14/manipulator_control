"""Multi-seed validation and conditional Post-DOB comparison for model-only tuning."""
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

from benchmark_dob_aware_nominal_shift import cuda_preflight, disturbance_profiles, metrics
from config import NOISE_VAR
from experiment_runner import run_mppi, waypoints
from utils.result_cache import environment_spec, source_fingerprint


STEPS = 250
SEEDS = (7, 17, 27)
PROFILES = ("constant", "step", "slow_sinusoidal")
MODEL_CASES = {
    "stable_r08_lambda1": {
        "noise_variance": list(NOISE_VAR),
        "temporal_noise_rho": 0.8,
        "temperature": 1.0,
        "disturbance_covariance_gain": 0.0,
    },
    "adaptive_a1_r0": {
        "noise_variance": list(NOISE_VAR),
        "temporal_noise_rho": 0.0,
        "temperature": 0.24,
        "disturbance_covariance_gain": 1.0,
    },
}


def extra_metrics(data: dict) -> dict:
    result = metrics(data)
    result.update({
        "weight_ess_mean": float(np.mean(data["mppi_weight_ess"])),
        "sample_clamp_fraction_mean": float(
            np.mean(data["mppi_sample_clamp_fraction"])
        ),
        "deadline_misses": int(np.sum(
            np.asarray(data["solve_t"])[np.isfinite(data["solve_t"])] > 0.02
        )),
    })
    return result


def summarize(rows: list[dict]) -> dict:
    output = {}
    for case in sorted({row["case"] for row in rows}):
        output[case] = {}
        for profile in PROFILES:
            selected = [row["metrics"] for row in rows
                        if row["case"] == case and row["profile"] == profile]
            if len(selected) != len(SEEDS):
                continue
            output[case][profile] = {
                field: {
                    "mean": float(np.mean([item[field] for item in selected])),
                    "std": float(np.std([item[field] for item in selected], ddof=1)),
                }
                for field in (
                    "tracking_rmse_mm", "tracking_max_mm",
                    "torque_rate_rms_nm_per_step", "saturated_steps",
                    "solve_mean_ms", "weight_ess_mean",
                    "sample_clamp_fraction_mean", "deadline_misses",
                )
            }
    return output


def save_plot(output: Path, report: dict) -> None:
    summary = report["summary"]
    cases = list(summary)
    x = np.arange(len(PROFILES))
    width = 0.8 / max(len(cases), 1)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for index, case in enumerate(cases):
        means = [summary[case][profile]["tracking_rmse_mm"]["mean"]
                 for profile in PROFILES]
        stds = [summary[case][profile]["tracking_rmse_mm"]["std"]
                for profile in PROFILES]
        axis.bar(x + (index-(len(cases)-1)/2)*width, means, width,
                 yerr=stds, capsize=3, label=case)
    axis.set_xticks(x, PROFILES)
    axis.set_ylabel("Tracking RMSE [mm]")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "validation_rmse.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-post", action="store_true")
    args = parser.parse_args()
    preflight = cuda_preflight()
    cases = dict(MODEL_CASES)
    if args.include_post:
        cases = {
            "stable_r08_lambda1": MODEL_CASES["stable_r08_lambda1"],
            "post_dob": {},
        }
    spec = {
        "include_post": args.include_post, "steps": STEPS,
        "seeds": SEEDS, "profiles": PROFILES, "cases": cases,
        "environment": environment_spec(),
        "source": source_fingerprint(Path.cwd(), "run", {}),
        "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output = Path("results/model_only_validation_gpu") / fingerprint[:12]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "metrics.json"
    rows = json.loads(path.read_text()).get("rows", []) if path.exists() else []
    completed = {(row["case"], row["profile"], row["seed"]) for row in rows}
    profiles = disturbance_profiles(STEPS, large_step_scale=8.0)
    reference = waypoints("circle", n_steps=STEPS)
    print("[CUDA PREFLIGHT]", json.dumps(preflight), flush=True)
    for case, settings in cases.items():
        for profile in PROFILES:
            for seed in SEEDS:
                if (case, profile, seed) in completed:
                    continue
                cuda_preflight()
                print("[RUN]", case, profile, seed, flush=True)
                post = case == "post_dob"
                data = run_mppi(
                    reference, has_disturbance=True, use_dob=True,
                    dob_compensation_mode="output" if post else "rollout_model_only",
                    disturbance_function=profiles[profile], require_cuda=True,
                    seed=seed, **settings,
                )
                if str(data["compute_device"]) != "cuda":
                    raise RuntimeError("CPU result rejected")
                row = {"case": case, "profile": profile, "seed": seed,
                       "settings": settings, "metrics": extra_metrics(data)}
                rows.append(row)
                report = {"fingerprint": fingerprint, "spec": spec,
                          "cuda_preflight": preflight, "rows": rows,
                          "summary": summarize(rows)}
                path.write_text(json.dumps(report, indent=2))
                print(json.dumps(row), flush=True)
                del data
                gc.collect()
                torch.cuda.empty_cache()
    report["summary"] = summarize(rows)
    path.write_text(json.dumps(report, indent=2))
    save_plot(output, report)
    print("[SUMMARY]", json.dumps(report["summary"]), flush=True)
    print("[SAVED]", output, flush=True)


if __name__ == "__main__":
    main()
