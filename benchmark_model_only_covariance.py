"""CUDA-only staged covariance search for model-only disturbance rollout.

The GUI/default controller is untouched.  This benchmark first screens only
model-only candidates; Post-DOB is run later only if a candidate approaches
the established Post-DOB tracking level.
"""
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

from benchmark_dob_aware_nominal_shift import cuda_preflight, disturbance_profiles
from config import D_AMP, DT, NOISE_VAR
from experiment_runner import run_mppi, waypoints
from utils.result_cache import environment_spec, source_fingerprint


STEPS = 250
SCREEN_SEED = 7
SCREEN_PROFILES = ("constant", "step", "slow_sinusoidal")
BASE = np.asarray(NOISE_VAR, dtype=float)
DISTURBANCE_SCALE = np.abs(np.asarray(D_AMP, dtype=float))


def _fixed_candidates() -> dict[str, dict]:
    candidates: dict[str, dict] = {}

    def add(name: str, variance: np.ndarray, rho: float) -> None:
        candidates[name] = {
            "noise_variance": variance.tolist(),
            "temporal_noise_rho": rho,
            "disturbance_covariance_gain": 0.0,
        }

    add("base_r0", BASE, 0.0)
    add("base_r08", BASE, 0.8)
    add("global_s1p5_r08", BASE * 1.5**2, 0.8)
    add("global_s2_r0", BASE * 2.0**2, 0.0)
    add("global_s2_r08", BASE * 2.0**2, 0.8)
    add("global_s3_r0", BASE * 3.0**2, 0.0)
    add("global_s3_r08", BASE * 3.0**2, 0.8)
    add("global_s4_r08", BASE * 4.0**2, 0.8)
    for c in (0.5, 0.75, 1.0):
        variance = np.maximum(BASE, (c * DISTURBANCE_SCALE) ** 2)
        add(f"matched_c{str(c).replace('.', 'p')}_r08", variance, 0.8)
    return candidates


def _adaptive_candidates() -> dict[str, dict]:
    candidates = {}
    for gain in (0.25, 0.5625, 1.0, 1.5):
        for rho in (0.0, 0.8):
            candidates[f"adaptive_a{str(gain).replace('.', 'p')}_r{int(rho*10)}"] = {
                "noise_variance": BASE.tolist(),
                "temporal_noise_rho": rho,
                "disturbance_covariance_gain": gain,
            }
    return candidates


def _temperature_candidates() -> dict[str, dict]:
    candidates = {}
    for temperature in (1.0, 4.0, 16.0, 64.0):
        candidates[f"base_r08_lambda_{str(temperature).replace('.', 'p')}"] = {
            "noise_variance": BASE.tolist(),
            "temporal_noise_rho": 0.8,
            "disturbance_covariance_gain": 0.0,
            "temperature": temperature,
        }
        candidates[f"global_s1p5_r08_lambda_{str(temperature).replace('.', 'p')}"] = {
            "noise_variance": (BASE * 1.5**2).tolist(),
            "temporal_noise_rho": 0.8,
            "disturbance_covariance_gain": 0.0,
            "temperature": temperature,
        }
    return candidates


def _metrics(data: dict) -> dict[str, float]:
    error = np.asarray(data["err_mm"], dtype=float)
    learned = np.asarray(data["u_mppi"] - data["u_ff"], dtype=float)
    estimate = np.asarray(data["d_hat_used"], dtype=float)
    solve_ms = np.asarray(data["solve_t"], dtype=float) * 1e3
    solve_ms = solve_ms[np.isfinite(solve_ms)]
    torque = np.asarray(data["u_applied"], dtype=float)
    return {
        "tracking_rmse_mm": float(np.sqrt(np.mean(error**2))),
        "tracking_mean_mm": float(np.mean(error)),
        "tracking_max_mm": float(np.max(error)),
        "learned_compensation_error_rms_nm": float(
            np.sqrt(np.mean((learned - estimate) ** 2))
        ),
        "torque_rate_rms_nm_per_step": float(
            np.sqrt(np.mean(np.linalg.norm(np.diff(torque, axis=0), axis=1) ** 2))
        ),
        "weight_ess_mean": float(np.mean(data["mppi_weight_ess"])),
        "weight_ess_p05": float(np.percentile(data["mppi_weight_ess"], 5)),
        "sample_clamp_fraction_mean": float(
            np.mean(data["mppi_sample_clamp_fraction"])
        ),
        "saturated_steps": int(np.sum(data["actuator_saturated"])),
        "solve_mean_ms": float(np.mean(solve_ms)),
        "deadline_misses": int(np.sum(solve_ms > DT * 1e3)),
    }


def _save(output: Path, report: dict, histories: dict[str, np.ndarray]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / "screen_histories.npz", **histories)


def _summarize(rows: list[dict], profiles: tuple[str, ...]) -> dict:
    summary = {}
    for candidate in sorted({row["candidate"] for row in rows}):
        selected = [row for row in rows if row["candidate"] == candidate]
        if len(selected) != len(profiles):
            continue
        rmse = np.asarray([row["metrics"]["tracking_rmse_mm"] for row in selected])
        summary[candidate] = {
            "mean_profile_rmse_mm": float(rmse.mean()),
            "worst_profile_rmse_mm": float(rmse.max()),
            "profile_rmse_mm": {
                row["profile"]: row["metrics"]["tracking_rmse_mm"]
                for row in selected
            },
            "mean_ess": float(np.mean([
                row["metrics"]["weight_ess_mean"] for row in selected
            ])),
            "mean_sample_clamp_fraction": float(np.mean([
                row["metrics"]["sample_clamp_fraction_mean"] for row in selected
            ])),
            "total_deadline_misses": int(sum(
                row["metrics"]["deadline_misses"] for row in selected
            )),
        }
    return summary


def _plot(output: Path, summary: dict, profiles: tuple[str, ...]) -> None:
    order = sorted(summary, key=lambda name: summary[name]["mean_profile_rmse_mm"])
    x = np.arange(len(order))
    width = 0.8 / len(profiles)
    fig, axis = plt.subplots(figsize=(max(11, len(order) * 0.85), 6))
    for index, profile in enumerate(profiles):
        values = [summary[name]["profile_rmse_mm"][profile] for name in order]
        axis.bar(x + (index - (len(profiles)-1)/2) * width, values,
                 width=width, label=profile)
    axis.axhline(7.5, color="tab:red", linestyle="--", label="initial model-only ~7.5 mm (slow)")
    axis.set_yscale("log")
    axis.set_ylabel("Tracking RMSE [mm, log scale]")
    axis.set_xticks(x, order, rotation=40, ha="right")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "screen_rmse.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("fixed", "adaptive", "temperature"),
        default="fixed",
    )
    args = parser.parse_args()
    preflight = cuda_preflight()
    if args.phase == "fixed":
        candidates = _fixed_candidates()
    elif args.phase == "adaptive":
        candidates = _adaptive_candidates()
    else:
        candidates = _temperature_candidates()
    spec = {
        "phase": args.phase,
        "steps": STEPS,
        "seed": SCREEN_SEED,
        "profiles": SCREEN_PROFILES,
        "candidates": candidates,
        "mode": "rollout_model_only",
        "environment": environment_spec(),
        "source": source_fingerprint(Path.cwd(), "run", {}),
        "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output = Path("results/model_only_covariance_gpu") / fingerprint[:12]
    report_path = output / "metrics.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        rows = report.get("rows", [])
    else:
        rows = []
    completed = {(row["candidate"], row["profile"]) for row in rows}
    histories: dict[str, np.ndarray] = {}
    history_path = output / "screen_histories.npz"
    if history_path.exists():
        with np.load(history_path) as old:
            histories.update({key: old[key] for key in old.files})

    profiles = disturbance_profiles(STEPS, large_step_scale=8.0)
    reference = waypoints("circle", n_steps=STEPS)
    print("[CUDA PREFLIGHT]", json.dumps(preflight), flush=True)
    for candidate, settings in candidates.items():
        for profile in SCREEN_PROFILES:
            if (candidate, profile) in completed:
                continue
            cuda_preflight()
            print(f"[RUN] {args.phase} {candidate} {profile}", flush=True)
            data = run_mppi(
                reference, has_disturbance=True, use_dob=True,
                dob_compensation_mode="rollout_model_only",
                disturbance_function=profiles[profile], require_cuda=True,
                seed=SCREEN_SEED, **settings,
            )
            if str(data["compute_device"]) != "cuda":
                raise RuntimeError("CPU result rejected")
            row = {
                "candidate": candidate,
                "profile": profile,
                "settings": settings,
                "metrics": _metrics(data),
            }
            rows.append(row)
            histories[f"{candidate}__{profile}__err_mm"] = data["err_mm"]
            histories[f"{candidate}__{profile}__u_mppi"] = data["u_mppi"]
            histories[f"{candidate}__{profile}__d_hat_used"] = data["d_hat_used"]
            report = {
                "fingerprint": fingerprint, "spec": spec,
                "cuda_preflight": preflight, "rows": rows,
                "summary": _summarize(rows, SCREEN_PROFILES),
            }
            _save(output, report, histories)
            print(json.dumps(row), flush=True)
            del data
            gc.collect()
            torch.cuda.empty_cache()

    report["summary"] = _summarize(rows, SCREEN_PROFILES)
    _save(output, report, histories)
    _plot(output, report["summary"], SCREEN_PROFILES)
    print("[RANKING]", flush=True)
    for name, item in sorted(report["summary"].items(),
                             key=lambda pair: pair[1]["mean_profile_rmse_mm"]):
        print(name, json.dumps(item), flush=True)
    print("[SAVED]", output, flush=True)


if __name__ == "__main__":
    main()
