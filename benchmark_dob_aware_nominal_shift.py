"""GPU-only comparison of baseline, post-DOB, and DOB-aware nominal shift."""
from __future__ import annotations

import argparse
import gc
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from benchmark_hardware_mppi import environment_metadata
from config import D_AMP, D_FREQ, DT, TAU_MAX
from experiment_runner import run_mppi, waypoints


MODES = {
    "baseline": {"use_dob": False, "dob_compensation_mode": "output"},
    "post_dob": {"use_dob": True, "dob_compensation_mode": "output"},
    "aware_nominal_shift": {
        "use_dob": True,
        "dob_compensation_mode": "rollout_nominal_shift",
    },
}


def _tensor(values, device, dtype):
    return torch.tensor(values, device=device, dtype=dtype)


def disturbance_profiles(steps: int, large_step_scale: float = 4.0
                         ) -> dict[str, Callable]:
    amplitudes = np.asarray(D_AMP, dtype=float)

    def constant(_step, device, dtype):
        return _tensor(amplitudes, device, dtype)

    def step(step, device, dtype):
        values = amplitudes if step >= steps // 4 else np.zeros(6)
        return _tensor(values, device, dtype)

    def slow_sinusoidal(step, device, dtype):
        time_s = step * DT
        values = [a * math.sin(2 * math.pi * f * time_s)
                  for a, f in zip(amplitudes, D_FREQ)]
        return _tensor(values, device, dtype)

    def fast_sinusoidal(step, device, dtype):
        time_s = step * DT
        frequencies = (8.0, 9.0, 10.0, 11.0, 12.0, 13.0)
        values = [a * math.sin(2 * math.pi * f * time_s)
                  for a, f in zip(amplitudes, frequencies)]
        return _tensor(values, device, dtype)

    def large_step(step, device, dtype):
        values = large_step_scale * amplitudes if step >= steps // 4 else np.zeros(6)
        return _tensor(values, device, dtype)

    return {
        "constant": constant,
        "step": step,
        "slow_sinusoidal": slow_sinusoidal,
        "fast_sinusoidal": fast_sinusoidal,
        "large_step": large_step,
    }


def cuda_preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA preflight failed; CPU fallback results will not be generated."
        )
    device = torch.device("cuda")
    probe = torch.randn((256, 256), device=device)
    _ = probe @ probe
    torch.cuda.synchronize(device)
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "torch": torch.__version__,
    }


def metrics(data: dict[str, Any]) -> dict[str, Any]:
    error = np.asarray(data["err_mm"], dtype=float)
    torque = np.asarray(data["u_applied"], dtype=float)
    delta = np.diff(torque, axis=0)
    delta_norm = np.linalg.norm(delta, axis=1)
    solve_ms = np.asarray(data["solve_t"], dtype=float) * 1e3
    solve_ms = solve_ms[np.isfinite(solve_ms)]
    saturated = np.asarray(data["actuator_saturated"], dtype=bool)
    residual_disturbance = np.asarray(data["d_tilde"], dtype=float)
    residual_norm = np.linalg.norm(residual_disturbance, axis=1)
    correlation = (
        float(np.corrcoef(residual_norm, error)[0, 1])
        if np.std(residual_norm) > 0 and np.std(error) > 0 else None
    )
    q = np.asarray(data["q"], dtype=float)[1:]
    q_reference = np.asarray(data["q_reference"], dtype=float)[1:len(q) + 1]
    joint_error = q - q_reference
    limits = np.asarray(TAU_MAX, dtype=float)
    margin = limits - np.abs(torque)
    return {
        "tracking_mean_mm": float(error.mean()),
        "tracking_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "tracking_max_mm": float(error.max()),
        "joint_rmse_rad": float(np.sqrt(np.mean(joint_error ** 2))),
        "actuator_torque_rms_nm": float(np.sqrt(np.mean(torque ** 2))),
        "actuator_torque_peak_nm": float(np.abs(torque).max()),
        "torque_rate_rms_nm_per_step": float(np.sqrt(np.mean(delta_norm ** 2))),
        "torque_rate_peak_nm_per_step": float(delta_norm.max()),
        "minimum_torque_margin_nm": float(margin.min()),
        "saturated_steps": int(saturated.sum()),
        "saturated_duration_s": float(saturated.sum() * DT),
        "solve_mean_ms": float(solve_ms.mean()),
        "solve_p95_ms": float(np.percentile(solve_ms, 95)),
        "solve_worst_ms": float(solve_ms.max()),
        "dob_residual_rms_nm": float(np.sqrt(np.mean(residual_norm ** 2))),
        "residual_disturbance_tracking_correlation": correlation,
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = [
        "tracking_rmse_mm", "tracking_max_mm", "joint_rmse_rad",
        "actuator_torque_rms_nm", "torque_rate_rms_nm_per_step",
        "saturated_steps", "solve_mean_ms", "solve_p95_ms",
    ]
    summary: dict[str, Any] = {}
    for profile in sorted({row["disturbance"] for row in rows}):
        summary[profile] = {}
        for mode in MODES:
            selected = [row["metrics"] for row in rows
                        if row["disturbance"] == profile and row["mode"] == mode]
            entry: dict[str, Any] = {"runs": len(selected)}
            for field in fields:
                values = np.asarray([item[field] for item in selected], dtype=float)
                entry[f"{field}_mean"] = float(values.mean())
                entry[f"{field}_std"] = float(values.std(ddof=1 if len(values) > 1 else 0))
            summary[profile][mode] = entry
    return summary


def run(seeds: list[int], steps: int, output_dir: Path,
        profile_names: list[str] | None = None,
        large_step_scale: float = 4.0) -> None:
    preflight = cuda_preflight()
    print("[CUDA PREFLIGHT] " + json.dumps(preflight))
    profiles = disturbance_profiles(steps, large_step_scale)
    if profile_names is not None:
        unknown = set(profile_names) - set(profiles)
        if unknown:
            raise ValueError(f"unknown disturbance profiles: {sorted(unknown)}")
        profiles = {name: profiles[name] for name in profile_names}
    reference = waypoints("circle", n_steps=steps)
    rows: list[dict[str, Any]] = []
    payload: dict[str, np.ndarray] = {}

    for profile_index, (profile_name, disturbance_fn) in enumerate(profiles.items()):
        for seed_index, seed in enumerate(seeds):
            mode_names = list(MODES)
            if (profile_index + seed_index) % 2:
                mode_names.reverse()
            for mode in mode_names:
                settings = MODES[mode]
                print(f"\n[RUN] disturbance={profile_name}, seed={seed}, mode={mode}")
                data = run_mppi(
                    reference, has_disturbance=True,
                    use_dob=settings["use_dob"],
                    dob_compensation_mode=settings["dob_compensation_mode"],
                    disturbance_function=disturbance_fn,
                    require_cuda=True, seed=seed,
                )
                if str(data["compute_device"]) != "cuda":
                    raise RuntimeError("non-CUDA result rejected before saving")
                row = {
                    "disturbance": profile_name,
                    "seed": seed,
                    "mode": mode,
                    "metrics": metrics(data),
                }
                rows.append(row)
                print(json.dumps(row, indent=2))
                prefix = f"{profile_name}__{mode}__seed_{seed}"
                for name, value in data.items():
                    payload[f"{prefix}__{name}"] = np.asarray(value)
                del data
                gc.collect()
                torch.cuda.empty_cache()

    report = {
        "benchmark": "dob_aware_nominal_shift",
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": environment_metadata(),
        "cuda_preflight": preflight,
        "trajectory": "circle",
        "duration_s": steps * DT,
        "steps": steps,
        "large_step_scale": large_step_scale,
        "seeds": seeds,
        "modes": MODES,
        "disturbance_profiles": list(profiles),
        "rows": rows,
        "summary": aggregate(rows),
    }
    if not report["environment"].get("cuda_available"):
        raise RuntimeError("CUDA metadata validation failed; refusing to save")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    np.savez_compressed(output_dir / "results.npz", **payload)
    print(f"[SAVE] {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--large-step-scale", type=float, default=4.0)
    parser.add_argument("--profiles", nargs="+", choices=(
        "constant", "step", "slow_sinusoidal", "fast_sinusoidal", "large_step",
    ))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dob_aware_nominal_shift_gpu"),
    )
    args = parser.parse_args()
    run(args.seeds, args.steps, args.output_dir, args.profiles,
        args.large_step_scale)


if __name__ == "__main__":
    main()
