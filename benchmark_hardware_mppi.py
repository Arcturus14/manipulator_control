"""Reproducible MPPI throughput benchmark for comparing CUDA machines."""
from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import (B_DAMP, DQ_MAX, DT, NOISE_VAR, TAU_MAX,
                    TIMING_WARMUP_STEPS, W_ACT, W_VEL)
from experiment_runner import run_mppi, waypoints


def _nvidia_smi() -> dict[str, str]:
    fields = ("driver_version", "vbios_version", "pstate", "power.limit")
    try:
        completed = subprocess.run(
            ["nvidia-smi", f"--query-gpu={','.join(fields)}",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        values = [part.strip() for part in completed.stdout.splitlines()[0].split(",")]
        return dict(zip(fields, values))
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        return {}


def environment_metadata() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    metadata: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pytorch_mppi": importlib.metadata.version("pytorch-mppi"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if cuda_available:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update({
            "cuda_device_index": index,
            "gpu_name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "vram_bytes": properties.total_memory,
        })
    metadata.update(_nvidia_smi())
    return metadata


def _run_metrics(data: dict) -> dict[str, float | int]:
    solve_ms = np.asarray(data["solve_t"], dtype=float) * 1e3
    solve_ms = solve_ms[np.isfinite(solve_ms)]
    if not len(solve_ms):
        raise RuntimeError("No finite MPPI solve-time samples were recorded")
    error = np.asarray(data["err_mm"], dtype=float)
    torque = np.asarray(data["u"], dtype=float)
    saturated = np.isclose(
        np.abs(torque), np.asarray(TAU_MAX), rtol=0.0, atol=1e-5
    ).any(axis=1)
    return {
        "timed_steps": int(len(solve_ms)),
        "mean_solve_ms": float(np.mean(solve_ms)),
        "median_solve_ms": float(np.median(solve_ms)),
        "std_solve_ms": float(np.std(solve_ms)),
        "p95_solve_ms": float(np.percentile(solve_ms, 95)),
        "p99_solve_ms": float(np.percentile(solve_ms, 99)),
        "worst_solve_ms": float(np.max(solve_ms)),
        "deadline_miss_pct": float(np.mean(solve_ms > DT * 1e3) * 100),
        "steps_per_second": float(1e3 / np.mean(solve_ms)),
        "realtime_margin": float(DT * 1e3 / np.mean(solve_ms)),
        "tracking_rmse_mm": float(np.sqrt(np.mean(error ** 2))),
        "tracking_mean_mm": float(np.mean(error)),
        "tracking_max_mm": float(np.max(error)),
        "saturated_step_pct": float(np.mean(saturated) * 100),
    }


def _aggregate(runs: list[dict], samples: list[np.ndarray]) -> dict[str, Any]:
    solve_ms = np.concatenate(samples)
    run_means = np.asarray([row["metrics"]["mean_solve_ms"] for row in runs])
    return {
        "runs": len(runs),
        "timed_steps": int(len(solve_ms)),
        "mean_solve_ms": float(np.mean(solve_ms)),
        "median_solve_ms": float(np.median(solve_ms)),
        "std_solve_ms": float(np.std(solve_ms)),
        "p95_solve_ms": float(np.percentile(solve_ms, 95)),
        "p99_solve_ms": float(np.percentile(solve_ms, 99)),
        "worst_solve_ms": float(np.max(solve_ms)),
        "deadline_miss_pct": float(np.mean(solve_ms > DT * 1e3) * 100),
        "steps_per_second": float(1e3 / np.mean(solve_ms)),
        "realtime_margin": float(DT * 1e3 / np.mean(solve_ms)),
        "run_mean_solve_ms_std": float(np.std(run_means, ddof=1)) if len(runs) > 1 else 0.0,
        "tracking_rmse_mm_mean": float(np.mean(
            [row["metrics"]["tracking_rmse_mm"] for row in runs]
        )),
    }


def run_benchmark(*, seeds: list[int], trajectory: str, output: Path,
                  n_samples: int, horizon: int, temperature: float,
                  mean_update_alpha: float,
                  noise_variance: list[float],
                  position_weight: float, terminal_weight: float,
                  smooth_weight: float, require_cuda: bool = True) -> None:
    environment = environment_metadata()
    if require_cuda and not environment["cuda_available"]:
        raise RuntimeError(
            "CUDA is required for the hardware benchmark, but torch.cuda.is_available() "
            "is false. Check the NVIDIA driver and CUDA-enabled PyTorch environment."
        )
    if not seeds:
        raise ValueError("Hardware benchmark requires at least one seed")

    reference = waypoints(trajectory)
    runs: list[dict] = []
    samples: list[np.ndarray] = []
    print(f"[HARDWARE] gpu={environment.get('gpu_name', 'CPU')}, seeds={seeds}")
    for run_index, seed in enumerate(seeds, start=1):
        print(f"\n[HARDWARE] run={run_index}/{len(seeds)}, seed={seed}")
        data = run_mppi(
            reference, has_disturbance=False, use_dob=False,
            dynamic_paths=None, static_centers=None, seed=seed,
            n_samples=n_samples, horizon=horizon, temperature=temperature,
            mean_update_alpha=mean_update_alpha,
            noise_variance=noise_variance,
            position_weight=position_weight, terminal_weight=terminal_weight,
            smooth_weight=smooth_weight,
        )
        solve_ms = np.asarray(data["solve_t"], dtype=float) * 1e3
        solve_ms = solve_ms[np.isfinite(solve_ms)]
        samples.append(solve_ms)
        row = {"run": run_index, "seed": seed, "metrics": _run_metrics(data)}
        runs.append(row)
        print(json.dumps(row, indent=2))
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    conditions = {
        "trajectory": trajectory,
        "obstacle": "none",
        "disturbance": False,
        "dob": False,
        "tracking_only": True,
        "dt_s": DT,
        "steps": int(len(reference)),
        "seeds": list(seeds),
        "software": {
            "python": environment["python"],
            "torch": environment["torch"],
            "cuda_runtime": environment["cuda_runtime"],
            "numpy": environment["numpy"],
            "pytorch_mppi": environment["pytorch_mppi"],
        },
        "plant": {
            "torque_limit": list(TAU_MAX),
            "velocity_limit": DQ_MAX,
            "damping": list(B_DAMP),
        },
        "mppi": {
            "dtype": "float32",
            "samples": n_samples,
            "horizon": horizon,
            "temperature": temperature,
            "mean_update_alpha": mean_update_alpha,
            "noise_variance": list(noise_variance),
            "position_weight": position_weight,
            "terminal_weight": terminal_weight,
            "velocity_weight": W_VEL,
            "action_weight": W_ACT,
            "smooth_weight": smooth_weight,
        },
        "timing": {
            "kernel_warmup_outside_control_loop": True,
            "discarded_control_steps": TIMING_WARMUP_STEPS,
            "cuda_synchronized_per_step": True,
        },
    }
    report = {
        "schema_version": 1,
        "benchmark": "mppi_hardware_tracking",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "conditions": conditions,
        "runs": runs,
        "aggregate": _aggregate(runs, samples),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n[HARDWARE SUMMARY]")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"[SAVE] {output}")
