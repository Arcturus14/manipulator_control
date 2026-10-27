"""GPU-only Model-only MPPI stress test near the official UR5 velocity limit."""
from __future__ import annotations

# Executable experiment module in the benchmarks package.

import gc
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import (
    cuda_preflight,
    disturbance_profiles,
    metrics,
)
from config import DQ_MAX, DT, NOISE_VAR, TAU_MAX
from experiment_runner import run_mppi, waypoints
from utils.result_cache import environment_spec, source_fingerprint


STEPS = 250
SEEDS = (7, 17, 27)
DISTURBANCE_SCALE = 4.0
SETTINGS = {
    "noise_variance": list(NOISE_VAR),
    "temporal_noise_rho": 0.8,
    "temperature": 1.0,
    "disturbance_covariance_gain": 0.0,
}


def _summary(rows: list[dict]) -> dict:
    fields = (
        "tracking_rmse_mm",
        "tracking_max_mm",
        "joint_velocity_abs_max_rad_s",
        "joint_position_limit_violation_steps",
        "joint_velocity_limit_violation_steps",
        "saturated_steps",
    )
    result = {}
    for field in fields:
        values = np.asarray([row["metrics"][field] for row in rows], dtype=float)
        result[field] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    return result


def _plot(output: Path, representative: dict) -> None:
    qd = np.asarray(representative["qd"], dtype=float)
    error = np.asarray(representative["err_mm"], dtype=float)
    torque = np.asarray(representative["u_applied"], dtype=float)
    time_state = np.arange(len(qd)) * DT
    time_control = np.arange(len(error)) * DT
    torque_ratio = np.max(np.abs(torque) / np.asarray(TAU_MAX), axis=1)
    violation = np.any(np.abs(qd) > DQ_MAX, axis=1)

    figure, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for joint in range(6):
        axes[0].plot(time_state, qd[:, joint], lw=1.2, label=f"J{joint + 1}")
    axes[0].axhline(DQ_MAX, color="black", ls="--", lw=1.2,
                    label=r"official limit $\pm\pi$ rad/s")
    axes[0].axhline(-DQ_MAX, color="black", ls="--", lw=1.2)
    axes[0].set_ylabel(r"Joint velocity [rad/s]")
    axes[0].legend(ncol=4, fontsize=8)

    axes[1].plot(time_control, error, color="#d62728", lw=1.5)
    axes[1].set_ylabel("EE error [mm]")

    axes[2].plot(time_control, torque_ratio, color="#2ca02c", lw=1.5)
    axes[2].axhline(1.0, color="black", ls="--", lw=1.2,
                    label="actuator torque limit")
    axes[2].set_ylabel(r"max $|u_j|/u_{j,max}$")
    axes[2].set_xlabel("Time [s]")
    axes[2].legend(fontsize=8)

    disturbance_start = (STEPS // 4) * DT
    for axis in axes:
        axis.axvline(disturbance_start, color="#9467bd", ls=":", lw=1.2)
        axis.grid(alpha=0.25)
    if np.any(violation):
        first = time_state[np.flatnonzero(violation)[0]]
        for axis in axes:
            axis.axvline(first, color="#ff7f0e", ls="--", lw=1.2)
    figure.suptitle(
        "Model-only velocity stress: 4x step disturbance, seed 7"
    )
    figure.tight_layout()
    figure.savefig(output / "velocity_stress.png", dpi=180)
    plt.close(figure)


def main() -> None:
    preflight = cuda_preflight()
    spec = {
        "benchmark": "model_only_velocity_stress",
        "steps": STEPS,
        "seeds": SEEDS,
        "disturbance_profile": "large_step",
        "disturbance_scale": DISTURBANCE_SCALE,
        "disturbance_start_step": STEPS // 4,
        "mode": "rollout_model_only",
        "settings": SETTINGS,
        "official_velocity_limit_rad_s": DQ_MAX,
        "environment": environment_spec(),
        "source": source_fingerprint(Path.cwd(), "run", {}),
        "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    output = Path("results/model_only_velocity_stress_gpu") / fingerprint[:12]
    required = ("metrics.json", "results.npz", "velocity_stress.png")
    if all((output / name).exists() for name in required):
        print("REUSE", output)
        return

    reference = waypoints("circle", n_steps=STEPS)
    disturbance = disturbance_profiles(
        STEPS, large_step_scale=DISTURBANCE_SCALE
    )["large_step"]
    rows: list[dict] = []
    payload: dict[str, np.ndarray] = {}
    representative = None
    for seed in SEEDS:
        cuda_preflight()
        data = run_mppi(
            reference,
            has_disturbance=True,
            use_dob=True,
            dob_compensation_mode="rollout_model_only",
            disturbance_function=disturbance,
            require_cuda=True,
            seed=seed,
            **SETTINGS,
        )
        if str(data["compute_device"]) != "cuda":
            raise RuntimeError("CPU result rejected")
        row = {"seed": seed, "metrics": metrics(data)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        for name, value in data.items():
            if not name.startswith("_"):
                payload[f"seed_{seed}__{name}"] = np.asarray(value)
        if seed == SEEDS[0]:
            representative = data
        else:
            del data
        gc.collect()
        torch.cuda.empty_cache()

    report = {
        "fingerprint": fingerprint,
        "spec": spec,
        "cuda_preflight": preflight,
        "rows": rows,
        "summary": _summary(rows),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(output / "results.npz", **payload)
    assert representative is not None
    _plot(output, representative)
    print("SAVED", output)


if __name__ == "__main__":
    main()
