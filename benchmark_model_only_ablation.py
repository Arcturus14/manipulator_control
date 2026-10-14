"""GPU oracle/terminal/sample-budget ablation. Saves data and plots, no MD."""
from __future__ import annotations

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
from experiment_runner import run_mppi, waypoints
from utils.result_cache import environment_spec, source_fingerprint

CASES = {
    "dob_zero": ("dob", "zero", 136),
    "oracle_zero": ("oracle", "zero", 136),
    "dob_hold": ("dob", "hold", 136),
    "oracle_hold": ("oracle", "hold", 136),
    "dob_zero_K544": ("dob", "zero", 544),
    "dob_hold_K544": ("dob", "hold", 544),
}
SEEDS = [7, 17, 27]
STEPS = 250


def main():
    print("CUDA", cuda_preflight(), flush=True)
    spec = {"cases": CASES, "seeds": SEEDS, "steps": STEPS,
            "profiles": ["constant", "step"], "trajectory": "circle",
            "source": source_fingerprint(Path.cwd(), "run", {}),
            "benchmark_source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": environment_spec()}
    fingerprint = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    output = Path("results/model_only_ablation_gpu") / fingerprint[:12]
    if all((output / name).exists() for name in ("metrics.json", "results.npz", "tracking.png", "diagnostics.png")):
        print("REUSE", output)
        return
    rows, payload, representative = [], {}, {}
    profiles = disturbance_profiles(STEPS)
    for pi, profile in enumerate(spec["profiles"]):
        for si, seed in enumerate(SEEDS):
            order = list(CASES)
            if (pi + si) % 2:
                order.reverse()
            for case in order:
                cuda_preflight()
                estimate, terminal, samples = CASES[case]
                print("RUN", profile, seed, case, flush=True)
                data = run_mppi(waypoints("circle", n_steps=STEPS),
                    has_disturbance=True, use_dob=True, require_cuda=True,
                    dob_compensation_mode="rollout_model_only", seed=seed,
                    disturbance_function=profiles[profile],
                    disturbance_estimate_mode=estimate,
                    residual_terminal_mode=terminal, n_samples=samples)
                assert str(data["compute_device"]) == "cuda"
                np.testing.assert_allclose(data["u_applied"], data["u_mppi"], atol=1e-5)
                np.testing.assert_allclose(data["u_effective_pred"], data["u_mppi"]-data["d_hat_used"], atol=1e-5)
                if estimate == "oracle":
                    np.testing.assert_array_equal(data["d_hat_used"], data["d_true"][1:])
                    np.testing.assert_allclose(data["u_effective_pred"], data["u_effective_actual"], atol=1e-5)
                m = metrics(data)
                t = np.asarray(data["solve_t"])
                m["deadline_miss_pct"] = float(100*np.mean(t[np.isfinite(t)] > 0.02))
                m["prediction_q_rmse_rad"] = float(np.sqrt(np.mean(data["state_prediction_error"][:, :6]**2)))
                rows.append({"profile": profile, "seed": seed, "case": case, "metrics": m})
                print(json.dumps(rows[-1]), flush=True)
                for key, value in data.items():
                    if not key.startswith("_"):
                        payload[f"{profile}__{case}__seed_{seed}__{key}"] = value
                if seed == SEEDS[0]:
                    representative[(profile, case)] = data
                del data
                gc.collect()
                torch.cuda.empty_cache()
    summary = {}
    for profile in spec["profiles"]:
        summary[profile] = {}
        for case in CASES:
            selected = [r["metrics"] for r in rows if r["profile"] == profile and r["case"] == case]
            summary[profile][case] = {key: {"mean": float(np.mean([m[key] for m in selected])),
                                                   "std": float(np.std([m[key] for m in selected], ddof=1))}
                for key in selected[0] if all(m[key] is not None for m in selected)}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "results.npz", **payload)
    (output / "metrics.json").write_text(json.dumps({"spec": spec, "fingerprint": fingerprint,
        "rows": rows, "summary": summary,
        "notes": ["Oracle knows exact current disturbance, held constant across horizon.",
                  "d_hat_used is the prediction estimate; d_hat is observer output history.",
                  "Prediction state error also includes rollout/RK4 discretization mismatch.",
                  "Terminal hold preserves last optimized residual without explicitly adding d_hat.",
                  "K544 is an offline sampling-budget diagnostic, not a realtime recommendation."]}, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for column, profile in enumerate(spec["profiles"]):
        for case in CASES:
            data = representative[(profile, case)]
            axes[0, column].plot(np.arange(STEPS)*0.02, data["err_mm"], label=case)
        means = [summary[profile][case]["tracking_rmse_mm"]["mean"] for case in CASES]
        stds = [summary[profile][case]["tracking_rmse_mm"]["std"] for case in CASES]
        axes[1, column].bar(np.arange(len(CASES)), means, yerr=stds, capsize=3)
        axes[1, column].set_xticks(np.arange(len(CASES)), list(CASES), rotation=40, ha="right")
        axes[0, column].set_title(profile + " (seed 7)")
        axes[0, column].set_ylabel("EE error [mm]")
        axes[0, column].set_xlabel("Time [s]")
        axes[1, column].set_ylabel("RMSE [mm], mean +/- SD")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "tracking.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for column, profile in enumerate(spec["profiles"]):
        for case in CASES:
            data = representative[(profile, case)]
            torque_error = data["u_mppi"] - data["u_ff"] - data["d_hat_used"]
            axes[0, column].plot(np.arange(STEPS)*0.02, np.linalg.norm(torque_error, axis=1), label=case, alpha=0.8)
        axes[0, column].set_title(profile + " (seed 7)")
        axes[0, column].set_ylabel("||u_mppi - u_ff - d_hat_used|| [Nm]")
        axes[0, column].set_xlabel("Time [s]")
        means = [summary[profile][case]["solve_mean_ms"]["mean"] for case in CASES]
        axes[1, column].bar(np.arange(len(CASES)), means)
        axes[1, column].axhline(20, color="red", linestyle="--", label="20 ms period")
        axes[1, column].set_xticks(np.arange(len(CASES)), list(CASES), rotation=40, ha="right")
        axes[1, column].set_ylabel("Mean solve [ms]")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend()
    fig.tight_layout()
    fig.savefig(output / "diagnostics.png", dpi=170)
    plt.close(fig)
    print("SAVED", output, flush=True)


if __name__ == "__main__":
    main()
