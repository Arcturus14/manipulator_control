"""Sequentially tune MPPI parameters for mean_update_alpha=0.9."""
from __future__ import annotations

import argparse
import gc
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark_hardware_mppi import environment_metadata
from benchmark_mean_update_mppi import _metrics
from experiment_runner import run_mppi, waypoints
from visualization.plotters import plot_joint_torques


ORIGINAL = {
    "n_samples": 136,
    "horizon": 10,
    "temperature": 0.2,
    "noise_variance": [16.0, 28.0, 16.0, 3.2, 3.2, 1.0],
    "position_weight": 960_000.0,
    "terminal_weight": 2_400_000.0,
    "smooth_weight": 0.25,
}
TUNED_ALPHA = 0.9
SCENARIOS = {
    "dob_off": {"has_disturbance": False, "use_dob": False},
    "dob_on": {"has_disturbance": True, "use_dob": True},
}


def _parameter_key(parameters: dict[str, Any]) -> str:
    return json.dumps(parameters, sort_keys=True, separators=(",", ":"))


def _aggregate(rows: list[dict]) -> dict[str, Any]:
    fields = (
        "rmse_err_mm", "mean_solve_ms", "worst_solve_ms",
        "mean_delta_tau_l2_nm", "rms_delta_tau_l2_nm",
        "p95_delta_tau_l2_nm", "max_delta_tau_l2_nm",
    )
    result: dict[str, Any] = {}
    for variant in ("original_alpha_1", "tuned_alpha_0.9"):
        for scenario in SCENARIOS:
            selected = [
                row for row in rows
                if row["variant"] == variant and row["scenario"] == scenario
            ]
            summary: dict[str, Any] = {"runs": len(selected)}
            for field in fields:
                values = np.asarray([row["metrics"][field] for row in selected])
                summary[f"{field}_mean"] = float(values.mean())
                summary[f"{field}_std"] = float(
                    values.std(ddof=1 if len(values) > 1 else 0)
                )
            per_joint = np.asarray([
                row["metrics"]["rms_delta_tau_per_joint_nm"]
                for row in selected
            ])
            summary["rms_delta_tau_per_joint_nm_mean"] = per_joint.mean(axis=0).tolist()
            result[f"{variant}__{scenario}"] = summary
    return result


class SequentialTuner:
    def __init__(self, reference: np.ndarray, output_dir: Path):
        self.reference = reference
        self.output_dir = output_dir
        self.cache: dict[tuple[str, float, int, str], dict[str, Any]] = {}
        self.stage_reports: list[dict[str, Any]] = []
        self.reference_metrics: dict[str, dict[str, Any]] = {}

    def evaluate(self, parameters: dict[str, Any], *, alpha: float,
                 seed: int, scenario: str) -> dict[str, Any]:
        key = (_parameter_key(parameters), alpha, seed, scenario)
        if key in self.cache:
            return self.cache[key]
        condition = SCENARIOS[scenario]
        print(
            f"\n[TUNE RUN] scenario={scenario}, seed={seed}, alpha={alpha:g}, "
            f"K={parameters['n_samples']}, T={parameters['horizon']}, "
            f"temp={parameters['temperature']:g}"
        )
        data = run_mppi(
            self.reference, seed=seed, mean_update_alpha=alpha,
            dynamic_paths=None, static_centers=None, **condition, **parameters,
        )
        metrics = _metrics(data)
        self.cache[key] = metrics
        print(json.dumps(metrics, indent=2))
        del data
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics

    def establish_reference(self, seed: int) -> None:
        self.reference_metrics = {
            scenario: self.evaluate(ORIGINAL, alpha=1.0, seed=seed,
                                    scenario=scenario)
            for scenario in SCENARIOS
        }

    def score(self, metrics: dict[str, dict[str, Any]]) -> dict[str, float]:
        tracking_ratio = float(np.mean([
            metrics[name]["rmse_err_mm"] / self.reference_metrics[name]["rmse_err_mm"]
            for name in SCENARIOS
        ]))
        smoothness_ratio = float(np.mean([
            metrics[name]["rms_delta_tau_l2_nm"]
            / self.reference_metrics[name]["rms_delta_tau_l2_nm"]
            for name in SCENARIOS
        ]))
        realtime_penalty = float(sum(
            max(0.0, metrics[name]["mean_solve_ms"] / 20.0 - 1.0) * 5.0
            for name in SCENARIOS
        ))
        instability_penalty = float(sum(
            max(0.0, metrics[name]["rmse_err_mm"] / 10.0 - 1.0) * 10.0
            for name in SCENARIOS
        ))
        return {
            "objective": 0.6 * tracking_ratio + 0.4 * smoothness_ratio
                         + realtime_penalty + instability_penalty,
            "tracking_ratio": tracking_ratio,
            "smoothness_ratio": smoothness_ratio,
            "realtime_penalty": realtime_penalty,
            "instability_penalty": instability_penalty,
        }

    def tune_stage(self, name: str, candidates: list[dict[str, Any]], *,
                   seed: int) -> dict[str, Any]:
        print(f"\n{'=' * 20} STAGE: {name} {'=' * 20}")
        evaluated = []
        for index, parameters in enumerate(candidates):
            metrics = {
                scenario: self.evaluate(
                    parameters, alpha=TUNED_ALPHA, seed=seed, scenario=scenario
                )
                for scenario in SCENARIOS
            }
            score = self.score(metrics)
            evaluated.append({
                "candidate": index,
                "parameters": parameters,
                "metrics": metrics,
                "score": score,
            })
            print(f"[TUNE SCORE] {name} candidate={index}: {score}")
        winner = min(evaluated, key=lambda row: row["score"]["objective"])
        report = {"stage": name, "candidates": evaluated, "winner": winner}
        self.stage_reports.append(report)
        self._save_progress()
        print(
            f"[STAGE WINNER] {name}: candidate={winner['candidate']}, "
            f"objective={winner['score']['objective']:.6f}"
        )
        return deepcopy(winner["parameters"])

    def _save_progress(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "objective": {
                "tracking_weight": 0.6,
                "torque_smoothness_weight": 0.4,
                "realtime_limit_ms": 20.0,
            },
            "reference_metrics_seed_7": self.reference_metrics,
            "stages": self.stage_reports,
        }
        (self.output_dir / "tuning_progress.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def _with(base: dict[str, Any], **changes: Any) -> dict[str, Any]:
    result = deepcopy(base)
    result.update(changes)
    return result


def _noise_scaled(base: dict[str, Any], scale: float) -> dict[str, Any]:
    return _with(
        base,
        noise_variance=[float(value * scale) for value in ORIGINAL["noise_variance"]],
    )


def _final_validation(reference: np.ndarray, tuned: dict[str, Any], *,
                      seeds: list[int], output_dir: Path) -> dict[str, Any]:
    rows: list[dict] = []
    representative: dict[tuple[str, str], dict] = {}
    result_payload: dict[str, np.ndarray] = {}
    variants = {
        "original_alpha_1": (ORIGINAL, 1.0),
        "tuned_alpha_0.9": (tuned, TUNED_ALPHA),
    }
    for seed_index, seed in enumerate(seeds):
        variant_items = list(variants.items())
        if seed_index % 2:
            variant_items.reverse()
        for scenario, condition in SCENARIOS.items():
            for variant, (parameters, alpha) in variant_items:
                print(
                    f"\n[FINAL] {variant}, {scenario}, seed={seed}, alpha={alpha:g}"
                )
                data = run_mppi(
                    reference, seed=seed, mean_update_alpha=alpha,
                    dynamic_paths=None, static_centers=None,
                    **condition, **parameters,
                )
                metrics = _metrics(data)
                rows.append({
                    "variant": variant, "scenario": scenario,
                    "seed": seed, "alpha": alpha, "metrics": metrics,
                })
                prefix = f"{variant}__{scenario}__seed_{seed}"
                for key, value in data.items():
                    result_payload[f"{prefix}__{key}"] = np.asarray(value)
                if seed == seeds[0]:
                    representative[(variant, scenario)] = data
                else:
                    del data
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    summary = _aggregate(rows)
    comparisons: dict[str, Any] = {}
    for scenario in SCENARIOS:
        original = summary[f"original_alpha_1__{scenario}"]
        candidate = summary[f"tuned_alpha_0.9__{scenario}"]

        def reduction(field: str) -> float:
            return 100.0 * (1.0 - candidate[field] / original[field])

        comparisons[scenario] = {
            "tracking_rmse_change_pct": -reduction("rmse_err_mm_mean"),
            "rms_delta_tau_reduction_pct": reduction(
                "rms_delta_tau_l2_nm_mean"
            ),
            "p95_delta_tau_reduction_pct": reduction(
                "p95_delta_tau_l2_nm_mean"
            ),
            "solve_time_change_pct": -reduction("mean_solve_ms_mean"),
        }

        plot_results = {
            "CURRENT": representative[("original_alpha_1", scenario)],
            "MEAN_UPDATE": representative[("tuned_alpha_0.9", scenario)],
        }
        figure = plot_joint_torques(
            plot_results,
            title=("Applied Joint Torque — "
                   f"{'DOB OFF / Disturbance OFF' if scenario == 'dob_off' else 'DOB ON / Disturbance ON'}"),
        )
        destination = output_dir / f"joint_torques_tuned_{scenario}.png"
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        plt.close(figure)
        print(f"[PLOT] {destination}")

    np.savez_compressed(output_dir / "final_results.npz", **result_payload)
    return {"rows": rows, "summary": summary, "comparisons": comparisons}


def run(output_dir: Path, *, require_cuda: bool = False) -> None:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required but unavailable; refusing to record CPU tuning"
        )
    reference = waypoints("infinity")
    tuner = SequentialTuner(reference, output_dir)
    tuner.establish_reference(seed=7)

    selected = deepcopy(ORIGINAL)
    selected = tuner.tune_stage(
        "temperature",
        [_with(selected, temperature=value)
         for value in (0.12, 0.16, 0.20, 0.24, 0.30)],
        seed=7,
    )
    selected = tuner.tune_stage(
        "noise_variance",
        [_noise_scaled(selected, scale)
         for scale in (0.70, 0.85, 1.00, 1.15, 1.30)],
        seed=7,
    )
    weight_pairs = (
        (960_000.0, 2_400_000.0),
        (1_200_000.0, 3_000_000.0),
        (1_440_000.0, 3_600_000.0),
        (1_200_000.0, 3_600_000.0),
        (1_440_000.0, 4_800_000.0),
    )
    selected = tuner.tune_stage(
        "tracking_terminal_weights",
        [_with(selected, position_weight=position, terminal_weight=terminal)
         for position, terminal in weight_pairs],
        seed=7,
    )
    selected = tuner.tune_stage(
        "smoothness_weight",
        [_with(selected, smooth_weight=value)
         for value in (0.10, 0.25, 0.50, 1.00, 2.00)],
        seed=7,
    )
    compute_pairs = ((136, 10), (160, 10), (192, 10), (136, 12), (160, 12))
    selected = tuner.tune_stage(
        "samples_horizon",
        [_with(selected, n_samples=samples, horizon=horizon)
         for samples, horizon in compute_pairs],
        seed=7,
    )

    validation = _final_validation(
        reference, selected, seeds=[7, 17, 27], output_dir=output_dir,
    )
    report = {
        "benchmark": "mppi_alpha_0.9_sequential_tuning",
        "environment": environment_metadata(),
        "require_cuda": require_cuda,
        "trajectory": "infinity",
        "conditions": SCENARIOS,
        "original_parameters": ORIGINAL,
        "tuned_alpha": TUNED_ALPHA,
        "selected_parameters": selected,
        "objective": {
            "tracking_weight": 0.6,
            "torque_smoothness_weight": 0.4,
            "realtime_limit_ms": 20.0,
        },
        "stages": tuner.stage_reports,
        "final_validation": validation,
    }
    destination = output_dir / "tuning_report.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\n[FINAL SELECTED PARAMETERS]")
    print(json.dumps(selected, indent=2))
    print("\n[FINAL COMPARISON]")
    print(json.dumps(validation["comparisons"], indent=2))
    print(f"[SAVE] {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output = Path("results") / f"mean_update_tuning_{stamp}"
    run(output, require_cuda=args.require_cuda)


if __name__ == "__main__":
    main()
