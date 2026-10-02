"""Generate plots comparing output-compensated and DOB-aware-rollout MPPI."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiment_runner import run_mppi, waypoints
from visualization import plotters


DISPLAY_KEYS = {
    "output": "OUTPUT_DOB",
    "rollout": "ROLLOUT_DOB",
}
plotters.LABELS.update({
    "OUTPUT_DOB": r"Original: $u_{MPPI}+\hat d$",
    "ROLLOUT_DOB": r"DOB-aware rollout: $u-\hat d$",
})
plotters.COLORS.update({
    "OUTPUT_DOB": "#6b7280",
    "ROLLOUT_DOB": "#d85b16",
})


def _save(figure: plt.Figure, destination: Path) -> None:
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    print(f"[PLOT] {destination}")


def _plot_aggregate(report: dict) -> plt.Figure:
    """Plot three-seed means and the paired per-seed tracking RMSE."""
    rows = report["rows"]
    seeds = report["seeds"]
    summary = report["summary"]
    modes = ("output", "rollout")
    labels = ["Original output\ncompensation", "DOB-aware\nrollout"]
    colors = [plotters.COLORS[DISPLAY_KEYS[mode]] for mode in modes]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=plotters.FIG_BG)
    fig.suptitle(
        "DOB Compensation Architecture — Three-Seed Comparison",
        fontsize=14, fontweight="bold",
    )

    fields = (
        ("rmse_err_mm", "Tracking RMSE", "error [mm]"),
        ("max_err_mm", "Maximum Tracking Error", "error [mm]"),
        ("rms_delta_tau_l2_nm", "Torque-Change RMS", r"$\|\Delta\tau\|_2$ [Nm]"),
    )
    for axis, (field, title, ylabel) in zip(axes.flat[:3], fields):
        means = [summary[mode][f"{field}_mean"] for mode in modes]
        stds = [summary[mode][f"{field}_std"] for mode in modes]
        bars = axis.bar(
            np.arange(2), means, yerr=stds, capsize=6,
            color=colors, width=0.64,
        )
        plotters._style_axis(axis, title, ylabel)
        axis.set_xticks(np.arange(2), labels)
        for bar, value in zip(bars, means):
            axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{value:.2f}", ha="center", va="bottom", fontsize=9,
            )

    axis = axes.flat[3]
    by_key = {
        (int(row["seed"]), row["mode"]): row["metrics"]["rmse_err_mm"]
        for row in rows
    }
    for mode in modes:
        values = [by_key[(int(seed), mode)] for seed in seeds]
        axis.plot(
            seeds, values, "o-", linewidth=2, markersize=6,
            color=plotters.COLORS[DISPLAY_KEYS[mode]],
            label=plotters.LABELS[DISPLAY_KEYS[mode]],
        )
    plotters._style_axis(axis, "Tracking RMSE by Seed", "RMSE [mm]")
    axis.set_xlabel("seed")
    axis.set_xticks(seeds)
    axis.legend(frameon=False)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
    return fig


def run(*, seed: int, metrics_path: Path, output_dir: Path) -> None:
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    steps = int(report["steps"])
    reference = waypoints("circle", n_steps=steps)
    results = {}

    for mode in ("output", "rollout"):
        print(f"\n[PLOT DATA] seed={seed}, mode={mode}")
        results[DISPLAY_KEYS[mode]] = run_mppi(
            reference,
            has_disturbance=True,
            use_dob=True,
            dob_compensation_mode=mode,
            dynamic_paths=None,
            static_centers=None,
            seed=seed,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {}
    saved_fields = (
        "q", "ee", "u", "err", "err_mm", "d_hat", "d_true",
        "dob_residual", "dob_tau_model", "solve_t", "solve_t_raw",
    )
    for key, data in results.items():
        for field in saved_fields:
            payload[f"{key}__{field}"] = np.asarray(data[field])
    payload["reference"] = reference
    np.savez_compressed(output_dir / f"representative_seed_{seed}.npz", **payload)

    _save(
        plotters.plot_3d_trajectory(results, reference),
        output_dir / f"seed_{seed}_trajectory.png",
    )
    _save(
        plotters.plot_timeseries(results),
        output_dir / f"seed_{seed}_tracking_and_step_time.png",
    )
    _save(
        plotters.plot_result_dashboard(results),
        output_dir / f"seed_{seed}_performance_dashboard.png",
    )
    _save(
        plotters.plot_joint_torques(
            results, title="Applied Joint Torque — DOB Architecture Comparison"
        ),
        output_dir / f"seed_{seed}_joint_torques.png",
    )
    _save(
        plotters.plot_disturbance_estimation_error(results),
        output_dir / f"seed_{seed}_disturbance_estimation_error.png",
    )
    _save(
        _plot_aggregate(report),
        output_dir / "three_seed_aggregate.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--metrics", type=Path,
        default=Path("results/dob_rollout_comparison/metrics.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/dob_rollout_comparison/plots"),
    )
    args = parser.parse_args()
    run(seed=args.seed, metrics_path=args.metrics, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
