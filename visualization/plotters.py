"""Result metrics, time-series, 3D trajectory, and dashboard plots."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import CROSS_RADIUS, DT, N_STEPS, TIMING_WARMUP_STEPS, TRAJ_CENTER
from utils.trajectory import crossing_obstacle_paths

LABELS = {"MPPI": "MPPI", "MPPI_DOB": "MPPI + DOB",
          "MPPI_CROSS": "MPPI", "MPPI_DOB_CROSS": "MPPI + DOB",
          "SQP": "SQP", "SQP_DOB": "SQP + DOB"}
COLORS = {"MPPI": "#d85b16", "MPPI_DOB": "#1261a6",
          "MPPI_CROSS": "#d85b16", "MPPI_DOB_CROSS": "#1261a6",
          "SQP": "#7c3aed", "SQP_DOB": "#059669"}
FIG_BG, AX_BG, GRID, TEXT, MUTED = "#f6f7f9", "#ffffff", "#d7dde5", "#20242a", "#6b7280"


def _style_axis(axis, title=None, ylabel=None):
    axis.set_facecolor(AX_BG)
    axis.grid(True, axis="y", color=GRID, linewidth=0.8, alpha=0.65)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors="#384150", labelsize=9)
    if title:
        axis.set_title(title, fontsize=11, fontweight="bold", color=TEXT, pad=10)
    if ylabel:
        axis.set_ylabel(ylabel, fontsize=9, color="#384150")


def compute_metrics(data: dict) -> dict:
    error = data["err"] if "err" in data else data["err_mm"] / 1e3
    solve_time = filtered_solve_times(data, preserve_length=False)
    estimate, actual = data.get("d_hat"), data.get("d_true")
    dob_rms = 0.0
    if estimate is not None and actual is not None:
        count = min(len(estimate), len(actual))
        dob_rms = float(np.sqrt(np.mean((actual[:count] - estimate[:count]) ** 2)))
    return {
        "mean_err_mm": float(np.mean(error) * 1e3),
        "rmse_err_mm": float(np.sqrt(np.mean(error ** 2)) * 1e3),
        "std_err_mm": float(np.std(error) * 1e3),
        "max_err_mm": float(np.max(error) * 1e3),
        "final_err_mm": float(error[-1] * 1e3),
        "mean_solve_ms": float(np.mean(solve_time) * 1e3) if len(solve_time) else 0.0,
        "worst_solve_ms": float(np.max(solve_time) * 1e3) if len(solve_time) else 0.0,
        "dob_rms_nm": dob_rms,
    }


def filtered_solve_times(data: dict, preserve_length: bool = True) -> np.ndarray:
    """Remove startup compilation/cache time from realtime measurements.

    New result files already store warm-up samples as NaN. Slicing is repeated
    here so legacy NPZ files receive the same treatment.
    """
    error = data["err"] if "err" in data else data["err_mm"] / 1e3
    values = np.asarray(data.get("solve_t", np.zeros_like(error)), dtype=float).copy()
    values[:min(TIMING_WARMUP_STEPS, len(values))] = np.nan
    if preserve_length:
        return values
    return values[np.isfinite(values)]


def plot_timeseries(results: dict):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.8), sharex=True, facecolor=FIG_BG)
    fig.suptitle("Time Series — Tracking Error and Step Time", fontsize=13, fontweight="bold")
    time = np.arange(N_STEPS) * DT
    for key, data in results.items():
        error = data["err"] if "err" in data else data["err_mm"] / 1e3
        solve_time = filtered_solve_times(data, preserve_length=True)
        axes[0].plot(time[:len(error)], error * 1e3, label=LABELS.get(key, key), color=COLORS.get(key), lw=2)
        axes[1].plot(time[:len(solve_time)], solve_time * 1e3, label=LABELS.get(key, key), color=COLORS.get(key), lw=2)
    _style_axis(axes[0], "Tracking Error", "EE error [mm]")
    _style_axis(axes[1], "Realtime Cost", "step time [ms]")
    axes[1].axhline(DT * 1e3, color="#111827", ls="--", label="control period")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.legend(frameon=False)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    return fig


def plot_3d_trajectory(results: dict, waypoints: np.ndarray, show_obstacles: bool = False):
    fig = plt.figure(figsize=(7.4, 6.2), facecolor=FIG_BG)
    axis = fig.add_subplot(111, projection="3d")
    axis.set_title("Final 3D Trajectory", fontsize=13, fontweight="bold")
    axis.plot(*waypoints.T, "--", color="#111827", lw=1.6, label="Reference")
    for key, data in results.items():
        ee = data["ee"]
        axis.plot(*ee.T, label=LABELS.get(key, key), color=COLORS.get(key), lw=2)
    if show_obstacles:
        for index, path in enumerate(crossing_obstacle_paths(N_STEPS + 1, TRAJ_CENTER)):
            axis.plot(*path.T, ":", lw=1.8, label=f"Obstacle {index + 1}")
    axis.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
    axis.set_zlim(0.1, 0.6)
    axis.view_init(elev=25, azim=-55)
    axis.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_result_dashboard(results: dict):
    keys = list(results)
    metrics = {key: compute_metrics(value) for key, value in results.items()}
    fig = plt.figure(figsize=(11.4, 5.8), facecolor=FIG_BG, layout="constrained")
    grid = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.5, 1.0], wspace=0.30)
    fig.suptitle("Controller Performance Metrics", fontsize=15, fontweight="bold")

    def bars(axis, names, fields, title, ylabel, legend=False):
        x = np.arange(len(names)); width = 0.72 / max(len(keys), 1)
        for index, key in enumerate(keys):
            offset = (index - (len(keys) - 1) / 2) * width
            values = [metrics[key][field] for field in fields]
            plotted = axis.bar(x + offset, values, width * 0.92,
                               color=COLORS.get(key), label=LABELS.get(key, key))
            upper = max(values, default=0.0)
            for bar, value in zip(plotted, values):
                axis.text(bar.get_x() + bar.get_width() / 2,
                          value + max(upper * 0.02, 0.01), f"{value:.2f}",
                          ha="center", va="bottom", fontsize=8)
        _style_axis(axis, title, ylabel); axis.set_xticks(x, names)
        if legend:
            axis.legend(frameon=False)

    bars(fig.add_subplot(grid[0, 0]), ["Mean err", "RMSE", "Std"],
         ["mean_err_mm", "rmse_err_mm", "std_err_mm"], "Tracking Error Metrics", "error [mm]", True)
    bars(fig.add_subplot(grid[0, 1]), ["MeanT", "WorstT"],
         ["mean_solve_ms", "worst_solve_ms"], "Computation Time", "step time [ms]")
    return fig
