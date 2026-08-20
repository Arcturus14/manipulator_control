"""Result metrics, time-series, 3D trajectory, and dashboard plots."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from config import (DT, DYNAMIC_OBSTACLE_SIZE, STATIC_OBSTACLE_SIZE,
                    TIMING_WARMUP_STEPS)

LABELS = {"MPPI": "MPPI", "MPPI_DOB": "MPPI + DOB",
          "MPPI_CROSS": "MPPI", "MPPI_DOB_CROSS": "MPPI + DOB",
          "SQP": "SQP", "SQP_DOB": "SQP + DOB"}
COLORS = {"MPPI": "#d85b16", "MPPI_DOB": "#1261a6",
          "MPPI_CROSS": "#d85b16", "MPPI_DOB_CROSS": "#1261a6",
          "SQP": "#7c3aed", "SQP_DOB": "#059669"}
FIG_BG, AX_BG, GRID, TEXT, MUTED = "#f6f7f9", "#ffffff", "#d7dde5", "#20242a", "#6b7280"


def _densify_polyline(points: np.ndarray, max_spacing: float) -> np.ndarray:
    """Sample every trajectory segment densely enough to detect cube crossings."""
    points = np.asarray(points, dtype=float)
    if len(points) < 2:
        return points.copy()
    dense = []
    for start, end in zip(points[:-1], points[1:]):
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / max_spacing)))
        fractions = np.arange(count, dtype=float)[:, None] / count
        dense.append(start + fractions * (end - start))
    dense.append(points[-1:])
    return np.vstack(dense)


def _segment_intersects_box(start: np.ndarray, end: np.ndarray,
                            lower: np.ndarray, upper: np.ndarray) -> bool:
    """Exact slab intersection test for one finite line segment and AABB."""
    direction = end - start
    enter, leave = 0.0, 1.0
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if start[axis] < lower[axis] or start[axis] > upper[axis]:
                return False
            continue
        first = (lower[axis] - start[axis]) / direction[axis]
        second = (upper[axis] - start[axis]) / direction[axis]
        enter = max(enter, min(first, second))
        leave = min(leave, max(first, second))
        if enter > leave:
            return False
    return True


def assess_static_obstacles(ee: np.ndarray, centers: np.ndarray,
                            size: float = STATIC_OBSTACLE_SIZE) -> list[dict]:
    """Return continuous-path EE contact and signed clearance for each cube."""
    ee = np.asarray(ee, dtype=float)
    sampled = _densify_polyline(ee, max_spacing=size / 100.0)
    half = size / 2.0
    assessments = []
    for center in np.asarray(centers):
        delta = np.abs(sampled - center)
        outside_distance = np.linalg.norm(np.maximum(delta - half, 0.0), axis=1)
        inside = np.all(delta <= half, axis=1)
        signed_clearance = outside_distance
        signed_clearance[inside] = -np.min(half - delta[inside], axis=1)
        minimum = float(np.min(signed_clearance))
        lower, upper = center - half, center + half
        contact = any(
            _segment_intersects_box(start, end, lower, upper)
            for start, end in zip(ee[:-1], ee[1:])
        ) or bool(np.any(np.all((ee >= lower) & (ee <= upper), axis=1)))
        assessments.append({
            "contact": contact,
            "clearance_mm": min(minimum, 0.0) * 1e3 if contact else minimum * 1e3,
        })
    return assessments


def assess_dynamic_obstacle(ee: np.ndarray, path: np.ndarray,
                            size: float = DYNAMIC_OBSTACLE_SIZE) -> dict:
    """Return continuous relative-motion contact and cube clearance.

    Both the EE and obstacle center are linearly interpolated between matching
    samples. Their difference is therefore tested against a stationary cube.
    """
    ee = np.asarray(ee, dtype=float)
    path = np.asarray(path, dtype=float)
    count = min(len(ee), len(path))
    if count == 0:
        raise ValueError("EE and dynamic obstacle paths must not be empty")
    relative = ee[:count] - path[:count]
    half = size / 2.0
    dense = _densify_polyline(relative, max_spacing=size / 100.0)
    delta = np.abs(dense)
    outside_distance = np.linalg.norm(np.maximum(delta - half, 0.0), axis=1)
    inside = np.all(delta <= half, axis=1)
    signed_clearance = outside_distance
    signed_clearance[inside] = -np.min(half - delta[inside], axis=1)
    contact = any(
        _segment_intersects_box(start, end, -np.full(3, half), np.full(3, half))
        for start, end in zip(relative[:-1], relative[1:])
    ) or bool(np.any(np.all(delta <= half, axis=1)))
    clearance = float(signed_clearance.min())
    return {
        "contact": contact,
        "clearance_mm": clearance * 1e3,
    }


def _set_equal_3d_scale(axis, arrays: list[np.ndarray]) -> None:
    """Use equal metres-per-display-unit so a physical cube looks cubic."""
    points = np.vstack([np.asarray(values).reshape(-1, 3) for values in arrays])
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = (lower + upper) / 2.0
    span = max(float(np.max(upper - lower)) * 1.12, 0.10)
    half_span = span / 2.0
    axis.set_xlim(center[0] - half_span, center[0] + half_span)
    axis.set_ylim(center[1] - half_span, center[1] + half_span)
    axis.set_zlim(center[2] - half_span, center[2] + half_span)
    axis.set_box_aspect((1, 1, 1))


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
    for key, data in results.items():
        error = data["err"] if "err" in data else data["err_mm"] / 1e3
        solve_time = filtered_solve_times(data, preserve_length=True)
        time = np.arange(len(error)) * DT
        axes[0].plot(time[:len(error)], error * 1e3, label=LABELS.get(key, key), color=COLORS.get(key), lw=2)
        axes[1].plot(time[:len(solve_time)], solve_time * 1e3, label=LABELS.get(key, key), color=COLORS.get(key), lw=2)
    _style_axis(axes[0], "Tracking Error", "EE error [mm]")
    _style_axis(axes[1], "Controller Step Time", "step time [ms]")
    axes[1].axhline(DT * 1e3, color="#111827", ls="--", label="control period")
    axes[1].set_xlabel("time [s]")
    for axis in axes:
        axis.legend(frameon=False)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.93])
    return fig


def plot_3d_trajectory(results: dict, waypoints: np.ndarray, *,
                       dynamic_paths: np.ndarray | None = None,
                       static_centers: np.ndarray | None = None):
    has_static = static_centers is not None
    has_dynamic = dynamic_paths is not None
    has_assessment = has_static or has_dynamic
    fig = plt.figure(figsize=(11.2 if has_assessment else 7.4, 6.2),
                     facecolor=FIG_BG)
    if has_assessment:
        grid = fig.add_gridspec(1, 2, width_ratios=(3.2, 1.65), wspace=0.04)
        axis = fig.add_subplot(grid[0, 0], projection="3d")
        assessment_axis = fig.add_subplot(grid[0, 1])
        assessment_axis.axis("off")
    else:
        axis = fig.add_subplot(111, projection="3d")
        assessment_axis = None
    axis.set_title("Final 3D Trajectory", fontsize=13, fontweight="bold")
    axis.plot(*waypoints.T, "--", color="#111827", lw=1.6, label="Reference")
    for key, data in results.items():
        ee = data["ee"]
        axis.plot(*ee.T, label=LABELS.get(key, key), color=COLORS.get(key), lw=2)
    if dynamic_paths is not None:
        dynamic_path = np.asarray(dynamic_paths)[0]
        axis.plot(*dynamic_path.T, ":", lw=1.8, label="Dynamic obstacle")
        assessments = {
            key: assess_dynamic_obstacle(data["ee"], dynamic_path)
            for key, data in results.items()
        }
        rows, row_colors = [], []
        for key, assessment in assessments.items():
            contact = assessment["contact"]
            rows.append([
                "O1", LABELS.get(key, key),
                "CONTACT" if contact else "CLEAR",
                f"{assessment['clearance_mm']:+.2f} mm",
            ])
            row_colors.append([
                "#ffffff", "#ffffff",
                "#fecaca" if contact else "#bbf7d0",
                "#fecaca" if contact else "#bbf7d0",
            ])
        assessment_axis.set_title(
            "Dynamic obstacle contact check\n"
            f"EE point vs {DYNAMIC_OBSTACLE_SIZE * 1e3:g} mm cube",
            fontsize=11, fontweight="bold", pad=14,
        )
        table = assessment_axis.table(
            cellText=rows,
            colLabels=["ID", "Run", "Result", "Min clearance"],
            cellColours=row_colors,
            colColours=["#e5e7eb"] * 4,
            cellLoc="center", colLoc="center", loc="center",
            colWidths=[0.13, 0.29, 0.27, 0.31],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.45)
        assessment_axis.text(
            0.5, 0.12, "Negative clearance = penetration",
            transform=assessment_axis.transAxes, ha="center",
            fontsize=8.5, color=MUTED,
        )
    if static_centers is not None:
        half = STATIC_OBSTACLE_SIZE / 2.0
        for index, center in enumerate(static_centers):
            axis.bar3d(*(center - half), *([STATIC_OBSTACLE_SIZE] * 3),
                       color="#ff4400", alpha=0.65,
                       label="Static obstacles" if index == 0 else None)
            axis.text(center[0], center[1], center[2] + half * 1.5,
                      f"O{index + 1}", color="#9a3412", fontweight="bold")
        assessments = {
            key: assess_static_obstacles(data["ee"], static_centers)
            for key, data in results.items()
        }
        rows, row_colors = [], []
        for key in results:
            for obstacle_index in range(len(static_centers)):
                assessment = assessments[key][obstacle_index]
                contact = assessment["contact"]
                rows.append([
                    f"O{obstacle_index + 1}", LABELS.get(key, key),
                    "CONTACT" if contact else "CLEAR",
                    f"{assessment['clearance_mm']:+.2f} mm",
                ])
                row_colors.append([
                    "#ffffff", "#ffffff",
                    "#fecaca" if contact else "#bbf7d0",
                    "#fecaca" if contact else "#bbf7d0",
                ])
        assessment_axis.set_title(
            "Static obstacle contact check\n"
            "EE point vs 10 mm cube",
            fontsize=11, fontweight="bold", pad=14,
        )
        table = assessment_axis.table(
            cellText=rows,
            colLabels=["ID", "Run", "Result", "Min clearance"],
            cellColours=row_colors,
            colColours=["#e5e7eb"] * 4,
            cellLoc="center", colLoc="center", loc="center",
            colWidths=[0.13, 0.29, 0.27, 0.31],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8.5)
        table.scale(1.0, 1.45)
        assessment_axis.text(
            0.5, 0.12, "Negative clearance = penetration",
            transform=assessment_axis.transAxes, ha="center",
            fontsize=8.5, color=MUTED,
        )
    axis.set(xlabel="X [m]", ylabel="Y [m]", zlabel="Z [m]")
    scale_arrays = [waypoints] + [data["ee"] for data in results.values()]
    if dynamic_paths is not None:
        scale_arrays.append(dynamic_paths)
    if static_centers is not None:
        half = STATIC_OBSTACLE_SIZE / 2.0
        scale_arrays.extend((static_centers - half, static_centers + half))
    _set_equal_3d_scale(axis, scale_arrays)
    axis.view_init(elev=25, azim=-55)
    axis.legend(frameon=False)
    if not has_assessment:
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
