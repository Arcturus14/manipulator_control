"""Interactive side-by-side UR5 arm animation engine."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from config import (CROSS_MARGIN, DT, DYNAMIC_OBSTACLE_SIZE,
                    STATIC_OBSTACLE_SIZE)
from kinematics.ur5_kinematics import fk_joints_np
from .plotters import COLORS, LABELS

LINK_COLORS = ["#003f8a", "#1560bd", "#2878d8", "#4a94e8", "#70b0f0", "#98ccff"]
OBSTACLE_COLORS = ["#ff4400", "#0044ff"]
SAMPLE_COLOR = "#6baed6"
ELITE_SAMPLE_COLOR = "#075b9a"
SAMPLE_MEAN_COLOR = "#f28e2b"
LOCAL_REFERENCE_COLOR = "#19a55a"


def animate_comparison(results: dict, waypoints: np.ndarray,
                       moving_obstacle_paths: np.ndarray | None = None,
                       moving_obstacle_size: float = DYNAMIC_OBSTACLE_SIZE,
                       static_obstacle_centers: np.ndarray | None = None):
    """Create an interactive arm animation and return ``(figure, animation)``."""
    from matplotlib import animation

    keys = list(results)
    fig = plt.figure(figsize=(9 * len(keys), 8), facecolor="white")
    axes = [fig.add_subplot(1, len(keys), i + 1, projection="3d") for i in range(len(keys))]
    arms, joints, ee_dots, traces, labels = [], [], [], [], []
    sample_lines, sample_means, local_references = [], [], []
    for axis, key in zip(axes, keys):
        axis.set(xlim=(0.18, 0.62), ylim=(-0.22, 0.22), zlim=(0.13, 0.57),
                 xlabel="X", ylabel="Y", zlabel="Z")
        axis.set_box_aspect((1, 1, 1))
        axis.set_title(LABELS.get(key, key), color=COLORS.get(key), fontweight="bold")
        axis.plot(*waypoints.T, "--", color="#008844", alpha=0.45)
        if static_obstacle_centers is not None:
            half = STATIC_OBSTACLE_SIZE / 2.0
            for center in static_obstacle_centers:
                axis.bar3d(*(center - half), *([STATIC_OBSTACLE_SIZE] * 3),
                           color="#ff4400", alpha=0.7)
        arms.append([axis.plot([], [], [], color=LINK_COLORS[j], lw=4)[0] for j in range(6)])
        joints.append(axis.plot([], [], [], "o", color="#cc2200", ms=5)[0])
        ee_dots.append(axis.plot([], [], [], "D", color="#e06000", ms=7)[0])
        traces.append(axis.plot([], [], [], color=COLORS.get(key), lw=1.5, alpha=0.8)[0])
        labels.append(axis.text2D(0.03, 0.95, "", transform=axis.transAxes))
        samples = results[key].get("_animation_samples")
        if samples:
            count = len(samples[0]["ee"])
            lines = []
            for rank in range(count):
                elite = rank < 5
                lines.append(axis.plot(
                    [], [], [],
                    color=ELITE_SAMPLE_COLOR if elite else SAMPLE_COLOR,
                    lw=1.3 if elite else 0.65,
                    alpha=0.7 if elite else 0.15,
                    label="sampled rollouts" if rank == 0 else None,
                )[0])
            sample_lines.append(lines)
            sample_means.append(axis.plot(
                [], [], [], color=SAMPLE_MEAN_COLOR, lw=3.0,
                label="weighted rollout mean",
            )[0])
            local_references.append(axis.plot(
                [], [], [], "--", color=LOCAL_REFERENCE_COLOR, lw=2.5,
                label="current horizon reference",
            )[0])
            axis.legend(loc="lower left", fontsize=8, framealpha=0.85)
        else:
            sample_lines.append([])
            sample_means.append(None)
            local_references.append(None)
        axis.view_init(elev=25, azim=-55)
    obstacle_artists = [[] for _ in axes]

    def update(frame):
        for index, (axis, key) in enumerate(zip(axes, keys)):
            q_history = results[key]["q"]
            frame_index = min(frame, len(q_history) - 1)
            positions = fk_joints_np(q_history[frame_index])
            for joint_index, segment in enumerate(arms[index]):
                points = positions[joint_index:joint_index + 2]
                segment.set_data(points[:, 0], points[:, 1]); segment.set_3d_properties(points[:, 2])
            joints[index].set_data(positions[:-1, 0], positions[:-1, 1]); joints[index].set_3d_properties(positions[:-1, 2])
            ee_dots[index].set_data([positions[-1, 0]], [positions[-1, 1]]); ee_dots[index].set_3d_properties([positions[-1, 2]])
            ee = results[key]["ee"][:frame_index + 1]
            traces[index].set_data(ee[:, 0], ee[:, 1]); traces[index].set_3d_properties(ee[:, 2])
            error = (results[key]["err_mm"] if "err_mm" in results[key]
                     else results[key]["err"] * 1e3)
            sample_history = results[key].get("_animation_samples")
            sample_text = ""
            if sample_history is not None and frame_index < len(sample_history):
                sample = sample_history[frame_index]
                current_ee = results[key]["ee"][frame_index]
                trajectories = np.asarray(sample["ee"])
                weights = np.asarray(sample["omega"])
                peak_weight = max(float(weights.max()), np.finfo(float).eps)
                for rank, line in enumerate(sample_lines[index]):
                    if rank >= len(trajectories):
                        line.set_data([], []); line.set_3d_properties([])
                        continue
                    points = np.concatenate((current_ee[None], trajectories[rank]), axis=0)
                    line.set_data(points[:, 0], points[:, 1])
                    line.set_3d_properties(points[:, 2])
                    relative_weight = float(weights[rank]) / peak_weight
                    if rank < 5:
                        line.set_alpha(0.55 + 0.4 * relative_weight)
                        line.set_linewidth(1.2 + 0.8 * relative_weight)
                    else:
                        line.set_alpha(0.06 + 0.26 * relative_weight)
                        line.set_linewidth(0.55 + 0.55 * relative_weight)

                mean_points = np.concatenate(
                    (current_ee[None], np.asarray(sample["ee_mean"])), axis=0
                )
                sample_means[index].set_data(mean_points[:, 0], mean_points[:, 1])
                sample_means[index].set_3d_properties(mean_points[:, 2])
                horizon = trajectories.shape[1]
                reference_end = min(frame_index + horizon + 1, len(waypoints))
                local_reference = waypoints[frame_index:reference_end]
                local_references[index].set_data(
                    local_reference[:, 0], local_reference[:, 1]
                )
                local_references[index].set_3d_properties(local_reference[:, 2])
                sample_text = f"  samples={len(trajectories)}  horizon={horizon}"
            elif sample_history is not None:
                for line in sample_lines[index]:
                    line.set_data([], []); line.set_3d_properties([])
                sample_means[index].set_data([], []); sample_means[index].set_3d_properties([])
                local_references[index].set_data([], []); local_references[index].set_3d_properties([])
            labels[index].set_text(
                f"step={frame_index}  err={error[max(frame_index-1, 0)]:.1f} mm"
                f"{sample_text}"
            )
            for artist in obstacle_artists[index]:
                artist.remove()
            obstacle_artists[index].clear()
            if moving_obstacle_paths is not None:
                obstacle_frame = min(frame, moving_obstacle_paths.shape[1] - 1)
                for obstacle_index, path in enumerate(moving_obstacle_paths):
                    center = path[obstacle_frame]
                    color = OBSTACLE_COLORS[obstacle_index % len(OBSTACLE_COLORS)]
                    half = moving_obstacle_size / 2.0
                    obstacle_artists[index].append(axis.bar3d(
                        *(center - half), *([moving_obstacle_size] * 3),
                        color=color, alpha=0.75,
                    ))
                    safe_size = moving_obstacle_size + 2.0 * CROSS_MARGIN
                    obstacle_artists[index].append(axis.bar3d(
                        *(center - safe_size / 2.0), *([safe_size] * 3),
                        color=color, alpha=0.06,
                    ))
        return []

    frame_count = max(len(data["q"]) for data in results.values())
    anim = animation.FuncAnimation(fig, update, frames=frame_count,
                                   interval=int(DT * 1000), blit=False, repeat=True)
    title = "UR5 Controller Simulation" if len(keys) == 1 else "UR5 Controller Comparison"
    fig.text(0.5, 0.98, title, ha="center", fontweight="bold")
    return fig, anim
