"""Interactive side-by-side UR5 arm animation engine."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from config import CROSS_MARGIN, CROSS_RADIUS, DT, N_STEPS
from kinematics.ur5_kinematics import fk_joints_np
from .plotters import COLORS, LABELS

LINK_COLORS = ["#003f8a", "#1560bd", "#2878d8", "#4a94e8", "#70b0f0", "#98ccff"]
OBSTACLE_COLORS = ["#ff4400", "#0044ff"]


def _sphere_wireframe(axis, center, radius, color):
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    return axis.plot_wireframe(x, y, z, color=color, linewidth=0.5, alpha=0.55)


def animate_comparison(results: dict, waypoints: np.ndarray,
                       moving_obstacle_paths: np.ndarray | None = None,
                       moving_obstacle_radius: float = CROSS_RADIUS):
    """Create an interactive arm animation and return ``(figure, animation)``."""
    from matplotlib import animation

    keys = list(results)
    fig = plt.figure(figsize=(9 * len(keys), 8), facecolor="white")
    axes = [fig.add_subplot(1, len(keys), i + 1, projection="3d") for i in range(len(keys))]
    arms, joints, ee_dots, traces, labels = [], [], [], [], []
    for axis, key in zip(axes, keys):
        axis.set(xlim=(0.18, 0.62), ylim=(-0.22, 0.22), zlim=(0.13, 0.57),
                 xlabel="X", ylabel="Y", zlabel="Z")
        axis.set_title(LABELS.get(key, key), color=COLORS.get(key), fontweight="bold")
        axis.plot(*waypoints.T, "--", color="#008844", alpha=0.45)
        arms.append([axis.plot([], [], [], color=LINK_COLORS[j], lw=4)[0] for j in range(6)])
        joints.append(axis.plot([], [], [], "o", color="#cc2200", ms=5)[0])
        ee_dots.append(axis.plot([], [], [], "D", color="#e06000", ms=7)[0])
        traces.append(axis.plot([], [], [], color=COLORS.get(key), lw=1.5, alpha=0.8)[0])
        labels.append(axis.text2D(0.03, 0.95, "", transform=axis.transAxes))
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
            error = results[key].get("err_mm", results[key].get("err") * 1e3)
            labels[index].set_text(f"step={frame_index}  err={error[max(frame_index-1, 0)]:.1f} mm")
            for artist in obstacle_artists[index]:
                artist.remove()
            obstacle_artists[index].clear()
            if moving_obstacle_paths is not None:
                obstacle_frame = min(frame, moving_obstacle_paths.shape[1] - 1)
                for obstacle_index, path in enumerate(moving_obstacle_paths):
                    obstacle_artists[index].append(_sphere_wireframe(
                        axis, path[obstacle_frame], moving_obstacle_radius,
                        OBSTACLE_COLORS[obstacle_index % len(OBSTACLE_COLORS)]))
                    center = path[obstacle_frame]
                    theta = np.linspace(0, 2*np.pi, 50)
                    obstacle_artists[index].append(axis.plot(
                        center[0] + (moving_obstacle_radius + CROSS_MARGIN) * np.cos(theta),
                        center[1] + (moving_obstacle_radius + CROSS_MARGIN) * np.sin(theta),
                        np.full_like(theta, center[2]), "--", alpha=0.25)[0])
        return []

    anim = animation.FuncAnimation(fig, update, frames=N_STEPS + 1,
                                   interval=int(DT * 1000), blit=False, repeat=True)
    title = "UR5 Controller Simulation" if len(keys) == 1 else "UR5 Controller Comparison"
    fig.text(0.5, 0.98, title, ha="center", fontweight="bold")
    return fig, anim
