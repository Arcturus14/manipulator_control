"""Reference trajectory generators."""
from __future__ import annotations

import math

import numpy as np


def circle(n: int, center: np.ndarray, radius: float) -> np.ndarray:
    phase = np.linspace(0, 2 * math.pi, n, endpoint=False)
    points = np.tile(np.asarray(center), (n, 1)).astype(float)
    points[:, 0] += radius * np.cos(phase)
    points[:, 1] += radius * np.sin(phase)
    return points


def infinity(n: int, center: np.ndarray, radius: float) -> np.ndarray:
    phase = np.linspace(0, 2 * math.pi, n, endpoint=False)
    denominator = 1 + np.sin(phase) ** 2
    points = np.tile(np.asarray(center), (n, 1)).astype(float)
    points[:, 0] += radius * np.cos(phase) / denominator
    points[:, 1] += radius * np.sin(phase) * np.cos(phase) / denominator
    return points


def rectangle(n: int, center: np.ndarray, radius: float) -> np.ndarray:
    cx, cy, cz = center
    width, height = radius * 1.6, radius
    corners = np.array([[cx-width, cy-height, cz], [cx+width, cy-height, cz],
                        [cx+width, cy+height, cz], [cx-width, cy+height, cz]])
    counts = np.full(4, n // 4)
    counts[:n % 4] += 1
    return np.vstack([np.linspace(corners[i], corners[(i + 1) % 4], counts[i], endpoint=False)
                      for i in range(4)])


TRAJECTORIES = {"circle": circle, "infinity": infinity, "rectangle": rectangle}


def crossing_obstacle_paths(n_steps: int, center: np.ndarray) -> np.ndarray:
    """Generate two spheres crossing in opposite Y directions."""
    phase = np.linspace(0.0, 1.0, n_steps)
    first = np.stack([np.full(n_steps, center[0]), -0.20 + 0.40 * phase,
                      np.full(n_steps, center[2])], axis=1)
    second = np.stack([np.full(n_steps, center[0]), 0.20 - 0.40 * phase,
                       np.full(n_steps, center[2])], axis=1)
    return np.stack([first, second]).astype(np.float32)
