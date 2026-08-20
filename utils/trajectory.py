"""Reference trajectory generators."""
from __future__ import annotations

import math

import numpy as np

from config import (DYNAMIC_OBSTACLE_PHASE_OFFSET,
                    DYNAMIC_OBSTACLE_SPEED_RATIO, STATIC_OBSTACLE_COUNT)


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


def complex_3d_wave(n: int, c: np.ndarray, r: float) -> np.ndarray:
    """3엽 반경 변조와 수직 파동을 결합한 3차원 기준 궤적을 생성한다.

    - X, Y: 3엽 장미 곡선과 주파수 변조가 결합된 입체 궤적
    - Z: 높낮이가 위아래로 유연하게 파동 치는 3D 이동
    """
    ts = np.linspace(0, 2 * math.pi, n, endpoint=False)
    points = np.tile(c.copy(), (n, 1))
    
    # X, Y 평면: 3엽 장미 변형 곡선 (r_scale이 주파수에 따라 변함)
    r_scale = r * (0.8 + 0.3 * np.cos(3 * ts))
    points[:, 0] += r_scale * np.cos(ts)
    points[:, 1] += r_scale * np.sin(ts)
    
    # Z 축: 높이 $z_c$ 중심으로 위아래로 $\pm 8\text{cm}$ 출렁임
    points[:, 2] += 0.08 * np.sin(2 * ts)
    
    return points


def multiscale_3d(n: int, center: np.ndarray, radius: float) -> np.ndarray:
    """Generate a smooth 3D path with several competing spatial frequencies.

    The low-frequency terms create the overall loop, while the higher
    harmonics add alternating tight turns, direction reversals, and vertical
    undulation.  The resulting path exercises nonlinear rollout optimization
    without introducing discontinuous, physically unreasonable waypoint
    jumps.
    """
    phase = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    points = np.tile(np.asarray(center, dtype=float), (n, 1))

    points[:, 0] += radius * (
        0.72 * np.cos(phase) + 0.20 * np.cos(5.0 * phase)
    )
    points[:, 1] += radius * (
        0.62 * np.sin(2.0 * phase) - 0.18 * np.sin(6.0 * phase)
    )
    points[:, 2] += radius * (
        0.38 * np.sin(3.0 * phase) + 0.14 * np.sin(7.0 * phase)
    )
    return points


TRAJECTORIES = {
    "circle": circle,
    "infinity": infinity,
    "rectangle": rectangle,
    "complex_3d_wave": complex_3d_wave,
    "multiscale_3d": multiscale_3d,
}


def chasing_obstacle_paths(
        reference: np.ndarray, extra_steps: int = 0, *,
        speed_ratio: float = DYNAMIC_OBSTACLE_SPEED_RATIO,
        phase_offset: float = DYNAMIC_OBSTACLE_PHASE_OFFSET) -> np.ndarray:
    """Move one obstacle cyclically along ``reference`` faster than the robot.

    The robot advances one reference sample per control step.  The obstacle
    starts half a cycle away by default and advances ``speed_ratio`` samples
    per step in the same direction, so it periodically catches the robot from
    behind. Fractional indices are linearly interpolated on the closed path.
    """
    reference = np.asarray(reference, dtype=float)
    if reference.ndim != 2 or reference.shape[1] != 3 or len(reference) < 2:
        raise ValueError("reference must have shape (N, 3) with N >= 2")
    if extra_steps < 0:
        raise ValueError("extra_steps must be non-negative")
    if speed_ratio <= 1.0:
        raise ValueError("speed_ratio must be greater than 1 for chasing")
    if not 0.0 <= phase_offset < 1.0:
        raise ValueError("phase_offset must be in [0, 1)")

    count = len(reference)
    steps = np.arange(count + extra_steps, dtype=float)
    indices = (phase_offset * count + speed_ratio * steps) % count
    lower = np.floor(indices).astype(int)
    fraction = indices - lower
    upper = (lower + 1) % count
    path = ((1.0 - fraction[:, None]) * reference[lower]
            + fraction[:, None] * reference[upper])
    return path[None, ...].astype(np.float32)


def static_obstacle_centers(reference: np.ndarray,
                            count: int = STATIC_OBSTACLE_COUNT) -> np.ndarray:
    """Uniformly distribute ``count`` obstacles without using the start point.

    The closed reference is split into ``count + 1`` equal progress gaps.  The
    initial boundary is intentionally omitted, leaving the same-sized clear
    gap on both sides of the trajectory start.
    """
    if count <= 0:
        raise ValueError("static obstacle count must be positive")
    reference = np.asarray(reference)
    if count >= len(reference):
        raise ValueError("static obstacle count must be smaller than reference length")
    indices = np.floor(
        np.arange(1, count + 1, dtype=float) * len(reference) / (count + 1)
    ).astype(int)
    return reference[indices].copy()
