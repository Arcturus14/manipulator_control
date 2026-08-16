"""Torch/Numpy forward kinematics, Jacobian, and numerical IK for UR5."""
from __future__ import annotations

import math

import numpy as np
import torch

from config import DH_PARAMS


def _dh_mat_batch(a: float, d: float, alpha: float, theta: torch.Tensor) -> torch.Tensor:
    count = theta.shape[0]
    zeros = torch.zeros(count, device=theta.device, dtype=theta.dtype)
    ones = torch.ones(count, device=theta.device, dtype=theta.dtype)
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    rows = (
        torch.stack([ct, -st * ca, st * sa, a * ct], dim=1),
        torch.stack([st, ct * ca, -ct * sa, a * st], dim=1),
        torch.stack([zeros, sa * ones, ca * ones, d * ones], dim=1),
        torch.stack([zeros, zeros, zeros, ones], dim=1),
    )
    return torch.stack(rows, dim=1)


@torch.compile(mode="reduce-overhead")
def fk_batch(q: torch.Tensor) -> torch.Tensor:
    """Return end-effector XYZ for a ``(K, 6)`` joint batch."""
    count = q.shape[0]
    transform = torch.eye(4, dtype=q.dtype, device=q.device).expand(count, -1, -1).clone()
    for index, (a, d, alpha, offset) in enumerate(DH_PARAMS):
        transform = torch.bmm(transform, _dh_mat_batch(a, d, alpha, q[:, index] + offset))
    return transform[:, :3, 3]


def fk_and_jacobian_batch(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return end-effector XYZ and translational Jacobian for a joint batch."""
    count = q.shape[0]
    transform = torch.eye(4, dtype=q.dtype, device=q.device).expand(count, -1, -1).clone()
    axes, origins = [], []
    for index, (a, d, alpha, offset) in enumerate(DH_PARAMS):
        axes.append(transform[:, :3, 2].clone())
        origins.append(transform[:, :3, 3].clone())
        transform = torch.bmm(transform, _dh_mat_batch(a, d, alpha, q[:, index] + offset))
    ee = transform[:, :3, 3]
    jacobian = torch.stack([torch.linalg.cross(axes[i], ee - origins[i]) for i in range(6)], dim=2)
    return ee, jacobian


def fk_joints_np(q: np.ndarray) -> np.ndarray:
    """Return base and all six joint positions as a ``(7, 3)`` array."""
    transform = np.eye(4)
    points = [np.zeros(3)]
    for index, (a, d, alpha, offset) in enumerate(DH_PARAMS):
        theta = q[index] + offset
        ct, st = math.cos(theta), math.sin(theta)
        ca, sa = math.cos(alpha), math.sin(alpha)
        local = np.array([
            [ct, -st * ca, st * sa, a * ct],
            [st, ct * ca, -ct * sa, a * st],
            [0.0, sa, ca, d],
            [0.0, 0.0, 0.0, 1.0],
        ])
        transform = transform @ local
        points.append(transform[:3, 3].copy())
    return np.asarray(points)


def ik_np(target: np.ndarray, q0: np.ndarray, n_iter: int = 300,
          tol: float = 1e-5, damping: float = 0.01, step_size: float = 0.5) -> np.ndarray:
    """Solve position-only IK with a damped least-squares Jacobian."""
    q = np.asarray(q0, dtype=np.float64).copy()
    target_t = torch.as_tensor(target, dtype=torch.float64).unsqueeze(0)
    for _ in range(n_iter):
        q_t = torch.as_tensor(q, dtype=torch.float64).unsqueeze(0)
        ee, jacobian = fk_and_jacobian_batch(q_t)
        error = target_t[0] - ee[0]
        if torch.linalg.vector_norm(error).item() < tol:
            break
        j = jacobian[0]
        delta = j.T @ torch.linalg.solve(j @ j.T + damping * torch.eye(3, dtype=j.dtype), error)
        q = np.clip(q + step_size * delta.numpy(), -math.pi, math.pi)
    return q


def ik_trajectory_np(targets: np.ndarray, q0: np.ndarray,
                     n_iter_per_step: int = 40) -> np.ndarray:
    """Solve a continuous position-IK sequence using the previous solution."""
    solutions = np.empty((len(targets), 6), dtype=np.float64)
    current = np.asarray(q0, dtype=np.float64).copy()
    for index, target in enumerate(targets):
        solved = ik_np(target, current, n_iter=n_iter_per_step,
                       tol=2e-6, damping=0.005, step_size=0.7)
        # Select the closest 2*pi-equivalent representation for continuity.
        delta = (solved - current + math.pi) % (2 * math.pi) - math.pi
        current = current + delta
        solutions[index] = current
    return solutions
