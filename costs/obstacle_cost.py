"""Static and time-indexed dynamic obstacle barrier costs."""
from __future__ import annotations

import numpy as np
import torch

from config import (CROSS_EXP_K, CROSS_EXP_W, CROSS_EXP_ZONE, CROSS_SAFE_RADIUS,
                    K_EXP, OBSTACLES, OBS_MARGIN, SOFT_ZONE, W_EXP)
from kinematics.ur5_kinematics import fk_batch

from .base_cost import BaseCost


class ObstacleCost(BaseCost):
    def __init__(self, horizon: int, device: torch.device, dtype: torch.dtype,
                 dynamic_paths: np.ndarray | None = None,
                 safe_radius: float = CROSS_SAFE_RADIUS,
                 exp_weight: float = CROSS_EXP_W,
                 exp_gain: float = CROSS_EXP_K,
                 exp_zone: float = CROSS_EXP_ZONE):
        self.horizon = horizon
        self.episode_step = 0
        self.safe_radius = safe_radius
        self.exp_weight, self.exp_gain, self.exp_zone = exp_weight, exp_gain, exp_zone
        self.dynamic_paths = None if dynamic_paths is None else torch.as_tensor(
            dynamic_paths, dtype=dtype, device=device
        )

    def evaluate_positions(self, ee: torch.Tensor, step: int) -> torch.Tensor:
        cost = torch.zeros(ee.shape[0], dtype=ee.dtype, device=ee.device)
        if self.dynamic_paths is not None:
            path_index = min(self.episode_step + step, self.dynamic_paths.shape[1] - 1)
            for center in self.dynamic_paths[:, path_index, :]:
                distance = torch.linalg.vector_norm(ee - center, dim=1).clamp_min(1e-8)
                penetration = torch.relu(self.safe_radius + self.exp_zone - distance)
                cost += self.exp_weight * torch.expm1(self.exp_gain * penetration)
            return cost
        for obstacle in OBSTACLES:
            center = ee.new_tensor(obstacle["center"])
            distance = torch.linalg.vector_norm(ee[:, :2] - center, dim=1).clamp_min(1e-8)
            outer_radius = obstacle["r_obs"] + OBS_MARGIN + SOFT_ZONE
            cost += W_EXP * torch.expm1(K_EXP * torch.relu(outer_radius - distance))
        return cost

    def running(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        return self.evaluate_positions(fk_batch(state[:, :6]), step)

    def terminal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        last_q = states[..., -1, :6].reshape(-1, 6)
        cost = 4.0 * self.evaluate_positions(fk_batch(last_q), self.horizon)
        return cost.reshape(states.shape[:2])
