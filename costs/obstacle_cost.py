"""Static and time-indexed dynamic obstacle barrier costs."""
from __future__ import annotations

import numpy as np
import torch

from config import (CROSS_EXP_K, CROSS_EXP_W, CROSS_EXP_ZONE, CROSS_MARGIN,
                    DYNAMIC_OBSTACLE_SIZE,
                    MPPI_STATIC_EXP_GAIN, MPPI_STATIC_EXP_WEIGHT,
                    MPPI_STATIC_MARGIN, MPPI_STATIC_ZONE,
                    STATIC_OBSTACLE_SIZE)
from kinematics.ur5_kinematics import fk_batch

from .base_cost import BaseCost


class ObstacleCost(BaseCost):
    def __init__(self, horizon: int, device: torch.device, dtype: torch.dtype,
                 dynamic_paths: np.ndarray | None = None,
                 static_centers: np.ndarray | None = None,
                 static_size: float = STATIC_OBSTACLE_SIZE,
                 static_margin: float = MPPI_STATIC_MARGIN,
                 static_zone: float = MPPI_STATIC_ZONE,
                 static_exp_weight: float = MPPI_STATIC_EXP_WEIGHT,
                 static_exp_gain: float = MPPI_STATIC_EXP_GAIN,
                 dynamic_size: float = DYNAMIC_OBSTACLE_SIZE,
                 dynamic_margin: float = CROSS_MARGIN,
                 exp_weight: float = CROSS_EXP_W,
                 exp_gain: float = CROSS_EXP_K,
                 exp_zone: float = CROSS_EXP_ZONE):
        self.horizon = horizon
        self.episode_step = 0
        self.dynamic_half_size = dynamic_size / 2.0
        self.dynamic_margin = dynamic_margin
        self.exp_weight, self.exp_gain, self.exp_zone = exp_weight, exp_gain, exp_zone
        if dynamic_paths is not None and static_centers is not None:
            raise ValueError("dynamic and static obstacles are mutually exclusive")
        self.dynamic_paths = None if dynamic_paths is None else torch.as_tensor(
            dynamic_paths, dtype=dtype, device=device
        )
        self.static_centers = None if static_centers is None else torch.as_tensor(
            static_centers, dtype=dtype, device=device
        )
        self.static_half_size = static_size / 2.0
        self.static_margin = static_margin
        self.static_zone = static_zone
        self.static_exp_weight = static_exp_weight
        self.static_exp_gain = static_exp_gain

    def evaluate_positions(self, ee: torch.Tensor, step: int) -> torch.Tensor:
        cost = torch.zeros(ee.shape[0], dtype=ee.dtype, device=ee.device)
        if self.dynamic_paths is not None:
            path_index = min(self.episode_step + step, self.dynamic_paths.shape[1] - 1)
            centers = self.dynamic_paths[:, path_index, :]
            outside = torch.relu(
                torch.abs(ee[:, None, :] - centers[None, :, :])
                - self.dynamic_half_size
            )
            distance = torch.linalg.vector_norm(outside, dim=2)
            penetration = torch.relu(
                self.dynamic_margin + self.exp_zone - distance
            )
            return (
                self.exp_weight * torch.expm1(self.exp_gain * penetration)
            ).sum(dim=1)
        if self.static_centers is None:
            return cost
        # Evaluate every cube in one batched GPU operation. Distance is zero
        # on/inside each cube and grows normally away from its faces.
        outside = torch.relu(
            torch.abs(ee[:, None, :] - self.static_centers[None, :, :])
            - self.static_half_size
        )
        distance = torch.linalg.vector_norm(outside, dim=2)
        penetration = torch.relu(
            self.static_margin + self.static_zone - distance
        )
        return (
            self.static_exp_weight
            * torch.expm1(self.static_exp_gain * penetration)
        ).sum(dim=1)

    def running(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        return self.evaluate_positions(fk_batch(state[:, :6]), step)

    def terminal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        last_q = states[..., -1, :6].reshape(-1, 6)
        cost = 4.0 * self.evaluate_positions(fk_batch(last_q), self.horizon)
        return cost.reshape(states.shape[:2])
