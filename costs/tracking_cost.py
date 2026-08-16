"""Reference tracking, control-effort, and input-smoothness costs."""
from __future__ import annotations

import torch

from config import MPPI_W_POS, MPPI_W_TERM, W_ACT, W_SMOOTH, W_VEL
from kinematics.ur5_kinematics import fk_batch

from .base_cost import BaseCost


class TrackingCost(BaseCost):
    def __init__(self, waypoints: torch.Tensor, horizon: int,
                 u_min: torch.Tensor, u_max: torch.Tensor,
                 smooth_weight: float = W_SMOOTH):
        self.waypoints = waypoints
        self.horizon = horizon
        self.u_min, self.u_max = u_min, u_max
        self.smooth_weight = smooth_weight
        self.episode_step = 0
        self.previous_action = torch.zeros_like(u_max)

    def running(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        torque = action.clamp(self.u_min, self.u_max)
        ee = fk_batch(state[:, :6])
        reference = self.waypoints[self.episode_step + step]
        cost = MPPI_W_POS * (ee - reference).square().sum(dim=1)
        cost += W_ACT * torque.square().sum(dim=1)
        cost += W_VEL * state[:, 6:].square().sum(dim=1)
        if step == 0:
            cost += self.smooth_weight * (torque - self.previous_action).square().sum(dim=1)
        return cost

    def terminal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        last_q = states[..., -1, :6].reshape(-1, 6)
        reference = self.waypoints[self.episode_step + self.horizon]
        cost = MPPI_W_TERM * (fk_batch(last_q) - reference).square().sum(dim=1)
        return cost.reshape(states.shape[:2])
