"""Reference tracking, control-effort, and input-smoothness costs."""
from __future__ import annotations

import torch

from config import (DQ_MAX, JOINT_LIMIT_SOFT_RATIO, MPPI_W_DQ_LIMIT,
                    MPPI_W_POS, MPPI_W_Q_LIMIT, MPPI_W_TERM, Q_MAX, Q_MIN,
                    W_ACT, W_SMOOTH, W_VEL)
from kinematics.ur5_kinematics import fk_batch

from .base_cost import BaseCost


class TrackingCost(BaseCost):
    def __init__(self, waypoints: torch.Tensor, horizon: int,
                 u_min: torch.Tensor, u_max: torch.Tensor,
                 smooth_weight: float = W_SMOOTH,
                 position_weight: float = MPPI_W_POS,
                 terminal_weight: float = MPPI_W_TERM,
                 joint_position_limit_weight: float = MPPI_W_Q_LIMIT,
                 joint_velocity_limit_weight: float = MPPI_W_DQ_LIMIT,
                 joint_limit_soft_ratio: float = JOINT_LIMIT_SOFT_RATIO):
        if not 0.0 < joint_limit_soft_ratio < 1.0:
            raise ValueError("joint_limit_soft_ratio must be in (0, 1)")
        if joint_position_limit_weight < 0 or joint_velocity_limit_weight < 0:
            raise ValueError("joint-limit cost weights must be nonnegative")
        self.waypoints = waypoints
        self.horizon = horizon
        self.u_min, self.u_max = u_min, u_max
        self.smooth_weight = smooth_weight
        self.position_weight = position_weight
        self.terminal_weight = terminal_weight
        self.joint_position_limit_weight = joint_position_limit_weight
        self.joint_velocity_limit_weight = joint_velocity_limit_weight
        self.joint_limit_soft_ratio = joint_limit_soft_ratio
        self.episode_step = 0
        # Optional fixed-address horizon buffer for complete-rollout compile.
        # The normal path remains unchanged unless explicitly enabled.
        self.reference_window = None
        self.previous_action = torch.zeros_like(u_max)
        self.q_min = torch.as_tensor(Q_MIN, dtype=u_max.dtype, device=u_max.device)
        self.q_max = torch.as_tensor(Q_MAX, dtype=u_max.dtype, device=u_max.device)
        self.dq_max = torch.full_like(u_max, DQ_MAX)
        self.q_soft_min = self.q_min * joint_limit_soft_ratio
        self.q_soft_max = self.q_max * joint_limit_soft_ratio
        self.dq_soft_max = self.dq_max * joint_limit_soft_ratio
        self.q_lower_margin = self.q_soft_min - self.q_min
        self.q_upper_margin = self.q_max - self.q_soft_max
        self.dq_margin = self.dq_max - self.dq_soft_max

    def state_limit_cost(self, state: torch.Tensor) -> torch.Tensor:
        """Normalized soft penalty near the official UR5 state limits."""
        q, velocity = state[..., :6], state[..., 6:]
        q_low = torch.relu(self.q_soft_min - q) / self.q_lower_margin
        q_high = torch.relu(q - self.q_soft_max) / self.q_upper_margin
        dq_high = torch.relu(velocity.abs() - self.dq_soft_max) / self.dq_margin
        return (
            self.joint_position_limit_weight
            * (q_low.square() + q_high.square()).sum(dim=-1)
            + self.joint_velocity_limit_weight * dq_high.square().sum(dim=-1)
        )

    def running(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        torque = action.clamp(self.u_min, self.u_max)
        ee = fk_batch(state[:, :6])
        reference = (self.reference_window[step]
                     if self.reference_window is not None
                     else self.waypoints[self.episode_step + step])
        cost = self.position_weight * (ee - reference).square().sum(dim=1)
        cost += W_ACT * torque.square().sum(dim=1)
        cost += W_VEL * state[:, 6:].square().sum(dim=1)
        cost += self.state_limit_cost(state)
        if step == 0:
            cost += self.smooth_weight * (torque - self.previous_action).square().sum(dim=1)
        return cost

    def terminal(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        last_q = states[..., -1, :6].reshape(-1, 6)
        reference = (self.reference_window[self.horizon]
                     if self.reference_window is not None
                     else self.waypoints[self.episode_step + self.horizon])
        cost = self.terminal_weight * (fk_batch(last_q) - reference).square().sum(dim=1)
        last_state = states[..., -1, :].reshape(-1, 12)
        cost += self.state_limit_cost(last_state)
        cost = cost.reshape(states.shape[:2])
        if self.horizon > 1:
            # The first-stage boundary term is evaluated in ``running``.
            # Penalize every internal torque change as well, normalized by the
            # number of transitions so the weight is independent of horizon.
            torque = actions.clamp(self.u_min, self.u_max)
            delta = torque[..., 1:, :] - torque[..., :-1, :]
            cost += self.smooth_weight * delta.square().sum(dim=(-1, -2)) / (
                self.horizon - 1
            )
        return cost
