from __future__ import annotations

# Unit tests are collected from the tests package.

import math
import unittest

import torch

from config import DQ_MAX, Q_MAX, Q_MIN
from costs.tracking_cost import TrackingCost
from dynamics.ur5_dynamics import UR5Dynamics


class StateLimitTests(unittest.TestCase):
    def test_official_ur5_limits(self):
        expected_position = (
            2 * math.pi, 2 * math.pi, math.pi,
            2 * math.pi, 2 * math.pi, 2 * math.pi,
        )
        self.assertEqual(Q_MAX, expected_position)
        self.assertEqual(Q_MIN, tuple(-value for value in expected_position))
        self.assertEqual(DQ_MAX, math.pi)

    def test_soft_penalty_starts_at_configured_ratio(self):
        dtype = torch.float64
        cost = TrackingCost(
            torch.zeros((3, 3), dtype=dtype), 1,
            torch.full((6,), -1.0, dtype=dtype),
            torch.full((6,), 1.0, dtype=dtype),
            joint_position_limit_weight=10.0,
            joint_velocity_limit_weight=20.0,
            joint_limit_soft_ratio=0.9,
        )
        state = torch.zeros((1, 12), dtype=dtype)
        state[0, 0] = cost.q_soft_max[0]
        state[0, 7] = cost.dq_soft_max[1]
        self.assertEqual(cost.state_limit_cost(state).item(), 0.0)

        state[0, 0] = cost.q_max[0]
        state[0, 7] = cost.dq_max[1]
        self.assertAlmostEqual(cost.state_limit_cost(state).item(), 30.0)

    def test_state_transition_does_not_clamp_velocity(self):
        dynamics = UR5Dynamics(torch.device("cpu"), torch.float64)
        dynamics.forward_dynamics = lambda _state, torque: torque
        state = torch.zeros((1, 12), dtype=torch.float64)
        effective_torque = torch.tensor(
            [[400.0, 0, 0, 0, 0, 0]], dtype=torch.float64
        )
        next_state = dynamics.rollout_effective(state, effective_torque)
        self.assertAlmostEqual(next_state[0, 6].item(), 8.0)

    def test_actuator_torque_clamp_is_retained(self):
        dynamics = UR5Dynamics(torch.device("cpu"), torch.float64)
        dynamics.forward_dynamics = lambda _state, torque: torque
        state = torch.zeros((1, 12), dtype=torch.float64)
        command = torch.tensor([[400.0, 0, 0, 0, 0, 0]], dtype=torch.float64)
        rollout_state = dynamics.rollout(state, command)
        plant_state = dynamics.plant_step(state, command)
        self.assertAlmostEqual(rollout_state[0, 6].item(), 3.0)
        self.assertAlmostEqual(plant_state[0, 6].item(), 3.0)


if __name__ == "__main__":
    unittest.main()
