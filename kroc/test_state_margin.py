"""Analytical checks for direction, projection, filtering, and raw states."""
import unittest

import torch

from costs.tracking_cost import TrackingCost
from dynamics.ur5_dynamics import UR5Dynamics
from kroc.state_margin import MarginConfig, StateMargins


class MarginTests(unittest.TestCase):
    def make(self, **kwargs):
        dyn = UR5Dynamics(torch.device("cpu"), torch.float64)
        cost = TrackingCost(torch.zeros(20, 3, dtype=torch.float64), 10,
                            dyn.u_min, dyn.u_max)
        dyn.mass_matrix_and_bias = lambda q, dq: (torch.eye(6, dtype=q.dtype), torch.zeros_like(q))
        return StateMargins(dyn, cost, MarginConfig(**kwargs))

    def test_direction_and_horizon_headroom_no_double_count(self):
        module = self.make(mode="headroom")
        state = torch.zeros(12, dtype=torch.float64)
        center = torch.zeros(10, 6, dtype=torch.float64)
        center[-1, 0] = -145
        d = torch.tensor([-1., 0, 0, 0, 0, 0], dtype=torch.float64)
        row = module.update(state, center, d)
        self.assertEqual(row["headroom_negative"][0].item(), 5)
        self.assertEqual(row["headroom_positive"][0].item(), 150)
        self.assertGreater(row["gain_upper"][0].item(), 1)
        self.assertEqual(row["gain_lower"][0].item(), 1)
        self.assertGreater(module.margin[1, 1, 0], module.margin[1, 0, 0])
        self.assertEqual(row["a_d_hat"][0].item(), 1)

    def test_asymmetric_velocity_penalty(self):
        module = self.make(mode="directional")
        state = torch.zeros(12, dtype=torch.float64)
        module.update(state, torch.zeros(10, 6, dtype=torch.float64),
                      torch.tensor([-10., 0, 0, 0, 0, 0], dtype=torch.float64))
        positive = state.clone()
        positive[6] = 2.7
        negative = positive.clone()
        negative[6] *= -1
        self.assertGreater(module.cost_value(positive), module.cost_value(negative))

    def test_clip_interval_and_slow_release(self):
        module = self.make(mode="headroom", max_fraction=.4,
                           min_internal_fraction=.8)
        state = torch.full((12,), 100., dtype=torch.float64)
        center = torch.zeros(10, 6, dtype=torch.float64)
        d = torch.full((6,), -10000., dtype=torch.float64)
        row = module.update(state, center, d)
        self.assertTrue(torch.all(row["d_hat_margin"].abs() <= module.dynamics.u_max))
        self.assertTrue(torch.all(module.margin.sum(1) <= module.span * .2 + 1e-12))
        previous = module.margin.clone()
        module.update(torch.zeros_like(state), center, torch.zeros_like(d))
        floor = module.span[:, None, :] * module.config.min_fraction
        torch.testing.assert_close(module.margin, previous + .1 * (floor - previous))

    def test_fixed_matches_legacy_at_five_percent_full_span(self):
        module = self.make(mode="fixed")
        state = torch.randn(200, 12, dtype=torch.float64) * 5
        torch.testing.assert_close(module.cost_value(state), module.cost.state_limit_cost(state))

    def test_plant_and_rollout_have_unclamped_raw_velocity(self):
        dyn = UR5Dynamics(torch.device("cpu"), torch.float64)
        dyn.forward_dynamics = lambda state, torque: torque
        state = torch.zeros(1, 12, dtype=torch.float64)
        torque = torch.full((1, 6), 400., dtype=torch.float64)
        for transition in (dyn.rollout_effective, dyn.plant_step_effective):
            result = transition(state, torque)
            torch.testing.assert_close(result[:, 6:], torch.full((1, 6), 8., dtype=torch.float64))

    def test_reset_restores_initial_margin(self):
        module = self.make(mode="headroom")
        initial = module.margin.clone()
        state = torch.ones(12, dtype=torch.float64) * 20
        module.update(state, torch.zeros(10, 6, dtype=torch.float64),
                      torch.ones(6, dtype=torch.float64) * 100)
        self.assertFalse(torch.equal(module.margin, initial))
        module.reset()
        torch.testing.assert_close(module.margin, initial)

    def test_acceleration_deadband_is_soft_thresholded(self):
        module = self.make(
            mode="directional",
            acceleration_deadband_rad_s2=(0.5, 1.0, 0, 0, 0, 0),
        )
        state = torch.zeros(12, dtype=torch.float64)
        row = module.update(
            state, torch.zeros(10, 6, dtype=torch.float64),
            torch.tensor([-0.4, -1.5, 0., 0., 0., 0.], dtype=torch.float64),
        )
        torch.testing.assert_close(
            row["a_d_hat_raw"][:2], torch.tensor([0.4, 1.5], dtype=torch.float64)
        )
        torch.testing.assert_close(
            row["a_d_hat"][:2], torch.tensor([0.0, 0.5], dtype=torch.float64)
        )
        self.assertEqual(row["acceleration_deadband"].shape, (6,))

    def test_acceleration_deadband_validation(self):
        with self.assertRaises(ValueError):
            self.make(mode="directional", acceleration_deadband_rad_s2=(1.0,))
        with self.assertRaises(ValueError):
            self.make(mode="directional",
                      acceleration_deadband_rad_s2=(0, 0, 0, 0, 0, -1))

    def test_zero_floor_softplus_uses_physical_boundary(self):
        module = self.make(
            mode="fixed", min_fraction=0.0, fixed_fraction=0.0,
            penalty_kind="softplus",
        )
        self.assertTrue(torch.all(module.margin == 0))
        state_inside = torch.zeros(12, dtype=torch.float64)
        state_at_limit = state_inside.clone()
        state_at_limit[0] = module.cost.q_max[0]
        self.assertGreater(module.cost_value(state_at_limit),
                           module.cost_value(state_inside))

    def test_post_dob_headroom_includes_output_compensation_when_enabled(self):
        module = self.make(
            mode="headroom", min_fraction=0.0, fixed_fraction=0.05,
            output_compensated_headroom=True,
        )
        row = module.update(
            torch.zeros(12, dtype=torch.float64),
            torch.zeros(10, 6, dtype=torch.float64),
            torch.tensor([140., 0, 0, 0, 0, 0], dtype=torch.float64),
        )
        self.assertEqual(row["headroom_positive"][0].item(), 10.0)
        self.assertEqual(row["center_horizon"][0, 0].item(), 0.0)
        self.assertEqual(row["headroom_center_horizon"][0, 0].item(), 140.0)


if __name__ == "__main__":
    unittest.main()
