from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import torch

from controllers.mppi_dob_controller import (
    UR5Tracker,
    compose_nominal_sequence,
    recover_residual_sequence,
    shift_control_sequence,
)
from dynamics.ur5_dynamics import UR5Dynamics


class DobAwareWarmStartTests(unittest.TestCase):
    def test_rollout_estimate_is_a_snapshot(self):
        tracker = object.__new__(UR5Tracker)
        tracker.dob = SimpleNamespace(d_hat=torch.tensor([2.0, 3.0]))
        tracker._rollout_disturbance = torch.zeros(2)
        tracker._set_rollout_disturbance(True)
        tracker.dob.d_hat.add_(10)
        self.assertTrue(torch.equal(tracker._rollout_disturbance, torch.tensor([2.0, 3.0])))
        tracker._set_rollout_disturbance(False)
        self.assertTrue(torch.equal(tracker._rollout_disturbance, torch.zeros(2)))

    def test_model_only_prediction_clamps_before_disturbance(self):
        tracker = object.__new__(UR5Tracker)
        tracker.dynamics = UR5Dynamics(torch.device("cpu"), torch.float64)
        tracker._rollout_disturbance = torch.tensor(
            [-20.0, 2.0, 0, 0, 0, 0], dtype=torch.float64
        )
        tracker._compiled_rollout_dynamics = lambda state, torque, step: torque
        action = torch.tensor([[200.0, 5.0, 0, 0, 0, 0]], dtype=torch.float64)
        predicted = tracker._stable_rollout_dynamics(None, action, 0)
        self.assertTrue(torch.equal(predicted, torch.tensor(
            [[170.0, 3.0, 0, 0, 0, 0]], dtype=torch.float64
        )))

    def test_model_only_retains_learned_compensation_in_residual(self):
        ff = torch.tensor([[5.0], [6.0], [7.0]])
        estimate = torch.tensor([2.0])
        optimum = ff + estimate
        residual = recover_residual_sequence(optimum, ff, estimate, False)
        shifted = shift_control_sequence(residual)
        center = compose_nominal_sequence(ff, shifted, estimate, False)
        self.assertTrue(torch.equal(center, torch.tensor([[7.0], [8.0], [7.0]])))

    def test_instruction_numeric_example(self):
        feedforward = torch.tensor([[5.0]])
        residual = torch.tensor([[0.5]])
        estimate = torch.tensor([2.0])
        nominal = compose_nominal_sequence(
            feedforward, residual, estimate, True
        )
        self.assertTrue(torch.equal(nominal, torch.tensor([[7.5]])))
        recovered = recover_residual_sequence(
            nominal, feedforward, estimate, True
        )
        self.assertTrue(torch.equal(recovered, residual))
        next_nominal = compose_nominal_sequence(
            torch.tensor([[5.1]]), recovered, torch.tensor([2.2]), True
        )
        self.assertTrue(torch.allclose(next_nominal, torch.tensor([[7.8]])))

    def test_zero_estimate_matches_non_aware_nominal(self):
        feedforward = torch.tensor([[5.0, -1.0]])
        residual = torch.tensor([[0.5, 0.25]])
        estimate = torch.zeros(2)
        self.assertTrue(torch.equal(
            compose_nominal_sequence(feedforward, residual, estimate, True),
            compose_nominal_sequence(feedforward, residual, estimate, False),
        ))

    def test_external_disturbance_is_not_actuator_clamped(self):
        dynamics = UR5Dynamics(torch.device("cpu"), torch.float64)
        dynamics.forward_dynamics = lambda _state, torque: torque
        state = torch.zeros((1, 12), dtype=torch.float64)
        torque = torch.tensor([[200.0, 0, 0, 0, 0, 0]], dtype=torch.float64)
        effective_next = dynamics.rollout_effective(state, torque)
        actuator_next = dynamics.rollout(state, torque)
        self.assertAlmostEqual(effective_next[0, 6].item(), 4.0)
        self.assertAlmostEqual(actuator_next[0, 6].item(), 3.0)

    def test_main_default_remains_output_compensation(self):
        default = inspect.signature(UR5Tracker.run).parameters[
            "dob_compensation_mode"
        ].default
        self.assertEqual(default, "output")

    def test_main_default_remains_feedforward_centered(self):
        default = inspect.signature(UR5Tracker.run).parameters[
            "sampling_center_mode"
        ].default
        self.assertEqual(default, "feedforward_residual")

    def test_zero_initial_sequence_shift(self):
        optimal = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        shifted = shift_control_sequence(optimal)
        expected = torch.tensor([
            [3.0, 4.0],
            [5.0, 6.0],
            [0.0, 0.0],
        ])
        self.assertTrue(torch.equal(shifted, expected))
        self.assertTrue(torch.equal(
            optimal,
            torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
        ))


if __name__ == "__main__":
    unittest.main()
