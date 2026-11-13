"""Structural checks for KRoC state-constraint approach references."""
import math
import unittest

import numpy as np

from kinematics.ur5_kinematics import fk_joints_np
from kroc.benchmark_noise_aware_state_margin import build_scenario


class ConstraintScenarioTests(unittest.TestCase):
    def test_position_reference_crosses_fixed_soft_boundary_only(self):
        reference, q0, disturbance, metadata = build_scenario(
            "position_limit", 250
        )
        self.assertGreater(metadata["reference_joint_position_max"][2], .9 * math.pi)
        self.assertLess(metadata["reference_joint_position_max"][2], math.pi)
        np.testing.assert_allclose(reference[0], fk_joints_np(q0)[-1], atol=1e-12)
        self.assertEqual(metadata["disturbance_nm"][2], -10.0)
        self.assertEqual(disturbance(250 // 4, "cpu", None).numpy()[2], -10.0)

    def test_position_scenario_accepts_in_limit_finish_and_signed_disturbance(self):
        _, _, _, metadata = build_scenario(
            "position_limit", 100,
            position_finish_fraction=0.995,
            position_disturbance_nm=-100.0,
        )
        self.assertAlmostEqual(
            metadata["reference_joint_position_max"][2], 0.995 * math.pi
        )
        self.assertEqual(metadata["disturbance_nm"][2], -100.0)

    def test_velocity_scenario_uses_signed_wrist_recovery_case(self):
        reference, q0, disturbance, metadata = build_scenario("velocity_limit", 250)
        self.assertEqual(metadata["reference_joint_velocity_abs_max"], 0.0)
        self.assertEqual(metadata["disturbance_duration_steps"], 10)
        self.assertEqual(metadata["disturbance_ramp_steps"], 5)
        self.assertEqual(metadata["disturbance_hold_steps"], 5)
        self.assertEqual(metadata["disturbance_onset_step"], 0)
        np.testing.assert_allclose(metadata["disturbance_nm"], [0, 0, 0, 0, 0, -16])
        np.testing.assert_allclose(metadata["initial_joint_velocity"], [0, 0, 0, 0, 0, 2.5])
        np.testing.assert_allclose(reference, np.repeat(reference[:1], 250, axis=0))
        np.testing.assert_allclose(reference[0], fk_joints_np(q0)[-1], atol=1e-12)

        onset = 0
        values = [
            disturbance(onset + offset, "cpu", None).numpy()[5]
            for offset in range(11)
        ]
        np.testing.assert_allclose(values[:5], [-3.2, -6.4, -9.6, -12.8, -16.0])
        np.testing.assert_allclose(values[5:10], -16.0)
        self.assertEqual(values[10], 0.0)

    def test_short_constraint_scenario_is_rejected(self):
        with self.assertRaises(ValueError):
            build_scenario("position_limit", 40)


if __name__ == "__main__":
    unittest.main()
