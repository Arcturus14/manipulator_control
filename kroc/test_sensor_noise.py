"""Checks for reproducible measurement noise and exact clean measurements."""
import unittest

import torch

from kroc.sensor_noise import SensorNoiseConfig, StateMeasurementNoise


class SensorNoiseTests(unittest.TestCase):
    def test_zero_noise_preserves_state(self):
        state = torch.randn(12, dtype=torch.float64)
        model = StateMeasurementNoise(
            torch.device("cpu"), torch.float64,
            SensorNoiseConfig(position_std_rad=0, velocity_std_rad_s=0),
        )
        measured = model.measure(state)
        torch.testing.assert_close(measured, state, rtol=0, atol=0)
        self.assertIsNot(measured, state)

    def test_reset_reproduces_noise_without_global_rng(self):
        state = torch.zeros(12, dtype=torch.float64)
        model = StateMeasurementNoise(
            torch.device("cpu"), torch.float64,
            SensorNoiseConfig(position_std_rad=.1,
                              velocity_std_rad_s=.2, seed=31),
        )
        global_before = torch.random.get_rng_state()
        first = model.measure(state)
        global_after = torch.random.get_rng_state()
        torch.testing.assert_close(global_after, global_before)
        model.reset()
        torch.testing.assert_close(model.measure(state), first)
        self.assertTrue(torch.all(first[:6] != 0))
        self.assertTrue(torch.all(first[6:] != 0))

    def test_invalid_standard_deviation_is_rejected(self):
        with self.assertRaises(ValueError):
            SensorNoiseConfig(position_std_rad=-1)


if __name__ == "__main__":
    unittest.main()
