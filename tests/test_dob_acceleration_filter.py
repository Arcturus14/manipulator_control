"""Unit tests for the DOB-only acceleration low-pass filter."""
import math
import unittest

import torch

from observers.dob import TorchDisturbanceObserver


class _IdentityDynamics:
    def __init__(self):
        self.damping = torch.zeros(6, dtype=torch.float64)

    def mass_matrix_and_bias(self, q, velocity):
        batch = q.shape[0]
        matrix = torch.eye(6, dtype=q.dtype).expand(batch, -1, -1).clone()
        bias = torch.zeros((batch, 6), dtype=q.dtype)
        return matrix, bias


class DOBAccelerationFilterTests(unittest.TestCase):
    def setUp(self):
        self.dynamics = _IdentityDynamics()
        self.zeros = torch.zeros(6, dtype=torch.float64)
        self.ones = torch.ones(6, dtype=torch.float64)

    def _update_for_raw_acceleration(self, observer, acceleration):
        observer.update(
            self.zeros, self.zeros, self.zeros,
            self.zeros, acceleration * observer.dt,
        )

    def test_zero_cutoff_preserves_unfiltered_acceleration(self):
        observer = TorchDisturbanceObserver(
            self.dynamics, dt=0.02, acceleration_lpf_cutoff_hz=0.0
        )
        self._update_for_raw_acceleration(observer, self.ones)
        torch.testing.assert_close(observer.raw_acceleration, self.ones)
        torch.testing.assert_close(observer.filtered_acceleration, self.ones)
        torch.testing.assert_close(observer.tau_model, self.ones)

    def test_positive_cutoff_applies_causal_first_order_filter(self):
        cutoff, dt = 5.0, 0.02
        beta = math.exp(-2.0 * math.pi * cutoff * dt)
        observer = TorchDisturbanceObserver(
            self.dynamics, dt=dt, acceleration_lpf_cutoff_hz=cutoff
        )
        self._update_for_raw_acceleration(observer, self.ones)
        torch.testing.assert_close(
            observer.filtered_acceleration,
            torch.full_like(self.ones, 1.0 - beta),
        )
        self._update_for_raw_acceleration(observer, self.ones)
        torch.testing.assert_close(
            observer.filtered_acceleration,
            torch.full_like(self.ones, 1.0 - beta ** 2),
        )

    def test_reset_clears_filter_memory(self):
        observer = TorchDisturbanceObserver(
            self.dynamics, dt=0.02, acceleration_lpf_cutoff_hz=5.0
        )
        self._update_for_raw_acceleration(observer, self.ones)
        observer.reset()
        torch.testing.assert_close(
            observer.filtered_acceleration, torch.zeros_like(self.ones)
        )

    def test_invalid_cutoff_is_rejected(self):
        for cutoff in (-1.0, float("nan"), float("inf")):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                TorchDisturbanceObserver(
                    self.dynamics, dt=0.02,
                    acceleration_lpf_cutoff_hz=cutoff,
                )

    def test_alpha_sets_residual_update_rate(self):
        observer = TorchDisturbanceObserver(
            self.dynamics, dt=0.02, alpha=10.0
        )
        self._update_for_raw_acceleration(observer, self.ones)
        # applied=0 and tau_model=1 gives residual=-1; dt*alpha=0.2.
        torch.testing.assert_close(
            observer.d_hat, torch.full_like(self.ones, -0.2)
        )

    def test_exact_cutoff_uses_exponential_discretization(self):
        cutoff, dt = 8.0, 0.02
        observer = TorchDisturbanceObserver(
            self.dynamics, dt=dt, alpha=40.0, cutoff_hz=cutoff
        )
        self._update_for_raw_acceleration(observer, self.ones)
        beta = math.exp(-2.0 * math.pi * cutoff * dt)
        torch.testing.assert_close(
            observer.d_hat, torch.full_like(self.ones, -(1.0 - beta))
        )

    def test_none_cutoff_preserves_legacy_euler_update(self):
        observer = TorchDisturbanceObserver(
            self.dynamics, dt=0.02, alpha=40.0, cutoff_hz=None
        )
        self._update_for_raw_acceleration(observer, self.ones)
        torch.testing.assert_close(
            observer.d_hat, torch.full_like(self.ones, -0.8)
        )

    def test_invalid_exact_cutoff_is_rejected(self):
        for cutoff in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(cutoff=cutoff), self.assertRaises(ValueError):
                TorchDisturbanceObserver(
                    self.dynamics, dt=0.02, cutoff_hz=cutoff
                )


if __name__ == "__main__":
    unittest.main()
