from __future__ import annotations

import inspect
import unittest

import torch
from pytorch_mppi import MPPI

from controllers.mppi_dob_controller import TemporalNoiseMPPI, UR5Tracker


def _dynamics(state, action):
    return state


def _running_cost(state, action):
    return torch.zeros(state.shape[:-1], device=state.device, dtype=state.dtype)


def _controller(cls=TemporalNoiseMPPI, *, rho=0.0, samples=32, horizon=5):
    kwargs = dict(
        dynamics=_dynamics,
        running_cost=_running_cost,
        nx=2,
        noise_sigma=torch.diag(torch.tensor([4.0, 9.0], dtype=torch.float64)),
        num_samples=samples,
        horizon=horizon,
        device="cpu",
        u_min=torch.full((2,), -1.0e6, dtype=torch.float64),
        u_max=torch.full((2,), 1.0e6, dtype=torch.float64),
    )
    if cls is TemporalNoiseMPPI:
        kwargs["temporal_noise_rho"] = rho
    return cls(**kwargs)


class TemporalNoiseTests(unittest.TestCase):
    def test_default_is_legacy_sampler(self):
        default = inspect.signature(UR5Tracker.__init__).parameters[
            "temporal_noise_rho"
        ].default
        self.assertEqual(default, 0.0)

    def test_rho_zero_exactly_matches_upstream_rng_path(self):
        torch.manual_seed(101)
        upstream = _controller(MPPI)
        torch.manual_seed(101)
        temporal = _controller(rho=0.0)

        torch.manual_seed(202)
        expected = upstream._sample_noise((32, 5))
        torch.manual_seed(202)
        actual = temporal._sample_noise((32, 5))
        self.assertTrue(torch.equal(actual, expected))

    def test_rho_zero_exactly_matches_upstream_command_result(self):
        torch.manual_seed(111)
        upstream = _controller(MPPI)
        torch.manual_seed(111)
        temporal = _controller(rho=0.0)
        state = torch.tensor([0.25, -0.5], dtype=torch.float64)

        torch.manual_seed(222)
        expected = upstream.command(state, shift_nominal_trajectory=False)
        torch.manual_seed(222)
        actual = temporal.command(state, shift_nominal_trajectory=False)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(temporal.U, upstream.U))
        self.assertTrue(torch.equal(temporal.noise, upstream.noise))

    def test_rho_point_eight_preserves_variance_and_adds_correlation(self):
        controller = _controller(rho=0.8, samples=30_000, horizon=4)
        torch.manual_seed(303)
        noise = controller._sample_noise((30_000, 4))

        marginal_variance = noise.var(dim=(0, 1), unbiased=True)
        self.assertTrue(torch.allclose(
            marginal_variance, torch.tensor([4.0, 9.0], dtype=torch.float64),
            rtol=0.03, atol=0.05,
        ))
        for joint in range(2):
            correlation = torch.corrcoef(torch.stack((
                noise[:, 0, joint], noise[:, 1, joint]
            )))[0, 1]
            self.assertAlmostEqual(correlation.item(), 0.8, delta=0.02)

    def test_rollout_and_weighted_update_share_bounded_epsilon(self):
        controller = _controller(rho=0.8)
        torch.manual_seed(404)
        controller._compute_perturbed_action_and_noise()
        self.assertTrue(torch.equal(
            controller.noise, controller.perturbed_action - controller.U
        ))

    def test_invalid_rho_is_rejected(self):
        for rho in (-0.01, 1.01):
            with self.subTest(rho=rho), self.assertRaises(ValueError):
                _controller(rho=rho)

    def test_diagonal_variance_can_be_updated_in_place(self):
        controller = _controller(rho=0.0, samples=30_000, horizon=2)
        variance = torch.tensor([16.0, 25.0], dtype=torch.float64)
        controller.set_diagonal_noise_variance(variance)
        torch.manual_seed(505)
        noise = controller._sample_noise((30_000, 2))
        self.assertTrue(torch.allclose(
            noise.var(dim=(0, 1), unbiased=True), variance,
            rtol=0.03, atol=0.08,
        ))
        self.assertTrue(torch.equal(
            controller._noise_sigma_inv_diag, variance.reciprocal()
        ))

    def test_disturbance_covariance_gain_defaults_to_disabled(self):
        default = inspect.signature(UR5Tracker.__init__).parameters[
            "disturbance_covariance_gain"
        ].default
        self.assertEqual(default, 0.0)


if __name__ == "__main__":
    unittest.main()
