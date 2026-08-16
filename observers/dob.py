"""First-order Q-filter disturbance observers for nonlinear robot dynamics."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from config import DOB_ALPHA


class DisturbanceObserver:
    """Numpy nonlinear DOB using an injected nominal inverse-dynamics model.

    ``nominal_model(q, dq, ddq)`` must return
    ``M(q)ddq + C(q,dq)dq + g(q) + B*dq``.
    """

    def __init__(self, nominal_model: Callable, dt: float, n_joints: int = 6,
                 alpha: float = DOB_ALPHA):
        self.nominal_model = nominal_model
        self.dt, self.alpha = dt, alpha
        self.d_hat = np.zeros(n_joints, dtype=np.float64)
        self.residual = np.zeros_like(self.d_hat)
        self.tau_model = np.zeros_like(self.d_hat)

    def reset(self) -> None:
        self.d_hat.fill(0.0)
        self.residual.fill(0.0)
        self.tau_model.fill(0.0)

    def update(self, applied, q_previous, velocity_previous,
               q_current, velocity_current):
        acceleration = (velocity_current - velocity_previous) / self.dt
        q_mid = 0.5 * (q_previous + q_current)
        velocity_mid = 0.5 * (velocity_previous + velocity_current)
        self.tau_model = np.asarray(
            self.nominal_model(q_mid, velocity_mid, acceleration),
            dtype=np.float64,
        )
        self.residual = np.asarray(applied) - self.tau_model
        self.d_hat += self.dt * self.alpha * (self.residual - self.d_hat)
        return self.d_hat


class TorchDisturbanceObserver:
    """Torch DOB whose nominal torque uses the same nonlinear UR5 model as MPPI.

    The residual convention matches the simulated plant, where an opposing
    external disturbance gives ``tau_effective = tau_applied - disturbance``.
    Consequently ``tau_applied - tau_model`` estimates that disturbance.
    """

    def __init__(self, dynamics, dt: float, alpha: float = DOB_ALPHA):
        self.dynamics = dynamics
        self.dt, self.alpha = dt, alpha
        self.d_hat = dynamics.damping.new_zeros(6)
        self.residual = torch.zeros_like(self.d_hat)
        self.tau_model = torch.zeros_like(self.d_hat)

    def reset(self) -> None:
        self.d_hat = torch.zeros_like(self.d_hat)
        self.residual = torch.zeros_like(self.residual)
        self.tau_model = torch.zeros_like(self.tau_model)

    def update(self, applied: torch.Tensor,
               q_previous: torch.Tensor, velocity_previous: torch.Tensor,
               q_current: torch.Tensor, velocity_current: torch.Tensor):
        measured_acceleration = (velocity_current - velocity_previous) / self.dt
        q_mid = 0.5 * (q_previous + q_current)
        velocity_mid = 0.5 * (velocity_previous + velocity_current)
        mass_matrix, bias = self.dynamics.mass_matrix_and_bias(
            q_mid.unsqueeze(0), velocity_mid.unsqueeze(0)
        )
        self.tau_model = (
            (mass_matrix[0] @ measured_acceleration)
            + bias[0]
            + self.dynamics.damping * velocity_mid
        )
        self.residual = applied - self.tau_model
        self.d_hat = self.d_hat + self.dt * self.alpha * (
            self.residual - self.d_hat
        )
        return self.d_hat
