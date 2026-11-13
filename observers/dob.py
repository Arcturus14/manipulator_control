"""First-order Q-filter disturbance observers for nonlinear robot dynamics."""
from __future__ import annotations

from collections.abc import Callable
import math

import numpy as np
import torch

from config import DOB_ALPHA


def _acceleration_lpf_retention(dt: float, cutoff_hz: float) -> float | None:
    """Return the previous-sample weight for a causal first-order LPF.

    A zero cutoff disables the extra acceleration filter and preserves the
    legacy DOB exactly.  For a positive cutoff, ``beta = exp(-2*pi*fc*dt)``.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not math.isfinite(cutoff_hz) or cutoff_hz < 0:
        raise ValueError("acceleration LPF cutoff must be finite and nonnegative")
    if cutoff_hz == 0:
        return None
    return math.exp(-2.0 * math.pi * cutoff_hz * dt)


def _exact_lpf_retention(dt: float, cutoff_hz: float | None) -> float | None:
    """Exact ZOH retention for a requested first-order cutoff in hertz.

    ``None`` deliberately selects the legacy Euler DOB update so old runs and
    the pre-noise controller remain bit-for-bit on their original path.
    """
    if cutoff_hz is None:
        return None
    if not math.isfinite(cutoff_hz) or cutoff_hz <= 0:
        raise ValueError("DOB cutoff must be finite and positive")
    return math.exp(-2.0 * math.pi * cutoff_hz * dt)


class DisturbanceObserver:
    """Numpy nonlinear DOB using an injected nominal inverse-dynamics model.

    ``nominal_model(q, dq, ddq)`` must return
    ``M(q)ddq + C(q,dq)dq + g(q) + B*dq``.
    """

    def __init__(self, nominal_model: Callable, dt: float, n_joints: int = 6,
                 alpha: float = DOB_ALPHA,
                 acceleration_lpf_cutoff_hz: float = 0.0,
                 cutoff_hz: float | None = None):
        self.nominal_model = nominal_model
        self.dt, self.alpha = dt, alpha
        self.acceleration_lpf_cutoff_hz = float(acceleration_lpf_cutoff_hz)
        self.acceleration_lpf_retention = _acceleration_lpf_retention(
            dt, self.acceleration_lpf_cutoff_hz
        )
        self.cutoff_hz = None if cutoff_hz is None else float(cutoff_hz)
        self.exact_lpf_retention = _exact_lpf_retention(dt, self.cutoff_hz)
        self.d_hat = np.zeros(n_joints, dtype=np.float64)
        self.residual = np.zeros_like(self.d_hat)
        self.tau_model = np.zeros_like(self.d_hat)
        self.raw_acceleration = np.zeros_like(self.d_hat)
        self.filtered_acceleration = np.zeros_like(self.d_hat)

    def reset(self) -> None:
        self.d_hat.fill(0.0)
        self.residual.fill(0.0)
        self.tau_model.fill(0.0)
        self.raw_acceleration.fill(0.0)
        self.filtered_acceleration.fill(0.0)

    def update(self, applied, q_previous, velocity_previous,
               q_current, velocity_current):
        self.raw_acceleration = (
            np.asarray(velocity_current) - np.asarray(velocity_previous)
        ) / self.dt
        if self.acceleration_lpf_retention is None:
            self.filtered_acceleration = self.raw_acceleration.copy()
        else:
            beta = self.acceleration_lpf_retention
            self.filtered_acceleration = (
                beta * self.filtered_acceleration
                + (1.0 - beta) * self.raw_acceleration
            )
        q_mid = 0.5 * (q_previous + q_current)
        velocity_mid = 0.5 * (velocity_previous + velocity_current)
        self.tau_model = np.asarray(
            self.nominal_model(q_mid, velocity_mid, self.filtered_acceleration),
            dtype=np.float64,
        )
        self.residual = np.asarray(applied) - self.tau_model
        if self.exact_lpf_retention is None:
            self.d_hat += self.dt * self.alpha * (self.residual - self.d_hat)
        else:
            beta = self.exact_lpf_retention
            self.d_hat = beta * self.d_hat + (1.0 - beta) * self.residual
        return self.d_hat


class TorchDisturbanceObserver:
    """Torch DOB whose nominal torque uses the same nonlinear UR5 model as MPPI.

    The residual convention matches the simulated plant, where an opposing
    external disturbance gives ``tau_effective = tau_applied - disturbance``.
    Consequently ``tau_applied - tau_model`` estimates that disturbance.
    """

    def __init__(self, dynamics, dt: float, alpha: float = DOB_ALPHA,
                 acceleration_lpf_cutoff_hz: float = 0.0,
                 cutoff_hz: float | None = None):
        self.dynamics = dynamics
        self.dt, self.alpha = dt, alpha
        self.acceleration_lpf_cutoff_hz = float(acceleration_lpf_cutoff_hz)
        self.acceleration_lpf_retention = _acceleration_lpf_retention(
            dt, self.acceleration_lpf_cutoff_hz
        )
        self.cutoff_hz = None if cutoff_hz is None else float(cutoff_hz)
        self.exact_lpf_retention = _exact_lpf_retention(dt, self.cutoff_hz)
        self.d_hat = dynamics.damping.new_zeros(6)
        self.residual = torch.zeros_like(self.d_hat)
        self.tau_model = torch.zeros_like(self.d_hat)
        self.raw_acceleration = torch.zeros_like(self.d_hat)
        self.filtered_acceleration = torch.zeros_like(self.d_hat)

    def reset(self) -> None:
        self.d_hat = torch.zeros_like(self.d_hat)
        self.residual = torch.zeros_like(self.residual)
        self.tau_model = torch.zeros_like(self.tau_model)
        self.raw_acceleration = torch.zeros_like(self.raw_acceleration)
        self.filtered_acceleration = torch.zeros_like(self.filtered_acceleration)

    def update(self, applied: torch.Tensor,
               q_previous: torch.Tensor, velocity_previous: torch.Tensor,
               q_current: torch.Tensor, velocity_current: torch.Tensor):
        self.raw_acceleration = (
            velocity_current - velocity_previous
        ) / self.dt
        if self.acceleration_lpf_retention is None:
            self.filtered_acceleration = self.raw_acceleration.clone()
        else:
            beta = self.acceleration_lpf_retention
            self.filtered_acceleration = (
                beta * self.filtered_acceleration
                + (1.0 - beta) * self.raw_acceleration
            )
        q_mid = 0.5 * (q_previous + q_current)
        velocity_mid = 0.5 * (velocity_previous + velocity_current)
        mass_matrix, bias = self.dynamics.mass_matrix_and_bias(
            q_mid.unsqueeze(0), velocity_mid.unsqueeze(0)
        )
        self.tau_model = (
            (mass_matrix[0] @ self.filtered_acceleration)
            + bias[0]
            + self.dynamics.damping * velocity_mid
        )
        self.residual = applied - self.tau_model
        if self.exact_lpf_retention is None:
            self.d_hat = self.d_hat + self.dt * self.alpha * (
                self.residual - self.d_hat
            )
        else:
            beta = self.exact_lpf_retention
            self.d_hat = beta * self.d_hat + (1.0 - beta) * self.residual
        return self.d_hat
