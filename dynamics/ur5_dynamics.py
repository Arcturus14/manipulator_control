"""Batched nonlinear rigid-body dynamics for the UR5 manipulator.

The model evaluates

    M(q) qdd + C(q, qd) qd + g(q) + B qd = tau

with a vectorized Recursive Newton-Euler Algorithm (RNEA). Seven inverse-
dynamics scenarios are evaluated in one batch to obtain the bias force and all
six columns of the mass matrix without numerical differentiation or autograd.
"""
from __future__ import annotations

import math

import torch

from config import (B_DAMP, DH_PARAMS, DQ_MAX, DT, GRAVITY, I_EFF,
                    JOINT_ARMATURE, LINK_COM, LINK_INERTIA_DIAG, LINK_MASSES,
                    MASS_MATRIX_REGULARIZATION, TAU_MAX)


def _skew(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(
        vector.shape[:-1] + (3, 3)
    )


def _motion_cross(motion: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    angular, linear = motion[..., :3], motion[..., 3:]
    other_angular, other_linear = other[..., :3], other[..., 3:]
    return torch.cat((torch.linalg.cross(angular, other_angular),
                      torch.linalg.cross(angular, other_linear)
                      + torch.linalg.cross(linear, other_angular)), dim=-1)


def _force_cross(motion: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
    angular, linear = motion[..., :3], motion[..., 3:]
    moment, resultant = force[..., :3], force[..., 3:]
    return torch.cat((torch.linalg.cross(angular, moment)
                      + torch.linalg.cross(linear, resultant),
                      torch.linalg.cross(angular, resultant)), dim=-1)


def _dh_motion_transform(theta: torch.Tensor, a: float, d: float,
                         alpha: float) -> torch.Tensor:
    """Return parent-to-child spatial motion transform for standard DH data."""
    ct, st = torch.cos(theta), torch.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    zero, one = torch.zeros_like(theta), torch.ones_like(theta)
    rotation = torch.stack((
        ct, -st * ca, st * sa,
        st, ct * ca, -ct * sa,
        zero, sa * one, ca * one,
    ), dim=-1).reshape(theta.shape + (3, 3))
    translation = torch.stack((a * ct, a * st, d * one), dim=-1)
    rotation_t = rotation.transpose(-1, -2)
    upper = torch.cat((rotation_t, torch.zeros_like(rotation_t)), dim=-1)
    lower = torch.cat((-rotation_t @ _skew(translation), rotation_t), dim=-1)
    return torch.cat((upper, lower), dim=-2)


class UR5Dynamics:
    """Nonlinear UR5 model shared by MPPI rollouts and the simulated plant."""

    def __init__(self, device: torch.device, dtype: torch.dtype, dt: float = DT):
        self.dt = dt
        self.damping = torch.tensor(B_DAMP, dtype=dtype, device=device)
        self.u_max = torch.tensor(TAU_MAX, dtype=dtype, device=device)
        self.u_min = -self.u_max
        # Kept only for the legacy diagonal DOB interface. The nonlinear plant
        # itself uses the spatial link inertias below.
        self.inertia = torch.tensor(I_EFF, dtype=dtype, device=device)
        self.armature = torch.tensor(JOINT_ARMATURE, dtype=dtype, device=device)
        self.gravity = torch.tensor(GRAVITY, dtype=dtype, device=device)
        self.mass_regularization = MASS_MATRIX_REGULARIZATION
        self.spatial_inertia = self._make_spatial_inertias(device, dtype)
        # Joint screw axes expressed in each child DH frame. Because standard
        # DH applies TransZ(d), TransX(a), RotX(alpha) after RotZ(q), the child
        # origin generally has a non-zero linear joint velocity.
        joint_axes = []
        for a, _d, alpha, _offset in DH_PARAMS:
            sa, ca = math.sin(alpha), math.cos(alpha)
            joint_axes.append((0.0, sa, ca, 0.0, a * ca, -a * sa))
        self.joint_axes = torch.tensor(joint_axes, dtype=dtype, device=device)

    @staticmethod
    def _make_spatial_inertias(device, dtype) -> torch.Tensor:
        matrices = []
        eye = torch.eye(3, dtype=dtype, device=device)
        for mass, com_values, inertia_values in zip(
                LINK_MASSES, LINK_COM, LINK_INERTIA_DIAG):
            com = torch.tensor(com_values, dtype=dtype, device=device)
            com_cross = _skew(com)
            inertia_com = torch.diag(torch.tensor(
                inertia_values, dtype=dtype, device=device
            ))
            upper = torch.cat((inertia_com + mass * com_cross @ com_cross.T,
                               mass * com_cross), dim=1)
            lower = torch.cat((mass * com_cross.T, mass * eye), dim=1)
            matrices.append(torch.cat((upper, lower), dim=0))
        return torch.stack(matrices)

    def inverse_dynamics(self, q: torch.Tensor, velocity: torch.Tensor,
                         acceleration: torch.Tensor,
                         base_acceleration: torch.Tensor | None = None) -> torch.Tensor:
        """Vectorized RNEA returning joint torque for arbitrary leading shape."""
        shape = q.shape[:-1]
        q_flat = q.reshape(-1, 6)
        velocity_flat = velocity.reshape(-1, 6)
        acceleration_flat = acceleration.reshape(-1, 6)
        count = q_flat.shape[0]
        if base_acceleration is None:
            base = q_flat.new_zeros((count, 6))
            base[:, 3:] = -self.gravity
        else:
            base = base_acceleration.reshape(count, 6)

        parent_velocity = q_flat.new_zeros((count, 6))
        parent_acceleration = base
        transforms, spatial_velocities, spatial_forces = [], [], []
        joint_motions = []
        for index, (a, d, alpha, offset) in enumerate(DH_PARAMS):
            transform = _dh_motion_transform(q_flat[:, index] + offset, a, d, alpha)
            axis = self.joint_axes[index].expand(count, -1)
            joint_velocity = axis * velocity_flat[:, index:index + 1]
            link_velocity = (transform @ parent_velocity.unsqueeze(-1)).squeeze(-1) + joint_velocity
            link_acceleration = (
                (transform @ parent_acceleration.unsqueeze(-1)).squeeze(-1)
                + axis * acceleration_flat[:, index:index + 1]
                + _motion_cross(link_velocity, joint_velocity)
            )
            momentum = (self.spatial_inertia[index] @ link_velocity.unsqueeze(-1)).squeeze(-1)
            force = ((self.spatial_inertia[index] @ link_acceleration.unsqueeze(-1)).squeeze(-1)
                     + _force_cross(link_velocity, momentum))
            transforms.append(transform)
            joint_motions.append(axis)
            spatial_velocities.append(link_velocity)
            spatial_forces.append(force)
            parent_velocity, parent_acceleration = link_velocity, link_acceleration

        torque = q_flat.new_empty((count, 6))
        accumulated = spatial_forces[-1]
        for index in range(5, -1, -1):
            if index < 5:
                accumulated = spatial_forces[index] + (
                    transforms[index + 1].transpose(-1, -2)
                    @ accumulated.unsqueeze(-1)
                ).squeeze(-1)
            torque[:, index] = (accumulated * joint_motions[index]).sum(dim=1)
        return torque.reshape(shape + (6,))

    def mass_matrix_and_bias(self, q: torch.Tensor,
                             velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return M(q) and h(q,qd)=C(q,qd)qd+g(q) in one batched RNEA call."""
        shape = q.shape[:-1]
        count = q.reshape(-1, 6).shape[0]
        q_scenarios = q.reshape(count, 1, 6).expand(-1, 7, -1)
        velocity_scenarios = q_scenarios.new_zeros((count, 7, 6))
        velocity_scenarios[:, 0] = velocity.reshape(count, 6)
        acceleration_scenarios = q_scenarios.new_zeros((count, 7, 6))
        acceleration_scenarios[:, 1:] = torch.eye(
            6, dtype=q.dtype, device=q.device
        ).unsqueeze(0)
        base = q_scenarios.new_zeros((count, 7, 6))
        base[:, 0, 3:] = -self.gravity
        all_torque = self.inverse_dynamics(
            q_scenarios, velocity_scenarios, acceleration_scenarios, base
        )
        bias = all_torque[:, 0]
        mass_matrix = all_torque[:, 1:].transpose(1, 2)
        eye = torch.eye(6, dtype=q.dtype, device=q.device).expand(count, -1, -1)
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.transpose(1, 2))
        mass_matrix = mass_matrix + torch.diag_embed(self.armature.expand(count, -1))
        mass_matrix = mass_matrix + self.mass_regularization * eye
        return mass_matrix.reshape(shape + (6, 6)), bias.reshape(shape + (6,))

    def forward_dynamics(self, state: torch.Tensor, torque: torch.Tensor) -> torch.Tensor:
        velocity = state[..., 6:]
        mass_matrix, bias = self.mass_matrix_and_bias(state[..., :6], velocity)
        rhs = torque - bias - self.damping * velocity
        return torch.linalg.solve(mass_matrix, rhs.unsqueeze(-1)).squeeze(-1)

    def state_derivative(self, state: torch.Tensor, torque: torch.Tensor) -> torch.Tensor:
        return torch.cat((state[..., 6:], self.forward_dynamics(state, torque)), dim=-1)

    def rollout(self, state: torch.Tensor, action: torch.Tensor, _step: int = 0) -> torch.Tensor:
        """Fast semi-implicit Euler step used for thousands of MPPI rollouts."""
        return self.rollout_effective(
            state, action.clamp(self.u_min, self.u_max), _step
        )

    def rollout_effective(self, state: torch.Tensor, effective_torque: torch.Tensor,
                          _step: int = 0) -> torch.Tensor:
        """Roll out an already composed actuator-minus-disturbance torque.

        Actuator saturation must be applied before an external disturbance is
        subtracted.  The resulting generalized torque is not actuator output
        and therefore must not be clamped to the actuator limits again.
        """
        acceleration = self.forward_dynamics(state, effective_torque)
        next_velocity = (state[..., 6:] + self.dt * acceleration).clamp(-DQ_MAX, DQ_MAX)
        next_position = state[..., :6] + self.dt * next_velocity
        return torch.cat((next_position, next_velocity), dim=-1)

    def plant_step(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """RK4 step for an actuator command, including actuator saturation."""
        return self.plant_step_effective(
            state, action.clamp(self.u_min, self.u_max)
        )

    def plant_step_effective(self, state: torch.Tensor,
                             effective_torque: torch.Tensor) -> torch.Tensor:
        """RK4 step for torque after external disturbance composition."""
        k1 = self.state_derivative(state, effective_torque)
        k2 = self.state_derivative(
            state + 0.5 * self.dt * k1, effective_torque
        )
        k3 = self.state_derivative(
            state + 0.5 * self.dt * k2, effective_torque
        )
        k4 = self.state_derivative(state + self.dt * k3, effective_torque)
        next_state = state + self.dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        return torch.cat((next_state[..., :6],
                          next_state[..., 6:].clamp(-DQ_MAX, DQ_MAX)), dim=-1)

    def gravity_compensation(self, q: torch.Tensor) -> torch.Tensor:
        zeros = torch.zeros_like(q)
        return self.inverse_dynamics(q, zeros, zeros)
