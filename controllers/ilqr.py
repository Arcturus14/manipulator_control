"""Iterative LQR controller sharing MPPI's model and cost structure."""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from config import (CROSS_EXP_K, CROSS_EXP_ZONE, CROSS_MARGIN,
                    DT, DYNAMIC_OBSTACLE_SIZE, HORIZON,
                    ILQR_CROSS_EXP_WEIGHT, ILQR_MAX_ITER,
                    ILQR_STATIC_EXP_WEIGHT, ILQR_W_ACT, ILQR_W_POS,
                    ILQR_W_SMOOTH, ILQR_W_SMOOTH_OBS, ILQR_W_TERM,
                    ILQR_W_VEL, MPPI_STATIC_EXP_GAIN, MPPI_STATIC_MARGIN,
                    MPPI_STATIC_ZONE,
                    STATIC_OBSTACLE_SIZE, TIMING_WARMUP_STEPS,
                    )
from dynamics.ur5_dynamics import UR5Dynamics
from kinematics.ur5_kinematics import (fk_and_jacobian_batch, fk_batch, ik_np,
                                      ik_trajectory_np)
from observers.dob import TorchDisturbanceObserver
from utils.disturbance import sinusoidal_disturbance

ILQR_STATIC_GUIDANCE_OFFSET = 0.012
ILQR_STATIC_GUIDANCE_HALF_WIDTH = 0.020


def _static_avoidance_reference(waypoints: np.ndarray,
                                centers: np.ndarray) -> np.ndarray:
    """Create a lateral warm-start seed without changing the cost reference."""
    guided = np.asarray(waypoints, dtype=float).copy()
    segment_length = np.linalg.norm(np.diff(guided, axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(segment_length)))
    for center in np.asarray(centers):
        obstacle_index = int(np.argmin(np.linalg.norm(guided - center, axis=1)))
        before, after = max(0, obstacle_index - 2), min(
            len(guided) - 1, obstacle_index + 2
        )
        tangent = guided[after] - guided[before]
        if np.linalg.norm(tangent) < 1e-9:
            continue
        tangent /= np.linalg.norm(tangent)
        normal = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(tangent, np.array([1.0, 0.0, 0.0]))
        normal /= np.linalg.norm(normal)
        distance = np.abs(arc_length - arc_length[obstacle_index])
        active = distance < ILQR_STATIC_GUIDANCE_HALF_WIDTH
        phase = distance[active] / ILQR_STATIC_GUIDANCE_HALF_WIDTH
        bump = ILQR_STATIC_GUIDANCE_OFFSET * 0.5 * (
            1.0 + np.cos(np.pi * phase)
        )
        guided[active] += bump[:, None] * normal
    return guided


class UR5ILQRTracker:
    """Receding-horizon iLQR tracker with MPPI-equivalent cost terms.

    Dynamics Jacobians use batched central differences.  Cost Hessians use a
    positive-semidefinite Gauss-Newton approximation, which is substantially
    more robust than exact second derivatives around FK and box barriers.
    """

    def __init__(self, waypoints: np.ndarray,
                 dynamic_paths: np.ndarray | None = None,
                 static_centers: np.ndarray | None = None, *,
                 horizon: int = HORIZON,
                 max_iterations: int = ILQR_MAX_ITER,
                 position_weight: float = ILQR_W_POS,
                 terminal_weight: float = ILQR_W_TERM,
                 velocity_weight: float = ILQR_W_VEL,
                 action_weight: float = ILQR_W_ACT,
                 smooth_weight: float | None = None,
                 static_exp_weight: float = ILQR_STATIC_EXP_WEIGHT,
                 dynamic_exp_weight: float = ILQR_CROSS_EXP_WEIGHT):
        if dynamic_paths is not None and static_centers is not None:
            raise ValueError("dynamic and static obstacles are mutually exclusive")
        if horizon < 1 or max_iterations < 1:
            raise ValueError("iLQR horizon and max_iterations must be positive")
        self.dtype = torch.float64
        self.device = torch.device("cpu")
        self.horizon = horizon
        self.max_iterations = max_iterations
        self.original_count = len(waypoints)
        padding = np.tile(waypoints[-1:], (horizon + 2, 1))
        self.waypoints = torch.as_tensor(
            np.vstack((waypoints, padding)), dtype=self.dtype
        )
        self.dynamic_paths = None if dynamic_paths is None else torch.as_tensor(
            dynamic_paths, dtype=self.dtype
        )
        self.static_centers = None if static_centers is None else torch.as_tensor(
            static_centers, dtype=self.dtype
        )
        self.warm_start_waypoints = (
            _static_avoidance_reference(waypoints, static_centers)
            if static_centers is not None else np.asarray(waypoints)
        )
        self.obstacle_mode = (
            "dynamic" if dynamic_paths is not None else
            "static" if static_centers is not None else "false"
        )
        self.position_weight = position_weight
        self.terminal_weight = terminal_weight
        self.velocity_weight = velocity_weight
        self.action_weight = action_weight
        self.static_exp_weight = static_exp_weight
        self.dynamic_exp_weight = dynamic_exp_weight
        self.smooth_weight = (
            ILQR_W_SMOOTH_OBS
            if self.obstacle_mode != "false" else ILQR_W_SMOOTH
        ) if smooth_weight is None else smooth_weight
        self.dynamics = UR5Dynamics(self.device, self.dtype)
        self._compiled_prediction = torch.compile(self.dynamics.rollout)
        self.dob = TorchDisturbanceObserver(self.dynamics, DT)
        self._regularization = 1e-3

    def _feedforward(self, q0: np.ndarray) -> torch.Tensor:
        targets = self.warm_start_waypoints
        joint = ik_trajectory_np(targets, q0)
        velocity = np.gradient(joint, DT, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, DT, axis=0, edge_order=2)
        torque = self.dynamics.inverse_dynamics(
            torch.as_tensor(joint, dtype=self.dtype),
            torch.as_tensor(velocity, dtype=self.dtype),
            torch.as_tensor(acceleration, dtype=self.dtype),
        )
        torque += self.dynamics.damping * torch.as_tensor(
            velocity, dtype=self.dtype
        )
        torque = torque.clamp(self.dynamics.u_min, self.dynamics.u_max)
        return torch.cat((torque, torque[-1:].expand(self.horizon + 2, -1)))

    def _rollout(self, state: torch.Tensor,
                 controls: torch.Tensor) -> torch.Tensor:
        states = [state]
        for control in controls:
            states.append(self._compiled_prediction(
                states[-1].unsqueeze(0), control.unsqueeze(0)
            )[0].clone())
        return torch.stack(states)

    def _obstacle_terms(self, ee: torch.Tensor, absolute_step: int,
                        horizon_step: int) -> tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor]:
        """Return barrier value, EE gradient, and PSD EE Hessian."""
        value = ee.new_zeros(())
        gradient = ee.new_zeros(3)
        hessian = ee.new_zeros((3, 3))
        if self.dynamic_paths is not None:
            path_index = min(
                absolute_step + horizon_step, self.dynamic_paths.shape[1] - 1
            )
            centers = self.dynamic_paths[:, path_index]
            half_size = DYNAMIC_OBSTACLE_SIZE / 2.0
            margin, zone = CROSS_MARGIN, CROSS_EXP_ZONE
            weight, gain = self.dynamic_exp_weight, CROSS_EXP_K
        elif self.static_centers is not None:
            centers = self.static_centers
            half_size = STATIC_OBSTACLE_SIZE / 2.0
            margin, zone = MPPI_STATIC_MARGIN, MPPI_STATIC_ZONE
            weight, gain = self.static_exp_weight, MPPI_STATIC_EXP_GAIN
        else:
            return value, gradient, hessian

        for center in centers:
            delta = ee - center
            outside = torch.relu(torch.abs(delta) - half_size)
            distance = torch.linalg.vector_norm(outside)
            penetration = margin + zone - distance
            if penetration.item() <= 0.0:
                continue
            exponential = torch.exp(gain * penetration)
            value = value + weight * torch.expm1(gain * penetration)
            if distance.item() <= 1e-12:
                continue
            distance_gradient = outside * torch.sign(delta) / distance
            scale = weight * gain * exponential
            gradient = gradient - scale * distance_gradient
            hessian = hessian + scale * gain * torch.outer(
                distance_gradient, distance_gradient
            )
        return value, gradient, hessian

    def _cost(self, states: torch.Tensor, controls: torch.Tensor,
              absolute_step: int, previous_action: torch.Tensor) -> torch.Tensor:
        ee = fk_batch(states[:, :6])
        references = self.waypoints[
            absolute_step:absolute_step + self.horizon + 1
        ]
        running_error = ee[:-1] - references[:-1]
        value = self.position_weight * running_error.square().sum()
        value += self.velocity_weight * states[:-1, 6:].square().sum()
        value += self.action_weight * controls.square().sum()
        value += self.smooth_weight * (controls[0] - previous_action).square().sum()
        value += self.terminal_weight * (ee[-1] - references[-1]).square().sum()
        for horizon_step in range(self.horizon):
            value += self._obstacle_terms(
                ee[horizon_step], absolute_step, horizon_step
            )[0]
        terminal_obstacle = self._obstacle_terms(
            ee[-1], absolute_step, self.horizon
        )[0]
        return value + 4.0 * terminal_obstacle

    def _cost_derivatives(
            self, states: torch.Tensor, controls: torch.Tensor,
            absolute_step: int, previous_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor]:
        nx, nu = 12, 6
        lx = states.new_zeros((self.horizon, nx))
        lu = states.new_zeros((self.horizon, nu))
        lxx = states.new_zeros((self.horizon, nx, nx))
        luu = states.new_zeros((self.horizon, nu, nu))
        lux = states.new_zeros((self.horizon, nu, nx))
        ee, jacobian = fk_and_jacobian_batch(states[:, :6])
        identity_u = torch.eye(nu, dtype=self.dtype)
        identity_v = torch.eye(6, dtype=self.dtype)
        for step in range(self.horizon):
            error = ee[step] - self.waypoints[absolute_step + step]
            j = jacobian[step]
            lx[step, :6] = 2.0 * self.position_weight * (j.T @ error)
            lxx[step, :6, :6] = 2.0 * self.position_weight * (j.T @ j)
            lx[step, 6:] = 2.0 * self.velocity_weight * states[step, 6:]
            lxx[step, 6:, 6:] = 2.0 * self.velocity_weight * identity_v
            lu[step] = 2.0 * self.action_weight * controls[step]
            luu[step] = 2.0 * self.action_weight * identity_u
            if step == 0:
                lu[step] += 2.0 * self.smooth_weight * (
                    controls[step] - previous_action
                )
                luu[step] += 2.0 * self.smooth_weight * identity_u
            _, obstacle_gradient, obstacle_hessian = self._obstacle_terms(
                ee[step], absolute_step, step
            )
            lx[step, :6] += j.T @ obstacle_gradient
            lxx[step, :6, :6] += j.T @ obstacle_hessian @ j

        terminal_error = (
            ee[-1] - self.waypoints[absolute_step + self.horizon]
        )
        terminal_j = jacobian[-1]
        terminal_x = states.new_zeros(nx)
        terminal_xx = states.new_zeros((nx, nx))
        terminal_x[:6] = 2.0 * self.terminal_weight * (
            terminal_j.T @ terminal_error
        )
        terminal_xx[:6, :6] = 2.0 * self.terminal_weight * (
            terminal_j.T @ terminal_j
        )
        _, obstacle_gradient, obstacle_hessian = self._obstacle_terms(
            ee[-1], absolute_step, self.horizon
        )
        terminal_x[:6] += 4.0 * terminal_j.T @ obstacle_gradient
        terminal_xx[:6, :6] += 4.0 * terminal_j.T @ obstacle_hessian @ terminal_j
        return lx, lu, lxx, luu, lux, terminal_x, terminal_xx

    def _linearize(self, states: torch.Tensor,
                   controls: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Central-difference every horizon stage in one batched model call."""
        nx, nu = 12, 6
        x_epsilon, u_epsilon = 1e-5, 1e-4
        batch_size = 2 * (nx + nu)
        state_batch = states[:-1, None, :].expand(-1, batch_size, -1).clone()
        control_batch = controls[:, None, :].expand(-1, batch_size, -1).clone()
        for index in range(nx):
            state_batch[:, 2 * index, index] += x_epsilon
            state_batch[:, 2 * index + 1, index] -= x_epsilon
        offset = 2 * nx
        for index in range(nu):
            control_batch[:, offset + 2 * index, index] += u_epsilon
            control_batch[:, offset + 2 * index + 1, index] -= u_epsilon
        next_batch = self._compiled_prediction(
            state_batch.reshape(-1, nx), control_batch.reshape(-1, nu)
        ).clone().reshape(self.horizon, batch_size, nx)
        a = torch.stack([
            (next_batch[:, 2 * index] - next_batch[:, 2 * index + 1])
            / (2.0 * x_epsilon) for index in range(nx)
        ], dim=2)
        b = torch.stack([
            (next_batch[:, offset + 2 * index]
             - next_batch[:, offset + 2 * index + 1]) / (2.0 * u_epsilon)
            for index in range(nu)
        ], dim=2)
        return a, b

    def _box_qp(self, hessian: torch.Tensor, gradient: torch.Tensor,
                cross: torch.Tensor, lower: torch.Tensor,
                upper: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve the tiny box-constrained local control QP by an active set."""
        step = torch.linalg.solve(hessian, -gradient).clamp(lower, upper)
        previous_free = None
        for _ in range(12):
            qp_gradient = gradient + hessian @ step
            free = ((step > lower + 1e-9) & (step < upper - 1e-9))
            free |= (step <= lower + 1e-9) & (qp_gradient < 0.0)
            free |= (step >= upper - 1e-9) & (qp_gradient > 0.0)
            if previous_free is not None and torch.equal(free, previous_free):
                break
            previous_free = free.clone()
            fixed = ~free
            if free.any():
                free_hessian = hessian[free][:, free]
                right_hand_side = -gradient[free]
                if fixed.any():
                    right_hand_side -= hessian[free][:, fixed] @ step[fixed]
                step[free] = torch.linalg.solve(
                    free_hessian, right_hand_side
                )
            step = step.clamp(lower, upper)

        feedback = cross.new_zeros(cross.shape)
        if previous_free is None:
            previous_free = torch.zeros_like(step, dtype=torch.bool)
        if previous_free.any():
            feedback[previous_free] = torch.linalg.solve(
                hessian[previous_free][:, previous_free],
                -cross[previous_free],
            )
        return step, feedback

    def _backward_pass(self, derivatives, a: torch.Tensor, b: torch.Tensor,
                       controls: torch.Tensor,
                       regularization: float) -> tuple[torch.Tensor, torch.Tensor]:
        lx, lu, lxx, luu, _lux, value_x, value_xx = derivatives
        feedforward = torch.empty((self.horizon, 6), dtype=self.dtype)
        feedback = torch.empty((self.horizon, 6, 12), dtype=self.dtype)
        identity = torch.eye(6, dtype=self.dtype)
        for step in range(self.horizon - 1, -1, -1):
            qx = lx[step] + a[step].T @ value_x
            qu = lu[step] + b[step].T @ value_x
            qxx = lxx[step] + a[step].T @ value_xx @ a[step]
            quu = luu[step] + b[step].T @ value_xx @ b[step]
            qux = b[step].T @ value_xx @ a[step]
            quu_regularized = 0.5 * (quu + quu.T) + regularization * identity
            # Cholesky is also an explicit positive-definiteness check before
            # the active-set box QP uses direct solves on its principal blocks.
            torch.linalg.cholesky(quu_regularized)
            feedforward[step], feedback[step] = self._box_qp(
                quu_regularized, qu, qux,
                self.dynamics.u_min - controls[step],
                self.dynamics.u_max - controls[step],
            )
            k, gain = feedforward[step], feedback[step]
            value_x = (qx + gain.T @ quu @ k + gain.T @ qu
                       + qux.T @ k)
            value_xx = (qxx + gain.T @ quu @ gain
                        + gain.T @ qux + qux.T @ gain)
            value_xx = 0.5 * (value_xx + value_xx.T)
        return feedforward, feedback

    def _optimize(self, state: torch.Tensor, controls: torch.Tensor,
                  absolute_step: int,
                  previous_action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nominal_states = self._rollout(state, controls)
        nominal_cost = self._cost(
            nominal_states, controls, absolute_step, previous_action
        )
        regularization = self._regularization
        for _ in range(self.max_iterations):
            derivatives = self._cost_derivatives(
                nominal_states, controls, absolute_step, previous_action
            )
            a, b = self._linearize(nominal_states, controls)
            try:
                feedforward, feedback = self._backward_pass(
                    derivatives, a, b, controls, regularization
                )
            except torch.linalg.LinAlgError:
                regularization = min(regularization * 10.0, 1e8)
                continue

            accepted = False
            for alpha in (1.0, 0.5, 0.25, 0.1, 0.05):
                candidate_states = [state]
                candidate_controls = []
                for horizon_step in range(self.horizon):
                    delta_state = candidate_states[-1] - nominal_states[horizon_step]
                    control = (controls[horizon_step]
                               + alpha * feedforward[horizon_step]
                               + feedback[horizon_step] @ delta_state)
                    control = control.clamp(
                        self.dynamics.u_min, self.dynamics.u_max
                    )
                    candidate_controls.append(control)
                    candidate_states.append(self._compiled_prediction(
                        candidate_states[-1].unsqueeze(0), control.unsqueeze(0)
                    )[0].clone())
                candidate_controls_t = torch.stack(candidate_controls)
                candidate_states_t = torch.stack(candidate_states)
                candidate_cost = self._cost(
                    candidate_states_t, candidate_controls_t,
                    absolute_step, previous_action
                )
                if torch.isfinite(candidate_cost) and candidate_cost < nominal_cost:
                    improvement = (nominal_cost - candidate_cost).item()
                    controls = candidate_controls_t
                    nominal_states = candidate_states_t
                    nominal_cost = candidate_cost
                    regularization = max(regularization * 0.5, 1e-6)
                    accepted = True
                    if improvement < 1e-5 * max(1.0, abs(nominal_cost.item())):
                        self._regularization = regularization
                        return controls, nominal_states
                    break
            if not accepted:
                regularization = min(regularization * 10.0, 1e8)
        self._regularization = regularization
        return controls, nominal_states

    def run(self, q0: np.ndarray, has_disturbance: bool = False,
            use_dob: bool = False) -> dict:
        state = torch.cat((torch.as_tensor(q0, dtype=self.dtype),
                           torch.zeros(6, dtype=self.dtype)))
        feedforward = self._feedforward(q0)
        controls = feedforward[:self.horizon].clone()
        previous_action = torch.zeros(6, dtype=self.dtype)
        self.dob.reset()
        print("[iLQR] prediction kernels warming up outside control loop...")
        warmup_start = time.perf_counter()
        warm_rollout = self._rollout(state, controls)
        warm_states = state.expand(self.horizon, -1)
        self._linearize(
            torch.cat((warm_states, warm_states[-1:])), controls
        )
        self._cost(warm_rollout, controls, 0, previous_action)
        # Exercise the complete optimization path once so lazy linear-algebra
        # initialization and any remaining compiled shapes are not charged to
        # the first real control step.  Discard both the solution and adaptive
        # regularization so the episode still starts from the original seed.
        initial_regularization = self._regularization
        self._optimize(state, controls.clone(), 0, previous_action)
        self._regularization = initial_regularization
        self.warmup_time = time.perf_counter() - warmup_start
        print(f"[iLQR] kernels ready: {self.warmup_time:.2f} s")
        q_history = [q0.copy()]
        ee_history = [fk_batch(state[:6].unsqueeze(0))[0].numpy()]
        control_history, error_history, solve_times_raw = [], [], []
        estimate_history = [np.zeros(6)]
        disturbance_history = [np.zeros(6)]
        residual_history = [np.zeros(6)]
        model_torque_history = [np.zeros(6)]

        print(f"iLQR simulation: {self.original_count} steps "
              f"(N={self.horizon}, max_iter={self.max_iterations}, "
              f"DOB={use_dob}, disturbance={has_disturbance}, "
              f"obstacle={self.obstacle_mode})")
        for simulation_step in range(self.original_count):
            solve_start = time.perf_counter()
            controls, _ = self._optimize(
                state, controls, simulation_step, previous_action
            )
            solve_times_raw.append(time.perf_counter() - solve_start)
            ilqr_torque = controls[0].clone()
            previous_action = ilqr_torque.clone()
            commanded = ilqr_torque + self.dob.d_hat if use_dob else ilqr_torque
            applied = commanded.clamp(self.dynamics.u_min, self.dynamics.u_max)
            true_disturbance = sinusoidal_disturbance(
                simulation_step, self.device, self.dtype
            ) if has_disturbance else torch.zeros(6, dtype=self.dtype)
            effective = applied - true_disturbance
            previous_state = state.clone()
            state = self.dynamics.plant_step_effective(
                state.unsqueeze(0), effective.unsqueeze(0)
            )[0]
            if use_dob:
                self.dob.update(applied, previous_state[:6], previous_state[6:],
                                state[:6], state[6:])

            terminal_index = simulation_step + self.horizon
            controls = torch.cat((
                controls[1:], feedforward[terminal_index:terminal_index + 1]
            ))
            ee = fk_batch(state[:6].unsqueeze(0))[0]
            error_mm = torch.linalg.vector_norm(
                ee - self.waypoints[simulation_step + 1]
            ).item() * 1e3
            q_history.append(state[:6].numpy().copy())
            ee_history.append(ee.numpy().copy())
            control_history.append(applied.numpy().copy())
            error_history.append(error_mm)
            estimate_history.append(self.dob.d_hat.numpy().copy())
            disturbance_history.append(true_disturbance.numpy().copy())
            residual_history.append(self.dob.residual.numpy().copy())
            model_torque_history.append(self.dob.tau_model.numpy().copy())
            if (simulation_step + 1) % 50 == 0:
                print(f"  step {simulation_step + 1:>3}/{self.original_count}  "
                      f"EE err={error_mm:.1f} mm  "
                      f"solve={solve_times_raw[-1] * 1e3:.2f} ms")

        solve_times_raw = np.asarray(solve_times_raw)
        solve_times = solve_times_raw.copy()
        solve_times[:min(TIMING_WARMUP_STEPS, len(solve_times))] = np.nan
        return {
            "q": np.asarray(q_history), "ee": np.asarray(ee_history),
            "u": np.asarray(control_history),
            "err_mm": np.asarray(error_history),
            "err": np.asarray(error_history) / 1e3,
            "d_hat": np.asarray(estimate_history),
            "d_true": np.asarray(disturbance_history),
            "dob_residual": np.asarray(residual_history),
            "dob_tau_model": np.asarray(model_torque_history),
            "solve_t": solve_times, "solve_t_raw": solve_times_raw,
            "elapsed": np.asarray(np.sum(solve_times_raw)),
            "warmup_t": np.asarray(self.warmup_time),
            "has_disturbance": np.asarray(has_disturbance),
            "use_dob": np.asarray(use_dob),
            "use_obstacle": np.asarray(self.obstacle_mode != "false"),
            "obstacle_mode": np.asarray(self.obstacle_mode),
            "controller": np.asarray("ilqr"),
            "dynamics_model": np.asarray("M(q)qdd+C(q,qd)qd+g(q)+Bqd=tau"),
            "ilqr_horizon": np.asarray(self.horizon),
            "ilqr_max_iterations": np.asarray(self.max_iterations),
            "ilqr_position_weight": np.asarray(self.position_weight),
            "ilqr_terminal_weight": np.asarray(self.terminal_weight),
            "ilqr_velocity_weight": np.asarray(self.velocity_weight),
            "ilqr_action_weight": np.asarray(self.action_weight),
            "ilqr_smooth_weight": np.asarray(self.smooth_weight),
            "ilqr_static_exp_weight": np.asarray(self.static_exp_weight),
            "ilqr_dynamic_exp_weight": np.asarray(self.dynamic_exp_weight),
            **({"cross_paths": self.dynamic_paths.numpy()}
               if self.dynamic_paths is not None else {}),
            **({"static_obstacle_centers": self.static_centers.numpy()}
               if self.static_centers is not None else {}),
            **({"ilqr_warm_start_waypoints": self.warm_start_waypoints}
               if self.static_centers is not None else {}),
        }


def run_ilqr(reference: np.ndarray, has_disturbance: bool = False,
             use_dob: bool = False,
             dynamic_paths: np.ndarray | None = None,
             static_centers: np.ndarray | None = None, *,
             horizon: int = HORIZON,
             max_iterations: int = ILQR_MAX_ITER,
             position_weight: float = ILQR_W_POS,
             terminal_weight: float = ILQR_W_TERM,
             velocity_weight: float = ILQR_W_VEL,
             action_weight: float = ILQR_W_ACT,
             smooth_weight: float | None = None,
             static_exp_weight: float = ILQR_STATIC_EXP_WEIGHT,
             dynamic_exp_weight: float = ILQR_CROSS_EXP_WEIGHT) -> dict:
    hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0 = ik_np(reference[0], hint)
    return UR5ILQRTracker(
        reference, dynamic_paths=dynamic_paths, static_centers=static_centers,
        horizon=horizon, max_iterations=max_iterations,
        position_weight=position_weight, terminal_weight=terminal_weight,
        velocity_weight=velocity_weight, action_weight=action_weight,
        smooth_weight=smooth_weight, static_exp_weight=static_exp_weight,
        dynamic_exp_weight=dynamic_exp_weight,
    ).run(q0, has_disturbance=has_disturbance, use_dob=use_dob)
