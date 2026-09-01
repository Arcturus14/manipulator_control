"""MPPI controller composed with UR5 dynamics, costs, and an optional DOB."""
from __future__ import annotations

import time

import numpy as np
import torch
from pytorch_mppi import MPPI

from config import (DT, HORIZON, HORIZON_OBS, LAMBDA,
                    MPPI_MEAN_UPDATE_ALPHA, MPPI_W_POS, MPPI_W_TERM,
                    N_SAMPLES, N_SAMPLES_OBS, NOISE_VAR, W_SMOOTH,
                    W_SMOOTH_OBS)
from costs.obstacle_cost import ObstacleCost
from costs.tracking_cost import TrackingCost
from dynamics.ur5_dynamics import UR5Dynamics
from kinematics.ur5_kinematics import fk_batch, ik_trajectory_np
from observers.dob import TorchDisturbanceObserver
from utils.disturbance import sinusoidal_disturbance

torch.set_float32_matmul_precision("high")


class UR5Tracker:
    """UR5 MPPI trajectory tracker with optional obstacle cost and DOB.

    The MPPI library owns sampling and importance weighting. This class only
    composes the plant rollout, stage/terminal costs, and episode simulation.
    """

    def __init__(self, waypoints: np.ndarray, device: torch.device,
                 dtype: torch.dtype = torch.float32,
                 obstacle_cost: ObstacleCost | None = None,
                 *, n_samples: int | None = None,
                 horizon: int | None = None,
                 temperature: float = LAMBDA,
                 mean_update_alpha: float = MPPI_MEAN_UPDATE_ALPHA,
                 noise_variance: tuple[float, ...] | list[float] = NOISE_VAR,
                 position_weight: float = MPPI_W_POS,
                 terminal_weight: float = MPPI_W_TERM,
                 smooth_weight: float | None = None):
        self.device, self.dtype = device, dtype
        self.step = 0
        self.obstacle_cost = obstacle_cost
        default_horizon = HORIZON_OBS if obstacle_cost else HORIZON
        self.horizon = default_horizon if horizon is None else horizon
        default_samples = N_SAMPLES_OBS if obstacle_cost else N_SAMPLES
        self.n_samples = default_samples if n_samples is None else n_samples
        if not 0 < mean_update_alpha <= 1:
            raise ValueError("mean_update_alpha must be in (0, 1]")
        self.mean_update_alpha = float(mean_update_alpha)
        if len(noise_variance) != 6 or any(value <= 0 for value in noise_variance):
            raise ValueError("noise_variance must contain six positive values")
        self.noise_variance = tuple(float(value) for value in noise_variance)
        self.dynamics = UR5Dynamics(device, dtype)
        # RNEA contains fixed six-link loops; compilation fuses them into a
        # small number of kernels and is essential for realtime rollout speed.
        self._compiled_rollout_dynamics = torch.compile(
            self.dynamics.rollout,
            options={
                # torch.linalg.solve is not CUDA-Graph-capture safe on the
                # target PyTorch/CUDA stack. Keep Inductor/Triton fusion while
                # disabling only graph capture for nonlinear rollouts.
                "triton.cudagraphs": False,
                "triton.cudagraph_trees": False,
            },
        )

        waypoint_tensor = torch.as_tensor(waypoints, dtype=dtype, device=device)
        self.episode_steps = len(waypoint_tensor)
        padding = waypoint_tensor[-1:].expand(self.horizon + 2, -1)
        self.waypoints = torch.cat([waypoint_tensor, padding])
        if smooth_weight is None:
            smooth_weight = W_SMOOTH_OBS if obstacle_cost else W_SMOOTH
        self.tracking_cost = TrackingCost(
            self.waypoints, self.horizon,
            self.dynamics.u_min, self.dynamics.u_max, smooth_weight,
            position_weight, terminal_weight,
        )
        if obstacle_cost is not None and obstacle_cost.horizon != self.horizon:
            raise ValueError("ObstacleCost horizon must match the obstacle MPPI horizon")

        self.dob = TorchDisturbanceObserver(self.dynamics, DT)
        noise_sigma = torch.diag(torch.tensor(
            self.noise_variance, dtype=dtype, device=device
        ))
        self.ctrl = MPPI(
            dynamics=self._stable_rollout_dynamics,
            running_cost=self._running_cost,
            nx=12,
            noise_sigma=noise_sigma,
            num_samples=self.n_samples,
            horizon=self.horizon,
            lambda_=temperature,
            device=device,
            terminal_state_cost=self._terminal_cost,
            u_min=self.dynamics.u_min,
            u_max=self.dynamics.u_max,
            step_dependent_dynamics=True,
        )

    @property
    def disturbance_estimate(self) -> torch.Tensor:
        return self.dob.d_hat

    def _stable_rollout_dynamics(self, state, action, rollout_step):
        """Return rollout state with storage independent of CUDA Graph buffers.

        ``reduce-overhead`` reuses output buffers on CUDA. MPPI passes each
        output to a separately compiled FK function before invoking the next
        rollout, so retaining the graph-owned tensor can otherwise trigger an
        overwritten-output error. The clone intentionally lives outside the
        compiled function.
        """
        return self._compiled_rollout_dynamics(
            state, action, rollout_step
        ).clone()

    def _sync_cost_step(self) -> None:
        self.tracking_cost.episode_step = self.step
        if self.obstacle_cost:
            self.obstacle_cost.episode_step = self.step

    def _running_cost(self, state, action, rollout_step):
        cost = self.tracking_cost.running(state, action, rollout_step)
        if self.obstacle_cost:
            cost += self.obstacle_cost.running(state, action, rollout_step)
        return cost

    def _terminal_cost(self, states, actions):
        cost = self.tracking_cost.terminal(states, actions)
        if self.obstacle_cost:
            cost += self.obstacle_cost.terminal(states, actions)
        return cost

    def _feedforward_torque(self, q0: torch.Tensor) -> torch.Tensor:
        """Create continuous-IK inverse-dynamics torque for the reference path."""
        waypoint_count = len(self.waypoints) - self.horizon - 2
        targets = self.waypoints[:waypoint_count].detach().cpu().numpy()
        joint_np = ik_trajectory_np(targets, q0.detach().cpu().numpy())
        velocity_np = np.gradient(joint_np, DT, axis=0, edge_order=2)
        acceleration_np = np.gradient(velocity_np, DT, axis=0, edge_order=2)
        q_ref = torch.as_tensor(joint_np, dtype=self.dtype, device=self.device)
        velocity_ref = torch.as_tensor(velocity_np, dtype=self.dtype, device=self.device)
        acceleration_ref = torch.as_tensor(acceleration_np, dtype=self.dtype, device=self.device)
        torque = self.dynamics.inverse_dynamics(q_ref, velocity_ref, acceleration_ref)
        torque = torque + self.dynamics.damping * velocity_ref
        padding = torque[-1:].expand(self.horizon + 2, -1)
        return torch.cat((torque, padding), dim=0).clamp(
            self.dynamics.u_min, self.dynamics.u_max
        )

    def _warm_up_controller(self, state: torch.Tensor,
                            feedforward: torch.Tensor) -> None:
        """Compile fixed-shape CUDA kernels before the realtime loop starts."""
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda" else None)
        original_u = self.ctrl.U.clone()
        indices = torch.arange(self.horizon, device=self.device) + self.step
        self.ctrl.U.copy_(feedforward[indices])
        if self.device.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        self._mppi_command(state)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.ctrl.U.copy_(original_u)
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state(cuda_rng, self.device)

    def _mppi_command(self, state: torch.Tensor) -> torch.Tensor:
        """Update and return MPPI input with optional CuRobo-style mean blending."""
        if self.mean_update_alpha == 1.0:
            return self.ctrl.command(state, shift_nominal_trajectory=False)
        warm_start = self.ctrl.U.clone()
        self.ctrl.command(state, shift_nominal_trajectory=False)
        self.ctrl.U.copy_(torch.lerp(
            warm_start, self.ctrl.U, self.mean_update_alpha
        ))
        return self.ctrl.get_action_sequence()[0]

    def run(self, q0: torch.Tensor, capture_samples: bool = False, n_show: int = 50,
            has_disturbance: bool = False, use_dob: bool = False):
        """Run one episode and return state, control, error, and DOB histories."""
        self.step = 0
        self.dob.reset()
        self.tracking_cost.previous_action.zero_()
        self._sync_cost_step()
        velocity = torch.zeros(6, dtype=self.dtype, device=self.device)
        state = torch.cat([q0.to(device=self.device, dtype=self.dtype), velocity])
        feedforward = self._feedforward_torque(state[:6])
        residual_sequence = torch.zeros_like(self.ctrl.U)

        print("[MPPI] rollout/cost kernels warming up outside control loop...")
        warmup_start = time.perf_counter()
        self._warm_up_controller(state, feedforward)
        print(f"[MPPI] kernels ready: {time.perf_counter() - warmup_start:.2f} s")

        q_history = [state[:6].cpu().numpy()]
        ee_history = [fk_batch(state[:6].unsqueeze(0))[0].cpu().numpy()]
        control_history, error_history = [], []
        sample_history = [] if capture_samples else None
        self.solve_time_history = []
        estimate_history = [self.dob.d_hat.cpu().numpy().copy()]
        disturbance_history = [np.zeros(6)]
        self.dob_residual_history = [np.zeros(6)]
        self.dob_model_torque_history = [np.zeros(6)]

        if use_dob and has_disturbance:
            mode = "DOB+외란"
        elif has_disturbance:
            mode = "외란(No-DOB)"
        elif use_dob:
            mode = "DOB(외란 없음)"
        else:
            mode = "기본"
        print(f"MPPI simulation [{mode}]: {self.episode_steps} steps "
              f"(K={self.n_samples}, T={self.horizon})")

        for simulation_step in range(self.episode_steps):
            indices = torch.arange(self.horizon, device=self.device) + self.step
            nominal_sequence = feedforward[indices]
            self.ctrl.U[:] = (nominal_sequence + residual_sequence).clamp(
                self.dynamics.u_min, self.dynamics.u_max
            )
            if self.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
                torch.cuda.synchronize(self.device)
            solve_start = time.perf_counter()
            mppi_action = self._mppi_command(state).clamp(
                self.dynamics.u_min, self.dynamics.u_max
            )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.solve_time_history.append(time.perf_counter() - solve_start)
            optimized_residual = self.ctrl.U - nominal_sequence
            residual_sequence[:-1] = optimized_residual[1:].detach()
            residual_sequence[-1].zero_()
            if capture_samples and self.ctrl.states is not None:
                q_all = self.ctrl.states[0, :, :, :6]
                ee_all = fk_batch(q_all.reshape(-1, 6)).reshape(
                    self.n_samples, self.horizon, 3
                )
                weights = self.ctrl.omega
                top = torch.argsort(weights, descending=True)[:n_show]
                normalized = weights / (weights.sum() + 1e-8)
                sample_history.append({
                    "ee": ee_all[top].cpu().numpy(),
                    "omega": weights[top].cpu().numpy(),
                    "ee_mean": (normalized[:, None, None] * ee_all).sum(dim=0).cpu().numpy(),
                })

            true_disturbance = sinusoidal_disturbance(
                self.step, self.device, self.dtype
            ) if has_disturbance else torch.zeros_like(mppi_action)
            commanded = mppi_action + self.dob.d_hat if use_dob else mppi_action
            applied = commanded.clamp(self.dynamics.u_min, self.dynamics.u_max)
            previous_state = state.clone()
            effective = (applied - true_disturbance).clamp(
                self.dynamics.u_min, self.dynamics.u_max
            )
            state = self.dynamics.plant_step(
                state.unsqueeze(0), effective.unsqueeze(0)
            ).squeeze(0)
            if use_dob:
                self.dob.update(
                    applied,
                    previous_state[:6], previous_state[6:],
                    state[:6], state[6:],
                )

            self.tracking_cost.previous_action = mppi_action.detach().clone()
            self.step += 1
            self._sync_cost_step()
            q = state[:6]
            ee = fk_batch(q.unsqueeze(0))[0]
            error_mm = torch.linalg.vector_norm(ee - self.waypoints[self.step]).item() * 1e3
            q_history.append(q.cpu().numpy())
            ee_history.append(ee.cpu().numpy())
            control_history.append(applied.cpu().numpy())
            error_history.append(error_mm)
            estimate_history.append(self.dob.d_hat.cpu().numpy().copy())
            disturbance_history.append(true_disturbance.cpu().numpy().copy())
            self.dob_residual_history.append(self.dob.residual.cpu().numpy().copy())
            self.dob_model_torque_history.append(self.dob.tau_model.cpu().numpy().copy())
            if (simulation_step + 1) % 50 == 0:
                print(f"  step {simulation_step + 1:>3}/{self.episode_steps}  "
                      f"EE err={error_mm:.1f} mm")

        print(f"Mean EE err: {np.mean(error_history):.1f} mm  "
              f"Max: {np.max(error_history):.1f} mm")
        return (np.asarray(q_history), np.asarray(ee_history),
                np.asarray(control_history), np.asarray(error_history), sample_history,
                np.asarray(estimate_history), np.asarray(disturbance_history))
