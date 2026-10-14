"""MPPI controller composed with UR5 dynamics, costs, and an optional DOB."""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import torch
from pytorch_mppi import MPPI

from config import (DT, HORIZON, HORIZON_OBS, LAMBDA,
                    MPPI_MEAN_UPDATE_ALPHA, MPPI_W_POS, MPPI_W_TERM,
                    N_SAMPLES, N_SAMPLES_OBS, NOISE_VAR,
                    TEMPORAL_NOISE_RHO, W_SMOOTH, W_SMOOTH_OBS)
from costs.obstacle_cost import ObstacleCost
from costs.tracking_cost import TrackingCost
from dynamics.ur5_dynamics import UR5Dynamics
from kinematics.ur5_kinematics import fk_batch, ik_trajectory_np
from observers.dob import TorchDisturbanceObserver
from utils.disturbance import sinusoidal_disturbance

torch.set_float32_matmul_precision("high")


class TemporalNoiseMPPI(MPPI):
    """MPPI with optional horizon-correlated control perturbations.

    For ``rho > 0``, every rollout receives one common Gaussian sample over
    the complete horizon plus the usual step-independent Gaussian samples::

        epsilon_t = sqrt(rho) * xi + sqrt(1 - rho) * zeta_t

    Both terms use the configured joint-space covariance.  Consequently each
    stage retains exactly the original marginal covariance, while distinct
    horizon stages have cross-covariance ``rho * noise_sigma``.

    The exact parent implementation is used at ``rho == 0``.  This preserves
    not only the distribution but also the existing random-number stream for
    fixed-seed regression compatibility.
    """

    def __init__(self, *args, temporal_noise_rho: float = 0.0, **kwargs):
        if not 0.0 <= temporal_noise_rho <= 1.0:
            raise ValueError("temporal_noise_rho must be in [0, 1]")
        # MPPI.__init__ samples the initial nominal sequence, so this must be
        # available before invoking the parent constructor.
        self.temporal_noise_rho = float(temporal_noise_rho)
        super().__init__(*args, **kwargs)

    def _sample_noise(self, shape):
        # Keep initialization and any future non-rollout sampling unchanged.
        # In particular, rho=0 delegates exactly to the upstream RNG path.
        if self.temporal_noise_rho == 0.0 or len(shape) != 2:
            return super()._sample_noise(shape)

        rollouts, horizon = shape
        common = torch.randn(
            rollouts, 1, self.nu, device=self.d, dtype=self.dtype
        )
        independent = torch.randn(
            rollouts, horizon, self.nu, device=self.d, dtype=self.dtype
        )
        standard_noise = (
            self.temporal_noise_rho ** 0.5 * common
            + (1.0 - self.temporal_noise_rho) ** 0.5 * independent
        )
        if self._diagonal_sigma:
            return standard_noise * self._noise_sigma_sqrt_diag + self.noise_mu
        return standard_noise @ self._noise_sigma_chol.T + self.noise_mu

    def set_diagonal_noise_variance(self, variance: torch.Tensor) -> None:
        """Update a diagonal proposal covariance without reallocating buffers."""
        if not self._diagonal_sigma:
            raise ValueError("adaptive variance requires diagonal noise_sigma")
        if variance.shape != (self.nu,) or torch.any(variance <= 0):
            raise ValueError("variance must contain one positive value per input")
        diagonal = self.noise_sigma.diagonal()
        diagonal.copy_(variance)
        self._noise_sigma_inv_diag.copy_(variance.reciprocal())
        self._noise_sigma_sqrt_diag.copy_(variance.sqrt())
        self.noise_sigma_inv.diagonal().copy_(self._noise_sigma_inv_diag)


def compose_nominal_sequence(feedforward: torch.Tensor,
                             residual: torch.Tensor,
                             disturbance_estimate: torch.Tensor,
                             aware_warm_start: bool) -> torch.Tensor:
    """Compose the actuator-command sampling mean without accumulating d_hat."""
    shift = disturbance_estimate if aware_warm_start else torch.zeros_like(
        disturbance_estimate
    )
    return feedforward + residual + shift


def recover_residual_sequence(optimal: torch.Tensor,
                              feedforward: torch.Tensor,
                              disturbance_estimate: torch.Tensor,
                              aware_warm_start: bool) -> torch.Tensor:
    """Return disturbance-free residual coordinates for the next warm start."""
    shift = disturbance_estimate if aware_warm_start else torch.zeros_like(
        disturbance_estimate
    )
    return optimal - feedforward - shift


def shift_control_sequence(optimal: torch.Tensor) -> torch.Tensor:
    """Receding-horizon warm start with a zero terminal control.

    This is the conventional absolute-control MPPI update used by the
    ``zero_initial_shift`` experiment: the first solve starts from all zeros,
    then each later solve starts from the previous optimum shifted by one
    stage.  It deliberately does not add inverse-dynamics feedforward.
    """
    shifted = torch.zeros_like(optimal)
    shifted[:-1].copy_(optimal[1:].detach())
    return shifted


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
                 temporal_noise_rho: float = TEMPORAL_NOISE_RHO,
                 disturbance_covariance_gain: float = 0.0,
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
        if not 0.0 <= temporal_noise_rho <= 1.0:
            raise ValueError("temporal_noise_rho must be in [0, 1]")
        self.temporal_noise_rho = float(temporal_noise_rho)
        if disturbance_covariance_gain < 0:
            raise ValueError("disturbance_covariance_gain must be nonnegative")
        self.disturbance_covariance_gain = float(disturbance_covariance_gain)
        self.dynamics = UR5Dynamics(device, dtype)
        # RNEA contains fixed six-link loops; compilation fuses them into a
        # small number of kernels and is essential for realtime rollout speed.
        self._compiled_rollout_dynamics = torch.compile(
            self.dynamics.rollout_effective,
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
        # Keep a fixed-address tensor so every rollout in one MPPI solve sees
        # the latest DOB estimate without forcing torch.compile to specialize
        # on a new Python object after each observer update.  The estimate is
        # held constant across the prediction horizon (zero-order hold).
        self._rollout_disturbance = torch.zeros(
            6, dtype=dtype, device=device
        )
        noise_sigma = torch.diag(torch.tensor(
            self.noise_variance, dtype=dtype, device=device
        ))
        self._base_noise_variance = noise_sigma.diagonal().clone()
        self.ctrl = TemporalNoiseMPPI(
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
            temporal_noise_rho=self.temporal_noise_rho,
        )

    @property
    def disturbance_estimate(self) -> torch.Tensor:
        return self.dob.d_hat

    def _stable_rollout_dynamics(self, state, action, rollout_step):
        """Roll out sampled actuator torque under the current DOB estimate.

        The plant convention is ``tau_effective = tau_applied - disturbance``.
        Passing that same effective torque to the prediction model makes MPPI
        include the required compensation in its optimized actuator command.

        ``reduce-overhead`` may reuse output buffers on CUDA. MPPI passes each
        output to a separately compiled FK function before the next rollout,
        so the clone intentionally remains outside the compiled function.
        """
        applied = action.clamp(self.dynamics.u_min, self.dynamics.u_max)
        return self._compiled_rollout_dynamics(
            state, applied - self._rollout_disturbance, rollout_step
        ).clone()

    def _set_rollout_disturbance(self, use_dob: bool) -> None:
        """Snapshot the current estimate for one MPPI prediction solve."""
        if use_dob:
            self._rollout_disturbance.copy_(self.dob.d_hat.detach())
        else:
            self._rollout_disturbance.zero_()

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
        self.joint_reference = torch.cat((
            q_ref, q_ref[-1:].expand(self.horizon + 2, -1)
        ), dim=0)
        padding = torque[-1:].expand(self.horizon + 2, -1)
        return torch.cat((torque, padding), dim=0).clamp(
            self.dynamics.u_min, self.dynamics.u_max
        )

    def _warm_up_controller(self, state: torch.Tensor,
                            feedforward: torch.Tensor,
                            sampling_center_mode: str) -> None:
        """Compile fixed-shape CUDA kernels before the realtime loop starts."""
        cpu_rng = torch.get_rng_state()
        cuda_rng = (torch.cuda.get_rng_state(self.device)
                    if self.device.type == "cuda" else None)
        original_u = self.ctrl.U.clone()
        indices = torch.arange(self.horizon, device=self.device) + self.step
        if sampling_center_mode == "feedforward_residual":
            self.ctrl.U.copy_(feedforward[indices])
        else:
            self.ctrl.U.zero_()
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
            has_disturbance: bool = False, use_dob: bool = False,
            dob_compensation_mode: str = "output",
            sampling_center_mode: str = "feedforward_residual",
            disturbance_estimate_mode: str = "dob",
            residual_terminal_mode: str = "zero",
            disturbance_function: Callable[[int, torch.device, torch.dtype],
                                           torch.Tensor] | None = None):
        """Run one episode and return state, control, error, and DOB histories."""
        if disturbance_estimate_mode not in {"dob", "oracle"}:
            raise ValueError("disturbance_estimate_mode must be dob or oracle")
        if residual_terminal_mode not in {"zero", "hold"}:
            raise ValueError("residual_terminal_mode must be zero or hold")
        if disturbance_estimate_mode == "oracle" and (not use_dob or dob_compensation_mode not in {"rollout", "rollout_model_only"}):
            raise ValueError("oracle is restricted to model-only diagnostic experiments")
        # Descriptive experiment name; keep the legacy name compatible.
        if dob_compensation_mode == "rollout_model_only":
            dob_compensation_mode = "rollout"
        if dob_compensation_mode not in {
                "output", "rollout", "rollout_nominal_shift"}:
            raise ValueError(
                "dob_compensation_mode must be 'output', 'rollout', or "
                "'rollout_nominal_shift'"
            )
        if sampling_center_mode not in {
                "feedforward_residual", "zero_initial_shift"}:
            raise ValueError(
                "sampling_center_mode must be 'feedforward_residual' or "
                "'zero_initial_shift'"
            )
        use_rollout_disturbance = (
            use_dob and dob_compensation_mode in {
                "rollout", "rollout_nominal_shift"
            }
        )
        if (use_dob and dob_compensation_mode in {
                "rollout", "rollout_nominal_shift"}
                and sampling_center_mode != "feedforward_residual"):
            raise ValueError("DOB-aware experiments require feedforward_residual")
        use_aware_warm_start = (
            use_dob and dob_compensation_mode == "rollout_nominal_shift"
        )
        if disturbance_function is None:
            disturbance_function = sinusoidal_disturbance
        self.step = 0
        self.dob.reset()
        self._set_rollout_disturbance(use_rollout_disturbance)
        self.tracking_cost.previous_action.zero_()
        self._sync_cost_step()
        velocity = torch.zeros(6, dtype=self.dtype, device=self.device)
        state = torch.cat([q0.to(device=self.device, dtype=self.dtype), velocity])
        feedforward = self._feedforward_torque(state[:6])
        residual_sequence = torch.zeros_like(self.ctrl.U)
        shifted_control_sequence = torch.zeros_like(self.ctrl.U)

        print("[MPPI] rollout/cost kernels warming up outside control loop...")
        warmup_start = time.perf_counter()
        self._warm_up_controller(state, feedforward, sampling_center_mode)
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
        self.feedforward_history = []
        self.sampling_center_history = []
        self.optimized_residual_history = []
        self.mppi_command_history = []
        self.predicted_effective_torque_history = []
        self.actual_effective_torque_history = []
        self.disturbance_error_history = []
        self.saturation_history = []
        self.prediction_error_history = []
        self.estimate_used_history = []
        self.weight_ess_history = []
        self.sample_clamp_fraction_history = []
        self.active_noise_variance_history = []

        if use_dob and has_disturbance:
            mode = "DOB + disturbance"
        elif has_disturbance:
            mode = "disturbance (no DOB)"
        elif use_dob:
            mode = "DOB (no disturbance)"
        else:
            mode = "nominal"
        print(f"MPPI simulation [{mode}]: {self.episode_steps} steps "
              f"(K={self.n_samples}, T={self.horizon})")

        for simulation_step in range(self.episode_steps):
            indices = torch.arange(self.horizon, device=self.device) + self.step
            nominal_sequence = feedforward[indices]
            if disturbance_estimate_mode == "oracle":
                # Noncausal diagnostic: exact current disturbance, still ZOH
                # over the horizon. Never available to normal GUI execution.
                self.dob.d_hat = (
                    disturbance_function(self.step, self.device, self.dtype)
                    if has_disturbance else torch.zeros_like(self.dob.d_hat)
                )
            self._set_rollout_disturbance(use_rollout_disturbance)
            active_variance = self._base_noise_variance
            if use_rollout_disturbance and self.disturbance_covariance_gain > 0:
                active_variance = (
                    self._base_noise_variance
                    + self.disturbance_covariance_gain
                    * self._rollout_disturbance.square()
                )
            self.ctrl.set_diagonal_noise_variance(active_variance)
            if sampling_center_mode == "feedforward_residual":
                warm_start = compose_nominal_sequence(
                    nominal_sequence, residual_sequence,
                    self._rollout_disturbance, use_aware_warm_start,
                )
            else:
                warm_start = shifted_control_sequence
            sampling_center = warm_start.detach().clone()
            self.ctrl.U[:] = warm_start.clamp(
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
            weights = self.ctrl.omega
            self.weight_ess_history.append(
                float(weights.square().sum().reciprocal().item())
            )
            sampled_actions = self.ctrl.perturbed_action
            at_limit = torch.isclose(
                sampled_actions.abs(), self.dynamics.u_max,
                rtol=0.0, atol=1e-5,
            )
            self.sample_clamp_fraction_history.append(
                float(at_limit.float().mean().item())
            )
            self.active_noise_variance_history.append(
                active_variance.detach().cpu().numpy().copy()
            )
            optimized_residual = recover_residual_sequence(
                self.ctrl.U, nominal_sequence, self._rollout_disturbance,
                use_aware_warm_start,
            )
            current_residual = optimized_residual[0].detach().clone()
            if sampling_center_mode == "feedforward_residual":
                residual_sequence[:-1] = optimized_residual[1:].detach()
                if residual_terminal_mode == "hold":
                    residual_sequence[-1].copy_(optimized_residual[-1].detach())
                else:
                    residual_sequence[-1].zero_()
            else:
                shifted_control_sequence.copy_(
                    shift_control_sequence(self.ctrl.U)
                )
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

            true_disturbance = disturbance_function(
                self.step, self.device, self.dtype
            ) if has_disturbance else torch.zeros_like(mppi_action)
            # Rollout compensation is already included in the optimized MPPI
            # command.  Output compensation reproduces the original external
            # DOB structure and is retained for controlled comparisons.
            compensation = (
                self.dob.d_hat
                if use_dob and dob_compensation_mode == "output"
                else torch.zeros_like(mppi_action)
            )
            commanded = mppi_action + compensation
            applied = commanded.clamp(
                self.dynamics.u_min, self.dynamics.u_max
            )
            previous_state = state.clone()
            effective = applied - true_disturbance
            predicted_effective = mppi_action - (
                self._rollout_disturbance
                if use_rollout_disturbance else torch.zeros_like(mppi_action)
            )
            estimate_used = self.dob.d_hat.detach().clone()
            self.estimate_used_history.append(estimate_used.cpu().numpy().copy())
            # Diagnostic only, outside the timed MPPI solve. Use the same
            # integrator as rollout so this includes discretization mismatch.
            predicted_state = self.dynamics.rollout_effective(
                previous_state.unsqueeze(0), predicted_effective.unsqueeze(0)
            ).squeeze(0)
            state = self.dynamics.plant_step_effective(
                state.unsqueeze(0), effective.unsqueeze(0)
            ).squeeze(0)
            self.prediction_error_history.append(
                (state - predicted_state).detach().cpu().numpy().copy()
            )
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
            self.feedforward_history.append(
                nominal_sequence[0].detach().cpu().numpy().copy()
            )
            self.sampling_center_history.append(
                sampling_center[0].cpu().numpy().copy()
            )
            self.optimized_residual_history.append(
                current_residual.cpu().numpy().copy()
            )
            self.mppi_command_history.append(
                mppi_action.detach().cpu().numpy().copy()
            )
            self.predicted_effective_torque_history.append(
                predicted_effective.detach().cpu().numpy().copy()
            )
            self.actual_effective_torque_history.append(
                effective.detach().cpu().numpy().copy()
            )
            self.disturbance_error_history.append(
                (true_disturbance - estimate_used).cpu().numpy().copy()
            )
            self.saturation_history.append(bool(torch.any(torch.isclose(
                applied.abs(), self.dynamics.u_max, rtol=0.0, atol=1e-5
            )).item()))
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
