"""Opt-in execution optimization of the frozen final N0/N1 equations.

No sampling, horizon, temperature, filter, penalty or plant-limit retuning.
Compiled arithmetic can change float32 rounding; validate closed-loop results.
The original implementations and GUI defaults remain independent references.
"""
import time
import types
from collections.abc import Callable

import numpy as np
import torch

from config import Q_MIN, Q_MAX, W_ACT, W_VEL
from controllers.mppi_dob_controller import (
    UR5Tracker, compose_nominal_sequence, recover_residual_sequence,
    shift_control_sequence,
)
from controllers.research_baseline_controller import asynchronous_forward_dynamics
from kinematics.ur5_kinematics import fk_batch
from kroc.position_input_headroom_study import PositionInputTracker
from utils.disturbance import sinusoidal_disturbance


COMPILE_OPTIONS = {'triton.cudagraphs': False, 'triton.cudagraph_trees': False}


class FastFinalTracker(PositionInputTracker):
    def __init__(self, *args, study_case='N0', fuse_costs=True,
                 compile_auxiliary=True, deferred_diagnostics=True, **kwargs):
        if study_case not in {'N0', 'N1'}:
            raise ValueError('FastFinalTracker supports the frozen N0/N1 equations only')
        kwargs.update(study_case=study_case, headroom_weight=0.,
                      headroom_fraction=.95, unclamped_diagnostic=False)
        self.clipping_weight = 1. if study_case == 'N1' else 0.
        self.fuse_costs = bool(fuse_costs)
        self.compile_auxiliary = bool(compile_auxiliary)
        self.deferred_diagnostics = bool(deferred_diagnostics)
        super().__init__(*args, **kwargs)
        if self.margin_config.mode != 'fixed':
            raise ValueError('Frozen final controller requires fixed state margins')
        if self.obstacle_cost is not None:
            raise ValueError('External obstacle costs have not been validated in this adapter')
        self.dynamics.forward_dynamics = types.MethodType(
            asynchronous_forward_dynamics, self.dynamics)
        if self.compile_auxiliary:
            for name in ('mass_matrix_and_bias', 'plant_step_effective', 'rollout_effective'):
                setattr(self.dynamics, name, torch.compile(getattr(self.dynamics, name),
                                                          options=COMPILE_OPTIONS))
            self.dob.update = torch.compile(self.dob.update, options=COMPILE_OPTIONS)
        self._fast_stage = torch.compile(self._stage_value, options=COMPILE_OPTIONS)
        self._fast_end = torch.compile(self._end_value, options=COMPILE_OPTIONS)
        self.diagnostic_export_time_s = 0.

    def _stage_value(self, state, action, reference, previous, compensation, first):
        c = self.tracking_cost
        value = c.position_weight * (fk_batch(state[:, :6])-reference).square().sum(-1)
        value = value + W_ACT * action.square().sum(-1)
        value = value + W_VEL * state[:, 6:].square().sum(-1)
        value = value + c.state_limit_cost(state)
        if first:
            value = value + self.boundary_weight * (action-previous).square().sum(-1)
        if self.collision_cost is not None:
            value = value + self.collision_geometry_cost(state)
        if self.clipping_weight:
            requested = action + compensation
            loss = requested-requested.clamp(self.dynamics.u_min, self.dynamics.u_max)
            value = value + self.clipping_weight * loss.square().sum(-1)
        return value

    def _end_value(self, states, actions, reference):
        c = self.tracking_cost
        last = states[..., -1, :].reshape(-1, 12)
        value = c.terminal_weight * (fk_batch(last[:, :6])-reference).square().sum(-1)
        value = value + c.state_limit_cost(last)
        if self.collision_cost is not None:
            value = value + self.collision_geometry_cost(last)
        value = value.reshape(states.shape[:2])
        if self.horizon > 1:
            value = value + c.smooth_weight * (actions[..., 1:, :]-actions[..., :-1, :]).square().sum((-1, -2))/(self.horizon-1)
        return value

    def _running_cost(self, state, action, rollout_step):
        if not self.fuse_costs:
            value = PositionInputTracker._running_cost(self, state, action, rollout_step)
            if self.clipping_weight:
                requested = action+self._compensation_snapshot
                loss = requested-requested.clamp(self.dynamics.u_min, self.dynamics.u_max)
                value = value+self.clipping_weight*loss.square().sum(-1)
            return value
        index = self.tracking_cost.state_index(rollout_step)
        return self._fast_stage(state, action, self.waypoints[self.step+index],
            self.tracking_cost.previous_action, self._compensation_snapshot, rollout_step == 0)

    def _terminal_cost(self, states, actions):
        if not self.fuse_costs:
            return PositionInputTracker._terminal_cost(self, states, actions)
        return self._fast_end(states, actions, self.waypoints[self.step+self.horizon])

    def _mppi_command(self, state):
        # Keep the original sample diagnostics; do not calculate a disabled
        # headroom penalty over K*H*6 elements merely to multiply it by zero.
        action = UR5Tracker._mppi_command(self, state)
        requested = self.ctrl.perturbed_action+self._compensation_snapshot
        self.request_sample_fraction.append((requested.abs()>self.dynamics.u_max).any(-1).float().mean().detach())
        self.headroom_sample_cost.append(requested.new_zeros(()))
        return action

    def _warm_up_controller(self, state, feedforward, sampling_center_mode):
        super()._warm_up_controller(state, feedforward, sampling_center_mode)
        if self.compile_auxiliary:
            self.dynamics.mass_matrix_and_bias(state[None, :6], state[None, 6:])
            torque = feedforward[0][None, :]
            self.dynamics.plant_step_effective(state[None, :], torque)
            self.dynamics.rollout_effective(state[None, :], torque)
            fields = ('d_hat', 'residual', 'tau_model', 'raw_acceleration', 'filtered_acceleration')
            saved = {key: getattr(self.dob, key).clone() for key in fields}
            self.dob.update(torque[0], state[:6], state[6:], state[:6], state[6:])
            for key, value in saved.items():
                setattr(self.dob, key, value)
            torch.cuda.synchronize(self.device)

    def run(self, *args, **kwargs):
        if kwargs.get('dob_compensation_mode', 'output') != 'output':
            raise ValueError('Final optimized adapter requires output compensation')
        self.request_sample_fraction.clear()
        self.headroom_sample_cost.clear()
        self.sensor_model.reset()
        self.measured_state_history.clear()
        self.margin_history.clear()
        self.margins.reset()
        self.diagnostic_export_time_s = 0.
        runner = self._run_deferred if self.deferred_diagnostics else UR5Tracker.run.__get__(self)
        return runner(*args, **kwargs)

    def export_sensor_data(self):
        result = super().export_sensor_data()
        result.update(fast_final_fused_costs=np.asarray(self.fuse_costs),
                      fast_final_compile_auxiliary=np.asarray(self.compile_auxiliary),
                      fast_final_deferred_diagnostics=np.asarray(self.deferred_diagnostics),
                      diagnostic_export_time_s=np.asarray(self.diagnostic_export_time_s))
        if self.study_case == 'N1':
            result.update(n1_cost_version=np.asarray('actual_clipping_loss_Nm_squared'),
                          n1_clipping_weight=np.asarray(self.clipping_weight),
                          n1_cost_activation_fraction=np.asarray(1.))
        return result

    # Execution loop mirrors UR5Tracker.run; only diagnostic storage/transfers
    # are deferred. Original reference remains intact for equivalence checks.
    def _run_deferred(self, q0: torch.Tensor, capture_samples: bool = False, n_show: int = 50,
            has_disturbance: bool = False, use_dob: bool = False,
            dob_compensation_mode: str = "output",
            sampling_center_mode: str = "feedforward_residual",
            disturbance_estimate_mode: str = "dob",
            residual_terminal_mode: str = "zero",
            initial_velocity: torch.Tensor | None = None,
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
                "output", "rollout", "rollout_nominal_shift",
                "estimate_only"}:
            raise ValueError(
                "dob_compensation_mode must be 'output', 'rollout', or "
                "'rollout_nominal_shift', or 'estimate_only'"
            )
        if sampling_center_mode not in {
                "feedforward_residual", "zero_initial_shift", "feedforward_initial_shift"}:
            raise ValueError(
                "sampling_center_mode must be 'feedforward_residual' or "
                "'zero_initial_shift', or 'feedforward_initial_shift'"
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
        velocity = (torch.zeros(6, dtype=self.dtype, device=self.device)
                    if initial_velocity is None else initial_velocity.to(
                        device=self.device, dtype=self.dtype))
        if velocity.shape != (6,) or not torch.isfinite(velocity).all():
            raise ValueError("initial_velocity must contain six finite values")
        state = torch.cat([q0.to(device=self.device, dtype=self.dtype), velocity])
        measured_state = self._measure_state(state)
        feedforward = self._feedforward_torque(state[:6])
        residual_sequence = torch.zeros_like(self.ctrl.U)
        shifted_control_sequence = torch.zeros_like(self.ctrl.U)
        if sampling_center_mode == "feedforward_initial_shift":
            # Initialize once; subsequent solves shift absolute optimal torques.
            shifted_control_sequence.copy_(feedforward[:self.horizon])

        print("[MPPI] rollout/cost kernels warming up outside control loop...")
        # Reset once per episode. A fixed proposal does not need an identical
        # covariance validation/copy (and CUDA host sync) every control step.
        self.ctrl.set_diagonal_noise_variance(self._base_noise_variance)
        warmup_start = time.perf_counter()
        self._warm_up_controller(state, feedforward, sampling_center_mode)
        print(f"[MPPI] kernels ready: {time.perf_counter() - warmup_start:.2f} s")

        q_history = [state[:6].cpu().numpy()]
        velocity_history = [state[6:].cpu().numpy()]
        ee_history = [fk_batch(state[:6].unsqueeze(0))[0].cpu().numpy()]
        control_history, error_history = [], []
        sample_history = [] if capture_samples else None
        self.solve_time_history = []
        self.control_cycle_time_history = []
        estimate_history = [self.dob.d_hat.cpu().numpy().copy()]
        disturbance_history = [np.zeros(6)]
        self.dob_residual_history = [np.zeros(6)]
        self.dob_model_torque_history = [np.zeros(6)]
        self.dob_raw_acceleration_history = [np.zeros(6)]
        self.dob_filtered_acceleration_history = [np.zeros(6)]
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
        self.joint_position_violation_history = []
        self.joint_velocity_violation_history = []
        q_min = torch.as_tensor(Q_MIN, dtype=self.dtype, device=self.device)
        q_max = torch.as_tensor(Q_MAX, dtype=self.dtype, device=self.device)

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

        deferred_records = []
        for simulation_step in range(self.episode_steps):
            cycle_start = time.perf_counter()
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
            self.ctrl.U[:] = self._bound_nominal_action(warm_start)
            if self.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()
                torch.cuda.synchronize(self.device)
            solve_start = time.perf_counter()
            self._prepare_solve(measured_state, sampling_center)
            mppi_action = self._bound_nominal_action(self._mppi_command(measured_state))
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.solve_time_history.append(time.perf_counter() - solve_start)
            weights = self.ctrl.omega
            self.weight_ess_history.append(
                weights.square().sum().reciprocal().detach().clone()
            )
            sampled_actions = self.ctrl.perturbed_action
            at_limit = torch.isclose(
                sampled_actions.abs(), self.dynamics.u_max,
                rtol=0.0, atol=1e-5,
            )
            self.sample_clamp_fraction_history.append(
                at_limit.float().mean().detach().clone()
            )
            self.active_noise_variance_history.append(
                active_variance.detach().clone()
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
                    shift_control_sequence(self.ctrl.U, terminal_mode=(
                        "hold" if sampling_center_mode == "feedforward_initial_shift" else "zero"))
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
                    "ee": ee_all[top].detach().clone(),
                    "omega": weights[top].detach().clone(),
                    "ee_mean": (normalized[:, None, None] * ee_all).sum(dim=0).detach().clone(),
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
            applied = self._apply_actuator_limits(commanded)
            previous_state = state.clone()
            effective = applied - true_disturbance
            predicted_effective = self._prediction_torque(mppi_action)
            estimate_used = self.dob.d_hat.detach().clone()
            self.estimate_used_history.append(estimate_used.detach().clone())
            # Diagnostic only, outside the timed MPPI solve. Use the same
            # integrator as rollout so this includes discretization mismatch.
            predicted_state = self.dynamics.rollout_effective(
                previous_state.unsqueeze(0), predicted_effective.unsqueeze(0)
            ).squeeze(0)
            state = self.dynamics.plant_step_effective(
                state.unsqueeze(0), effective.unsqueeze(0)
            ).squeeze(0)
            next_measured_state = self._measure_state(state)
            self.prediction_error_history.append(
                (state - predicted_state).detach().clone()
            )
            if use_dob:
                self.dob.update(
                    applied,
                    measured_state[:6], measured_state[6:],
                    next_measured_state[:6], next_measured_state[6:],
                )
            measured_state = next_measured_state

            self.tracking_cost.previous_action = mppi_action.detach().clone()
            self.step += 1
            self._sync_cost_step()
            q = state[:6]
            ee = fk_batch(q.unsqueeze(0))[0]
            # One host transfer for the unchanged per-step diagnostic fields,
            # instead of synchronizing after each small .cpu()/.item() call.
            vector_fields = (q, state[6:], applied, nominal_sequence[0],
                sampling_center[0], current_residual, mppi_action,
                predicted_effective, effective, true_disturbance-estimate_used,
                self.dob.d_hat, true_disturbance, self.dob.residual,
                self.dob.tau_model, self.dob.raw_acceleration,
                self.dob.filtered_acceleration)
            scalars = torch.stack((
                torch.linalg.vector_norm(ee-self.waypoints[self.step]),
                torch.any((q<q_min)|(q>q_max)),
                torch.any(state[6:].abs()>self.tracking_cost.dq_max),
                torch.any(torch.isclose(applied.abs(),self.dynamics.u_max,rtol=0.,atol=1e-5))))
            # Retain a snapshot, not a reference to mutable controller buffers.
            deferred_records.append(torch.cat((*vector_fields, ee, scalars)).detach())
            # Honest full-cycle latency: include completion of queued plant,
            # DOB and diagnostics, even though their host copies are deferred.
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            self.control_cycle_time_history.append(time.perf_counter() - cycle_start)
            if (simulation_step + 1) % 50 == 0:
                print(f"  step {simulation_step + 1:>3}/{self.episode_steps}")

        export_start = time.perf_counter()
        records = torch.stack(deferred_records).cpu().numpy()
        for packed in records:
            vectors = packed[:96].reshape(16,6)
            error_mm = float(packed[99])*1e3
            q_history.append(vectors[0]);velocity_history.append(vectors[1])
            self.joint_position_violation_history.append(bool(packed[100]))
            self.joint_velocity_violation_history.append(bool(packed[101]))
            ee_history.append(packed[96:99]);control_history.append(vectors[2])
            error_history.append(error_mm)
            histories = (self.feedforward_history,self.sampling_center_history,
                self.optimized_residual_history,self.mppi_command_history,
                self.predicted_effective_torque_history,self.actual_effective_torque_history,
                self.disturbance_error_history,estimate_history,disturbance_history,
                self.dob_residual_history,self.dob_model_torque_history,
                self.dob_raw_acceleration_history,self.dob_filtered_acceleration_history)
            for history, value in zip(histories,vectors[3:]):
                history.append(value)
            self.saturation_history.append(bool(packed[102]))
        for name in ('weight_ess_history', 'sample_clamp_fraction_history',
                     'active_noise_variance_history', 'estimate_used_history',
                     'prediction_error_history'):
            array = torch.stack(getattr(self, name)).cpu().numpy()
            setattr(self, name, list(array))
        if capture_samples:
            sample_history = [{key: value.cpu().numpy() for key, value in row.items()}
                              for row in sample_history]
        self.diagnostic_export_time_s = time.perf_counter()-export_start

        print(f"Mean EE err: {np.mean(error_history):.1f} mm  "
              f"Max: {np.max(error_history):.1f} mm")
        self.velocity_history = velocity_history
        return (np.asarray(q_history), np.asarray(ee_history),
                np.asarray(control_history), np.asarray(error_history), sample_history,
                np.asarray(estimate_history), np.asarray(disturbance_history))
