"""Isolated C3 study on the frozen pre-input-study baseline.

v is an unconstrained nominal/effective input, not actuator torque. Only the
physical post-DOB command is hard-clamped. No dynamic input tightening.
Existing effort/smoothness costs remain in v coordinates for every arm.
"""
import numpy as np
import torch

from config import W_ACT
from kroc.finalize_pre_input_baseline import BoundaryPaperTracker


CASES = {
    'Base': dict(fixed_clip=True, saturation_rollout=False, headroom=False),
    'N0': dict(fixed_clip=False, saturation_rollout=False, headroom=False),
    'N1': dict(fixed_clip=False, saturation_rollout=False, headroom=True),
    'S0': dict(fixed_clip=False, saturation_rollout=True, headroom=False),
    'S1': dict(fixed_clip=False, saturation_rollout=True, headroom=True),
}


def headroom_cost(requested, limit, fraction, weight):
    excess = torch.relu(requested.abs() - fraction * limit)
    return weight * (excess / ((1. - fraction) * limit)).square().sum(dim=-1)


def saturation_prediction(nominal, compensation, lower, upper):
    requested = nominal + compensation
    # Algebraically clip(v+c)-c, but preserve v BIT-FOR-BIT when no clamp
    # changes the request. Avoid cancellation roundoff masquerading as a
    # model effect in low-ESS MPPI comparisons.
    return nominal + (requested.clamp(lower, upper) - requested)


def unbounded_action(action):
    return action


class InputHeadroomTracker(BoundaryPaperTracker):
    def __init__(self, *args, study_case='N1', headroom_weight=10.,
                 headroom_fraction=.85, **kwargs):
        if study_case not in CASES:
            raise ValueError('Unknown input-headroom case')
        if not np.isfinite(headroom_weight) or headroom_weight < 0:
            raise ValueError('Headroom weight must be finite and nonnegative')
        if not 0. < headroom_fraction < 1.:
            raise ValueError('Headroom onset must lie in (0,1)')
        self.study_case = study_case
        self.study = CASES[study_case]
        self.headroom_weight = float(headroom_weight) if self.study['headroom'] else 0.
        self.headroom_fraction = float(headroom_fraction)
        super().__init__(*args, **kwargs)
        self._compensation_snapshot = torch.zeros(6, device=self.device, dtype=self.dtype)
        if not self.study['fixed_clip']:
            self.ctrl._bound_action = unbounded_action
        self.request_sample_fraction = []
        self.headroom_sample_cost = []

    def run(self, *args, **kwargs):
        if not kwargs.get('use_dob', False) or kwargs.get('dob_compensation_mode', 'output') != 'output':
            raise ValueError('This study requires measured post-DOB output compensation')
        self.request_sample_fraction.clear()
        self.headroom_sample_cost.clear()
        return super().run(*args, **kwargs)

    def _set_rollout_disturbance(self, use_dob):
        super()._set_rollout_disturbance(use_dob)
        self._compensation_snapshot.copy_(self.dob.d_hat.detach())

    def _bound_nominal_action(self, action):
        return super()._bound_nominal_action(action) if self.study['fixed_clip'] else action

    def _prediction_torque(self, action):
        if self.study['fixed_clip']:
            return super()._prediction_torque(action)
        if self.study['saturation_rollout']:
            return saturation_prediction(action, self._compensation_snapshot,
                                         self.dynamics.u_min, self.dynamics.u_max)
        return action

    def _running_cost(self, state, action, rollout_step):
        cost = super()._running_cost(state, action, rollout_step)
        if not self.study['fixed_clip']:
            # Remove hidden clipping ONLY from the existing nominal-input costs.
            # This is identical to the frozen cost wherever v is within bounds.
            bounded = action.clamp(self.dynamics.u_min, self.dynamics.u_max)
            cost = cost + W_ACT * (action.square() - bounded.square()).sum(-1)
            if rollout_step == 0:
                previous = self.tracking_cost.previous_action
                cost = cost + self.boundary_weight * (
                    (action-previous).square() - (bounded-previous).square()).sum(-1)
        if self.headroom_weight:
            cost = cost + headroom_cost(action + self._compensation_snapshot,
                self.dynamics.u_max, self.headroom_fraction, self.headroom_weight)
        return cost

    def _terminal_cost(self, states, actions):
        cost = super()._terminal_cost(states, actions)
        if not self.study['fixed_clip'] and self.horizon > 1:
            bounded = actions.clamp(self.dynamics.u_min, self.dynamics.u_max)
            delta_raw = actions[..., 1:, :] - actions[..., :-1, :]
            delta_bounded = bounded[..., 1:, :] - bounded[..., :-1, :]
            cost = cost + self.tracking_cost.smooth_weight * (
                delta_raw.square() - delta_bounded.square()).sum((-1, -2)) / (self.horizon-1)
        return cost

    def _mppi_command(self, state):
        action = super()._mppi_command(state)
        requested = self.ctrl.perturbed_action + self._compensation_snapshot
        self.request_sample_fraction.append((requested.abs() > self.dynamics.u_max).any(-1).float().mean().detach())
        self.headroom_sample_cost.append(headroom_cost(requested, self.dynamics.u_max,
            self.headroom_fraction, self.headroom_weight).mean().detach())
        return action

    def _warm_up_controller(self, *args, **kwargs):
        super()._warm_up_controller(*args, **kwargs)
        self.request_sample_fraction.clear()
        self.headroom_sample_cost.clear()

    def export_sensor_data(self):
        result = super().export_sensor_data()
        result.update(self.export_margin_data())
        result.update(input_study_case=np.asarray(self.study_case),
            input_study_fixed_clip=np.asarray(self.study['fixed_clip']),
            input_study_saturation_rollout=np.asarray(self.study['saturation_rollout']),
            input_study_headroom_weight=np.asarray(self.headroom_weight),
            input_study_headroom_fraction=np.asarray(self.headroom_fraction),
            input_study_cost_coordinates=np.asarray('nominal v; original effort and smoothness coefficients'),
            input_study_sample_request_excess_fraction=torch.stack(self.request_sample_fraction).cpu().numpy(),
            input_study_sample_headroom_cost=torch.stack(self.headroom_sample_cost).cpu().numpy(),
            input_tightening_enabled=np.asarray(False))
        return result
