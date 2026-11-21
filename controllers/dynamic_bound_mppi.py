"""Comparison only: N0 objective, nominal rollout, compensation-shifted bounds.

The interval is translated, not symmetrically tightened. Projection happens
before rollout and after the weighted update; the physical clamp remains.
The library computes effective noise as projected samples minus the center.
"""
import numpy as np
import torch

from controllers.fast_final_mppi import FastFinalTracker
from controllers.graph_final_mppi import GraphFinalTracker


class DynamicBoundMixin:
    def __init__(self, *args, **kwargs):
        kwargs['study_case'] = 'N0'
        super().__init__(*args, **kwargs)
        self._feasible_lower = self.dynamics.u_min.clone()
        self._feasible_upper = self.dynamics.u_max.clone()
        self.ctrl._bound_action = self._bound_sample_action
        self.dynamic_bounds_history = []
        self.dynamic_projection_history = []
        self._raw_sample_excess_fraction = self.dynamics.u_min.new_zeros(())

    def _set_rollout_disturbance(self, use_dob):
        super()._set_rollout_disturbance(use_dob)
        self._feasible_lower.copy_(self.dynamics.u_min-self._compensation_snapshot)
        self._feasible_upper.copy_(self.dynamics.u_max-self._compensation_snapshot)

    def _project(self, action):
        return action.clamp(self._feasible_lower, self._feasible_upper)

    def _bound_sample_action(self, action):
        self._raw_sample_excess_fraction = (
            (action<self._feasible_lower)|(action>self._feasible_upper)
        ).any(-1).float().mean().detach()
        return self._project(action)

    def _bound_nominal_action(self, action):
        return self._project(action)

    def _mppi_command(self, state):
        super()._mppi_command(state)
        # The weighted average of projected samples is feasible in exact
        # arithmetic. Reproject the WHOLE sequence for float32 roundoff and
        # compatibility with a future blended-mean implementation.
        projected = self._project(self.ctrl.U)
        correction = (projected-self.ctrl.U).abs().amax().detach()
        self.ctrl.U.copy_(projected)
        self.dynamic_bounds_history.append(torch.stack((self._feasible_lower, self._feasible_upper)).clone())
        self.dynamic_projection_history.append(torch.stack((
            self._raw_sample_excess_fraction, correction)).clone())
        return self.ctrl.get_action_sequence()[0]

    def _warm_up_controller(self, *args, **kwargs):
        super()._warm_up_controller(*args, **kwargs)
        self.dynamic_bounds_history.clear()
        self.dynamic_projection_history.clear()

    def run(self, *args, **kwargs):
        self.dynamic_bounds_history.clear()
        self.dynamic_projection_history.clear()
        return super().run(*args, **kwargs)

    def export_sensor_data(self):
        result = super().export_sensor_data()
        result.update(input_study_case=np.asarray('D0'), dynamic_input_bounds=np.asarray(True),
            input_tightening_enabled=np.asarray(False),
            input_bounds_translated=np.asarray(True),
            dynamic_nominal_input_bounds=torch.stack(self.dynamic_bounds_history).cpu().numpy(),
            dynamic_sample_projection=torch.stack(self.dynamic_projection_history).cpu().numpy(),
            dynamic_bound_cost=np.asarray('N0; no soft input penalty; same raw nominal effort/smoothness'))
        return result


class DynamicBoundTracker(DynamicBoundMixin, FastFinalTracker):
    pass


class DynamicBoundGraphTracker(DynamicBoundMixin, GraphFinalTracker):
    pass
