"""Payload N0/N1/D0 on generated GPU kernels: same equations, faster evaluation.

Select with ``controller.rollout_backend: kernel``. The captured backend stays
the reference implementation and the only one used for frozen replays.
"""
import numpy as np
import torch

from controllers.payload_stabilized_mppi import (PayloadStabilizedD0Tracker,
                                                 PayloadStabilizedTracker)
from controllers.rollout_kernels import ObserverKernel, RolloutKernels


class KernelBackendMixin:
    def _warm_up_controller(self, state, feedforward, sampling_center_mode):
        super()._warm_up_controller(state, feedforward, sampling_center_mode)
        self._rollout_kernels = RolloutKernels(self)
        self._observer_kernel = ObserverKernel(self.dob, self.dynamics)
        saved_state = self.ctrl.state
        saved_observer = {key: getattr(self.dob, key) for key in self.DOB_STATE}
        self.ctrl.state = state
        actions = feedforward[:self.horizon][None].expand(self.n_samples, -1, -1)
        for _ in range(20):
            self._rollout_kernels(actions)
            self._observer_kernel(feedforward[0], state[:6], state[6:], state[:6], state[6:])
        torch.cuda.synchronize(self.device)
        self.ctrl.state = saved_state
        for key, value in saved_observer.items():
            setattr(self.dob, key, value)
        self.ctrl._compute_rollout_costs_single = self._rollout_kernels
        self.dob.update = self._observer_kernel
        self._margin_inputs = []
        cost = self.tracking_cost
        self._fixed_bounds = dict(
            q_soft_min_applied=cost.q_soft_min.clone(),
            q_soft_max_applied=cost.q_soft_max.clone(),
            dq_soft_max_applied=cost.dq_soft_max.clone(),
            q_upper_normalizer=cost.q_upper_margin.clone())

    DOB_STATE = ('d_hat', 'residual', 'tau_model', 'raw_acceleration', 'filtered_acceleration')

    def _prepare_solve(self, state, sampling_center):
        self.posture_window.copy_(self.joint_reference[self.step:self.step+self.horizon+1])
        # Fixed margins do not depend on the state. Their diagnostics do not
        # enter the control, so only their inputs are kept during the episode.
        self._margin_inputs.append((state.clone(), sampling_center, self.dob.d_hat.clone()))

    def export_sensor_data(self):
        for inputs in self._margin_inputs:
            row = dict(zip(self.MARGIN_KEYS,
                           (value.detach().clone() for value in self._fixed_diagnostics(*inputs))))
            row.update(self._fixed_bounds)
            self.margin_history.append(row)
        self._margin_inputs = []
        data = super().export_sensor_data()
        data['payload_rollout_backend'] = np.asarray('kernel')
        return data


class PayloadKernelTracker(KernelBackendMixin, PayloadStabilizedTracker):
    pass


class PayloadKernelD0Tracker(KernelBackendMixin, PayloadStabilizedD0Tracker):
    pass
