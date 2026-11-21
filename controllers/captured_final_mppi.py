"""Explicit CUDA graph replay of the same warmed kernels; no model/cost changes."""
import numpy as np
import torch

from controllers.fast_final_mppi import FastFinalTracker


class CapturedFunction:
    """Fixed-shape pure tensor call with owned inputs and cloned outputs.

    Only construct outside the timed loop, after torch.compile warmup. There
    are no random operations inside these graphs; MPPI sampling stays outside.
    """
    def __init__(self, function, arguments, device):
        self.inputs = tuple(value.clone() for value in arguments)
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                function(*self.inputs)
        torch.cuda.current_stream(device).wait_stream(stream)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.outputs = function(*self.inputs)
        torch.cuda.synchronize(device)

    def __call__(self, *arguments):
        if len(arguments) != len(self.inputs):
            raise ValueError('Captured argument count changed')
        for target, value in zip(self.inputs, arguments):
            if target.shape != value.shape or target.dtype != value.dtype or target.device != value.device:
                raise ValueError('Captured tensor shape/dtype/device changed')
            target.copy_(value)
        self.graph.replay()
        if isinstance(self.outputs, tuple):
            return tuple(value.clone() for value in self.outputs)
        return self.outputs.clone()


class CapturedFinalTracker(FastFinalTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.fuse_costs or not self.compile_auxiliary:
            raise ValueError('Explicit replay requires fused costs and auxiliary compilation')
        self.tracking_cost.reference_window = self.waypoints[:self.horizon+1].clone()
        self._original_core = self.ctrl._compute_rollout_costs_single
        self._original_plant = self.dynamics.plant_step_effective
        self._original_prediction = self.dynamics.rollout_effective

    def _running_cost(self, state, action, rollout_step):
        return self._fast_stage(state, action, self.tracking_cost.reference_window[rollout_step],
            self.tracking_cost.previous_action, self._compensation_snapshot, rollout_step == 0)

    def _terminal_cost(self, states, actions):
        return self._fast_end(states, actions, self.tracking_cost.reference_window[self.horizon])

    def _warm_up_controller(self, state, feedforward, sampling_center_mode):
        self.ctrl._compute_rollout_costs_single = self._original_core
        self.dynamics.plant_step_effective = self._original_plant
        self.dynamics.rollout_effective = self._original_prediction
        super()._warm_up_controller(state, feedforward, sampling_center_mode)
        saved_state = self.ctrl.state
        saved_previous = self.tracking_cost.previous_action

        def core(static_state, actions, previous):
            self.ctrl.state = static_state
            self.tracking_cost.previous_action = previous
            return self._original_core(actions)

        try:
            self._core_graph = CapturedFunction(core, (saved_state, self.ctrl.perturbed_action,
                                                      saved_previous), self.device)
        finally:
            self.ctrl.state = saved_state
            self.tracking_cost.previous_action = saved_previous
        self.ctrl._compute_rollout_costs_single = self._replay_core
        torque = feedforward[0][None]
        self.dynamics.plant_step_effective = CapturedFunction(
            self._original_plant, (state[None], torque), self.device)
        self.dynamics.rollout_effective = CapturedFunction(
            self._original_prediction, (state[None], torque), self.device)

    def _replay_core(self, actions):
        return self._core_graph(self.ctrl.state, actions, self.tracking_cost.previous_action)

    def export_sensor_data(self):
        result = super().export_sensor_data()
        result['fast_final_explicit_cuda_graphs'] = np.asarray(True)
        return result
