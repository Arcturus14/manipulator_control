"""Optional whole-horizon compilation of the final nominal-rollout controller."""
import torch

from controllers.fast_final_mppi import FastFinalTracker, COMPILE_OPTIONS


class GraphFinalTracker(FastFinalTracker):
    def __init__(self, *args, cuda_graphs=False, **kwargs):
        self.cuda_graphs = bool(cuda_graphs)
        super().__init__(*args, **kwargs)
        # Fixed margins still produce the original acceleration/headroom
        # diagnostics. Fuse that work instead of deleting those fields.
        self.margins.update = torch.compile(self.margins.update, options=COMPILE_OPTIONS)
        self.tracking_cost.reference_window = self.waypoints[:self.horizon+1].clone()
        options = dict(COMPILE_OPTIONS)
        if self.cuda_graphs:
            options.update({'triton.cudagraphs': True, 'triton.cudagraph_trees': True})
        self.ctrl._compute_rollout_costs_single = torch.compile(
            self.ctrl._compute_rollout_costs_single, fullgraph=True, options=options)

    def _running_cost(self, state, action, rollout_step):
        index = self.tracking_cost.state_index(rollout_step)
        return self._fast_stage(state, action, self.tracking_cost.reference_window[index],
            self.tracking_cost.previous_action, self._compensation_snapshot, rollout_step == 0)

    def _terminal_cost(self, states, actions):
        return self._fast_end(states, actions, self.tracking_cost.reference_window[self.horizon])

    def _warm_up_controller(self, state, feedforward, sampling_center_mode):
        self._prepare_solve(state, feedforward[:self.horizon])
        super()._warm_up_controller(state, feedforward, sampling_center_mode)
        self.margin_history.clear()

    def export_sensor_data(self):
        result = super().export_sensor_data()
        result.update(fast_final_compiled_horizon=True, fast_final_compiled_margin=True,
                      fast_final_cuda_graphs=self.cuda_graphs)
        return result
