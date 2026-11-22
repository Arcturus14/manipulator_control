"""Stage t of pytorch_mppi costs x(t+1): every time-indexed state reference must too."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pytorch_mppi import MPPI

from controllers.captured_final_mppi import CapturedFinalTracker
from controllers.fast_final_mppi import FastFinalTracker
from controllers.graph_final_mppi import GraphFinalTracker
from controllers.payload_stabilized_mppi import PayloadStabilizedTracker
from controllers.research_baseline_controller import ResearchBaselineTracker
from costs.obstacle_cost import ObstacleCost
from costs.tracking_cost import TrackingCost

H = 4


def tracking(offset=1, window=False):
    waypoints = torch.arange(20.)[:, None].expand(20, 3)
    cost = TrackingCost(waypoints, H, -torch.ones(6), torch.ones(6))
    cost.reference_offset = offset
    cost.episode_step = 3
    if window:
        cost.reference_window = waypoints[3:3+H+1].clone()
    return cost


class ReferenceAlignmentTests(unittest.TestCase):
    def test_library_calls_stage_cost_on_the_next_state(self):
        seen = []

        def running(state, action, t):
            seen.append((t, float(state[0, 0])))
            return torch.zeros(len(state))
        ctrl = MPPI(lambda state, action, t: state+1., running, 1, torch.eye(1),
                    num_samples=2, horizon=H, step_dependent_dynamics=True,
                    terminal_state_cost=lambda states, actions: torch.zeros(1, 2))
        ctrl.command(torch.zeros(1), shift_nominal_trajectory=False)
        self.assertEqual(seen, [(t, t+1.) for t in range(H)])

    def test_tracking_cost_with_and_without_compiled_window(self):
        q = torch.zeros(1, 12)
        for window in (False, True):
            for offset in (0, 1):
                cost = tracking(offset, window)
                for t in range(H):
                    with patch('costs.tracking_cost.fk_batch', lambda _q: cost.waypoints[3+t+offset][None]):
                        self.assertEqual(float(cost.running(q, torch.zeros(1, 6), t)), 0.)

    def test_dynamic_obstacle_uses_the_same_time(self):
        path = np.zeros((1, 20, 3), np.float32)
        path[0, :, 0] = np.arange(20)
        cost = ObstacleCost(H, torch.device('cpu'), torch.float32, dynamic_paths=path)
        cost.episode_step = 3
        with patch('costs.obstacle_cost.fk_batch', lambda q: torch.zeros(1, 3)), \
                patch.object(cost, 'evaluate_positions', side_effect=lambda ee, step: step):
            self.assertEqual([cost.running(torch.zeros(1, 12), None, t) for t in range(H)], [1, 2, 3, 4])

    def test_every_fused_tracker(self):
        for cls in (FastFinalTracker, CapturedFinalTracker, GraphFinalTracker,
                    ResearchBaselineTracker, PayloadStabilizedTracker):
            for offset in (0, 1):
                obj = cls.__new__(cls)
                obj.tracking_cost = tracking(offset, window=True)
                obj.waypoints = obj.tracking_cost.waypoints
                obj.step, obj.horizon, obj.fuse_costs = 3, H, True
                obj.obstacle_cost = None
                obj._compensation_snapshot = torch.zeros(6)
                seen = []

                def stage(state, action, reference, *rest):
                    seen.append((float(reference[0]), bool(rest[-1])))
                    return torch.zeros(1)
                obj._fast_stage = obj._stage = stage
                if cls is PayloadStabilizedTracker:
                    obj.posture_weight = 1.
                    obj.posture_window = torch.arange(H+1.)[:, None].expand(H+1, 6)
                for t in range(H):
                    state = torch.full((1, 12), float(t+offset))
                    value = obj._running_cost(state, torch.zeros(1, 6), t)
                    if cls is PayloadStabilizedTracker:
                        self.assertEqual(float(value[0]), 0.)  # posture reference is aligned as well
                self.assertEqual(seen, [(3.+t+offset, t == 0) for t in range(H)], cls.__name__)


if __name__ == '__main__':
    unittest.main()
