"""GPU checks of the cost algebra, including nominal inputs beyond actuator bounds."""
import inspect
import unittest

import torch

from controllers.fast_final_mppi import FastFinalTracker
from controllers.mppi_dob_controller import UR5Tracker
from kroc.improve_n1_clipping_cost import clipping_loss_cost
from kroc.optimize_final_mppi import options
from kroc.search_lowfreq_estimator_margin import geometry


def tracker(case='N0', cls=FastFinalTracker):
    opts = options(6807, case, 'deferred')
    accepted = inspect.signature(UR5Tracker.__init__).parameters
    kwargs = {k: v for k, v in opts.items() if k in accepted}
    kwargs.update(opts['tracker_options'])
    kwargs['compile_auxiliary'] = False
    ref, _, _ = geometry(.89)
    return cls(ref, torch.device('cuda'), dtype=torch.float32, **kwargs)


class CostEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError('These checks require CUDA; no CPU fallback')

    def test_compiled_costs_match_reference_outside_nominal_physical_bounds(self):
        torch.manual_seed(391771)
        for case in ['N0', 'N1']:
            ctrl = tracker(case)
            _, q0, _ = geometry(.89)
            state = torch.randn(136, 12, device='cuda')*.1
            state[:, :6] += torch.as_tensor(q0, device='cuda')
            # Include state-limit penalties and torque requests on both sides.
            state[::4, :6] = ctrl.tracking_cost.q_max+.02
            action = torch.randn(136, 6, device='cuda')*ctrl.dynamics.u_max*1.4
            ctrl._compensation_snapshot.copy_(torch.tensor([20., -30., -130., 5., -5., 3.], device='cuda'))
            ctrl.tracking_cost.previous_action.copy_(action[0])
            for step in [0, 5]:
                ctrl.fuse_costs = False
                expected = ctrl._running_cost(state, action, step)
                ctrl.fuse_costs = True
                actual = ctrl._running_cost(state, action, step)
                torch.testing.assert_close(actual, expected, rtol=5e-6, atol=.005)
            states = state[None, :, None, :].expand(1, -1, 10, -1).clone()
            actions = action[None, :, None, :].expand(1, -1, 10, -1).clone()
            actions[..., 1::2, :] *= -.7
            ctrl.fuse_costs = False
            expected = ctrl._terminal_cost(states, actions)
            ctrl.fuse_costs = True
            actual = ctrl._terminal_cost(states, actions)
            torch.testing.assert_close(actual, expected, rtol=5e-6, atol=.005)

    def test_n1_starts_only_beyond_physical_limit(self):
        bound = torch.tensor([150., 150., 150., 28., 28., 28.], device='cuda')
        request = torch.stack([.95*bound, bound, bound+3., -bound-3.])
        actual = clipping_loss_cost(request, -bound, bound, 1.)
        torch.testing.assert_close(actual, torch.tensor([0., 0., 54., 54.], device='cuda'))

    def test_dynamic_bounds_translate_and_update_uses_projected_noise(self):
        from controllers.dynamic_bound_mppi import DynamicBoundTracker
        ctrl = tracker('N0', DynamicBoundTracker)
        ctrl.dob.d_hat[2] = -130.
        ctrl._set_rollout_disturbance(False)
        torch.testing.assert_close(ctrl._feasible_upper-ctrl._feasible_lower,
                                   ctrl.dynamics.u_max-ctrl.dynamics.u_min)
        trial = torch.zeros(2, 6, device='cuda')
        trial[:, 2] = torch.tensor([-30., 200.], device='cuda')
        projected = ctrl._project(trial)
        torch.testing.assert_close(projected[:, 2], torch.tensor([-20., 200.], device='cuda'))
        self.assertTrue(torch.all((projected+ctrl._compensation_snapshot).abs()<=ctrl.dynamics.u_max))
        mppi = ctrl.ctrl
        mppi.U.zero_()
        center = mppi.U.clone()
        noise = torch.zeros(mppi.K, mppi.T, 6, device='cuda')
        noise[:, :, 2] = torch.linspace(-400., 400., mppi.K, device='cuda')[:, None]
        mppi._sample_noise = lambda shape: noise.clone()
        mppi._compute_perturbed_action_and_noise()
        expected = ctrl._project(center[None]+noise)
        torch.testing.assert_close(mppi.perturbed_action, expected)
        torch.testing.assert_close(mppi.noise, expected-center)
        weight = torch.full((mppi.K,), 1./mppi.K, device='cuda')
        updated = center+torch.einsum('k,ktn->tn', weight, mppi.noise)
        torch.testing.assert_close(updated, torch.einsum('k,ktn->tn', weight, expected))
        self.assertTrue(torch.all(updated>=ctrl._feasible_lower-3e-5))
        self.assertTrue(torch.all(updated<=ctrl._feasible_upper+3e-5))
        self.assertGreater(float(updated[0, 2]), 20.)


if __name__ == '__main__':
    unittest.main()
