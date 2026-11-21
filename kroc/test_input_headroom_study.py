"""Coordinate/cost regression checks; no CPU performance experiments."""
import unittest
from types import SimpleNamespace
import torch
from config import W_ACT
from kroc.input_headroom_study import InputHeadroomTracker, headroom_cost, saturation_prediction
from controllers.mppi_dob_controller import UR5Tracker


class InputHeadroomTests(unittest.TestCase):
    def test_cost_onset_limit_and_excess(self):
        requested = torch.tensor([[23.8], [28.], [32.2], [-28.]])
        actual = headroom_cost(requested, torch.tensor([28.]), .85, 10.)
        torch.testing.assert_close(actual, torch.tensor([0., 10., 40., 10.]), atol=1e-4, rtol=1e-5)

    def test_prediction_preserves_nominal_until_clipping(self):
        v = torch.tensor([[5., -12.], [12., -12.]])
        c = torch.tensor([20., -20.])
        actual = saturation_prediction(v, c, -28., 28.)
        torch.testing.assert_close(actual, torch.tensor([[5., -8.], [8., -8.]]))

    def test_unsaturated_model_is_bitwise_nominal_with_noisy_estimate(self):
        torch.manual_seed(9)
        v=torch.randn(1000,6);c=torch.randn(6)
        self.assertTrue(torch.equal(saturation_prediction(v,c,-28.,28.),v))

    def test_unrestricted_v_can_exceed_physical_limit_and_remain_feasible(self):
        v, c = torch.tensor([40.]), torch.tensor([-20.])
        torch.testing.assert_close(saturation_prediction(v, c, -28., 28.), v)
        adapter = SimpleNamespace(study={'fixed_clip': False})
        self.assertIs(InputHeadroomTracker._bound_nominal_action(adapter, v), v)
        nominal=torch.zeros(1,6);nominal[0,-1]=40.
        raw,base=self.tracker('N1'),self.tracker('Base')
        self.assertTrue(torch.equal(raw.ctrl._bound_action(nominal),nominal))
        self.assertEqual(base.ctrl._bound_action(nominal)[0,-1].item(),28.)

    def test_default_prediction_retains_legacy_coordinates(self):
        adapter = SimpleNamespace(dynamics=SimpleNamespace(u_min=-28., u_max=28.),
                                  _rollout_disturbance=torch.tensor([3.]))
        torch.testing.assert_close(UR5Tracker._prediction_torque(adapter, torch.tensor([40.])), torch.tensor([25.]))

    def tracker(self, case):
        return InputHeadroomTracker(torch.zeros(25,3).numpy(), torch.device('cpu'),
            study_case=case, headroom_weight=10., proposal='single', n_samples=4,
            horizon=3, collision_weight=0., smooth_weight=.25, boundary_weight=.25,
            position_weight=0., terminal_weight=0., joint_position_limit_weight=0.,
            joint_velocity_limit_weight=0.)

    def test_existing_input_costs_use_raw_v_without_silent_clipping(self):
        base, raw = self.tracker('Base'), self.tracker('N0')
        states = torch.zeros(2,12)
        action = torch.tensor([[0.,0.,0.,40.,0.,0.]]).expand(2,-1)
        bounded = action.clamp(base.dynamics.u_min, base.dynamics.u_max)
        expected = (W_ACT+.25)*(action.square()-bounded.square()).sum(-1)
        torch.testing.assert_close(raw._running_cost(states,action,0)-base._running_cost(states,bounded,0), expected)
        v = action[:,None].expand(-1,3,-1).clone()
        v[:,1] *= -1
        s = torch.zeros(1,2,3,12)
        vb = v.clamp(base.dynamics.u_min,base.dynamics.u_max)
        expected_terminal = .25*((v[:,1:]-v[:,:-1]).square()-(vb[:,1:]-vb[:,:-1]).square()).sum((-1,-2))/2
        torch.testing.assert_close(raw._terminal_cost(s,v[None])-base._terminal_cost(s,vb[None]),expected_terminal[None])

    def test_nominal_rollout_never_uses_compensation_but_cost_does(self):
        off,on=self.tracker('N0'),self.tracker('N1')
        on._compensation_snapshot[-1]=-26.
        action=torch.zeros(2,6)
        torch.testing.assert_close(on._prediction_torque(action),action)
        expected=headroom_cost(action+on._compensation_snapshot,on.dynamics.u_max,.85,10.)
        torch.testing.assert_close(on._running_cost(torch.zeros(2,12),action,1)
            -off._running_cost(torch.zeros(2,12),action,1),expected)

    def test_saturation_snapshot_is_causal_and_shared_by_cost_and_model(self):
        tracker=self.tracker('S1')
        tracker.dob.d_hat[-1]=-26.
        tracker._set_rollout_disturbance(False)
        tracker.dob.d_hat[-1]=0.
        self.assertEqual(tracker._compensation_snapshot[-1].item(),-26.)
        self.assertEqual(tracker._rollout_disturbance[-1].item(),0.)
        action=torch.zeros(1,6);action[0,-1]=-4.
        self.assertEqual(tracker._prediction_torque(action)[0,-1].item(),-2.)


if __name__ == '__main__':
    unittest.main()
