"""GUI baseline wiring and algebra; CPU checks are not performance benchmarks."""
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from utils.yaml_config import load_config, validate_config
from main import _mppi_options
from costs.tracking_cost import TrackingCost
from kroc.state_margin import StateMargins, MarginConfig
from controllers.research_baseline_controller import ResearchBaselineTracker, local_reference, asynchronous_forward_dynamics
from kinematics.ur5_kinematics import fk_batch, fk_joints_np


class GuiBaselineTests(unittest.TestCase):
    def test_gui_default_matches_frozen_preset(self):
        import json
        from simulation_gui import GUI_DEFAULT_CONFIG
        cfg=load_config(GUI_DEFAULT_CONFIG,{})
        frozen=json.loads(Path('kroc/configs/pre_input_study_baseline.json').read_text())
        opt=_mppi_options(cfg)
        from controllers.final_n0_controller import FinalN0Tracker
        self.assertIs(opt['tracker_class'],FinalN0Tracker)
        for key in ['horizon','mean_update_alpha']:
            self.assertEqual(opt[key],frozen[key])
        self.assertEqual(opt['n_samples'],frozen['samples'])
        self.assertEqual(opt['temperature'],64.)
        self.assertEqual(opt['noise_variance'],[9.,9.,9.,.3136,.3136,.3136])
        self.assertEqual(opt['tracker_options']['margin_config']['fixed_fraction'],.025)
        self.assertEqual(opt['tracker_options']['dob_acceleration_lpf_cutoff_hz'],6.)
        self.assertEqual(opt['tracker_options']['collision_weight'],50000.)
        self.assertTrue(cfg['experiment']['dob'])
        for key in ['fuse_costs','compile_auxiliary','asynchronous_solve']:
            self.assertFalse(cfg['controller'][key])
        self.assertFalse(cfg['output']['capture_samples'])

    def test_invalid_single_instance_rho_rejected(self):
        cfg=load_config(Path('configs/gui_pre_input_baseline.yaml'),{})
        cfg['controller']['temporal_noise_rho']=0.
        with self.assertRaises(ValueError):validate_config(cfg)

    def test_local_ik_reproduces_stationary_reference(self):
        q=np.array([3.14,-1.2,1.2,-1.5,-1.57,5.5])
        actual=local_reference(np.repeat(fk_joints_np(q)[-1:,:],4,axis=0),q)
        np.testing.assert_allclose(actual,np.repeat(q[None,:],4,axis=0),atol=1e-12)

    def test_async_solve_preserves_equations_and_residual(self):
        from dynamics.ur5_dynamics import UR5Dynamics
        torch.manual_seed(7)
        dynamics=UR5Dynamics(torch.device('cpu'),torch.float64)
        state=torch.randn(12,12);torque=torch.randn(12,6)
        state=state.double();torque=torque.double()
        expected=dynamics.forward_dynamics(state,torque)
        actual=asynchronous_forward_dynamics(dynamics,state,torque)
        torch.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
        mass,bias=dynamics.mass_matrix_and_bias(state[:,:6],state[:,6:])
        torch.testing.assert_close((mass@actual[...,None]).squeeze(-1),
            torque-bias-dynamics.damping*state[:,6:],rtol=1e-12,atol=1e-12)

    def test_fused_algebra_matches_original_stage_and_terminal(self):
        obj=ResearchBaselineTracker.__new__(ResearchBaselineTracker)
        obj.horizon=3;obj.step=0;obj.fuse_costs=True
        obj.obstacle_cost=None;obj.collision_geometry_cost=None
        obj.dynamics=SimpleNamespace(u_min=-torch.ones(6)*28,u_max=torch.ones(6)*28)
        obj.waypoints=torch.zeros(10,3)
        c=TrackingCost(obj.waypoints,3,obj.dynamics.u_min,obj.dynamics.u_max)
        obj.tracking_cost=c
        margins=StateMargins(obj.dynamics,c,MarginConfig(fixed_fraction=.025))
        c.state_limit_cost=margins.cost_value
        state=torch.randn(8,12);action=torch.randn(8,6)
        # Use the same eager FK on both sides, avoiding CPU compiler benchmarks.
        eager=fk_batch.__wrapped__
        with patch('costs.tracking_cost.fk_batch',eager),patch('controllers.research_baseline_controller.fk_batch',eager):
            for i in range(3):
                torch.testing.assert_close(obj._stage_value(state,action,obj.waypoints[i],i==0),c.running(state,action,i))
            states=torch.randn(1,8,3,12);actions=torch.randn(1,8,3,6)
            torch.testing.assert_close(obj._end_value(states,actions,obj.waypoints[3]),c.terminal(states,actions))


if __name__=='__main__':unittest.main()
