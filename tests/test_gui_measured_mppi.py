"""GUI wiring tests; no dynamics simulation or CUDA context required."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
from controllers.measured_mppi_controller import MeasuredStateTracker
from controllers.mppi_dob_controller import UR5Tracker
from costs.tracking_cost import TrackingCost
from utils.yaml_config import load_config, validate_config


class GuiMeasuredMppiTests(unittest.TestCase):
    def make_tracker(self,**kwargs):
        def fake_init(obj,*args,**kw):
            obj.device=torch.device('cpu');obj.dtype=torch.float32
            obj.tracking_cost=TrackingCost(torch.zeros(20,3),10,-torch.ones(6),torch.ones(6))
        with patch.object(UR5Tracker,'__init__',fake_init):
            return MeasuredStateTracker(**kwargs)

    def test_physical_onset_preserves_nonzero_normalizers(self):
        tracker=self.make_tracker(physical_limit_penalty_only=True)
        cost=tracker.tracking_cost
        self.assertTrue((cost.q_lower_margin>0).all())
        self.assertTrue((cost.q_upper_margin>0).all())
        torch.testing.assert_close(cost.q_soft_min,cost.q_min)
        torch.testing.assert_close(cost.q_soft_max,cost.q_max)
        inside=torch.cat((cost.q_max-.01,torch.zeros(6)))[None]
        outside=inside.clone();outside[0,0]=cost.q_max[0]+.01
        self.assertEqual(float(cost.state_limit_cost(inside)),0.)
        self.assertGreater(float(cost.state_limit_cost(outside)),0.)

    def test_measurement_changes_not_true_state_and_resets(self):
        tracker=self.make_tracker()
        state=torch.zeros(12);measured=tracker._measure_state(state)
        torch.testing.assert_close(state,torch.zeros(12))
        self.assertFalse(torch.equal(state,measured))
        tracker.sensor_model.reset()
        torch.testing.assert_close(tracker._measure_state(state),measured)

    def test_disabled_noise_is_exact_clone(self):
        tracker=self.make_tracker(sensor_noise={'position_std_rad':0.,'velocity_std_rad_s':0.})
        state=torch.ones(12);measured=tracker._measure_state(state)
        torch.testing.assert_close(state,measured)
        self.assertNotEqual(state.data_ptr(),measured.data_ptr())

    def test_gui_preset_wiring(self):
        from main import _mppi_options
        cfg=load_config(Path('configs/kroc_gui_rho0_physical_limits.yaml'),{})
        options=_mppi_options(cfg)
        self.assertEqual(options['temporal_noise_rho'],0.)
        self.assertEqual(options['temperature'],.24)
        self.assertIs(options['tracker_class'],MeasuredStateTracker)
        self.assertTrue(options['tracker_options']['physical_limit_penalty_only'])
        self.assertEqual(options['tracker_options']['dob_cutoff_hz'],8.)
        self.assertEqual(options['tracker_options']['dob_acceleration_lpf_cutoff_hz'],10.)
        self.assertEqual(options['tracker_options']['sensor_noise']['velocity_std_rad_s'],.01)
        cfg['measurement']['enabled']=False
        self.assertEqual(_mppi_options(cfg)['tracker_options']['sensor_noise']['velocity_std_rad_s'],0.)

    def test_bad_noise_rejected(self):
        cfg=load_config(None,{})
        cfg['measurement']['position_std_rad']=-.1
        with self.assertRaises(ValueError):validate_config(cfg)

    def test_animation_backend_reenabled(self):
        from visualization.backend import enable_gui_backend
        with patch('matplotlib.pyplot.switch_backend') as switch:
            enable_gui_backend()
        switch.assert_called_once_with('TkAgg')

    def test_cached_single_run_can_animate_without_resimulation(self):
        from main import _show_cached_result
        cfg=load_config(Path('configs/kroc_gui_rho0_physical_limits.yaml'),{})
        cfg['experiment']['duration']=.06
        with TemporaryDirectory() as temporary,patch('main._show_animation') as show:
            directory=Path(temporary)
            np.savez(directory/'results.npz',q=np.zeros((4,6)),dob_cutoff_hz=8.)
            _show_cached_result('run',cfg,directory)
        self.assertEqual(show.call_count,1)
        results=show.call_args.args[0]
        self.assertEqual(results['MPPI_DOB']['q'].shape,(4,6))
        self.assertEqual(float(results['MPPI_DOB']['dob_cutoff_hz']),8.)


if __name__=='__main__':unittest.main()
