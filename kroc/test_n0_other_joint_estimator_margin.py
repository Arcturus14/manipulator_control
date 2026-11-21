"""Search eligibility tests; no CPU MPPI performance runs."""
import copy
import unittest
import numpy as np
from config import Q_MIN,Q_MAX
from kroc.search_n0_other_joint_estimator_margin import SCENES,geometry,static_failure,separation
from kroc.compare_n0_state_margins import options


class OtherJointSearchTests(unittest.TestCase):
    def test_all_common_starts(self):
        span=np.asarray(Q_MAX)-Q_MIN
        for scene in SCENES:
            ref,q=geometry(scene)
            self.assertEqual(ref.shape,(300,3))
            self.assertGreater(np.minimum(q-np.asarray(Q_MIN)-.05*span,
                np.asarray(Q_MAX)-.05*span-q).min(),0)
            np.testing.assert_allclose(np.linalg.norm(ref-ref.mean(0),axis=1),scene['radius'],atol=1e-12)

    def test_frozen_estimator_only_options(self):
        opts=options(8607,'dynamic_estimate',None)
        self.assertEqual(opts['dob_compensation_mode'],'estimate_only')
        self.assertEqual(opts['temperature'],64.)
        self.assertEqual(opts['tracker_options']['margin_config']['mode'],'directional')
        self.assertEqual(opts['tracker_options']['margin_config']['max_fraction'],.05)
        self.assertEqual(opts['tracker_options']['dob_cutoff_hz'],3.)
        self.assertEqual(opts['tracker_options']['dob_acceleration_lpf_cutoff_hz'],6.)

    def test_static_failure_and_collision_rejection(self):
        row=dict(eligible=True,metrics=dict(max_position_excess_rad=0.,
            velocity_violation_steps=0,rmse_mm=10.))
        low=copy.deepcopy(row);low['metrics']['max_position_excess_rad']=.01
        self.assertTrue(static_failure(low));self.assertTrue(separation(low,row))
        low['eligible']=False
        self.assertFalse(static_failure(low));self.assertFalse(separation(low,row))
        low['eligible']=True;row['metrics']['max_position_excess_rad']=.002
        self.assertFalse(separation(low,row))


if __name__=='__main__':unittest.main()
