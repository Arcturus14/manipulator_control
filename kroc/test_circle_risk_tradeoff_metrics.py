import unittest
import numpy as np
from config import Q_MAX,Q_MIN
from kroc.circle_risk_tradeoff_assessment import position_metrics,comparison_gates


class RiskMetricTests(unittest.TestCase):
    def test_counts_all_joints_and_excludes_initial_state(self):
        q=np.zeros((4,6));q[0,0]=Q_MAX[0]+1
        q[1,0]=Q_MAX[0]+.1;q[1,2]=Q_MIN[2]-.2;q[2,4]=Q_MAX[4]+.05
        m=position_metrics(q,dt=.02)
        self.assertEqual(m['violating_steps'],2)
        self.assertAlmostEqual(m['maximum_violation_rad'],.2)
        self.assertAlmostEqual(m['cumulative_violation_rad_s'],.007)
        self.assertAlmostEqual(m['violation_step_fraction'],2/3)

    def test_numerical_tolerance_is_not_counted(self):
        q=np.zeros((3,6));q[1,0]=Q_MAX[0]+.5e-6
        m=position_metrics(q)
        self.assertFalse(m['violation_episode'])
        self.assertEqual(m['cumulative_violation_rad_s'],0.)

    def test_improvement_does_not_require_zero_dynamic_violation(self):
        def record(episodes,fraction,peak,integral,rms):
            return dict(violation_episodes=episodes,mean_violation_step_fraction=fraction,
                worst_violation_rad=peak,mean_cumulative_violation_rad_s=integral,
                tracking_rmse_mean_mm=rms,mesh_collision_trajectories=0)
        a={'static25':record(4,.1,.2,.08,3.),
           'directional25_5':record(2,.05,.1,.03,4.),
           'static5':record(0,0.,0.,0.,6.)}
        self.assertTrue(all(comparison_gates(a).values()))


if __name__=='__main__':unittest.main()
