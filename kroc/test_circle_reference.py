import unittest
import numpy as np
from config import Q_MAX,Q_MIN
from kinematics.ur5_kinematics import fk_joints_np
from kroc.search_circle_floor25 import local_ik_reference


class ContinuousCircleIK(unittest.TestCase):
    def test_branch_above_pi_is_continuous_and_tracks_circle(self):
        q0=np.array([.94*Q_MAX[0],-1.2,1.2,-1.5,-1.57,0])
        start=fk_joints_np(q0)[-1]
        radial=np.r_[start[:2],0.];radial/=np.linalg.norm(radial)
        tangent=np.array([-radial[1],radial[0],0.])
        th=np.linspace(0,2*np.pi,201)
        targets=start+.08*((np.cos(th)-1)[:,None]*radial+np.sin(th)[:,None]*tangent)
        q=local_ik_reference(targets,q0)
        ee=np.array([fk_joints_np(x)[-1] for x in q])
        self.assertLess(np.linalg.norm(ee-targets,axis=1).max(),2e-6)
        self.assertLess(np.abs(np.diff(q,axis=0)).max(),.02)
        self.assertTrue((q>=np.array(Q_MIN)-1e-9).all())
        self.assertTrue((q<=np.array(Q_MAX)+1e-9).all())
        self.assertGreater(q[:,0].min(),np.pi)


if __name__=='__main__':unittest.main()
