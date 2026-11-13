import unittest
import numpy as np
from config import DT,Q_MAX
from kinematics.ur5_kinematics import fk_joints_np
from kroc.screen_shape_margin_bank import build_reference,local_curve


class ShapeBankReferenceTests(unittest.TestCase):
    def test_start_and_periodic_kinematic_limits(self):
        q=np.array([.88*Q_MAX[0],-1.,1.2,-1.5,-1.57,0.])
        for shape in ('circle','infinity','rectangle'):
            with self.subTest(shape=shape):
                ref,_,period,speed,acc=build_reference(q,shape,.4)
                np.testing.assert_allclose(ref[0],fk_joints_np(q)[-1],atol=1e-12)
                self.assertAlmostEqual(len(ref)*DT,period)
                self.assertLessEqual(speed,.2501)
                self.assertLessEqual(acc,.351)
                self.assertTrue(np.isfinite(ref).all())

    def test_rectangle_uses_straight_sides_and_stops_at_corners(self):
        n=4000;ref=local_curve('rectangle',n)
        for k in range(4):
            side=ref[k*n//4:(k+1)*n//4]
            self.assertLess(np.ptp(side[:,1 if k%2==0 else 0]),1e-12)
            i=k*n//4
            self.assertLess(np.linalg.norm(ref[(i+1)%n]-ref[(i-1)%n])*n/2,.001)


if __name__=='__main__':unittest.main()
