"""Reference continuity and finite corner accelerations; no control runs."""
import unittest
import numpy as np
from kroc.payload_scene import rounded_rectangle,make_reference


class RectangleReferenceTests(unittest.TestCase):
    def test_closed_bounds_and_actual_straight_edges(self):
        p=rounded_rectangle(np.linspace(0,1,10001),.16,.12,.02)
        np.testing.assert_allclose(p[[0,-1]],[[.16,0],[.16,0]],atol=1e-15)
        self.assertLessEqual(abs(p[:,0]).max(),.16+1e-15)
        self.assertLessEqual(abs(p[:,1]).max(),.12+1e-15)
        self.assertGreater((np.isclose(abs(p[:,0]),.16,atol=1e-12)).sum(),1000)
        self.assertGreater((np.isclose(abs(p[:,1]),.12,atol=1e-12)).sum(),1000)

    def test_c2_at_line_corner_joins(self):
        a,b,r=.16,.12,.02;L=4*(a+b-2*r)+2*np.pi*r
        lengths=[b-r,np.pi*r/2,2*(a-r),np.pi*r/2,2*(b-r),np.pi*r/2,2*(a-r),np.pi*r/2,b-r]
        eps=1e-6
        for s in np.cumsum(lengths)[:-1]/L:
            q=rounded_rectangle(s+np.arange(-3,4)*eps,a,b,r)
            vl=(q[3]-q[2])/eps;vr=(q[4]-q[3])/eps
            al=(q[3]-2*q[2]+q[1])/eps**2;ar=(q[5]-2*q[4]+q[3])/eps**2
            np.testing.assert_allclose(vl,vr,atol=2e-6)
            np.testing.assert_allclose(al,ar,atol=.1)

    def test_bad_corner_sizes(self):
        for r in [0,-.1,.12,.2,float('nan')]:
            with self.assertRaises(ValueError):rounded_rectangle(np.array([0.,.5]),.16,.12,r)

    def test_start_ramp_begins_at_rest(self):
        class TCP:
            def position_and_jacobian_np(self,q):return np.zeros(3),None
        s=dict(trajectory='rounded_rectangle_yz',circle_radius_m=.16,period_s=6.,reference_start_ramp_s=1.)
        q=make_reference(TCP(),np.zeros(6),4,1e-3,s)
        np.testing.assert_array_equal(q[0],np.zeros(3))
        self.assertLess(np.linalg.norm((q[1]-q[0])/1e-3),1e-8)

if __name__=='__main__':unittest.main()
