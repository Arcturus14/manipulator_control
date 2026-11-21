"""Check the removal profile, phase alignment, and seed separation."""
import unittest
import numpy as np
from kroc.final_input_headroom_checks import load_fraction,SEEDS,SCENARIOS


class FinalChecksTests(unittest.TestCase):
    def test_original_rise_and_hold_are_unchanged(self):
        t=np.array([0.,.5,1.5,2.5,3.5,5.,6.])
        np.testing.assert_allclose(load_fraction(t),[0.,0.,.5,1.,1.,1.,1.],atol=1e-15)

    def test_removal_is_continuous_and_has_recovery_window(self):
        t=np.array([2.5,3.5,4.25,5.,5.5,6.])
        np.testing.assert_allclose(load_fraction(t,True),[1.,1.,.5,0.,0.,0.],atol=1e-15)
        self.assertLess(abs(float(load_fraction(3.5+1e-6,True))-1.),1e-10)

    def test_parameters_are_not_selected_from_confirmation_seeds(self):
        self.assertFalse(set(SEEDS)&{5107,5207,5217,5227,5237,5247})
        self.assertEqual(sum(2*len(SEEDS) for _ in SCENARIOS),70)


if __name__=='__main__':unittest.main()
