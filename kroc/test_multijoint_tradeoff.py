import unittest
from kroc.search_circle_multijoint_tradeoff import pilot_gate,nominal_gate
from kroc.multijoint_margin_assessment import dominates


class MultijointCriteriaTests(unittest.TestCase):
    def rows(self):
        def m(steps,peak,integral,rms):
            return dict(position_violation_steps=steps,position_violation_max_rad=peak,
                position_violation_integral_rad_s=integral,tracking_rmse_mm=rms,first_collision_index=None)
        return [{'case':n,'metrics':v} for n,v in zip(
            ['static25','directional25_5','static5'],
            [m(40,.2,.1,5.),m(20,.1,.04,6.),m(0,0.,0.,9.)])]

    def test_nonzero_dynamic_violation_can_pass(self):
        self.assertTrue(pilot_gate(self.rows())['pass'])

    def test_tracking_loss_rejects_pilot(self):
        rows=self.rows();rows[1]['metrics']['tracking_rmse_mm']=12.
        self.assertFalse(pilot_gate(rows)['pass'])

    def test_collision_rejects_pilot(self):
        rows=self.rows();rows[0]['metrics']['first_collision_index']=20
        self.assertFalse(pilot_gate(rows)['pass'])

    def test_nominal_gate_requires_actual_tracking_pressure(self):
        rows=self.rows()
        for r in rows:r['metrics']['position_violation_max_rad']=0.
        self.assertTrue(nominal_gate(rows)['pass'])
        rows[2]['metrics']['tracking_rmse_mm']=6.1
        self.assertFalse(nominal_gate(rows)['pass'])

    def test_pareto_dominance_does_not_mean_only_better_rmse(self):
        a=dict(mean_violation_step_fraction=.1,worst_violation_rad=.1,
            mean_cumulative_violation_rad_s=.1,tracking_rmse_mean_mm=3.)
        b={**a,'tracking_rmse_mean_mm':4.}
        self.assertTrue(dominates(a,b))
        b['worst_violation_rad']=.01
        self.assertFalse(dominates(a,b))


if __name__=='__main__':unittest.main()
