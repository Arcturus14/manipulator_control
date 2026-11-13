"""Locked circle-only replication of a selected risk/tracking trade-off.

No all-seed dominance or zero-violation requirement. No automatic MD report.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    prior=Path('kroc/results/circle_wrist_common_start_validation_20261113')
    seeds=[1307,1317,1327,1337,1347]
    assert set(seeds).isdisjoint([17,27,37,107,117,127,137,147])
    files=['controllers/mppi_dob_controller.py','costs/tracking_cost.py','experiment_runner.py',
        'observers/dob.py','config.py','kroc/state_margin.py','kroc/sensor_noise.py',
        'kroc/search_circle_floor25.py','kroc/search_postdob_floor25.py',
        'kroc/benchmark_postdob_small_directional_margin.py','kroc/self_collision_cost.py']
    plan={'prior_selection_source':str(prior),'prior_is_not_new_validation':True,
        'fresh_seeds':seeds,'headroom_enabled':False,'circle_radius_m':.08,'period_s':4.,
        'episode_s':8.,'disturbance_joint_zero_index':4,'disturbance_nm':-27.55,
        'onset_s':1.,'rise_s':.2,'held_until_episode_end':True,
        'initial_wrist_fraction_of_positive_limit':.89,'initial_base_rad':0.,
        'cases':['static25','directional25_5','static5'],
        'primary_gates':[
            'dynamic has fewer violation episodes AND lower violating-control-step fraction than static25',
            'dynamic has lower worst all-joint position violation than static25',
            'dynamic has lower mean cumulative all-joint position violation than static25',
            'dynamic mean per-seed EE RMSE is lower than static5',
            'all paired trajectories are collision-free in offline 5ms interpolated mesh audit'],
        'violation_tolerance_rad':1e-6,'zero_dynamic_violation_is_not_required':True,
        'same_seed_simultaneous_dominance_is_not_required':True,
        'stopping_rule':'If this locked independent replication fails, stop circle search this turn; no parameter retuning on these seeds.',
        'normalization':'unchanged fixed native 5%-span denominator in all cases',
        'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
        'prior_metrics_sha256':hashlib.sha256((prior/'metrics.json').read_bytes()).hexdigest()}
    (a.output/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    from kroc import search_circle_floor25 as search
    fresh=a.output/'fresh'
    sys.argv=['search','--fractions','.89','--radii','.08','--poses','0',
        '--amplitudes','27.55','--onsets','1.','--ramps','.2','--signs','1',
        '--joint','4','--scale','1.','--period','4.','--seconds','8.',
        '--seeds',*[str(s) for s in seeds],'--output',str(fresh),'--base-fraction','0.',
        '--collision-check','--collision-weight','50000.','--collision-clearance','.005',
        '--collision-backend','triton','--compile-rollout-loop','--require-common-start']
    search.main()
    from kroc import audit_circle_validation as audit
    sys.argv=['audit',str(fresh),'--output',str(a.output/'audit'),
        '--mesh-dt','.005','--mesh-workers','6','--require-common-start']
    audit.main()
    from kroc.circle_risk_tradeoff_assessment import assess,plot_assessment
    historical=assess(prior,prior/'figures/audited_results.json')
    result=assess(fresh,a.output/'audit/audited_results.json')
    (a.output/'historical_reassessment.json').write_text(json.dumps(historical,indent=2))
    (a.output/'fresh_assessment.json').write_text(json.dumps(result,indent=2))
    plot_assessment(result,a.output/'tradeoff_results.png')
    from kroc import plot_circle_floor25 as plot
    sys.argv=['plot',str(fresh),'--output',str(a.output/'fresh_circle_comparison.png')]
    plot.main()
    verdict={'fresh_primary_pass':result['primary_pass'],
        'historical_primary_pass':historical['primary_pass'],
        'status':'circle_tradeoff_replicated_in_fresh_sample' if result['primary_pass'] else 'circle_search_stopped_no_confirmed_replication',
        'not_a_formal_safety_or_statistical_significance_guarantee':True,
        'fresh_and_historical_statistics_are_separate':True,
        'headroom_enabled':False,'further_scenario_search_performed':False}
    (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print('CIRCLE_TRADEOFF_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
