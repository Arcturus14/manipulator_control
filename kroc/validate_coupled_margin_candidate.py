"""Lock the first successful coupled-joint pilot before held-out testing."""
import argparse
import json
from pathlib import Path
import sys
from kroc import benchmark_nominal_margin_pressure as benchmark


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan={'bank':'kroc/results/coupled_wrist_margin_bank_20261113','candidate':9,
          'amplitude_nm':27.4,'joint':4,'onset_s':1.,'ramp_s':.2,'held_to_episode_end':True,
          'heldout_seeds':[907,917,927],'cycles':1,'headroom_enabled':False,
          'selection':'first/lower-amplitude pilot full-gate pass; selected before viewing 27.55Nm outcome',
          'pilot_source':'kroc/results/margin_mechanism_probe_20261113/coupled_wrist; seed 807',
          'full_gate':'low violation >1e-4 rad plus wrist saturation; dynamic/high safe; all mesh-free; dynamic RMSE <=25mm and >=2mm better than high'}
    (a.output/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    dest=a.output/'heldout'
    sys.argv=['benchmark','--screen',plan['bank'],'--candidates','9',
        '--static-fractions','.025','.05','--dynamic-max','.05','--seeds','907','917','927',
        '--output',str(dest),'--use-banked-reference','--disturbance-joint','4',
        '--onset','1.','--ramp','.2','--amplitudes','27.4']
    benchmark.main()
    rows=json.loads((dest/'metrics.json').read_text());paired={}
    for seed in plan['heldout_seeds']:
        by={r['case']:r['metrics'] for r in rows if r['seed']==seed}
        low,dyn,high=[by[n] for n in ('static25','directional25_5','static5')]
        paired[seed]={'low_violated':low['position_violation_max_rad']>1e-4,
            'low_saturated':low['test_joint_saturation_steps']>0,
            'dynamic_safe':dyn['position_violation_max_rad']<=1e-6,
            'high_safe':high['position_violation_max_rad']<=1e-6,
            'all_mesh_free_20ms':all(m['first_collision_index'] is None for m in by.values()),
            'dynamic_tracking_acceptable':dyn['tracking_rmse_mm']<=25.,
            'dynamic_beats_high_by_2mm':high['tracking_rmse_mm']-dyn['tracking_rmse_mm']>=2.}
        paired[seed]['full_gate_pass']=all(paired[seed].values())
    verdict={'paired':paired,'all_heldout_pass':all(v['full_gate_pass'] for v in paired.values()),
        'dense_mesh_audit_pending':True,'not_a_general_robustness_guarantee':True}
    (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print('COUPLED_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
