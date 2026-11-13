"""Bounded wrist-margin pilot, then disjoint-seed validation if a case passes.

Keep one Python process to reuse full-rollout compilation between stages.
Never enables headroom or changes controller/margin/collision cost settings.
"""
import argparse
import json
from pathlib import Path
import sys
from kroc.search_circle_floor25 import main as circle_main


def invoke(output,amplitudes,seeds,base_fraction):
    previous=sys.argv
    sys.argv=['search_circle_floor25','--poses','0','--joint','4','--fractions','.94',
        '--base-fraction',str(base_fraction),'--radii','.08','--collision-check',
        '--collision-weight','50000','--collision-backend','triton','--compile-rollout-loop',
        '--amplitudes',*map(str,amplitudes),'--onsets','1.0','--ramps','.2',
        '--seconds','8','--seeds',*map(str,seeds),'--output',str(output)]
    try:circle_main()
    finally:sys.argv=previous


def passing(rows):
    cases={r['case']:r['metrics'] for r in rows}
    if set(cases)!={'static25','directional25_5','static5'}:return False
    if any(m['self_collision_states'] or m['floor_collision_states'] for m in cases.values()):return False
    return (cases['static25']['position_violation_max_rad']>0
        and cases['directional25_5']['position_violation_steps']==0
        and cases['static5']['position_violation_steps']==0
        and cases['directional25_5']['tracking_rmse_mm']<cases['static5']['tracking_rmse_mm'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--amplitudes',type=float,nargs='+',default=[27.5,27.55,27.6])
    p.add_argument('--base-fraction',type=float,default=.92)
    p.add_argument('--validation-seeds',type=int,nargs='+',default=[47,57])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if {7,17,27,37}.intersection(a.validation_seeds):
        p.error('validation seeds must exclude pilot 7 and nominal deadband calibration 17/27/37')
    invoke(a.output/'pilot',a.amplitudes,[7],a.base_fraction)
    rows=json.loads((a.output/'pilot'/'metrics.json').read_text())
    selected=None
    for scenario in sorted({r['scenario'] for r in rows}):
        group=[r for r in rows if r['scenario']==scenario]
        if passing(group):selected=group[0]['amplitude'];break
    result={'pilot_seed':7,'base_fraction':a.base_fraction,'headroom_enabled':False,'selected_amplitude_nm':selected,
            'status':'no_pilot_case' if selected is None else 'pilot_case_found',
            'criteria':'static25 position violation; dynamic and static5 no position violation; all collision-free; dynamic RMSE < static5'}
    (a.output/'selection.json').write_text(json.dumps(result,indent=2))
    if selected is not None:
        print('DISJOINT_SEED_VALIDATION',selected,flush=True)
        invoke(a.output/'validation',[selected],a.validation_seeds,a.base_fraction)
        valid=json.loads((a.output/'validation'/'metrics.json').read_text())
        result['validation_pass_by_seed']={seed:passing([r for r in valid if r['seed']==seed])
                                            for seed in a.validation_seeds}
        result['status']='validated' if all(result['validation_pass_by_seed'].values()) else 'pilot_not_reproduced_all_seeds'
        (a.output/'selection.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
