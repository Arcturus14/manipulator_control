"""Bounded sustained-disturbance pilot and locked held-out final verification.

Never enables headroom or changes costs, dynamics, limits, noise, or margins.
Final duration is two circles (16 s), even when pilot runs last one circle.
"""
import argparse
import json
from pathlib import Path
import sys
from kroc.benchmark_nominal_margin_pressure import main as benchmark

NAMES=['static25','directional25_5','static5']


def invoke(output, amplitudes, seeds, seconds):
    previous=sys.argv
    sys.argv=['benchmark_nominal_margin_pressure','--screen',
        'kroc/results/nominal_margin5_geometry_screen_20261113','--output',str(output),
        '--candidates','61','--static-fractions','.025','.05','--period','8','--seconds',str(seconds),
        '--amplitudes',*map(str,amplitudes),'--onset','.5','--ramp','.4',
        '--seeds',*map(str,seeds)]
    try:benchmark()
    finally:sys.argv=previous


def assessment(rows):
    c={r['case']:r['metrics'] for r in rows}
    if set(c)!=set(NAMES):return {'complete':False,'full_relation':False}
    result={'complete':True,
        'all_collision_free':all(not m['self_collision_states'] and not m['floor_collision_states'] for m in c.values()),
        'low_static_violates':c['static25']['position_violation_max_rad']>1e-4,
        'dynamic_safe':c['directional25_5']['position_violation_steps']==0,
        'high_static_safe':c['static5']['position_violation_steps']==0,
        'dynamic_tracks_better_than_high_static':c['directional25_5']['tracking_rmse_mm']<c['static5']['tracking_rmse_mm'],
        'low_static_saturates':c['static25']['test_joint_saturation_steps']>0}
    result['full_relation']=all(v for k,v in result.items() if k not in ['complete','full_relation'])
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan={'pilot_seed':507,'pilot_amplitudes_nm':[145.,147.,148.,149.,149.5],
          'final_seeds':[607,617,627,637,647], 'pilot_seconds':8,'final_seconds':16,
          'headroom_enabled':False,'candidate':61,'period_seconds':8,
          'disturbance_joint_zero_indexed':0,'onset_seconds':.5,'ramp_seconds':.4,
          'disturbance_released':False,
          'success':'Same-seed low static position violation >1e-4 rad and saturation; dynamic/high static no position violation; dynamic lower RMSE than high static; ALL cases collision free.',
          'selection':'first/smallest passing pilot amplitude; if none, closest pilot is validated explicitly as exploratory, not a success',
          'mesh_final_audit':'5 ms interpolated official-mesh audit is mandatory after final runs'}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
    invoke(a.output/'pilot',plan['pilot_amplitudes_nm'],[plan['pilot_seed']],plan['pilot_seconds'])
    rows=json.loads((a.output/'pilot/metrics.json').read_text())
    summaries={amp:assessment([r for r in rows if r['amplitude']==amp]) for amp in plan['pilot_amplitudes_nm']}
    passing=[amp for amp in plan['pilot_amplitudes_nm'] if summaries[amp]['full_relation']]
    if passing:
        selected=passing[0];selection='pilot_pass'
    else:
        # Prefer safe, collision-free dynamic/high-static runs. Do not select a
        # favorable validation seed, alter the model, or remove failed records.
        def score(amp):
            c={r['case']:r['metrics'] for r in rows if r['amplitude']==amp};s=summaries[amp]
            return (s['all_collision_free'],s['dynamic_safe'] and s['high_static_safe'],
                    s['low_static_violates'],s['dynamic_tracks_better_than_high_static'],
                    -c['directional25_5']['position_violation_max_rad'],-amp)
        selected=max(plan['pilot_amplitudes_nm'],key=score);selection='no_pilot_pass_exploratory'
    result={'plan':plan,'pilot_assessments':summaries,'selected_amplitude_nm':selected,'selection':selection}
    (a.output/'selection.json').write_text(json.dumps(result,indent=2));print('FINAL_LOCKED',selected,selection,flush=True)
    invoke(a.output/'final', [selected],plan['final_seeds'],plan['final_seconds'])
    final=json.loads((a.output/'final/metrics.json').read_text())
    result['final_by_seed']={s:assessment([r for r in final if r['seed']==s]) for s in plan['final_seeds']}
    result['status']='pending_dense_mesh_audit'
    result['all_final_seeds_meet_state_tracking_criteria']=all(s['full_relation'] for s in result['final_by_seed'].values())
    (a.output/'selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
