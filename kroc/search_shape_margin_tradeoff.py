"""Bounded exploratory search; no headroom and no automatic research report.

Pilot only: do not interpret a selected seed as held-out validation.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from kroc import benchmark_nominal_margin_pressure as benchmark


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--onset-mode',choices=['early','nominal-phase'],default='early')
    p.add_argument('--reuse-nominal',type=Path)
    p.add_argument('--disturbed-candidates',type=int,nargs='+')
    p.add_argument('--amplitudes',type=float,nargs='+',default=[140.,145.,147.])
    p.add_argument('--ramp',type=float,default=.4)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    bank=json.loads((a.bank/'screen.json').read_text())['selected']
    candidates=[18,22,20,48]
    plan={'candidate_ids':candidates,'seed':807,'amplitudes_nm':a.amplitudes,
          'disturbance':'base joint negative ramp, held until end','ramp_seconds':a.ramp,
          'onset_mode':a.onset_mode,
          'disturbed_onset_rule':'0.5 s' if a.onset_mode=='early' else 'first nominal low-static positive-velocity crossing of 70% of its initial-to-maximum base excursion; minimum 0.5 s',
          'nominal_gate':{'max_dynamic_rmse_mm':10.,'min_high_minus_dynamic_rmse_mm':2.,
                          'min_high_over_dynamic_ratio':1.2},
          'full_gate':{'low_violation_min_rad':1e-4,'dynamic_and_high_violation_max_rad':1e-6,
                       'dynamic_rmse_max_mm':25.,'high_minus_dynamic_rmse_min_mm':2.,
                       'all_three_collision_free':True,'low_saturation_required':True},
          'max_disturbed_candidates':2,'headroom_enabled':False,
          'reuse_nominal':str(a.reuse_nominal) if a.reuse_nominal else None,
          'requested_disturbed_candidates':a.disturbed_candidates,
          'stage':'exploratory pilot, not held-out validation',
          'normalization':'existing native fixed 5% denominator, unchanged'}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
    summary=[];eligible=[]
    def run(candidate,amplitudes,stage,onset=.5):
        c=next(x for x in bank if x['candidate']==candidate)
        high=c['high_fraction'];dest=a.output/f'{stage}_c{candidate}'
        sys.argv=['benchmark','--screen',str(a.bank),'--candidates',str(candidate),
            '--static-fractions','.025',str(high),'--dynamic-max',str(high),
            '--seeds','807','--output',str(dest),'--use-banked-reference',
            '--onset',str(onset),
            '--ramp',str(a.ramp),
            '--amplitudes',*[str(v) for v in amplitudes]]
        benchmark.main()
        rows=json.loads((dest/'metrics.json').read_text())
        for row in rows:
            row['directory']=str(dest);row['stage']=stage
        summary.extend(rows)
        (a.output/'search_results.json').write_text(json.dumps(summary,indent=2))
        return rows
    def safe(m):
        return m['position_violation_max_rad']<=1e-6 and m['first_collision_index'] is None
    for candidate in candidates:
        if a.reuse_nominal:
            directory=a.reuse_nominal/f'nominal_c{candidate}'
            rows=json.loads((directory/'metrics.json').read_text())
            rows=[{**r,'directory':str(directory),'stage':'nominal'} for r in rows]
            summary.extend(rows)
            (a.output/'search_results.json').write_text(json.dumps(summary,indent=2))
        else:rows=run(candidate,[0.],'nominal')
        by={x['case']:x['metrics'] for x in rows}
        dyn=next(m for n,m in by.items() if n.startswith('directional'))
        high=by['static75' if next(x for x in bank if x['candidate']==candidate)['high_fraction']==.075 else 'static5']
        if (all(safe(m) for m in by.values()) and dyn['tracking_rmse_mm']<=10.
                and high['tracking_rmse_mm']-dyn['tracking_rmse_mm']>=2.
                and high['tracking_rmse_mm']>=1.2*dyn['tracking_rmse_mm']):
            eligible.append(candidate)
    eligible=[c for c in eligible if c in a.disturbed_candidates] if a.disturbed_candidates else eligible[:2]
    successes=[]
    for candidate in eligible:
        with np.load((a.reuse_nominal or a.output)/f'nominal_c{candidate}'/'s0_seed807_static25.npz') as z:
            q=z['q'][:,0]
            threshold=q[0]+.7*(q.max()-q[0])
            crossings=np.flatnonzero((q[:-1]>=threshold)&(np.diff(q)>0))
            onset=max(.5,float(crossings[0]*.02)) if len(crossings) else .5
        rows=run(candidate,plan['amplitudes_nm'],'disturbed',onset if a.onset_mode=='nominal-phase' else .5)
        for amplitude in plan['amplitudes_nm']:
            group=[r for r in rows if r['amplitude']==amplitude]
            by={r['case']:r['metrics'] for r in group}
            low=by['static25'];dyn=next(m for n,m in by.items() if n.startswith('directional'))
            high=by['static75' if 'static75' in by else 'static5']
            passes=(low['position_violation_max_rad']>1e-4 and low['test_joint_saturation_steps']>0
                and safe(dyn) and safe(high) and all(m['first_collision_index'] is None for m in by.values())
                and dyn['tracking_rmse_mm']<=25.
                and high['tracking_rmse_mm']-dyn['tracking_rmse_mm']>=2.)
            if passes:successes.append({'candidate':candidate,'amplitude':amplitude})
    verdict={'nominal_eligible':eligible,'pilot_full_gate_passes':successes,
             'status':'pilot_candidate_requires_heldout_validation' if successes else 'not_found_in_bounded_pilot',
             'not_an_impossibility_certificate':True}
    (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print('SEARCH_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
