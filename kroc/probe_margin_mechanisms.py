"""Predeclared bounded follow-up: disturbance phase and joint-role separation."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from kroc import benchmark_nominal_margin_pressure as benchmark


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path('kroc/results')
    nominal=root/'shape_margin_tradeoff_20261113/nominal_c22/s0_seed807_static25.npz'
    with np.load(nominal) as z:q=z['q'][:,0].copy()
    onset=float(np.flatnonzero((q[:-1]>=q[0]+.7*(q.max()-q[0]))&(np.diff(q)>0))[0]*.02)
    jobs=[dict(name='phase_rectangle',bank='shape_margin_bank_20261113',candidate=22,
               amplitudes=[147.],joint=0,onset=onset,ramp=.1),
          dict(name='coupled_wrist',bank='coupled_wrist_margin_bank_20261113',candidate=9,
               amplitudes=[0.,27.4,27.55],joint=4,onset=1.,ramp=.2)]
    (a.output/'plan.json').write_text(json.dumps({'seed':807,'jobs':jobs,
        'headroom_enabled':False,'phase_selected_from_nominal_only':True,
        'full_gate':'low violation >1e-4 rad plus tested-joint saturation; dynamic and high safe; all collision-free; dynamic RMSE <=25mm and >=2mm better than high',
        'status':'exploratory, not held-out validation'},indent=2))
    summary=[];passes=[]
    for job in jobs:
        dest=a.output/job['name']
        sys.argv=['benchmark','--screen',str(root/job['bank']),'--candidates',str(job['candidate']),
            '--static-fractions','.025','.05','--dynamic-max','.05','--seeds','807',
            '--output',str(dest),'--use-banked-reference','--disturbance-joint',str(job['joint']),
            '--onset',str(job['onset']),'--ramp',str(job['ramp']),
            '--amplitudes',*[str(x) for x in job['amplitudes']]]
        benchmark.main()
        rows=json.loads((dest/'metrics.json').read_text())
        summary.extend([{**r,'directory':str(dest),'stage':job['name']} for r in rows])
        (a.output/'search_results.json').write_text(json.dumps(summary,indent=2))
        for amplitude in job['amplitudes']:
            by={r['case']:r['metrics'] for r in rows if r['amplitude']==amplitude}
            low,dyn,high=[by[n] for n in ('static25','directional25_5','static5')]
            if (low['position_violation_max_rad']>1e-4 and low['test_joint_saturation_steps']>0
                and dyn['position_violation_max_rad']<=1e-6 and high['position_violation_max_rad']<=1e-6
                and all(m['first_collision_index'] is None for m in by.values())
                and dyn['tracking_rmse_mm']<=25.
                and high['tracking_rmse_mm']-dyn['tracking_rmse_mm']>=2.):
                passes.append({'job':job['name'],'amplitude':amplitude})
    verdict={'pilot_full_gate_passes':passes,'heldout_validated':False,
        'status':'pilot_candidate_requires_validation' if passes else 'not_found_in_bounded_followup'}
    (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print('PROBE_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
