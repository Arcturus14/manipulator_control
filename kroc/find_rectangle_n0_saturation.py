"""Empirical J2 DOB compensation saturation onset for fixed rounded-rectangle N0.

Runs only after the feedforward choice is fixed. CUDA required. Each outcome is
stored even if tracking or geometry fails; there is no controller retuning.
"""
import copy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import yaml

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.compare_payload_feedforward import BASE, OLD, SEEDS
from kroc.find_payload_n0_saturation import statistics
from kroc.run_payload_cube_tcp import run_payload
from kroc.study_payload_torque_stability import write
from kroc.view_payload_tcp import load_result, save_outputs


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--resume',type=Path)
    args=p.parse_args()
    cuda=cuda_preflight()
    decision=json.loads(args.selection.read_text())
    chosen=decision['selected']
    cfg=yaml.safe_load(BASE.read_text())
    if chosen=='reflected_inertia_ff_on':
        cfg['controller']['experimental_stabilization']['reflected_inertia_feedforward']=True
    elif chosen!='existing_residual_only':raise ValueError(chosen)
    cfg['scene'].update(use_dob=True,disturbance_joint=2,disturbance_amplitude_nm=0.)
    root=args.resume or Path('kroc/results/payload_rectangle_saturation_20261122')/datetime.now().strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True,exist_ok=True)
    write(root/'plan.json',dict(cuda=cuda,feedforward_selection=str(args.selection),selected=chosen,
        seeds=SEEDS,trajectory='5 kg cube center TCP, rounded rectangle XZ 24x20 cm, corner radius 2 cm, period 5 s, 1 s start ramp; 24 s run.',
        disturbance='J2 negative raised cosine from t=0.5 to2.5 s, then constant. Plant effective torque = applied - true disturbance.',
        coarse_amplitudes_nm=[10,20,30,40,50],resolution_nm=1,clip_threshold_nm=1e-5,
        nominal_gate='3-seed DOB-on zero disturbance safe/no clipping, <=1.5x paired DOB-off RMSE and <=5 mm RMSE.',
        scope='N0 only. No N1/D0 or gain/filter/noise retuning. Stop upward search on unsafe or RMSE>10 mm.'))
    rows={}
    if (root/'runs.json').exists():
        for row in json.loads((root/'runs.json').read_text()):rows[(row['seed'],row['amplitude_abs_nm'])]=row
    def run(seed,amp):
        key=(seed,amp)
        if key in rows:return rows[key]
        config=copy.deepcopy(cfg);config['scene']['disturbance_amplitude_nm']=-float(amp)
        path=root/f'a{amp:02d}_seed{seed}'
        if (path/'verification.json').exists():
            z=load_result(path);v=json.loads((path/'verification.json').read_text())
        else:
            if path.exists():path=path/'retry'
            z,v,path=run_payload(config,seed,'N0',output=path)
        row=statistics(z,v)
        row.update(seed=seed,amplitude_abs_nm=amp,path=str(path))
        rows[key]=row;write(root/'runs.json',list(rows.values()))
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=True,csv=True)
        print('ONSET',seed,amp,json.dumps(dict(rmse=row['rmse_err_mm'],safe=row['safe'],
            clip_steps=row['clip_steps_J2'],clip_max=row['clip_J2_abs_max_nm'],jclip=row['jclip_J2_nm_s'],
            first=row['first_clip_s'])),flush=True)
        return row
    paired=json.loads((args.selection.parent/'paired_metrics.json').read_text())
    nominal={r['seed']:r for r in paired if r['variant']==chosen}
    zero={seed:run(seed,0) for seed in SEEDS}
    gate=all(zero[s]['safe'] and zero[s]['clip_steps_J2']==0 and
        zero[s]['rmse_err_mm']<=min(5.,1.5*nominal[s]['rmse_err_mm']) for s in SEEDS)
    write(root/'zero_gate.json',dict(passed=gate,paired_nominal=nominal,zero=zero))
    if not gate:
        write(root/'assessment.json',dict(status='stopped_at_DOB_zero_gate',selected=chosen,zero=zero))
        print('STOP_DOB_ZERO','[OUTPUT]',root,flush=True);return
    low=0;high=None
    for amp in [10,20,30,40,50]:
        row=run(6807,amp)
        if not row['safe'] or row['rmse_err_mm']>10:
            write(root/'assessment.json',dict(status='stopped_at_tracking_or_safety_failure',
                last_safe_no_clip_amplitude_nm=low,failed_amplitude_nm=amp,selected=chosen,failed=row,
                inference='No saturation onset established before failure.'))
            print('STOP_FAILURE','[OUTPUT]',root,flush=True);return
        if row['clip_steps_J2']:
            high=amp;break
        low=amp
    if high is None:
        write(root/'assessment.json',dict(status='no_clipping_through_50Nm',selected=chosen))
        print('NO_CLIP','[OUTPUT]',root,flush=True);return
    while high-low>1:
        mid=(high+low)//2
        row=run(6807,mid)
        if not row['safe'] or row['rmse_err_mm']>10:
            write(root/'assessment.json',dict(status='stopped_during_refinement_due_to_failure',
                checked_bracket_nm=[low,high],failed_amplitude_nm=mid,selected=chosen))
            print('STOP_REFINE','[OUTPUT]',root,flush=True);return
        if row['clip_steps_J2']:high=mid
        else:low=mid
    brackets={'6807':dict(no_clip_nm=low,first_clip_nm=high)}
    # Independent seeds: assess the fixed pilot amplitudes plus local neighbors.
    for seed in SEEDS[1:]:
        for amp in sorted(set([max(0,low-2),low,high,min(50,high+2),30])):
            run(seed,amp)
        seen=sorted([r for r in rows.values() if r['seed']==seed],key=lambda r:r['amplitude_abs_nm'])
        unclipped=[r['amplitude_abs_nm'] for r in seen if r['safe'] and r['rmse_err_mm']<=10 and r['clip_steps_J2']==0]
        clipped=[r['amplitude_abs_nm'] for r in seen if r['safe'] and r['rmse_err_mm']<=10 and r['clip_steps_J2']>0]
        brackets[str(seed)]=dict(highest_tested_no_clip_nm=max(unclipped) if unclipped else None,
            lowest_tested_clip_nm=min(clipped) if clipped else None)
    write(root/'assessment.json',dict(status='completed_empirical_N0_onset',selected=chosen,
        brackets=brackets,pilot_seed_resolution_nm=1,
        validation_30nm={str(seed):dict(clip_steps_J2=rows[(seed,30)]['clip_steps_J2'],
            jclip_J2_nm_s=rows[(seed,30)]['jclip_J2_nm_s'],
            rmse_mm=rows[(seed,30)]['rmse_err_mm'],safe=rows[(seed,30)]['safe']) for seed in SEEDS},
        interpretation='Observed onset for this trajectory, estimator, controller, and stochastic seeds; not a physical or universal disturbance threshold.',
        limits='N0 only; N1/D0 comparison remains future work.',all_outcomes_retained=True))
    print('DONE','[OUTPUT]',root,flush=True)

if __name__=='__main__':main()
