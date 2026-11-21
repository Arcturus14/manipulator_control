"""CUDA-only stage2: frozen N0 on C2 rounded rectangles, no DOB/load."""
import json
from pathlib import Path
import numpy as np
import yaml
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.run_payload_cube_tcp import run_payload
from kroc.validate_payload_circle_workload import summarize
from kroc.study_payload_torque_stability import write
from kroc.view_payload_tcp import load_result,save_outputs
from kroc.screen_payload_rectangle_workload import ROOT


def main():
    cuda=cuda_preflight();root=ROOT/'cuda';root.mkdir(parents=True,exist_ok=True)
    ranked=json.loads((ROOT/'offline/ranked_candidates.json').read_text())
    grid=[('xz12_r0.01',8.),('yz20_r0.02',6.),('yz20_r0.04',4.),('yz16_r0.02',6.),
          ('xz12_r0.02',5.),('yz16_r0.02',5.)]
    plan_path=root/('plan_followup.json' if (root/'plan.json').exists() else 'plan.json')
    write(plan_path,dict(cuda=cuda,pilot_seed=6807,grid=grid,heldout_seeds=[6817,6827],
        adaptive_pilot_note='Last two candidates added after first four pilot outcomes: broaden XZ corner while increasing speed; increase YZ16 period6 to5s. Both already passed offline physical-demand screen. Same gate; no heldout reselection.',
        controller='Frozen N0 only; no DOB/disturbance, no covariance/lambda/cost/feedforward changes.',
        selection='Same tracking/loading gate as circle: safe,no clipping,total+post-first-cycle RMSE<=3mm,negative J2>=130Nm for>=1% of post-first-cycle steps. Fix best pilot then two independent seeds; no reselection.',
        reference='Quintic Bezier C2 rounded rectangle,1s C2 start speed ramp,24s run; corner traversal>=0.12s. No physical limit lowering or payload increase.'))
    rows=[];configs={}
    def run(item,seed):
        name=item['name'];T=item['period_s'];label=f'{name}_T{T:g}_seed{seed}';cfg=item['config'];configs[label]=cfg;path=root/label
        if (path/'verification.json').exists():z=load_result(path);v=json.loads((path/'verification.json').read_text())
        else:z,v,path=run_payload(cfg,seed,'N0',output=path)
        row=summarize(z,v,T);row.update(name=name,period_s=T,seed=seed,path=str(path),
            corner_radius_m=item['corner_radius_m'],corner_traversal_s=item['corner_traversal_s'],
            prescreen_j2_negative_peak_nm=item['j2_negative_peak_nm'])
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        print('RESULT',label,json.dumps(row),flush=True)
        return row
    for name,T in grid:
        item=next(r for r in ranked if r['name']==name and r['period_s']==T)
        rows.append(run(item,6807));write(root/'pilot_metrics.json',rows)
    def qualifies(r):return r['safe'] and not r['clipping_steps'] and r['rmse_err_mm']<=3 and r['steady_rmse_mm']<=3 and r['steady_J2_fraction_below_minus130']>=.01
    eligible=[r for r in rows if qualifies(r)]
    write(root/'selection.json',dict(target_met=bool(eligible),eligible=eligible))
    if eligible:
        best=min(eligible,key=lambda r:(r['rmse_err_mm'],r['torque_hp5_rms']))
        item=next(r for r in ranked if r['name']==best['name'] and r['period_s']==best['period_s'])
        held=[]
        for seed in [6817,6827]:held.append(run(item,seed));write(root/'heldout_metrics.json',held)
        passed=all(qualifies(r) for r in held)
        write(root/'fixed_candidate_validation.json',dict(pilot=best,heldout=held,passed=passed))
        if passed:
            p=Path('configs/payload_cube_tcp_high_workload_rectangle_n0.yaml')
            if p.exists():raise FileExistsError(p)
            p.write_text('# Additional high-workload C2 rectangle; frozen N0, DOB/disturbance OFF.\n'+yaml.safe_dump(item['config'],sort_keys=False))
    print('[OUTPUT]',root,flush=True)

if __name__=='__main__':main()
