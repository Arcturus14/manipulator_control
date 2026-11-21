"""Frozen N0, CUDA-only circular scene validation; no disturbance/DOB."""
import copy
import json
from pathlib import Path
import numpy as np
import yaml
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.run_payload_cube_tcp import run_payload
from kroc.study_payload_torque_stability import metrics,write
from kroc.view_payload_tcp import save_outputs,load_result
from kroc.screen_payload_circle_workload import ROOT,BASE


def summarize(z,v,period):
    row=metrics(z,v);u=z['u_requested'];n=len(u);t=np.arange(n)*.02
    active=t>=max(1.,period);j2=-u[:,1]
    edges=np.diff(np.r_[False,(j2>=130)&active,False].astype(int))
    start=np.flatnonzero(edges==1);end=np.flatnonzero(edges==-1)
    row.update(steady_start_s=max(1.,period),
        steady_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][active]**2))),
        steady_max_err_mm=float(z['err_mm'][active].max()),
        steady_J2_abs_peak_nm=float(np.abs(u[active,1]).max()),
        steady_J2_negative_peak_nm=float(j2[active].max()),
        steady_J2_q99_abs_nm=float(np.quantile(np.abs(u[active,1]),.99)),
        steady_J2_fraction_below_minus120=float((j2[active]>=120).mean()),
        steady_J2_fraction_below_minus130=float((j2[active]>=130).mean()),
        steady_J2_fraction_below_minus140=float((j2[active]>=140).mean()),
        longest_130Nm_s=float(max(end-start,default=0)*.02),
        steady_ess_mean=float(z['mppi_weight_ess'][active].mean()),
        joint_speed_abs_max_rad_s=np.abs(z['qd']).max(0).tolist())
    assert not bool(z['payload_use_dob']) and not np.any(z['d_true'])
    assert str(z['compute_device'])=='cuda' and str(z['input_study_case'])=='N0'
    np.testing.assert_allclose(z['u_requested'],z['u_mppi'],atol=0)
    return row


def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--heldout',action='store_true');a=ap.parse_args()
    cuda=cuda_preflight();base=yaml.safe_load(BASE.read_text());root=ROOT/'cuda';root.mkdir(parents=True,exist_ok=True)
    candidates=json.loads((ROOT/'offline/demands.json').read_text());rows=[];configs={}
    grid=[('yz_far16',4.),('yz_far16',2.),('yz_far16',1.5),('yz20',1.5),('xz12',1.5),('xy12',1.5)]
    write(root/'plan.json',dict(cuda=cuda,pilot_seed=6807,grid=grid,heldout=[6817,6827],
        constants='N0 frozen,5kg,6.3cm cube,physical torque limits/covariance/lambda/costs unchanged; DOB/disturbance OFF.',
        selection='Safe and no clipping; total and post-first-cycle RMSE<=3mm; repeated nominal J2 negative torque>=130Nm for >=1% of steady steps. All outcomes retained. If none, circle target not met; consider rectangle next.',
        tracking_tradeoff='3mm is a prospective screen tolerance, not equality with the previous1.2mm nominal tracking. Report both. Startup peaks excluded from loading score, not from safety/clipping/RMSE.'))
    def run(name,T,seed):
        label=f'{name}_T{T:g}_seed{seed}';path=root/label
        cfg=copy.deepcopy(base);d=next(r for r in candidates if r['name']==name and r['period_s']==T)
        cfg['scene'].update(initial_q=d['initial_q'],trajectory=d['shape'],circle_radius_m=d['radius_m'],period_s=T,
            duration_s=20.,use_dob=False,disturbance_amplitude_nm=0.)
        cfg['output_root']=str(ROOT/'replay');configs[label]=cfg
        if (path/'verification.json').exists():z=load_result(path);v=json.loads((path/'verification.json').read_text())
        else:z,v,path=run_payload(cfg,seed,'N0',output=path)
        row=summarize(z,v,T);row.update(name=name,period_s=T,seed=seed,path=str(path))
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        return label,row
    for name,T in grid:
        label,row=run(name,T,6807);rows.append(row);write(root/'pilot_metrics.json',rows)
        print('RESULT',label,json.dumps(row),flush=True)
    eligible=[r for r in rows if r['safe'] and r['clipping_steps']==0 and r['rmse_err_mm']<=3 and r['steady_rmse_mm']<=3 and r['steady_J2_fraction_below_minus130']>=.01]
    write(root/'selection.json',dict(eligible=eligible,target_met=bool(eligible),scope='Pilot geometry/timing selection only; no N1/D0 comparison.'))
    if eligible and a.heldout:
        best=min(eligible,key=lambda r:(r['rmse_err_mm'],r['torque_hp5_rms']));label=f'{best["name"]}_T{best["period_s"]:g}_seed6807';cfg=configs[label]
        held=[]
        for seed in [6817,6827]:
            _,r=run(best['name'],best['period_s'],seed);held.append(r);write(root/'heldout_metrics.json',held)
        passed=all(r['safe'] and not r['clipping_steps'] and r['rmse_err_mm']<=3 and r['steady_rmse_mm']<=3 and r['steady_J2_fraction_below_minus130']>=.01 for r in held)
        write(root/'fixed_candidate_validation.json',dict(pilot=best,heldout=held,passed=passed))
        if passed:
            preset=Path('configs/payload_cube_tcp_high_workload_circle_n0.yaml')
            if preset.exists():raise FileExistsError(preset)
            preset.write_text('# Additional high-workload circle. Frozen controller, DOB/disturbance OFF.\n'+yaml.safe_dump(cfg,sort_keys=False))
    print('[OUTPUT]',root,flush=True)

if __name__=='__main__':main()
