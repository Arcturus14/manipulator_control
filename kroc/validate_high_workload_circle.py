"""Staged N0 validation of the opt-in fast YZ payload circle.

CUDA only. Stage A: DOB/disturbance off. Stage B: DOB on, disturbance off.
Stage C: fixed low-frequency J2 disturbance screen. No N1/D0 or retuning.
"""
import copy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import yaml

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT
from kroc.find_payload_n0_saturation import statistics
from kroc.run_payload_cube_tcp import run_payload
from kroc.study_payload_torque_stability import metrics, write
from kroc.view_payload_tcp import load_result, save_outputs

DEFAULT_CONFIG=Path('configs/payload_cube_tcp_high_workload_yz_circle_n0.yaml')
SEEDS=[6807,6817,6827]
PERIOD=1.5
RAMP=3.0


def cycle_metrics(z):
    n=len(z['u_requested']);t=np.arange(n)*DT
    starts=np.arange(RAMP+PERIOD,t[-1]-PERIOD+1e-9,PERIOD)
    cycles=[]
    for start in starts:
        mask=(t>=start)&(t<start+PERIOD)
        loss=np.abs(z['u_clamp_loss'][mask,1])
        cycles.append(dict(start_s=float(start),rmse_mm=float(np.sqrt(np.mean(z['err_mm'][mask]**2))),
            max_error_mm=float(z['err_mm'][mask].max()),j2_requested_min_nm=float(z['u_requested'][mask,1].min()),
            j2_clipped_steps=int((loss>1e-5).sum()),j2_clipping_nm_s=float(loss.sum()*DT),
            ess_mean=float(z['mppi_weight_ess'][mask].mean())))
    return cycles


def summarize(z,v):
    row=metrics(z,v);n=len(z['u_requested']);t=np.arange(n)*DT;steady=t>=RAMP+PERIOD
    loss=np.abs(z['u_clamp_loss'][:,1]);mask=loss>1e-5
    edges=np.diff(np.r_[False,mask,False].astype(int));lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    row.update(steady_start_s=RAMP+PERIOD,
        steady_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][steady]**2))),
        steady_max_error_mm=float(z['err_mm'][steady].max()),
        steady_j2_requested_min_nm=float(z['u_requested'][steady,1].min()),
        steady_j2_fraction_below_minus120=float((z['u_requested'][steady,1]<=-120).mean()),
        steady_j2_fraction_below_minus130=float((z['u_requested'][steady,1]<=-130).mean()),
        j2_clip_steps=int(mask.sum()),j2_clip_max_nm=float(loss.max()),
        j2_clip_nm_s=float(loss.sum()*DT),longest_j2_clip_s=float(max(lengths,default=0)*DT),
        q3_min_rad=float(z['q'][:,2].min()),q3_max_rad=float(z['q'][:,2].max()),
        joint_speed_abs_max_rad_s=np.abs(z['qd']).max(0).tolist(),
        ess_steady_mean=float(z['mppi_weight_ess'][steady].mean()),
        d_hat_j2_steady_rms_nm=float(np.sqrt(np.mean(z['d_hat_used'][steady,1]**2))),
        d_hat_j2_steady_peak_nm=float(np.abs(z['d_hat_used'][steady,1]).max()),
        cycles=cycle_metrics(z))
    return row


def main():
    global PERIOD, RAMP
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--resume',type=Path)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);args=p.parse_args()
    cuda=cuda_preflight();cfg=yaml.safe_load(args.config.read_text())
    PERIOD=float(cfg['scene']['period_s']);RAMP=float(cfg['scene']['reference_start_ramp_s'])
    root=args.resume or Path('kroc/results/payload_high_workload_circle_20261122')/datetime.now().strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True,exist_ok=True)
    write(root/'plan.json',dict(cuda=cuda,config=str(args.config),seeds=SEEDS,
        geometry=(f"5 kg cube-center TCP; YZ circle radius{cfg['scene']['circle_radius_m']:.2f}m; "
                  f"center horizontal radius0.72m, z0.30m; period{PERIOD:g}s; speed ramp{RAMP:g}s; "
                  f"duration{cfg['scene']['duration_s']:g}s."),
        controller='Frozen N0: K136,H10,lambda64,normalized covariance,posture5000,boundary/smooth4; physical limits unchanged.',
        stages=['A: DOB/disturbance OFF','B: DOB ON, disturbance OFF','C: DOB ON, constant-after-ramp low-frequency negative J2 disturbance'],
        gate_A='All seeds safe, no clipping, q3>0.2rad, total RMSE<=8mm, steady RMSE<=9mm, max joint speed<pi.',
        gate_B='All seeds safe, no clipping, total RMSE<=9mm and <=1.2x paired A. Record spurious d_hat; no filter retuning.',
        disturbance_pilot_nm=[15,20,25,30,35,37.5,40],scope='N0 only. Stop before disturbance if a gate fails; retain every outcome.'))
    records=[]
    if (root/'runs.json').exists():records=json.loads((root/'runs.json').read_text())
    def run(stage,seed,amp=0.):
        existing=next((r for r in records if r['stage']==stage and r['seed']==seed and r['amplitude_abs_nm']==amp),None)
        if existing:return existing
        conf=copy.deepcopy(cfg);conf['scene']['use_dob']=stage!='A';conf['scene']['disturbance_amplitude_nm']=-float(amp)
        path=root/f'{stage}_a{amp:g}_seed{seed}'
        if (path/'verification.json').exists():z=load_result(path);v=json.loads((path/'verification.json').read_text())
        else:
            if path.exists():path=path/'retry'
            z,v,path=run_payload(conf,seed,'N0',output=path)
        row=summarize(z,v);row.update(stage=stage,seed=seed,amplitude_abs_nm=amp,path=str(path))
        records.append(row);write(root/'runs.json',records)
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=stage!='A',csv=True)
        print('RESULT',stage,seed,amp,json.dumps({k:row[k] for k in ['rmse_err_mm','steady_rmse_mm','safe','clipping_steps','j2_clip_steps','longest_j2_clip_s','steady_j2_requested_min_nm','q3_min_rad','ess_mean','d_hat_j2_steady_rms_nm']}),flush=True)
        return row
    A=[run('A',s) for s in SEEDS]
    gateA=all(r['safe'] and r['clipping_steps']==0 and r['q3_min_rad']>.2 and
              r['rmse_err_mm']<=8 and r['steady_rmse_mm']<=9 and
              max(r['joint_speed_abs_max_rad_s'])<np.pi for r in A)
    write(root/'gate_A.json',dict(passed=gateA,rows=A))
    if not gateA:
        write(root/'assessment.json',dict(status='stopped_at_nominal_gate',gate_A=A,
            conclusion='Fast-circle candidate did not establish a clean N0 nominal baseline; no DOB/disturbance runs started.'))
        print('STOP_GATE_A','[OUTPUT]',root,flush=True);return
    B=[run('B',s) for s in SEEDS]
    paired={r['seed']:r for r in A}
    gateB=all(r['safe'] and r['clipping_steps']==0 and r['rmse_err_mm']<=9 and
              r['rmse_err_mm']<=1.2*paired[r['seed']]['rmse_err_mm'] for r in B)
    write(root/'gate_B.json',dict(passed=gateB,rows=B))
    if not gateB:
        write(root/'assessment.json',dict(status='stopped_at_DOB_zero_gate',gate_A=A,gate_B=B,
            conclusion='DOB-on zero-disturbance baseline failed; no disturbance runs started and filters were not retuned.'))
        print('STOP_GATE_B','[OUTPUT]',root,flush=True);return
    pilot=[]
    for amp in [15.,20.,25.,30.,35.,37.5,40.]:
        row=run('C',SEEDS[0],amp);pilot.append(row)
        if not row['safe'] or row['rmse_err_mm']>15:break
    eligible=[r for r in pilot if r['safe'] and r['j2_clip_steps']>0 and r['longest_j2_clip_s']>=.2]
    if not eligible:
        write(root/'assessment.json',dict(status='no_sustained_clipping_candidate',gate_A=A,gate_B=B,pilot=pilot,
            conclusion='No safe pilot produced >=0.2s continuous J2 clipping; no N1/D0 comparison selected.'))
        print('NO_CANDIDATE','[OUTPUT]',root,flush=True);return
    selected=min(eligible,key=lambda r:r['amplitude_abs_nm'])['amplitude_abs_nm']
    held=[run('C',s,selected) for s in SEEDS[1:]]
    validation=[next(r for r in pilot if r['amplitude_abs_nm']==selected),*held]
    passed=all(r['safe'] and r['j2_clip_steps']>0 and r['longest_j2_clip_s']>=.2 for r in validation)
    write(root/'assessment.json',dict(status='completed_N0_candidate_validation',gate_A=A,gate_B=B,
        disturbance_pilot=pilot,selected_amplitude_abs_nm=selected,validation=validation,passed=passed,
        conclusion=('Candidate is ready for fixed N0/N1/D0 comparison.' if passed else
                    'Sustained clipping did not reproduce across seeds; do not start N0/N1/D0 comparison.')))
    print('DONE',passed,'selected',selected,'[OUTPUT]',root,flush=True)

if __name__=='__main__':main()
