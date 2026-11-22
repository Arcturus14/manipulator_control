"""Screen a clean nominal payload-circle baseline before changing to J3 disturbance.

CUDA only. Existing configurations and results remain unchanged.
"""
import copy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import yaml

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT
from kroc.run_payload_cube_tcp import run_payload
from kroc.study_payload_torque_stability import metrics, write
from kroc.view_payload_tcp import load_result, save_outputs

BASE = Path('configs/payload_cube_tcp_high_workload_yz_circle_t2_n0.yaml')
REUSED_T2 = Path('kroc/results/payload_high_workload_circle_20261122/20261122_024806/A_a0_seed6807')
PILOT_SEED = 6807
HELDOUT = [6817, 6827]
VARIANTS = [(2.0, False), (2.0, True), (2.5, False), (2.5, True), (3.0, False), (3.0, True)]


def summarize(z, v, period):
    row = metrics(z, v)
    n = len(z['u_requested'])
    t = np.arange(n)*DT
    steady = t >= 3.0+period
    du = np.diff(z['u_applied'], axis=0)/DT
    row.update(
        steady_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][steady]**2))),
        steady_max_error_mm=float(np.max(z['err_mm'][steady])),
        torque_rate_rms=float(np.sqrt(np.mean(du**2))),
        torque_rate_per_joint=np.sqrt(np.mean(du**2, axis=0)).tolist(),
        ess_steady_mean=float(np.mean(z['mppi_weight_ess'][steady])),
        j2_request_abs_peak_nm=float(np.max(np.abs(z['u_requested'][steady, 1]))),
        j3_request_abs_peak_nm=float(np.max(np.abs(z['u_requested'][steady, 2]))),
        q3_min_rad=float(np.min(z['q'][:, 2])),
        q3_max_rad=float(np.max(z['q'][:, 2])))
    return row


def config_for(base, period, reflected):
    cfg = copy.deepcopy(base)
    cfg['scene']['period_s'] = float(period)
    cfg['scene']['duration_s'] = 18.0 if period >= 2.5 else 16.0
    cfg['scene']['use_dob'] = False
    cfg['scene']['disturbance_amplitude_nm'] = 0.0
    cfg['controller']['experimental_stabilization']['reflected_inertia_feedforward'] = bool(reflected)
    return cfg


def main():
    cuda = cuda_preflight()
    base = yaml.safe_load(BASE.read_text())
    root = Path('kroc/results/payload_high_workload_nominal_improvement_20261122')/datetime.now().strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True)
    write(root/'plan.json', dict(cuda=cuda, base=str(BASE), pilot_seed=PILOT_SEED,
        heldout=HELDOUT, variants=[dict(period_s=p, reflected_inertia_feedforward=f) for p, f in VARIANTS],
        fixed='5 kg cube-center TCP, YZ radius 0.20 m, same center/posture, N0 K136 H10 lambda64, DOB and disturbance off.',
        selection='Safe/no clipping; torque-rate <=1.2x current T2 baseline; steady RMSE <=6 mm. Select shortest period, then lowest RMSE. Stop if none.',
        next='Only after heldout validation: keep fixed candidate and change disturbance joint to J3.'))
    rows=[]
    baseline=None
    for period, reflected in VARIANTS:
        name=f'T{period:g}_ff{int(reflected)}'
        if period == 2.0 and not reflected:
            z=load_result(REUSED_T2);v=json.loads((REUSED_T2/'verification.json').read_text());path=REUSED_T2
        else:
            cfg=config_for(base,period,reflected)
            z,v,path=run_payload(cfg,PILOT_SEED,'N0',output=root/name)
            save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        row=summarize(z,v,period)
        row.update(name=name,period_s=period,reflected_inertia_feedforward=reflected,
                   seed=PILOT_SEED,path=str(path))
        rows.append(row)
        if name=='T2_ff0':baseline=row
        write(root/'pilot_metrics.json',rows)
        print('PILOT',name,json.dumps({k:row[k] for k in ['rmse_err_mm','steady_rmse_mm','steady_max_error_mm','torque_rate_rms','j2_request_abs_peak_nm','j3_request_abs_peak_nm','ess_steady_mean','safe','clipping_steps']}),flush=True)
    eligible=[r for r in rows if r['safe'] and r['clipping_steps']==0 and
              r['steady_rmse_mm']<=6.0 and r['torque_rate_rms']<=1.2*baseline['torque_rate_rms']]
    if not eligible:
        write(root/'selection.json',dict(selected=None,status='no_candidate',baseline=baseline,
            reason='No pilot met the predeclared nominal accuracy and torque-rate gates; J3 disturbance was not started.'))
        print('NO_CANDIDATE','[OUTPUT]',root,flush=True);return
    selected=min(eligible,key=lambda r:(r['period_s'],r['steady_rmse_mm']))
    held=[]
    cfg=config_for(base,selected['period_s'],selected['reflected_inertia_feedforward'])
    for seed in HELDOUT:
        path=root/f"{selected['name']}_seed{seed}"
        z,v,path=run_payload(cfg,seed,'N0',output=path)
        row=summarize(z,v,selected['period_s']);row.update(name=selected['name'],seed=seed,path=str(path))
        held.append(row);write(root/'heldout_metrics.json',held)
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
    passed=all(r['safe'] and r['clipping_steps']==0 and r['steady_rmse_mm']<=6.0 and
               r['torque_rate_rms']<=1.2*baseline['torque_rate_rms'] for r in held)
    fixed=copy.deepcopy(cfg)
    fixed_path=root/'selected_config.yaml'
    if passed:fixed_path.write_text(yaml.safe_dump(fixed,sort_keys=False))
    write(root/'selection.json',dict(selected=selected,heldout=held,passed=passed,
        fixed_config=str(fixed_path) if passed else None,
        next=('Screen J3 DOB/disturbance saturation only.' if passed else 'Do not start J3 screen.')))
    print('DONE',passed,selected['name'],'[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
