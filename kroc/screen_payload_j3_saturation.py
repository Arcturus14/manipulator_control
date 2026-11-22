"""N0-only J3 saturation screen on the fixed improved payload baseline."""
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
from kroc.view_payload_tcp import save_outputs

BASE=Path('kroc/results/payload_high_workload_nominal_improvement_20261122/20261122_033823/selected_config.yaml')
SEEDS=[6807,6817,6827]
AMPLITUDES=[80.,90.,95.,100.,105.,110.]
STEADY_START=6.0


def summarize(z,v):
    row=metrics(z,v);n=len(z['u_requested']);t=np.arange(n)*DT;steady=t>=STEADY_START
    loss=np.abs(z['u_clamp_loss'][:,2]);mask=loss>1e-5
    edges=np.diff(np.r_[False,mask,False].astype(int));lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    row.update(steady_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][steady]**2))),
        steady_max_error_mm=float(np.max(z['err_mm'][steady])),
        j3_request_min_nm=float(np.min(z['u_requested'][steady,2])),
        j3_clip_steps=int(mask.sum()),j3_clip_nm_s=float(loss.sum()*DT),
        j3_clip_energy_nm2_s=float(np.square(loss).sum()*DT),
        j3_clip_max_nm=float(loss.max()),longest_j3_clip_s=float(max(lengths,default=0)*DT),
        d_hat_j3_steady_rms_nm=float(np.sqrt(np.mean(z['d_hat_used'][steady,2]**2))),
        ess_steady_mean=float(np.mean(z['mppi_weight_ess'][steady])))
    return row


def main():
    cuda=cuda_preflight();base=yaml.safe_load(BASE.read_text())
    root=Path('kroc/results/payload_j3_saturation_20261122')/datetime.now().strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True)
    write(root/'plan.json',dict(cuda=cuda,base=str(BASE),seeds=SEEDS,amplitudes_nm=AMPLITUDES,
        fixed='T3.0s, radius0.20m, 5kg payload, N0 frozen controller; disturbance joint J3; raised-cosine 0.5--2.5s then hold.',
        candidate='Smallest safe amplitude with longest J3 clipping >=0.2s and full RMSE >=1.05x paired DOB-zero baseline.',
        scope='N0 only. No N1/D0 until saturation demonstrably worsens tracking.'))
    rows=[]
    def run(seed,amp):
        cfg=copy.deepcopy(base);cfg['scene'].update(use_dob=True,disturbance_joint=3,disturbance_amplitude_nm=-float(amp))
        path=root/f"a{amp:g}_seed{seed}"
        z,v,path=run_payload(cfg,seed,'N0',output=path)
        row=summarize(z,v);row.update(seed=seed,amplitude_abs_nm=amp,path=str(path));rows.append(row);write(root/'runs.json',rows)
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=True,csv=True)
        print('RESULT',seed,amp,json.dumps({k:row[k] for k in ['rmse_err_mm','steady_rmse_mm','safe','j3_clip_steps','longest_j3_clip_s','j3_clip_nm_s','j3_clip_max_nm','j3_request_min_nm','d_hat_j3_steady_rms_nm','ess_steady_mean']}),flush=True)
        return row
    baseline=run(SEEDS[0],0.)
    if not baseline['safe'] or baseline['clipping_steps']:
        write(root/'assessment.json',dict(status='failed_DOB_zero_baseline',baseline=baseline));print('STOP_BASELINE','[OUTPUT]',root);return
    pilot=[]
    for amp in AMPLITUDES:
        row=run(SEEDS[0],amp);pilot.append(row)
        if not row['safe'] or row['rmse_err_mm']>1.8*baseline['rmse_err_mm']:break
    eligible=[r for r in pilot if r['safe'] and r['longest_j3_clip_s']>=.2 and r['rmse_err_mm']>=1.05*baseline['rmse_err_mm']]
    if not eligible:
        write(root/'assessment.json',dict(status='no_tracking_degradation_candidate',baseline=baseline,pilot=pilot,
            conclusion='J3 screen did not jointly produce horizon-length clipping and >=5% RMSE degradation.'))
        print('NO_CANDIDATE','[OUTPUT]',root);return
    selected=min(eligible,key=lambda r:r['amplitude_abs_nm'])['amplitude_abs_nm']
    held=[run(seed,selected) for seed in SEEDS[1:]]
    validation=[next(r for r in pilot if r['amplitude_abs_nm']==selected),*held]
    # Each heldout seed gets its own DOB-zero baseline before the paired claim.
    held_base=[run(seed,0.) for seed in SEEDS[1:]]
    base_by_seed={baseline['seed']:baseline,**{r['seed']:r for r in held_base}}
    passed=all(r['safe'] and r['longest_j3_clip_s']>=.2 and
               r['rmse_err_mm']>=1.05*base_by_seed[r['seed']]['rmse_err_mm'] for r in validation)
    write(root/'assessment.json',dict(status='completed',selected_amplitude_abs_nm=selected,
        baselines=[baseline,*held_base],validation=validation,passed=passed,
        conclusion=('Ready for fixed N0/N1/D0 comparison.' if passed else 'Tracking degradation did not reproduce across seeds.')))
    print('DONE',passed,'selected',selected,'[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
