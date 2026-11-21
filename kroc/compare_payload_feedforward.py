"""Paired N0 rounded-rectangle feedforward ablation, then fixed choice for DOB screening.

The only controller change in the candidate is adding reflected armature inertia
times reference acceleration to the feedforward proposal. CUDA runs only.
"""
import copy
from datetime import datetime
import json
from pathlib import Path

import numpy as np
import yaml

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT, JOINT_ARMATURE
from kroc.run_payload_cube_tcp import run_payload
from kroc.study_payload_torque_stability import metrics, write
from kroc.view_payload_tcp import load_result, save_outputs

BASE = Path('configs/payload_cube_tcp_workload_rectangle_trial_n0.yaml')
OLD = Path('kroc/results/payload_rectangle_workload_20261121/cuda')
SEEDS = [6807, 6817, 6827]


def summarize(z, v):
    result = metrics(z, v)
    n = len(z['u_mppi'])
    ff = z['u_ff'][:n]
    raw = z['payload_u_ff_unclamped'][:n]
    residual = z['u_mppi'] - ff
    accel = np.gradient(np.gradient(z['q_reference'][:n], DT, axis=0, edge_order=2),
                        DT, axis=0, edge_order=2)
    armature = accel * np.asarray(JOINT_ARMATURE)
    target = armature[:, 1]
    r2 = residual[:, 1]
    steady = np.arange(n)*DT >= 5.
    active = np.abs(target[steady]) >= 2.
    result.update(
        ff_reflected_inertia_enabled=bool(z.get('payload_reflected_inertia_feedforward',False)),
        j2_missing_reflected_inertia_peak_nm=float(np.max(np.abs(target[steady]))),
        j2_missing_reflected_inertia_rms_nm=float(np.sqrt(np.mean(target[steady]**2))),
        j2_residual_rms_nm=float(np.sqrt(np.mean(r2[steady]**2))),
        j2_residual_vs_inertia_correlation=None if not active.any() else
            float(np.corrcoef(r2[steady][active], target[steady][active])[0, 1]),
        j2_ff_raw_peak_nm=float(np.max(np.abs(raw[steady, 1]))),
        j2_ff_clipped_fraction=float(np.mean(np.abs(raw[:, 1]-ff[:, 1]) > 1e-4)),
        j2_torque_peak_after_first_cycle_nm=float(np.max(np.abs(z['u_requested'][steady, 1]))),
        j2_torque_q99_after_first_cycle_nm=float(np.quantile(np.abs(z['u_requested'][steady, 1]), .99)),
        j2_torque_mean_abs_after_first_cycle_nm=float(np.mean(np.abs(z['u_requested'][steady, 1]))),
        j2_fraction_below_minus130_after_first_cycle=float(np.mean(z['u_requested'][steady, 1]<=-130.)),
        steady_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][steady]**2))))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    cuda = cuda_preflight()
    root = args.resume or Path('kroc/results/payload_feedforward_comparison_20261122') / datetime.now().strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True, exist_ok=True)
    base = yaml.safe_load(BASE.read_text())
    write(root/'plan.json', dict(cuda=cuda, seeds=SEEDS, base=str(BASE),
        comparison='Existing N0 vs opt-in N0 with armature*reference acceleration in feedforward; all else fixed.',
        original_primary_target='J2 peak>=130 Nm for >=1% steady steps with full RMSE<=3mm, zero clipping, safe.',
        selection='Choose FF only if safe/no clipping all seeds and improves mean RMSE by >=5% without >20% torque-rate increase; otherwise keep original. A higher J2 peak alone is not a benefit.',
        study_scope='No N1/D0 comparison or feedforward retuning. DOB and disturbance start only after fixed selection.'))
    rows=[]
    for seed in SEEDS:
        path_old=OLD/f'xz12_r0.02_T5_seed{seed}'
        z_old=load_result(path_old)
        v_old=json.loads((path_old/'verification.json').read_text())
        old=summarize(z_old,v_old)
        old.update(seed=seed,variant='existing_residual_only',path=str(path_old))
        rows.append(old)
        cfg=copy.deepcopy(base)
        cfg['controller']['experimental_stabilization']['reflected_inertia_feedforward']=True
        path=root/f'feedforward_on_seed{seed}'
        if (path/'verification.json').exists():
            z=load_result(path);v=json.loads((path/'verification.json').read_text())
        else:
            if path.exists(): path=path/'retry'
            z,v,path=run_payload(cfg,seed,'N0',output=path)
        current=summarize(z,v)
        current.update(seed=seed,variant='reflected_inertia_ff_on',path=str(path))
        rows.append(current)
        write(root/'paired_metrics.json',rows)
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        print('PAIRED',seed,json.dumps(dict(old_rmse=old['rmse_err_mm'],new_rmse=current['rmse_err_mm'],
            old_peak=old['j2_torque_peak_after_first_cycle_nm'],new_peak=current['j2_torque_peak_after_first_cycle_nm'],
            old_rate=old['torque_rate_rms'],new_rate=current['torque_rate_rms'],safe=current['safe'])),flush=True)
    old=[r for r in rows if r['variant']=='existing_residual_only']
    new=[r for r in rows if r['variant']=='reflected_inertia_ff_on']
    old_mean=float(np.mean([r['rmse_err_mm'] for r in old]))
    new_mean=float(np.mean([r['rmse_err_mm'] for r in new]))
    rate_ratio=float(np.mean([r['torque_rate_rms'] for r in new])/np.mean([r['torque_rate_rms'] for r in old]))
    selected='reflected_inertia_ff_on' if all(r['safe'] and r['clipping_steps']==0 for r in new) and new_mean<=.95*old_mean and rate_ratio<=1.2 else 'existing_residual_only'
    target_by_seed={str(r['seed']):bool(r['safe'] and r['clipping_steps']==0 and r['rmse_err_mm']<=3.
        and r['j2_fraction_below_minus130_after_first_cycle']>=.01) for r in new}
    write(root/'selection.json',dict(selected=selected,old_mean_rmse_mm=old_mean,new_mean_rmse_mm=new_mean,
        rmse_change_percent=100*(new_mean/old_mean-1),torque_rate_ratio=rate_ratio,
        reflected_ff_primary_target_met_per_seed=target_by_seed,
        reflected_ff_primary_target_met_all_seeds=all(target_by_seed.values()),
        reason='Preset remains trial until empirical DOB and disturbance checks pass.'))
    print('SELECTION',selected,'[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
