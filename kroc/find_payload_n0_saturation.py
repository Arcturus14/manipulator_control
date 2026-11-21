"""Steps1-2 only: frozen small-cube N0, DOB-on check, J2 clipping onset.

CUDA control only. No N1/D0 runs or controller/filter/covariance retuning.
All outcomes/configurations/mesh audits retained; no archives generated.
"""
import copy
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import yaml
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.run_payload_cube_tcp import run_payload
from kroc.study_payload_torque_stability import metrics,write
from kroc.view_payload_tcp import load_result,save_outputs

BASE=Path('kroc/results/payload_quarter_volume_20261121/20261121_211735')
SEEDS=[6807,6817,6827]


def statistics(z,v):
    m=metrics(z,v);u=z['u_mppi'];req=z['u_requested'];app=z['u_applied'];loss=req-app
    n=len(u);t=np.arange(n)*.02;delta=np.abs(loss[:,1]);mask=delta>1e-5
    edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    i=int(delta.argmax());comp=z['d_hat_used'];true=z['d_true'][1:n+1]
    limits=np.array([150,150,150,28,28,28]);nominal_excess=np.maximum(np.abs(u)-limits,0.)
    purely_comp=mask & (nominal_excess[:,1]<=1e-5)
    ess=z['mppi_weight_ess'];steady=t>=2.5
    first=None if not mask.any() else int(np.flatnonzero(mask)[0])
    m.update(clip_steps_J2=int(mask.sum()),clip_fraction_J2=float(mask.mean()),
        clip_steps_gt0p1Nm=int((delta>.1).sum()),clip_steps_gt1Nm=int((delta>1).sum()),
        first_clip_s=None if first is None else first*.02,
        longest_clip_s=float(max(ends-starts,default=0)*.02),clip_J2_abs_max_nm=float(delta.max()),
        jclip_J2_nm_s=float(delta.sum()*.02),
        exclusively_compensation_induced_clip_steps_J2=int(purely_comp.sum()),
        nominal_alone_excess_steps_J2=int((nominal_excess[:,1]>1e-5).sum()),
        first_clip_components=None if first is None else dict(time_s=first*.02,nominal_nm=float(u[first,1]),
            compensation_nm=float(comp[first,1]),requested_nm=float(req[first,1]),applied_nm=float(app[first,1]),loss_nm=float(loss[first,1])),
        max_clip_components=dict(time_s=i*.02,nominal_nm=float(u[i,1]),compensation_nm=float(comp[i,1]),
            requested_nm=float(req[i,1]),applied_nm=float(app[i,1]),loss_nm=float(loss[i,1])),
        steady_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][steady]**2))),
        prediction_q_rmse_rad=float(np.sqrt(np.mean(z['state_prediction_error'][:,:6]**2))),
        ess_median=float(np.median(ess)),ess_p05=float(np.percentile(ess,5)),ess_lt2_fraction=float((ess<2).mean()),
        ess_clipped_mean=None if not mask.any() else float(ess[mask].mean()),
        steady_compensation_J2_mean_nm=float(comp[steady,1].mean()),
        steady_estimation_error_J2_rms_nm=float(np.sqrt(np.mean((comp[steady,1]-true[steady,1])**2))),
        compensation_J2_min_nm=float(comp[:,1].min()),compensation_J2_max_nm=float(comp[:,1].max()))
    np.testing.assert_allclose(req,u+comp,atol=1e-6)
    np.testing.assert_allclose(app,np.clip(req,-limits,limits),atol=3e-6,rtol=1e-6)
    assert str(z['compute_device'])=='cuda' and str(z['input_study_case'])=='N0'
    assert float(z['input_study_headroom_weight'])==0 and not bool(z['payload_diagnostic_mask_base_upper_arm'])
    return m


def main():
    cuda=cuda_preflight();cfg=yaml.safe_load(Path('configs/payload_cube_tcp_stabilized_small_cube_n0.yaml').read_text())
    cfg['scene'].update(use_dob=True,disturbance_joint=2,disturbance_amplitude_nm=0.)
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--resume',type=Path);args=parser.parse_args()
    root=args.resume or Path('kroc/results/payload_N0_saturation_onset_20261121')/datetime.now().strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True,exist_ok=True)
    write(root/'plan.json',dict(cuda=cuda,seeds=SEEDS,pilot_seed=6807,controller='N0 only',
        disturbance='Plant tau_effective=tau_app-d. J2 d ramps0 to negative amplitude from0.5 to2.5s with a raised cosine, held until20s.',
        coarse_absolute_amplitudes_nm=[20,40,50,60],onset_resolution_nm=1,
        numerical_clip_tolerance_nm=1e-5,engineering_excess_thresholds_nm=[.1,1.],
        zero_load_gate='All3 seeds safe, no clipping, EE RMSE<=2mm and <=1.5 times paired DOB-OFF; otherwise stop/report before disturbance sweep.',
        scope='No N1/D0 comparisons and no parameter retuning. Search empirical amplitude bracket; monotonicity of stochastic closed loop is not guaranteed.',
        physical_limits_nm=[150,150,150,28,28,28]))
    runs={};results={}
    if (root/'runs.json').exists():
        for row in json.loads((root/'runs.json').read_text()):
            key=f'a{row["amplitude_abs_nm"]:02d}_seed{row["seed"]}'
            runs[key]=row;results[key]=load_result(Path(row['path']))
    def run(seed,amplitude):
        key=f'a{amplitude:02d}_seed{seed}'
        if key in runs:return runs[key]
        config=copy.deepcopy(cfg);config['scene']['disturbance_amplitude_nm']=-float(amplitude)
        path=root/key
        if (path/'verification.json').exists():
            z=load_result(path);v=json.loads((path/'verification.json').read_text())
        else:
            if path.exists():path=path/'retry'
            z,v,path=run_payload(config,seed,'N0',output=path)
        row=statistics(z,v);row.update(seed=seed,amplitude_abs_nm=amplitude,path=str(path))
        runs[key]=row;results[key]=z;write(root/'runs.json',list(runs.values()))
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=True,csv=True)
        print('RESULT',key,json.dumps(row),flush=True)
        return row
    zero={};off={}
    for seed in SEEDS:
        old=BASE/f'seed{seed}';off[str(seed)]=metrics(load_result(old),json.loads((old/'verification.json').read_text()))
        zero[str(seed)]=run(seed,0)
    write(root/'dob_only_comparison.json',dict(DOB_OFF=off,DOB_ON=zero))
    if not all(r['safe'] and not r['clipping_steps'] and r['rmse_err_mm']<=2 and r['rmse_err_mm']<=1.5*off[s]['rmse_err_mm'] for s,r in zero.items()):
        write(root/'assessment.json',dict(status='stopped_at_DOB_only_gate',dob_zero=zero,reason='DOB-on no-disturbance gate did not pass; controller/filter settings not changed.'))
        print('[OUTPUT]',root,flush=True);return
    # 60Nm screening caused loss of tracking; retain it, but only refine <=50Nm.
    for amplitude in [20,40,50]:
        row=run(6807,amplitude)
        if not row['safe']:
            print('UNSAFE_COARSE_OUTCOME_RETAINED',amplitude,flush=True);break
    sampled=sorted([r for r in runs.values() if r['seed']==6807],key=lambda r:r['amplitude_abs_nm'])
    clipped=[r for r in sampled if r['clip_steps_J2']>0]
    if not clipped:
        write(root/'assessment.json',dict(status='no_clipping_through_coarse_range',runs=list(runs.values())));print('[OUTPUT]',root);return
    hi=min(r['amplitude_abs_nm'] for r in clipped)
    lo=max(r['amplitude_abs_nm'] for r in sampled if r['amplitude_abs_nm']<hi and r['clip_steps_J2']==0)
    while hi-lo>1:
        mid=(lo+hi)//2;row=run(6807,mid)
        if row['clip_steps_J2']:hi=mid
        else:lo=mid
    # Local neighbors make the transition interpretable, rather than relying on one spike.
    for amp in [max(0,lo-1),min(50,hi+2),min(50,hi+5)]:run(6807,amp)
    brackets={'6807':dict(no_clip_amp_nm=lo,first_clip_amp_nm=hi)}
    for seed in [6817,6827]:
        low,high=lo,hi
        while run(seed,low)['clip_steps_J2'] and low>max(0,lo-6):low=max(0,low-2)
        while not run(seed,high)['clip_steps_J2'] and high<min(50,hi+6):high=min(50,high+2)
        if run(seed,low)['clip_steps_J2'] or not run(seed,high)['clip_steps_J2']:
            brackets[str(seed)]=dict(unresolved=True,checked_low_nm=low,checked_high_nm=high);continue
        while high-low>1:
            mid=(high+low)//2
            if run(seed,mid)['clip_steps_J2']:high=mid
            else:low=mid
        brackets[str(seed)]=dict(no_clip_amp_nm=low,first_clip_amp_nm=high)
    visible=[r for r in runs.values() if r['seed']==6807 and r['safe'] and r['clip_steps_J2']>0 and (r['longest_clip_s']>=.1 or r['clip_fraction_J2']>=.01)]
    selected=min(visible,key=lambda r:r['amplitude_abs_nm']) if visible else run(6807,hi)
    for label,amp in [('onset',hi),('intermittent',selected['amplitude_abs_nm'])]:
        preset=Path(f'configs/payload_cube_tcp_N0_J2_clip_{label}.yaml')
        if preset.exists():raise FileExistsError(preset)
        config=copy.deepcopy(cfg);config['scene']['disturbance_amplitude_nm']=-float(amp)
        preset.write_text('# Additional frozen N0 saturation-screen scenario; no N1/D0 comparison yet.\n'+yaml.safe_dump(config,sort_keys=False))
    write(root/'assessment.json',dict(status='completed_N0_only',dob_only_gate_passed=True,amplitude_brackets_nm=brackets,
        pilot_onset=run(6807,hi),visible_saturation_candidate=selected,disturbance_definition='J2 negative raised-cosine ramp0.5--2.5s, then constant until20s',
        interpretation='Empirical clipping onset under specified noise/seeds and trajectory; not a universal physical-feasibility limit or N1/D0 superiority claim.',
        all_outcomes_retained=True,abnormal_60Nm_outcome=runs.get('a60_seed6807'),
        safety_scope='60Nm pilot lost tracking; further upward search stopped. Refinement and other seeds limited to <=50Nm.'))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,1,figsize=(11,12),sharex=True)
    for seed in SEEDS:
        rows=sorted([r for r in runs.values() if r['seed']==seed],key=lambda r:r['amplitude_abs_nm']);x=[r['amplitude_abs_nm'] for r in rows]
        for ax,key in zip(axes,['J_clip_l2_nm_s','rmse_err_mm','clip_fraction_J2','ess_mean']):
            good=[r for r in rows if r['safe']];bad=[r for r in rows if not r['safe']]
            ax.plot([r['amplitude_abs_nm'] for r in good],[r[key] for r in good],'o-',label=f'seed{seed}')
            if bad:ax.scatter([r['amplitude_abs_nm'] for r in bad],[r[key] for r in bad],marker='x',color='red',label='failed screening')
    for ax,label in zip(axes,['J_clip [Nm s]','TCP RMSE [mm]','J2 clipping fraction','Mean ESS']):ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Magnitude of negative J2 disturbance [Nm]');fig.tight_layout();fig.savefig(root/'amplitude_sweep.png',dpi=165);plt.close(fig)
    key=f'a{selected["amplitude_abs_nm"]:02d}_seed6807';z=results[key];n=len(z['u_requested']);t=np.arange(n)*.02
    fig,axes=plt.subplots(5,1,figsize=(13,13),sharex=True)
    axes[0].plot(t,z['d_true'][1:n+1,1],label='d true');axes[0].plot(t,z['d_hat_used'][:,1],label='d_hat used')
    for label,a in [('nominal MPPI',z['u_mppi']),('requested',z['u_requested']),('applied',z['u_applied'])]:axes[1].plot(t,a[:,1],label=label,lw=1)
    axes[1].axhline(-150,color='red',ls='--',label='physical lower bound')
    axes[2].plot(t,np.abs(z['u_clamp_loss'][:,1]),label='|request-applied|')
    axes[3].plot(t,z['err_mm'],label='TCP error');axes[4].plot(t,z['mppi_weight_ess'],label='ESS')
    for ax,label in zip(axes,['J2 disturbance [Nm]','J2 torque [Nm]','J2 clipping loss [Nm]','Error [mm]','ESS']):ax.set_ylabel(label);ax.legend(fontsize=8);ax.grid(alpha=.25)
    axes[-1].set_xlabel('Time [s]');fig.suptitle(f'N0 only: J2 disturbance amplitude -{selected["amplitude_abs_nm"]} Nm');fig.tight_layout();fig.savefig(root/'saturation_time_series.png',dpi=165);plt.close(fig)
    print('[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
