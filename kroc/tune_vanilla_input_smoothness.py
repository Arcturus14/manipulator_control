"""Tune ONLY the existing torque-difference cost; vanilla MPPI and rho=0.

All experiments GPU-only. No dynamic margin, output/rollout compensation,
output LPF, correlated sampling, mean blending or automatic markdown reports.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gc
import json
import multiprocessing
from pathlib import Path
import numpy as np
import torch
from config import DT,Q_MIN,Q_MAX
from config import NOISE_VAR
from experiment_runner import run_mppi
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.search_lowfreq_estimator_margin import geometry,smooth_load,CASES,STATIC
from kroc.find_estimator_directional_separation import AuditedCircleTracker
from kroc.audit_circle_validation import dense_mesh_check


def metrics(z,n=None):
    n=len(z['u_applied']) if n is None else n
    u=z['u_applied'][:n];delta=np.diff(u,axis=0);ess=z['mppi_weight_ess'][:n]
    excess=np.maximum(np.maximum(z['q'][:n+1]-Q_MAX,np.asarray(Q_MIN)-z['q'][:n+1]),0.)
    return dict(tracking_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][:n]**2))),
        delta_torque_rms_nm=float(np.sqrt(np.mean(delta**2))),
        delta_torque6_rms_nm=float(np.sqrt(np.mean(delta[:,5]**2))),
        torque_rate_rms_nm_s=float(np.sqrt(np.mean(delta**2)))/DT,
        delta_torque6_p95_nm=float(np.percentile(np.abs(delta[:,5]),95)),
        ess_median=float(np.median(ess)),ess_lt2_fraction=float(np.mean(ess<2)),
        max_position_excess_rad=float(excess.max()),
        solve_mean_ms=float(np.nanmean(z['solve_t'][:n])*1000))


def run(root,phase,seeds,weights,temperature=.24,tracker_class=AuditedCircleTracker,noise_variance=NOISE_VAR):
    directory=root/phase;directory.mkdir(exist_ok=False)
    reference,q0,_=geometry(.89);rows=[]
    for profile,amplitude in [('nominal',0.),('slow_disturbance',5.5)]:
        for seed in seeds:
            for w in weights:
                print('SMOOTH_RUN',phase,profile,seed,w,flush=True)
                z=run_mppi(reference,initial_q=q0,has_disturbance=amplitude!=0,use_dob=True,
                    dob_compensation_mode='estimate_only',disturbance_function=smooth_load(amplitude),
                    tracker_class=tracker_class,tracker_options=dict(margin_config=asdict(CASES[STATIC]),
                        sensor_noise=dict(position_std_rad=.001,velocity_std_rad_s=.01,seed=98000+seed),
                        dob_acceleration_lpf_cutoff_hz=0.,dob_cutoff_hz=3.),
                    require_cuda=True,seed=seed,temperature=temperature,temporal_noise_rho=0.,
                    mean_update_alpha=1.,smooth_weight=w,n_samples=136,horizon=10,
                    noise_variance=noise_variance,
                    compile_rollout_loop=False,joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.)
                assert str(z['compute_device'])=='cuda'
                assert float(z['mppi_temporal_noise_rho'])==0. and float(z['mppi_temperature'])==temperature
                assert float(z['mppi_mean_update_alpha'])==1. and float(z['mppi_smooth_weight'])==w
                assert np.max(np.abs(z['u_applied']-z['u_mppi']))<1e-5
                assert np.max(np.abs(z['u_effective_pred']-z['u_mppi']))<1e-5
                p=directory/f'{profile}_seed{seed}_w{w:g}.npz'
                np.savez_compressed(p,**{k:v for k,v in z.items() if not k.startswith('_')},reference=reference)
                row=dict(profile=profile,seed=seed,weight=w,temperature=temperature,path=str(p),full_metrics=metrics(z));rows.append(row)
                (directory/'metrics.json').write_text(json.dumps(rows,indent=2))
                print('SMOOTH_RESULT',json.dumps(row),flush=True)
                del z;gc.collect();torch.cuda.empty_cache()
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row,mesh in zip(rows,pool.map(dense_mesh_check,[(r['path'],4) for r in rows])):
            row['mesh_5ms']=mesh
    (directory/'audited.json').write_text(json.dumps(rows,indent=2))
    return rows


def paired_metrics(rows,weights):
    result=[]
    for profile in ['nominal','slow_disturbance']:
        for seed in sorted({r['seed'] for r in rows}):
            rr=[r for r in rows if r['profile']==profile and r['seed']==seed and r['weight'] in weights]
            contacts=[r['mesh_5ms']['first_collision_index']*.005 for r in rr if r['mesh_5ms']['first_collision_index'] is not None]
            n=min(300,int(np.floor((min(contacts)-DT)/DT+1e-9))) if contacts else 300
            assert n>=30,'Insufficient collision-free prefix'
            for r in rr:
                with np.load(r['path']) as z:m=metrics(z,n)
                result.append(dict(profile=profile,seed=seed,weight=r['weight'],safe_end_s=n*DT,
                    metrics=m,contact=r['mesh_5ms']['first_collision_index'] is not None))
    return result


def select(rows,weights):
    comparisons=[]
    for w in weights[1:]:
        rr=paired_metrics(rows,[.25,w]);by={(r['profile'],r['weight']):r for r in rr}
        ratios={p:{k:by[p,w]['metrics'][k]/max(by[p,.25]['metrics'][k],1e-12)
            for k in ['delta_torque_rms_nm','tracking_rmse_mm']} for p in ['nominal','slow_disturbance']}
        # Keep gain modest and do not select an attractive post-contact trace.
        candidate=[r for r in rows if r['weight']==w]
        baseline=[r for r in rows if r['weight']==.25]
        eligible=(all(not r['contact'] for r in rr if r['weight']==w)
            and ratios['nominal']['tracking_rmse_mm']<=1.20
            and ratios['slow_disturbance']['tracking_rmse_mm']<=1.20
            and all(r['delta_torque_rms_nm']<=.80 for r in ratios.values())
            and all(c['full_metrics']['max_position_excess_rad']<=b['full_metrics']['max_position_excess_rad']+.001
                    for c,b in zip(candidate,baseline)))
        comparisons.append(dict(weight=w,ratios=ratios,eligible=bool(eligible)))
    eligible=[r for r in comparisons if r['eligible']]
    # Least intrusive weight meeting a meaningful smoothness threshold.
    selected=min((r['weight'] for r in eligible),default=None)
    return dict(selected_weight=selected,candidates=comparisons,
        rule='Smallest weight with >=20% all-joint delta torque RMS reduction in BOTH profiles, <=20% RMSE increase, no candidate contacts or increased position excess')


def analyse(root):
    selection=json.loads((root/'selection.json').read_text());w=selection['selected_weight']
    summaries={}
    for phase in ['calibration','validation']:
        p=root/phase/'audited.json'
        if not p.exists():continue
        rows=json.loads(p.read_text());ww=[.25,w] if phase=='validation' else sorted({r['weight'] for r in rows})
        pairs=paired_metrics(rows,ww);groups=[]
        for profile in ['nominal','slow_disturbance']:
            for weight in ww:
                rr=[r for r in pairs if r['profile']==profile and r['weight']==weight]
                groups.append(dict(profile=profile,weight=weight,runs=len(rr),contact_runs=sum(r['contact'] for r in rr),
                    mean_metrics={k:float(np.mean([r['metrics'][k] for r in rr])) for k in rr[0]['metrics']},
                    safe_end_s_per_seed=[r['safe_end_s'] for r in rr]))
        summaries[phase]=dict(groups=groups,paired_prefix_records=pairs)
    (root/'summary.json').write_text(json.dumps(summaries,indent=2))
    if 'validation' in summaries:
        groups=summaries['validation']['groups'];checks=[]
        for profile in ['nominal','slow_disturbance']:
            b=next(r for r in groups if (r['profile'],r['weight'])==(profile,.25))
            c=next(r for r in groups if (r['profile'],r['weight'])==(profile,w))
            checks.append(dict(profile=profile,
                delta_torque_ratio=c['mean_metrics']['delta_torque_rms_nm']/b['mean_metrics']['delta_torque_rms_nm'],
                tracking_ratio=c['mean_metrics']['tracking_rmse_mm']/b['mean_metrics']['tracking_rmse_mm'],
                contacts_not_increased=c['contact_runs']<=b['contact_runs']))
        passed=all(r['delta_torque_ratio']<=.8 and r['tracking_ratio']<=1.2 and r['contacts_not_increased'] for r in checks)
        (root/'validation_verdict.json').write_text(json.dumps(dict(passed=passed,checks=checks,selected_weight=w),indent=2))
    plot(root,summaries,w)


def plot(root,summaries,w):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    phase='validation' if 'validation' in summaries else 'calibration'
    rows=json.loads((root/phase/'audited.json').read_text());seed=min(r['seed'] for r in rows)
    for profile in ['nominal','slow_disturbance']:
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for weight,color in [(.25,'tab:orange'),(w,'tab:blue')]:
            if weight is None:continue
            r=next(r for r in rows if (r['profile'],r['seed'],r['weight'])==(profile,seed,weight))
            safe=next(x for x in summaries[phase]['paired_prefix_records'] if (x['profile'],x['seed'],x['weight'])==(profile,seed,weight))
            with np.load(r['path']) as z:
                n=round(safe['safe_end_s']/DT);t=np.arange(n)*DT;label=f'weight={weight:g}'
                axes[0,0].plot(t,z['u_applied'][:n,5],color=color,label=label)
                axes[0,1].plot(t[1:],np.diff(z['u_applied'][:n,5])/DT,color=color,label=label)
                axes[1,0].plot(t,z['err_mm'][:n],color=color,label=label)
                axes[1,1].plot(t,z['mppi_weight_ess'][:n],color=color,label=label)
        axes[0,0].set(xlabel='Time [s]',ylabel='Applied torque6 [Nm]')
        axes[0,1].set(xlabel='Time [s]',ylabel='Torque6 rate [Nm/s]')
        axes[1,0].set(xlabel='Time [s]',ylabel='EE error [mm]')
        axes[1,1].set(xlabel='Time [s]',ylabel='ESS / 136')
        for ax in axes.flat:ax.grid(alpha=.3);ax.legend()
        fig.suptitle(f'{phase}, {profile}, first predeclared seed {seed}; common collision-free prefix\nONLY existing smoothness weight changed; rho=0, lambda=.24, alpha=1; no dynamic margin or DOB compensation')
        fig.savefig(root/f'{profile}_comparison.png',dpi=150);plt.close(fig)


def main(args):
    if args.analyse_only:return analyse(args.output)
    cuda_preflight();root=args.output;root.mkdir(parents=True,exist_ok=False)
    weights=[.25,1.,4.,16.,64.]
    plan=dict(weights=weights,calibration_seed=2007,validation_seeds=[2507,2517,2527],
        rho=0.,temperature=.24,mean_update_alpha=1.,dob_compensation='estimate_only',
        no_dynamic_margin=True,fixed_margin_fraction=.025,observer_acceleration_lpf_hz=0.,observer_q_hz=3.,
        K=136,H=10,DT=DT,only_changed_parameter='Existing TrackingCost.smooth_weight; formula unchanged',
        reference='R4cm circle, period6s, initial q6=.89QMAX; nominal and 2s raised-cosine to sustained d6=-5.5Nm',
        selection_rule='See selection.json: lowest weight meeting smoothness/tracking/contact/violation gates',
        baseline_cost='w*(||u0-u_previous||^2 + sum internal ||u_t-u_t-1||^2/(H-1))')
    (root/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    rows=run(root,'calibration',[2007],weights);choice=select(rows,weights)
    (root/'selection.json').write_text(json.dumps(choice,indent=2));print('SMOOTH_SELECTION',json.dumps(choice),flush=True)
    if choice['selected_weight'] is not None:run(root,'validation',plan['validation_seeds'],[.25,choice['selected_weight']])
    analyse(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyse-only',action='store_true');main(p.parse_args())
