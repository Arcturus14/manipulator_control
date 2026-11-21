"""Low-frequency estimator-only 2.5% versus 2.5--5% state-margin search.

Bounded, paired GPU experiments. No Post-DOB, rollout compensation, headroom,
or automatic markdown reports. Physical limits and cost normalizers unchanged.
"""
import argparse
from dataclasses import replace,asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from config import DT,Q_MIN,Q_MAX
from kroc.check_estimate_only_rho0 import run_group,CASES as OLD_CASES
from kroc.find_estimator_directional_separation import (
    AuditedCircleTracker,VelocityOnlyAuditedCircle,replay_margin,geometry)
from kroc.circle_risk_tradeoff_assessment import position_metrics

STATIC='static25';DYNAMIC='dynamic25_5'
CFG=replace(OLD_CASES['dynamic5_10'],min_fraction=.025,fixed_fraction=.025,max_fraction=.05)
CASES={STATIC:replace(CFG,mode='fixed'),DYNAMIC:CFG,'dynamic5_10':CFG}
CLASSES={STATIC:AuditedCircleTracker,DYNAMIC:AuditedCircleTracker,'velocity_only':VelocityOnlyAuditedCircle}


def smooth_load(amplitude):
    """C1 raised-cosine onset: .25-Hz half-cycle over 2s; then sustained."""
    def function(step,device,dtype):
        phase=np.clip((step*DT-.5)/2.,0.,1.)
        value=-amplitude*.5*(1-np.cos(np.pi*phase))
        result=torch.zeros(6,device=device,dtype=dtype);result[5]=value
        return result
    return function


def audit(directory):
    rows=json.loads((directory/'audited_metrics.json').read_text())
    span=np.asarray(Q_MAX)-Q_MIN;paired={}
    for row in rows:
        with np.load(row['path']) as z:
            assert str(z['dob_compensation_mode'])=='estimate_only'
            assert float(z['dob_cutoff_hz'])==3. and float(z['dob_acceleration_lpf_cutoff_hz'])==0.
            assert float(z['mppi_temporal_noise_rho'])==0. and float(z['mppi_temperature'])==.24
            assert np.max(np.abs(z['u_applied']-z['u_mppi']))<1e-5
            assert np.max(np.abs(z['u_effective_pred']-z['u_mppi']))<1e-5
            n=len(z['kroc_margin']);actual=z['kroc_margin'];v=z['measured_state'][:n,6:]
            np.testing.assert_allclose(z['kroc_q_soft_max_applied'],np.asarray(Q_MAX)-actual[:,0,1],atol=1e-6)
            np.testing.assert_allclose(z['kroc_q_soft_min_applied'],np.asarray(Q_MIN)+actual[:,0,0],atol=1e-6)
            np.testing.assert_allclose(z['kroc_q_upper_normalizer'],np.broadcast_to(.05*span,(n,6)),rtol=1e-6)
            assert actual[:,0].min()>=.025*span.min()-1e-6
            assert np.all(actual[:,0]<=.05*span+1e-6)
            # Check initial margins really equal 2.5%, not merely config labels.
            # At the first solve measured velocity can add a tiny amount.
            initial=z['q'][0]
            row['initial_common_2p5pct_slack_rad']=float(np.minimum(initial-np.asarray(Q_MIN)-.025*span,
                np.asarray(Q_MAX)-.025*span-initial).min())
            row['initial_maximum_5pct_box_slack_rad']=float(np.minimum(initial-np.asarray(Q_MIN)-.05*span,
                np.asarray(Q_MAX)-.05*span-initial).min())
            row['common_start_slack_rad']=row['initial_common_2p5pct_slack_rad']
            first=paired.setdefault(row['seed'],{k:z[k].copy() for k in
                ['circle_reference','d_true','initial_q','sensor_noise_seed']})
            for key,value in first.items():np.testing.assert_array_equal(value,z[key])
            row['q6_upper_clearance_rad']=float(Q_MAX[5]-z['q'][:,5].max())
            collision=row['mesh_5ms']['first_collision_index']
            contact_s=None if collision is None else collision*.005
            excess=np.maximum(z['q']-Q_MAX,np.asarray(Q_MIN)-z['q']).max(axis=1)
            violation=np.flatnonzero(excess>1e-6)
            first_violation=None if len(violation)==0 else float(violation[0]*DT)
            row['first_physical_position_violation_s']=first_violation
            row['first_contact_s']=contact_s
            last=len(z['q'])-1 if contact_s is None else max(0,int(np.floor((contact_s-DT)/DT)))
            row['precontact_position']=position_metrics(z['q'][:last+1])
            row['precontact_end_s']=last*DT
            row['full_run_safe']=bool(contact_s is None and not row['position']['violation_episode'])
            fractions=actual[:,0,1,5]/span[5]
            row['q6_upper_margin_min_fraction']=float(fractions.min())
            row['q6_upper_margin_max_fraction']=float(fractions.max())
            error=z['d_hat_used'][:,5]-z['d_true'][1:,5]
            post=np.arange(n)*DT>=3.
            row['q6_estimation_rmse_nm']=float(np.sqrt(np.mean(error**2)))
            row['q6_steady_estimation_rmse_nm']=float(np.sqrt(np.mean(error[post]**2)))
            row['q6_steady_estimation_bias_nm']=float(error[post].mean())
            if row['case']!=STATIC:
                reconstructed=replay_margin(v,z['kroc_a_d_hat'],CFG)
                shadow=replay_margin(v,np.zeros_like(v),CFG)
                np.testing.assert_allclose(actual,reconstructed,atol=2e-6,rtol=1e-6)
                row['dhat_upper_extra_max_rad']=float((actual-shadow)[:,0,1,5].max())
                row['dhat_upper_extra_steady_mean_rad']=float((actual-shadow)[post,0,1,5].mean())
                np.savez_compressed(directory/f"seed{row['seed']}_{row['case']}_shadow.npz",zero_dhat_margin=shadow)
            else:
                np.testing.assert_allclose(actual[:,0],np.broadcast_to(.025*span,(n,2,6)),atol=1e-6)
    (directory/'separation_audit.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    return rows


def gate(rows):
    by={r['case']:r for r in rows};low=by[STATIC];dyn=by[DYNAMIC]
    clear=all(r['mesh_5ms']['first_collision_index'] is None for r in rows)
    return dict(pass_candidate=bool(clear and low['position']['maximum_violation_rad']>.001
        and not dyn['position']['violation_episode']
        and min(r['initial_common_2p5pct_slack_rad'] for r in rows)>0
        and dyn['metrics']['tracking_rmse_mm']<30
        and dyn['q6_steady_estimation_rmse_nm']<.2
        and dyn['dhat_upper_extra_max_rad']>.005),
        collision_free=clear,static_max_excess_rad=low['position']['maximum_violation_rad'],
        dynamic_max_excess_rad=dyn['position']['maximum_violation_rad'],
        dynamic_q6_estimation_rmse_nm=dyn['q6_steady_estimation_rmse_nm'])


def execute(directory,fraction,amplitude,seeds,names=(STATIC,DYNAMIC)):
    ref,q0,geo=geometry(fraction)
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'condition.json').write_text(json.dumps(dict(initial_fraction=fraction,initial_q=q0.tolist(),
        amplitude_nm=amplitude,rise_start_s=.5,rise_end_s=2.5,held_to_end_s=6.,
        profile='raised cosine onset (.25Hz half-cycle) then constant; NOT a band-limited signal',
        radius_m=.04,period_s=6.,reference_geometry=geo),indent=2))
    run_group(directory/'raw',ref,q0,smooth_load(amplitude),seeds,names,98000,
        case_configs=CASES,tracker_classes=CLASSES,
        observer_options={'dob_acceleration_lpf_cutoff_hz':0.,'dob_cutoff_hz':3.})
    return audit(directory/'raw')


def analyse(root):
    results={}
    for path in sorted(root.glob('*/raw/separation_audit.json')):
        rows=json.loads(path.read_text());aggregate={}
        for name in sorted({r['case'] for r in rows}):
            rr=[r for r in rows if r['case']==name];clear=all(r['mesh_5ms']['first_collision_index'] is None for r in rr)
            aggregate[name]=dict(runs=len(rr),contact_runs=sum(r['mesh_5ms']['first_collision_index'] is not None for r in rr),
                full_episode_metrics_valid=clear,
                violation_runs=sum(r['position']['violation_episode'] for r in rr) if clear else None,
                worst_excess_rad=max(r['position']['maximum_violation_rad'] for r in rr) if clear else None,
                mean_integral_rad_s=float(np.mean([r['position']['cumulative_violation_rad_s'] for r in rr])) if clear else None,
                mean_rmse_mm=float(np.mean([r['metrics']['tracking_rmse_mm'] for r in rr])) if clear else None,
                q6_steady_estimation_rmse_nm=float(np.mean([r['q6_steady_estimation_rmse_nm'] for r in rr])) if clear else None,
                mean_solve_ms=float(np.mean([r['metrics']['compute_mean_ms'] for r in rr])))
        results[path.parent.parent.name]=dict(aggregate=aggregate,rows=rows)
    (root/'assessment.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    if 'validation' in results:
        counters={}
        for name in [STATIC,DYNAMIC,'velocity_only']:
            rr=[r for r in results['validation']['rows'] if r['case']==name]
            if rr and 'full_run_safe' in rr[0]:
                counters[name]=dict(runs=len(rr),full_run_safe_runs=sum(r['full_run_safe'] for r in rr),
                    contact_runs=sum(r['first_contact_s'] is not None for r in rr),
                    position_violation_observed_before_contact_runs=sum(r['precontact_position']['violation_episode'] for r in rr))
        (root/'validation_classification.json').write_text(json.dumps(counters,indent=2))
    plot(root,results)
    return results


def plot(root,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for condition,result in results.items():
        seed=min(r['seed'] for r in result['rows'])
        fig,axes=plt.subplots(3,2,figsize=(14,10),layout='constrained')
        for name,color in [(STATIC,'tab:orange'),(DYNAMIC,'tab:blue'),('velocity_only','tab:purple')]:
            rr=[r for r in result['rows'] if r['case']==name and r['seed']==seed]
            if not rr:continue
            r=rr[0]
            with np.load(r['path']) as z:
                t=np.arange(len(z['err_mm']))*DT
                label=name+(' [CONTACT: invalid]' if r['mesh_5ms']['first_collision_index'] is not None else '')
                axes[0,0].plot(np.arange(len(z['q']))*DT,z['q'][:,5],color=color,label=label)
                axes[0,0].plot(t,z['kroc_q_soft_max_applied'][:,5],color=color,ls='--',alpha=.6)
                axes[0,1].plot(t,z['d_hat_used'][:,5],color=color,label=label)
                axes[1,0].plot(t,100*z['kroc_margin'][:,0,1,5]/(Q_MAX[5]-Q_MIN[5]),color=color,label=label)
                axes[1,1].plot(t+DT,z['err_mm'],color=color,label=label)
                axes[2,0].plot(z['ee'][:,0],z['ee'][:,1],color=color,label=label)
                axes[2,1].plot(t,z['u_applied'][:,5],color=color,label=label)
                if name==DYNAMIC:
                    with np.load(Path(r['path']).parent/f'seed{seed}_{name}_shadow.npz') as a:
                        axes[1,0].plot(t,100*a['zero_dhat_margin'][:,0,1,5]/(Q_MAX[5]-Q_MIN[5]),'k:',label='Same measured states, d_hat=0 shadow')
                ref=z['circle_reference'];truth=z['d_true'][1:,5]
        axes[0,0].axhline(Q_MAX[5],color='k',ls='--',label='Actual physical upper')
        axes[0,0].set(xlabel='Time [s]',ylabel='q6 [rad]',title='Solid: joint; dashed colored: actual soft boundary')
        axes[0,1].plot(t,truth,'k--',label='True disturbance');axes[0,1].set(xlabel='Time [s]',ylabel='d6 / d_hat6 [Nm]')
        axes[1,0].set(xlabel='Time [s]',ylabel='Upper margin [% full span]')
        axes[1,1].set(xlabel='Time [s]',ylabel='EE tracking error [mm]')
        axes[2,0].plot(ref[:,0],ref[:,1],'k--',label='Circle reference');axes[2,0].set(xlabel='X [m]',ylabel='Y [m]',aspect='equal')
        axes[2,1].set(xlabel='Time [s]',ylabel='Applied torque6 [Nm]',title='MPPI only: NO output or rollout compensation')
        for ax in axes.flat:ax.legend(fontsize=7);ax.grid(alpha=.3)
        fig.suptitle(f'{condition}: first predeclared seed {seed}; static2.5% vs dynamic2.5-5%\nNo acceleration LPF, Q=3Hz; rho=0, lambda=.24; noisy 4cm circle; no headroom')
        fig.savefig(root/condition/'comparison.png',dpi=150);plt.close(fig)
        if condition=='validation':
            seeds=sorted({r['seed'] for r in result['rows']})
            fig,axes=plt.subplots(len(seeds),1,figsize=(11,2.5*len(seeds)),layout='constrained',squeeze=False)
            for seed,ax in zip(seeds,axes[:,0]):
                for name,color in [(STATIC,'tab:orange'),(DYNAMIC,'tab:blue'),('velocity_only','tab:purple')]:
                    r=next(r for r in result['rows'] if (r['seed'],r['case'])==(seed,name))
                    with np.load(r['path']) as z:
                        contact=r['mesh_5ms']['first_collision_index']
                        n=len(z['q']) if contact is None else max(1,int(np.floor(contact*.005/DT))+1)
                        ax.plot(np.arange(n)*DT,z['q'][:n,5],color=color,label=name)
                        if contact is not None:ax.axvline(contact*.005,color=color,ls=':',alpha=.7,label=f'{name} contact; later data excluded')
                ax.axhline(Q_MAX[5],color='k',ls='--',label='Physical upper')
                ax.set(xlabel='Time [s]',ylabel='q6 [rad]',title=f'Held-out seed {seed}',xlim=(0,6))
                ax.grid(alpha=.3);ax.legend(fontsize=6,ncol=2)
            fig.suptitle('Locked 5.5Nm low-frequency case: ALL held-out seeds; no output/rollout compensation\nCurves stop at first mesh contact; do not interpret post-contact state excursions')
            fig.savefig(root/'heldout_joint6.png',dpi=150);plt.close(fig)


def main(args):
    if args.analyse_only:return analyse(args.output)
    root=args.output;root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    pilots=[(.94,4.),(.94,5.),(.94,5.5),(.94,6.),(.94,4.5),(.89,5.),(.89,5.5),(.89,4.5)]
    plan=dict(cases={k:asdict(CASES[k]) for k in [STATIC,DYNAMIC]},
        pilot_conditions=pilots,pilot_seed=2007,heldout_seeds=[2407,2417,2427,2437,2447],
        max_pilot_conditions=8,wall_budget_s=600,
        observer=dict(acceleration_lpf_hz=0.,q_cutoff_hz=3.),rho=0.,temperature=.24,
        compensation_mode='estimate_only',output_compensation=False,rollout_compensation=False,headroom=False,
        sensor_noise=dict(q_std_rad=.001,dq_std_rad_s=.01,seed_offset=98000),
        disturbance='Joint6 negative d pushes upper: C1 2s raised-cosine ramp starts .5s, reaches full at2.5s and is held to6s. Smooth onset dominated by .25Hz, NOT strictly band-limited.',
        reference='Cartesian circle R=.04m, period6s; continuous local IK; same feedforward in both cases',
        initial_requirement='Inside the identical initial2.5% box; .89 also inside the maximum5% box, .94 does not. Record both slacks explicitly.',
        physical_bounds_and_cost_normalizers='Unchanged; native5%-span denominator in BOTH arms; only soft onset differs',
        deadband='Existing p99 acceleration thresholds unchanged; not recalibrated for Q3Hz',
        collision_cost=False,collision_audit_dt_s=.005,
        selection='First contact-free pilot with static physical excess>1mrad, dynamic all-joint zero violation, EE_RMSE<30mm, steady d6 estimation_RMSE<.2Nm and actual d_hat margin contribution>.005rad; then heldout seeds without retuning',
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            [__file__,'kroc/check_estimate_only_rho0.py','kroc/state_margin.py','observers/dob.py']})
    (root/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    eligible={};gates=[];selection=None;validated=False
    for fraction in [.94,.89]:
        rows=execute(root/f'nominal_{fraction:g}',fraction,0.,[2007])
        eligible[fraction]=all(r['mesh_5ms']['first_collision_index'] is None and not r['position']['violation_episode']
            and r['metrics']['tracking_rmse_mm']<15 for r in rows)
    for i,(fraction,amplitude) in enumerate(pilots):
        if time.monotonic()-start>600:break
        if not eligible[fraction]:continue
        rows=execute(root/f'pilot_{i}',fraction,amplitude,[2007]);g=gate(rows)
        gates.append(dict(index=i,fraction=fraction,amplitude_nm=amplitude,gate=g))
        (root/'pilot_gates.json').write_text(json.dumps(gates,indent=2))
        print('LOWFREQ_GATE',json.dumps(gates[-1]),flush=True)
        if g['pass_candidate']:selection=(i,fraction,amplitude);break
    if selection is not None:
        i,fraction,amplitude=selection
        execute(root/'validation',fraction,amplitude,plan['heldout_seeds'],[STATIC,DYNAMIC,'velocity_only']);validated=True
    results=analyse(root)
    verdict=dict(selection=selection,validated=validated,nominal_eligible=eligible,
        no_compensation=True,headroom=False,elapsed_wall_s=time.monotonic()-start,
        completed_gpu_runs=sum(len(r['rows']) for r in results.values()),no_further_search=True)
    if validated:
        a=results['validation']['aggregate'];verdict['aggregate']=a
        verdict['separation_confirmed']=bool(all(r['contact_runs']==0 for r in a.values())
            and a[STATIC]['violation_runs']>=3 and a[DYNAMIC]['violation_runs']==0)
    (root/'verdict.json').write_text(json.dumps(verdict,indent=2,allow_nan=False))
    print('LOWFREQ_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--analyse-only',action='store_true')
    main(p.parse_args())
