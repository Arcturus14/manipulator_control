"""Bounded estimator-only retest: old hold diagnostic, then at most 3 circle loads.

No output/rollout disturbance compensation, no headroom, no automatic MD.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import gc
import hashlib
import json
import multiprocessing
from pathlib import Path
import time
import numpy as np
import torch
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight,disturbance_profiles
from config import DT,Q_MIN,Q_MAX,TAU_MAX
from experiment_runner import run_mppi,waypoints
from kroc.benchmark_estimator_only_position_margin import CASES as ORIGINAL,ramp_step,reference_and_initial
from kroc.benchmark_estimator_only_circle_margin import metrics
from kroc.state_margin import NoiseAwareKrocTracker
from kroc.circle_risk_tradeoff_assessment import position_metrics
from kroc.audit_circle_validation import dense_mesh_check

CASES={'static5':ORIGINAL['fixed_5pct'],
       'dynamic5_10':ORIGINAL['dynamic_5_to_10pct'],
       'static10':replace(ORIGINAL['fixed_5pct'],fixed_fraction=.10)}
LABELS={'static5':'Static 5%','dynamic5_10':'Dynamic 5-10%',
        'static10':'Static 10%','velocity_only':'Velocity-only 5-10%'}
OLD_CIRCLE=Path('kroc/results/estimator_only_circle_margin_scale2_20261111_multiseed')
OLD_HOLD=Path('kroc/results/estimator_only_position_validation_amp9p5_20261111')


class VelocityOnlyEstimatorTracker(NoiseAwareKrocTracker):
    def _prepare_solve(self,state,sampling_center):
        values=self.margins.update(state,sampling_center,torch.zeros_like(self.dob.d_hat))
        self.margin_history.append({k:v.detach().clone() for k,v in values.items()})


def candidate_gate(rows):
    by={r['case']:r for r in rows}
    low,dyn,high=[by[n] for n in ['static5','dynamic5_10','static10']]
    clear=all(r['mesh_5ms']['first_collision_index'] is None for r in rows)
    common=all(r['common_start_slack_rad']>0 for r in rows)
    risk=(low['position']['cumulative_violation_rad_s']>1e-6 and
        dyn['position']['cumulative_violation_rad_s']<=.9*low['position']['cumulative_violation_rad_s'] and
        dyn['position']['maximum_violation_rad']<=low['position']['maximum_violation_rad']+1e-8)
    tracking=high['metrics']['tracking_rmse_mm']>=1.03*dyn['metrics']['tracking_rmse_mm']
    return dict(pass_candidate=bool(clear and common and risk and tracking),
        collision_free=clear,common_start=common,risk_reduction=bool(risk),tracking_advantage=bool(tracking))


def run_group(directory,reference,initial,disturbance,seeds,names,noise_offset,
              case_configs=None,tracker_classes=None,observer_options=None):
    directory.mkdir(parents=True,exist_ok=False)
    rows=[];span=np.asarray(Q_MAX)-Q_MIN
    configs=CASES if case_configs is None else case_configs
    tracker_classes={} if tracker_classes is None else tracker_classes
    filters={'dob_acceleration_lpf_cutoff_hz':10.,'dob_cutoff_hz':8.}
    if observer_options is not None:
        if set(observer_options)-set(filters):raise ValueError('Unknown observer option')
        filters.update(observer_options)
    for seed in seeds:
        for name in names:
            cfg=configs['dynamic5_10'] if name=='velocity_only' else configs[name]
            cls=tracker_classes.get(name,VelocityOnlyEstimatorTracker if name=='velocity_only' else NoiseAwareKrocTracker)
            holder=[]
            def factory(*args,**kwargs):
                tracker=cls(*args,**kwargs);holder.append(tracker);return tracker
            print('ESTIMATE_ONLY_RUN',str(directory),seed,name,flush=True)
            data=run_mppi(reference,initial_q=initial,has_disturbance=True,use_dob=True,
                dob_compensation_mode='estimate_only',disturbance_function=disturbance,
                require_cuda=True,seed=seed,temperature=.24,temporal_noise_rho=0.,
                n_samples=136,horizon=10,compile_rollout_loop=False,
                joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.,
                tracker_class=factory,tracker_options={'margin_config':asdict(cfg),
                    'sensor_noise':{'position_std_rad':.001,'velocity_std_rad_s':.01,'seed':noise_offset+seed},
                    **filters})
            data.update(holder[0].export_margin_data())
            data['circle_reference']=reference
            assert str(data['compute_device'])=='cuda'
            assert str(data['dob_compensation_mode'])=='estimate_only'
            assert np.max(np.abs(data['u_applied']-data['u_mppi']))<1e-5
            assert np.max(np.abs(data['u_effective_pred']-data['u_mppi']))<1e-5
            slack=np.minimum(data['q'][0]-np.asarray(Q_MIN)-.10*span,
                np.asarray(Q_MAX)-.10*span-data['q'][0])
            row=dict(seed=seed,case=name,metrics=metrics(data),position=position_metrics(data['q']),
                common_start_slack_rad=float(slack.min()),
                margin_max_fraction=float(np.max(data['kroc_margin'][:,0]/span)),
                q6_max_rad=float(np.max(data['q'][:,5])),
                output_compensation_audit_max_nm=float(np.abs(data['u_applied']-data['u_mppi']).max()),
                rollout_compensation_audit_max_nm=float(np.abs(data['u_effective_pred']-data['u_mppi']).max()))
            path=directory/f'seed{seed}_{name}.npz';row['path']=str(path)
            np.savez_compressed(path,**{k:v for k,v in data.items() if not k.startswith('_')})
            rows.append(row)
            (directory/'metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
            print('ESTIMATE_ONLY_RESULT',json.dumps(row),flush=True)
            del data,holder;gc.collect();torch.cuda.empty_cache()
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row,mesh in zip(rows,pool.map(dense_mesh_check,[(r['path'],4) for r in rows])):
            row['mesh_5ms']=mesh
    (directory/'audited_metrics.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    return rows


def analyse(root):
    groups={}
    for path in sorted(root.glob('*/audited_metrics.json')):
        rows=json.loads(path.read_text());aggregate={}
        for name in sorted({r['case'] for r in rows}):
            rr=[r for r in rows if r['case']==name]
            clear=all(r['mesh_5ms']['first_collision_index'] is None for r in rr)
            errors=[r['metrics']['tracking_rmse_mm'] for r in rr]
            aggregate[name]=dict(runs=len(rr),contact_runs=sum(r['mesh_5ms']['first_collision_index'] is not None for r in rr),
                full_episode_comparison_valid=clear,
                mean_rmse_mm=float(np.mean(errors)) if clear else None,
                sd_rmse_mm=float(np.std(errors,ddof=1)) if clear and len(errors)>1 else None,
                # These full-episode position metrics are not valid after contact.
                violation_runs=sum(r['position']['violation_episode'] for r in rr) if clear else None,
                worst_excess_rad=max(r['position']['maximum_violation_rad'] for r in rr) if clear else None,
                mean_integral_rad_s=float(np.mean([r['position']['cumulative_violation_rad_s'] for r in rr])) if clear else None,
                mean_violation_duration_s=float(np.mean([r['position']['violation_duration_s'] for r in rr])) if clear else None,
                mean_solve_ms=float(np.mean([r['metrics']['compute_mean_ms'] for r in rr])),
                saturation_steps_mean=float(np.mean([r['metrics']['saturation_steps'] for r in rr])),
                common_start=all(r['common_start_slack_rad']>0 for r in rr))
        per_seed={str(seed):candidate_gate([r for r in rows if r['seed']==seed and r['case']!='velocity_only'])
            for seed in sorted({r['seed'] for r in rows})}
        groups[path.parent.name]=dict(aggregate=aggregate,per_seed_gate=per_seed,per_run=rows)
        # Audit matched reference, actual disturbance, noise settings and actual disabled compensation.
        for seed in sorted({r['seed'] for r in rows}):
            first=None
            for row in [r for r in rows if r['seed']==seed]:
                with np.load(row['path']) as z:
                    if first is None:first={k:z[k].copy() for k in z.files}
                    assert float(z['mppi_temperature'])==.24 and float(z['mppi_temporal_noise_rho'])==0.
                    for key in ['circle_reference','d_true','initial_q','sensor_noise_seed',
                        'sensor_position_std_rad','sensor_velocity_std_rad_s']:
                        np.testing.assert_array_equal(first[key],z[key])
                    assert np.max(np.abs(z['u_applied']-z['u_mppi']))<1e-5
                    assert np.max(np.abs(z['u_effective_pred']-z['u_mppi']))<1e-5
                    if row['case']=='velocity_only':assert np.abs(z['kroc_d_hat_margin']).max()==0
        from kroc.validate_temperature_margin import safe_count
        for row in rows:
            first=row['mesh_5ms']['first_collision_index']
            with np.load(row['path']) as z:
                count=safe_count(first,len(z['err_mm']))
                row['first_detected_contact_s']=None if first is None else first*.005
                row['position_before_contact_or_end']=position_metrics(z['q'][:count+1]) if count else None
    summary=dict(rho=0.,temperature=.24,output_compensation=False,rollout_compensation=False,
        groups=groups,mesh_dt_s=.005,mesh_sampling_is_not_continuous_safety_proof=True,
        small_seed_count_is_exploratory=True,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'assessment.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    plot(root,summary)
    for group,result in groups.items():
        print('FINAL_ESTIMATE_ONLY',group,json.dumps(result['aggregate']),flush=True)
    return summary


def plot(root,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cells=[]
    order=['historical_hold','circle_weak0p5','circle_weak1p0','circle_scale2p0','circle_scale2p5','circle_scale3p0','circle_validation']
    for group in order:
        if group not in summary['groups']:continue
        for name in ['static5','dynamic5_10','static10','velocity_only']:
            m=summary['groups'][group]['aggregate'].get(name)
            if m is None:continue
            valid=m['full_episode_comparison_valid']
            cells.append([group,LABELS[name],str(m['runs']),f"{m['contact_runs']}/{m['runs']}",
                'Invalid (contact)' if not valid else f"{m['mean_rmse_mm']:.2f}",
                'Invalid (contact)' if not valid else f"{m['violation_runs']}/{m['runs']}",
                'Yes' if m['common_start'] else 'NO: historical diagnostic'])
    fig,ax=plt.subplots(figsize=(16,2+len(cells)*.34));ax.axis('off')
    table=ax.table(cellText=cells,colLabels=['Condition','Margin policy','Runs','Contact runs',
        'EE RMSE [mm]','Position violation runs','Initial common 10% box'],loc='center',cellLoc='center',
        colWidths=[.17,.17,.06,.10,.16,.15,.19])
    table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,1.8)
    ax.set_title('Estimator-only DOB: rho=0, lambda=0.24; no output/rollout compensation\n'
        'Contact-affected full episodes INVALID; hold is NOT circular tracking and starts outside tightened boxes',pad=16)
    fig.tight_layout();fig.savefig(root/'results_table.png',dpi=160);plt.close(fig)
    for group,result in summary['groups'].items():
        rows=result['per_run'];seed=min(r['seed'] for r in rows)
        fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        joint=5 if group=='historical_hold' else 2
        for name,color in [('static5','tab:orange'),('dynamic5_10','tab:blue'),('static10','tab:green'),('velocity_only','tab:purple')]:
            rr=[r for r in rows if r['case']==name]
            if not rr:continue
            row=next(r for r in rr if r['seed']==seed)
            with np.load(row['path']) as z:
                count=len(z['err_mm']);t=np.arange(count)*DT
                label=LABELS[name]+(' [CONTACT: diagnostic]' if row['mesh_5ms']['first_collision_index'] is not None else '')
                axes[0,0].plot(z['ee'][:,0],z['ee'][:,1],color=color,label=label)
                axes[0,1].plot(t+DT,z['err_mm'],color=color,label=label)
                axes[1,0].plot(np.arange(count+1)*DT,z['q'][:,joint],color=color,label=label)
                axes[1,1].plot(t,100*z['kroc_margin'][:,0,1,joint]/(Q_MAX[joint]-Q_MIN[joint]),color=color,label=label)
                ref=z['circle_reference'].copy()
        axes[0,0].plot(ref[:,0],ref[:,1],'k--',label='Reference')
        axes[0,0].set(xlabel='X [m]',ylabel='Y [m]',aspect='equal',title=f'Representative EE path, seed {seed}')
        axes[0,1].set(xlabel='Time [s]',ylabel='True EE error [mm]')
        axes[1,0].axhline(Q_MAX[joint],ls='--',color='k',label='Physical upper limit')
        axes[1,0].set(xlabel='Time [s]',ylabel=f'Joint {joint+1} position [rad]')
        axes[1,1].set(xlabel='Time [s]',ylabel=f'Joint {joint+1} upper margin [% full span]')
        for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=7)
        title=f'{group}: estimator-only DOB, rho=0, temperature=0.24; NO torque/rollout compensation'
        if group=='historical_hold':title+='\nHistorical diagnostic: initial state OUTSIDE common tightened box; NOT a circle'
        if any(m['contact_runs'] for m in result['aggregate'].values()):title+='\nCONTACT: full-episode trajectories diagnostic only'
        fig.suptitle(title,fontsize=10)
        fig.savefig(root/group/'comparison.png',dpi=160);plt.close(fig)


def main(args):
    if args.analyse_only:return analyse(args.output)
    if args.weak_followup:return weak_followup(args.output)
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    plan=dict(rho=0.,temperature=.24,max_gpu_runs=30,max_wall_budget_s=600,
        hold_diagnostic_seeds=[17,27,37],circle_pilot_seed=17,validation_seeds=[27,37,47],
        circle_scale_order=[2.,3.,2.5],steps_hold=100,steps_circle=250,
        output_compensation=False,rollout_compensation=False,headroom=False,
        cases={k:asdict(v) for k,v in CASES.items()},
        collision_cost=False,collision_evaluation='offline official meshes, 5ms interpolated joints; reject contacts',
        reason_no_collision_cost='Preserve the old estimator-only control setup; do NOT claim post-contact physics is valid',
        hold_initial_outside_common_box=True,
        candidate_gate='Common 10% start box, no mesh contact in all 3 cases; low static violates, dynamic integral >=10% lower with no larger worst excess, high static RMSE >=3% larger than dynamic',
        stop='At most 3 predeclared circle loads; validate only first passing candidate, no further geometry/margin/cost tuning',
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            [__file__,'controllers/mppi_dob_controller.py','kroc/state_margin.py','kroc/sensor_noise.py','observers/dob.py','costs/tracking_cost.py','experiment_runner.py']},cuda=cuda_preflight())
    (args.output/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    ref,q0=reference_and_initial(100,5,.92)
    run_group(args.output/'historical_hold',ref,q0,ramp_step(-9.5,10,10,5),plan['hold_diagnostic_seeds'],list(CASES),70000)
    selection=None;pilots=[];reference=waypoints('circle',250)
    for scale in plan['circle_scale_order']:
        if time.monotonic()-started>plan['max_wall_budget_s']:break
        name='circle_scale'+str(scale).replace('.','p')
        rows=run_group(args.output/name,reference,None,disturbance_profiles(250,scale)['large_step'],[17],list(CASES),30000)
        gate=candidate_gate(rows);pilots.append(dict(group=name,scale=scale,gate=gate))
        (args.output/'pilot_gates.json').write_text(json.dumps(pilots,indent=2))
        if gate['pass_candidate']:selection=scale;break
    validated=False
    if selection is not None and time.monotonic()-started<plan['max_wall_budget_s']:
        run_group(args.output/'circle_validation',reference,None,disturbance_profiles(250,selection)['large_step'],
            plan['validation_seeds'],list(CASES)+['velocity_only'],30000)
        validated=True
    assessment=analyse(args.output)
    verdict=dict(status='candidate_validated' if validated else 'bounded_search_stopped_no_qualifying_circle',
        selected_scale=selection,validated=validated,headroom=False,
        completed_gpu_runs=sum(len(x['per_run']) for x in assessment['groups'].values()),
        elapsed_wall_s=time.monotonic()-started,wall_budget_s=plan['max_wall_budget_s'],
        budget_exhausted=time.monotonic()-started>plan['max_wall_budget_s'],
        no_further_search=True,not_an_impossibility_proof=True)
    if validated:verdict['validation_per_seed']=assessment['groups']['circle_validation']['per_seed_gate']
    (args.output/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print('ESTIMATE_ONLY_VERDICT',json.dumps(verdict),flush=True)


def weak_followup(root):
    """Explicit bounded amendment: weaker loads, not more severe failed loads."""
    previous=json.loads((root/'verdict.json').read_text())
    assert not previous['validated'] and previous['completed_gpu_runs']==18
    plan=dict(reason='All original circle stress loads failed; check only two weaker loads before stopping',
        scales=[.5,1.],seed=17,max_additional_gpu_runs=6,no_further_validation_or_search=True,
        rho=0.,temperature=.24,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'weaker_followup_plan.json').write_text(json.dumps(plan,indent=2))
    reference=waypoints('circle',250);gates=[];started=time.monotonic()
    for scale in plan['scales']:
        name='circle_weak'+str(scale).replace('.','p')
        rows=run_group(root/name,reference,None,disturbance_profiles(250,scale)['large_step'],
            [17],list(CASES),30000)
        gates.append(dict(scale=scale,gate=candidate_gate(rows)))
    result=analyse(root)
    verdict=dict(status='weak_candidate_unvalidated' if any(x['gate']['pass_candidate'] for x in gates)
        else 'bounded_retest_stopped_no_qualifying_circle',validated=False,
        initial_stage=previous,weaker_followup_gates=gates,
        completed_gpu_runs=sum(len(x['per_run']) for x in result['groups'].values()),
        weak_followup_wall_s=time.monotonic()-started,headroom=False,
        no_further_search=True,not_an_impossibility_proof=True)
    (root/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print('ESTIMATE_ONLY_FOLLOWUP_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--analyse-only',action='store_true')
    p.add_argument('--weak-followup',action='store_true')
    main(p.parse_args())
