"""Bounded slow-ramp, uncompensated circle tracking and causal margin audit.

No output compensation, no rollout disturbance model, no headroom, no MD.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from config import DT,Q_MIN,Q_MAX,DQ_MAX
from kinematics.ur5_kinematics import fk_joints_np
from kroc.check_estimate_only_rho0 import CASES,run_group
from kroc.benchmark_estimator_only_position_margin import ramp_step
from kroc.search_circle_floor25 import CircleTracker,local_ik_reference
from kroc.screen_nominal_margin_pressure import circle
from kroc.collision_geometry import Geometry

NAMES=['static5','dynamic5_10']


class AuditedCircleTracker(CircleTracker):
    def _record_applied_bounds(self):
        cost=self.tracking_cost
        self.margin_history[-1].update(
            q_soft_min_applied=cost.q_soft_min.detach().clone(),
            q_soft_max_applied=cost.q_soft_max.detach().clone(),
            dq_soft_max_applied=cost.dq_soft_max.detach().clone(),
            q_upper_normalizer=cost.q_upper_margin.detach().clone())

    def _prepare_solve(self,state,center):
        super()._prepare_solve(state,center)
        self._record_applied_bounds()


class VelocityOnlyAuditedCircle(AuditedCircleTracker):
    def _prepare_solve(self,state,center):
        values=self.margins.update(state,center,torch.zeros_like(self.dob.d_hat))
        self.margin_history.append({k:v.detach().clone() for k,v in values.items()})
        self._record_applied_bounds()


def replay_margin(velocity,acceleration,cfg):
    """Exact existing 5-10% rule, no sum projection active within these caps."""
    span=np.stack((np.asarray(Q_MAX)-Q_MIN,np.full(6,2*DQ_MAX)))
    v=np.stack((np.maximum(-velocity,0),np.maximum(velocity,0)),axis=1)
    a=np.stack((np.maximum(-acceleration,0),np.maximum(acceleration,0)),axis=1)
    target=span[None,:,None,:]*cfg.min_fraction+np.stack((
        cfg.position_gain*(v*cfg.reaction_time_s+.5*a*cfg.reaction_time_s**2),
        cfg.velocity_gain*a*cfg.reaction_time_s),axis=1)
    target=np.clip(target,span[None,:,None,:]*cfg.min_fraction,span[None,:,None,:]*cfg.max_fraction)
    previous=np.broadcast_to(span[:,None,:]*cfg.min_fraction,(2,2,6)).copy()
    values=[]
    for row in target:
        previous=previous+np.where(row>previous,cfg.beta_up,cfg.beta_down)*(row-previous)
        values.append(previous.copy())
    return np.asarray(values)


def audit_group(directory):
    rows=json.loads((directory/'audited_metrics.json').read_text())
    cfg=CASES['dynamic5_10'];span=np.asarray(Q_MAX)-Q_MIN
    paired={}
    for row in rows:
        with np.load(row['path']) as z:
            n=len(z['kroc_margin']);actual=z['kroc_margin'].copy()
            v=z['measured_state'][:n,6:]
            shadow=replay_margin(v,np.zeros_like(v),cfg)
            target=replay_margin(v,z['kroc_a_d_hat'],cfg)
            np.testing.assert_allclose(z['kroc_q_soft_min_applied'],np.asarray(Q_MIN)+actual[:,0,0],atol=1e-6)
            np.testing.assert_allclose(z['kroc_q_soft_max_applied'],np.asarray(Q_MAX)-actual[:,0,1],atol=1e-6)
            np.testing.assert_allclose(z['kroc_q_upper_normalizer'],np.broadcast_to(.05*span,(n,6)),rtol=1e-6)
            assert str(z['dob_compensation_mode'])=='estimate_only'
            assert float(z['mppi_temperature'])==.24 and float(z['mppi_temporal_noise_rho'])==0.
            assert np.abs(z['u_applied']-z['u_mppi']).max()<1e-5
            assert np.abs(z['u_effective_pred']-z['u_mppi']).max()<1e-5
            first=paired.setdefault(row['seed'],{k:z[k].copy() for k in
                ['circle_reference','d_true','initial_q','sensor_noise_seed']})
            for key,value in first.items():np.testing.assert_array_equal(value,z[key])
            slack=np.minimum(z['q'][0]-np.asarray(Q_MIN)-.05*span,np.asarray(Q_MAX)-.05*span-z['q'][0])
            row['initial_common_5pct_slack_rad']=float(slack.min())
            row['margin_q6_upper_min_fraction']=float(actual[:,0,1,5].min()/span[5])
            row['margin_q6_upper_max_fraction']=float(actual[:,0,1,5].max()/span[5])
            row['margin_q6_lower_max_fraction']=float(actual[:,0,0,5].max()/span[5])
            row['q6_min_upper_clearance_rad']=float(Q_MAX[5]-z['q'][:,5].max())
            if row['case'] in ['dynamic5_10','velocity_only']:
                np.testing.assert_allclose(actual,target,atol=2e-6,rtol=1e-6)
                if row['case']=='velocity_only':assert np.abs(z['kroc_d_hat_margin']).max()==0.
                row['replay_max_error_rad']=float(np.abs(actual-target).max())
                row['same_state_dhat_upper_extra_max_rad']=float((actual-shadow)[:,0,1,5].max())
                post=np.arange(n)*DT>=1.5
                row['same_state_dhat_upper_extra_mean_post_rad']=float((actual-shadow)[post,0,1,5].mean())
                row['dhat_q6_mean_post_nm']=float(z['d_hat_used'][post,5].mean())
                np.savez_compressed(directory/f"seed{row['seed']}_{row['case']}_margin_audit.npz",
                    actual_margin=actual,same_state_zero_dhat_margin=shadow,replayed_margin=target)
            else:
                np.testing.assert_allclose(actual,np.broadcast_to(np.stack((span,np.full(6,2*DQ_MAX)))[:,None,:]*.05,actual.shape),atol=1e-6)
    (directory/'separation_audit.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    return rows


def gate(rows):
    by={r['case']:r for r in rows};low,dyn=[by[n] for n in NAMES]
    return dict(pass_candidate=bool(
        all(r['mesh_5ms']['first_collision_index'] is None for r in rows)
        and min(r['initial_common_5pct_slack_rad'] for r in rows)>0
        and low['position']['maximum_violation_rad']>.001
        and not dyn['position']['violation_episode']
        and dyn['metrics']['tracking_rmse_mm']<30.
        and dyn['same_state_dhat_upper_extra_max_rad']>.005),
        collision_free=all(r['mesh_5ms']['first_collision_index'] is None for r in rows),
        static_max_excess_rad=low['position']['maximum_violation_rad'],
        dynamic_max_excess_rad=dyn['position']['maximum_violation_rad'],
        dhat_extra_max_rad=dyn['same_state_dhat_upper_extra_max_rad'])


def geometry(fraction):
    q=np.array([np.pi,-1.2,1.2,-1.5,-1.57,fraction*Q_MAX[5]])
    ratio=.04/np.linalg.norm(fk_joints_np(q)[-1,:2])
    reference,radius=circle(q,ratio,period=6.,seconds=6.)
    joints=local_ik_reference(reference,q)
    check=Geometry().trajectory(joints[::10])
    assert not check['self_collision_states'] and not check['floor_collision_states']
    return reference,q,check


def execute(directory,fraction,amplitude,seeds,names=NAMES):
    reference,q0,check=geometry(fraction)
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'condition.json').write_text(json.dumps(dict(initial_fraction=fraction,
        amplitude_nm=amplitude,onset_s=.5,ramp_s=1.,held_to_end_s=6.,
        radius_m=.04,period_s=6.,reference_mesh_prefilter=check),indent=2))
    run_group(directory/'raw',reference,q0,ramp_step(-amplitude,50,25,5),seeds,names,90000,
        tracker_classes={'static5':AuditedCircleTracker,'dynamic5_10':AuditedCircleTracker,
            'velocity_only':VelocityOnlyAuditedCircle})
    return audit_group(directory/'raw')


def analyse(root):
    records={}
    for p in sorted(root.glob('*/raw/separation_audit.json')):
        rows=json.loads(p.read_text());groups={}
        for name in sorted({r['case'] for r in rows}):
            rr=[r for r in rows if r['case']==name]
            clear=all(r['mesh_5ms']['first_collision_index'] is None for r in rr)
            groups[name]=dict(runs=len(rr),contact_runs=sum(r['mesh_5ms']['first_collision_index'] is not None for r in rr),
                violation_runs=sum(r['position']['violation_episode'] for r in rr) if clear else None,
                worst_excess_rad=max(r['position']['maximum_violation_rad'] for r in rr) if clear else None,
                mean_integral_rad_s=float(np.mean([r['position']['cumulative_violation_rad_s'] for r in rr])) if clear else None,
                mean_rmse_mm=float(np.mean([r['metrics']['tracking_rmse_mm'] for r in rr])) if clear else None,
                mean_q6_clearance_rad=float(np.mean([r['q6_min_upper_clearance_rad'] for r in rr])) if clear else None,
                mean_solve_ms=float(np.mean([r['metrics']['compute_mean_ms'] for r in rr])))
        records[p.parent.parent.name]=dict(aggregate=groups,per_run=rows)
    (root/'assessment.json').write_text(json.dumps(records,indent=2,allow_nan=False))
    plot(root,records)
    return records


def plot(root,records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for condition,result in records.items():
        rows=result['per_run'];seed=min(r['seed'] for r in rows)
        fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained')
        for name,color in [('static5','tab:orange'),('dynamic5_10','tab:blue'),('velocity_only','tab:purple')]:
            rr=[r for r in rows if r['case']==name]
            if not rr:continue
            r=next(r for r in rr if r['seed']==seed)
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
                if name=='dynamic5_10':
                    with np.load(Path(r['path']).parent/f'seed{seed}_{name}_margin_audit.npz') as a:
                        axes[1,0].plot(t,100*a['same_state_zero_dhat_margin'][:,0,1,5]/(Q_MAX[5]-Q_MIN[5]),'k:',label='Same states, d_hat=0 shadow')
                ref=z['circle_reference'].copy();true=z['d_true'][1:,5].copy()
        axes[0,0].axhline(Q_MAX[5],color='k',ls='--',label='Physical upper')
        axes[0,0].set(xlabel='Time [s]',ylabel='q6 [rad]',title='Solid: true joint; dashed colored: actual soft boundary')
        axes[0,1].plot(t,true,'k--',label='True disturbance');axes[0,1].set(xlabel='Time [s]',ylabel='d6, d_hat6 [Nm]')
        axes[1,0].set(xlabel='Time [s]',ylabel='Upper margin [% full span]',title='Actual margin vs counterfactual on identical measured states')
        axes[1,1].set(xlabel='Time [s]',ylabel='True EE error [mm]')
        axes[2,0].plot(ref[:,0],ref[:,1],'k--',label='Reference');axes[2,0].set(xlabel='X [m]',ylabel='Y [m]',aspect='equal')
        axes[2,1].set(xlabel='Time [s]',ylabel='Applied torque6 [Nm]',title='MPPI only: no addition of d_hat')
        for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=7)
        fig.suptitle(f'{condition}, first predeclared seed {seed}: rho=0, lambda=.24, noisy estimator-only DOB\nR=.04m circle; continuous local IK; no headroom; 5-10% margins')
        fig.savefig(root/condition/'comparison.png',dpi=160);plt.close(fig)


def main(args):
    if args.analyse_only:return analyse(args.output)
    root=args.output;root.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    pilots=[(.78,.5),(.78,1.),(.78,1.5),(.78,2.),(.78,3.),(.88,.5),(.88,1.),(.88,2.)]
    plan=dict(rho=0.,temperature=.24,output_compensation=False,rollout_compensation=False,
        headroom=False,static_floor=.05,dynamic_cap=.10,noise_q_rad=.001,noise_dq_rad_s=.01,
        dob_acceleration_lpf_hz=10.,dob_q_hz=8.,reaction_time_s=.1,beta_up=1.,beta_down=.1,
        pilot_seed=2007,validation_seeds=[2107,2117,2127,2137,2147],
        pilot_fraction_amplitude_nm=pilots,max_pilot_conditions=8,max_gpu_runs=33,wall_budget_s=600,
        reference='R=.04m Cartesian circle, period/duration6s, no orientation/joint objective',
        risk_joint=6,disturbance='negative d6 ramp begins .5s, rises over 1s then held through6s',
        continuous_local_ik='Same physically bounded IK for ALL arms; no legacy +/-pi clipping of q6',
        collision_cost=False,collision_audit_dt_s=.005,
        initial_requirement='Both arms strictly inside the identical initial 5% tightened box; fraction .78 also inside maximum 10% box',
        stop='First pilot with static excess>1e-3rad and dynamic zero, no collision, dynamic RMSE<30mm and same-state d_hat margin extra>.005rad; then five locked validation seeds without retuning',
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,'kroc/check_estimate_only_rho0.py','kroc/state_margin.py','kroc/search_circle_floor25.py']})
    (root/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    nominal=execute(root/'nominal',.78,0.,[2007])
    if any(r['mesh_5ms']['first_collision_index'] is not None or r['position']['violation_episode'] or r['metrics']['tracking_rmse_mm']>15 for r in nominal):
        analyse(root);(root/'verdict.json').write_text(json.dumps({'status':'stop_nominal_failed'}));return
    selection=None;gates=[]
    for i,(fraction,amplitude) in enumerate(pilots):
        if time.monotonic()-started>600:break
        directory=root/f'pilot_{i}'
        rows=execute(directory,fraction,amplitude,[2007]);g=gate(rows)
        gates.append(dict(index=i,fraction=fraction,amplitude_nm=amplitude,gate=g))
        (root/'pilot_gates.json').write_text(json.dumps(gates,indent=2))
        print('SEPARATION_GATE',json.dumps(gates[-1]),flush=True)
        if g['pass_candidate']:selection=(i,fraction,amplitude);break
    validated=False
    if selection is not None and time.monotonic()-started<600:
        i,fraction,amplitude=selection
        execute(root/'validation',fraction,amplitude,plan['validation_seeds'],NAMES+['velocity_only']);validated=True
    results=analyse(root)
    verdict=dict(status='no_qualifying_pilot_stopped' if not validated else 'validation_completed',
        selection=selection,validated=validated,headroom=False,
        completed_gpu_runs=sum(len(x['per_run']) for x in results.values()),elapsed_wall_s=time.monotonic()-started,
        no_further_search=True,not_a_safety_guarantee=True)
    if validated:
        aggregate=results['validation']['aggregate'];low=aggregate['static5'];full=aggregate['dynamic5_10']
        verdict['separation_confirmed']=bool(all(a['contact_runs']==0 for a in aggregate.values())
            and low['violation_runs']>=3 and full['violation_runs']==0)
        verdict['aggregate']=aggregate
    (root/'verdict.json').write_text(json.dumps(verdict,indent=2,allow_nan=False))
    print('SEPARATION_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--analyse-only',action='store_true')
    main(p.parse_args())
