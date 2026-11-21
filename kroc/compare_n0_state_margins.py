"""Four-way estimator-only state-margin check on frozen N0 settings.

No GUI/default changes, compensation, input tightening, headroom or tuning.
Oracle disturbance is supplied ONLY to the margin generator.
"""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import DT, Q_MIN, Q_MAX, DQ_MAX, TAU_MAX
from controllers.final_n0_controller import FinalN0Tracker
from controllers.mppi_dob_controller import UR5Tracker
from experiment_runner import run_mppi
from kroc.optimize_final_mppi import options as n0_options
from kroc.run_pre_input_baseline import load_preset
from kroc.state_margin import MarginConfig
from kroc.search_lowfreq_estimator_margin import geometry, smooth_load
from kroc.find_estimator_directional_separation import replay_margin
from kroc.audit_circle_validation import dense_mesh_check
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight

ROOT=Path('kroc/results/n0_state_margin_fourway_20261118')
SEEDS=[8207,8217,8227]
CASES=['static25','static5','dynamic_estimate','dynamic_oracle']
LABELS=['Static 2.5%','Static 5%','Dynamic estimated d','Dynamic oracle d']
COLORS=['tab:orange','tab:green','tab:blue','tab:purple']


def margin_config(case):
    base=MarginConfig(**load_preset()['state_margin'])
    return replace(base,mode='directional' if case.startswith('dynamic') else 'fixed',
        min_fraction=.025,fixed_fraction=.05 if case=='static5' else .025,max_fraction=.05)


class N0MarginTracker(FinalN0Tracker):
    def __init__(self,*args,margin_source='estimate',margin_disturbance=None,**kwargs):
        self.margin_source=margin_source
        self.margin_disturbance=margin_disturbance
        super().__init__(*args,**kwargs)

    def run(self,*args,**kwargs):
        if not kwargs.get('use_dob') or kwargs.get('dob_compensation_mode')!='estimate_only':
            raise ValueError('This adapter requires DOB estimation without compensation')
        self.request_sample_fraction.clear();self.headroom_sample_cost.clear()
        self.sensor_model.reset();self.measured_state_history.clear()
        self.margin_history.clear();self.margins.reset()
        return UR5Tracker.run(self,*args,**kwargs)

    def _prepare_solve(self,state,center):
        signal=(self.margin_disturbance(self.step,self.device,self.dtype)
                if self.margin_source=='oracle' else self.dob.d_hat)
        values=self.margins.update(state,center,signal)
        values['margin_input_d']=signal.detach().clone()
        self.margin_history.append({k:v.detach().clone() for k,v in values.items()})
        self._record_applied_bounds()


def options(seed,case,disturbance):
    opts=n0_options(seed,'N0','reference')
    opts['tracker_class']=N0MarginTracker
    opts['dob_compensation_mode']='estimate_only'
    opts['tracker_options'].update(margin_config=asdict(margin_config(case)),
        margin_source='oracle' if case=='dynamic_oracle' else 'estimate',
        margin_disturbance=disturbance)
    return opts


def audit(z,case):
    assert str(z['compute_device'])=='cuda'
    assert float(z['mppi_temperature'])==64.
    assert float(z['dob_cutoff_hz'])==3.
    assert float(z['dob_acceleration_lpf_cutoff_hz'])==6.
    assert str(z['dob_compensation_mode'])=='estimate_only'
    assert float(z['input_study_headroom_weight'])==0.
    assert not bool(z['input_study_fixed_clip'])
    assert not bool(z['input_study_saturation_rollout'])
    assert not bool(z['input_tightening_enabled'])
    np.testing.assert_array_equal(z['u_effective_pred'],z['u_mppi'])
    np.testing.assert_allclose(z['u_applied'],np.clip(z['u_mppi'],-np.array(TAU_MAX),TAU_MAX),atol=2e-6)
    np.testing.assert_allclose(z['mppi_active_noise_variance'],
        np.broadcast_to(np.array([9.,9.,9.,.3136,.3136,.3136],np.float32),(300,6)),rtol=1e-6)
    for key in ['q','qd','u_mppi','u_applied','d_hat_used','kroc_margin','err_mm']:
        assert np.isfinite(z[key]).all(),key
    signal=z['d_true'][1:] if case=='dynamic_oracle' else z['d_hat_used']
    np.testing.assert_array_equal(z['kroc_margin_input_d'],signal)
    actual=z['kroc_margin'];span=np.array(Q_MAX)-Q_MIN
    np.testing.assert_allclose(z['kroc_q_soft_max_applied'],np.array(Q_MAX)-actual[:,0,1],atol=1e-6)
    np.testing.assert_allclose(z['kroc_q_upper_normalizer'],np.broadcast_to(.05*span,(300,6)),rtol=1e-6)
    if case.startswith('dynamic'):
        reconstructed=replay_margin(z['measured_state'][:300,6:],z['kroc_a_d_hat'],margin_config(case))
        np.testing.assert_allclose(actual,reconstructed,rtol=1e-5,atol=2e-6)
    else:
        target=np.stack((span,np.full(6,2*DQ_MAX)))[:,None,:]*margin_config(case).fixed_fraction
        np.testing.assert_allclose(actual,np.broadcast_to(target,actual.shape),atol=2e-6)


def metrics(z,n):
    q=z['q'][1:n+1];excess=np.maximum(np.maximum(np.array(Q_MIN)-q,q-np.array(Q_MAX)),0.)
    peak=excess.max(1);u=z['u_applied'][:n];cut=z['u_mppi'][:n]-u
    frac=z['kroc_margin'][:n,0,1,5]/(Q_MAX[5]-Q_MIN[5])
    shadow=replay_margin(z['measured_state'][:n,6:],np.zeros((n,6)),margin_config('dynamic_estimate'))
    return dict(end_s=n*DT,rmse_mm=float(np.sqrt(np.mean(z['err_mm'][:n]**2))),
        position_violation_steps=int(np.sum(peak>1e-6)),
        position_violation_duration_s=float(np.sum(peak>1e-6)*DT),
        max_position_excess_rad=float(excess.max()),
        position_excess_integral_rad_s=float(peak.sum()*DT),
        q6_max_rad=float(q[:,5].max()),q6_upper_clearance_rad=float(Q_MAX[5]-q[:,5].max()),
        saturation_steps=int(np.sum((np.abs(cut)>1e-5).any(1))),
        max_clamp_loss_nm=float(np.abs(cut).max()),
        upper_margin6_min_pct=float(100*frac.min()),upper_margin6_max_pct=float(100*frac.max()),
        same_state_dhat_upper_extra_max_rad=float((z['kroc_margin'][:n,0,1,5]-shadow[:,0,1,5]).max()),
        dob6_rmse_nm=float(np.sqrt(np.mean((z['d_hat_used'][:n,5]-z['d_true'][1:n+1,5])**2))),
        solve_mean_ms=float(np.mean(z['solve_t'][min(20,n-1):n])*1000))


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False))


def run_one(seed,case):
    path=ROOT/f'{case}_seed{seed}.npz'
    if not path.exists():
        ref,q0,_=geometry(.89);disturbance=smooth_load(5.5)
        print('RUN',seed,case,flush=True)
        z=run_mppi(ref,initial_q=q0,has_disturbance=True,
            disturbance_function=disturbance,**options(seed,case,disturbance))
        audit(z,case)
        z.update(circle_reference=ref,comparison_case=np.asarray(case))
        np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')})
        del z;gc.collect();torch.cuda.empty_cache()
    mesh=dense_mesh_check((str(path),4))
    with np.load(path) as z:
        audit(z,case)
        full=metrics(z,300)
    row=dict(seed=seed,case=case,path=str(path),mesh_5ms=mesh,
             full_episode_valid=mesh['first_collision_index'] is None,full_metrics=full)
    write(path.with_suffix('.json'),row)
    print('RESULT',seed,case,json.dumps(full),'mesh',mesh['first_collision_index'],flush=True)
    return row


def plot(seed):
    loaded=[dict(np.load(ROOT/f'{case}_seed{seed}.npz')) for case in CASES]
    t=np.arange(300)*DT;span=Q_MAX[5]-Q_MIN[5]
    fig,axes=plt.subplots(3,2,figsize=(14,11),layout='constrained')
    for case,label,color,z in zip(CASES,LABELS,COLORS,loaded):
        axes[0,0].plot(t,z['err_mm'],label=label,color=color)
        axes[0,1].plot(t+DT,z['q'][1:,5],label=label,color=color)
        axes[1,0].plot(t,z['kroc_margin_input_d'][:,5],label=label,color=color)
        axes[1,1].plot(t,100*z['kroc_margin'][:,0,1,5]/span,label=label,color=color)
        axes[2,0].plot(t,z['u_applied'][:,5],label=label,color=color,lw=.8)
        axes[2,1].plot(t,z['qd'][1:,5],label=label,color=color)
    axes[0,1].axhline(Q_MAX[5],color='k',ls='--',label='Physical upper')
    axes[1,0].plot(t,loaded[0]['d_true'][1:,5],'k--',label='True disturbance')
    axes[2,0].axhline(TAU_MAX[5],color='k',ls='--');axes[2,0].axhline(-TAU_MAX[5],color='k',ls='--')
    for ax,y in zip(axes.flat,['EE error [mm]','q6 [rad]','Margin input d6 [Nm]',
                            'q6 upper margin [% full span]','Applied torque6 [Nm]','dq6 [rad/s]']):
        ax.set(xlabel='Time [s]',ylabel=y);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle(f'N0, lambda64; estimator ONLY; seed{seed}; 4cm/6s circle, sustained d6=-5.5Nm\nSame sensor/MPPI seeds; NO output or rollout compensation; NO input tightening/headroom')
    fig.savefig(ROOT/f'fourway_seed{seed}.png',dpi=160);plt.close(fig)
    z=loaded[2];cfg=margin_config('dynamic_estimate')
    shadow=replay_margin(z['measured_state'][:300,6:],np.zeros((300,6)),cfg)
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained',sharex=True)
    axes[0].plot(t,z['d_true'][1:,5],'k--',label='True d6')
    axes[0].plot(t,z['kroc_margin_input_d'][:,5],label='Estimated d6 used by margin')
    axes[1].plot(t,z['kroc_a_d_hat_raw'][:,5],label='Raw -M^-1 d_hat')
    axes[1].plot(t,z['kroc_a_d_hat'][:,5],label='After deadband')
    axes[2].plot(t,100*z['kroc_margin'][:,0,1,5]/span,label='Upper margin (actual)')
    axes[2].plot(t,100*z['kroc_margin'][:,0,0,5]/span,label='Lower margin (actual)')
    axes[2].plot(t,100*shadow[:,0,1,5]/span,'--',label='Same-state zero-d_hat shadow (not a controller run)')
    axes[2].axhline(2.5,color='grey',ls=':');axes[2].axhline(5,color='grey',ls=':')
    for ax,y in zip(axes,['Disturbance [Nm]','Disturbance acceleration [rad/s^2]','Margin [% full span]']):
        ax.set_ylabel(y);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'N0 dynamic margin generation; seed{seed}; no compensation\nTrue d -> DOB estimate -> directional acceleration -> upper/lower state margin')
    fig.savefig(ROOT/f'dhat_margin_timeseries_seed{seed}.png',dpi=170);plt.close(fig)
    data=np.column_stack((t,z['d_true'][1:,5],z['kroc_margin_input_d'][:,5],
        z['kroc_a_d_hat'][:,5],100*z['kroc_margin'][:,0,1,5]/span,
        100*z['kroc_margin'][:,0,0,5]/span,100*shadow[:,0,1,5]/span))
    np.savetxt(ROOT/f'dhat_margin_timeseries_seed{seed}.csv',data,delimiter=',',
        header='t_s,true_d6_Nm,estimate_d6_Nm,filtered_a6,upper_margin_pct,lower_margin_pct,zero_dhat_shadow_upper_pct',comments='')


def summarize(rows):
    common={}
    for seed in SEEDS:
        rr=[r for r in rows if r['seed']==seed]
        if len(rr)!=4:continue
        hits=[r['mesh_5ms']['first_collision_index']*.005 for r in rr if r['mesh_5ms']['first_collision_index'] is not None]
        n=300 if not hits else max(2,min(300,int(np.floor((min(hits)-DT)/DT))))
        common[str(seed)]=dict(end_s=n*DT,metrics={})
        paired=None
        for r in rr:
            with np.load(r['path']) as z:
                fields={k:z[k] for k in ['circle_reference','d_true','initial_q','sensor_noise_seed']}
                if paired is None:paired=fields
                else:
                    for k in fields:np.testing.assert_array_equal(fields[k],paired[k])
                common[str(seed)]['metrics'][r['case']]=metrics(z,n)
        plot(seed)
    groups={}
    for case in CASES:
        rr=[r for r in rows if r['case']==case]
        groups[case]=dict(runs=len(rr),collision_runs=sum(not r['full_episode_valid'] for r in rr),
            mean_full_rmse_mm=float(np.mean([r['full_metrics']['rmse_mm'] for r in rr])),
            violation_runs=sum(r['full_metrics']['position_violation_steps']>0 for r in rr))
    write(ROOT/'summary.json',dict(groups=groups,paired_common_precontact=common,rows=rows,
        conclusion='Descriptive fixed scenario, not a safety guarantee or search for a successful case. Full-episode safety/tracking claims invalid after mesh contact.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    args=parser.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    cuda=cuda_preflight();print('PREFLIGHT',cuda,flush=True)
    plan=dict(N0='final N0 nominal unbounded input, single-instance task-cost weighting',
        temperature=64.,covariance=[9,9,9,.3136,.3136,.3136],K=136,H=10,
        dob_Q_hz=3.,acceleration_lpf_hz=6.,output_compensation=False,rollout_compensation=False,
        headroom=False,input_tightening=False,seeds=SEEDS,cases=CASES,
        scenario='geometry(.89): R=.04m circle, period/duration6s; d6: 0 until.5s, raised cosine to -5.5Nm at2.5s then hold',
        oracle_scope='True d used only by margin generation; observer still estimates normally',
        cfg={case:asdict(margin_config(case)) for case in CASES},cuda=cuda,
        source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
            [__file__,'controllers/final_n0_controller.py','kroc/input_headroom_study.py','kroc/state_margin.py']})
    path=ROOT/'plan.json'
    if path.exists():
        old=json.loads(path.read_text());assert old==plan,'Frozen comparison settings changed'
    else:write(path,plan)
    for seed in ([args.seed] if args.seed is not None else SEEDS):
        for case in CASES:run_one(seed,case)
        plot(seed)
    rows=[json.loads(p.read_text()) for p in ROOT.glob('*_seed*.json')]
    summarize(rows)


if __name__=='__main__':main()
