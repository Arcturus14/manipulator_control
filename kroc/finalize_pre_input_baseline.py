"""Freeze the simulation baseline BEFORE input tightening; bounded GPU checks."""
from dataclasses import asdict
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import DT,NOISE_VAR,TAU_MAX,W_ACT,W_VEL
from experiment_runner import run_mppi
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.search_lowfreq_estimator_margin import geometry,smooth_load,CASES,STATIC
from kroc.benchmark_single_instance_mppi import PaperWeightTracker
from kroc.audit_circle_validation import dense_mesh_check
from kroc.tune_vanilla_input_smoothness import metrics

ROOT=Path('kroc/results/pre_input_baseline_20261115')
PROFILES=[('nominal',0.),('slow_disturbance',5.5)]
PILOT=4507
HELDOUT=[4607,4617,4627]


def boundary_extra(action,previous,weight,horizon_weight):
    return (weight-horizon_weight)*(action-previous).square().sum(dim=-1)


class BoundaryPaperTracker(PaperWeightTracker):
    def __init__(self,*args,boundary_weight=.25,**kwargs):
        if not np.isfinite(boundary_weight) or boundary_weight<0:
            raise ValueError('boundary weight must be finite and nonnegative')
        super().__init__(*args,**kwargs)
        self.boundary_weight=float(boundary_weight)

    def _running_cost(self,state,action,rollout_step):
        cost=super()._running_cost(state,action,rollout_step)
        if rollout_step==0 and self.boundary_weight!=self.tracking_cost.smooth_weight:
            torque=action.clamp(self.dynamics.u_min,self.dynamics.u_max)
            cost=cost+boundary_extra(torque,self.tracking_cost.previous_action,
                                    self.boundary_weight,self.tracking_cost.smooth_weight)
        return cost


def measurement(z,n):
    m=metrics(z,n)
    applied=z['u_applied'][:n];mppi=z['u_mppi'][:n]
    estimate=z['d_hat_used'][:n]
    post=str(z['dob_compensation_mode'])=='output'
    requested=mppi+estimate if post else mppi
    cut=requested-applied
    excess=np.maximum(np.abs(requested)-np.asarray(TAU_MAX),0.)
    cycle=z['control_cycle_t_raw'][min(20,n-1):n]
    # Verify the actual implementation and sign, not merely requested settings.
    np.testing.assert_allclose(applied,np.clip(requested,-np.asarray(TAU_MAX),TAU_MAX),atol=2e-6)
    m.update(ess_mean=float(np.mean(z['mppi_weight_ess'][:n])),
        mppi_delta_rms_nm=float(np.sqrt(np.mean(np.diff(mppi,axis=0)**2))),
        cut_rms_nm=float(np.sqrt(np.mean(cut**2))),cut_max_nm=float(np.max(np.abs(cut))),
        requested_excess_max_nm=float(excess.max()),
        saturation_fraction=float(np.mean(np.any(excess>1e-5,axis=1))),
        per_joint_delta_rms_nm=np.sqrt(np.mean(np.diff(applied,axis=0)**2,axis=0)).tolist(),
        solve_p95_ms=float(np.nanpercentile(z['solve_t'][:n],95)*1000),
        control_cycle_mean_ms=float(np.mean(cycle)*1000),
        control_cycle_p95_ms=float(np.percentile(cycle,95)*1000),
        dob6_rmse_nm=float(np.sqrt(np.mean((estimate[:,5]-z['d_true'][1:n+1,5])**2))))
    return m


def run_one(weight,mode,profile,amp,seed,phase):
    folder=ROOT/phase;folder.mkdir(exist_ok=True)
    path=folder/f'{mode}_{profile}_wb{weight:g}_seed{seed}.npz'
    if not path.exists():
        reference,q0,_=geometry(.89)
        print('RUN',phase,weight,mode,profile,seed,flush=True)
        z=run_mppi(reference,initial_q=q0,has_disturbance=bool(amp),use_dob=True,
            dob_compensation_mode=mode,tracker_class=BoundaryPaperTracker,
            disturbance_function=smooth_load(amp),
            tracker_options=dict(proposal='single',boundary_weight=weight,
                margin_config=asdict(CASES[STATIC]),
                sensor_noise=dict(position_std_rad=.001,velocity_std_rad_s=.01,seed=98000+seed),
                dob_acceleration_lpf_cutoff_hz=0.,dob_cutoff_hz=3.,
                collision_weight=50000.,collision_clearance=.005,collision_backend='triton'),
            require_cuda=True,seed=seed,temperature=1.,temporal_noise_rho=1.,
            mean_update_alpha=1.,smooth_weight=.25,n_samples=136,horizon=10,
            noise_variance=NOISE_VAR,compile_rollout_loop=False,
            position_weight=1200000.,terminal_weight=3600000.,
            joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.)
        assert str(z['compute_device'])=='cuda'
        np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')},
                            boundary_weight=np.asarray(weight))
        del z;gc.collect();torch.cuda.empty_cache()
    mesh=dense_mesh_check((str(path),4));hit=mesh['first_collision_index']
    n=300 if hit is None else max(2,min(300,int(np.floor((hit*.005-DT)/DT+1e-9))))
    with np.load(path) as z:m=measurement(z,n)
    row=dict(boundary_weight=weight,mode=mode,profile=profile,seed=seed,phase=phase,
             path=str(path),mesh=mesh,safe_end_s=n*DT,metrics=m)
    print('METRIC',json.dumps(row),flush=True)
    return row


def group(weight,mode,seeds,phase):
    return [run_one(weight,mode,p,a,s,phase) for s in seeds for p,a in PROFILES]


def paired(baseline,candidate):
    result=[]
    for b,c in zip(sorted(baseline,key=lambda x:(x['seed'],x['profile'])),
                   sorted(candidate,key=lambda x:(x['seed'],x['profile']))):
        assert (b['seed'],b['profile'])==(c['seed'],c['profile'])
        n=round(min(b['safe_end_s'],c['safe_end_s'])/DT)
        with np.load(b['path']) as z:bm=measurement(z,n)
        with np.load(c['path']) as z:cm=measurement(z,n)
        result.append(dict(seed=b['seed'],profile=b['profile'],safe_end_s=n*DT,
            baseline=bm,candidate=cm,candidate_contact=c['mesh']['first_collision_index'] is not None,
            tracking_ratio=cm['tracking_rmse_mm']/bm['tracking_rmse_mm'],
            delta_ratio=cm['delta_torque_rms_nm']/bm['delta_torque_rms_nm']))
    return result


def usable(row):
    return (row['mesh']['first_collision_index'] is None and row['safe_end_s']==6.
            and row['metrics']['max_position_excess_rad']<1e-6
            and row['metrics']['tracking_rmse_mm']<=5.
            and row['metrics']['delta_torque_rms_nm']<=3.5)


def plot(rows,weight,seed,phase):
    for profile,_ in PROFILES:
        rr=[r for r in rows if r['seed']==seed and r['profile']==profile and r['boundary_weight']==weight]
        if len(rr)<2:continue
        n=round(min(r['safe_end_s'] for r in rr)/DT);t=np.arange(n)*DT
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
        for r in rr:
            label='Post-DOB' if r['mode']=='output' else 'DOB estimate only'
            with np.load(r['path']) as z:
                axes[0,0].plot(t,z['err_mm'][:n],label=label)
                axes[0,1].plot(t,z['u_applied'][:n,5],label=label,linewidth=.85)
                axes[1,0].plot(t,z['d_hat_used'][:n,5],label=label)
                axes[1,1].plot(t,z['q'][:n,5],label=label)
        axes[0,0].set_ylabel('EE error [mm]');axes[0,1].set_ylabel('Applied torque6 [Nm]')
        axes[1,0].set_ylabel('Estimated disturbance6 [Nm]');axes[1,1].set_ylabel('q6 [rad]')
        for ax in axes.flat:ax.grid(alpha=.3);ax.legend(fontsize=8);ax.set_xlabel('Time [s]')
        fig.suptitle(f'{profile}; seed{seed}; wb={weight:g}, wh=.25, lambda=1\nSingle-instance, online collision cost, sensor noise ON; NO input tightening')
        fig.savefig(ROOT/f'{phase}_{profile}_seed{seed}.png',dpi=150);plt.close(fig)


def main():
    ROOT.mkdir(exist_ok=True);print('PREFLIGHT',json.dumps(cuda_preflight()),flush=True)
    plan=dict(pilot_seed=PILOT,heldout_seeds=HELDOUT,weights=[.25,.5,1.],
        tuning_rule='Smallest increased boundary weight with >=10% delta RMS reduction in BOTH profiles, <=10% RMSE increase, no contact or position violation; else retain .25',
        acceptance='Each heldout run: full6s no contact/no position violation, RMSE <=5mm, delta RMS<=3.5Nm; Post-DOB must also have no applied clamp loss',
        fixed=dict(proposal='single',temperature=1.,horizon_weight=.25,collision_weight=50000.,
            state_margin='fixed2.5%',no_input_tightening=True,output_filter=False),
        boundary_anchor='previous MPPI command (not previous post-DOB applied torque)',
        timing='20ms simulation step; no actuation delay model; realtime readiness evaluated separately')
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2))
    rows=[];base=group(.25,'estimate_only',[PILOT],'pilot');rows.extend(base)
    selected=.25;candidates=[]
    for w in [.5,1.]:
        rr=group(w,'estimate_only',[PILOT],'pilot');rows.extend(rr);rec=paired(base,rr)
        ok=all(usable(r) for r in rr) and all(r['tracking_ratio']<=1.1 and r['delta_ratio']<=.9 for r in rec)
        candidates.append(dict(weight=w,eligible=ok,records=rec))
        if ok and selected==.25:selected=w
        (ROOT/'metrics.json').write_text(json.dumps(rows,indent=2))
    (ROOT/'selection.json').write_text(json.dumps(dict(selected_boundary_weight=selected,candidates=candidates),indent=2))
    for mode in ['estimate_only','output']:
        rr=group(selected,mode,HELDOUT,'heldout');rows.extend(rr)
        (ROOT/'metrics.json').write_text(json.dumps(rows,indent=2))
    held=[r for r in rows if r['phase']=='heldout']
    simulation_passed=all(usable(r) and (r['mode']!='output' or r['metrics']['cut_max_nm']<1e-5) for r in held)
    result=dict(simulation_baseline_passed=simulation_passed,selected_boundary_weight=selected,
        heldout_seeds=HELDOUT,heldout_runs=len(held),input_tightening_applied=False,
        realtime_20ms_passed=all(r['metrics']['control_cycle_p95_ms']<=20 for r in held),
        production_defaults_changed=False,
        summary=[dict(mode=mode,profile=profile,runs=len(rr),
            tracking_rmse_mean_mm=float(np.mean([r['metrics']['tracking_rmse_mm'] for r in rr])),
            delta_torque_mean_nm=float(np.mean([r['metrics']['delta_torque_rms_nm'] for r in rr])),
            ess_median_mean=float(np.mean([r['metrics']['ess_median'] for r in rr])),
            contacts=sum(r['mesh']['first_collision_index'] is not None for r in rr),
            position_violations=sum(r['metrics']['max_position_excess_rad']>1e-6 for r in rr),
            max_clamp_loss_nm=max(r['metrics']['cut_max_nm'] for r in rr),
            solve_mean_ms=float(np.mean([r['metrics']['solve_mean_ms'] for r in rr])),
            cycle_p95_max_ms=max(r['metrics']['control_cycle_p95_ms'] for r in rr))
            for mode in ['estimate_only','output'] for profile,_ in PROFILES
            for rr in [[r for r in held if r['mode']==mode and r['profile']==profile]]])
    (ROOT/'verdict.json').write_text(json.dumps(result,indent=2))
    for seed in HELDOUT:plot(held,selected,seed,'heldout')
    # Freeze only when the predeclared checks pass; never pretend realtime passed.
    if simulation_passed:
        config=dict(schema_version=1,scope='validated simulation baseline; NOT realtime20ms certified',
            tracker='kroc.finalize_pre_input_baseline.BoundaryPaperTracker',
            proposal='single',weighting='exp(-rollout_cost/temperature)',temperature=1.,
            boundary_weight=selected,horizon_smooth_weight=.25,mean_update_alpha=1.,
            samples=136,horizon=10,dt_s=DT,noise_variance=list(NOISE_VAR),
            collision_weight=50000.,collision_clearance_m=.005,collision_backend='triton',
            fixed_effort_weight=W_ACT,fixed_velocity_weight=W_VEL,
            position_weight=1200000.,terminal_weight=3600000.,
            joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.,
            sensor_position_std_rad=.001,sensor_velocity_std_rad_s=.01,
            dob_cutoff_hz=3.,dob_acceleration_lpf_hz=0.,
            sampling_center_mode='feedforward_residual',state_margin=asdict(CASES[STATIC]),
            output_filter=False,input_tightening=False,default_dob_mode='output',
            supported_dob_modes=['estimate_only','output'],
            evidence=str(ROOT/'verdict.json'),validation_seeds=HELDOUT,
            realtime_20ms_passed=result['realtime_20ms_passed'])
        folder=Path('kroc/configs');folder.mkdir(exist_ok=True)
        (folder/'pre_input_study_baseline.json').write_text(json.dumps(config,indent=2))
    print('DONE',json.dumps(result),flush=True)


if __name__=='__main__':main()
