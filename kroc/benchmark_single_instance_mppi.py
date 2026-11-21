"""Bounded GPU comparison of IID/constant-horizon proposals and task-cost weights.

Experiment-only adapters. No changes to production controller/defaults. This
borrows the paper's sampling/weighting components, not its complete controller.
"""
from dataclasses import asdict
import functools
import gc
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import DT,NOISE_VAR
from experiment_runner import run_mppi
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.search_lowfreq_estimator_margin import geometry,smooth_load,CASES,STATIC
from kroc.find_estimator_directional_separation import AuditedCircleTracker
from kroc.audit_circle_validation import dense_mesh_check
from kroc.tune_vanilla_input_smoothness import metrics

ROOT=Path('kroc/results/single_instance_mppi_20261115')
BASE=Path('kroc/results/vanilla_input_smoothness_20261115/calibration')
PROFILES=[('nominal',0.),('slow_disturbance',5.5)]
VALIDATION_SEEDS=[4307,4317,4327]


def sample_single(ctrl,shape,original):
    if len(shape)!=2:
        return original(shape)
    k,h=shape
    z=torch.randn(k,1,ctrl.nu,device=ctrl.d,dtype=ctrl.dtype)
    n=(z*ctrl._noise_sigma_sqrt_diag if ctrl._diagonal_sigma else z@ctrl._noise_sigma_chol.T)+ctrl.noise_mu
    return n.expand(k,h,ctrl.nu)


def paper_total_cost(ctrl):
    ctrl._compute_perturbed_action_and_noise()
    cost,ctrl.states,actions=ctrl._compute_rollout_costs(ctrl.perturbed_action)
    ctrl.actions=actions/ctrl.u_scale if actions is not None else None
    ctrl.cost_total=cost
    return cost


class PaperWeightTracker(AuditedCircleTracker):
    def __init__(self,*args,proposal='iid',**kwargs):
        super().__init__(*args,**kwargs)
        if proposal not in {'iid','single','mixture'}:
            raise ValueError('Unknown proposal')
        self.ctrl._compute_total_cost_batch=functools.partial(paper_total_cost,self.ctrl)
        if proposal=='single':
            self.ctrl._sample_noise=functools.partial(sample_single,self.ctrl,original=self.ctrl._sample_noise)


def spec(name,proposal='single',temperature=.24,smooth=.25,paper=True):
    return dict(name=name,proposal=proposal,temperature=temperature,smooth=smooth,paper=paper)


BASELINE=spec('original_iid',proposal='iid',paper=False)
IID=spec('paper_iid',proposal='iid')
SINGLE=spec('paper_single')


def run_one(case,profile,amp,seed,phase):
    folder=ROOT/phase;folder.mkdir(exist_ok=True)
    path=folder/f'{case["name"]}_{profile}_seed{seed}.npz'
    if case==BASELINE and seed==2007:
        path=BASE/f'{profile}_seed2007_w0.25.npz'
    if not path.exists():
        reference,q0,_=geometry(.89)
        rho=1. if case['proposal']=='single' else (.8 if case['proposal']=='mixture' else 0.)
        tracker=PaperWeightTracker if case['paper'] else AuditedCircleTracker
        options=dict(margin_config=asdict(CASES[STATIC]),
            sensor_noise=dict(position_std_rad=.001,velocity_std_rad_s=.01,seed=98000+seed),
            dob_acceleration_lpf_cutoff_hz=0.,dob_cutoff_hz=3.)
        if case.get('collision_weight',0.):
            options.update(collision_weight=case['collision_weight'],collision_clearance=.005,
                           collision_backend='triton')
        if case['paper']:options['proposal']=case['proposal']
        print('RUN',phase,case['name'],profile,seed,flush=True)
        z=run_mppi(reference,initial_q=q0,has_disturbance=bool(amp),use_dob=True,
            dob_compensation_mode='estimate_only',tracker_class=tracker,
            disturbance_function=smooth_load(amp),tracker_options=options,
            require_cuda=True,seed=seed,temperature=case['temperature'],temporal_noise_rho=rho,
            mean_update_alpha=1.,smooth_weight=case['smooth'],n_samples=136,horizon=10,
            noise_variance=NOISE_VAR,compile_rollout_loop=False,
            joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.)
        assert str(z['compute_device'])=='cuda'
        assert np.allclose(z['u_applied'],z['u_mppi'])
        np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')},
            experimental_proposal=np.asarray(case['proposal']),
            experimental_weighting=np.asarray('rollout_cost_only' if case['paper'] else 'upstream_importance'))
        del z;gc.collect();torch.cuda.empty_cache()
    mesh=dense_mesh_check((str(path),4))
    row=dict(case=case,profile=profile,seed=seed,path=str(path),mesh=mesh,phase=phase)
    print('AUDIT',case['name'],profile,seed,'first_contact',mesh['first_collision_index'],flush=True)
    return row


def group(case,seeds,phase):
    return [run_one(case,p,a,seed,phase) for seed in seeds for p,a in PROFILES]


def summarize(rows):
    output=[]
    for r in rows:
        hit=r['mesh']['first_collision_index']
        n=300 if hit is None else max(2,min(300,int(np.floor((hit*.005-DT)/DT+1e-9))))
        with np.load(r['path']) as z:
            m=metrics(z,n);u=z['u_applied'][:n];du=np.diff(u,axis=0)
            m.update(ess_mean=float(np.mean(z['mppi_weight_ess'][:n])),
                delta_per_joint_rms_nm=np.sqrt(np.mean(du**2,axis=0)).tolist(),
                saturation_fraction=float(np.mean(z['actuator_saturated'][:n])),
                sample_clamp_fraction=float(np.mean(z['mppi_sample_clamp_fraction'][:n])),
                solve_p95_ms=float(np.nanpercentile(z['solve_t'][:n],95)*1000))
        output.append(dict(**r,safe_end_s=n*DT,first_contact_s=None if hit is None else hit*.005,metrics=m))
    (ROOT/'metrics.json').write_text(json.dumps(output,indent=2))
    return output


def compare(base,candidate):
    records=[]
    for b,c in zip(sorted(base,key=lambda r:(r['seed'],r['profile'])),sorted(candidate,key=lambda r:(r['seed'],r['profile']))):
        assert (b['seed'],b['profile'])==(c['seed'],c['profile'])
        hits=[r['mesh']['first_collision_index']*.005 for r in [b,c] if r['mesh']['first_collision_index'] is not None]
        n=300 if not hits else max(2,min(300,int(np.floor((min(hits)-DT)/DT+1e-9))))
        with np.load(b['path']) as z:bm=metrics(z,n)
        with np.load(c['path']) as z:cm=metrics(z,n)
        records.append(dict(profile=b['profile'],seed=b['seed'],safe_end_s=n*DT,
            baseline=bm,candidate=cm,candidate_contact=c['mesh']['first_collision_index'] is not None,
            baseline_contact=b['mesh']['first_collision_index'] is not None,
            tracking_ratio=cm['tracking_rmse_mm']/bm['tracking_rmse_mm'],
            torque_ratio=cm['delta_torque_rms_nm']/bm['delta_torque_rms_nm']))
    return records


def eligible(records):
    return all(not r['candidate_contact'] and r['tracking_ratio']<=1.2 and r['torque_ratio']<=.8
        and r['candidate']['max_position_excess_rad']<=r['baseline']['max_position_excess_rad']+.01
        for r in records)


def plot(base,others,phase,seed=2007):
    for profile,_ in PROFILES:
        rr=[next(r for r in group_rows if r['profile']==profile and r['seed']==seed) for group_rows in [base]+others]
        hits=[r['mesh']['first_collision_index']*.005 for r in rr if r['mesh']['first_collision_index'] is not None]
        n=300 if not hits else max(2,min(300,int(np.floor((min(hits)-DT)/DT+1e-9))))
        fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained');t=np.arange(n)*DT
        for r in rr:
            label=r['case']['name']
            with np.load(r['path']) as z:
                axes[0,0].plot(t,z['err_mm'][:n],label=label)
                axes[0,1].plot(t,z['u_applied'][:n,5],label=label,linewidth=.8)
                axes[1,0].plot(t,z['mppi_weight_ess'][:n],label=label)
                d=np.sqrt(np.mean(np.diff(z['u_applied'][:n],axis=0)**2,axis=0))
                axes[1,1].plot(np.arange(1,7),d,'-o',label=label)
        for ax,y in zip(axes.flat,['EE error [mm]','Applied torque6 [Nm]','ESS /136','Delta torque RMS [Nm/step]']):
            ax.set_ylabel(y);ax.grid(alpha=.25);ax.legend(fontsize=7)
        for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.set_xlabel('Time [s]')
        axes[1,1].set_xlabel('Joint')
        collision_note='; online collision cost ON' if any(r['case'].get('collision_weight',0) for r in rr) else '; online collision cost OFF'
        fig.suptitle(f'{phase}: {profile}; seed{seed}; common pre-contact window {n*DT:.2f}s\nSensor noise ON; no DOB compensation; fixed2.5%; FF/residual warm start retained{collision_note}')
        fig.savefig(ROOT/f'{phase}_{profile}.png',dpi=150);plt.close(fig)


def main():
    ROOT.mkdir(exist_ok=True);print('PREFLIGHT',json.dumps(cuda_preflight()),flush=True)
    plan=dict(pilot_seed=2007,heldout_seeds=VALIDATION_SEEDS,initial_cases=[BASELINE,IID,SINGLE],
        tuning='single temperatures 1,4,16; then smooth=1 at best single temperature; mixture rho=.8 at same temperature',
        gate='No candidate contacts; tracking RMSE <=1.2x baseline; input delta RMS <=.8x baseline; position excess <=baseline+.01rad, BOTH profiles',
        defaults_changed=False,not_full_paper_replication=True)
    (ROOT/'plan.json').write_text(json.dumps(plan,indent=2))
    rows=[];base=group(BASELINE,[2007],'pilot');rows.extend(base)
    initial=[];candidates=[]
    for case in [IID,SINGLE]:
        rr=group(case,[2007],'pilot');rows.extend(rr);initial.append(rr)
        rec=compare(base,rr);candidates.append(dict(case=case,records=rec,eligible=eligible(rec)))
        summarize(rows)
    plot(base,initial,'initial')
    singles=[initial[1]]
    for temp in [1.,4.,16.]:
        case=spec(f'single_lambda{temp:g}',temperature=temp)
        rr=group(case,[2007],'tuning');rows.extend(rr);singles.append(rr)
        rec=compare(base,rr);candidates.append(dict(case=case,records=rec,eligible=eligible(rec)));summarize(rows)
    # Rank safe full-horizon single cases; fallback ranks collision-free prefixes,
    # but a colliding candidate can never pass the selection gate.
    def score(rr):
        rec=compare(base,rr)
        return (sum(r['candidate_contact'] for r in rec),max(r['tracking_ratio'] for r in rec),
                np.mean([r['torque_ratio'] for r in rec]))
    best_single=min(singles,key=score);temp=best_single[0]['case']['temperature']
    for case in [spec(f'single_lambda{temp:g}_smooth1',temperature=temp,smooth=1.),
                 spec(f'mixture08_lambda{temp:g}',proposal='mixture',temperature=temp)]:
        rr=group(case,[2007],'tuning');rows.extend(rr)
        rec=compare(base,rr);candidates.append(dict(case=case,records=rec,eligible=eligible(rec)));summarize(rows)
    accepted=[c for c in candidates if c['eligible']]
    chosen=min(accepted,key=lambda c:np.mean([r['torque_ratio']*r['tracking_ratio'] for r in c['records']])) if accepted else None
    selection=dict(candidates=candidates,selected=None if chosen is None else chosen['case'])
    (ROOT/'selection.json').write_text(json.dumps(selection,indent=2))
    plot(base,[best_single],'best_single')
    verdict=dict(defaults_changed=False,pilot_passed=chosen is not None,heldout_completed=False)
    if chosen:
        selected_rows=[r for r in rows if r['case']==chosen['case']]
        plot(base,[selected_rows],'selected')
        vb=group(BASELINE,VALIDATION_SEEDS,'heldout');rows.extend(vb);summarize(rows)
        vc=group(chosen['case'],VALIDATION_SEEDS,'heldout');rows.extend(vc);summarize(rows)
        records=compare(vb,vc)
        verdict.update(heldout_completed=True,heldout_passed=eligible(records),selected=chosen['case'],records=records)
    (ROOT/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print('DONE',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
