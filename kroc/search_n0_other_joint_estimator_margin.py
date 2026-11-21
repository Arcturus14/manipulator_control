"""Static-failure-first, bounded CUDA search with estimator-only margins.

No default/controller tuning, post compensation, rollout disturbance, headroom,
oracle or MD reports. Invalid mesh-contact episodes are excluded, not successes.
"""
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import DT, Q_MIN, Q_MAX, TAU_MAX
from kinematics.ur5_kinematics import fk_joints_np
from experiment_runner import run_mppi
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.compare_n0_state_margins import options, audit, margin_config
from kroc.find_estimator_directional_separation import replay_margin
from kroc.position_input_headroom_study import raised_cosine_load
from kroc.screen_nominal_margin_pressure import circle
from kroc.search_circle_floor25 import local_ik_reference
from kroc.collision_geometry import Geometry
from kroc.audit_circle_validation import dense_mesh_check

ROOT=Path('kroc/results/n0_other_joint_estimator_margin_20261118')
PILOT=8607
HELDOUT=[8707,8717,8727]
CASES=['static25','dynamic_estimate','static5']
LABELS=['Static 2.5%','Dynamic 2.5-5%','Static 5%']
COLORS=['tab:orange','tab:blue','tab:green']
SCENES=[dict(joint=0,radius=.08,amplitudes=[100.,140.,150.,160.]),
        dict(joint=3,radius=.04,amplitudes=[16.,24.,28.,32.]),
        dict(joint=4,radius=.04,amplitudes=[16.,24.,28.,32.]),
        dict(joint=1,radius=.04,amplitudes=[100.,140.,160.,180.])]
MAX_RUNS=40
RUN_COUNT=0


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False))


def geometry(scene):
    q=np.array([np.pi,-1.2,1.2,-1.5,-1.57,0.])
    if scene['joint']==1:q[2]=-1.2
    q[scene['joint']]=.89*Q_MAX[scene['joint']]
    ref,_=circle(q,scene['radius']/np.linalg.norm(fk_joints_np(q)[-1,:2]),'xy',6.,6.)
    return ref,q


def prefilter():
    geom=Geometry();rows=[];valid=[]
    for scene in SCENES:
        ref,q=geometry(scene);qr=local_ik_reference(ref[::5],q)
        mesh=geom.trajectory(qr)
        corridor=np.broadcast_to(q,(101,6)).copy()
        corridor[:,scene['joint']]=np.linspace(q[scene['joint']],Q_MAX[scene['joint']]+.1,101)
        risk=geom.trajectory(corridor)
        error=max(np.linalg.norm(fk_joints_np(x)[-1]-y) for x,y in zip(qr,ref[::5]))
        accepted=bool(mesh['first_collision_index'] is None and risk['first_collision_index'] is None and error<1e-4)
        row=dict(**scene,initial_q=q.tolist(),reference_mesh=mesh,risk_corridor_mesh=risk,
                 max_local_ik_error_mm=float(error*1000),accepted=accepted)
        rows.append(row)
        if accepted:valid.append(scene)
    write(ROOT/'prefilter.json',rows)
    print('PREFILTER accepted joints',[s['joint']+1 for s in valid],flush=True)
    return valid


def measure(z,joint):
    q=z['q'];n=len(z['u_applied']);j=joint
    excess=np.maximum(np.maximum(q-np.asarray(Q_MAX),np.asarray(Q_MIN)-q),0.)
    peak=excess[1:].max(1);hit=np.flatnonzero(peak>1e-6)
    cut=z['u_mppi']-z['u_applied'];margin=z['kroc_margin'][:,0,1,j]
    shadow=replay_margin(z['measured_state'][:n,6:],np.zeros((n,6)),margin_config('dynamic_estimate'))
    return dict(rmse_mm=float(np.sqrt(np.mean(z['err_mm']**2))),
        max_position_excess_rad=float(excess.max()),per_joint_max_excess_rad=excess.max(0).tolist(),
        violation_steps=int((peak>1e-6).sum()),violation_duration_s=float((peak>1e-6).sum()*DT),
        excess_integral_rad_s=float(peak.sum()*DT),
        first_violation_s=None if len(hit)==0 else float((hit[0]+1)*DT),
        risk_joint_upper_clearance_rad=float(Q_MAX[j]-q[:,j].max()),
        risk_joint_q_max_rad=float(q[:,j].max()),
        saturation_steps=int(np.any(np.abs(cut)>1e-5,axis=1).sum()),
        risk_joint_saturation_steps=int((np.abs(cut[:,j])>1e-5).sum()),
        max_clamp_loss_nm=float(np.abs(cut).max()),
        margin_upper_min_pct=float(100*margin.min()/(Q_MAX[j]-Q_MIN[j])),
        margin_upper_max_pct=float(100*margin.max()/(Q_MAX[j]-Q_MIN[j])),
        same_state_dhat_upper_extra_max_rad=float((margin-shadow[:,0,1,j]).max()),
        dob_risk_joint_rmse_nm=float(np.sqrt(np.mean((z['d_hat_used'][:,j]-z['d_true'][1:,j])**2))),
        velocity_violation_steps=int(z['joint_velocity_limit_violated'].sum()),
        solve_mean_ms=float(np.mean(z['solve_t'][20:])*1000))


def run(scene,amplitude,case,seed=PILOT,phase='pilot'):
    global RUN_COUNT
    folder=ROOT/phase;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'j{scene["joint"]+1}_a{amplitude:g}_{case}_seed{seed}.npz';meta=path.with_suffix('.json')
    if meta.exists():return json.loads(meta.read_text())
    if RUN_COUNT>=MAX_RUNS:raise RuntimeError('Predeclared GPU run budget exhausted')
    if not path.exists():
        ref,q0=geometry(scene);load=raised_cosine_load(scene['joint'],-amplitude)
        print('RUN',scene['joint']+1,amplitude,case,seed,flush=True)
        z=run_mppi(ref,initial_q=q0,has_disturbance=bool(amplitude),disturbance_function=load,
                   **options(seed,case,load));RUN_COUNT+=1
        audit(z,case)
        z.update(circle_reference=ref,disturbed_joint=np.asarray(scene['joint']+1),
                 disturbance_amplitude_nm=np.asarray(amplitude))
        np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')})
        del z;gc.collect();torch.cuda.empty_cache()
    mesh=dense_mesh_check((str(path),4))
    with np.load(path) as z:
        audit(z,case);metrics=measure(z,scene['joint'])
        n=len(z['u_applied']);expected=np.zeros((n,6));t=np.arange(n)*DT
        expected[:,scene['joint']]=-amplitude*.5*(1-np.cos(np.pi*np.clip((t-.5)/2,0,1)))
        np.testing.assert_allclose(z['d_true'][1:],expected,atol=3e-5,rtol=1e-6)
        span=np.asarray(Q_MAX)-Q_MIN
        assert np.minimum(z['q'][0]-np.asarray(Q_MIN)-.05*span,
                          np.asarray(Q_MAX)-.05*span-z['q'][0]).min()>0
    row=dict(joint=scene['joint'],amplitude_nm=amplitude,case=case,seed=seed,phase=phase,
             path=str(path),metrics=metrics,mesh_5ms=mesh,
             eligible=mesh['first_collision_index'] is None)
    write(meta,row)
    # Do not print rejected collision details; retain private raw audit for reproducibility.
    if row['eligible']:print('VALID',json.dumps({k:row[k] for k in ['joint','amplitude_nm','case','metrics']}),flush=True)
    else:print('EXCLUDED',scene['joint']+1,amplitude,case,flush=True)
    return row


def static_failure(row):
    return bool(row['eligible'] and row['metrics']['max_position_excess_rad']>.001)


def separation(low,dyn):
    return bool(static_failure(low) and dyn['eligible']
        and dyn['metrics']['max_position_excess_rad']<=1e-6
        and dyn['metrics']['velocity_violation_steps']==0
        and max(low['metrics']['rmse_mm'],dyn['metrics']['rmse_mm'])<=50.)


def plot(rows,scene,prefix):
    j=scene['joint'];fig,axs=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for row,label,color in zip(rows,LABELS,COLORS):
        assert row['eligible']
        with np.load(row['path']) as z:
            t=np.arange(300)*DT;tq=np.arange(301)*DT
            if row==rows[0]:axs[0,0].plot(t,z['d_true'][1:,j],'k',label='Actual d')
            axs[0,0].plot(t,z['d_hat_used'][:,j],color=color,label=label+' DOB')
            axs[0,1].plot(t,100*z['kroc_margin'][:,0,1,j]/(Q_MAX[j]-Q_MIN[j]),color=color,label=label)
            axs[1,0].plot(tq,z['q'][:,j],color=color,label=label)
            axs[1,0].plot(t,Q_MAX[j]-z['kroc_margin'][:,0,1,j],color=color,ls=':',lw=.8)
            axs[1,1].plot(t+DT,z['err_mm'],color=color,label=label)
            axs[2,0].plot(t,z['u_applied'][:,j],color=color,label=label)
            axs[2,1].plot(t,z['u_mppi'][:,j]-z['u_applied'][:,j],color=color,label=label)
    axs[1,0].axhline(Q_MAX[j],color='k',ls='--',label='Actual upper limit')
    axs[2,0].axhline(-TAU_MAX[j],color='k',ls='--');axs[2,0].axhline(TAU_MAX[j],color='k',ls='--')
    for ax,title,unit in zip(axs.flat,['Actual disturbance / DOB','Upper state margin',f'Joint{j+1} position / soft onset','EE tracking error','Applied torque','Torque cut by final clamp'],['Nm','% full position span','rad','mm','Nm','Nm']):
        ax.set(title=title,xlabel='Time [s]',ylabel=unit);ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle(f'Frozen N0; NO output/rollout compensation; joint{j+1}, d -> -{rows[0]["amplitude_nm"]:g}Nm\n'
                 f'Circle R={scene["radius"]:g}m / 6s; no mesh contact at5ms sampling; seed{rows[0]["seed"]}')
    fig.savefig(ROOT/f'{prefix}.png',dpi=160);plt.close(fig)


def main():
    ROOT.mkdir(parents=True,exist_ok=True);valid=prefilter()
    cuda=cuda_preflight();print('PREFLIGHT',cuda,flush=True)
    plan=dict(scenes=SCENES,pilot=PILOT,heldout=HELDOUT,max_new_gpu_runs=MAX_RUNS,
        no_output_or_rollout_compensation=True,no_headroom=True,no_controller_tuning=True,
        static_failure_first=True,margin_fractions=[.025,.05],
        initial_fraction_of_upper=.89,circle_period_s=6.,duration_s=6.,
        load='negative, .5s onset, 2s raised cosine rise then held to6s',
        refinement='up to4 bisections of first safe-static / violating-static interval per scene; dynamic only at eligible static failures',
        primary='static25 excess>1mrad, dynamic <=1urad, both collision-free, dynamic no velocity violation, both RMSE<=50mm',
        secondary='collision-free static5 RMSE >1.05*dynamic RMSE',cuda=cuda,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            [__file__,'kroc/compare_n0_state_margins.py','kroc/state_margin.py','controllers/final_n0_controller.py']})
    plan_path=ROOT/'plan.json'
    if plan_path.exists():assert json.loads(plan_path.read_text())==plan
    else:write(plan_path,plan)
    history=[];candidate=None;selected_scene=None
    for scene in valid:
        if RUN_COUNT>=28:break
        # Sanity check actual tracking before disturbance search; no collision results are reported.
        nominal=run(scene,0.,'static25');history.append(nominal)
        if not nominal['eligible'] or nominal['metrics']['rmse_mm']>20.:continue
        previous=0.;bracket=None
        for amplitude in scene['amplitudes']:
            if RUN_COUNT>=29:break
            low=run(scene,amplitude,'static25');history.append(low)
            if static_failure(low):
                dyn=run(scene,amplitude,'dynamic_estimate');history.append(dyn)
                if separation(low,dyn):candidate=[low,dyn];selected_scene=scene;break
                bracket=(previous,amplitude);break
            if low['eligible']:previous=amplitude
            write(ROOT/'progress.json',dict(rows=history,new_runs=RUN_COUNT))
        if candidate:break
        if bracket:
            left,right=bracket
            for _ in range(4):
                if RUN_COUNT>=29:break
                amplitude=round((left+right)/2,6)
                low=run(scene,amplitude,'static25');history.append(low)
                if not low['eligible']:break
                if static_failure(low):
                    right=amplitude
                    dyn=run(scene,amplitude,'dynamic_estimate');history.append(dyn)
                    if separation(low,dyn):candidate=[low,dyn];selected_scene=scene;break
                else:left=amplitude
                write(ROOT/'progress.json',dict(rows=history,new_runs=RUN_COUNT))
        if candidate:break
        if RUN_COUNT>=MAX_RUNS-12:break
    validation=[];upper=None
    if candidate:
        amp=candidate[0]['amplitude_nm'];upper=run(selected_scene,amp,'static5')
        if upper['eligible']:plot(candidate+[upper],selected_scene,'pilot_selected')
        for seed in HELDOUT:
            rows=[run(selected_scene,amp,c,seed,'validation') for c in CASES]
            pair=dict(seed=seed,primary=separation(rows[0],rows[1]),
                secondary=bool(rows[2]['eligible'] and rows[2]['metrics']['rmse_mm']>1.05*rows[1]['metrics']['rmse_mm']),rows=rows)
            validation.append(pair)
            if all(r['eligible'] for r in rows):plot(rows,selected_scene,f'validation_seed{seed}')
            write(ROOT/'validation_progress.json',validation)
    # Full audit retained, while public comparison includes only collision-free runs.
    write(ROOT/'search_audit.json',dict(rows=history,validation=validation))
    summary=dict(status='validated' if validation and all(p['primary'] for p in validation)
        else 'pilot_found_not_fully_reproduced' if candidate else 'stopped_no_qualifying_case',
        selected_scene=selected_scene,selected_amplitude_nm=None if not candidate else candidate[0]['amplitude_nm'],
        valid_pilot_rows=[r for r in history if r['eligible']],validation=validation,
        static5_pilot=upper if upper and upper['eligible'] else None,
        new_gpu_runs=RUN_COUNT,search_stopped=True,
        limitation='Finite search, not a proof that a separating scenario does not exist. No post-DOB saturation-compensation claim is supported by estimator-only experiments.')
    write(ROOT/'verdict.json',summary)
    print('DONE',summary['status'],'new GPU runs',RUN_COUNT,flush=True)


if __name__=='__main__':main()
