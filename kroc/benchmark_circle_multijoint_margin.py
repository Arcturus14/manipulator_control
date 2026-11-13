"""Experiment-only multi-joint sustained loads; production MPPI is unchanged."""
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT,Q_MAX,Q_MIN,TAU_MAX
from experiment_runner import run_mppi
from kroc.benchmark_noise_aware_state_margin import ACCELERATION_DEADBANDS
from kroc.benchmark_postdob_small_directional_margin import measure
from kroc.circle_risk_tradeoff_assessment import position_metrics
from kroc.collision_geometry import Geometry
from kroc.search_circle_floor25 import CircleTracker,local_ik_reference
from kroc.screen_nominal_margin_pressure import circle
from kroc.state_margin import MarginConfig

CASES={
    'static25':MarginConfig(mode='fixed',min_fraction=.025,fixed_fraction=.025,max_fraction=.05),
    'directional25_5':MarginConfig(mode='directional',min_fraction=.025,fixed_fraction=.025,
        max_fraction=.05,reaction_time_s=.10,beta_up=1.,beta_down=.1,
        acceleration_deadband_rad_s2=ACCELERATION_DEADBANDS['nominal_p99']),
    'static5':MarginConfig(mode='fixed',min_fraction=.025,fixed_fraction=.05,max_fraction=.05),
    'static35':MarginConfig(mode='fixed',min_fraction=.025,fixed_fraction=.035,max_fraction=.05)}


def load_circle(bank,candidate,risk_joint):
    bank=Path(bank)
    c=next(r for r in json.loads((bank/'screen.json').read_text())['selected'] if r['candidate']==candidate)
    assert c.get('shape','circle')=='circle'
    with np.load(bank/f'candidate_{candidate}.npz') as z:q0=z['initial_q'].copy()
    ref,radius=circle(q0,c['ratio'],c['plane'],period=8.,seconds=8.)
    qr=local_ik_reference(ref,q0);v=np.gradient(qr,DT,axis=0)
    threshold=qr[0,risk_joint]+.7*(qr[:,risk_joint].max()-qr[0,risk_joint])
    indices=np.flatnonzero((qr[:,risk_joint]>=threshold)&(v[:,risk_joint]>0))
    onset=max(.5,float(indices[0]*DT)) if len(indices) else 1.
    assert onset<=6.,'Need at least two seconds of sustained disturbance'
    return {'bank':str(bank),'candidate':candidate,'q0':q0,'reference':ref,
        'radius':radius,'metadata':c,'risk_joint':risk_joint,'onset':onset}


def run_experiment(output,geometry,profiles,seeds,names=('static25','directional25_5','static5')):
    cuda=cuda_preflight();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    q0=geometry['q0'];ref=geometry['reference'];j=geometry['risk_joint']
    span=np.asarray(Q_MAX)-np.asarray(Q_MIN)
    slack=np.minimum(q0-np.asarray(Q_MIN)-.05*span,np.asarray(Q_MAX)-.05*span-q0)
    assert slack.min()>0
    files=['controllers/mppi_dob_controller.py','costs/tracking_cost.py','experiment_runner.py',
        'observers/dob.py','config.py','kroc/state_margin.py','kroc/search_circle_floor25.py',__file__]
    manifest={'arguments':{'joint':j,'scale':1.,'seeds':list(seeds),'period':8.,'seconds':8.,
        'bank':geometry['bank'],'candidate':geometry['candidate']},'profiles':profiles,
        'cases':{n:asdict(CASES[n]) for n in names},'cuda':cuda,'headroom_enabled':False,
        'sensor_noise':{'q_std_rad':.001,'dq_std_rad_s':.01},
        'dob':{'acceleration_lpf_hz':10.,'q_hz':8.},
        'collision_cost':{'enabled':True,'backend':'triton','weight':50000.,'clearance_m':.005},
        'reference':'position-only Cartesian circle; no orientation/joint objective',
        'initial_q':q0.tolist(),'reference_sha256':hashlib.sha256(ref.tobytes()).hexdigest(),
        'normalization':'same fixed native 5%-span denominator for all cases; only onset boundaries move',
        'disturbance':'per-profile vector times common ramp, held to end; plant effective torque u_applied-d_true',
        'source_sha256':{f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    geom=Geometry();rows=[]
    for scenario,profile in enumerate(profiles):
        vector=np.asarray(profile['vector_nm'],dtype=float)
        assert vector.shape==(6,) and np.isfinite(vector).all()
        onset=float(profile['onset']);ramp=float(profile['ramp'])
        assert 0<=onset<8 and ramp>0 and onset+ramp<8
        def disturbance(step,device,dtype):
            factor=float(np.clip((step*DT-onset+DT)/ramp,0,1))
            return torch.as_tensor(vector*factor,device=device,dtype=dtype)
        for seed in seeds:
            for name in names:
                holder=[]
                def factory(*args,**kw):
                    tracker=CircleTracker(*args,**kw);holder.append(tracker);return tracker
                print('MULTI_RUN',profile['name'],'seed',seed,name,vector.tolist(),flush=True)
                data=run_mppi(ref,initial_q=q0,has_disturbance=bool(np.any(vector)),use_dob=True,
                    disturbance_function=disturbance,dob_compensation_mode='output',
                    require_cuda=True,seed=seed,temperature=1.,temporal_noise_rho=.8,
                    compile_rollout_loop=True,joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.,
                    tracker_class=factory,tracker_options={'margin_config':asdict(CASES[name]),
                        'torque_scale':1.,'collision_weight':50000.,'collision_clearance':.005,
                        'collision_backend':'triton','sensor_noise':{'position_std_rad':.001,
                            'velocity_std_rad_s':.01,'seed':80000+seed},
                        'dob_acceleration_lpf_cutoff_hz':10.,'dob_cutoff_hz':8.})
                data.update(holder[0].export_margin_data());data.update(holder[0].export_sensor_data())
                data['circle_reference']=ref
                q=data['q'];margin=data['kroc_margin'];risk=position_metrics(q)
                distance=np.minimum(q[1:]-np.asarray(Q_MIN),np.asarray(Q_MAX)-q[1:])
                soft=(q[:-1]<np.asarray(Q_MIN)+margin[:,0,0])|(q[:-1]>np.asarray(Q_MAX)-margin[:,0,1])
                sat=np.isclose(np.abs(data['u_applied']),np.asarray(TAU_MAX),atol=1e-5,rtol=0)
                clamp_error=float(np.max(np.abs(data['u_applied']-np.clip(
                    data['u_mppi']+data['d_hat_used'],-np.asarray(TAU_MAX),np.asarray(TAU_MAX)))))
                assert clamp_error<1e-4
                m=measure(data,j,1.)
                m.update(position_violation_steps=risk['violating_steps'],
                    position_violation_max_rad=risk['maximum_violation_rad'],
                    position_violation_integral_rad_s=risk['cumulative_violation_rad_s'],
                    position_violation_step_fraction=risk['violation_step_fraction'],
                    position_minimum_distance_rad=float(distance.min()),
                    initial_common_box_min_slack_rad=float(slack.min()),
                    requested_clamp_audit_max=clamp_error,
                    tracking_peak_mm=float(data['err_mm'].max()),
                    position_soft_active_fraction=float(soft.any(axis=1).mean()),
                    position_soft_active_fraction_per_joint=soft.mean(axis=0).tolist(),
                    saturation_duration_per_joint_s=(sat.sum(axis=0)*DT).tolist())
                m.update(geom.trajectory(q))
                row={'scenario':scenario,'profile':profile['name'],'candidate':geometry['candidate'],
                    'fraction':float(q0[0]/Q_MAX[0]),'radius':geometry['radius'],
                    'pose':geometry['metadata']['pose'],'amplitude':float(np.max(np.abs(vector))),
                    'onset':onset,'ramp':ramp,'sign':1,'shape':'circle','period_seconds':8.,
                    'duration_seconds':8.,'disturbance_vector_nm':vector.tolist(),
                    'seed':seed,'case':name,'metrics':m}
                rows.append(row)
                np.savez_compressed(output/f's{scenario}_seed{seed}_{name}.npz',
                    **{k:v for k,v in data.items() if not k.startswith('_')})
                (output/'metrics.json').write_text(json.dumps(rows,indent=2))
                print('MULTI_RESULT',json.dumps(row),flush=True)
                del data,holder;gc.collect();torch.cuda.empty_cache()
    return rows
