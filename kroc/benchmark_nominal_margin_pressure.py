"""CUDA nominal tracking only: geometry changes vs larger static margins.

No disturbance, no orientation/joint tracking term, no headroom, no default
controller changes. Keep the existing cost normalization unchanged.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import torch
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT, Q_MIN, Q_MAX, TAU_MAX
from experiment_runner import run_mppi
from kinematics.ur5_kinematics import fk_joints_np
from kroc.benchmark_noise_aware_state_margin import ACCELERATION_DEADBANDS
from kroc.benchmark_postdob_small_directional_margin import measure
from kroc.collision_geometry import Geometry
from kroc.search_circle_floor25 import CircleTracker
from kroc.state_margin import MarginConfig
from kroc.screen_nominal_margin_pressure import circle


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--candidates',type=int,nargs='+',required=True)
    p.add_argument('--static-fractions',type=float,nargs='+',default=[.025,.05,.075,.10])
    p.add_argument('--seeds',type=int,nargs='+',default=[207])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--period',type=float,default=4.)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--amplitudes',type=float,nargs='+',default=[0.])
    p.add_argument('--onset',type=float,default=.5)
    p.add_argument('--ramp',type=float,default=.4)
    p.add_argument('--disturbance-joint',type=int,default=0)
    p.add_argument('--use-banked-reference',action='store_true')
    p.add_argument('--cycles',type=int,default=1)
    p.add_argument('--dynamic-max',type=float,choices=[.05,.075],default=.05)
    a=p.parse_args();cuda=cuda_preflight();a.output.mkdir(parents=True,exist_ok=False)
    if a.ramp<=0 or a.onset<0 or min(a.amplitudes)<0 or a.disturbance_joint not in range(6):
        p.error('Require nonnegative disturbance amplitudes/onset, positive ramp, joint 0..5')
    candidates=json.loads((a.screen/'screen.json').read_text())['selected']
    names={.025:'static25',.035:'static35',.05:'static5',.075:'static75',.10:'static10'}
    configs={names[f]:MarginConfig(mode='fixed',min_fraction=.025,fixed_fraction=f,
                                 max_fraction=max(.05,f),penalty_kind='hinge') for f in a.static_fractions}
    dynamic_name='directional25_5' if a.dynamic_max==.05 else 'directional25_75'
    configs[dynamic_name]=MarginConfig(mode='directional',min_fraction=.025,fixed_fraction=.025,
        max_fraction=a.dynamic_max,penalty_kind='hinge',reaction_time_s=.10,
        acceleration_deadband_rad_s2=ACCELERATION_DEADBANDS['nominal_p99'],beta_up=1.,beta_down=.10)
    configs={n:configs[n] for n in ['static25',dynamic_name,'static5','static75','static10','static35'] if n in configs}
    common=max(a.static_fractions+[a.dynamic_max]);span=np.asarray(Q_MAX)-np.asarray(Q_MIN)
    manifest={'arguments':{**vars(a),'screen':str(a.screen),'output':str(a.output),'joint':a.disturbance_joint,'scale':1.},
        'cases':{n:asdict(c) for n,c in configs.items()},'cuda':cuda,'headroom_enabled':False,
        'sensor_noise':{'q_std_rad':.001,'dq_std_rad_s':.01},'dob':{'acceleration_lpf_hz':10.,'q_hz':8.},
        'collision_cost':{'enabled':True,'backend':'triton','weight':50000.,'clearance_m':.005},
        'reference':'position-only banked trajectory; no orientation/joint objective' if a.use_banked_reference else 'position-only Cartesian circle; no orientation/joint objective',
        'external_disturbance':bool(max(a.amplitudes)),'common_start_fraction':common,
        'disturbance':'negative torque-coordinate ramp then held to episode end; plant effective torque u_applied-d_true',
        'normalization':'existing fixed native 5% full-span denominator, unchanged across cases; only soft boundaries move',
        'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in map(Path,
            ['kroc/state_margin.py','kroc/search_circle_floor25.py','controllers/mppi_dob_controller.py','config.py'])}}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    geom=Geometry();rows=[]
    for scenario,(idx,amplitude) in enumerate(itertools.product(a.candidates,a.amplitudes)):
        c=next(c for c in candidates if c['candidate']==idx)
        with np.load(a.screen/f'candidate_{idx}.npz') as z:
            q0=z['initial_q'].copy()
            bank_ref=z['reference'].copy()
        if a.use_banked_reference:
            assert a.cycles>=1
            ref=np.tile(bank_ref,(a.cycles,1));period_seconds=len(bank_ref)*DT
        else:
            ref,_=circle(q0,c['ratio'],c['plane'],a.period,a.seconds);period_seconds=a.period
        slack=np.minimum(q0-np.asarray(Q_MIN)-common*span,np.asarray(Q_MAX)-common*span-q0)
        assert slack.min()>0
        def disturbance(step,device,dtype):
            value=amplitude*np.clip((step*DT-a.onset+DT)/a.ramp,0,1)
            d=torch.zeros(6,device=device,dtype=dtype);d[a.disturbance_joint]=-float(value)
            return d
        for seed in a.seeds:
            for name,cfg in configs.items():
                holder=[]
                def factory(*args,**kw):
                    tracker=CircleTracker(*args,**kw);holder.append(tracker);return tracker
                print('RUN',idx,name,seed,'disturbance',amplitude,flush=True)
                data=run_mppi(ref,initial_q=q0,has_disturbance=bool(amplitude),use_dob=True,
                    disturbance_function=disturbance,
                    dob_compensation_mode='output',require_cuda=True,seed=seed,temperature=1.,temporal_noise_rho=.8,
                    compile_rollout_loop=True,joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.,
                    tracker_class=factory,tracker_options={'margin_config':asdict(cfg),'torque_scale':1.,
                        'collision_weight':50000.,'collision_clearance':.005,'collision_backend':'triton',
                        'sensor_noise':{'position_std_rad':.001,'velocity_std_rad_s':.01,'seed':80000+seed},
                        'dob_acceleration_lpf_cutoff_hz':10.,'dob_cutoff_hz':8.})
                data.update(holder[0].export_margin_data());data.update(holder[0].export_sensor_data())
                data['circle_reference']=ref
                m=measure(data,a.disturbance_joint,1.);q=data['q'];margin=data['kroc_margin']
                distance=np.minimum(q[1:]-np.asarray(Q_MIN),np.asarray(Q_MAX)-q[1:])
                qsoft=(q[:-1]<np.asarray(Q_MIN)+margin[:,0,0])|(q[:-1]>np.asarray(Q_MAX)-margin[:,0,1])
                vsoft=(data['qd'][:-1]<-np.pi+margin[:,1,0])|(data['qd'][:-1]>np.pi-margin[:,1,1])
                err=data['err_mm'];late=np.arange(len(err))*DT>=4
                m.update(position_violation_steps=int((distance<0).any(1).sum()),
                    position_violation_max_rad=float(np.maximum(-distance,0).max()),
                    position_minimum_distance_rad=float(distance.min()),
                    tracking_late_rmse_mm=float(np.sqrt(np.mean(err[late]**2))),
                    position_soft_active_fraction=float(qsoft.any(1).mean()),
                    velocity_soft_active_fraction=float(vsoft.any(1).mean()),
                    initial_common_box_min_slack_rad=float(slack.min()),
                    reference_ik_max_error_mm=float(max(np.linalg.norm(fk_joints_np(qr)[-1]-target)*1000
                        for qr,target in zip(data['q_reference'],ref))),
                    requested_clamp_audit_max=float(np.abs(data['u_applied']-np.clip(
                        data['u_mppi']+data['d_hat_used'],-np.asarray(TAU_MAX),np.asarray(TAU_MAX))).max()))
                m.update(geom.trajectory(q))
                row={'scenario':scenario,'candidate':idx,'fraction':q0[0]/Q_MAX[0],'radius':c['radius'],
                     'pose':c['pose'],'plane':c['plane'],'amplitude':amplitude,'onset':a.onset,'ramp':a.ramp,
                     'sign':1,'seed':seed,'case':name,'metrics':m,
                     'shape':c.get('shape','circle'),'period_seconds':period_seconds,'duration_seconds':len(ref)*DT}
                rows.append(row);np.savez_compressed(a.output/f's{scenario}_seed{seed}_{name}.npz',
                    **{k:v for k,v in data.items() if not k.startswith('_')})
                (a.output/'metrics.json').write_text(json.dumps(rows,indent=2))
                print(json.dumps(row),flush=True)
                del data,holder;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
