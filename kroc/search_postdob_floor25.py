"""Directional-only search with a fixed 2.5% span floor; no headroom."""
import argparse
from dataclasses import asdict, replace
import gc
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT, Q_MAX, Q_MIN, TAU_MAX
from experiment_runner import run_mppi
from kinematics.ur5_kinematics import fk_joints_np
from kroc.benchmark_postdob_small_directional_margin import CASES, measure, stat
from kroc.state_margin import MarginConfig, NoiseAwareKrocTracker

CASES['fixed_1pct'] = MarginConfig(mode='fixed', min_fraction=.025, fixed_fraction=.025, max_fraction=.05, penalty_kind='hinge')
CASES['dynamic_directional_1_to_5pct'] = replace(CASES['dynamic_directional_1_to_5pct'], min_fraction=.025, fixed_fraction=.025)
NAMES = dict(zip(CASES, ['static25', 'directional25_5', 'static5']))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fractions', nargs='+', type=float, default=[.90, .94])
    p.add_argument('--amplitudes', nargs='+', type=float, default=[10., 11.])
    p.add_argument('--durations', nargs='+', type=int, default=[30, 50, 80])
    p.add_argument('--signs', nargs='+', type=int, default=[1])
    p.add_argument('--poses', nargs='+', type=int, default=[0])
    p.add_argument('--joint', type=int, default=5)
    p.add_argument('--scale', type=float, default=.35)
    p.add_argument('--ramp', type=int, default=20)
    p.add_argument('--steps', type=int, default=150)
    p.add_argument('--seeds', nargs='+', type=int, default=[7])
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cuda = cuda_preflight()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(arguments={**vars(args), 'output': str(args.output)}, cuda=cuda,
                    headroom_enabled=False, margin_basis='fraction of full span per side',
                    cases={NAMES[k]: asdict(v) for k,v in CASES.items()},
                    sensor_noise={'q_std_rad': .001, 'dq_std_rad_s': .01},
                    dob={'acceleration_lpf_hz':10., 'q_hz':8.},
                    onset_step=10, reference='constant Cartesian position, no orientation cost')
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    scenarios = list(itertools.product(args.fractions,args.amplitudes,args.durations,args.signs,args.poses))
    rows=[]
    for idx,(fraction,amp,duration,sign,pose) in enumerate(scenarios):
        q0=np.array([[np.pi,-1.2,1.2,-1.5,-1.57,0],
                     [2.4,-.7,1.7,-2.,-.8,0],
                     [3.5,-1.7,.7,-.8,-2.,0]][pose],dtype=float)
        j=args.joint
        q0[j]=fraction*(Q_MAX[j] if sign==1 else Q_MIN[j])
        ref=np.repeat(fk_joints_np(q0)[-1][None,:],args.steps,axis=0)
        def disturbance(step,device,dtype):
            elapsed=step-10
            factor=0 if elapsed<0 or elapsed>=duration else min((elapsed+1)/args.ramp,1)
            d=torch.zeros(6,device=device,dtype=dtype)
            d[j]=-sign*amp*factor
            return d
        pair={}
        for seed in args.seeds:
            for key,cfg in CASES.items():
                name=NAMES[key]
                print(f'SCENARIO {idx+1}/{len(scenarios)} q={fraction} amp={amp} duration={duration} sign={sign} pose={pose} seed={seed} {name}',flush=True)
                holder=[]
                def factory(*a,**kw):
                    t=NoiseAwareKrocTracker(*a,**kw);holder.append(t);return t
                data=run_mppi(ref, initial_q=q0, has_disturbance=True,use_dob=True,
                    dob_compensation_mode='output',disturbance_function=disturbance,
                    require_cuda=True,seed=seed,temperature=1.,temporal_noise_rho=.8,
                    joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.,
                    tracker_class=factory,tracker_options={'margin_config':asdict(cfg),'torque_scale':args.scale,
                    'sensor_noise':{'position_std_rad':.001,'velocity_std_rad_s':.01,'seed':80000+seed},
                    'dob_acceleration_lpf_cutoff_hz':10.,'dob_cutoff_hz':8.})
                data.update(holder[0].export_margin_data()); data.update(holder[0].export_sensor_data())
                m=measure(data,j,args.scale)
                q=data['q'][1:,j]
                violation=np.maximum(q-Q_MAX[j],Q_MIN[j]-q).clip(0)
                dist=(Q_MAX[j]-q) if sign==1 else (q-Q_MIN[j])
                margin=data['kroc_margin'][:,0,1 if sign==1 else 0,j]/(Q_MAX[j]-Q_MIN[j])
                m.update(hard_violation_steps=int((violation>0).sum()),hard_violation_max_rad=float(violation.max()),
                         hard_violation_integral_rad_s=float(violation.sum()*DT),risk_distance_min_rad=float(dist.min()),
                         margin_max_fraction=float(margin.max()),margin_mean_fraction=float(margin.mean()),
                         all_joint_violation_steps=int(np.asarray(data['joint_position_limit_violated']).sum()),
                         actuator_clamp_audit_max=float(np.abs(data['u_applied']-np.clip(data['u_mppi']+data['d_hat_used'],-np.asarray(TAU_MAX)*args.scale,np.asarray(TAU_MAX)*args.scale)).max()))
                row=dict(scenario=idx,fraction=fraction,amplitude=amp,duration=duration,sign=sign,pose=pose,seed=seed,case=name,metrics=m)
                rows.append(row);pair[(seed,name)]=m
                # Retain all trajectories for audit and candidate follow-up.
                np.savez_compressed(args.output/f's{idx}_seed{seed}_{name}.npz',**{k:v for k,v in data.items() if not k.startswith('_')})
                (args.output/'metrics.json').write_text(json.dumps(rows,indent=2))
                print(json.dumps(row),flush=True)
                del data,holder;gc.collect();torch.cuda.empty_cache()
            a,b,c=(pair[(seed,n)] for n in NAMES.values())
            if a['hard_violation_steps']>0 and b['all_joint_violation_steps']==0 and c['all_joint_violation_steps']==0 and a['test_joint_saturation_steps']>0:
                print('CANDIDATE',json.dumps(rows[-3:]),flush=True)
    aggregate={name:{f:stat([r['metrics'][f] for r in rows if r['case']==name]) for f in rows[0]['metrics']} for name in NAMES.values()}
    (args.output/'aggregate.json').write_text(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    main()
