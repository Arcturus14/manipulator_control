"""Post-DOB circular tracking: 2.5% fixed floor, directional-only margins."""
import argparse
from dataclasses import asdict
import gc
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT, Q_MAX, Q_MIN, TAU_MAX, DH_PARAMS
from experiment_runner import run_mppi
from kinematics.ur5_kinematics import fk_joints_np
from kroc.search_postdob_floor25 import CASES, NAMES
from kroc.benchmark_postdob_small_directional_margin import measure, stat
from kroc.state_margin import NoiseAwareKrocTracker


def local_ik_reference(targets, initial):
    """Continuous position IK in physical bounds, avoiding legacy +/-pi clipping."""
    q=np.array(initial,dtype=float).copy(); result=[]
    for target in targets:
        for _ in range(50):
            tr=np.eye(4);origins=[];axes=[]
            for i,(aa,dd,alpha,offset) in enumerate(DH_PARAMS):
                origins.append(tr[:3,3].copy());axes.append(tr[:3,2].copy())
                ct,st=np.cos(q[i]+offset),np.sin(q[i]+offset);ca,sa=np.cos(alpha),np.sin(alpha)
                tr=tr@np.array([[ct,-st*ca,st*sa,aa*ct],[st,ct*ca,-ct*sa,aa*st],[0,sa,ca,dd],[0,0,0,1]])
            err=target-tr[:3,3]
            if np.linalg.norm(err)<2e-7:break
            jac=np.stack([np.cross(ax,tr[:3,3]-o) for ax,o in zip(axes,origins)],axis=1)
            delta=jac.T@np.linalg.solve(jac@jac.T+1e-5*np.eye(3),err)
            q=np.clip(q+.8*np.clip(delta,-.05,.05),Q_MIN,Q_MAX)
        result.append(q.copy())
    return np.array(result)


class CircleTracker(NoiseAwareKrocTracker):
    def __init__(self,*args,collision_weight=0.,collision_clearance=.005,
                 collision_backend='triton',**kwargs):
        super().__init__(*args,**kwargs)
        self.collision_cost=None
        if collision_weight:
            from kroc.self_collision_cost import PolytopeCollisionCost
            self.collision_geometry_cost=PolytopeCollisionCost(
                self.device,self.dtype,collision_weight,collision_clearance,
                backend=collision_backend)
            self.collision_cost=torch.compile(self.collision_geometry_cost.__call__,
                options={'triton.cudagraphs':False,'triton.cudagraph_trees':False})

    def _running_cost(self,state,action,rollout_step):
        cost=super()._running_cost(state,action,rollout_step)
        if self.collision_cost is not None:cost=cost+self.collision_cost(state)
        return cost

    def _terminal_cost(self,states,actions):
        cost=super()._terminal_cost(states,actions)
        if self.collision_cost is not None:
            cost=cost+self.collision_cost(states[...,-1,:].reshape(-1,12)).reshape(states.shape[:2])
        return cost

    def _feedforward_torque(self,q0):
        count=len(self.waypoints)-self.horizon-2
        targets=self.waypoints[:count].detach().cpu().numpy()
        joints=local_ik_reference(targets,q0.detach().cpu().numpy())
        vel=np.gradient(joints,DT,axis=0,edge_order=2)
        acc=np.gradient(vel,DT,axis=0,edge_order=2)
        qr=torch.as_tensor(joints,device=self.device,dtype=self.dtype)
        vr=torch.as_tensor(vel,device=self.device,dtype=self.dtype)
        ar=torch.as_tensor(acc,device=self.device,dtype=self.dtype)
        torque=self.dynamics.inverse_dynamics(qr,vr,ar)+self.dynamics.damping*vr
        self.joint_reference=torch.cat((qr,qr[-1:].expand(self.horizon+2,-1)),dim=0)
        return torch.cat((torque,torque[-1:].expand(self.horizon+2,-1)),dim=0).clamp(self.dynamics.u_min,self.dynamics.u_max)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fractions',nargs='+',type=float,default=[.92,.94])
    p.add_argument('--radii',nargs='+',type=float,default=[.04,.08])
    p.add_argument('--poses',nargs='+',type=int,default=[0])
    p.add_argument('--amplitudes',nargs='+',type=float,default=[0])
    p.add_argument('--onsets',nargs='+',type=float,default=[2.])
    p.add_argument('--ramps',nargs='+',type=float,default=[.4])
    p.add_argument('--signs',nargs='+',type=int,default=[1])
    p.add_argument('--joint',type=int,default=0)
    p.add_argument('--scale',type=float,default=1.)
    p.add_argument('--period',type=float,default=4.)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--seeds',nargs='+',type=int,default=[7])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--candidate-file',type=Path)
    p.add_argument('--outward',action='store_true')
    p.add_argument('--collision-check',action='store_true')
    p.add_argument('--base-fraction',type=float)
    p.add_argument('--collision-weight',type=float,default=0.)
    p.add_argument('--collision-clearance',type=float,default=.005)
    p.add_argument('--collision-backend',choices=['reference','fused','triton'],default='triton')
    p.add_argument('--compile-rollout-loop',action='store_true')
    p.add_argument('--require-common-start',action='store_true',
                   help='Require q0 strictly inside the maximum (5%% span) tightened position box.')
    a=p.parse_args(); cuda=cuda_preflight()
    if a.collision_weight<0 or a.collision_clearance<0:p.error('collision settings must be nonnegative')
    a.output.mkdir(parents=True,exist_ok=False)
    manifest=dict(arguments={**vars(a),'output':str(a.output),'candidate_file':str(a.candidate_file)},cuda=cuda,
        headroom_enabled=False,margin_basis='full physical span per side',
        cases={NAMES[k]:asdict(c) for k,c in CASES.items()},
        sensor_noise={'q_std_rad':.001,'dq_std_rad_s':.01},
        dob={'acceleration_lpf_hz':10.,'q_hz':8.},
        collision_cost={'enabled':bool(a.collision_weight),'method':'official mesh enclosing 26-DOP; face-normal axes',
                        'backend':a.collision_backend,
                        'weight':a.collision_weight,'clearance_m':a.collision_clearance,
                        'distance_normalization_m':.05,'floor_z_m':0.,
                        'excluded_pairs':'directly adjacent only','safety_guarantee':False},
        reference='Cartesian xy circle, exact initial FK, no orientation or joint reference objective',
        feedforward_ik='continuous local IK within physical joint bounds; experiment-only override',
        disturbance='ramp then held until episode ends; no release')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    candidates=json.loads(a.candidate_file.read_text())['selected'] if a.candidate_file else None
    if a.collision_check:
        from kroc.collision_geometry import Geometry
        geom=Geometry()
    rows=[]; scenarios=list(itertools.product(a.fractions,a.radii,a.poses,a.amplitudes,a.onsets,a.ramps,a.signs))
    for idx,(frac,radius,pose,amp,onset,ramp,sign) in enumerate(scenarios):
        q0=np.array([[0,-1.2,1.2,-1.5,-1.57,0],
                     [0,-.5,.6,-1.5,-1.57,0],
                     [0,-1.7,.7,-.8,-2.,0],
                     [.94*Q_MAX[0],-1.2,1.2,-1.5,-1.57,0]][pose if candidates is None else 0],dtype=float)
        if candidates is not None:q0=np.array(candidates[pose]['q'],dtype=float)
        if a.base_fraction is not None:q0[0]=a.base_fraction*Q_MAX[0]
        j=a.joint;q0[j]=frac*(Q_MAX[j] if sign>0 else Q_MIN[j])
        span=np.asarray(Q_MAX)-np.asarray(Q_MIN)
        common_slack=np.minimum(q0-(np.asarray(Q_MIN)+.05*span),
                                np.asarray(Q_MAX)-.05*span-q0)
        if a.require_common_start and np.min(common_slack)<=0:
            raise ValueError(f'Initial state outside strict common 5% tightened box: {common_slack}')
        start=fk_joints_np(q0)[-1];n=round(a.seconds/DT)
        theta=np.arange(n)*DT*2*np.pi/a.period
        # Place circle radially inward from initial point; tangent is azimuthal.
        radial=np.array([*start[:2],0.]);radial/=np.linalg.norm(radial)
        tangent=np.array([-radial[1],radial[0],0.])
        radial_offset=(1-np.cos(theta)) if a.outward else (np.cos(theta)-1)
        ref=start+radius*(radial_offset[:,None]*radial+sign*np.sin(theta)[:,None]*tangent)
        def disturbance(step,device,dtype):
            d=torch.zeros(6,device=device,dtype=dtype)
            d[j]=-sign*amp*np.clip((step*DT-onset+DT)/ramp,0,1)
            return d
        for seed in a.seeds:
            for key,cfg in CASES.items():
                name=NAMES[key];holder=[]
                def factory(*x,**kw):
                    tracker=CircleTracker(*x,**kw);holder.append(tracker);return tracker
                print(f'SCENARIO {idx+1}/{len(scenarios)} fraction={frac} R={radius} pose={pose} d={amp} onset={onset} ramp={ramp} sign={sign} seed={seed} {name}',flush=True)
                data=run_mppi(ref,initial_q=q0,has_disturbance=bool(amp),use_dob=True,
                    dob_compensation_mode='output',disturbance_function=disturbance,
                    require_cuda=True,seed=seed,temperature=1.,temporal_noise_rho=.8,
                    compile_rollout_loop=a.compile_rollout_loop,
                    joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.,
                    tracker_class=factory,tracker_options={'margin_config':asdict(cfg),'torque_scale':a.scale,
                    'collision_weight':a.collision_weight,'collision_clearance':a.collision_clearance,
                    'collision_backend':a.collision_backend,
                    'sensor_noise':{'position_std_rad':.001,'velocity_std_rad_s':.01,'seed':80000+seed},
                    'dob_acceleration_lpf_cutoff_hz':10.,'dob_cutoff_hz':8.})
                data.update(holder[0].export_margin_data());data.update(holder[0].export_sensor_data())
                data['circle_reference']=ref
                m=measure(data,j,a.scale);q=data['q'][1:];dq=data['qd'][1:]
                distance=np.minimum(q-np.asarray(Q_MIN),np.asarray(Q_MAX)-q)
                risk=(Q_MAX[j]-q[:,j]) if sign>0 else (q[:,j]-Q_MIN[j])
                margin=data['kroc_margin'][:,0,1 if sign>0 else 0,j]/(Q_MAX[j]-Q_MIN[j])
                err=data['err_mm']; t=np.arange(n)*DT
                end=t>=max(a.seconds-2.,onset+ramp+1.)
                true=np.asarray(data['d_true'])[1:]
                m.update(position_violation_steps=int((distance<0).any(1).sum()),
                    initial_common_box_min_slack_rad=float(common_slack.min()),
                    position_violation_max_rad=float((-distance).clip(0).max()),
                    risk_distance_min_rad=float(risk.min()),position_minimum_distance_rad=float(distance.min()),
                    risk_violation_steps=int((risk<0).sum()),risk_violation_max_rad=float((-risk).clip(0).max()),
                    margin_max_fraction=float(margin.max()),margin_mean_fraction=float(margin.mean()),
                    tracking_late_rmse_mm=float(np.sqrt(np.mean(err[end]**2))) if end.any() else None,
                    peak_speed_rad_s=float(np.abs(dq).max()),
                    risk_joint_final_rad=float(q[-1,j]),risk_joint_final_speed_rad_s=float(dq[-1,j]),
                    dob_late_rmse_nm=float(np.sqrt(np.mean((data['d_hat_used'][end,j]-true[end,j])**2))) if end.any() else None,
                    requested_clamp_audit_max=float(np.abs(data['u_applied']-np.clip(data['u_mppi']+data['d_hat_used'],-np.asarray(TAU_MAX)*a.scale,np.asarray(TAU_MAX)*a.scale)).max()))
                m['ik_max_step_rad']=float(np.abs(np.diff(data['q_reference'],axis=0)).max())
                if a.collision_check:
                    check=geom.trajectory(data['q']);m.update(check)
                    violations=np.flatnonzero((distance<0).any(1))
                    pre=geom.trajectory(data['q'][:violations[0]+2]) if len(violations) else check
                    m['pre_state_failure_self_collision_states']=pre['self_collision_states']
                    m['pre_state_failure_floor_collision_states']=pre['floor_collision_states']
                row=dict(scenario=idx,fraction=frac,radius=radius,pose=pose,amplitude=amp,onset=onset,ramp=ramp,sign=sign,seed=seed,case=name,metrics=m)
                rows.append(row); print(json.dumps(row),flush=True)
                np.savez_compressed(a.output/f's{idx}_seed{seed}_{name}.npz',**{k:v for k,v in data.items() if not k.startswith('_')})
                (a.output/'metrics.json').write_text(json.dumps(rows,indent=2))
                del data,holder;gc.collect();torch.cuda.empty_cache()
    aggregate={str(idx):{name:{k:stat([r['metrics'][k] for r in rows if r['scenario']==idx and r['case']==name])
                for k in rows[0]['metrics'] if all(r['metrics'][k] is not None for r in rows if r['scenario']==idx and r['case']==name)} for name in NAMES.values()} for idx in range(len(scenarios))}
    (a.output/'aggregate.json').write_text(json.dumps(aggregate,indent=2))


if __name__=='__main__':main()
