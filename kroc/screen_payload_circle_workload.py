"""Offline circular-workload screen. No controller runs or tuning here.

Exact payload rigid dynamics plus reflected armature, regularization, damping.
Continuous local IK and finite-difference demands are feasibility estimates,
not MPPI results. Preserve all accepted/rejected geometry and timing cases.
"""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from config import Q_MIN,Q_MAX,DQ_MAX
from dynamics.ur5_payload import CubePayload,PayloadUR5Dynamics
from kinematics.payload_tcp import PayloadTCP
from kroc.collision_geometry import Geometry
from kroc.payload_geometry import audit_cube_trajectory


ROOT=Path('kroc/results/payload_circle_workload_20261121')
BASE=Path('configs/payload_cube_tcp_stabilized_small_cube_n0.yaml')

def write(p,obj):p.write_text(json.dumps(obj,indent=2,allow_nan=False))

def main():
    root=ROOT/'offline';root.mkdir(parents=True,exist_ok=False)
    cfg=yaml.safe_load(BASE.read_text());payload=CubePayload(**cfg['payload'])
    tcp=PayloadTCP(payload,'cpu',torch.float64);dyn=PayloadUR5Dynamics('cpu',torch.float64,payload)
    geo=Geometry();qbase=np.array(cfg['scene']['initial_q']);start=tcp.position_and_jacobian_np(qbase)[0]
    center=start-[0,.16,0];cy=float(center[1]);z=float(center[2])
    paths=[('yz_far16','circle_yz',center,.16),
           ('yz_inward16','circle_yz',center+[-.04,0,.04],.16),
           ('yz_far12','circle_yz',center,.12),
           ('xz16','circle_xz',np.array([.70,cy,.30]),.16),
           ('xz12','circle_xz',np.array([.75,cy,.30]),.12),
           ('xy16','circle_xy',np.array([.70,cy,.30]),.16),
           ('xy12','circle_xy',np.array([.75,cy,.30]),.12),
           ('yz20','circle_yz',np.array([.83,cy,.30]),.20)]
    periods=[20.,10.,8.,6.,4.,3.,2.,1.5]
    write(root/'plan.json',dict(base=str(BASE),paths=[dict(name=n,shape=s,center_m=c.tolist(),radius_m=r) for n,s,c,r in paths],periods_s=periods,
        scope='Offline only. N0 weights/noise/filters, mass, limits unchanged. No disturbance/DOB. No reciprocal paths.',
        target='Repeated negative J2 peaks130--145Nm with no nominal saturation; reject near-limit velocity and failed geometry. Validate selected candidates on CUDA.',
        methodology='Two continuous IK cycles at128 samples/cycle. Score second cycle, include armature and regularization. Dynamics scale inverse-square with period, damping inverse-linear. Coarse prescreen is not proof of feasibility.'))
    records=[];summary=[]
    for name,shape,c,r in paths:
        row=dict(name=name,shape=shape,center_m=c.tolist(),radius_m=r)
        try:
            a,b={'circle_xy':(0,1),'circle_xz':(0,2),'circle_yz':(1,2)}[shape]
            phase=2*np.pi*np.arange(257)/128
            ref=np.tile(c,(257,1));ref[:,a]+=r*np.cos(phase);ref[:,b]+=r*np.sin(phase)
            q0=tcp.reference_np(ref[:1],qbase)[0];q=tcp.reference_np(ref,q0)
            robot=geo.trajectory(q,stop_on_collision=True);cube=audit_cube_trajectory(q,payload,subdivisions=1,dt=1/128)
            sigma=np.array([np.linalg.svd(tcp.position_and_jacobian_np(qi)[1],compute_uv=False)[-1] for qi in q])
            dq=np.gradient(q,1/128,axis=0,edge_order=2);ddq=np.gradient(dq,1/128,axis=0,edge_order=2)
            qt,vt,at=[torch.as_tensor(x,dtype=torch.float64) for x in [q,dq,ddq]]
            with torch.no_grad():
                gravity=dyn.gravity_compensation(qt)
                dynamic=dyn.inverse_dynamics(qt,vt,at)-gravity+(dyn.armature+dyn.mass_regularization)*at
                damping=dyn.damping*vt
            gravity,dynamic,damping=[x.numpy() for x in [gravity,dynamic,damping]]
            geometric_ok=not(robot['self_collision_states'] or robot['floor_collision_states'] or cube['payload_collision_states'])
            geometric_ok=bool(geometric_ok and np.all(q>=np.array(Q_MIN)) and np.all(q<=np.array(Q_MAX)))
            row.update(initial_q=q0.tolist(),robot_mesh=robot,payload_mesh=cube,geometry_pass=geometric_ok,
                min_jacobian_sigma_m=float(sigma.min()),q3_range_rad=[float(q[:,2].min()),float(q[:,2].max())],
                cycle_joint_drift_rad=(q[-1]-q[128]).tolist())
            sl=slice(128,255)
            for T in periods:
                ff=gravity+dynamic/T**2+damping/T
                maxspeed=np.abs(dq[sl]/T).max(0);limits=np.array([150,150,150,28,28,28])
                item=dict(name=name,period_s=T,radius_m=r,shape=shape,geometry_pass=geometric_ok,
                    min_jacobian_sigma_m=float(sigma.min()),initial_q=q0.tolist(),center_m=c.tolist(),
                    ff_min_nm=ff[sl].min(0).tolist(),ff_max_nm=ff[sl].max(0).tolist(),
                    j2_negative_peak_nm=float(-ff[sl,1].min()),j2_fraction_below_minus120=float((ff[sl,1]<-120).mean()),
                    joint_speed_abs_max_rad_s=maxspeed.tolist(),velocity_pass=bool(maxspeed.max()<.9*DQ_MAX),
                    predicted_any_torque_excess=bool((np.abs(ff[sl])>limits).any()),
                    reference_torque_rate_rms_nm_s=np.sqrt(np.mean((np.gradient(ff[sl],T/128,axis=0))**2,axis=0)).tolist())
                summary.append(item)
            np.savez_compressed(root/f'{name}.npz',q=q,reference=ref,gravity=gravity,dynamic_unit_period=dynamic,damping_unit_period=damping)
        except ValueError as e:row.update(geometry_pass=False,error=str(e))
        records.append(row);write(root/'geometry.json',records);write(root/'demands.json',summary)
        print(json.dumps(row),flush=True)
    eligible=[r for r in summary if r['geometry_pass'] and r['velocity_pass'] and not r['predicted_any_torque_excess'] and r['min_jacobian_sigma_m']>=.03]
    eligible.sort(key=lambda r:abs(r['j2_negative_peak_nm']-135))
    write(root/'ranked_candidates.json',eligible)
    for row in eligible[:12]:print('CANDIDATE',row['name'],row['period_s'],row['j2_negative_peak_nm'],row['joint_speed_abs_max_rad_s'],flush=True)

if __name__=='__main__':main()
