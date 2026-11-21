"""Stage2 only after frozen-N0 circle screen failed its prospective gate."""
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
from kroc.payload_scene import rounded_rectangle
from kroc.screen_payload_circle_workload import BASE
from kroc.study_payload_torque_stability import write

ROOT=Path('kroc/results/payload_rectangle_workload_20261121')

def main():
    circle=json.loads(Path('kroc/results/payload_circle_workload_20261121/cuda/selection.json').read_text())
    if circle['target_met']:raise RuntimeError('Circle gate passed; do not advance to rectangle automatically')
    root=ROOT/'offline';root.mkdir(parents=True,exist_ok=False)
    base=yaml.safe_load(BASE.read_text());payload=CubePayload(**base['payload'])
    tcp=PayloadTCP(payload,'cpu',torch.float64);dyn=PayloadUR5Dynamics('cpu',torch.float64,payload);geo=Geometry()
    qb=np.array(base['scene']['initial_q']);start=tcp.position_and_jacobian_np(qb)[0];center=start-[0,.16,0]
    paths=[('yz16','yz',center,.16,.12),('yz20','yz',np.array([.83,center[1],.30]),.20,.12),
           ('xz12','xz',np.array([.75,center[1],.30]),.12,.10)]
    periods=[12.,10.,8.,6.,5.,4.,3.];corners=[.04,.02,.01]
    write(root/'plan.json',dict(stage='2: circles did not meet frozen N0 tracking/loading gate',
        paths=[dict(name=n,plane=s,center_m=c.tolist(),half_width_m=a,half_height_m=b) for n,s,c,a,b in paths],
        periods_s=periods,corner_radii_m=corners,start_ramp_s=1.,
        corner='Exact straight sides connected by quintic Bezier C2 turns; no infinite acceleration/sharp instantaneous turns.',
        target='130--145Nm physical inverse-dynamics negative J2 peaks, no predicted torque or velocity excess; corner traversal>=0.12s. Validate CUDA N0 without changing controller.',
        inverse_dynamics_note='Physical demand includes reflected armature+mass regularization; existing controller feedforward RNEA does not. No change to frozen controller in this study.'))
    rows=[];geometries=[]
    for name,plane,center,a,b in paths:
        for corner in corners:
            label=f'{name}_r{corner:g}';record=dict(name=label)
            try:
                N=1024;u=np.arange(2*N+1)/N;xy=rounded_rectangle(u,a,b,corner)
                ref=np.tile(center,(len(u),1));first,second={'xy':(0,1),'xz':(0,2),'yz':(1,2)}[plane]
                ref[:,first]+=xy[:,0];ref[:,second]+=xy[:,1]
                q0=tcp.reference_np(ref[:1],qb)[0];q=tcp.reference_np(ref,q0)
                # Mesh screen every fourth point; full control trajectories audited5ms later.
                robot=geo.trajectory(q[::4],stop_on_collision=True);cube=audit_cube_trajectory(q[::4],payload,subdivisions=1,dt=4/N)
                jacmin=min(np.linalg.svd(tcp.position_and_jacobian_np(qi)[1],compute_uv=False)[-1] for qi in q[::4])
                geometry=bool(not(robot['self_collision_states'] or robot['floor_collision_states'] or cube['payload_collision_states']) and np.all(q>=np.array(Q_MIN)) and np.all(q<=np.array(Q_MAX)))
                record.update(geometry_pass=geometry,robot_mesh=robot,payload_mesh=cube,min_jacobian_sigma_m=float(jacmin))
                v=np.gradient(q,1/N,axis=0,edge_order=2);acc=np.gradient(v,1/N,axis=0,edge_order=2)
                qt,vt,at=[torch.as_tensor(x,dtype=torch.float64) for x in [q,v,acc]]
                with torch.no_grad():
                    g=dyn.gravity_compensation(qt)
                    dynamic=dyn.inverse_dynamics(qt,vt,at)-g+(dyn.armature+dyn.mass_regularization)*at
                    damping=dyn.damping*vt
                g,dynamic,damping=[x.numpy() for x in [g,dynamic,damping]];sl=slice(N,2*N-2)
                L=4*(a+b-2*corner)+2*np.pi*corner
                for T in periods:
                    f=g+dynamic/T**2+damping/T;speed=abs(v[sl]/T).max(0)
                    cfg=copy.deepcopy(base);cfg['scene'].update(initial_q=q0.tolist(),trajectory='rounded_rectangle_'+plane,
                        circle_radius_m=a,rectangle_half_height_m=b,rectangle_corner_radius_m=corner,reference_start_ramp_s=1.,
                        period_s=T,duration_s=24.,use_dob=False,disturbance_amplitude_nm=0.)
                    cfg['output_root']=str(ROOT/'replay')
                    rows.append(dict(name=label,period_s=T,corner_radius_m=corner,geometry_pass=geometry,
                        min_jacobian_sigma_m=float(jacmin),ff_min_nm=f[sl].min(0).tolist(),ff_max_nm=f[sl].max(0).tolist(),
                        j2_negative_peak_nm=float(-f[sl,1].min()),j2_fraction_below_minus130=float((f[sl,1]<=-130).mean()),
                        predicted_any_torque_excess=bool((abs(f[sl])>np.array([150,150,150,28,28,28])).any()),
                        joint_speed_abs_max_rad_s=speed.tolist(),velocity_pass=bool(speed.max()<.9*DQ_MAX),
                        corner_traversal_s=float((np.pi*corner/2)/L*T),config=cfg))
                np.savez_compressed(root/f'{label}.npz',q=q,reference=ref,gravity=g,dynamic_unit_period=dynamic,damping_unit_period=damping)
            except ValueError as e:record.update(geometry_pass=False,error=str(e))
            geometries.append(record);write(root/'geometry.json',geometries);write(root/'demands.json',rows)
            print(json.dumps(record),flush=True)
    ok=[r for r in rows if r['geometry_pass'] and r['velocity_pass'] and not r['predicted_any_torque_excess'] and r['min_jacobian_sigma_m']>=.03 and r['corner_traversal_s']>=.12]
    ok.sort(key=lambda r:abs(r['j2_negative_peak_nm']-135))
    write(root/'ranked_candidates.json',ok)
    for r in ok[:12]:print('CANDIDATE',r['name'],r['period_s'],r['j2_negative_peak_nm'],r['corner_traversal_s'],flush=True)

if __name__=='__main__':main()
