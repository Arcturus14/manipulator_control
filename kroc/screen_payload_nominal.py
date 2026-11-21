"""CUDA-only N0 payload scene screen; no disturbance or controller retuning.

Prescreen geometric references/inverse-dynamics demand, then compare one
intermediate pose and three trajectory planes at the most extended pose.
Keep every prescreen and rollout outcome. This does not validate saturation.
"""
import copy
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from dynamics.ur5_payload import CubePayload,PayloadUR5Dynamics
from kinematics.payload_tcp import PayloadTCP
from kroc.payload_scene import make_reference
from kroc.payload_geometry import audit_cube_trajectory
from kroc.collision_geometry import Geometry
from kroc.run_payload_cube_tcp import run_payload
from kroc.view_payload_tcp import save_outputs,torque_metrics


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def safe(v):
    return (v['no_position_violation'] and v['no_velocity_violation'] and
        not v['robot_mesh']['self_collision_states'] and not v['robot_mesh']['floor_collision_states'] and
        not v['payload_mesh']['payload_collision_states'])


def main():
    cuda=cuda_preflight();base=yaml.safe_load(Path('configs/payload_cube_tcp.yaml').read_text())
    root=Path('kroc/results/payload_nominal_screen_20261121')/datetime.now().strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True)
    write(root/'plan.json',dict(cuda=cuda,seed=6807,heldout_seeds=[6817,6827],
        external_disturbance=False,DOB=False,payload_mass_kg=5.,torque_limits_nm=[150,150,150,28,28,28],
        selection='Among collision/limit-free runs with EE RMSE <=2mm, maximize mean absolute J2 requested torque. No N1/D0 performance selection.',
        scope='Nominal operating-point selection, not N0/N1/D0 research comparison.'))
    tcp=PayloadTCP(CubePayload(**base['payload']),'cpu',torch.float64)
    dynamics=PayloadUR5Dynamics('cuda',torch.float64,CubePayload(**base['payload']))
    geometry=Geometry();records=[];configs={}
    poses={'mid':(-.9,.7),'far':(-.6,.4),'extended':(-.5,.25)}
    for pose,(q2,q3) in poses.items():
        for shape in ['circle_xy','circle_xz','circle_yz']:
            name=f'{pose}_{shape}';cfg=copy.deepcopy(base);scene=cfg['scene']
            scene.update(trajectory=shape,use_dob=False,disturbance_amplitude_nm=0.)
            scene['initial_q'][1:3]=[q2,q3]
            configs[name]=cfg
            row=dict(name=name,q2=q2,q3=q3,shape=shape)
            try:
                dt=.12;reference=make_reference(tcp,scene['initial_q'],51,dt,scene)
                q=tcp.reference_np(reference,scene['initial_q'])
                dq=np.gradient(q,dt,axis=0,edge_order=2);ddq=np.gradient(dq,dt,axis=0,edge_order=2)
                args=[torch.tensor(a,device='cuda',dtype=torch.float64) for a in [q,dq,ddq]]
                ff=dynamics.inverse_dynamics(*args)+dynamics.damping*args[1]
                ff=ff.cpu().numpy()
                robot=geometry.trajectory(q);cube=audit_cube_trajectory(q,CubePayload(**cfg['payload']),subdivisions=1,dt=dt)
                row.update(tcp_start_m=reference[0].tolist(),radius_horizontal_m=float(np.linalg.norm(reference[0,:2])),
                    ff_peak_abs_nm=np.abs(ff).max(0).tolist(),ff_mean_abs_nm=np.abs(ff).mean(0).tolist(),
                    jacobian_min_singular_value_m=float(min(np.linalg.svd(tcp.position_and_jacobian_np(qi)[1],compute_uv=False)[-1] for qi in q)),
                    robot_mesh=robot,payload_mesh=cube,
                    prescreen_pass=not(robot['self_collision_states'] or robot['floor_collision_states'] or cube['payload_collision_states']))
            except ValueError as error:row.update(prescreen_pass=False,error=str(error))
            records.append(row);print(json.dumps(row),flush=True)
    write(root/'prescreen.json',records)
    # Fix the rollout set by geometry/pose, before observing tracking performance.
    viable=[p for p in poses if all(r.get('prescreen_pass',False) for r in records if r['name'].startswith(p+'_'))]
    if not viable:raise RuntimeError('No geometrically feasible three-plane payload pose')
    farthest=max(viable,key=lambda p:next(r['radius_horizontal_m'] for r in records if r['name']==p+'_circle_xy'))
    selected=['mid_circle_xy'] if 'mid' in viable and farthest!='mid' else []
    selected += [f'{farthest}_{shape}' for shape in ['circle_xy','circle_xz','circle_yz']]
    write(root/'rollout_plan.json',dict(candidates=selected,geometric_farthest_pose=farthest,all_prescreen_outcomes_retained=True))
    rows=[];results={}
    for name in selected:
        cfg=configs[name]
        z,v,path=run_payload(cfg,seed=6807,case='N0',output=root/name)
        m=torque_metrics(z);m.update(name=name,safe=safe(v),mean_abs_j2_nm=float(np.abs(z['u_requested'][:,1]).mean()),path=str(path))
        rows.append(m);results[name]=z
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        write(root/'rollout_metrics.json',rows)
    eligible=[r for r in rows if r['safe'] and r['rmse_err_mm']<=2.]
    if not eligible:raise RuntimeError('No candidate satisfies prespecified safety/RMSE criteria; inspect retained outcomes')
    best=max(eligible,key=lambda r:r['mean_abs_j2_nm']);cfg=configs[best['name']]
    # Additional independent seeds assess the fixed scene; no reselection.
    validation=[]
    for seed in [6817,6827]:
        z,v,path=run_payload(cfg,seed=seed,case='N0',output=root/f"{best['name']}_seed{seed}")
        m=torque_metrics(z);m.update(seed=seed,safe=safe(v));validation.append(m)
        save_outputs({'N0':z},path,torque=True,components=False,disturbance_error=False,csv=True)
    write(root/'validation.json',validation)
    recommendation=Path('configs/payload_cube_tcp_extended_n0.yaml')
    if recommendation.exists():raise FileExistsError(recommendation)
    recommendation.write_text('# Additional scene; original payload and legacy presets remain unchanged.\n'+yaml.safe_dump(cfg,sort_keys=False))
    write(root/'selected_scene.json',dict(name=best['name'],preset=str(recommendation),pilot=best,heldout=validation,
        interpretation='Known 5kg payload increases nominal J2 loading. Disturbance saturation and N0/N1/D0 comparison remain untested at this new scene.'))
    # Common scale makes J2 loading changes visible; all nominal scenes have zero clipping.
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,8),sharex=True)
    for name,z in results.items():
        t=np.arange(len(z['u_requested']))*.02
        axes[0].plot(t,z['u_requested'][:,1],label=name)
        axes[1].plot(t,z['u_requested'][:,2],label=name)
    for j,axis in enumerate(axes):
        axis.axhline(-150,color='r',ls='--',label='physical lower bound')
        axis.set_ylabel(f'J{j+2} u_req [Nm]');axis.grid(alpha=.25);axis.legend(fontsize=8)
    axes[1].set_xlabel('Time [s]');fig.tight_layout();fig.savefig(root/'nominal_scene_torques.png',dpi=170);plt.close(fig)
    print('[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
