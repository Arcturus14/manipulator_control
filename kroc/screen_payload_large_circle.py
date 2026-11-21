"""Larger YZ circles, fixed center, frozen N0/payload settings; CUDA only."""
import copy
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from dynamics.ur5_payload import CubePayload
from kinematics.payload_tcp import PayloadTCP
from kroc.payload_scene import make_reference
from kroc.payload_geometry import audit_cube_trajectory
from kroc.collision_geometry import Geometry
from kroc.run_payload_cube_tcp import run_payload
from kroc.screen_payload_nominal import safe
from kroc.view_payload_tcp import torque_metrics,save_outputs,load_result


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def main():
    cuda=cuda_preflight()
    base=yaml.safe_load(Path('configs/payload_cube_tcp_extended_n0.yaml').read_text())
    root=Path('kroc/results/payload_large_circle_20261121')/datetime.now().strftime('%Y%m%d_%H%M%S')
    root.mkdir(parents=True)
    payload=CubePayload(**base['payload']);tcp=PayloadTCP(payload,'cpu',torch.float64);geometry=Geometry()
    qbase=np.array(base['scene']['initial_q']);start=tcp.position_and_jacobian_np(qbase)[0]
    center=start-np.array([0,base['scene']['circle_radius_m'],0])
    grid=[(.08,10.),(.12,15.),(.16,20.)]
    write(root/'plan.json',dict(cuda=cuda,center_m=center.tolist(),candidates=grid,seed=6807,
        heldout=[6817,6827],disturbance=False,DOB=False,control_changes=False,
        criteria='Largest radius with RMSE<=2mm, no actual collision/joint violation/clipping, mean abs J2>=80Nm; fixed candidate then two heldout seeds. Preserve old presets.'))
    configs={};screen=[]
    for radius,period in grid:
        name=f'r{round(radius*100):02d}cm_T{period:g}s'
        cfg=copy.deepcopy(base);row=dict(name=name,radius_m=radius,period_s=period)
        try:
            # Existing generator starts at center+[0,r,0]. Move initial joints
            # by IK too: changing radius alone would shift the center in -Y.
            q0=tcp.reference_np([center+np.array([0,radius,0])],qbase)[0]
            cfg['scene'].update(initial_q=q0.tolist(),circle_radius_m=radius,period_s=period,duration_s=period)
            ref=make_reference(tcp,q0,101,period/100,cfg['scene'])
            np.testing.assert_allclose(ref[0]-np.array([0,radius,0]),center,atol=2e-7)
            q=tcp.reference_np(ref,q0)
            robot=geometry.trajectory(q);cube=audit_cube_trajectory(q,payload,subdivisions=1,dt=period/100)
            row.update(center_m=(ref[0]-np.array([0,radius,0])).tolist(),initial_q=q0.tolist(),
                v_tcp_m_s=2*np.pi*radius/period,a_tcp_m_s2=radius*(2*np.pi/period)**2,
                min_jacobian_singular_value_m=float(min(np.linalg.svd(tcp.position_and_jacobian_np(qi)[1],compute_uv=False)[-1] for qi in q)),
                robot_mesh=robot,payload_mesh=cube,
                feasible=not(robot['self_collision_states'] or robot['floor_collision_states'] or cube['payload_collision_states']))
            if row['feasible']:configs[name]=cfg
        except ValueError as error:row.update(feasible=False,error=str(error))
        screen.append(row);write(root/'prescreen.json',screen);print(json.dumps(row),flush=True)
    metrics=[]
    for name,cfg in configs.items():
        z,v,path=run_payload(cfg,seed=6807,case='N0',output=root/name)
        m=torque_metrics(z);m.update(name=name,radius_m=cfg['scene']['circle_radius_m'],period_s=cfg['scene']['period_s'],
            safe=safe(v),mean_abs_j2_nm=float(np.abs(z['u_requested'][:,1]).mean()))
        metrics.append(m);save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        write(root/'metrics.json',metrics)
    eligible=[r for r in metrics if r['safe'] and r['rmse_err_mm']<=2. and r['clipping_steps']==0 and r['mean_abs_j2_nm']>=80.]
    if not eligible:
        write(root/'selection.json',dict(selected=None,reason='No candidate passed predefined criteria'));print('[OUTPUT]',root);return
    best=max(eligible,key=lambda row:row['radius_m']);cfg=configs[best['name']];heldout=[]
    for seed in [6817,6827]:
        z,v,path=run_payload(cfg,seed=seed,case='N0',output=root/f"{best['name']}_seed{seed}")
        m=torque_metrics(z);m.update(seed=seed,safe=safe(v),mean_abs_j2_nm=float(np.abs(z['u_requested'][:,1]).mean()));heldout.append(m)
        save_outputs({'N0':z},path,torque=True,components=False,disturbance_error=False,csv=True)
    valid=all(r['safe'] and r['rmse_err_mm']<=2. and r['clipping_steps']==0 and r['mean_abs_j2_nm']>=80 for r in heldout)
    preset=Path('configs/payload_cube_tcp_large_yz_n0.yaml')
    if valid:
        if preset.exists():raise FileExistsError(preset)
        preset.write_text('# Additional large-circle scene; center preserved, previous presets unchanged.\n'+yaml.safe_dump(cfg,sort_keys=False))
    write(root/'selection.json',dict(pilot=best,heldout=heldout,heldout_pass=valid,preset=str(preset) if valid else None,center_m=center.tolist()))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for name,radius,period in [('previous',.04,6.),*[(r['name'],r['radius_m'],r['period_s']) for r in screen if r['feasible']]]:
        phase=np.linspace(0,2*np.pi,301)
        axes[0].plot(center[1]+radius*np.cos(phase),center[2]+radius*np.sin(phase),label=f'{name}: r={radius*100:g} cm, T={period:g}s')
    axes[0].set(xlabel='Y [m]',ylabel='Z [m]',title=f'Fixed-center YZ references, X={center[0]:.3f} m')
    axes[0].axis('equal');axes[0].legend(fontsize=8);axes[0].grid(alpha=.3)
    for row in metrics:
        z=load_result(root/row['name']);axes[1].plot(np.arange(len(z['u_requested']))*.02,z['u_requested'][:,1],label=row['name'])
    axes[1].set(xlabel='Time [s]',ylabel='J2 requested torque [Nm]',title='N0, 5kg, no disturbance / DOB')
    axes[1].legend();axes[1].grid(alpha=.3);fig.tight_layout();fig.savefig(root/'large_circle_comparison.png',dpi=170);plt.close(fig)
    z=load_result(root/best['name'])
    from visualization.animator import animate_comparison
    fig,animation=animate_comparison({'N0':z},z['tcp_reference'])
    for frame in [0,len(z['q'])//4,len(z['q'])//2,len(z['q'])-1]:
        animation._func(frame);fig.savefig(root/f'large_circle_frame_{frame:04d}.png',dpi=150)
    animation._draw_was_started=True;plt.close(fig)
    print('[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
