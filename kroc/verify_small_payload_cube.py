"""Quarter-volume, equal-mass payload; preserve world TCP circle and old preset.

No controller/filter/covariance tuning. Known inertia, COM/TCP, all models and
collision/animation geometry use the same physically smaller rigid cube.
"""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from dynamics.ur5_payload import CubePayload
from kinematics.payload_tcp import PayloadTCP
from kroc.payload_scene import make_reference
from kroc.run_payload_cube_tcp import run_payload
from kroc.study_payload_torque_stability import metrics,write
from kroc.view_payload_tcp import load_result,save_outputs

OLD_PRESET=Path('configs/payload_cube_tcp_stabilized_n0.yaml')
NEW_PRESET=Path('configs/payload_cube_tcp_stabilized_small_cube_n0.yaml')
GUI_RESULT=Path('kroc/results/gui_final_n0/20261121_211045_647395_payload_seed6807/N0')
BASE=Path('kroc/results/payload_torque_stability_20261121/20261121_201719_stage2/posture5000_rate4')


def main():
    cuda=cuda_preflight();old=yaml.safe_load(OLD_PRESET.read_text());old_hash=hashlib.sha256(OLD_PRESET.read_bytes()).hexdigest()
    previous=load_result(GUI_RESULT);confirmed=load_result(BASE)
    for key in ['q','qd','u_requested','u_applied','err_mm']:np.testing.assert_array_equal(previous[key],confirmed[key])
    root=Path('kroc/results/payload_quarter_volume_20261121')/datetime.now().strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True)
    screenshot=dict(source=str(GUI_RESULT),bitwise_matches_verified_candidate=True,
        requested_min_nm=previous['u_requested'].min(0).tolist(),requested_max_nm=previous['u_requested'].max(0).tolist(),
        max_abs_requested_minus_applied_nm=float(np.abs(previous['u_requested']-previous['u_applied']).max()),
        rmse_mm=float(np.sqrt(np.mean(previous['err_mm']**2))))
    write(root/'user_gui_run_check.json',screenshot)
    side=old['payload']['side_m']*4**(-1/3)
    cfg=copy.deepcopy(old);cfg['payload'].update(side_m=side,center_in_flange_m=[0.,0.,side/2])
    cfg['output_root']='kroc/results/payload_quarter_volume_20261121'
    big_tcp=PayloadTCP(CubePayload(**old['payload']),'cpu',torch.float64)
    small_tcp=PayloadTCP(CubePayload(**cfg['payload']),'cpu',torch.float64)
    q0=np.asarray(old['scene']['initial_q']);world_start=big_tcp.position_and_jacobian_np(q0)[0]
    newq=small_tcp.reference_np([world_start],q0)[0];cfg['scene']['initial_q']=newq.tolist()
    oldref=make_reference(big_tcp,q0,1000,.02,old['scene']);newref=make_reference(small_tcp,newq,1000,.02,cfg['scene'])
    np.testing.assert_allclose(newref,oldref,atol=2e-7,rtol=0)
    if NEW_PRESET.exists():raise FileExistsError(NEW_PRESET)
    NEW_PRESET.write_text('# Quarter volume, same 5kg mass; smaller COM/TCP and fixed world circle. Original preset preserved.\n'+yaml.safe_dump(cfg,sort_keys=False))
    write(root/'plan.json',dict(cuda=cuda,old_preset=str(OLD_PRESET),new_preset=str(NEW_PRESET),
        old_side_m=old['payload']['side_m'],new_side_m=side,volume_ratio=(side/old['payload']['side_m'])**3,
        mass_kg=cfg['payload']['mass_kg'],new_tcp_offset_m=cfg['payload']['center_in_flange_m'],
        old_inertia_com_kg_m2=CubePayload(**old['payload']).inertia_com_kg_m2,new_inertia_com_kg_m2=CubePayload(**cfg['payload']).inertia_com_kg_m2,
        world_reference_max_change_m=float(np.abs(newref-oldref).max()),
        seed_list=[6807,6817,6827],two_cycle_seed=6807,
        unchanged='Controller costs, sensor noise, covariance, lambda, DOB/disturbance OFF, radius16cm period20s.',
        criteria='No retuning. Check RMSE<=2mm, J2peak<=110Nm, no clipping/J3 sign crossing/joint or mesh collision violation.'))
    rows={};cases={}
    for seed in [6807,6817,6827]:
        z,v,path=run_payload(cfg,seed,'N0',output=root/f'seed{seed}')
        row=metrics(z,v);rows[str(seed)]=row;write(root/'metrics.json',rows)
        np.testing.assert_allclose(z['payload_side_m'],side)
        np.testing.assert_allclose(z['payload_com_in_flange_m'],[0,0,side/2])
        np.testing.assert_allclose(z['payload_inertia_com_diag'],[CubePayload(**cfg['payload']).inertia_com_kg_m2]*3)
        edge=np.linalg.norm(z['payload_cube_vertices'][:,0]-z['payload_cube_vertices'][:,1],axis=1)
        np.testing.assert_allclose(edge,side,atol=1e-6)
        np.testing.assert_allclose(z['tcp_reference'],oldref,atol=3e-7,rtol=0)
        cases[str(seed)]=z
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
    long=copy.deepcopy(cfg);long['scene']['duration_s']=40.
    z,v,path=run_payload(long,6807,'N0',output=root/'two_cycles_seed6807')
    two=metrics(z,v);save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
    valid=all(r['safe'] and r['rmse_err_mm']<=2 and r['clipping_steps']==0 and not r['q3_zero_crossings_s'] and r['max_abs_j2_req_nm']<=110 for r in [*rows.values(),two])
    assert hashlib.sha256(OLD_PRESET.read_bytes()).hexdigest()==old_hash
    write(root/'assessment.json',dict(passed=bool(valid),seeds=rows,two_cycles=two,preset=str(NEW_PRESET),old_preset_unchanged=True,
        matched_shared_payload_models=True,cube_edges_and_world_tcp_reference_verified=True,
        scope='Known 5kg, no external disturbance or DOB; N1/D0 disturbance comparison remains untested.'))
    import matplotlib.pyplot as plt
    from visualization.animator import animate_comparison
    fig,anim=animate_comparison({'N0':cases['6807']},cases['6807']['tcp_reference'])
    for frame in [0,500,1000]:
        anim._func(frame);fig.savefig(root/f'small_cube_animation_{frame:04d}.png',dpi=150)
    anim._draw_was_started=True;plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for label,z in [('10cm cube',previous),('6.30cm cube',cases['6807'])]:
        t=np.arange(len(z['u_requested']))*.02
        for j,ax in enumerate(axes.flat):ax.plot(t,z['u_requested'][:,j],label=label,lw=.9)
    for j,ax in enumerate(axes.flat):ax.set_title(f'J{j+1}');ax.set_ylabel('Nm');ax.grid(alpha=.25)
    axes[0,0].legend();axes[-1,0].set_xlabel('Time [s]');axes[-1,1].set_xlabel('Time [s]');fig.tight_layout();fig.savefig(root/'size_torque_comparison.png',dpi=160);plt.close(fig)
    print('[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
