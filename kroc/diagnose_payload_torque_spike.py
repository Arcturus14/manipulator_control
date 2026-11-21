"""Offline CUDA torque decomposition of the saved large-circle N0 spike."""
import json
from pathlib import Path
import numpy as np
import torch
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from dynamics.ur5_payload import CubePayload,PayloadUR5Dynamics
from kinematics.payload_tcp import PayloadTCP
from kroc.self_collision_cost import PolytopeCollisionCost
from kroc.payload_geometry import CubeCollisionCost
from kroc.view_payload_tcp import load_result

ROOT=Path('kroc/results/payload_large_circle_20261121/20261121_174431')


def main():
    cuda=cuda_preflight();out=ROOT/'torque_spike_diagnosis';out.mkdir(exist_ok=False)
    dyn=PayloadUR5Dynamics('cuda',torch.float64,CubePayload())
    tcp=PayloadTCP(CubePayload(),'cuda',torch.float64)
    robot=PolytopeCollisionCost('cuda',torch.float64,backend='reference');cube=CubeCollisionCost(tcp,robot)
    records={};arrays={}
    for name in ['r16cm_T20s','r16cm_T20s_seed6817','r16cm_T20s_seed6827']:
        z=load_result(ROOT/name);q=z['q'].astype(float);v=z['qd'].astype(float)
        qt=torch.as_tensor((q[:-1]+q[1:])/2,device='cuda',dtype=torch.float64)
        vt=torch.as_tensor((v[:-1]+v[1:])/2,device='cuda',dtype=torch.float64)
        at=torch.as_tensor(np.diff(v,axis=0)/.02,device='cuda',dtype=torch.float64)
        mass,bias=dyn.mass_matrix_and_bias(qt,vt)
        gravity=dyn.gravity_compensation(qt);coriolis=bias-gravity
        inertia=(mass@at[...,None]).squeeze(-1);damping=dyn.damping*vt
        rebuilt=inertia+gravity+coriolis+damping
        values={key:value.cpu().numpy() for key,value in dict(gravity=gravity,coriolis=coriolis,inertia=inertia,damping=damping,reconstructed=rebuilt,acceleration=at).items()}
        # Diagnostics at the same pre-control states as the optimization inputs.
        s=torch.tensor(np.c_[q[:-1],v[:-1]],device='cuda',dtype=torch.float64)
        values['robot_cost']=robot(s).cpu().numpy();values['cube_cost']=cube(s).cpu().numpy()
        null_speed=[];singular=[];tcp_speed=[]
        for qi,vi in zip(q[:-1],v[:-1]):
            _,jac=tcp.position_and_jacobian_np(qi)
            task=np.linalg.pinv(jac)@(jac@vi)
            null_speed.append(np.linalg.norm(vi-task));singular.append(np.linalg.svd(jac,compute_uv=False)[-1]);tcp_speed.append(np.linalg.norm(jac@vi))
        values.update(null_speed=np.array(null_speed),jacobian_sigma_min=np.array(singular),tcp_speed=np.array(tcp_speed))
        arrays[name]=values
        i=int(np.abs(z['u_requested'][:,1]).argmax());rows=[]
        for t in [15.,16.,16.4,16.74,17.,18.]:
            k=round(t/.02)
            rows.append(dict(t_s=t,req_J2_nm=float(z['u_requested'][k,1]),ff_J2_nm=float(z['u_ff'][k,1]),
                residual_J2_nm=float(z['u_requested'][k,1]-z['u_ff'][k,1]),
                q_J2J3_rad=q[k,1:3].tolist(),dq_J2J3_rad_s=v[k,1:3].tolist(),
                ddq_J2J3_rad_s2=values['acceleration'][k,1:3].tolist(),
                decomposition_J2_nm={key:float(values[key][k,1]) for key in ['gravity','coriolis','inertia','damping','reconstructed']},
                ee_error_mm=float(z['err_mm'][k]),ess=float(z['mppi_weight_ess'][k]),
                null_speed_rad_s=float(values['null_speed'][k]),tcp_speed_m_s=float(values['tcp_speed'][k]),
                jacobian_sigma_min_m=float(values['jacobian_sigma_min'][k]),
                robot_collision_cost=float(values['robot_cost'][k]),cube_collision_cost=float(values['cube_cost'][k])))
        crossed=np.flatnonzero(q[:-1,2]*q[1:,2]<0)*.02
        records[name]=dict(peak_t_s=i*.02,peak_J2_nm=float(z['u_requested'][i,1]),q3_zero_crossings_s=crossed.tolist(),
            torque_reconstruction_max_error_nm=float(np.abs(values['reconstructed']-z['u_requested']).max()),
            torque_reconstruction_rmse_nm=float(np.sqrt(np.mean((values['reconstructed']-z['u_requested'])**2))),
            robot_collision_cost_max=float(values['robot_cost'].max()),cube_collision_cost_max=float(values['cube_cost'].max()),rows=rows)
        print(name,json.dumps(records[name],indent=2),flush=True)
    (out/'diagnosis.json').write_text(json.dumps(dict(cuda=cuda,method='Offline decomposition at midpoint states; acceleration from stored velocity differences / dt, including reflected armature in M. No controller rerun or modification.',records=records),indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,2,figsize=(14,11),sharex=True)
    for name in records:
        z=load_result(ROOT/name);t=np.arange(len(z['u_requested']))*.02;mask=(t>=15)&(t<=18)
        label=name.replace('r16cm_T20s','seed6807' if name=='r16cm_T20s' else '').strip('_')
        axes[0,0].plot(t[mask],z['u_requested'][mask,1],label=label)
        axes[0,1].plot(t[mask],z['q'][:-1][mask,2],label=label)
        axes[2,0].plot(t[mask],arrays[name]['null_speed'][mask],label=label)
    name='r16cm_T20s';z=load_result(ROOT/name);a=arrays[name]
    axes[0,0].plot(t[mask],z['u_ff'][mask,1],'k--',label='feedforward seed6807')
    for j in [1,2]:
        axes[1,0].plot(t[mask],z['qd'][:-1][mask,j],label=f'J{j+1}')
        axes[1,1].plot(t[mask],a['acceleration'][mask,j],label=f'J{j+1}')
    for key in ['gravity','inertia','coriolis','damping']:
        axes[2,1].plot(t[mask],a[key][mask,1],label=key)
    titles=['J2 requested torque / feedforward','Elbow angle J3','Joint velocity, seed6807','Joint acceleration, seed6807','Position-task null-space speed','J2 torque decomposition, seed6807']
    labels=['Nm','rad','rad/s','rad/s^2','rad/s','Nm']
    for axis,title,label in zip(axes.flat,titles,labels):
        axis.set_title(title);axis.set_ylabel(label);axis.axvline(16.74,color='gray',ls=':');axis.legend(fontsize=8);axis.grid(alpha=.25)
    axes[0,1].axhline(0,color='black',lw=.7)
    for axis in axes[-1]:axis.set_xlabel('Time [s]')
    fig.tight_layout();fig.savefig(out/'spike_mechanism.png',dpi=160);plt.close(fig)
    print('[OUTPUT]',out,flush=True)


if __name__=='__main__':main()
