"""Collision kernel against float64 on 680000 postures, plus its all-branch timing. CUDA only.

 python -m kroc.check_collision_kernel
"""
import math
from pathlib import Path
import time

import numpy as np

from kroc.evidence_common import EVIDENCE,load_config,provenance,simulate,write_json

OUTPUT=EVIDENCE/'equivalence_20261123/collision_kernel.json'
TRIALS=500


def main():
    import torch
    from kinematics.payload_tcp import PayloadTCP
    from kroc.payload_geometry import CubeCollisionCost
    from kroc.self_collision_cost import PolytopeCollisionCost
    box={}

    def modify(options):
        class Capture(options['tracker_class']):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs);box['tracker']=self
        options['tracker_class']=Capture

    result=simulate(load_config(),7207,'N0',modify)
    tracker=box['tracker'];kernels=tracker._rollout_kernels
    count=tracker.n_samples*tracker.horizon
    cpu,float64=torch.device('cpu'),torch.float64
    robot=PolytopeCollisionCost(cpu,float64,tracker.collision_geometry_cost.weight,
        tracker.collision_geometry_cost.clearance,backend='fused')
    cube=CubeCollisionCost(PayloadTCP(tracker.payload,cpu,float64),robot,
        weight=tracker.cube_cost.weight,clearance=tracker.cube_cost.clearance)

    def collide(frames,bounds=None,ranges=None):
        kernels.collision[(count,)](frames,kernels.vertices,kernels.normals,kernels.offsets,
            kernels.box if bounds is None else bounds,kernels.range if ranges is None else ranges,
            kernels.i,kernels.j,kernels.collision_cost,num_warps=1)

    torch.set_num_threads(16);random=np.random.default_rng(5)
    missed=false=near=0;kernel_error=reference_error=0.
    for trial in range(TRIALS):
        kind=trial%4
        q=(random.uniform(-math.pi,math.pi,(count,6)) if kind==0 else
           result['q'][random.integers(0,len(result['q']),count)]
           +random.normal(0,(0,.15,.5,1.2)[kind],(count,6)))
        rotation=np.broadcast_to(np.eye(3),(count,3,3)).copy();position=np.zeros((count,3))
        frames=[np.c_[rotation.reshape(count,9),position]]
        for joint,coefficients in enumerate(robot.origin_coefficients):
            origin=np.reshape(coefficients,(-1,4))[:3]
            position=position+rotation@origin[:,3];rotation=rotation@origin[:,:3]
            c,s=np.cos(q[:,joint]),np.sin(q[:,joint]);turn=np.zeros((count,3,3))
            turn[:,0,0]=c;turn[:,0,1]=-s;turn[:,1,0]=s;turn[:,1,1]=c;turn[:,2,2]=1
            rotation=rotation@turn;frames.append(np.c_[rotation.reshape(count,9),position])
        q64=torch.as_tensor(q,dtype=float64)
        flange=cube.tcp.flange_transform(q64).numpy()
        frames.append(np.c_[flange[:,:3,:3].reshape(count,9),flange[:,:3,3]])
        collide(torch.as_tensor(np.stack(frames,1),device=tracker.device,dtype=torch.float32)
                .reshape(tracker.n_samples,tracker.horizon,96))
        state=torch.cat((q64,torch.zeros_like(q64)),1)
        truth=(robot(state)+cube(state)).numpy()
        single=state.to(tracker.device,torch.float32)
        existing=(tracker.collision_geometry_cost(single)+tracker.cube_cost(single)).cpu().numpy().astype(float)
        fast=kernels.collision_cost.reshape(count).cpu().numpy().astype(float)
        missed+=int(((truth>1.)&(fast==0)).sum());false+=int(((truth==0)&(fast>1.)).sum())
        near+=int(((truth>0)&(truth<500)).sum())
        scale=np.maximum(truth,1000.)
        kernel_error=max(kernel_error,float((np.abs(fast-truth)/scale).max()))
        reference_error=max(reference_error,float((np.abs(existing-truth)/scale).max()))

    tracker.ctrl.state=torch.as_tensor(np.r_[result['q'][300],result['qd'][300]],
        device=tracker.device,dtype=tracker.dtype)
    kernels(tracker.ctrl.perturbed_action.clone())
    open_box=kernels.box.clone();open_box[:,3:]+=100.
    open_range=kernels.range.clone();open_range[:,:32]-=100.;open_range[:,32:]+=100.
    timing={}
    for name,arguments in (('nominal',()),('every_branch',(open_box,open_range))):
        samples=[]
        for repeat in range(350):
            torch.cuda.synchronize();start=time.perf_counter()
            collide(kernels.frames,*arguments)
            torch.cuda.synchronize()
            if repeat>=50:samples.append((time.perf_counter()-start)*1000)
        timing[name]=float(np.median(samples))
    report=dict(configurations=TRIALS*count,threshold_cost=1.,missed_penalties=missed,
        false_penalties=false,near_threshold_cases=near,
        max_relative_error_kernel=kernel_error,max_relative_error_captured_float32=reference_error,
        collision_kernel_median_ms=timing,provenance=provenance(Path('kroc/check_collision_kernel.py')))
    write_json(OUTPUT,report)
    print({k:v for k,v in report.items() if k!='provenance'},flush=True)


if __name__=='__main__':main()
