"""Additional cube collision cost and independent offline exact geometry audit."""
import numpy as np
import torch


class CubeCollisionCost:
    """Conservative face-axis SAT against robot links except attached link6.

    Uses the existing enclosing robot polytopes and cube faces; edge axes are
    omitted, so this is a soft conservative cost, not a safety guarantee.
    """
    def __init__(self,tcp,robot,weight=50000.,clearance=.005):
        self.tcp,self.robot=tcp,robot
        self.weight,self.clearance=weight,clearance

    def __call__(self,state):
        q=state[...,:6].reshape(-1,6)
        cube=self.tcp.vertices(q)
        value=(torch.relu(self.clearance-cube[...,2].amin(-1))/.05).square()
        if self.robot is not None:
            v,n,_=self.robot.world(q)
            # Index6 is wrist3, the mechanically attached collision link.
            v,n=v[:,:6],n[:,:6]
            a=torch.einsum('bldc,blvc->bldv',n,v)
            b=torch.einsum('bldc,bvc->bldv',n,cube)
            gap=torch.maximum(b.amin(-1)-a.amax(-1),a.amin(-1)-b.amax(-1)).amax(-1)
            axes=self.tcp.flange_transform(q)[:,:3,:3].transpose(-1,-2)
            a=torch.einsum('bdc,blvc->bldv',axes,v)
            b=torch.einsum('bdc,bvc->bdv',axes,cube)[:,None]
            gap=torch.maximum(gap,torch.maximum(b.amin(-1)-a.amax(-1),a.amin(-1)-b.amax(-1)).amax(-1))
            value=value+(torch.relu(self.clearance-gap)/.05).square().sum(-1)
        return self.weight*value


def audit_cube_trajectory(q,payload,subdivisions=4,dt=.02):
    """Exact FCL cube/robot intersections and cube/floor over interpolated q.

    The original offline robot mesh audit remains separate and unchanged.
    Collision at the attached wrist3 link is intentionally excluded only.
    """
    from config import DH_PARAMS
    from kroc.collision_geometry import Geometry
    from kroc.check_circle_mesh_collisions import frames
    import fcl
    g=Geometry();floor=float('inf');first=None;hits=0
    alpha=np.arange(subdivisions)/subdivisions
    dense=(q[:-1,None]*(1-alpha[None,:,None])+q[1:,None]*alpha[None,:,None]).reshape(-1,6)
    dense=np.concatenate((dense,q[-1:]))
    cube_shape=fcl.Box(payload.side_m,payload.side_m,payload.side_m)
    for i,qi in enumerate(dense):
        tr=np.eye(4)
        for angle,(a,d,aa,off) in zip(qi,DH_PARAMS):
            c,s=np.cos(angle+off),np.sin(angle+off);ca,sa=np.cos(aa),np.sin(aa)
            tr=tr@np.array([[c,-s*ca,s*sa,a*c],[s,c*ca,-c*sa,a*s],[0,sa,ca,d],[0,0,0,1]])
        center=tr[:3,3]+tr[:3,:3]@np.asarray(payload.center_in_flange_m)
        low=center[2]-payload.side_m*.5*np.abs(tr[2,:3]).sum()
        floor=min(floor,float(low))
        cube=fcl.CollisionObject(cube_shape,fcl.Transform(tr[:3,:3],center))
        bad=low < -1e-6
        for model,tf in zip(g.models[:6],[t@o for t,o in zip(frames(qi,g.kin),g.origins)][:6]):
            body=fcl.CollisionObject(model,fcl.Transform(tf[:3,:3],tf[:3,3]))
            bad=bool(fcl.collide(cube,body,fcl.CollisionRequest(),fcl.CollisionResult())) or bad
        if bad:
            hits+=1
            if first is None:
                first=i
    return dict(cube_floor_min_m=floor,payload_collision_states=hits,first_payload_collision_index=first,
                interpolation_dt_s=dt/subdivisions,excluded_attached_robot_link='wrist3 only')
