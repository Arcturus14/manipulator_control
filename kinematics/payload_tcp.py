"""Position TCP at a rigid cube center; no changes to flange kinematics."""
import itertools
import numpy as np
import torch

from config import DH_PARAMS, Q_MIN, Q_MAX
from kinematics.ur5_kinematics import _dh_mat_batch, fk_batch as flange_fk


class PayloadTCP:
    def __init__(self, payload, device, dtype):
        self.payload = payload
        self.center = torch.tensor(payload.center_in_flange_m, device=device, dtype=dtype)
        self.zero_offset = all(x == 0 for x in payload.center_in_flange_m)
        self.corners_local = torch.tensor(list(itertools.product((-.5,.5),repeat=3)),device=device,dtype=dtype)*payload.side_m

    def flange_transform(self, q):
        shape=q.shape[:-1]
        flat=q.reshape(-1,6)
        tr=torch.eye(4,device=q.device,dtype=q.dtype).expand(len(flat),-1,-1).clone()
        for j,(a,d,alpha,offset) in enumerate(DH_PARAMS):
            tr=torch.bmm(tr,_dh_mat_batch(a,d,alpha,flat[:,j]+offset))
        return tr.reshape(shape+(4,4))

    def fk_batch(self,q):
        if self.zero_offset:
            return flange_fk(q.reshape(-1,6)).reshape(q.shape[:-1]+(3,))
        tr=self.flange_transform(q)
        return tr[...,:3,3]+(tr[...,:3,:3]@self.center[...,None]).squeeze(-1)

    def vertices(self,q):
        tr=self.flange_transform(q)
        return ((self.corners_local+self.center)@tr[...,:3,:3].transpose(-1,-2)
                +tr[...,:3,3][...,None,:])

    def position_and_jacobian_np(self,q):
        tr=np.eye(4);axes=[];origins=[]
        for angle,(a,d,alpha,offset) in zip(q,DH_PARAMS):
            axes.append(tr[:3,2].copy());origins.append(tr[:3,3].copy())
            ct,st=np.cos(angle+offset),np.sin(angle+offset);ca,sa=np.cos(alpha),np.sin(alpha)
            tr=tr@np.array([[ct,-st*ca,st*sa,a*ct],[st,ct*ca,-ct*sa,a*st],[0,sa,ca,d],[0,0,0,1]])
        point=tr[:3,3]+tr[:3,:3]@np.asarray(self.payload.center_in_flange_m)
        jac=np.stack([np.cross(a,point-o) for a,o in zip(axes,origins)],axis=1)
        return point,jac

    def reference_np(self,targets,initial):
        """Same bounded local position IK as baseline, evaluated at the TCP."""
        q=np.asarray(initial,dtype=float).copy();result=[]
        for index,target in enumerate(targets):
            for _ in range(80):
                p,jac=self.position_and_jacobian_np(q)
                err=target-p
                if np.linalg.norm(err)<2e-7:
                    break
                delta=jac.T@np.linalg.solve(jac@jac.T+1e-5*np.eye(3),err)
                q=np.clip(q+.8*np.clip(delta,-.05,.05),Q_MIN,Q_MAX)
            if np.linalg.norm(target-self.position_and_jacobian_np(q)[0])>1e-5:
                raise ValueError(f'TCP reference IK failed at waypoint{index}; do not silently track flange')
            result.append(q.copy())
        return np.array(result)
