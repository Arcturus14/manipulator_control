"""Opt-in GPU 26-DOP enclosing-polytope cost; verify with triangle meshes.

Supporting planes enclose the official UR5 CB3 meshes. Face-normal axes are
conservative (edge/edge axes omitted), so false positives are possible.
This soft cost is not a safety filter or a collision-free guarantee.

Backends preserve the same geometry/penalty: reference uses the original
matrix operations, fused uses scalar affine FK and traceable reductions,
triton (default) additionally avoids distance workspaces on CUDA float32.
CPU and float64 use the fused Torch path for numerical validation.
"""
import itertools
import numpy as np
import torch
from scipy.spatial import HalfspaceIntersection
from kroc.collision_geometry import Geometry
from kroc.check_circle_mesh_collisions import NAMES, transform


class PolytopeCollisionCost:
    def __init__(self, device, dtype, weight=50000., clearance=.005,
                 backend='triton'):
        if backend not in {'reference','fused','triton'}:
            raise ValueError('unknown collision backend')
        self.backend=backend
        g=Geometry();vertices=[];origins=[];normals=[];offsets=[]
        directions=np.array([v for v in itertools.product((-1,0,1),repeat=3) if any(v)],float)
        directions/=np.linalg.norm(directions,axis=1)[:,None]
        for mesh,origin in zip(g.vertices,g.origins):
            v=mesh@origin[:3,:3].T+origin[:3,3]
            support=(directions@v.T).max(1)+1e-5
            points=HalfspaceIntersection(np.column_stack((directions,-support)),v.mean(0)).intersections
            vertices.append(points);normals.append(directions);offsets.append(-support)
        maxn=max(map(len,vertices))
        vertices=[np.concatenate((v,np.repeat(v[:1],maxn-len(v),axis=0))) for v in vertices]
        for name in NAMES[1:]:
            k=g.kin[name]
            origins.append(transform([k[x] for x in ('x','y','z')],
                                     [k[x] for x in ('roll','pitch','yaw')]))
        tensor=lambda v:torch.as_tensor(np.array(v),device=device,dtype=dtype)
        self.vertices=tensor(vertices);self.normals=tensor(normals)
        self.offsets=tensor(offsets);self.origins=tensor(origins)
        self.origin_coefficients=tuple(tuple(float(x) for x in o.reshape(-1))
                                       for o in origins)
        self.i=torch.tensor([i for i,j in g.pairs],device=device)
        self.j=torch.tensor([j for i,j in g.pairs],device=device)
        self.weight=float(weight);self.clearance=float(clearance)

    def world_reference(self,q):
        count=q.shape[0]
        tr=torch.eye(4,device=q.device,dtype=q.dtype).expand(count,4,4)
        rotations=[tr[:,:3,:3]];translations=[tr[:,:3,3]]
        for i in range(6):
            angle=q[:,i];c=angle.cos();s=angle.sin()
            zero=torch.zeros_like(c);one=torch.ones_like(c)
            rz=torch.stack((c,-s,zero,zero,s,c,zero,zero,
                            zero,zero,one,zero,zero,zero,zero,one),-1).reshape(-1,4,4)
            tr=tr@self.origins[i]@rz
            rotations.append(tr[:,:3,:3]);translations.append(tr[:,:3,3])
        rotation=torch.stack(rotations,1);translation=torch.stack(translations,1)
        v=self.vertices@rotation.transpose(-1,-2)+translation[:,:,None,:]
        n=self.normals@rotation.transpose(-1,-2)
        b=self.offsets-(n*translation[:,:,None,:]).sum(-1)
        return v,n,b

    def world(self,q):
        if self.backend=='reference':
            return self.world_reference(q)
        # Fixed six-link affine FK: scalar expressions can fuse across joints.
        # Keep the reference 4x4 chain above for independent regression checks.
        zero=torch.zeros_like(q[:,0]);one=torch.ones_like(zero)
        r=(one,zero,zero,zero,one,zero,zero,zero,one)
        p=(zero,zero,zero);rotations=[torch.stack(r,-1).reshape(-1,3,3)]
        translations=[torch.stack(p,-1)]
        for i,o in enumerate(self.origin_coefficients):
            p=tuple(p[j]+r[3*j]*o[3]+r[3*j+1]*o[7]+r[3*j+2]*o[11]
                    for j in range(3))
            a=tuple(r[3*j]*o[k]+r[3*j+1]*o[4+k]+r[3*j+2]*o[8+k]
                    for j in range(3) for k in range(3))
            c=q[:,i].cos();s=q[:,i].sin()
            r=tuple(value for j in range(3) for value in
                    (a[3*j]*c+a[3*j+1]*s,-a[3*j]*s+a[3*j+1]*c,a[3*j+2]))
            rotations.append(torch.stack(r,-1).reshape(-1,3,3))
            translations.append(torch.stack(p,-1))
        rotation=torch.stack(rotations,1);translation=torch.stack(translations,1)
        def rotate(points):
            return sum(rotation[:,:,None,:,axis]*points[None,:,:,axis,None]
                       for axis in range(3))
        v=rotate(self.vertices)+translation[:,:,None,:]
        n=rotate(self.normals)
        b=self.offsets-(n*translation[:,:,None,:]).sum(-1)
        return v,n,b

    def separations_reference(self,q):
        v,n,b=self.world_reference(q)
        ab=(n[:,self.i]@v[:,self.j].transpose(-1,-2)+b[:,self.i,:,None]).amin(-1).amax(-1)
        ba=(n[:,self.j]@v[:,self.i].transpose(-1,-2)+b[:,self.j,:,None]).amin(-1).amax(-1)
        return torch.maximum(ab,ba),v[:,1:,:,2].amin(-1)

    def separations(self,q):
        if self.backend=='reference':
            return self.separations_reference(q)
        v,n,b=self.world(q)
        if self.backend=='triton' and q.is_cuda and q.dtype==torch.float32:
            from kroc.collision_kernels import pair_gaps
            return pair_gaps(v,n,b,self.i,self.j).amax(-1),v[:,1:,:,2].amin(-1)
        # Explicit 3-D dot products let Inductor fuse pair indexing, distances
        # and both reductions. Avoid general GEMM for a contraction of size 3.
        def gap(i,j):
            ni=n[:,i];vj=v[:,j]
            d=(ni[...,0,None]*vj[...,None,:,0]
               +ni[...,1,None]*vj[...,None,:,1]
               +ni[...,2,None]*vj[...,None,:,2]+b[:,i,:,None])
            return d.amin(-1).amax(-1)
        return torch.maximum(gap(self.i,self.j),gap(self.j,self.i)),v[:,1:,:,2].amin(-1)

    def __call__(self,state):
        pair,floor=self.separations(state[...,:6])
        return self.weight*((torch.relu(self.clearance-pair)/.05).square().sum(-1)
                            +(torch.relu(self.clearance-floor)/.05).square().sum(-1))
