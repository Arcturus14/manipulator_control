"""Reusable offline UR5/CB3 collision geometry (no control-cost modification)."""
from pathlib import Path
import sys
sys.path.insert(0,'/tmp/kroc_collision_deps')
import fcl
import numpy as np
import trimesh
import yaml
from kroc.check_circle_mesh_collisions import frames,collision_origins,NAMES,FILES


class Geometry:
    def __init__(self, assets=Path('kroc/assets/ur5_cb3_collision')):
        self.kin=yaml.safe_load((assets/'default_kinematics.yaml').read_text())['kinematics']
        self.origins=collision_origins(yaml.safe_load((assets/'physical_parameters.yaml').read_text())['offsets'])
        self.vertices=[];self.models=[]
        for name in FILES:
            mesh=trimesh.load_mesh(assets/f'{name}.stl',process=True)
            v=np.array(mesh.vertices);f=np.array(mesh.faces,dtype=np.int32)
            model=fcl.BVHModel();model.beginModel(len(v),len(f));model.addSubModel(v,f);model.endModel()
            self.models.append(model);self.vertices.append(v)
        self.pairs=[(i,j) for i in range(7) for j in range(i+2,7)]

    def check(self,q):
        ts=[t@o for t,o in zip(frames(q,self.kin),self.origins)]
        floor=min(float((v@t[:3,:3].T+t[:3,3])[:,2].min()) for v,t in zip(self.vertices[1:],ts[1:]))
        objs=[fcl.CollisionObject(m,fcl.Transform(t[:3,:3],t[:3,3])) for m,t in zip(self.models,ts)]
        hits=[];distance=np.inf;far_distance=np.inf
        for i,j in self.pairs:
            n=fcl.collide(objs[i],objs[j],fcl.CollisionRequest(),fcl.CollisionResult())
            d=0. if n else float(fcl.distance(objs[i],objs[j],fcl.DistanceRequest(),fcl.DistanceResult()))
            if n:hits.append(f'{NAMES[i]}--{NAMES[j]}')
            distance=min(distance,d)
            if j-i>=3:far_distance=min(far_distance,d)
        return dict(floor_clearance_m=floor,self_clearance_m=distance,far_pair_clearance_m=far_distance,hit_pairs=hits)

    def trajectory(self,q,stop_on_collision=False):
        floor=np.inf;clearance=np.inf;hits=0;ground=0;first=None
        for i,state in enumerate(q):
            r=self.check(state);floor=min(floor,r['floor_clearance_m']);clearance=min(clearance,r['self_clearance_m'])
            collision=bool(r['hit_pairs']);g=r['floor_clearance_m']<-1e-6
            hits+=collision;ground+=g
            if (collision or g) and first is None:first=i
            if stop_on_collision and first is not None:break
        return dict(floor_min_m=floor,self_min_m=clearance,self_collision_states=hits,floor_collision_states=ground,first_collision_index=first)
