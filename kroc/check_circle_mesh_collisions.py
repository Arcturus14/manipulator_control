"""Offline UR5/CB3 collision mesh checks; never changes controller costs."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0,'/tmp/kroc_collision_deps')
import fcl
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import yaml
from config import DT
from kinematics.ur5_kinematics import fk_joints_np

NAMES=['base','shoulder','upper_arm','forearm','wrist_1','wrist_2','wrist_3']
FILES=['base','shoulder','upperarm','forearm','wrist1','wrist2','wrist3']
ROOT='https://raw.githubusercontent.com/ros-industrial/universal_robot/noetic-devel/ur_description/'


def transform(xyz=(0,0,0),rpy=(0,0,0)):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler('xyz',rpy).as_matrix();t[:3,3]=xyz
    return t


def frames(q,kin):
    t=np.eye(4);result=[t.copy()]
    for name,angle in zip(NAMES[1:],q):
        k=kin[name]
        t=t@transform([k[a] for a in ['x','y','z']],[k[a] for a in ['roll','pitch','yaw']])@transform(rpy=(0,0,float(angle)))
        result.append(t.copy())
    return result


def collision_origins(offsets):
    return [transform(rpy=(0,0,math.pi)),transform(rpy=(0,0,math.pi)),
        transform((0,0,offsets['shoulder_offset']),(math.pi/2,0,-math.pi/2)),
        transform((0,0,offsets['elbow_offset']),(math.pi/2,0,-math.pi/2)),
        transform((0,0,-.093),(math.pi/2,0,0)),
        transform((0,0,-.095)),transform((0,0,-.0818),(math.pi/2,0,0))]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('kroc/results/circle_floor25_independent_validation'))
    p.add_argument('--assets',type=Path,default=Path('kroc/assets/ur5_cb3_collision'))
    p.add_argument('--output',type=Path,default=Path('kroc/results/circle_floor25_mesh_collision_check'))
    p.add_argument('--download',action='store_true')
    a=p.parse_args();a.assets.mkdir(parents=True,exist_ok=True)
    sources={f'{name}.stl':f'meshes/ur5/collision/{name}.stl' for name in FILES}
    sources.update({f'{name}.yaml':f'config/ur5/{name}.yaml' for name in ['default_kinematics','physical_parameters','visual_parameters']})
    sources['ur_macro.xacro']='urdf/inc/ur_macro.xacro'
    if a.download:
        for name,url in sources.items():
            path=a.assets/name
            if not path.exists():
                with urllib.request.urlopen(ROOT+url,timeout=30) as response:path.write_bytes(response.read())
    metadata={name:{'url':ROOT+url,'sha256':hashlib.sha256((a.assets/name).read_bytes()).hexdigest()} for name,url in sources.items()}
    kin=yaml.safe_load((a.assets/'default_kinematics.yaml').read_text())['kinematics']
    physical=yaml.safe_load((a.assets/'physical_parameters.yaml').read_text())
    offsets=physical['offsets']
    # Origins taken from the official ur_macro.xacro, expressed in its internal
    # base_link_inertia frame. Its wrist_3 origin equals this project's DH TCP.
    origins=collision_origins(offsets)
    models=[];vertices=[];faces=[]
    for name in FILES:
        m=trimesh.load_mesh(a.assets/f'{name}.stl',process=True)
        v=np.asarray(m.vertices,dtype=np.float64);f=np.asarray(m.faces,dtype=np.int32)
        b=fcl.BVHModel();b.beginModel(len(v),len(f));b.addSubModel(v,f);b.endModel()
        models.append(b);vertices.append(v);faces.append(f)
    # Exclude directly connected links only. A two-joint-neighbor pair is kept
    # and reported by name rather than silently masked by a generated SRDF.
    pairs=[(i,j) for i,j in itertools.combinations(range(7),2) if j-i>1]
    a.output.mkdir(parents=True,exist_ok=False)
    rows=[];worst_snapshot=None;worst_clearance=np.inf;fk_max=0.
    for path in sorted(a.input.glob('s*_seed*.npz')):
        z=np.load(path);q=z['q'];pair_counts={f'{NAMES[i]}--{NAMES[j]}':0 for i,j in pairs}
        mins={name:np.inf for name in NAMES};collision_steps=[];ground_steps=[]
        pair_min={k:np.inf for k in pair_counts};first_pairs={}
        first_collision=None;first_ground=None
        for step,angles in enumerate(q):
            chain=frames(angles,kin)
            error=np.linalg.norm(chain[-1][:3,3]-fk_joints_np(angles)[-1]);fk_max=max(fk_max,float(error))
            assert error<1e-7, f'URDF/DH frame mismatch: {error}'
            ts=[t@o for t,o in zip(chain,origins)]
            objs=[fcl.CollisionObject(m,fcl.Transform(t[:3,:3],t[:3,3])) for m,t in zip(models,ts)]
            world=[v@t[:3,:3].T+t[:3,3] for v,t in zip(vertices,ts)]
            lowest={name:float(v[:,2].min()) for name,v in zip(NAMES,world)}
            for name,val in lowest.items():mins[name]=min(mins[name],val)
            # The static base is mounted on the plane and intentionally excluded.
            ground=any(lowest[name]<-1e-6 for name in NAMES[1:])
            hit=[]
            for i,j in pairs:
                name=f'{NAMES[i]}--{NAMES[j]}'
                result=fcl.CollisionResult();n=fcl.collide(objs[i],objs[j],fcl.CollisionRequest(),result)
                if n:
                    pair_counts[name]+=1;hit.append(name);pair_min[name]=min(pair_min[name],0.)
                    first_pairs.setdefault(name,step*DT)
                else:
                    d=fcl.distance(objs[i],objs[j],fcl.DistanceRequest(),fcl.DistanceResult())
                    pair_min[name]=min(pair_min[name],float(d))
            collision_steps.append(bool(hit));ground_steps.append(ground)
            if hit and first_collision is None:first_collision=step*DT
            if ground and first_ground is None:first_ground=step*DT
            clearance=min(lowest[name] for name in NAMES[1:])
            if worst_snapshot is None or (step==0 and worst_snapshot['step']!=0):
                worst_snapshot={'source':path.name,'step':step,'q':angles.tolist(),'hit_pairs':hit,'ground_min_m':clearance}
                np.savez_compressed(a.output/'initial_meshes.npz',**{f'{name}_vertices':v for name,v in zip(NAMES,world)},**{f'{name}_faces':f for name,f in zip(NAMES,faces)})
            worst_clearance=min(worst_clearance,clearance)
        row=dict(file=path.name,states=len(q),floor_z_m=0.,floor_collision_states=sum(ground_steps),first_floor_collision_s=first_ground,
            minimum_link_z_m=mins,self_collision_states=sum(collision_steps),first_self_collision_s=first_collision,
            pair_collision_states={k:v for k,v in pair_counts.items() if v},pair_minimum_separation_m=pair_min,
            first_pair_collision_s=first_pairs)
        rows.append(row);print(json.dumps({k:row[k] for k in ['file','floor_collision_states','self_collision_states','pair_collision_states']}),flush=True)
        (a.output/'results.json').write_text(json.dumps({'rows':rows},indent=2))
    summary={'sources':metadata,'method':'official UR5 CB3 triangle meshes; python-fcl BVH triangle intersection and distance',
        'frame':'project DH base = ROS internal base_link_inertia; verified wrist3 origin against project FK at every saved state',
        'dh_fk_max_error_m':fk_max,'floor_assumption':'infinite plane z=0 at base mounting plane; static base excluded',
        'excluded_self_pairs':'only directly adjacent links; no other exemptions',
        'sampling':'all saved true states, 20 ms; discrete tests, not a continuous collision-free certificate',
        'initial_state':worst_snapshot,'overall_minimum_link_z_m':worst_clearance,'rows':rows}
    (a.output/'results.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
