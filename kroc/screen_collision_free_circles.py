"""Screen posture and continuous IK circles before costly CUDA experiments."""
import itertools
import json
from pathlib import Path
import numpy as np
from config import Q_MAX,DT
from kinematics.ur5_kinematics import fk_joints_np
from kroc.search_circle_floor25 import local_ik_reference
from kroc.collision_geometry import Geometry


def circle(q,radius=.04,period=4,seconds=8):
    start=fk_joints_np(q)[-1];theta=np.arange(round(seconds/DT))*DT*2*np.pi/period
    radial=np.r_[start[:2],0.];radial/=np.linalg.norm(radial)
    tangent=np.array([-radial[1],radial[0],0.])
    return start+radius*((1-np.cos(theta))[:,None]*radial+np.sin(theta)[:,None]*tangent)


def main():
    output=Path('kroc/results/collision_free_circle_screen_outward');output.mkdir(parents=True,exist_ok=False)
    geom=Geometry();candidates=[];n=0
    for q2,q4,q5 in itertools.product([-2.4,-2.,-1.6,-1.2,-.8,-.4],[-3,-2,-1,0,1,2,3],[-2,-1.57,-.7,0,.7,1.57,2]):
        n+=1;q=np.array([.94*Q_MAX[0],q2,.9*Q_MAX[2],q4,q5,0.])
        ee=fk_joints_np(q)[-1]
        if ee[2]<.15 or np.linalg.norm(ee[:2])<.16:continue
        r=geom.check(q)
        if r['hit_pairs'] or r['floor_clearance_m']<.015 or r['self_clearance_m']<.002 or r['far_pair_clearance_m']<.015:continue
        candidates.append(dict(q=q.tolist(),ee=ee.tolist(),initial_geometry=r,score=r['far_pair_clearance_m']))
    candidates.sort(key=lambda r:r['score'],reverse=True)
    print('INITIAL_SAFE',len(candidates),'/',n,flush=True)
    selected=[];rejected={'ik_error':0,'reference_collision':0}
    for c in candidates:
        if len(selected)>=12:break
        # Keep different shoulder/wrist configurations, not duplicate yaw-only cases.
        if any(np.linalg.norm(np.array(c['q'])[1:5]-np.array(s['q'])[1:5])<.65 for s in selected):continue
        ref=circle(c['q']);qref=local_ik_reference(ref,np.array(c['q']))
        err=max(np.linalg.norm(fk_joints_np(q)[-1]-t) for q,t in zip(qref,ref))
        if err>1e-4:rejected['ik_error']+=1;continue
        check=geom.trajectory(qref[::4],stop_on_collision=True)
        if check['first_collision_index'] is not None:rejected['reference_collision']+=1;continue
        c.update(ik_geometry=check,ik_error_m=err,candidate=len(selected))
        selected.append(c);print('CANDIDATE',json.dumps(c),flush=True)
    (output/'candidates.json').write_text(json.dumps({'sampled_postures':n,'initial_safe_count':len(candidates),'rejected':rejected,'initial_candidates':candidates,'selected':selected},indent=2))


if __name__=='__main__':main()
