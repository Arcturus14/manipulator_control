"""Refine wrist orientation rather than reusing the rejected folded posture.

Offline geometry/IK screening only; controller experiments remain CUDA-only.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from config import Q_MAX
from kinematics.ur5_kinematics import fk_joints_np
from kroc.collision_geometry import Geometry
from kroc.screen_collision_free_circles import circle
from kroc.search_circle_floor25 import local_ik_reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=6)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    g=Geometry();initial=[];tested=0
    for q2,q4,q5 in itertools.product([-1.4,-1.3,-1.2,-1.1],
            [1.8,1.9,2.,2.1,2.2,2.3,2.4],[-2.4,-1.57,-.7,0.,.7,1.57]):
        tested+=1;q=np.array([.92*Q_MAX[0],q2,.9*Q_MAX[2],q4,q5,0.])
        r=g.check(q)
        if r['hit_pairs'] or r['floor_clearance_m']<.015 or r['self_clearance_m']<.005:continue
        initial.append({'q':q.tolist(),'initial_geometry':r,
                        'ee':fk_joints_np(q)[-1].tolist()})
    initial.sort(key=lambda c:(abs(c['q'][1]+1.2),-c['initial_geometry']['self_clearance_m']))
    selected=[];rejected=[]
    for c in initial:
        if len(selected)>=a.count:break
        if any(np.linalg.norm(np.array(c['q'])[1:5]-np.array(s['q'])[1:5])<.45 for s in selected):continue
        ref=circle(c['q']);qref=local_ik_reference(ref,np.array(c['q']))
        err=max(np.linalg.norm(fk_joints_np(q)[-1]-t) for q,t in zip(qref,ref))
        if err>1e-4:rejected.append({'q':c['q'],'reason':'IK error','max_error_m':err});continue
        check=g.trajectory(qref[::2],stop_on_collision=True)
        if check['first_collision_index'] is not None:
            rejected.append({'q':c['q'],'reason':'reference collision'});continue
        c.update(candidate=len(selected),ik_geometry=check,ik_error_m=err)
        selected.append(c);print(json.dumps(c),flush=True)
    result={'sampled_postures':tested,'initial_safe_count':len(initial),
            'selected':selected,'rejected':rejected,'headroom_enabled':False,
            'reference':'outward XY circle R=.04m, period4s, duration8s',
            'sampling':'reference mesh checks every40ms; actual controller states require rechecking'}
    (a.output/'candidates.json').write_text(json.dumps(result,indent=2))
    print('selected',len(selected),'of',tested,flush=True)


if __name__=='__main__':main()
