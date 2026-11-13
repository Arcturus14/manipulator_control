"""Bounded geometry/IK screening of native shapes with controlled kinematics.

Rectangle uses the existing four straight sides, with quintic segment timing
(zero velocity/acceleration at corners), NOT the native constant-speed timing.
Local bounded IK is a prefilter, never a global infeasibility certificate.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from config import DT,Q_MIN,Q_MAX
from utils.trajectory import TRAJECTORIES
from kinematics.ur5_kinematics import fk_joints_np


def local_curve(shape,n):
    if shape!='rectangle':
        raw=TRAJECTORIES[shape](n,np.zeros(3),1.)
        return raw-raw[0]
    corners=TRAJECTORIES['rectangle'](4,np.zeros(3),1./1.6)
    progress=np.arange(n)*4/n;segment=progress.astype(int);u=progress-segment
    smooth=10*u**3-15*u**4+6*u**5
    raw=corners[segment]*(1-smooth[:,None])+corners[(segment+1)%4]*smooth[:,None]
    raw=raw-raw[0];raw[:,0]*=-1
    return raw


def build_reference(q,shape,ratio,vmax=.25,amax=.35):
    start=fk_joints_np(q)[-1];radial=np.r_[start[:2],0.];r=np.linalg.norm(radial);radial/=r
    tangent=np.array([-radial[1],radial[0],0.]);radius=ratio*r
    unit=local_curve(shape,2048)
    v=(np.roll(unit,-1,axis=0)-np.roll(unit,1,axis=0))*1024
    acc=(np.roll(unit,-1,axis=0)-2*unit+np.roll(unit,1,axis=0))*2048**2
    period=max(8.,radius*np.linalg.norm(v,axis=1).max()/vmax,
               np.sqrt(radius*np.linalg.norm(acc,axis=1).max()/amax))
    period=np.ceil(period/DT)*DT;n=round(period/DT)
    points=local_curve(shape,n)
    ref=start+radius*(points[:,0,None]*radial+points[:,1,None]*tangent)
    velocity=(np.roll(ref,-1,axis=0)-np.roll(ref,1,axis=0))/(2*DT)
    acceleration=(np.roll(ref,-1,axis=0)-2*ref+np.roll(ref,1,axis=0))/DT**2
    return ref,radius,float(period),float(np.linalg.norm(velocity,axis=1).max()),float(np.linalg.norm(acceleration,axis=1).max())


def main():
    from kroc.collision_geometry import Geometry
    from kroc.screen_nominal_margin_pressure import bounded_ik
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--coupled-wrist',action='store_true',
                   help='Base workspace pressure plus near-limit wrist disturbance candidate bank.')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    geom=Geometry();selected=[];rows=[];span=np.asarray(Q_MAX)-np.asarray(Q_MIN)
    scenarios=list(itertools.product([.05] if a.coupled_wrist else [.05,.075],
        [-1.4,-1.0],[-1.5,-.5],['circle'] if a.coupled_wrist else ['circle','infinity','rectangle'],
        [.35,.40] if a.coupled_wrist else [.35,.40,.45],
        [.88*Q_MAX[4],.89*Q_MAX[4]] if a.coupled_wrist else [-1.57]))
    for high,q2,q4,shape,ratio,q5 in scenarios:
        q=np.array([(1-2*high-.02)*Q_MAX[0],q2,1.2,q4,q5,0.])
        slack=np.minimum(q-np.asarray(Q_MIN)-high*span,np.asarray(Q_MAX)-high*span-q)
        assert slack.min()>0
        ref,radius,period,speed,acc=build_reference(q,shape,ratio)
        sample=ref[::max(1,round(.10/DT))]
        qphys,ep=bounded_ik(sample,q)
        row={'shape':shape,'high_fraction':high,'q0':q.tolist(),'ratio':ratio,
             'radius':radius,'period_s':period,'peak_cartesian_speed_m_s':speed,
             'peak_cartesian_acceleration_m_s2':acc,'physical_ik_max_error_mm':float(ep.max()*1000)}
        if ep.max()>1e-4:rows.append({**row,'rejected':'physical local IK error'});continue
        geo=geom.trajectory(qphys,stop_on_collision=True);row['reference_mesh']=geo
        if geo['first_collision_index'] is not None or geo['floor_min_m']<.01:
            rows.append({**row,'rejected':'reference collision/clearance'});continue
        qlow,el=bounded_ik(sample,q,.025);_,eh=bounded_ik(sample,q,high)
        row.update(low_ik_rmse_mm=float(np.sqrt(np.mean(el**2))*1000),
            high_ik_rmse_mm=float(np.sqrt(np.mean(eh**2))*1000),
            pressure_mm=float((np.sqrt(np.mean(eh**2))-np.sqrt(np.mean(el**2)))*1000),
            physical_ik_peak_base_rad=float(qphys[:,0].max()),
            low_ik_peak_base_rad=float(qlow[:,0].max()),
            initial_common_slack_rad=float(slack.min()),plane='xy',pose=len(rows),candidate=len(selected))
        np.savez_compressed(a.output/f'candidate_{len(selected)}.npz',reference=ref,initial_q=q)
        selected.append(row);rows.append(row);print(json.dumps(row),flush=True)
        (a.output/'screen.json').write_text(json.dumps({'selected':selected,'rows':rows,
            'headroom_enabled':False,'speed_limit_m_s':.25,'acceleration_limit_m_s2':.35,
            'reference_mesh_sample_seconds':.1,'rectangle_timing':'quintic stop at corners',
            'bounded_local_ik_is_not_global_certificate':True},indent=2))
    print('SCREEN_COMPLETE',len(selected),'/',len(scenarios),flush=True)


if __name__=='__main__':main()
