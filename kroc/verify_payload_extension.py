"""Bounded CUDA integration checks, not a payload/pose/disturbance search."""
import copy
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT
from controllers.payload_mppi import PayloadFinalTracker
from dynamics.ur5_payload import CubePayload
from experiment_runner import run_mppi
from kinematics.ur5_kinematics import fk_batch
from kroc.optimize_final_mppi import options
from kroc.run_payload_cube_tcp import run_payload

ROOT=Path('kroc/results/payload_cube_tcp_20261121/verification')


def clean():
    gc.collect();torch.cuda.empty_cache()


def main():
    cuda=cuda_preflight();ROOT.mkdir(parents=True,exist_ok=True)
    cfg=yaml.safe_load(Path('configs/payload_cube_tcp.yaml').read_text())
    cfg['scene']['use_dob']=True
    checks=[]
    for case,steps in [('N0',None),('N1',40),('D0',40)]:
        path=ROOT/case
        if (path/'verification.json').exists():
            verified=json.loads((path/'verification.json').read_text())
            with np.load(path/'trajectory.npz') as z:
                assert bool(z['payload_models_shared'])
                if case=='D0':
                    assert str(z['input_study_case'])=='D0'
                    assert np.max(np.abs(z['u_mppi']+z['d_hat_used']-z['u_applied']))<=3.1e-5
            checks.append(verified);continue
        z,verified,_=run_payload(copy.deepcopy(cfg),case=case,steps=steps,output=path)
        assert verified['no_position_violation'] and verified['no_velocity_violation']
        assert verified['robot_mesh']['first_collision_index'] is None
        assert verified['payload_mesh']['first_payload_collision_index'] is None
        if case=='D0':
            assert np.max(np.abs(z['u_mppi']+z['d_hat_used']-z['u_applied']))<=3.1e-5
        checks.append(verified);del z;clean()

    # New code must not perturb the original controller. Compare a zero-load,
    # zero-TCP-offset adapter with the old selected N0 execution directly.
    q0=np.asarray(cfg['scene']['initial_q']);q=torch.tensor(q0[None],device='cuda',dtype=torch.float32)
    p0=fk_batch(q)[0].cpu().numpy();t=np.arange(40)*DT
    ref=p0+np.column_stack((.004*(np.cos(2*np.pi*t/6)-1),.004*np.sin(2*np.pi*t/6),np.zeros(40)))
    oldopts=options(6807,'N0','margin_capture')
    oldopts['tracker_options']['sensor_noise'].update(position_std_rad=0.,velocity_std_rad_s=0.)
    old=run_mppi(ref,initial_q=q0,has_disturbance=False,**oldopts)
    newopts=options(6807,'N0','margin_capture')
    newopts['tracker_options']['sensor_noise'].update(position_std_rad=0.,velocity_std_rad_s=0.)
    newopts['tracker_class']=PayloadFinalTracker
    newopts['tracker_options'].update(payload=CubePayload(0.,.1,(0.,0.,0.)),payload_collision=False)
    new=run_mppi(ref,initial_q=q0,has_disturbance=False,**newopts)
    differences={key:float(np.max(np.abs(old[key]-new[key]))) for key in ['q','qd','ee','err_mm','u_mppi','u_applied','d_hat_used']}
    for key in differences:
        np.testing.assert_array_equal(old[key],new[key],err_msg='Zero-extension regression:'+key)
    np.savez_compressed(ROOT/'zero_extension_reference.npz',**{k:old[k] for k in differences})
    np.savez_compressed(ROOT/'zero_extension_new.npz',**{k:new[k] for k in differences})
    before=json.loads(Path('/tmp/payload_existing_sources_before_20261121.json').read_text())
    changed=[name for name,digest in before.items() if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest]
    assert not changed,changed
    result=dict(passed=True,cuda=cuda,unit_checks=8,loaded_checks=checks,
        zero_payload_zero_tcp_bitwise_equal=True,zero_extension_max_differences=differences,
        existing_source_config_files_unchanged=len(before),existing_sources_modified=changed,
        new_files=['configs/payload_cube_tcp.yaml','dynamics/ur5_payload.py','kinematics/payload_tcp.py',
            'controllers/payload_mppi.py','kroc/payload_geometry.py','kroc/run_payload_cube_tcp.py',
            'kroc/verify_payload_extension.py','tests/test_payload_cube_tcp.py','kroc/requirements_payload.txt'],
        meaning='New optional model/TCP path implemented and checked. Payload/pose/load scenario not tuned; GUI defaults and frozen N0/N1/D0 unchanged.')
    (ROOT/'completed.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('VERIFICATION PASSED',ROOT/'completed.json',flush=True)


if __name__=='__main__':main()
