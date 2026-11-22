"""Run an explicit payload/TCP configuration without touching legacy defaults.

Examples:
 python -m kroc.run_payload_cube_tcp --config configs/payload_cube_tcp.yaml
 python -m kroc.run_payload_cube_tcp --case D0 --dob --amplitude -10 --joint 2
Outputs use a new timestamped directory; existing data is never overwritten.
"""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT,TAU_MAX,Q_MIN,Q_MAX
from controllers.payload_mppi import PayloadFinalTracker,PayloadD0Tracker
from dynamics.ur5_payload import CubePayload
from experiment_runner import run_mppi
from kinematics.payload_tcp import PayloadTCP
from kroc.audit_circle_validation import dense_mesh_check
from kroc.optimize_final_mppi import options
from kroc.payload_geometry import audit_cube_trajectory
from kroc.position_input_headroom_study import raised_cosine_load
from kroc.payload_scene import make_reference


def run_payload(config,seed=6807,case=None,steps=None,output=None,capture_samples=False,sample_count=30):
    cuda=cuda_preflight()
    if config.get('schema_version')!=1:
        raise ValueError('Unsupported payload configuration schema')
    payload=CubePayload(**config['payload'])
    scene=config['scene'];settings=config['controller']
    case=case or settings['case']
    if case not in {'N0','N1','D0'}:
        raise ValueError('Case must be N0,N1,D0')
    q0=np.asarray(scene['initial_q'],dtype=float)
    if q0.shape!=(6,) or not np.isfinite(q0).all():
        raise ValueError('Initial q must have six finite entries')
    if np.any(q0<np.asarray(Q_MIN)) or np.any(q0>np.asarray(Q_MAX)):
        raise ValueError('Initial q violates the existing physical joint bounds')
    n=round(scene['duration_s']/DT) if steps is None else steps
    if n<4 or (steps is None and not np.isclose(n*DT,scene['duration_s'])):
        raise ValueError('Duration must be a dt multiple with at least four steps')
    if scene['disturbance_joint'] not in range(1,7):
        raise ValueError('Disturbance joint is numbered1--6')
    amplitude=float(scene['disturbance_amplitude_nm'])
    if not np.isfinite(amplitude):
        raise ValueError('Nonfinite disturbance amplitude')
    # Reference coordinates always describe the OBJECT CENTER, not flange.
    tcp=PayloadTCP(payload,torch.device('cpu'),torch.float64)
    ref=make_reference(tcp,q0,n,DT,scene)
    opts=options(seed,case,'margin_capture')
    opts['tracker_class']=PayloadD0Tracker if case=='D0' else PayloadFinalTracker
    opts['tracker_options'].update(payload=payload,
        payload_collision=bool(settings['payload_collision']),
        dob_cutoff_hz=float(settings['q_cutoff_hz']),
        dob_acceleration_lpf_cutoff_hz=float(settings['acceleration_lpf_hz']))
    # Explicit opt-in only. Existing payload/legacy presets take the same path.
    stabilization=settings.get('experimental_stabilization')
    if stabilization is not None:
        from controllers.payload_stabilized_mppi import PayloadStabilizedTracker,PayloadStabilizedD0Tracker
        allowed={'posture_weight','boundary_weight','smooth_weight','diagnostic_mask_base_upper_arm','diagnostic_sensor_noise_off','reflected_inertia_feedforward','align_running_reference'}
        if set(stabilization)-allowed:raise ValueError('Unknown experimental stabilization option')
        opts['tracker_class']=PayloadStabilizedD0Tracker if case=='D0' else PayloadStabilizedTracker
        for key in ['posture_weight','diagnostic_mask_base_upper_arm','boundary_weight','reflected_inertia_feedforward','align_running_reference']:
            if key in stabilization:opts['tracker_options'][key]=stabilization[key]
        if 'smooth_weight' in stabilization:opts['smooth_weight']=stabilization['smooth_weight']
        if stabilization.get('diagnostic_sensor_noise_off',False):
            opts['tracker_options']['sensor_noise'].update(position_std_rad=0.,velocity_std_rad_s=0.)
    opts.update(use_dob=bool(scene['use_dob']),n_samples=int(settings['samples']),
        horizon=int(settings['horizon']),temperature=float(settings['temperature']),
        noise_variance=settings['noise_variance'])
    destination=(Path(output) if output is not None else Path(config['output_root'])/
        f'{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}_{case}_seed{seed}')
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    resolved=dict(config,case=case,seed=seed,actual_steps=n,
        scene=dict(scene,duration_s=n*DT),controller=dict(settings,case=case),
        payload=dict(mass_kg=payload.mass_kg,side_m=payload.side_m,center_in_flange_m=payload.center_in_flange_m),
        payload_derived=payload.to_dict(),cuda=cuda)
    (destination/'resolved_config.json').write_text(json.dumps(resolved,indent=2,allow_nan=False))
    result=run_mppi(ref,initial_q=q0,has_disturbance=bool(amplitude),
        disturbance_function=raised_cosine_load(scene['disturbance_joint']-1,amplitude),
        capture_samples=capture_samples,sample_count=sample_count,**opts)
    for key in ['q','qd','ee','err_mm','u_mppi','u_applied','d_hat_used']:
        if not np.isfinite(result[key]).all():
            raise RuntimeError(f'Nonfinite payload trajectory:{key}; do not audit as valid geometry')
    result.update(tcp_reference=ref,circle_reference=ref,payload_config_json=np.asarray(json.dumps(payload.to_dict())))
    assert bool(result['payload_models_shared'])
    requested=result['u_mppi']+(result['d_hat_used'] if scene['use_dob'] else 0)
    result.update(u_requested=requested,u_clamp_loss=requested-result['u_applied'],
                  payload_use_dob=np.asarray(bool(scene['use_dob'])))
    if result.get('_animation_samples'):
        samples=result['_animation_samples']
        result.update(payload_sample_ee=np.stack([s['ee'] for s in samples]),
                      payload_sample_omega=np.stack([s['omega'] for s in samples]),
                      payload_sample_mean=np.stack([s['ee_mean'] for s in samples]))
    np.testing.assert_allclose(result['u_applied'],np.clip(requested,-np.asarray(TAU_MAX),TAU_MAX),atol=3e-6,rtol=1e-6)
    q=result['q']
    expected=np.array([tcp.position_and_jacobian_np(qi)[0] for qi in q])
    np.testing.assert_allclose(result['ee'],expected,atol=1e-6,rtol=1e-5)
    np.testing.assert_allclose(result['err_mm'],np.linalg.norm(expected[1:]-np.vstack((ref[1:],ref[-1:])),axis=1)*1000,atol=3e-4,rtol=1e-3)
    result['payload_cube_vertices']=tcp.vertices(torch.as_tensor(q,dtype=torch.float64)).numpy()
    result['flange_position']=tcp.flange_transform(torch.as_tensor(q,dtype=torch.float64))[:,:3,3].numpy()
    path=destination/'trajectory.npz'
    np.savez_compressed(path,**{k:v for k,v in result.items() if not k.startswith('_')})
    robot=dense_mesh_check((str(path),4))
    cube=audit_cube_trajectory(q,payload)
    verification=dict(cuda=cuda,case=case,steps=n,
        tcp_fk_error_max_m=float(np.abs(result['ee']-expected).max()),
        tcp_tracking_rmse_mm=float(np.sqrt(np.mean(result['err_mm']**2))),
        shared_payload_model=True,robot_mesh=robot,payload_mesh=cube,
        no_position_violation=not bool(result['joint_position_limit_violated'].any()),
        no_velocity_violation=not bool(result['joint_velocity_limit_violated'].any()),
        actual_torque_abs_max_per_joint_nm=np.abs(result['u_applied']).max(0).tolist(),
        trajectory_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        meaning='Implementation check only; not a tuned saturation scenario or N0/N1/D0 research validation.')
    (destination/'verification.json').write_text(json.dumps(verification,indent=2,allow_nan=False))
    print(json.dumps(verification,indent=2));print('[OUTPUT]',destination,flush=True)
    return result,verification,destination


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/payload_cube_tcp.yaml'))
    p.add_argument('--case',choices=['N0','N1','D0'])
    p.add_argument('--seed',type=int,default=6807)
    p.add_argument('--mass-kg',type=float)
    p.add_argument('--steps',type=int,help='Short implementation smoke check; does not tune a scenario')
    p.add_argument('--dob',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--amplitude',type=float)
    p.add_argument('--joint',type=int)
    p.add_argument('--output',type=Path)
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text())
    if args.mass_kg is not None:cfg['payload']['mass_kg']=args.mass_kg
    if args.dob is not None:cfg['scene']['use_dob']=args.dob
    if args.amplitude is not None:cfg['scene']['disturbance_amplitude_nm']=args.amplitude
    if args.joint is not None:cfg['scene']['disturbance_joint']=args.joint
    run_payload(cfg,args.seed,args.case,args.steps,args.output)


if __name__=='__main__':
    main()
