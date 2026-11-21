"""Reusable entry point for the validated KROC simulation baseline.

Does NOT change GUI/default controller settings, add input tightening, or claim
20 ms realtime operation. Use the frozen preset instead of copying kwargs.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from config import DT,W_ACT,W_VEL
from experiment_runner import run_mppi
from kroc.finalize_pre_input_baseline import BoundaryPaperTracker
from kroc.search_lowfreq_estimator_margin import geometry,smooth_load

PRESET=Path(__file__).resolve().parent/'configs/pre_input_study_baseline.json'


def load_preset(path=PRESET):
    config=json.loads(Path(path).read_text())
    if config['schema_version']!=1 or config['input_tightening']:
        raise ValueError('Unsupported baseline preset or unexpected input tightening')
    if config['dt_s']!=DT:
        raise ValueError('Simulation dt differs from validated preset')
    if config['fixed_effort_weight']!=W_ACT or config['fixed_velocity_weight']!=W_VEL:
        raise ValueError('Non-configurable effort/velocity costs changed since validation')
    return config


def run_baseline(reference,initial_q,*,disturbance_function=None,seed=4607,
                 dob_mode='output',preset=PRESET):
    """Run on CUDA only; callers own scenario design and result storage."""
    c=load_preset(preset)
    if dob_mode not in c['supported_dob_modes']:
        raise ValueError('Unsupported DOB mode')
    result=run_mppi(reference,initial_q=initial_q,
        has_disturbance=disturbance_function is not None,use_dob=True,
        dob_compensation_mode=dob_mode,sampling_center_mode=c['sampling_center_mode'],
        disturbance_function=disturbance_function,tracker_class=BoundaryPaperTracker,
        tracker_options=dict(proposal=c['proposal'],boundary_weight=c['boundary_weight'],
            margin_config=c['state_margin'],
            sensor_noise=dict(position_std_rad=c['sensor_position_std_rad'],
                velocity_std_rad_s=c['sensor_velocity_std_rad_s'],seed=98000+seed),
            dob_acceleration_lpf_cutoff_hz=c['dob_acceleration_lpf_hz'],dob_cutoff_hz=c['dob_cutoff_hz'],
            collision_weight=c['collision_weight'],collision_clearance=c['collision_clearance_m'],
            collision_backend=c['collision_backend']),
        require_cuda=True,seed=seed,temperature=c['temperature'],temporal_noise_rho=1.,
        mean_update_alpha=c['mean_update_alpha'],smooth_weight=c['horizon_smooth_weight'],
        n_samples=c['samples'],horizon=c['horizon'],noise_variance=c['noise_variance'],
        position_weight=c['position_weight'],terminal_weight=c['terminal_weight'],
        joint_position_limit_weight=c['joint_position_limit_weight'],
        joint_velocity_limit_weight=c['joint_velocity_limit_weight'],compile_rollout_loop=False)
    result['u_requested']=result['u_mppi']+(result['d_hat_used'] if dob_mode=='output' else 0.)
    result['u_clamp_loss']=result['u_requested']-result['u_applied']
    result['baseline_preset']=np.asarray(str(preset))
    result['baseline_boundary_weight']=np.asarray(c['boundary_weight'])
    result['input_tightening_enabled']=np.asarray(False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=['nominal','slow_disturbance'],default='nominal')
    p.add_argument('--dob-mode',choices=['estimate_only','output'],default='output')
    p.add_argument('--seed',type=int,default=4607)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    print('SIMULATION baseline only; NOT certified for realtime 20 ms.')
    ref,q0,_=geometry(.89)
    z=run_baseline(ref,q0,disturbance_function=(smooth_load(5.5) if args.profile=='slow_disturbance' else None),
                   seed=args.seed,dob_mode=args.dob_mode)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**{k:v for k,v in z.items() if not k.startswith('_')})
    print(args.output)


if __name__=='__main__':main()
