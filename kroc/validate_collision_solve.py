"""Same CUDA RNG/state/center MPPI solve parity, including optional fullgraph."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.search_circle_floor25 import CircleTracker,CASES,NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path(
        'kroc/results/circle_collision_cost_boundary_pilot/s1_seed7_directional25_5.npz'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();cuda=cuda_preflight();a.output.mkdir(parents=True,exist_ok=False)
    z=np.load(a.input);device=torch.device('cuda')
    cfg=next(v for k,v in CASES.items() if NAMES[k]=='directional25_5')
    tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.float32)
    ref=z['circle_reference'];q0=tensor(z['q'][0]);results={};baseline={}
    for name,backend,compiled in [('reference','reference',False),
                                   ('triton','triton',False),('triton_fullgraph','triton',True)]:
        torch.manual_seed(7)
        tracker=CircleTracker(ref,device,torch.float32,n_samples=136,horizon=10,
            temperature=1.,temporal_noise_rho=.8,
            joint_position_limit_weight=50000.,joint_velocity_limit_weight=50000.,
            margin_config=asdict(cfg),collision_weight=50000.,collision_backend=backend,
            compile_rollout_loop=compiled,
            sensor_noise={'position_std_rad':.001,'velocity_std_rad_s':.01,'seed':80007},
            dob_acceleration_lpf_cutoff_hz=10.,dob_cutoff_hz=8.)
        ff=tracker._feedforward_torque(q0)
        tracker._warm_up_controller(tensor(np.r_[z['q'][0],z['qd'][0]]),ff,'feedforward_residual')
        results[name]={}
        for step in (0,100,300):
            state=tensor(z['measured_state'][step]);center=tensor(z['kroc_center_horizon'][step])
            tracker.step=step;tracker._sync_cost_step();tracker.margins.reset()
            tracker.tracking_cost.previous_action.zero_()
            tracker.dob.d_hat.copy_(tensor(z['d_hat_used'][step]))
            tracker._prepare_solve(state,center)
            tracker.ctrl.U.copy_(center.clamp(tracker.dynamics.u_min,tracker.dynamics.u_max))
            torch.manual_seed(1234+step)
            action=tracker._mppi_command(state);torch.cuda.synchronize()
            outputs={'action':action.clone(),'states':tracker.ctrl.states.clone(),
                     'cost':tracker.ctrl.cost_total.clone(),'weights':tracker.ctrl.omega.clone()}
            if name=='reference':baseline[step]=outputs
            differences={key:float((value-baseline[step][key]).abs().max())
                         for key,value in outputs.items()}
            # A tiny floating-point perturbation in large costs can affect
            # lambda=1 importance weights. Check the resulting torque directly.
            torch.testing.assert_close(outputs['action'],baseline[step]['action'],rtol=0,atol=.02)
            torch.testing.assert_close(outputs['states'],baseline[step]['states'],rtol=2e-5,atol=2e-5)
            results[name][step]=differences
            print(name,step,differences,flush=True)
        del tracker
    result={'cuda':cuda,'input':str(a.input),'action_absolute_tolerance_nm':.02,
            'status':'passed','max_abs_differences':results}
    (a.output/'results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
