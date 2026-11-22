"""N0/N1/D0 validation of the time-aligned payload circle preset. CUDA only.

 python -m kroc.validate_aligned_circle                   # 168 runs, about 70 min
 python -m kroc.validate_aligned_circle --seeds 7207 --output <new directory>

Existing trajectories in --output are reused, never overwritten. One process per
controller class keeps torch.compile below its recompile limit (past it the
kernels fall back to eager execution and float32 rounding changes).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys

import numpy as np
import yaml

from config import DT
from kroc.view_payload_tcp import evaluation_metrics

CONFIG=Path('configs/payload_circle_T2_J2_45_n0.yaml')
OUTPUT=Path('kroc/results/aligned_circle_20261122')
SEEDS=[7207+10*i for i in range(10)]
CASES=('N0','N1','D0')


def jobs(seeds):
    """name, case, seed, |disturbance|, DOB, aligned, variant, duration."""
    for s in seeds:
        yield 'original_dob_off','N0',s,0.,False,False,'',16.
        yield 'original_dob_on','N0',s,0.,True,False,'',16.
        yield 'aligned_dob_off','N0',s,0.,False,True,'',16.
        yield 'aligned_dob_on','N0',s,0.,True,True,'',16.
        for c in CASES:
            yield 'original_45',c,s,45.,True,False,'',16.
            yield 'aligned_45',c,s,45.,True,True,'',16.
            yield 'aligned_47p5',c,s,47.5,True,True,'',16.
        yield 'aligned_45_unclamped','N0',s,45.,True,True,'unclamped',16.
        yield 'aligned_45_mean_only','D0',s,45.,True,True,'mean_only',16.
    for s in seeds[:3]:
        for c in CASES:
            yield 'aligned_45_lowfreq',c,s,45.,True,True,'lowfreq',16.
            yield 'aligned_45_long',c,s,45.,True,True,'',40.


def group(job):
    _,case,_,_,_,aligned,variant,_=job
    return f"{'aligned' if aligned else 'original'}-{variant or 'main'}-{'D0' if case=='D0' else 'N'}"


def folder(root,job):
    return root/job[0]/f'{job[1]}_seed{job[2]}'


class Simulated(Exception):
    """Carries the raw result out before the slow serial mesh audit of run_payload."""


def run_group(base,root,selected):
    from kroc import run_payload_cube_tcp as runner
    simulate=runner.run_mppi;classes={}
    for job in selected:
        name,case,seed,amplitude,dob,aligned,variant,duration=job
        path=folder(root,job)
        if (path/'trajectory.npz').exists():continue
        cfg=copy.deepcopy(base);cfg['validation_variant']=variant or 'none'
        cfg['scene'].update(use_dob=dob,disturbance_amplitude_nm=-amplitude,duration_s=duration)
        cfg['controller']['experimental_stabilization']['align_running_reference']=aligned

        def intercept(reference,**options):
            tracker=options['tracker_class']
            if variant in ('unclamped','mean_only'):
                override=({'_apply_actuator_limits':lambda self,requested:requested} if variant=='unclamped'
                          else {'_bound_sample_action':lambda self,action:action})
                options['tracker_class']=classes.setdefault((tracker,variant),type(variant,(tracker,),override))
            if variant=='lowfreq':
                load=options['disturbance_function']
                options['disturbance_function']=lambda k,d,t:load(k,d,t)*(1+.05*np.sin(2*np.pi*.1*k*DT))
            raise Simulated(simulate(reference,**options))

        runner.run_mppi=intercept
        try:
            runner.run_payload(cfg,seed,case,output=path)
        except Simulated as done:
            result=done.args[0]
        finally:
            runner.run_mppi=simulate
        np.savez_compressed(path/'trajectory.npz',**{k:result[k] for k in (
            'q','qd','ee','err_mm','u_mppi','u_applied','d_hat_used','solve_t_raw',
            'joint_position_limit_violated','joint_velocity_limit_violated')})
        print('DONE',name,case,seed,flush=True)


def audit(task):
    path,payload=task
    from dynamics.ur5_payload import CubePayload
    from kroc.audit_circle_validation import dense_mesh_check
    from kroc.payload_geometry import audit_cube_trajectory
    robot=dense_mesh_check((str(path),4))
    with np.load(path) as z:cube=audit_cube_trajectory(z['q'],CubePayload(**payload))
    return not (robot['self_collision_states'] or robot['floor_collision_states'] or cube['payload_collision_states'])


def summarize(base,root,planned,workers):
    paths=[folder(root,job)/'trajectory.npz' for job in planned]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        clear=list(pool.map(audit,[(p,base['payload']) for p in paths]))
    rows=[]
    for job,path,collision_free in zip(planned,paths,clear):
        window=dict(base['evaluation'],end_s=39. if job[-1]==40. else base['evaluation']['end_s'])
        with np.load(path) as z:
            row=evaluation_metrics(z,window)
            limits=not bool(z['joint_position_limit_violated'].any() or z['joint_velocity_limit_violated'].any())
        row.update(name=job[0],case=job[1],seed=job[2],amplitude_abs_nm=job[3],
            collision_free=collision_free,valid=bool(collision_free and limits and row['elbow_ok']))
        rows.append(row)
    table=[]
    for name,case in dict.fromkeys((r['name'],r['case']) for r in rows):
        selected=[r for r in rows if (r['name'],r['case'])==(name,case)]
        e=np.array([r['eval_rmse_mm'] for r in selected])
        table.append(dict(name=name,case=case,runs=len(e),valid=sum(r['valid'] for r in selected),
            rmse_mean_mm=float(e.mean()),rmse_std_mm=float(e.std(ddof=1)) if len(e)>1 else 0.,
            max_error_mean_mm=float(np.mean([r['eval_max_mm'] for r in selected]))))
    (root/'summary.json').write_text(json.dumps(dict(table=table,runs=rows),indent=2,allow_nan=False))
    for r in table:
        print(f"{r['name']:24s} {r['case']:3s} RMSE {r['rmse_mean_mm']:6.2f} +/- {r['rmse_std_mm']:.2f}"
              f"  max {r['max_error_mean_mm']:6.2f}  valid {r['valid']}/{r['runs']}")
    print('[OUTPUT]',root,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config',type=Path,default=CONFIG)
    p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--group',help=argparse.SUPPRESS)
    a=p.parse_args();base=yaml.safe_load(a.config.read_text());planned=list(jobs(a.seeds))
    if a.group:
        run_group(base,a.output,[j for j in planned if group(j)==a.group]);return
    missing=[j for j in planned if not (folder(a.output,j)/'trajectory.npz').exists()]
    for name in dict.fromkeys(group(j) for j in missing):
        subprocess.run([sys.executable,'-m','kroc.validate_aligned_circle','--config',str(a.config),
            '--output',str(a.output),'--group',name,'--seeds',*map(str,a.seeds)],check=True)
    summarize(base,a.output,planned,a.workers)


if __name__=='__main__':main()
