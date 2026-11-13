"""Summarize existing CUDA closed-loop artifacts without generating a report."""
import argparse
import json
from pathlib import Path
import numpy as np
from config import DT,TIMING_WARMUP_STEPS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
    for directory in a.inputs:
        manifest=json.loads((directory/'manifest.json').read_text())
        assert manifest['cuda']['device']=='cuda'
        args=manifest['arguments']
        for r in json.loads((directory/'metrics.json').read_text()):
            path=directory/f"s{r['scenario']}_seed{r['seed']}_{r['case']}.npz"
            z=np.load(path);t=z['solve_t_raw'][TIMING_WARMUP_STEPS:]*1000
            cycle=z['control_cycle_t_raw'][TIMING_WARMUP_STEPS:]*1000
            m=r['metrics']
            rows.append({'directory':str(directory),'file':str(path),'case':r['case'],
                'seed':r['seed'],'backend':args['collision_backend'],
                'compile_rollout_loop':args['compile_rollout_loop'],
                'mean_ms':float(t.mean()),'p95_ms':float(np.percentile(t,95)),
                'p99_ms':float(np.percentile(t,99)),'max_ms':float(t.max()),
                'deadline_miss_rate_20ms':float((t>DT*1000).mean()),
                'simulation_cycle_mean_ms':float(cycle.mean()),
                'tracking_rmse_mm':m['tracking_rmse_mm'],
                'position_violation_steps':m['position_violation_steps'],
                'self_collision_states':m['self_collision_states'],
                'floor_collision_states':m['floor_collision_states'],
                'saturation_steps':m['saturation_steps']})
    for r in rows:
        base=next(x for x in rows if x['backend']=='reference'
                  and x['case']==r['case'] and x['seed']==r['seed'])
        r['time_reduction_percent_vs_reference']=100*(1-r['mean_ms']/base['mean_ms'])
    result={'timing':'synchronized MPPI solve including margin update; first 5 steps excluded',
            'simulation_cycle_note':'includes simulated plant RK4 and diagnostics, not hardware latency',
            'geometry_and_penalty_unchanged':True,'headroom_enabled':False,'rows':rows}
    (a.output/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
