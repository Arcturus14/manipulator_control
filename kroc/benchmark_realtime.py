"""Control-computation latency at a 50 Hz release, with every step saved. CUDA only.

 python -m kroc.benchmark_realtime                # about 40 min
 python -m kroc.benchmark_realtime --summarize    # rebuild the table from the saved steps

All times are measured from the scheduled release of the step:

 wake         the loop resumes after waiting (release/wake-up lateness)
 command      MPPI, d_hat compensation and clamp done, torque copied to the CPU
 update_done  DOB update done as well; simulated plant and prediction are subtracted
 loop_ready   whole simulation loop done (plant and diagnostics included)
 release_slip how far the release was postponed because the loop was still busy

The measurement used by a solve is simulated at the end of the previous step, so
this is computation latency after waiting. It is not sensor-to-actuator latency.
The CPU-to-GPU state upload is included; robot communication is not.
"""
import argparse
import gzip
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from config import DT
from kroc.evidence_common import (EVIDENCE,TEST_SEEDS,load_config,provenance,
                                  simulate,write_json)

ROOT=EVIDENCE/'realtime_20261123'
MODES=(('kernel','sleep'),('kernel','busy'),('captured','sleep'))
CASES=('N0','N1','D0')
COLUMNS=('seed','step','release_slip_ms','wake_ms','command_ms','update_done_ms',
         'simulated_ms','loop_ready_ms')


def timed_tracker(base,wait,torch):
    class Timed(base):
        timing=False

        def _warm_up_controller(self,*args):
            super()._warm_up_controller(*args)
            self.rows=[];self.due=None;self.simulated=0.
            for name in ('plant_step_effective','rollout_effective'):
                setattr(self.dynamics,name,self._simulation_only(getattr(self.dynamics,name)))
            update=self.dob.update

            def finish(*arguments):
                value=update(*arguments)
                torch.cuda.synchronize(self.device)
                self.rows.append([self.slip,self.wake-self.release,self.output-self.release,
                                  time.perf_counter()-self.release-self.simulated,
                                  self.simulated,float('nan')])
                return value

            self.dob.update=finish;self.timing=True

        def _simulation_only(self,function):
            def run(*arguments):
                torch.cuda.synchronize(self.device);start=time.perf_counter()
                value=function(*arguments)
                torch.cuda.synchronize(self.device)
                self.simulated+=time.perf_counter()-start
                return value
            return run

        def _set_rollout_disturbance(self,use):
            if self.timing:
                torch.cuda.synchronize(self.device);now=time.perf_counter()
                if self.rows:self.rows[-1][5]=now-self.release
                if self.due is None:self.due=now
                self.slip=max(now-self.due,0.)
                self.release=self.due+self.slip
                if wait=='sleep':
                    if self.release>now:time.sleep(self.release-now)
                else:
                    while time.perf_counter()<self.release:pass
                self.wake=time.perf_counter();self.due=self.release+DT;self.simulated=0.
            return super()._set_rollout_disturbance(use)

        def _prepare_solve(self,state,center):
            state.copy_(torch.from_numpy(state.cpu().numpy()).to(self.device))
            return super()._prepare_solve(state,center)

        def _apply_actuator_limits(self,requested):
            applied=super()._apply_actuator_limits(requested)
            if self.timing:
                applied.cpu().numpy();self.output=time.perf_counter()
            return applied

        def export_sensor_data(self):
            self.timing=False
            data=super().export_sensor_data()
            data['timing_ms']=np.asarray(self.rows[-self.episode_steps:])*1000
            return data
    return Timed


def csv_path(backend,wait,case):
    return ROOT/f'{backend}_{wait}_{case}.csv.gz'


def measure(backend,wait,case,seeds):
    import torch
    classes={}

    def modify(options):
        base=options['tracker_class']
        options['tracker_class']=classes.setdefault(base,timed_tracker(base,wait,torch))

    rows,runs=[],[]
    for seed in seeds:
        config=load_config();config['controller']['rollout_backend']=backend
        result=simulate(config,seed,case,modify)
        for step,values in enumerate(result['timing_ms']):rows.append([seed,step,*values])
        runs.append(dict(seed=seed,
            eval_rmse_mm=float(np.sqrt(np.mean(result['err_mm'][250:750]**2)))))
        print('TIMED',backend,wait,case,seed,flush=True)
    path=csv_path(backend,wait,case);path.parent.mkdir(parents=True,exist_ok=True)
    text=','.join(COLUMNS)+'\n'+''.join(
        f'{seed},{step},'+','.join(f'{v:.4f}' for v in values)+'\n' for seed,step,*values in rows)
    path.write_bytes(gzip.compress(text.encode(),mtime=0))
    write_json(ROOT/f'{backend}_{wait}_{case}.runs.json',
        dict(runs=runs,provenance=provenance(Path('kroc/benchmark_realtime.py'))))


def summarize():
    table=[]
    for backend,wait in MODES:
        for case in CASES:
            path=csv_path(backend,wait,case)
            if not path.exists():continue
            data=np.genfromtxt(gzip.open(path,'rt'),delimiter=',',skip_header=1)
            row=dict(backend=backend,wait=wait,case=case,steps=len(data),
                     seeds=len(np.unique(data[:,0])),
                     release_slip_steps=int((data[:,2]>0.05).sum()))
            for name,column in (('wake',3),('command',4),('update_done',5),('loop_ready',7)):
                x=data[:,column];x=x[np.isfinite(x)]
                row[name]=dict(mean=float(x.mean()),p99=float(np.percentile(x,99)),
                    p999=float(np.percentile(x,99.9)),max=float(x.max()),
                    over_20ms=int((x>20).sum()))
            table.append(row)
            c,u=row['command'],row['update_done']
            print(f"{backend:8s} {wait:5s} {case} n={len(data)}"
                  f" | command {c['mean']:.2f}/{c['p99']:.2f}/{c['max']:.2f} >20={c['over_20ms']}"
                  f" | update {u['mean']:.2f}/{u['p99']:.2f}/{u['max']:.2f} >20={u['over_20ms']}"
                  f" | wake max {row['wake']['max']:.2f} | slip {row['release_slip_steps']}",flush=True)
    write_json(ROOT/'summary.json',dict(unit='ms from scheduled release',
        definition=__doc__,table=table))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds',type=int,nargs='+',default=TEST_SEEDS)
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--group',nargs=3,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.group:
        measure(*a.group,a.seeds);return
    if not a.summarize:
        for backend,wait in MODES:
            for case in CASES:
                subprocess.run([sys.executable,'-m','kroc.benchmark_realtime','--group',backend,wait,case,
                                '--seeds',*map(str,a.seeds)],check=True)
    summarize()


if __name__=='__main__':main()
