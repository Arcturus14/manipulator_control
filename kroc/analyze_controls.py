"""Causal controls and sensitivity of the payload saturation study. CUDA only.

 python -m kroc.analyze_controls                         # all suites, about 50 min
 python -m kroc.analyze_controls --suites causes

causes       interventions for the two interpretations of README 10.3-10.4
sensitivity  K/H/lambda around the fixed values on tuning seeds (post-hoc, not a selection)

Interventions are diagnostics. None of them changes the N0/N1/D0 definitions.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys

import numpy as np

from config import DT
from kroc.evidence_common import (EVIDENCE,TEST_SEEDS,TUNING_SEEDS,load_config,
                                  provenance,simulate,write_json)

ROOT=EVIDENCE/'controls_20261123'
RAW=Path('kroc/results/controls_20261123')
SUITES=('causes','sensitivity')


def conditions(suite):
    """variant -> settings and the (case, |disturbance|) runs. One process per variant."""
    if suite=='causes':
        zero=[('N0',0.)]
        return {
            'dob_off':dict(runs=zero,use_dob=False),
            'dob_off_noiseless':dict(runs=zero,use_dob=False,noiseless=True),
            'lpf6':dict(runs=zero),
            'lpf12':dict(runs=zero,lpf=12.),
            'lpf_off':dict(runs=zero,lpf=0.),
            'lpf6_noiseless':dict(runs=zero,noiseless=True),
            'lpf_off_noiseless':dict(runs=zero,lpf=0.,noiseless=True),
            'unchanged':dict(runs=[(c,a) for a in (45.,47.5) for c in ('N0','N1','D0')]),
            'previous_projected':dict(runs=[(c,a) for a in (45.,47.5) for c in ('N0','N1')],project=('previous',)),
            'center_projected':dict(runs=[('N0',45.),('N0',47.5)],project=('center',)),
            'center_previous_projected':dict(runs=[('N0',45.),('N0',47.5)],project=('center','previous')),
            'mean_only':dict(runs=[('D0',45.),('D0',47.5)],mean_only=True),
            'boundary_weight_zero':dict(runs=zero+[(c,45.) for c in ('N0','N1','D0')],boundary=0.)}
    runs=[('N0',0.)]+[(c,a) for a in (45.,47.5) for c in ('N0','N1','D0')]
    variants={'K136_H10_lambda64':dict(runs=runs)}
    for k in (256,512,1024,2048):variants[f'K{k}_H10_lambda64']=dict(runs=runs,samples=k)
    for h in (6,8,9,11,12,16):variants[f'K136_H{h}_lambda64']=dict(runs=runs,horizon=h)
    for value in (32,80,96,128):variants[f'K136_H10_lambda{value}']=dict(runs=runs,temperature=float(value))
    return variants


def intervention(base,settings):
    project=settings.get('project',())

    class Intervened(base):
        def _warm_up_controller(self,*args):
            super()._warm_up_controller(*args)
            self.eta_history=[]
            weighting=self.ctrl._compute_weighting

            def record(cost):
                value=weighting(cost)
                self.eta_history.append(self.ctrl.cost_total_non_zero.sum().detach())
                return value

            self.ctrl._compute_weighting=record

        def _prepare_solve(self,state,center):
            lower=self.dynamics.u_min-self.dob.d_hat;upper=self.dynamics.u_max-self.dob.d_hat
            if 'center' in project:self.ctrl.U.copy_(self.ctrl.U.clamp(lower,upper))
            if 'previous' in project:
                self.tracking_cost.previous_action=self.tracking_cost.previous_action.clamp(lower,upper)
            return super()._prepare_solve(state,center)

        def _bound_sample_action(self,action):
            return action if settings.get('mean_only') else super()._bound_sample_action(action)

        def export_sensor_data(self):
            import torch
            data=super().export_sensor_data()
            data['eta']=torch.stack(self.eta_history[-self.episode_steps:]).cpu().numpy()
            return data
    return Intervened


def run_variant(suite,variant,seeds):
    settings=conditions(suite)[variant];classes={};rows=[]

    def modify(options):
        base=options['tracker_class']
        options['tracker_class']=classes.setdefault(base,intervention(base,settings))
        if settings.get('noiseless'):
            options['tracker_options']['sensor_noise'].update(position_std_rad=0.,velocity_std_rad_s=0.)

    for case,amplitude in settings['runs']:
        for seed in seeds:
            config=load_config();controller=config['controller']
            config['scene'].update(disturbance_amplitude_nm=-amplitude,use_dob=settings.get('use_dob',True))
            for key,name in (('lpf','acceleration_lpf_hz'),('samples','samples'),('horizon','horizon'),
                             ('temperature','temperature')):
                if key in settings:controller[name]=settings[key]
            if 'boundary' in settings:controller['experimental_stabilization']['boundary_weight']=settings['boundary']
            z=simulate(config,seed,case,modify)
            window=slice(250,750);error=z['err_mm'][window];ess=z['mppi_weight_ess'][window]
            estimate=z['d_hat_used'][:,1];truth=z['d_true'][1:len(estimate)+1,1]
            requested=z['u_mppi'][:,1]+(estimate if settings.get('use_dob',True) else 0.)
            raw=RAW/suite/variant/f'{case}_a{amplitude:g}_seed{seed}.npz'
            raw.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(raw,**{k:z[k] for k in ('q','err_mm','u_mppi','u_applied','d_hat_used')})
            rows.append(dict(variant=variant,case=case,amplitude_abs_nm=amplitude,seed=seed,raw=str(raw),
                rmse_mm=float(np.sqrt(np.mean(error**2))),max_mm=float(error.max()),
                d_hat_j2_rms_nm=float(np.sqrt(np.mean(estimate[window]**2))),
                d_error_j2_rms_nm=float(np.sqrt(np.mean((estimate-truth)[window]**2))),
                d_error_j2_ramp_rms_nm=float(np.sqrt(np.mean((estimate-truth)[25:150]**2))),
                torque_rate_j2_rms_nm_s=float(np.sqrt(np.mean(np.diff(z['u_applied'][window,1])**2))/DT),
                u_mppi_j2_min_nm=float(z['u_mppi'][window,1].min()),
                j2_clip_percent=float(np.mean(np.abs(requested[window])>150.+1e-5)*100),
                ess_mean=float(ess.mean()),ess_median=float(np.median(ess)),
                ess_p05=float(np.percentile(ess,5)),ess_below7_percent=float(np.mean(ess<7)*100),
                eta_median=float(np.median(z['eta'][window])),
                solve_mean_ms=float(z['solve_t_raw'][10:].mean()*1000),
                q3_min_rad=float(z['q'][:,2].min()),elbow_ok=bool(z['q'][:,2].min()>.2),
                limits_ok=not bool(z['joint_position_limit_violated'].any() or z['joint_velocity_limit_violated'].any())))
        print('RUN',suite,variant,case,amplitude,f'{np.mean([r["rmse_mm"] for r in rows[-len(seeds):]]):.3f}',flush=True)
    write_json(RAW/suite/f'{variant}.rows.json',rows)


def collision_free(path):
    from dynamics.ur5_payload import CubePayload
    from kroc.audit_circle_validation import dense_mesh_check
    from kroc.payload_geometry import audit_cube_trajectory
    robot=dense_mesh_check((path,4))
    with np.load(path) as z:cube=audit_cube_trajectory(z['q'],CubePayload(**load_config()['payload']))
    return not (robot['self_collision_states'] or robot['floor_collision_states'] or cube['payload_collision_states'])


def summarize(suite,seeds,workers):
    rows=[r for v in conditions(suite) for r in json.loads((RAW/suite/f'{v}.rows.json').read_text())]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        checked=list(pool.map(collision_free,[r['raw'] for r in rows]))
    for row,free in zip(rows,checked):
        row['collision_free']=free;row['valid']=bool(free and row['elbow_ok'] and row['limits_ok'])
        row.pop('raw')
    table=[]
    for key in dict.fromkeys((r['variant'],r['case'],r['amplitude_abs_nm']) for r in rows):
        group=[r for r in rows if (r['variant'],r['case'],r['amplitude_abs_nm'])==key]
        mean=lambda name:float(np.mean([r[name] for r in group]))
        item=dict(variant=key[0],case=key[1],amplitude_abs_nm=key[2],runs=len(group),
            valid=sum(r['valid'] for r in group),
            rmse_mean_mm=mean('rmse_mm'),rmse_std_mm=float(np.std([r['rmse_mm'] for r in group],ddof=1)))
        item.update({name:mean(name) for name in ('max_mm','d_hat_j2_rms_nm','d_error_j2_rms_nm',
            'd_error_j2_ramp_rms_nm','torque_rate_j2_rms_nm_s','u_mppi_j2_min_nm','j2_clip_percent',
            'ess_mean','ess_median','ess_p05','ess_below7_percent','eta_median','solve_mean_ms')})
        table.append(item)
        print(f"{suite:11s} {key[0]:30s} {key[1]} {key[2]:5g} {item['rmse_mean_mm']:6.2f}+-{item['rmse_std_mm']:.2f}"
              f" max {item['max_mm']:6.2f} dhat {item['d_hat_j2_rms_nm']:6.2f} eta {item['eta_median']:5.1f}"
              f" valid {item['valid']}/{item['runs']}",flush=True)
    write_json(ROOT/f'{suite}.json',dict(suite=suite,seeds=seeds,window_s=[5,15],
        provenance=provenance(Path('kroc/analyze_controls.py')),table=table,runs=rows))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--suites',nargs='+',choices=SUITES,default=SUITES)
    p.add_argument('--workers',type=int,default=20)
    p.add_argument('--group',nargs=2,help=argparse.SUPPRESS)
    a=p.parse_args()
    seeds=lambda suite:TUNING_SEEDS if suite=='sensitivity' else TEST_SEEDS
    if a.group:
        run_variant(*a.group,seeds(a.group[0]));return
    for suite in a.suites:
        for variant in conditions(suite):
            subprocess.run([sys.executable,'-m','kroc.analyze_controls','--group',suite,variant],check=True)
        summarize(suite,seeds(suite),a.workers)


if __name__=='__main__':main()
