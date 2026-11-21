"""Cost-only N1 redesign: penalize actual lost torque, retain nominal rollout.

Do not change plant bounds, sampler, tracking cost, or frozen experiments.
No success-based disturbance/seed changes, no GUI updates, no ZIP creation.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from config import DT,TAU_MAX,Q_MIN,Q_MAX
from experiment_runner import run_mppi
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.run_pre_input_baseline import load_preset,PRESET
from kroc.run_input_headroom_study import run_options
from kroc.position_input_headroom_study import PositionInputTracker,metrics,usable,raised_cosine_load
from kroc.compare_covariance_headroom import joint_metrics
from kroc.final_input_headroom_checks import window_metrics
from kroc.search_lowfreq_estimator_margin import geometry
from kroc.audit_circle_validation import dense_mesh_check

ROOT=Path('kroc/results/n1_clipping_cost_redesign_20261116')
PILOT=[6507,6517]
HELDOUT=[6607,6617,6627,6637,6647]
WEIGHTS=[1.,10.,100.,1000.]
LOADS={'near132p5':-132.5,'boundary133':-133.}
COVARIANCE=[9.,9.,9.,.3136,.3136,.3136]
TEMPERATURE=64.


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))

def clipping_loss_cost(requested,lower,upper,weight):
    loss=requested-requested.clamp(lower,upper)
    return weight*loss.square().sum(-1)


class ClippingLossN1(PositionInputTracker):
    def __init__(self,*args,clipping_weight=10.,**kwargs):
        if not np.isfinite(clipping_weight) or clipping_weight<0:raise ValueError('Invalid clipping weight')
        kwargs.update(study_case='N1',headroom_weight=0.,headroom_fraction=.95,unclamped_diagnostic=False)
        self.clipping_weight=float(clipping_weight)
        super().__init__(*args,**kwargs)

    def _running_cost(self,state,action,rollout_step):
        cost=super()._running_cost(state,action,rollout_step)
        if self.clipping_weight:
            cost=cost+clipping_loss_cost(action+self._compensation_snapshot,
                self.dynamics.u_min,self.dynamics.u_max,self.clipping_weight)
        return cost

    def export_sensor_data(self):
        data=super().export_sensor_data()
        data.update(n1_cost_version=np.asarray('actual_clipping_loss_Nm_squared'),
            n1_clipping_weight=np.asarray(self.clipping_weight),n1_cost_activation_fraction=np.asarray(1.))
        return data


def spec(name):
    if name=='N0':return dict(kind='N0',weight=0.)
    if name=='legacy_N1':return dict(kind='legacy_N1',weight=10.)
    if name=='zero_regression':return dict(kind='loss_N1',weight=0.)
    if name.startswith('loss_w'):
        w=float(name[6:])
        if w not in WEIGHTS:raise ValueError('Unplanned cost weight')
        return dict(kind='loss_N1',weight=w)
    raise ValueError('Unknown planned variant')


def audit(z,variant,amplitude):
    settings=spec(variant)
    assert str(z['compute_device'])=='cuda'
    assert float(z['mppi_temperature'])==TEMPERATURE
    assert not bool(z['input_study_fixed_clip']) and not bool(z['input_study_saturation_rollout'])
    assert not bool(z['nonphysical_unclamped_diagnostic'])
    if settings['kind']=='loss_N1':
        assert float(z['input_study_headroom_weight'])==0.
        assert float(z['n1_clipping_weight'])==settings['weight']
    else:assert float(z['input_study_headroom_weight'])==settings['weight']
    for key in ['q','qd','u_mppi','u_applied','d_hat_used','err_mm','state_prediction_error']:
        if not np.isfinite(z[key]).all():raise RuntimeError('Nonfinite '+key)
    np.testing.assert_array_equal(z['u_effective_pred'],z['u_mppi'])
    np.testing.assert_allclose(z['u_applied'],np.clip(z['u_mppi']+z['d_hat_used'],-np.asarray(TAU_MAX),TAU_MAX),atol=3e-6,rtol=1e-6)
    np.testing.assert_array_equal(z['mppi_active_noise_variance'],np.broadcast_to(np.asarray(COVARIANCE,np.float32),(300,6)))
    t=np.arange(300)*DT;phase=np.clip((t-.5)/2.,0.,1.)
    d=np.zeros((300,6));d[:,2]=amplitude*.5*(1.-np.cos(np.pi*phase))
    np.testing.assert_allclose(z['d_true'][1:],d,atol=1e-5,rtol=1e-6)
    assert float(z['mppi_temporal_noise_rho'])==1. and float(z['mppi_mean_update_alpha'])==1.


def freeze(cuda):
    ROOT.mkdir(parents=True,exist_ok=True);path=ROOT/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        for file,digest in plan['source_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
        return plan
    prior=json.loads(Path('kroc/results/covariance_headroom_transfer_20261116/plan.json').read_text())
    files=list(prior['source_sha256'])+[str(Path(__file__))]
    for file,digest in prior['source_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    for file in files:
        dst=ROOT/'source'/Path(file).resolve().relative_to(Path.cwd());dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dst)
    plan=dict(cuda=cuda,base=load_preset(),sampler=dict(variance=COVARIANCE,temperature=TEMPERATURE),pilot_seeds=PILOT,heldout_seeds=HELDOUT,
        loads=LOADS,primary='boundary133',secondary='near132p5',weights=WEIGHTS,
        cost='w * sum_j (requested_j - clip(requested_j, physical_lower_j, physical_upper_j))^2; Nm squared, no feasible-input penalty',
        control='Same nominal rollout and unbounded nominal input for all cases; only final actuator clamp. All original costs retained.',
        design_rationale='95% headroom penalizes feasible near-limit torque. Penalize only actual compensation-inclusive clipping loss, with steeper penalty outside bounds.',
        selection='Pilot only: all4 load-seed runs safe; boundary133 mean RMSE <=.98*N0 and Jclip <=.8*N0; near132p5 mean RMSE <=1.05*N0; no individual RMSE >1.1*N0. Rank boundary mean RMSE then Jclip then weight.',
        fallback='If no eligible pilot candidate, freeze one safe diagnostic best by boundary RMSE then Jclip; validation still performed but no pilot success claimed.',
        validation='N0,legacy_N1,selected loss_N1 on two fixed loads x5 fresh seeds; N0/selected on nominal x5. Primary success requires all10 loaded selected runs safe, boundary mean RMSE/Jclip simultaneous improvement, at least4/5 boundary paired simultaneous wins; report secondary and nominal regressions.',
        intended_runs=65,pilot_runs=24,heldout_loaded_runs=30,heldout_nominal_runs=10,zero_weight_regression_runs=1,
        no_default_change=True,no_zip=True,source_sha256={file:hashlib.sha256(Path(file).read_bytes()).hexdigest() for file in files})
    write(path,plan);return plan


def run_one(phase,load,amplitude,variant,seed):
    folder=ROOT/phase/load;folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'{variant}_seed{seed}.npz';meta=path.with_suffix('.json')
    if meta.exists():return json.loads(meta.read_text())
    params=spec(variant);print('RUN',phase,load,variant,seed,flush=True)
    if not path.exists():
        ref,q0,_=geometry(.89);case='N0' if params['kind']=='N0' else 'N1'
        opts=run_options(load_preset(),seed,case,params['weight'] if params['kind']=='legacy_N1' else 0.)
        opts.update(temperature=TEMPERATURE,noise_variance=COVARIANCE,
            tracker_class=ClippingLossN1 if params['kind']=='loss_N1' else PositionInputTracker)
        if params['kind']=='loss_N1':opts['tracker_options']['clipping_weight']=params['weight']
        opts['tracker_options'].update(headroom_fraction=.95,unclamped_diagnostic=False)
        z=run_mppi(ref,initial_q=q0,has_disturbance=bool(amplitude),disturbance_function=raised_cosine_load(2,amplitude),**opts)
        z.update(circle_reference=ref,study_joint=np.asarray(2),study_amplitude_nm=np.asarray(amplitude),baseline_preset=np.asarray(str(PRESET)))
        audit(z,variant,amplitude);np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')})
        del z;gc.collect();torch.cuda.empty_cache()
    mesh=dense_mesh_check((str(path),4))
    with np.load(path) as z:
        audit(z,variant,amplitude);full=metrics(z);joints=joint_metrics(z)
        windows={label:window_metrics(z,start,end) for label,start,end in [('before',0,.5),('rise',.5,2.5),('plateau',2.5,6.)]}
    row=dict(phase=phase,load=load,amplitude=amplitude,variant=variant,seed=seed,path=str(path),params=params,mesh_5ms=mesh,full_metrics=full,joints=joints,windows=windows)
    row['safe']=usable(row);write(meta,row)
    print('RESULT',phase,load,variant,seed,'RMSE',full['rmse_mm'],'Jclip',full['jclip_nm_s'],'safe',row['safe'],flush=True)
    return row


def select(rows):
    candidates=[]
    for weight in WEIGHTS:
        name=f'loss_w{weight:g}';runs=[r for r in rows if r['variant']==name]
        if not all(r['safe'] for r in runs):continue
        info=dict(variant=name,weight=weight)
        for load in LOADS:
            b=[r for r in rows if r['variant']=='N0' and r['load']==load]
            n=[r for r in runs if r['load']==load]
            info[load]={key:float(np.mean([r['full_metrics'][key] for r in n])/max(np.mean([r['full_metrics'][key] for r in b]),1e-12)) for key in ['rmse_mm','jclip_nm_s']}
        worst=max(r['full_metrics']['rmse_mm']/next(b['full_metrics']['rmse_mm'] for b in rows if b['variant']=='N0' and b['load']==r['load'] and b['seed']==r['seed']) for r in runs)
        info['max_paired_rmse_ratio']=worst
        info['eligible']=(info['boundary133']['rmse_mm']<=.98 and info['boundary133']['jclip_nm_s']<=.8 and info['near132p5']['rmse_mm']<=1.05 and worst<=1.1)
        candidates.append(info)
    if not candidates:raise RuntimeError('No safe redesigned candidate; stop and report')
    eligible=[c for c in candidates if c['eligible']]
    winner=min(eligible or candidates,key=lambda c:(c['boundary133']['rmse_mm'],c['boundary133']['jclip_nm_s'],c['weight']))
    return dict(selected=winner,candidates=candidates,pilot_gate_passed=bool(eligible),frozen_before_validation=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--phase',choices=['all','pilot','validation'],default='all');args=parser.parse_args()
    cuda=cuda_preflight();print('PREFLIGHT',cuda,flush=True);freeze(cuda)
    selection_path=ROOT/'selection.json'
    if args.phase in ['all','pilot']:
        rows=[]
        for load,amp in LOADS.items():
            for seed in PILOT:
                for variant in ['N0','legacy_N1']+[f'loss_w{w:g}' for w in WEIGHTS]:
                    rows.append(run_one('pilot',load,amp,variant,seed));write(ROOT/'pilot_progress.json',rows)
        zero=run_one('regression','near132p5',-132.5,'zero_regression',PILOT[0])
        base=next(r for r in rows if r['variant']=='N0' and r['load']=='near132p5' and r['seed']==PILOT[0])
        with np.load(base['path']) as b,np.load(zero['path']) as z:
            equal={k:bool(np.array_equal(b[k],z[k])) for k in ['q','qd','u_mppi','u_applied','d_hat_used','err_mm']}
        write(ROOT/'zero_weight_regression.json',equal);assert all(equal.values()),'Zero-cost N1 must reproduce N0'
        chosen=select(rows)
        if selection_path.exists():assert json.loads(selection_path.read_text())==chosen
        else:write(selection_path,chosen)
        print('PILOT_SELECTION',chosen,flush=True)
    if args.phase in ['all','validation']:
        chosen=json.loads(selection_path.read_text());variant=chosen['selected']['variant'];rows=[]
        for load,amp in list(LOADS.items())+[('nominal',0.)]:
            for seed in HELDOUT:
                for name in (['N0',variant] if load=='nominal' else ['N0','legacy_N1',variant]):
                    rows.append(run_one('validation',load,amp,name,seed));write(ROOT/'validation_progress.json',rows)
        write(ROOT/'completed.json',dict(pilot=24,regression=1,validation=len(rows),no_retuning=True))
    print('DONE',args.phase,flush=True)


if __name__=='__main__':main()
