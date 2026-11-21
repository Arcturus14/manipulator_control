"""Fixed-parameter confirmation for the manuscript and topic presentation.

Priorities: adjacent loads, nominal/small load, removal/recovery, new primary
seeds. No selection or retuning from these outcomes. All plant runs CUDA-only.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from config import DT
from experiment_runner import run_mppi
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.run_pre_input_baseline import load_preset,PRESET
from kroc.run_input_headroom_study import run_options
from kroc.position_input_headroom_study import PositionInputTracker,audit,metrics,usable,write
from kroc.search_lowfreq_estimator_margin import geometry
from kroc.audit_circle_validation import dense_mesh_check

ROOT=Path('kroc/results/input_headroom_final_checks_20261115')
PREVIOUS=Path('kroc/results/position_input_headroom_20261115')
SEEDS=[5307,5317,5327,5337,5347]
FRACTION=.95
WEIGHT=10.
SCENARIOS={
    'adjacent_131p5':dict(priority=1,amplitude=-131.5,release=False),
    'adjacent_132':dict(priority=1,amplitude=-132.,release=False),
    'adjacent_133':dict(priority=1,amplitude=-133.,release=False),
    'nominal':dict(priority=2,amplitude=0.,release=False),
    'small_50':dict(priority=2,amplitude=-50.,release=False),
    'release_132p5':dict(priority=3,amplitude=-132.5,release=True),
    'primary_132p5':dict(priority=4,amplitude=-132.5,release=False),
}


def load_fraction(time,release=False):
    phase=np.clip((np.asarray(time)-.5)/2.,0.,1.)
    fraction=.5*(1.-np.cos(np.pi*phase))
    if release:
        down=np.clip((np.asarray(time)-3.5)/1.5,0.,1.)
        fraction=fraction*.5*(1.+np.cos(np.pi*down))
    return fraction


def disturbance(spec):
    def function(step,device,dtype):
        value=spec['amplitude']*float(load_fraction(step*DT,spec['release']))
        result=torch.zeros(6,device=device,dtype=dtype);result[2]=value
        return result
    return function


def window_metrics(z,start,end):
    t=np.arange(len(z['u_mppi']))*DT
    mask=(t>=start-1e-9)&(t<end-1e-9)
    if not mask.any():return None
    requested=z['u_mppi'][mask]+z['d_hat_used'][mask]
    loss=requested-z['u_applied'][mask]
    return dict(start_s=start,end_s=end,samples=int(mask.sum()),
        rmse_mm=float(np.sqrt(np.mean(z['err_mm'][mask]**2))),
        max_error_mm=float(z['err_mm'][mask].max()),
        jclip_nm_s=float(np.linalg.norm(loss,axis=1).sum()*DT),
        residual3_rms_nm=float(np.sqrt(np.mean((z['d_hat_used'][mask,2]-z['d_true'][1:][mask,2])**2))))


def freeze(root,cuda):
    root.mkdir(exist_ok=True,parents=True)
    path=root/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        for file,sha in manifest['source_sha256'].items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:
                raise RuntimeError(f'Frozen source changed: {file}')
        return
    prior=json.loads((PREVIOUS/'manifest.json').read_text())
    for file,sha in prior['source_sha256'].items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:
            raise RuntimeError(f'Original confirmation implementation changed: {file}')
    files=[Path(file) for file in prior['source_sha256']]+[Path(__file__)]
    snapshot=root/'source'
    for file in files:
        target=snapshot/file.resolve().relative_to(Path.cwd())
        target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(file,target)
    write(path,dict(cuda=cuda,baseline=load_preset(),parameters=dict(fraction=FRACTION,weight=WEIGHT),
        scenarios=SCENARIOS,new_seeds=SEEDS,old_primary_seeds=[5207,5217,5227,5237,5247],
        total_new_gpu_runs=70,retuning_allowed=False,
        load='joint3 only; raised cosine .5--2.5s; held until6s unless removal begins3.5s and ends5s',
        release_evaluation='down3.5--5s and post-removal5--6s; original full6s circle unchanged',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        interpretation='All paired outcomes retained. Unsafe full-run performance never pooled as valid improvement; report common pre-contact prefixes separately.'))


def run_one(root,name,seed,case):
    folder=root/name;folder.mkdir(exist_ok=True)
    path=folder/f'{case}_seed{seed}.npz';meta=path.with_suffix('.json')
    if meta.exists():return json.loads(meta.read_text())
    print('RUN',name,seed,case,flush=True)
    spec=SCENARIOS[name]
    if not path.exists():
        reference,q0,_=geometry(.89)
        opts=run_options(load_preset(),seed,case,WEIGHT if case=='N1' else 0.)
        opts['tracker_class']=PositionInputTracker
        opts['tracker_options'].update(headroom_fraction=FRACTION,unclamped_diagnostic=False)
        z=run_mppi(reference,initial_q=q0,has_disturbance=bool(spec['amplitude']),
            disturbance_function=disturbance(spec),**opts)
        audit(z,case,False)
        expected=np.zeros((len(z['u_mppi']),6));expected[:,2]=spec['amplitude']*load_fraction(np.arange(len(expected))*DT,spec['release'])
        np.testing.assert_allclose(z['d_true'][1:],expected,atol=1e-5,rtol=1e-6)
        z.update(circle_reference=reference,study_joint=np.asarray(2),
            study_amplitude_nm=np.asarray(spec['amplitude']),final_check_scenario=np.asarray(name),
            release_profile=np.asarray(spec['release']),baseline_preset=np.asarray(str(PRESET)))
        np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')})
        del z;gc.collect();torch.cuda.empty_cache()
    mesh=dense_mesh_check((str(path),4))
    with np.load(path) as z:
        audit(z,case,False);full=metrics(z)
        windows={label:window_metrics(z,start,end) for label,start,end in [
            ('before',0.,.5),('rise',.5,2.5),('plateau',2.5,3.5),('release',3.5,5.),('recovery',5.,6.)]}
        hit=mesh['first_collision_index']
        n=len(z['u_mppi']) if hit is None else max(2,min(len(z['u_mppi']),int(np.floor((hit*.005-DT)/DT))))
        prefix=metrics(z,n)
    row=dict(scenario=name,priority=spec['priority'],joint=2,amplitude_nm=spec['amplitude'],
        release=spec['release'],seed=seed,case=case,fraction=FRACTION,weight=WEIGHT if case=='N1' else 0.,
        path=str(path),mesh_5ms=mesh,full_metrics=full,windows=windows,
        precontact_metrics=prefix,precontact_end_s=n*DT)
    write(meta,row)
    print('RESULT',name,seed,case,'RMSE',full['rmse_mm'],'Jclip',full['jclip_nm_s'],
        'contact',mesh['first_collision_index'],'q/dq',full['position_violation_steps'],full['velocity_violation_steps'],flush=True)
    return row


def pair_rows(rows):
    groups={}
    for row in rows:groups.setdefault((row['scenario'],row['seed']),{})[row['case']]=row
    result=[]
    for (scenario,seed),by in sorted(groups.items()):
        if set(by)!={'N0','N1'}:continue
        b,n=by['N0'],by['N1']
        with np.load(b['path']) as bz,np.load(n['path']) as nz:
            for key in ['circle_reference','d_true','initial_q','sensor_noise_seed']:
                np.testing.assert_array_equal(bz[key],nz[key])
            bitwise=all(np.array_equal(bz[key],nz[key]) for key in ['q','qd','u_mppi','u_applied','d_hat_used'])
            max_input_difference=float(np.abs(bz['u_mppi']-nz['u_mppi']).max())
            common_n=round(min(b['precontact_end_s'],n['precontact_end_s'])/DT)
            common={case:metrics(z,common_n) for case,z in [('N0',bz),('N1',nz)]}
        result.append(dict(scenario=scenario,seed=seed,n0=b,n1=n,
            safe=usable(b) and usable(n),bitwise_equal=bitwise,max_input_difference_nm=max_input_difference,
            rmse_ratio=n['full_metrics']['rmse_mm']/b['full_metrics']['rmse_mm'],
            clip_difference_nm_s=n['full_metrics']['jclip_nm_s']-b['full_metrics']['jclip_nm_s'],
            simultaneous_improvement=n['full_metrics']['rmse_mm']<b['full_metrics']['rmse_mm'] and n['full_metrics']['jclip_nm_s']<b['full_metrics']['jclip_nm_s'],
            common_precontact_end_s=common_n*DT,common_precontact_metrics=common))
    return result


def summarize(root):
    rows=[json.loads(f.read_text()) for name in SCENARIOS for f in sorted((root/name).glob('*.json'))]
    pairs=pair_rows(rows)
    groups=[]
    for name in SCENARIOS:
        pp=[p for p in pairs if p['scenario']==name]
        if not pp:continue
        safe=all(p['safe'] for p in pp)
        group=dict(scenario=name,pairs=len(pp),safe_pairs=sum(p['safe'] for p in pp),
            all_safe=safe,bitwise_pairs=sum(p['bitwise_equal'] for p in pp),
            simultaneous_wins=sum(p['simultaneous_improvement'] and p['safe'] for p in pp),
            whole_run_performance_valid=safe)
        # Keep all full metrics for failure diagnosis, explicitly flagged.
        for case in ['n0','n1']:
            mm=[p[case]['full_metrics'] for p in pp]
            scalar=[key for key,value in mm[0].items() if isinstance(value,(int,float))]
            group[case]=dict(mean={key:float(np.mean([m[key] for m in mm])) for key in scalar},
                std={key:float(np.std([m[key] for m in mm],ddof=1)) if len(mm)>1 else 0. for key in scalar})
        groups.append(group)
    write(root/'summary.json',dict(groups=groups,pairs=pairs,completed_runs=len(rows),planned_runs=70))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--priority',choices=['1','2','3','4','all','summarize'],default='all')
    p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args()
    if args.priority=='summarize':return summarize(args.output)
    cuda=cuda_preflight();print('PREFLIGHT',cuda,flush=True);freeze(args.output,cuda)
    for name,spec in SCENARIOS.items():
        if args.priority!='all' and spec['priority']!=int(args.priority):continue
        for seed in SEEDS:
            for case in ['N0','N1']:
                run_one(args.output,name,seed,case)
                summarize(args.output)
    print('DONE',args.priority,flush=True)


if __name__=='__main__':main()
