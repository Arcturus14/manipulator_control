"""Frozen covariance/temperature transfer and N0/N1 comparison, CUDA only."""
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
from kroc.final_input_headroom_checks import window_metrics
from kroc.search_lowfreq_estimator_margin import geometry
from kroc.audit_circle_validation import dense_mesh_check

ROOT=Path('kroc/results/covariance_headroom_transfer_20261116')
SEEDS=[6307,6317,6327,6337,6347]
R=.95;W=10.;AMPLITUDE=-132.5
CANDIDATE=Path('kroc/reports/normalized_covariance_20261116/candidate_config.json')


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))

def specifications():
    base=load_preset();candidate=json.loads(CANDIDATE.read_text())
    assert candidate['temperature']==64. and np.allclose(candidate['noise_variance'],[9,9,9,.3136,.3136,.3136])
    return dict(old_l1=dict(variance=base['noise_variance'],temperature=1.,cases=['N0','N1']),
        new_l64=dict(variance=candidate['noise_variance'],temperature=64.,cases=['N0','N1']),
        old_l64=dict(variance=base['noise_variance'],temperature=64.,cases=['N0']),
        new_l1=dict(variance=candidate['noise_variance'],temperature=1.,cases=['N0']))


def joint_metrics(z,n=None):
    n=len(z['u_mppi']) if n is None else n
    def rates(a):return np.sqrt(np.mean((np.diff(a[:n],axis=0)/DT)**2,axis=0)).tolist()
    return dict(applied_rate_per_joint_nm_s=rates(z['u_applied']),nominal_rate_per_joint_nm_s=rates(z['u_mppi']),
        compensation_rate_per_joint_nm_s=rates(z['d_hat_used']),
        ik_configuration_deviation_rms_rad=np.sqrt(np.mean((z['q'][1:n+1]-z['q_reference'][1:n+1])**2,axis=0)).tolist(),
        velocity_rms_per_joint_rad_s=np.sqrt(np.mean(z['qd'][1:n+1]**2,axis=0)).tolist())


def audit(z,spec,case):
    assert str(z['compute_device'])=='cuda'
    assert str(z['input_study_case'])==case and not bool(z['input_study_fixed_clip'])
    assert not bool(z['input_study_saturation_rollout']) and not bool(z['nonphysical_unclamped_diagnostic'])
    assert float(z['mppi_temperature'])==spec['temperature']
    assert float(z['mppi_temporal_noise_rho'])==1. and float(z['mppi_mean_update_alpha'])==1.
    assert np.isclose(float(z['input_study_headroom_fraction']),R)
    assert float(z['input_study_headroom_weight'])==(W if case=='N1' else 0.)
    for key in ['q','qd','u_mppi','u_applied','d_hat_used','err_mm','state_prediction_error']:
        if not np.isfinite(z[key]).all():raise RuntimeError('Nonfinite '+key)
    np.testing.assert_array_equal(z['u_effective_pred'],z['u_mppi'])
    np.testing.assert_allclose(z['u_applied'],np.clip(z['u_mppi']+z['d_hat_used'],-np.asarray(TAU_MAX),TAU_MAX),atol=3e-6,rtol=1e-6)
    np.testing.assert_array_equal(z['mppi_active_noise_variance'],np.broadcast_to(np.asarray(spec['variance'],np.float32),(300,6)))
    phase=np.clip((np.arange(300)*DT-.5)/2.,0.,1.)
    expected=np.zeros((300,6));expected[:,2]=AMPLITUDE*.5*(1.-np.cos(np.pi*phase))
    np.testing.assert_allclose(z['d_true'][1:],expected,atol=1e-5,rtol=1e-6)
    span=np.asarray(Q_MAX)-Q_MIN
    np.testing.assert_allclose(z['kroc_margin'][:,0],np.broadcast_to(.025*span,z['kroc_margin'][:,0].shape),atol=2e-6)


def freeze(cuda):
    ROOT.mkdir(exist_ok=True,parents=True);path=ROOT/'plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        for file,digest in plan['source_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
        return plan
    prior=json.loads(Path('kroc/results/input_headroom_final_checks_20261115/manifest.json').read_text())
    files=list(prior['source_sha256'])+[str(Path(__file__)),str(CANDIDATE)]
    for file,digest in prior['source_sha256'].items():assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==digest,file
    for file in files:
        relative=Path(file).resolve().relative_to(Path.cwd());dst=ROOT/'source'/relative
        dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(file,dst)
    plan=dict(cuda=cuda,seeds=SEEDS,specifications=specifications(),baseline=load_preset(),fraction=R,weight=W,
        amplitude_nm=AMPLITUDE,joint_1based=3,rise_s=[.5,2.5],plateau_s=[2.5,6.],total_new_gpu_runs=30,
        selection='No selection or retuning. Candidate remains exploratory; original no-DOB gate was narrowly missed.',
        primary='old/new setting each N0 vs N1; new vs old in N0 only; shared fresh5 seeds',
        secondary='N0 factorial old/new covariance x lambda1/64. Two mixed settings isolate covariance and temperature effects.',
        limits='No generalization to all low frequency signals; no20-30 seed nominal expansion in this stage; no GUI changes.',
        source_sha256={file:hashlib.sha256(Path(file).read_bytes()).hexdigest() for file in files})
    write(path,plan);return plan


def run_one(name,spec,case,seed):
    folder=ROOT/name;folder.mkdir(exist_ok=True)
    path=folder/f'{case}_seed{seed}.npz';meta=path.with_suffix('.json')
    if meta.exists():return json.loads(meta.read_text())
    print('RUN',name,case,seed,flush=True)
    if not path.exists():
        ref,q0,_=geometry(.89);opts=run_options(load_preset(),seed,case,W if case=='N1' else 0.)
        opts.update(temperature=spec['temperature'],noise_variance=spec['variance'],tracker_class=PositionInputTracker)
        opts['tracker_options'].update(headroom_fraction=R,unclamped_diagnostic=False)
        z=run_mppi(ref,initial_q=q0,has_disturbance=True,disturbance_function=raised_cosine_load(2,AMPLITUDE),**opts)
        z.update(circle_reference=ref,study_joint=np.asarray(2),study_amplitude_nm=np.asarray(AMPLITUDE),baseline_preset=np.asarray(str(PRESET)))
        audit(z,spec,case);np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')})
        del z;gc.collect();torch.cuda.empty_cache()
    mesh=dense_mesh_check((str(path),4))
    with np.load(path) as z:
        audit(z,spec,case);full=metrics(z);joints=joint_metrics(z)
        windows={label:window_metrics(z,start,end) for label,start,end in [('before',0,.5),('rise',.5,2.5),('plateau',2.5,6.)]}
        hit=mesh['first_collision_index']
        n=300 if hit is None else max(2,min(300,int(np.floor((hit*.005-DT)/DT))))
        prefix=metrics(z,n)
    row=dict(setting=name,spec=spec,case=case,seed=seed,path=str(path),mesh_5ms=mesh,full_metrics=full,
        joints=joints,windows=windows,precontact_metrics=prefix,precontact_end_s=n*DT)
    row['safe']=usable(row);write(meta,row)
    print('RESULT',name,case,seed,'RMSE',full['rmse_mm'],'Jclip',full['jclip_nm_s'],'ESS',full['ess_mean'],'safe',row['safe'],flush=True)
    return row


def main():
    cuda=cuda_preflight();print('PREFLIGHT',cuda,flush=True);plan=freeze(cuda);rows=[]
    for seed in SEEDS:
        for name,spec in plan['specifications'].items():
            for case in spec['cases']:
                rows.append(run_one(name,spec,case,seed));write(ROOT/'progress.json',rows)
    write(ROOT/'completed.json',dict(runs=len(rows),no_retuning=True));print('DONE',len(rows),flush=True)


if __name__=='__main__':main()
