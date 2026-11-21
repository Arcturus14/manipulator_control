"""Staged joint2/3 N0--N1 study. Fixed controller, causal nominal rollouts.

The unclamped plant is a labelled, nonphysical diagnostic only. Never a
candidate controller. No production defaults or DOB estimation are changed.
"""
from pathlib import Path
import argparse
import gc
import hashlib
import json
import shutil
import numpy as np
import torch

from config import DT, TAU_MAX, Q_MIN, Q_MAX
from experiment_runner import run_mppi
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.input_headroom_study import InputHeadroomTracker
from kroc.run_input_headroom_study import run_options, audit as previous_audit
from kroc.run_pre_input_baseline import load_preset, PRESET
from kroc.search_lowfreq_estimator_margin import geometry
from kroc.audit_circle_validation import dense_mesh_check

ROOT=Path('kroc/results/position_input_headroom_20261115')
PILOT=5107
HELDOUT=[5207,5217,5227,5237,5247]
FRACTIONS=[.85,.90,.95]
WEIGHTS=[.3,1.,3.,10.,30.]
SCREEN=[(1,a) for a in [-70.,-90.,-105.,-115.,-125.,140.]] + [
    (2,a) for a in [-100.,-120.,-130.,-140.,-150.,145.]]


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False))


def raised_cosine_load(joint,amplitude):
    def disturbance(step,device,dtype):
        phase=np.clip((step*DT-.5)/2.,0.,1.)
        value=amplitude*.5*(1.-np.cos(np.pi*phase))
        result=torch.zeros(6,device=device,dtype=dtype);result[joint]=value
        return result
    return disturbance


class PositionInputTracker(InputHeadroomTracker):
    def __init__(self,*args,unclamped_diagnostic=False,**kwargs):
        self.unclamped_diagnostic=bool(unclamped_diagnostic)
        super().__init__(*args,**kwargs)

    def _apply_actuator_limits(self,requested):
        if self.unclamped_diagnostic:
            return requested
        return super()._apply_actuator_limits(requested)

    def export_sensor_data(self):
        result=super().export_sensor_data()
        result['nonphysical_unclamped_diagnostic']=np.asarray(self.unclamped_diagnostic)
        return result


def burst_length(mask):
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    return int(max(lengths,default=0))


def metrics(z,n=None):
    n=len(z['u_mppi']) if n is None else n
    v=z['u_mppi'][:n];estimate=z['d_hat_used'][:n]
    a=z['u_applied'][:n];requested=v+estimate;loss=a-requested
    true=z['d_true'][1:n+1];residual=estimate-true
    limit=np.asarray(TAU_MAX)
    clipped=(np.abs(loss)>1e-5).any(-1)
    q=z['q'][1:n+1];qref=z['q_reference'][1:n+1]
    violations=np.maximum(np.maximum(q-Q_MAX,np.asarray(Q_MIN)-q),0.)
    pred=z['state_prediction_error'][:n]
    error=z['err_mm'][:n]
    ratio=np.abs(a)/limit
    steady=np.arange(n)*DT>=2.5
    smooth=np.diff(a,axis=0)
    reference_request=z['u_ff'][:n]+true
    reference_excess=np.maximum(np.abs(reference_request)-limit,0.)
    return dict(rmse_mm=float(np.sqrt(np.mean(error**2))),max_error_mm=float(error.max()),
        jclip_nm_s=float(np.linalg.norm(loss,axis=-1).sum()*DT),
        clip_steps=int(clipped.sum()),clip_fraction=float(clipped.mean()),
        longest_clip_s=burst_length(clipped)*DT,
        clip_max_nm=float(np.abs(loss).max()),
        clip_per_joint_s=((np.abs(loss)>1e-5).sum(0)*DT).tolist(),
        prediction_q_rmse_rad=float(np.sqrt(np.mean(pred[:,:6]**2))),
        prediction_dq_rmse_rad_s=float(np.sqrt(np.mean(pred[:,6:]**2))),
        near98_fraction=float((ratio>=.98).any(-1).mean()),
        near95_fraction=float((ratio>=.95).any(-1).mean()),
        near90_fraction=float((ratio>=.90).any(-1).mean()),
        near85_fraction=float((ratio>=.85).any(-1).mean()),
        torque_rate_rms_nm_s=float(np.sqrt(np.mean(smooth**2))/DT),
        joint_configuration_rmse_rad=np.sqrt(np.mean((q-qref)**2,axis=0)).tolist(),
        dob_residual_rms_per_joint_nm=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
        steady_rmse_mm=None if not steady.any() else float(np.sqrt(np.mean(error[steady]**2))),
        steady_dob_residual_rms_per_joint_nm=None if not steady.any() else np.sqrt(np.mean(residual[steady]**2,axis=0)).tolist(),
        steady_clip_rms_per_joint_nm=None if not steady.any() else np.sqrt(np.mean(loss[steady]**2,axis=0)).tolist(),
        position_violation_steps=int((violations>1e-6).any(-1).sum()),
        max_position_excess_rad=float(violations.max()),
        velocity_violation_steps=int(z['joint_velocity_limit_violated'][:n].sum()),
        ess_mean=float(z['mppi_weight_ess'][:n].mean()),
        ess_median=float(np.median(z['mppi_weight_ess'][:n])),
        requested_excess_max_nm=float(np.maximum(np.abs(requested)-limit,0.).max()),
        reference_ff_plus_true_excess_fraction=float((reference_excess>1e-5).any(-1).mean()),
        reference_ff_plus_true_excess_max_nm=float(reference_excess.max()),
        applied_physical_excess_max_nm=float(np.maximum(np.abs(a)-limit,0.).max()),
        solve_mean_ms=float(z['solve_t'][min(20,n-1):n].mean()*1000))


def audit(z,case,diagnostic):
    if not diagnostic:
        previous_audit(z,case)
    else:
        assert str(z['compute_device'])=='cuda'
        assert bool(z['nonphysical_unclamped_diagnostic'])
        np.testing.assert_allclose(z['u_applied'],z['u_mppi']+z['d_hat_used'],atol=3e-6)
        np.testing.assert_array_equal(z['u_effective_pred'],z['u_mppi'])
    for key in ['q','qd','u_applied','d_hat_used','err_mm','state_prediction_error']:
        if not np.isfinite(z[key]).all():
            raise RuntimeError(f'Nonfinite trajectory: {key}; stop study and inspect')


def run_one(root,phase,joint,amplitude,case='N0',seed=PILOT,fraction=.85,weight=0.,diagnostic=False):
    directory=root/phase;directory.mkdir(exist_ok=True)
    label=f'j{joint+1}_a{amplitude:g}_{case}_r{fraction:g}_w{weight:g}_s{seed}'+('_UNCLAMPED' if diagnostic else '')
    path=directory/f'{label}.npz';meta=path.with_suffix('.json')
    if meta.exists():return json.loads(meta.read_text())
    print('RUN',phase,label,flush=True)
    if not path.exists():
        reference,q0,_=geometry(.89)
        opts=run_options(load_preset(),seed,case,weight)
        opts['tracker_class']=PositionInputTracker
        opts['tracker_options'].update(headroom_fraction=fraction,unclamped_diagnostic=diagnostic)
        z=run_mppi(reference,initial_q=q0,has_disturbance=bool(amplitude),
            disturbance_function=raised_cosine_load(joint,amplitude),**opts)
        audit(z,case,diagnostic)
        z.update(circle_reference=reference,study_joint=np.asarray(joint),
            study_amplitude_nm=np.asarray(amplitude),baseline_preset=np.asarray(str(PRESET)))
        np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')})
        del z;gc.collect();torch.cuda.empty_cache()
    mesh=dense_mesh_check((str(path),4))
    with np.load(path) as z:
        audit(z,case,diagnostic)
        full=metrics(z)
        hit=mesh['first_collision_index']
        n=len(z['u_mppi']) if hit is None else max(2,min(len(z['u_mppi']),int(np.floor((hit*.005-DT)/DT))))
        safe=metrics(z,n)
    row=dict(phase=phase,joint=joint,amplitude_nm=amplitude,case=case,seed=seed,
        fraction=fraction,weight=weight,nonphysical_diagnostic=diagnostic,path=str(path),
        mesh_5ms=mesh,full_metrics=full,precontact_metrics=safe,precontact_end_s=n*DT)
    write(meta,row)
    print('RESULT',label,'RMSE',full['rmse_mm'],'Jclip',full['jclip_nm_s'],
          'longest',full['longest_clip_s'],'contact',hit,flush=True)
    return row


def usable(row):
    m=row['full_metrics']
    return row['mesh_5ms']['first_collision_index'] is None and m['position_violation_steps']==0 and m['velocity_violation_steps']==0


def screen_gate(row,nominal):
    m=row['full_metrics'];b=nominal['full_metrics']
    return (usable(row) and .02<=m['clip_fraction']<=.5 and m['longest_clip_s']<=.8
        and m['rmse_mm']>=max(1.2*b['rmse_mm'],b['rmse_mm']+.2))


def freeze(root,cuda):
    root.mkdir(parents=True,exist_ok=True)
    path=root/'manifest.json'
    if path.exists():
        manifest=json.loads(path.read_text())
        for name,sha in manifest['source_sha256'].items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=sha:
                raise RuntimeError(f'Frozen source changed: {name}')
        return
    files=[PRESET,Path(__file__),Path('kroc/input_headroom_study.py'),Path('kroc/run_input_headroom_study.py'),
        Path('controllers/mppi_dob_controller.py'),Path('costs/tracking_cost.py'),
        Path('kroc/finalize_pre_input_baseline.py'),Path('kroc/benchmark_single_instance_mppi.py'),
        Path('kroc/state_margin.py'),Path('kroc/search_circle_floor25.py'),Path('kroc/search_lowfreq_estimator_margin.py'),
        Path('kroc/find_estimator_directional_separation.py'),Path('kroc/sensor_noise.py'),
        Path('dynamics/ur5_dynamics.py'),Path('observers/dob.py'),Path('experiment_runner.py'),Path('config.py')]
    snapshot=root/'source'
    for file in files:
        dst=snapshot/file.resolve().relative_to(Path.cwd());dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(file,dst)
    write(path,dict(cuda=cuda,baseline=load_preset(),pilot_seed=PILOT,heldout_seeds=HELDOUT,
        screen_joint_amplitudes=SCREEN,fractions=FRACTIONS,weights=WEIGHTS,
        scene='unchanged geometry(.89), 6s R=.04m circle',
        load='.5s onset, 2s raised-cosine rise, then held; single joint2 or3',
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        screen_gate='safe, clipping2--50%, max burst<=.8s, RMSE>=max(1.2*nominal,nominal+.2mm)',
        diagnostic_gate='unclamped nonphysical reference recovers >=20% RMSE, safe both; prioritize lower clipping fraction, at most 2 diagnoses',
        sweep_rule='prefer simultaneous >=5% RMSE and >=10% Jclip reduction vs N0, no contact or increased q/dq violations; rank RMSE then Jclip; do not select from heldout',
        nominal_input_costs='same as frozen baseline; unconstrained v',
        production_defaults_changed=False,diagnostic_is_not_a_physical_controller=True))


def rows_in(root,phase):
    return [json.loads(f.read_text()) for f in sorted((root/phase).glob('*.json'))]


def screen(root):
    rows=[run_one(root,'screen',1,0.)]
    nominal=rows[0]
    for joint,amplitude in SCREEN:
        rows.append(run_one(root,'screen',joint,amplitude))
        write(root/'screen_metrics.json',rows)
    candidates=sorted([r for r in rows[1:] if screen_gate(r,nominal)],
        key=lambda r:(r['full_metrics']['clip_fraction'],r['full_metrics']['rmse_mm']))
    diagnoses=[];chosen=None
    for r in candidates[:2]:
        diagnostic=run_one(root,'diagnostic',r['joint'],r['amplitude_nm'],diagnostic=True)
        ratio=diagnostic['full_metrics']['rmse_mm']/r['full_metrics']['rmse_mm']
        accepted=usable(diagnostic) and ratio<=.8
        diagnoses.append(dict(candidate=r,diagnostic=diagnostic,rmse_ratio=ratio,accepted=bool(accepted)))
        if accepted:
            chosen=dict(joint=r['joint'],amplitude_nm=r['amplitude_nm'],n0=r,diagnostic=diagnostic)
            break
    write(root/'condition_selection.json',dict(selected=chosen,nominal=nominal,
        candidates=len(candidates),diagnoses=diagnoses,
        status='condition_selected' if chosen else 'stopped_no_confirmed_recoverable_condition'))


def pareto(records):
    valid=[r for r in records if usable(r)]
    return [r for r in valid if not any(
        s['full_metrics']['rmse_mm']<=r['full_metrics']['rmse_mm']
        and s['full_metrics']['jclip_nm_s']<=r['full_metrics']['jclip_nm_s']
        and (s['full_metrics']['rmse_mm']<r['full_metrics']['rmse_mm'] or s['full_metrics']['jclip_nm_s']<r['full_metrics']['jclip_nm_s'])
        for s in valid)]


def sweep(root):
    condition=json.loads((root/'condition_selection.json').read_text())['selected']
    if not condition:raise RuntimeError('No confirmed recoverable condition; stop before tuning')
    baseline=condition['n0']['full_metrics'];rows=[]
    for fraction in FRACTIONS:
        for weight in WEIGHTS:
            rows.append(run_one(root,'sweep',condition['joint'],condition['amplitude_nm'],'N1',fraction=fraction,weight=weight))
            write(root/'sweep_metrics.json',rows)
    eligible=[r for r in rows if usable(r)
        and r['full_metrics']['rmse_mm']<=.95*baseline['rmse_mm']
        and r['full_metrics']['jclip_nm_s']<=.9*baseline['jclip_nm_s']]
    selected=min(eligible,key=lambda r:(r['full_metrics']['rmse_mm'],r['full_metrics']['jclip_nm_s'])) if eligible else None
    write(root/'parameter_selection.json',dict(selected=selected,pareto=pareto(rows),
        status='parameters_selected' if selected else 'stopped_no_simultaneous_pilot_improvement'))


def validation(root):
    condition=json.loads((root/'condition_selection.json').read_text())['selected']
    parameter=json.loads((root/'parameter_selection.json').read_text())['selected']
    if parameter is None:raise RuntimeError('No simultaneous pilot improvement; stop before validation')
    rows=[]
    for seed in HELDOUT:
        for case in ['N0','N1']:
            rows.append(run_one(root,'validation',condition['joint'],condition['amplitude_nm'],case,seed,
                parameter['fraction'],parameter['weight'] if case=='N1' else 0.))
            write(root/'validation_metrics.json',rows)
    pairs=[]
    for seed in HELDOUT:
        rr={r['case']:r for r in rows if r['seed']==seed};b,n=rr['N0'],rr['N1']
        with np.load(b['path']) as bz,np.load(n['path']) as nz:
            for key in ['circle_reference','d_true','initial_q','sensor_noise_seed']:
                np.testing.assert_array_equal(bz[key],nz[key])
        pairs.append(dict(seed=seed,n0=b,n1=n,usable=usable(b) and usable(n),
            rmse_ratio=n['full_metrics']['rmse_mm']/b['full_metrics']['rmse_mm'],
            clip_ratio=n['full_metrics']['jclip_nm_s']/max(b['full_metrics']['jclip_nm_s'],1e-12)))
    write(root/'validation_summary.json',dict(pairs=pairs,
        all_safe=all(r['usable'] for r in pairs),
        simultaneous_wins=sum(r['rmse_ratio']<1. and r['clip_ratio']<1. for r in pairs),
        rmse_mean_ratio=np.mean([r['n1']['full_metrics']['rmse_mm'] for r in pairs])/np.mean([r['n0']['full_metrics']['rmse_mm'] for r in pairs]),
        clip_mean_ratio=np.mean([r['n1']['full_metrics']['jclip_nm_s'] for r in pairs])/np.mean([r['n0']['full_metrics']['jclip_nm_s'] for r in pairs])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['screen','sweep','validation'],required=True)
    p.add_argument('--output',type=Path,default=ROOT)
    args=p.parse_args();cuda=cuda_preflight();print('PREFLIGHT',cuda,flush=True)
    freeze(args.output,cuda)
    {'screen':screen,'sweep':sweep,'validation':validation}[args.phase](args.output)
    print('DONE',args.phase,flush=True)


if __name__=='__main__':main()
