"""CUDA-only Base/N0/N1/S0/S1 experiment; frozen baseline and bounded pilot.

Run pilot first, then validation. Raw trajectories and mesh audits are retained.
The pilot selects one common headroom weight using N1 only; heldout seeds never
change the weight or the scenario. Nominal-input effort/smoothness unchanged.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
import torch

from config import DT, TAU_MAX, Q_MIN, Q_MAX
from experiment_runner import run_mppi
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.run_pre_input_baseline import load_preset, PRESET
from kroc.search_lowfreq_estimator_margin import geometry, smooth_load
from kroc.input_headroom_study import InputHeadroomTracker, CASES
from kroc.audit_circle_validation import dense_mesh_check

ROOT = Path('kroc/results/input_headroom_study_20261115')
PILOT = 4707
HELDOUT = [4807, 4817, 4827]
PROFILES = {'nominal': 0., 'slow5p5': 5.5, 'near26': 26.}
WEIGHTS = [1., 10., 100.]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def run_options(c, seed, case, weight):
    return dict(use_dob=True, dob_compensation_mode='output',
        sampling_center_mode=c['sampling_center_mode'], tracker_class=InputHeadroomTracker,
        tracker_options=dict(study_case=case, headroom_weight=weight, headroom_fraction=.85,
            proposal=c['proposal'], boundary_weight=c['boundary_weight'], margin_config=c['state_margin'],
            sensor_noise=dict(position_std_rad=c['sensor_position_std_rad'],
                velocity_std_rad_s=c['sensor_velocity_std_rad_s'], seed=98000+seed),
            dob_acceleration_lpf_cutoff_hz=c['dob_acceleration_lpf_hz'], dob_cutoff_hz=c['dob_cutoff_hz'],
            collision_weight=c['collision_weight'], collision_clearance=c['collision_clearance_m'],
            collision_backend=c['collision_backend']),
        require_cuda=True, seed=seed, temperature=c['temperature'], temporal_noise_rho=1.,
        mean_update_alpha=c['mean_update_alpha'], smooth_weight=c['horizon_smooth_weight'],
        n_samples=c['samples'], horizon=c['horizon'], noise_variance=c['noise_variance'],
        position_weight=c['position_weight'], terminal_weight=c['terminal_weight'],
        joint_position_limit_weight=c['joint_position_limit_weight'],
        joint_velocity_limit_weight=c['joint_velocity_limit_weight'], compile_rollout_loop=False)


def measurement(z, n=None):
    n = len(z['u_mppi']) if n is None else n
    v = z['u_mppi'][:n]; c = z['d_hat_used'][:n]; a = z['u_applied'][:n]
    req = v+c; loss = a-req; limit = np.asarray(TAU_MAX)
    q = z['q'][:n+1]; dq = z['qd'][:n+1]
    excess = np.maximum(np.maximum(q-Q_MAX, np.asarray(Q_MIN)-q), 0.)
    clipped = (np.abs(loss)>1e-5).any(-1)
    feasible = (np.abs(req)<=limit+1e-5).all(-1)
    expanded = (np.abs(v)>limit+1e-5).any(-1) & feasible
    pred = z['u_effective_pred'][:n]
    error = z['state_prediction_error'][:n]
    du = np.diff(a, axis=0)
    head = 1.-np.abs(a)/limit
    timing = z['solve_t'][min(20,n-1):n]*1000.
    return dict(tracking_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][:n]**2))),
        clipping_steps=int(clipped.sum()), clipping_fraction=float(clipped.mean()),
        clipping_integral_nm_s=float(np.linalg.norm(loss,axis=-1).sum()*DT),
        clipping_max_nm=float(np.abs(loss).max()),
        near98_fraction=float((np.abs(a)>=.98*limit).any(-1).mean()),
        near85_fraction=float((np.abs(a)>=.85*limit).any(-1).mean()),
        minimum_headroom_fraction=float(head.min()),
        expanded_feasible_steps=int(expanded.sum()),
        nominal_limit_excess_steps=int((np.abs(v)>limit+1e-5).any(-1).sum()),
        nominal_abs_max_per_joint_nm=np.abs(v).max(0).tolist(),
        clipping_per_joint_s=((np.abs(loss)>1e-5).sum(0)*DT).tolist(),
        torque_rate_rms_nm_s=float(np.sqrt(np.mean(du**2))/DT),
        delta_torque_rms_nm=float(np.sqrt(np.mean(du**2))),
        prediction_q_rmse_rad=float(np.sqrt(np.mean(error[:,:6]**2))),
        prediction_dq_rmse_rad_s=float(np.sqrt(np.mean(error[:,6:]**2))),
        predicted_effective_abs_max_nm=float(np.abs(pred).max()),
        max_position_excess_rad=float(excess.max()),
        position_violation_steps=int((excess[1:]>1e-6).any(-1).sum()),
        max_speed_rad_s=float(np.abs(dq).max()),
        velocity_violation_steps=int(z['joint_velocity_limit_violated'][:n].sum()),
        ess_mean=float(z['mppi_weight_ess'][:n].mean()),
        ess_median=float(np.median(z['mppi_weight_ess'][:n])),
        solve_mean_ms=float(timing.mean()), solve_p95_ms=float(np.percentile(timing,95)),
        sample_request_excess_fraction=float(z['input_study_sample_request_excess_fraction'][:n].mean()),
        dob6_rmse_nm=float(np.sqrt(np.mean((c[:,5]-z['d_true'][1:n+1,5])**2))))


def audit(z, case):
    assert str(z['compute_device']) == 'cuda'
    for key in ['q','qd','u_mppi','u_applied','d_hat_used','err_mm','state_prediction_error']:
        assert np.isfinite(z[key]).all(), key
    c = load_preset()
    for key, expected in [('mppi_temperature',c['temperature']),('mppi_temporal_noise_rho',1.),
        ('mppi_mean_update_alpha',c['mean_update_alpha']),('mppi_smooth_weight',c['horizon_smooth_weight']),
        ('dob_cutoff_hz',c['dob_cutoff_hz']),('dob_acceleration_lpf_cutoff_hz',c['dob_acceleration_lpf_hz'])]:
        assert np.isclose(float(z[key]),expected), key
    np.testing.assert_array_equal(z['mppi_active_noise_variance'],
        np.broadcast_to(np.asarray(c['noise_variance'],dtype=np.float32),z['mppi_active_noise_variance'].shape))
    v = z['u_mppi']; comp = z['d_hat_used']; limit = np.asarray(TAU_MAX)
    np.testing.assert_allclose(z['u_applied'],np.clip(v+comp,-limit,limit),atol=3e-6,rtol=1e-6)
    expected = np.clip(v+comp,-limit,limit)-comp if CASES[case]['saturation_rollout'] else v
    np.testing.assert_allclose(z['u_effective_pred'],expected,atol=4e-6,rtol=1e-6)
    assert len(z['input_study_sample_request_excess_fraction']) == len(v)
    if case == 'Base':
        assert (np.abs(v)<=limit+1e-5).all()
    margin = z['kroc_margin']
    span = np.array(Q_MAX)-np.array(Q_MIN)
    np.testing.assert_allclose(margin[:,0], np.broadcast_to(.025*span,margin[:,0].shape),atol=2e-6)
    np.testing.assert_allclose(z['kroc_q_upper_normalizer'],np.broadcast_to(.05*span,(len(v),6)),atol=2e-6)


def run_one(root, phase, case, profile, seed, weight):
    directory = root/phase; directory.mkdir(exist_ok=True)
    effective_weight = weight if CASES[case]['headroom'] else 0.
    path = directory/f'{case}_{profile}_w{effective_weight:g}_seed{seed}.npz'
    meta = path.with_suffix('.json')
    if meta.exists():
        return json.loads(meta.read_text())
    print('RUN',phase,case,profile,seed,effective_weight,flush=True)
    if not path.exists():
        reference,q0,_ = geometry(.89)
        amp = PROFILES[profile]
        z = run_mppi(reference,initial_q=q0,has_disturbance=bool(amp),
            disturbance_function=smooth_load(amp),**run_options(load_preset(),seed,case,weight))
        audit(z,case)
        z.update(circle_reference=reference,baseline_preset=np.asarray(str(PRESET)))
        np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')})
        del z; gc.collect(); torch.cuda.empty_cache()
    mesh = dense_mesh_check((str(path),4))
    with np.load(path) as z:
        audit(z,case)
        m = measurement(z)
    row = dict(case=case,profile=profile,seed=seed,headroom_weight=effective_weight,
               phase=phase,path=str(path),mesh_5ms=mesh,full_metrics=m)
    write(meta,row)
    print('RESULT',case,profile,seed,json.dumps(m),flush=True)
    return row


def select_weight(rows):
    baseline = next(r for r in rows if r['case']=='N0' and r['profile']=='near26')
    b = baseline['full_metrics']; candidates=[]
    for r in rows:
        if r['case'] != 'N1': continue
        m = r['full_metrics']
        eligible = (r['mesh_5ms']['first_collision_index'] is None
            and m['max_position_excess_rad']<=b['max_position_excess_rad']+1e-6
            and m['tracking_rmse_mm']<=1.2*b['tracking_rmse_mm']
            and m['clipping_integral_nm_s']<=.9*b['clipping_integral_nm_s']
            and b['clipping_integral_nm_s']>1e-5)
        candidates.append(dict(weight=r['headroom_weight'],eligible=bool(eligible),metrics=m))
    accepted = [r['weight'] for r in candidates if r['eligible']]
    return dict(selected_weight=min(accepted) if accepted else 10.,
        pilot_gate_passed=bool(accepted),fallback_is_descriptive_only=not bool(accepted),candidates=candidates,
        rule='Smallest N1 weight with >=10% clipping-integral reduction vs N0, <=20% RMSE increase, no contact or increased position excess; otherwise predeclared w=10 descriptive validation. Same weight for S1.')


def summarize(root, rows):
    paired=[]
    for profile in PROFILES:
        for seed in HELDOUT:
            rr = [r for r in rows if r['profile']==profile and r['seed']==seed]
            if len(rr)!=len(CASES): continue
            hits = [r['mesh_5ms']['first_collision_index'] for r in rr if r['mesh_5ms']['first_collision_index'] is not None]
            n = 300 if not hits else max(2,min(300,int(np.floor((min(hits)*.005-DT)/DT))))
            first=None
            for r in rr:
                with np.load(r['path']) as z:
                    common = {k:z[k].copy() for k in ['circle_reference','d_true','initial_q','sensor_noise_seed']}
                    if first is None: first=common
                    else:
                        for key in first: np.testing.assert_array_equal(common[key],first[key])
                    m=measurement(z,n)
                paired.append(dict(case=r['case'],profile=profile,seed=seed,common_end_s=n*DT,
                    contact=r['mesh_5ms']['first_collision_index'] is not None,metrics=m))
    groups=[]
    for profile in PROFILES:
        for case in CASES:
            rr=[r for r in paired if r['profile']==profile and r['case']==case]
            if not rr: continue
            keys=[k for k,v in rr[0]['metrics'].items() if isinstance(v,(int,float))]
            groups.append(dict(case=case,profile=profile,runs=len(rr),
                contact_runs=sum(r['contact'] for r in rr),common_end_s=[r['common_end_s'] for r in rr],
                mean={k:float(np.mean([r['metrics'][k] for r in rr])) for k in keys},
                std={k:float(np.std([r['metrics'][k] for r in rr],ddof=1)) if len(rr)>1 else 0. for k in keys}))
    write(root/'summary.json',dict(groups=groups,paired=paired,full_runs=rows,
        note='Collision cases use common pre-contact windows; full trajectories never count as valid performance after contact. 3 heldout seeds are exploratory, not a general guarantee.'))
    plot(root,rows)


def plot(root,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for profile in PROFILES:
        rr=[r for r in rows if r['profile']==profile and r['seed']==HELDOUT[0]]
        if len(rr)!=5: continue
        fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
        for r in rr:
            with np.load(r['path']) as z:
                t=np.arange(len(z['u_mppi']))*DT
                req=z['u_mppi']+z['d_hat_used']; label=r['case']
                axes[0,0].plot(t,z['err_mm'],label=label)
                axes[0,1].plot(t,req[:,5],label=label)
                axes[1,0].plot(t,z['u_applied'][:,5],label=label)
                axes[1,1].plot(t,np.linalg.norm(z['u_applied']-req,axis=1),label=label)
                axes[2,0].plot(t+DT,z['q'][1:,5],label=label)
                axes[2,1].plot(t,z['d_hat_used'][:,5],label=label)
                true=z['d_true'][1:,5].copy()
        axes[2,1].plot(t,true,'k--',label='true d6')
        for ax in [axes[0,1],axes[1,0]]:
            for lim in [-28.,28.]: ax.axhline(lim,color='k',ls='--',lw=.8)
            for lim in [-23.8,23.8]: ax.axhline(lim,color='gray',ls=':',lw=.8)
        axes[2,0].axhline(Q_MAX[5],color='k',ls='--')
        for ax,y in zip(axes.flat,['EE error [mm]','Requested torque6 [Nm]',
            'Applied torque6 [Nm]','Clipping loss norm [Nm]','q6 [rad]','Disturbance6 [Nm]']):
            ax.set(xlabel='Time [s]',ylabel=y);ax.grid(alpha=.25);ax.legend(fontsize=7)
        fig.suptitle(f'{profile}; heldout seed {HELDOUT[0]}; frozen pre-input baseline; no dynamic input bounds')
        fig.savefig(root/f'{profile}_comparison.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=['pilot','validation','summarize'],required=True)
    p.add_argument('--output',type=Path,default=ROOT)
    a=p.parse_args();root=a.output
    if a.phase=='summarize':
        return summarize(root,[json.loads(f.read_text()) for f in sorted((root/'validation').glob('*.json'))])
    cuda=cuda_preflight();print('PREFLIGHT',json.dumps(cuda),flush=True)
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'manifest.json').exists():
        sources=[PRESET,Path(__file__),Path('kroc/input_headroom_study.py'),
            Path('controllers/mppi_dob_controller.py'),Path('costs/tracking_cost.py'),
            Path('kroc/finalize_pre_input_baseline.py'),Path('kroc/benchmark_single_instance_mppi.py'),
            Path('kroc/state_margin.py'),Path('kroc/search_circle_floor25.py'),
            Path('kroc/search_lowfreq_estimator_margin.py'),Path('kroc/find_estimator_directional_separation.py'),
            Path('experiment_runner.py'),Path('observers/dob.py'),Path('config.py')]
        sha={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources}
        snapshot=root/'source';snapshot.mkdir(exist_ok=True)
        for f in sources:
            relative=f.resolve().relative_to(Path.cwd())
            target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
        write(root/'manifest.json',dict(cuda=cuda,baseline=load_preset(),baseline_preset=str(PRESET),
            cases=CASES,pilot_seed=PILOT,heldout_seeds=HELDOUT,profiles=PROFILES,pilot_weights=WEIGHTS,
            headroom_fraction=.85,source_sha256=sha,production_defaults_changed=False,
            scene='same geometry(.89): R=.04m circle, period/duration6s, initial q unchanged',
            disturbance='negative joint6; .5s onset; 2s raised-cosine rise then held; only amplitude varies',
            common_costs='original nominal-v effort and smoothness, no clipping within costs for unrestricted arms',
            optimized_gui_adapter_used=False,reference_tracker='frozen BoundaryPaperTracker',
            note='Baseline validated without realtime certification. CPU fallback prohibited.'))
    else:
        saved=json.loads((root/'manifest.json').read_text())
        for file,sha in saved['source_sha256'].items():
            if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:
                raise RuntimeError(f'Source changed since experiment freeze: {file}; use a fresh output directory')
    rows=[]
    if a.phase=='pilot':
        for profile in PROFILES: rows.append(run_one(root,'pilot','Base',profile,PILOT,0.))
        for case in ['N0','S0']: rows.append(run_one(root,'pilot',case,'near26',PILOT,0.))
        for weight in WEIGHTS: rows.append(run_one(root,'pilot','N1','near26',PILOT,weight))
        selection=select_weight(rows);write(root/'selection.json',selection)
        rows.append(run_one(root,'pilot','S1','near26',PILOT,selection['selected_weight']))
        write(root/'pilot_metrics.json',rows)
        print('SELECTION',json.dumps(selection),flush=True)
    else:
        selection=json.loads((root/'selection.json').read_text());weight=selection['selected_weight']
        for profile in PROFILES:
            for seed in HELDOUT:
                for case in CASES:
                    rows.append(run_one(root,'validation',case,profile,seed,weight))
                    write(root/'validation_metrics.json',rows)
        summarize(root,rows)
    print('DONE',a.phase,flush=True)


if __name__=='__main__':
    main()
