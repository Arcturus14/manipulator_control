"""Locked temperature-only pairs, then a fresh small-circle margin comparison.

CUDA simulations only. No headroom, scenario search, retuning, or MD reports.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np
from config import DT, Q_MIN, Q_MAX, TAU_MAX
from kroc.benchmark_circle_multijoint_margin import CASES, run_experiment
from kroc.validate_disturbance_margin_value import VelocityOnlyCircleTracker, NAMES, LABELS
from kroc.audit_circle_validation import dense_mesh_check
from kroc.circle_risk_tradeoff_assessment import position_metrics

OLD=Path('kroc/results/circle_multijoint_tradeoff_20261114/heldout')
SMALL=Path('kroc/results/rho0_small_circle_check_20261114')
SETTINGS=['seed','horizon','samples','mean_update_alpha','noise_variance',
    'temporal_noise_rho','disturbance_covariance_gain','position_weight','terminal_weight',
    'smooth_weight','joint_position_limit_weight','joint_velocity_limit_weight',
    'joint_limit_soft_ratio','compile_rollout_loop']


def load_conditions():
    old=json.loads((OLD/'manifest.json').read_text())
    small=json.loads((SMALL/'plan.json').read_text())
    with np.load(OLD/'s0_seed1507_directional25_5.npz') as z:ref=z['circle_reference'].copy()
    assert hashlib.sha256(ref.tobytes()).hexdigest()==old['reference_sha256']
    big=dict(bank=old['arguments']['bank'],candidate=old['arguments']['candidate'],
        q0=np.array(old['initial_q']),reference=ref,radius=.24935464438369306,
        risk_joint=4,metadata={'pose':9})
    with np.load(SMALL/'strong/s0_seed1807_static25.npz') as z:small_ref=z['circle_reference'].copy()
    little=dict(bank='previously_contact_free_rho0_small_circle',candidate=0,
        q0=np.array(small['initial_q']),reference=small_ref,radius=small['radius_m'],
        risk_joint=4,metadata={'pose':'rho0_small_R013'})
    return big,old['profiles'],little,[next(p for p in small['profiles'] if p['name']=='strong')]


def run_conditions(root):
    big,profiles,small,strong=load_conditions()
    root.mkdir(parents=True,exist_ok=False)
    plan=dict(rho=0.,temperatures=[1.,.24],cases=NAMES,headroom=False,
        stage_a_seeds=[1607,1617,1627],stage_b_seeds=[1817,1827,1837],
        stage_a='Same archived circle, initial pose, sustained disturbance; fresh reruns at BOTH temperatures',
        stage_b='Same small R=0.13m strong case as previous check, but three fresh seeds, temperature=0.24',
        max_runs=36,compile_rollout_loop=True,no_scenario_search=True,
        stage_a_profiles=profiles,stage_b_profiles=strong,
        comparison='All four policies retain post-DOB compensation; velocity_only zeros d_hat ONLY for margins',
        contact_policy='Invalidate full-episode RMSE after any detected contact; also report common precontact windows',
        mesh_sample_dt_s=.005,small_seed_count_is_exploratory=True,
        source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            [__file__,'kroc/benchmark_circle_multijoint_margin.py','kroc/state_margin.py',
             'controllers/mppi_dob_controller.py','observers/dob.py','experiment_runner.py']})
    (root/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    cases={**CASES,'velocity_only':CASES['directional25_5']}
    for label,temperature in [('lambda1',1.),('lambda024',.24)]:
        run_experiment(root/'stage_a'/label,big,profiles,plan['stage_a_seeds'],NAMES,
            case_configs=cases,tracker_classes={'velocity_only':VelocityOnlyCircleTracker},
            temporal_noise_rho=0.,temperature=temperature,compile_rollout_loop=True)
    audit_stage(root/'stage_a',['lambda1','lambda024'])
    run_experiment(root/'stage_b'/'lambda024',small,strong,plan['stage_b_seeds'],NAMES,
        case_configs=cases,tracker_classes={'velocity_only':VelocityOnlyCircleTracker},
        temporal_noise_rho=0.,temperature=.24,compile_rollout_loop=True)
    audit_stage(root/'stage_b',['lambda024'])
    summarize(root)


def audit_stage(stage,labels):
    tasks=[];records=[]
    for label in labels:
        directory=stage/label
        manifest=json.loads((directory/'manifest.json').read_text())
        for row in json.loads((directory/'metrics.json').read_text()):
            path=directory/f"s0_seed{row['seed']}_{row['case']}.npz"
            records.append(dict(label=label,path=str(path),**row))
            tasks.append((str(path),4))
    with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row,mesh in zip(records,pool.map(dense_mesh_check,tasks)):
            row['mesh_5ms']=mesh
            print('TEMPERATURE_MESH',row['label'],row['seed'],row['case'],mesh,flush=True)
    (stage/'audited_runs.json').write_text(json.dumps(records,indent=2,allow_nan=False))


def safe_count(contact_index,n_steps):
    """Conservatively omit the whole control interval containing sampled contact."""
    if contact_index is None:return n_steps
    return max(0,int(np.floor((contact_index*.005-1e-9)/DT)))


def validate_pairs(stage,records):
    labels=sorted({r['label'] for r in records})
    first_manifest=json.loads((stage/labels[0]/'manifest.json').read_text())
    for label in labels[1:]:
        manifest=json.loads((stage/label/'manifest.json').read_text())
        for key in ['arguments','profiles','cases','sensor_noise','dob','collision_cost',
            'initial_q','reference_sha256','source_sha256','compile_rollout_loop','tracker_class_per_case']:
            assert first_manifest[key]==manifest[key],key
    seeds=sorted({r['seed'] for r in records})
    checks=0
    for seed in seeds:
        group=[r for r in records if r['seed']==seed]
        loaded=[]
        for r in group:
            with np.load(r['path'],allow_pickle=False) as z:loaded.append({k:z[k].copy() for k in z.files})
        first=loaded[0]
        for r,z in zip(group,loaded):
            assert str(z['compute_device'])=='cuda' and str(z['dob_compensation_mode'])=='output'
            assert float(z['mppi_temporal_noise_rho'])==0.
            assert float(z['mppi_temperature'])==({'lambda1':1.,'lambda024':.24}[r['label']])
            for field in SETTINGS:np.testing.assert_array_equal(first['mppi_'+field],z['mppi_'+field])
            for field in ['circle_reference','d_true','initial_q','initial_qd',
                'sensor_noise_seed','sensor_position_std_rad','sensor_velocity_std_rad_s',
                'dob_cutoff_hz','dob_acceleration_lpf_cutoff_hz']:
                np.testing.assert_array_equal(first[field],z[field])
            for field in ['q','qd','err_mm','u_applied','d_hat_used']:assert np.isfinite(z[field]).all()
            request=z['u_mppi']+z['d_hat_used']
            assert np.max(np.abs(z['u_applied']-np.clip(request,-np.asarray(TAU_MAX),TAU_MAX)))<1e-4
            if r['case']=='velocity_only':assert np.max(np.abs(z['kroc_d_hat_margin']))==0.
            # Independent measurement RNG gives the same additive noise in paired arms.
            noise=z['measured_state']-np.column_stack((z['q'],z['qd']))
            base_noise=first['measured_state']-np.column_stack((first['q'],first['qd']))
            np.testing.assert_allclose(noise,base_noise,atol=2e-5,rtol=0)
            checks+=1
        counts=[safe_count(r['mesh_5ms']['first_collision_index'],len(z['err_mm'])) for r,z in zip(group,loaded)]
        common=min(counts)
        for r,z,count in zip(group,loaded,counts):
            assert count>0 and common>0,'No usable precontact interval'
            risk=position_metrics(z['q'][:count+1])
            r.update(contact_free=r['mesh_5ms']['first_collision_index'] is None,
                first_contact_s=None if r['mesh_5ms']['first_collision_index'] is None else r['mesh_5ms']['first_collision_index']*.005,
                own_precontact_steps=count,common_precontact_steps=common,
                common_precontact_s=common*DT,precontact_position=risk,
                common_precontact_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][:common]**2))),
                first_2s_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][:100]**2))) if count>=100 else None,
                valid_full_episode_rmse_mm=r['metrics']['tracking_rmse_mm'] if count==len(z['err_mm']) else None,
                wrist_upper_min_distance_precontact_rad=float(Q_MAX[4]-z['q'][:count+1,4].max()))
    return checks


def aggregate(records):
    result={}
    for label in sorted({r['label'] for r in records}):
        result[label]={}
        for name in NAMES:
            rows=[r for r in records if r['label']==label and r['case']==name]
            clear=all(r['contact_free'] for r in rows)
            result[label][name]=dict(runs=len(rows),contact_runs=sum(not r['contact_free'] for r in rows),
                precontact_violating_runs=sum(r['precontact_position']['violation_episode'] for r in rows),
                worst_precontact_excess_rad=max(r['precontact_position']['maximum_violation_rad'] for r in rows),
                mean_precontact_integral_rad_s=float(np.mean([r['precontact_position']['cumulative_violation_rad_s'] for r in rows])),
                full_episode_rmse_mean_mm=float(np.mean([r['valid_full_episode_rmse_mm'] for r in rows])) if clear else None,
                common_precontact_rmse_mean_mm=float(np.mean([r['common_precontact_rmse_mm'] for r in rows])),
                first_2s_rmse_mean_mm=(float(np.mean([r['first_2s_rmse_mm'] for r in rows]))
                    if all(r['first_2s_rmse_mm'] is not None for r in rows) else None),
                mean_wrist_saturation_s=float(np.mean([r['metrics']['test_joint_saturation_steps']*DT for r in rows])) if clear else None,
                mean_min_wrist_clearance_rad=float(np.mean([r['wrist_upper_min_distance_precontact_rad'] for r in rows])),
                mean_solve_ms=float(np.mean([r['metrics']['compute_mean_ms'] for r in rows])),
                mean_weight_ess=None)
            ess=[]
            for r in rows:
                with np.load(r['path']) as z:ess.append(float(np.mean(z['mppi_weight_ess'][:r['common_precontact_steps']])))
            result[label][name]['mean_weight_ess']=float(np.mean(ess))
    return result


def summarize(root):
    summaries={}
    for stage,labels in [('stage_a',['lambda1','lambda024']),('stage_b',['lambda024'])]:
        records=json.loads((root/stage/'audited_runs.json').read_text())
        checks=validate_pairs(root/stage,records)
        result=dict(setting_audits_passed=checks,aggregate=aggregate(records),per_run=records,
            common_windows={str(seed):next(r['common_precontact_s'] for r in records if r['seed']==seed)
                for seed in sorted({r['seed'] for r in records})},
            mesh_sampling_not_continuous_guarantee=True,seed_count_is_exploratory=True)
        (root/stage/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False))
        summaries[stage]=result
    small=summaries['stage_b']['aggregate']['lambda024']
    vel,full,low,high=[small[n] for n in ['velocity_only','directional25_5','static25','static5']]
    valid=all(m['contact_runs']==0 for m in small.values())
    differences=[]
    runs=summaries['stage_b']['per_run']
    for seed in sorted({r['seed'] for r in runs}):
        pair={r['case']:r for r in runs if r['seed']==seed}
        differences.append(dict(seed=seed,full_minus_velocity_rmse_mm=
            pair['directional25_5']['metrics']['tracking_rmse_mm']-pair['velocity_only']['metrics']['tracking_rmse_mm']))
    verdict=dict(completed_runs=sum(s['setting_audits_passed'] for s in summaries.values()),
        headroom=False,scenario_search_performed=False,new_default_temperature=.24,new_default_rho=0.,
        stage_b_all_contact_free=valid,
        stage_b_paired_dhat_minus_velocity_rmse=differences,
        dynamic_risk_reduction_vs_small_static_observed=bool(valid and full['mean_precontact_integral_rad_s']<low['mean_precontact_integral_rad_s']),
        dynamic_tracking_improvement_vs_large_static_observed=bool(valid and full['full_episode_rmse_mean_mm']<high['full_episode_rmse_mean_mm']),
        dhat_tracking_improvement_vs_velocity_only_observed=bool(valid and full['full_episode_rmse_mean_mm']<vel['full_episode_rmse_mean_mm']),
        not_a_proof_of_general_uselessness_or_robustness=True,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    archived=Path('kroc/results/dhat_margin_value_rho0_20261114/stage_a')
    reproduction=[]
    for row in summaries['stage_a']['per_run']:
        if row['label']!='lambda1':continue
        previous=archived/Path(row['path']).name
        with np.load(previous) as old,np.load(row['path']) as new:
            reproduction.append(dict(seed=row['seed'],case=row['case'],
                exact_state_reproduction=bool(np.array_equal(old['q'],new['q'])),
                exact_applied_torque_reproduction=bool(np.array_equal(old['u_applied'],new['u_applied'])),
                max_state_difference_rad=float(np.abs(old['q']-new['q']).max())))
    (root/'lambda1_reproduction_audit.json').write_text(json.dumps(reproduction,indent=2,allow_nan=False))
    verdict['lambda1_archived_reproductions_exact']=all(r['exact_state_reproduction'] and r['exact_applied_torque_reproduction'] for r in reproduction)
    (root/'verdict.json').write_text(json.dumps(verdict,indent=2,allow_nan=False))
    plot_summary(root,summaries)
    print('TEMPERATURE_VERDICT',json.dumps(verdict),flush=True)


def plot_summary(root,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage,result in summaries.items():
        cells=[]
        for label,policies in result['aggregate'].items():
            for name,m in policies.items():
                rmse=m['full_episode_rmse_mean_mm']
                cells.append([label,LABELS[name],f"{m['contact_runs']}/{m['runs']}",
                    f"{m['precontact_violating_runs']}/{m['runs']}",f"{m['worst_precontact_excess_rad']:.4f}",
                    f"{m['common_precontact_rmse_mean_mm']:.2f}",'Invalid (contact)' if rmse is None else f'{rmse:.2f}'])
        fig,ax=plt.subplots(figsize=(15,3+len(cells)*.35));ax.axis('off')
        table=ax.table(cellText=cells,colLabels=['Temperature','Margin policy','Contact runs','q excess runs\nbefore contact/end',
            'Worst excess\n[rad]','Common precontact\nEE RMSE [mm]','Full episode\nEE RMSE [mm]'],cellLoc='center',loc='center',
            colWidths=[.10,.23,.10,.14,.12,.15,.16])
        table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,2)
        ax.set_title(f'{stage}: rho=0, noisy Post-DOB, three paired seeds, no headroom\n'
            f"Common no-contact windows by seed [s]: { {seed:round(value,2) for seed,value in result['common_windows'].items()} }")
        fig.tight_layout();fig.savefig(root/stage/'results_table.png',dpi=170);plt.close(fig)
    # Compare both temperatures only inside the SAME contact-free time window.
    a=summaries['stage_a'];n=min(r['common_precontact_steps'] for r in a['per_run'])
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,name in zip(axes.flat,NAMES):
        for label,color in [('lambda1','tab:red'),('lambda024','tab:blue')]:
            values=[]
            for row in a['per_run']:
                if row['case']==name and row['label']==label:
                    with np.load(row['path']) as z:values.append(z['err_mm'][:n].copy())
            values=np.array(values);time=(np.arange(n)+1)*DT
            ax.plot(time,values.mean(0),color=color,label=label)
            ax.fill_between(time,values.min(0),values.max(0),color=color,alpha=.15)
        ax.set(title=LABELS[name],xlabel='Time [s]',ylabel='True EE error [mm]')
        ax.grid(alpha=.25);ax.legend()
    fig.suptitle(f'Stage A temperature-only comparison: mean and min/max over three seeds\n'
        f'All curves censored to common no-contact interval: 0-{n*DT:.2f}s; rho=0')
    fig.savefig(root/'stage_a/temperature_comparison.png',dpi=160);plt.close(fig)
    rows=summaries['stage_b']['per_run']
    if any(not r['contact_free'] for r in rows):return
    seeds=sorted({r['seed'] for r in rows})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors=['tab:orange','tab:purple','tab:blue','tab:green']
    for name,color in zip(NAMES,colors):
        data=[]
        for seed in seeds:
            path=next(r['path'] for r in rows if r['seed']==seed and r['case']==name)
            with np.load(path) as z:data.append({k:z[k].copy() for k in ['ee','err_mm','q','kroc_margin','circle_reference']})
        time=np.arange(400)*DT
        axes[0,0].plot(data[0]['ee'][:,0],data[0]['ee'][:,1],color=color,label=LABELS[name])
        for ax,values,t in [(axes[0,1],np.array([z['err_mm'] for z in data]),time+DT),
            (axes[1,0],np.array([z['q'][:,4] for z in data]),np.arange(401)*DT),
            (axes[1,1],np.array([100*z['kroc_margin'][:,0,1,4]/(Q_MAX[4]-Q_MIN[4]) for z in data]),time)]:
            ax.plot(t,values.mean(0),color=color,label=LABELS[name])
            ax.fill_between(t,values.min(0),values.max(0),color=color,alpha=.15)
    ref=data[0]['circle_reference'];axes[0,0].plot(ref[:,0],ref[:,1],'k--',label='Reference')
    axes[0,0].set(xlabel='X [m]',ylabel='Y [m]',aspect='equal',title=f'EE trajectory, first predeclared seed {seeds[0]}')
    axes[0,1].set(xlabel='Time [s]',ylabel='True EE error [mm]',title='Mean and min/max over 3 seeds')
    axes[1,0].axhline(Q_MAX[4],color='k',ls='--');axes[1,0].set(xlabel='Time [s]',ylabel='Joint 5 position [rad]')
    axes[1,1].set(xlabel='Time [s]',ylabel='Upper margin [% full joint span]')
    for ax in axes.flat:ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Stage B: lambda=0.24, rho=0, R=0.13m, sustained disturbance [-120,0,0,0,-25,0] Nm')
    fig.savefig(root/'stage_b/comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--analyse-only',action='store_true')
    args=parser.parse_args()
    if args.analyse_only:summarize(args.output)
    else:run_conditions(args.output)
