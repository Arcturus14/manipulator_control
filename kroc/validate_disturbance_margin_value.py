"""Locked, stop-gated ablation of d_hat's incremental value in state margins.

No scenario selection, no headroom, no production-controller changes, no MD.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import torch

from config import DT, Q_MIN, Q_MAX, TAU_MAX, LAMBDA
from kroc.benchmark_circle_multijoint_margin import CASES, run_experiment
from kroc.search_circle_floor25 import CircleTracker

OLD=Path('kroc/results/circle_multijoint_tradeoff_20261114')
NAMES=['static25','velocity_only','directional25_5','static5']
LABELS={'static25':'Static 2.5%','velocity_only':'Velocity-only 2.5-5%',
        'directional25_5':'Velocity + d_hat 2.5-5%','static5':'Static 5%',
        'static30':'Static 3%','static35':'Static 3.5%',
        'static40':'Static 4%','static45':'Static 4.5%'}


class VelocityOnlyCircleTracker(CircleTracker):
    """Remove d_hat ONLY from margin update; keep true DOB state/compensation."""
    def _prepare_solve(self,state,sampling_center):
        values=self.margins.update(state,sampling_center,torch.zeros_like(self.dob.d_hat))
        self.margin_history.append({k:v.detach().clone() for k,v in values.items()})


def bootstrap(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(26111401)
    means=values[rng.integers(len(values),size=(20000,len(values)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()


def incremental_gate(aggregate,paired,collision_free):
    full=aggregate['directional25_5'];vel=aggregate['velocity_only']
    risk_keys=['mean_violation_fraction','worst_violation_rad','mean_integral_rad_s']
    no_worse=all(full[k]<=vel[k]+1e-8 for k in risk_keys)
    differences=[p['full_minus_velocity_rmse_mm'] for p in paired]
    integral=[p['full_minus_velocity_integral_rad_s'] for p in paired]
    risk_wins=sum(p['full_minus_velocity_integral_rad_s'] < -1e-6 for p in paired)
    risk_losses=sum(p['full_minus_velocity_integral_rad_s'] > 1e-6 for p in paired)
    tracking=(no_worse and full['mean_rmse_mm']<=.95*vel['mean_rmse_mm']
              and bootstrap(differences)[1]<0)
    safety=(vel['mean_integral_rad_s']>1e-6
            and full['mean_integral_rad_s']<=.8*vel['mean_integral_rad_s']
            and no_worse and risk_wins>=2 and risk_wins>risk_losses
            and full['mean_rmse_mm']<=1.05*vel['mean_rmse_mm'])
    return {'pass':bool(collision_free and (tracking or safety)),
        'tracking_route':bool(tracking),'risk_route':bool(safety),
        'all_trajectories_collision_free':bool(collision_free),
        'risk_improved_seed_count':risk_wins,'risk_worsened_seed_count':risk_losses,
        'paired_rmse_bootstrap95_ci_mm':bootstrap(differences),
        'paired_integral_bootstrap95_ci_rad_s':bootstrap(integral),
        'small_sample_exploratory_not_formal_significance':True,
        'failure_means_no_justified_incremental_value_not_mathematical_equivalence':True}


def analyse(directory,names,mesh=True):
    rows=json.loads((directory/'metrics.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    seeds=sorted({r['seed'] for r in rows});scenarios=sorted({r['scenario'] for r in rows})
    expected={(s,seed,n) for s in scenarios for seed in seeds for n in names}
    assert {(r['scenario'],r['seed'],r['case']) for r in rows}==expected
    seen={};references={};disturbances={};margin_diagnostics=[]
    for r in rows:
        path=directory/f's{r["scenario"]}_seed{r["seed"]}_{r["case"]}.npz'
        with np.load(path,allow_pickle=False) as z:
            assert str(z['compute_device'])=='cuda'
            assert str(z['dob_compensation_mode'])=='output'
            assert float(z['mppi_temporal_noise_rho'])==manifest.get('temporal_noise_rho',.8)
            for key in ['q','qd','err_mm','u_applied','d_hat_used','kroc_margin']:
                assert np.isfinite(z[key]).all(),key
            condition=r['scenario'];key=(condition,r['seed'])
            if condition not in references:
                references[condition]=z['circle_reference'].copy();disturbances[condition]=z['d_true'].copy()
            assert np.array_equal(references[condition],z['circle_reference'])
            assert np.array_equal(disturbances[condition],z['d_true'])
            assert np.allclose(z['q'][0],manifest['initial_q'])
            assert np.max(np.abs(z['qd'][0]))<1e-8
            req=z['u_mppi']+z['d_hat_used']
            assert np.max(np.abs(z['u_applied']-np.clip(req,-np.asarray(TAU_MAX),TAU_MAX)))<1e-4
            if r['case']=='velocity_only':
                assert np.max(np.abs(z['kroc_a_d_hat']))==0
                assert np.max(np.abs(z['kroc_d_hat_margin']))==0
                # Still a genuine post-DOB run, NOT a no-DOB baseline.
                if np.max(np.abs(z['d_true']))>0:assert np.max(np.abs(z['d_hat_used']))>1
            if r['case']=='directional25_5':
                # Measurements include the terminal state; solves/margins do not.
                vel=z['measured_state'][:len(z['kroc_a_d_hat']),6:]
                span=np.asarray(Q_MAX)-np.asarray(Q_MIN)
                a=z['kroc_a_d_hat'];sides=np.stack([np.maximum(-a,0),np.maximum(a,0)],axis=1)
                vs=np.stack([np.maximum(-vel,0),np.maximum(vel,0)],axis=1)
                mv=np.clip(.025*span+vs*.1,.025*span,.05*span)
                mf=np.clip(.025*span+vs*.1+.5*sides*.1**2,.025*span,.05*span)
                margin_diagnostics.append({'scenario':condition,'seed':r['seed'],
                    'same_state_position_target_extra_mean_fraction':float(((mf-mv)/span).mean()),
                    'same_state_position_target_extra_max_fraction':float(((mf-mv)/span).max()),
                    'wrist_upper_extra_mean_fraction':float(((mf-mv)/span)[:,1,4].mean()),
                    'wrist_upper_extra_max_fraction':float(((mf-mv)/span)[:,1,4].max())})
            seen.setdefault(key,{})[r['case']]=r['metrics']
    if mesh:
        from kroc.audit_circle_validation import dense_mesh_check
        tasks=[(str(directory/f's{r["scenario"]}_seed{r["seed"]}_{r["case"]}.npz'),4) for r in rows]
        with ProcessPoolExecutor(max_workers=6,mp_context=multiprocessing.get_context('spawn')) as pool:
            for r,check in zip(rows,pool.map(dense_mesh_check,tasks)):
                r['mesh_5ms']=check
                print('ABLATION_MESH',r['scenario'],r['seed'],r['case'],check,flush=True)
    aggregate={}
    for name in names:
        rr=[r for r in rows if r['case']==name];mm=[r['metrics'] for r in rr]
        aggregate[name]={'runs':len(rr),
            'violating_runs':sum(m['position_violation_steps']>0 for m in mm),
            'mean_violation_fraction':float(np.mean([m['position_violation_step_fraction'] for m in mm])),
            'worst_violation_rad':float(max(m['position_violation_max_rad'] for m in mm)),
            'mean_integral_rad_s':float(np.mean([m['position_violation_integral_rad_s'] for m in mm])),
            'mean_rmse_mm':float(np.mean([m['tracking_rmse_mm'] for m in mm])),
            'sd_rmse_mm':float(np.std([m['tracking_rmse_mm'] for m in mm],ddof=1)) if len(mm)>1 else 0.,
            'mean_compute_ms':float(np.mean([m['compute_mean_ms'] for m in mm])),
            'mean_wrist_saturation_s':float(np.mean([m['saturation_duration_per_joint_s'][4] for m in mm])),
            'mesh_colliding_runs':sum(r.get('mesh_5ms',{}).get('first_collision_index') is not None for r in rr)}
    paired=[]
    for (scenario,seed),by in sorted(seen.items()):
        if all(n in by for n in ['velocity_only','directional25_5']):
            full,vel=by['directional25_5'],by['velocity_only']
            paired.append({'scenario':scenario,'seed':seed,
                'full_minus_velocity_rmse_mm':full['tracking_rmse_mm']-vel['tracking_rmse_mm'],
                'full_minus_velocity_integral_rad_s':full['position_violation_integral_rad_s']-vel['position_violation_integral_rad_s'],
                'full_minus_velocity_max_rad':full['position_violation_max_rad']-vel['position_violation_max_rad']})
    clear=all(v['mesh_colliding_runs']==0 for v in aggregate.values())
    result={'aggregate':aggregate,'paired':paired,'per_run':rows,
        'same_state_target_diagnostics_unfiltered':margin_diagnostics,
        'mesh_dt_s':.005 if mesh else None,'mesh_is_not_continuous_safety_certificate':True,
        'headroom_enabled':False,'post_dob_compensation_in_all_cases':True}
    if paired:
        result['incremental_gate']=incremental_gate(aggregate,paired,clear)
    return result


def plots(directory,result,names):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for name in names:
        m=result['aggregate'][name]
        rows.append([LABELS[name],f'{m["violating_runs"]}/{m["runs"]}',
            f'{100*m["mean_violation_fraction"]:.2f}',f'{m["worst_violation_rad"]:.4f}',
            f'{m["mean_integral_rad_s"]:.5f}',f'{m["mean_rmse_mm"]:.2f} +/- {m["sd_rmse_mm"]:.2f}',
            str(m['mesh_colliding_runs'])])
    fig,ax=plt.subplots(figsize=(16,4));ax.axis('off')
    t=ax.table(cellText=rows,colLabels=['Margin policy','Violating runs','Violating steps [%]',
        'Worst excess [rad]','Mean integral [rad s]','EE RMSE [mm]','Mesh contact runs'],
        colWidths=[.24,.10,.14,.13,.15,.17,.10],cellLoc='center',loc='center')
    t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2)
    rho=json.loads((directory/'manifest.json').read_text()).get('temporal_noise_rho',.8)
    ax.set_title(f'Locked d_hat-margin ablation; rho={rho:g}; noisy Post-DOB in ALL cases; no headroom')
    if any(m['mesh_colliding_runs'] for m in result['aggregate'].values()):
        fig.suptitle('DIAGNOSTIC: contact dynamics are not modeled',color='red')
    fig.tight_layout();fig.savefig(directory/'results_table.png',dpi=170);plt.close(fig)
    if 'velocity_only' not in names:return
    seeds=sorted({r['seed'] for r in result['per_run']});fig,axes=plt.subplots(2,2,figsize=(13,9))
    for name in ['velocity_only','directional25_5']:
        zs=[np.load(directory/f's0_seed{s}_{name}.npz') for s in seeds]
        time=np.arange(400)*DT
        for ax,values in [(axes[0,0],np.array([z['err_mm'] for z in zs])),
            (axes[0,1],np.array([z['q'][:-1,4] for z in zs])),
            (axes[1,0],np.array([z['kroc_margin'][:,0,1,4]*100/(Q_MAX[4]-Q_MIN[4]) for z in zs]))]:
            ax.plot(time,values.mean(0),label=LABELS[name]);ax.fill_between(time,values.min(0),values.max(0),alpha=.15)
        for z in zs:z.close()
    axes[0,0].set(ylabel='EE error [mm]',title='Mean and seed range')
    axes[0,1].axhline(Q_MAX[4],color='red',ls='--');axes[0,1].set(ylabel='q5 [rad]',title='Physical upper limit shown in red')
    axes[1,0].set(ylabel='Upper q5 margin [% full span]',title='Mean and seed range')
    pp=result['paired'];axes[1,1].bar([str(p['seed']) for p in pp],[p['full_minus_velocity_rmse_mm'] for p in pp])
    axes[1,1].axhline(0,color='black');axes[1,1].set(ylabel='Full - velocity-only RMSE [mm]',title='Negative = d_hat margin improves tracking')
    axes[1,1].tick_params(axis='x',rotation=45)
    for ax in axes.flat:ax.grid(alpha=.25)
    for ax in [axes[0,0],axes[0,1],axes[1,0]]:ax.legend(fontsize=8);ax.set_xlabel('Time [s]')
    fig.tight_layout();fig.savefig(directory/'dhat_vs_velocity.png',dpi=160);plt.close(fig)


def save_analysis(directory,names):
    result=analyse(directory,names)
    (directory/'assessment.json').write_text(json.dumps(result,indent=2))
    plots(directory,result,names)
    print('ABLATION_ASSESSMENT',json.dumps({k:v for k,v in result.items() if k in ['aggregate','incremental_gate']}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-stage-a',action='store_true',help='Analyse an already completed locked stage A without rerunning it')
    p.add_argument('--temporal-noise-rho',type=float,default=0.)
    p.add_argument('--temperature',type=float,default=LAMBDA)
    p.add_argument('--stage-a-only',action='store_true',help='Repeat the locked center comparison only; do not run neighboring-condition tests')
    args=p.parse_args()
    if not args.resume_stage_a:args.output.mkdir(parents=True,exist_ok=False)
    elif not (args.output/'locked_plan.json').is_file():raise ValueError('Missing locked plan')
    old_manifest=json.loads((OLD/'heldout/manifest.json').read_text())
    with np.load(OLD/'heldout/s0_seed1507_directional25_5.npz') as z:reference=z['circle_reference'].copy()
    assert hashlib.sha256(reference.tobytes()).hexdigest()==old_manifest['reference_sha256']
    core=['controllers/mppi_dob_controller.py','costs/tracking_cost.py','experiment_runner.py',
          'observers/dob.py','config.py','kroc/state_margin.py','kroc/search_circle_floor25.py']
    for f in core:assert hashlib.sha256(Path(f).read_bytes()).hexdigest()==old_manifest['source_sha256'][f],f
    geometry={'bank':old_manifest['arguments']['bank'],'candidate':9,
        'q0':np.array(old_manifest['initial_q']),'reference':reference,
        'radius':.24935464438369306,'risk_joint':4,'metadata':{'pose':9}}
    center=old_manifest['profiles'][0]
    seeds=list(range(1607,1707,10))
    configs=dict(CASES,velocity_only=CASES['directional25_5'])
    for fraction in [.03,.04,.045]:
        configs[f'static{round(fraction*1000):02d}']=replace(CASES['static25'],fixed_fraction=fraction)
    plan={'seeds':seeds,'stage_a_names':NAMES,'center_profile':center,
        'temporal_noise_rho':args.temporal_noise_rho,'stage_a_only':args.stage_a_only,
        'temperature':args.temperature,
        'reference_sha256':old_manifest['reference_sha256'],
        'velocity_only':'zero only d_hat argument to margins.update; keep DOB state and output compensation',
        'headroom_enabled':False,'same_initial_pose_reference_noise_DOB_weights_as_archived':True,
        'criterion_tracking_route':'full <=95% velocity RMSE; paired bootstrap upper<0; no worse mean violating time, worst excess, mean integral',
        'criterion_risk_route':'full integral <=80% velocity integral; >=2 paired risk wins and more wins than losses; no worse risk metrics; RMSE <=105% velocity',
        'require_all_cases_5ms_collision_free':True,
        'stop_if_stage_a_not_passed':'no scenario tuning, no additional disturbance/margin search',
        'conditional_stage_b':'static 3/3.5/4/4.5% on center seeds 1707,1717,1727; plus all 8 policies on three locked neighboring conditions seeds 1737,1747,1757',
        'neighbor_conditions':[{'scale':.95,'onset':.5},{'scale':1.05,'onset':.5},{'scale':1.,'onset':1.}],
        'no_retuning_using_new_results':True,'source_sha256':{
            f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in [__file__,'kroc/benchmark_circle_multijoint_margin.py']}}
    if args.resume_stage_a:
        locked=json.loads((args.output/'locked_plan.json').read_text())
        for key in plan:
            if key!='source_sha256':
                legacy={'temporal_noise_rho':.8,'stage_a_only':False,'temperature':1.}
                assert plan[key]==locked.get(key,legacy.get(key)),key
        (args.output/'analysis_resume_source_sha256.json').write_text(json.dumps(plan['source_sha256'],indent=2))
    else:
        (args.output/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    stage=args.output/'stage_a'
    if not args.resume_stage_a:
        run_experiment(stage,geometry,[center],seeds,NAMES,case_configs=configs,
            tracker_classes={'velocity_only':VelocityOnlyCircleTracker},temporal_noise_rho=args.temporal_noise_rho,temperature=args.temperature)
    result=save_analysis(stage,NAMES)
    if not result['incremental_gate']['all_trajectories_collision_free']:
        verdict={'status':'stop_collision_affected_validation','stage_a_pass':False,
            'stage_b_run':False,'temporal_noise_rho':args.temporal_noise_rho,
            'full_episode_tracking_is_not_valid_contact_free_physics':True,
            'lack_of_dhat_value_not_established_by_invalid_comparison':True}
        (args.output/'verdict.json').write_text(json.dumps(verdict,indent=2));print('FINAL_VERDICT',json.dumps(verdict),flush=True)
        return
    if not result['incremental_gate']['pass']:
        verdict={'status':'stop_no_justified_incremental_dhat_margin_value',
            'stage_a_pass':False,'further_search_performed':False,'stage_b_run':False,
            'DOB_compensation_not_tested_for_necessity':True,
            'not_a_proof_that_disturbance_information_is_generally_useless':True}
        (args.output/'verdict.json').write_text(json.dumps(verdict,indent=2));print('FINAL_VERDICT',json.dumps(verdict),flush=True)
        return
    if args.stage_a_only:
        verdict={'status':'stage_a_incremental_value_passed_no_further_conditions_tested',
            'stage_a_pass':True,'stage_b_run':False,'temporal_noise_rho':args.temporal_noise_rho,
            'headroom_enabled':False}
        (args.output/'verdict.json').write_text(json.dumps(verdict,indent=2));print('FINAL_VERDICT',json.dumps(verdict),flush=True)
        return
    # A passed: finish a small PREDECLARED static frontier + neighborhood check.
    all_names=['static25','static30','static35','static40','static45','static5','velocity_only','directional25_5']
    neighbors=[]
    for i,item in enumerate(plan['neighbor_conditions']):
        neighbors.append({**center,'name':f'neighbor_{i}',
            'vector_nm':(np.array(center['vector_nm'])*item['scale']).tolist(),'onset':item['onset']})
    run_experiment(args.output/'stage_b_static',geometry,[center],[1707,1717,1727],all_names,
        case_configs=configs,tracker_classes={'velocity_only':VelocityOnlyCircleTracker},temporal_noise_rho=args.temporal_noise_rho,temperature=args.temperature)
    save_analysis(args.output/'stage_b_static',all_names)
    run_experiment(args.output/'stage_b_neighbors',geometry,neighbors,[1737,1747,1757],all_names,
        case_configs=configs,tracker_classes={'velocity_only':VelocityOnlyCircleTracker},temporal_noise_rho=args.temporal_noise_rho,temperature=args.temperature)
    result_b=save_analysis(args.output/'stage_b_neighbors',all_names)
    verdict={'status':'completed_locked_validation_no_further_search','stage_a_pass':True,
        'neighbor_incremental_gate':result_b['incremental_gate'],'headroom_enabled':False}
    (args.output/'verdict.json').write_text(json.dumps(verdict,indent=2));print('FINAL_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
