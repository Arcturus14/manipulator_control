"""Bounded staged circle search: nominal pressure, multi-joint loads, held-out ablation."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from kroc.benchmark_circle_multijoint_margin import load_circle,run_experiment

NAMES=['static25','directional25_5','static5']


def nominal_gate(rows):
    by={n:[r['metrics'] for r in rows if r['case']==n] for n in NAMES}
    dyn=np.mean([m['tracking_rmse_mm'] for m in by['directional25_5']])
    high=np.mean([m['tracking_rmse_mm'] for m in by['static5']])
    safe=all(m['position_violation_max_rad']<=1e-6 and m['first_collision_index'] is None
        for rr in by.values() for m in rr)
    return {'pass':bool(safe and dyn<=10. and high-dyn>=2. and high>=1.2*dyn),
        'dynamic_rmse_mm':float(dyn),'high_static_rmse_mm':float(high),'all_safe_mesh20ms':safe}


def pilot_gate(rows):
    by={r['case']:r['metrics'] for r in rows}
    low,dyn,high=[by[n] for n in NAMES]
    checks={'all_collision_free_20ms':all(m['first_collision_index'] is None for m in by.values()),
        'low_has_position_violation':low['position_violation_max_rad']>1e-4,
        'dynamic_less_violating_time':dyn['position_violation_steps']<low['position_violation_steps'],
        'dynamic_smaller_worst_violation':dyn['position_violation_max_rad']<low['position_violation_max_rad'],
        'dynamic_less_integral':dyn['position_violation_integral_rad_s']<low['position_violation_integral_rad_s'],
        'dynamic_tracking_5percent_better_than_high':dyn['tracking_rmse_mm']<=.95*high['tracking_rmse_mm'],
        'dynamic_tracking_below_30mm':dyn['tracking_rmse_mm']<=30.}
    return {'pass':all(checks.values()),'checks':checks}


def profile(name,vector,onset,ramp=.2):
    return {'name':name,'vector_nm':vector,'onset':onset,'ramp':ramp}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    primary=load_circle('kroc/results/nominal_margin5_geometry_screen_20261113',61,0)
    secondary=load_circle('kroc/results/coupled_wrist_margin_bank_20261113',9,4)
    old=Path('kroc/results/nominal_margin5_search_20261113/validation')
    old_manifest=json.loads((old/'manifest.json').read_text())
    matches={f:hashlib.sha256(Path(f).read_bytes()).hexdigest()==sha
             for f,sha in old_manifest['source_sha256'].items()}
    assert all(matches.values()),'Reused nominal results require unchanged core source'
    baseline=nominal_gate(json.loads((old/'metrics.json').read_text()))
    assert baseline['pass']
    primary_profiles=[profile(f'body_{i}',v,primary['onset'],r) for i,(v,r) in enumerate([
        ([-140,-90,0,0,0,0],.2),([-144,-90,0,0,0,0],.2),([-147,-90,0,0,0,0],.2),
        ([-144,-90,0,0,0,0],.1),([-147,-90,0,0,0,0],.1),([-144,-105,0,0,0,0],.2),
        ([-140,-90,-110,0,0,0],.2),([-144,-90,-110,0,0,0],.2),
        ([-147,-90,-110,0,0,0],.2),([-144,-105,-125,0,0,0],.2)])]
    secondary_profiles=[profile(f'wrist_body_{i}',v,secondary['onset']) for i,v in enumerate([
        [-130,0,0,0,-26,0],[-140,0,0,0,-26,0],[-145,0,0,0,-26,0],
        [-130,-90,0,0,-26,0],[-140,-90,0,0,-26,0],[-140,-90,0,0,-27.2,0]])]
    seeds=[1507,1517,1527,1537,1547]
    plan={'pilot_seed':1407,'heldout_seeds':seeds,'headroom_enabled':False,
        'primary_nominal_source':str(old),'primary_core_hash_matches':matches,'primary_nominal_gate':baseline,
        'primary_profiles':primary_profiles,'secondary_profiles':secondary_profiles,
        'secondary_nominal_gate':'same nominal criterion on seed 1407 before any secondary disturbances',
        'nominal_criterion':'all safe/contact-free; dynamic RMSE <=10mm; static5 RMSE >= dynamic +2mm AND >=1.2 times dynamic',
        'selection':'first passing pilot in listed order; primary family before secondary',
        'max_disturbed_pilot_profiles':16,'stop_on_failed_heldout':True,
        'heldout_cases':NAMES+['static35'],
        'heldout_primary_criterion':'lower mean violating-step fraction, lower worst position excess, lower mean cumulative excess than static25; lower mean EE RMSE than static5; all trajectories mesh-free at 5ms',
        'episode_count_reported_not_required_to_strictly_decrease':True,
        'zero_dynamic_violations_not_required':True,'per_seed_uniform_dominance_not_required':True,
        'no_parameter_retuning_after_viewing_heldout':True,
        'ramp_is_held_until_8s_episode_end':True,
        'onset_rule':'70% of initial-to-max upper-risk-joint excursion in continuous reference IK, positive velocity; minimum 0.5s',
        'normalization':'all cases retain fixed native 5%-span denominator'}
    (a.output/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    (a.output/'nominal_evidence.json').write_text(json.dumps({'primary':baseline,'source':str(old)},indent=2))
    summaries=[];selected=None
    for family,geometry,profiles in [('primary',primary,primary_profiles),('secondary',secondary,secondary_profiles)]:
        if family=='secondary':
            rows=run_experiment(a.output/'secondary_nominal',geometry,
                [profile('nominal',[0]*6,geometry['onset'])],[1407])
            gate=nominal_gate(rows)
            (a.output/'secondary_nominal_gate.json').write_text(json.dumps(gate,indent=2))
            if not gate['pass']:break
        for pr in profiles:
            directory=a.output/'pilot'/pr['name']
            rows=run_experiment(directory,geometry,[pr],[1407])
            gate=pilot_gate(rows)
            summaries.append({'family':family,'profile':pr,'directory':str(directory),'assessment':gate,
                'rows':rows})
            (a.output/'pilot_assessments.json').write_text(json.dumps(summaries,indent=2))
            print('PILOT_GATE',pr['name'],json.dumps(gate),flush=True)
            if gate['pass']:
                selected=(family,geometry,pr);break
        if selected:break
    if not selected:
        verdict={'status':'bounded_search_stopped_no_pilot_tradeoff','pilot_profiles_tested':len(summaries),
            'heldout_run':False,'headroom_enabled':False,'not_an_impossibility_certificate':True}
        (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2));print('MULTI_VERDICT',json.dumps(verdict),flush=True)
        return
    family,geometry,pr=selected
    (a.output/'locked_selection.json').write_text(json.dumps({'family':family,'profile':pr,
        'bank':geometry['bank'],'candidate':geometry['candidate'],'heldout_seeds':seeds},indent=2))
    run_experiment(a.output/'heldout',geometry,[pr],seeds,NAMES+['static35'])
    from kroc import audit_circle_validation as audit
    sys.argv=['audit',str(a.output/'heldout'),'--output',str(a.output/'audit'),
        '--mesh-dt','.005','--mesh-workers','6','--require-common-start',
        '--cases',*NAMES,'static35']
    audit.main()
    from kroc.multijoint_margin_assessment import assess,plot_results
    result=assess(a.output/'heldout',a.output/'audit/audited_results.json')
    (a.output/'heldout_assessment.json').write_text(json.dumps(result,indent=2))
    plot_results(result,a.output/'tradeoff_results.png')
    from kroc import plot_circle_floor25 as plot
    sys.argv=['plot',str(a.output/'heldout'),'--output',str(a.output/'heldout_comparison.png'),
        '--cases',*NAMES,'static35']
    plot.main()
    verdict={'status':'outer_tradeoff_replicated' if result['primary_pass'] else 'heldout_tradeoff_not_replicated_search_stopped',
        'primary_pass':result['primary_pass'],'middle_static_dominates_dynamic':result['middle_static_dominates_dynamic'],
        'headroom_enabled':False,'pilot_profiles_tested':len(summaries),
        'not_a_formal_safety_or_significance_guarantee':True}
    (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2));print('MULTI_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
