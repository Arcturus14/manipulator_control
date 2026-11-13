"""One final bounded radius refinement, only if coarse search found no pilot."""
import argparse
import json
from pathlib import Path
import sys
from kroc.benchmark_circle_multijoint_margin import load_circle,run_experiment
from kroc.search_circle_multijoint_tradeoff import nominal_gate,pilot_gate,profile,NAMES


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    previous=Path('kroc/results/circle_multijoint_tradeoff_20261114')
    prior=json.loads((previous/'verdict.json').read_text())
    assert prior['status']=='bounded_search_stopped_no_pilot_tradeoff','Never retune after observing held-out data'
    a.output.mkdir(parents=True,exist_ok=False)
    geometry=load_circle('kroc/results/shape_margin_bank_20261113',19,0)
    profiles=[profile(f'larger_{i}',v,geometry['onset'],r) for i,(v,r) in enumerate([
        ([-144,-90,0,0,0,0],.2),([-147,-90,0,0,0,0],.2),
        ([-144,-90,0,0,0,0],.1),([-147,-90,-110,0,0,0],.2)])]
    seeds=[1507,1517,1527,1537,1547]
    plan={'previous_search':str(previous),'reason':'Test closer constraint approach by increasing circle radius, not by escalating torque beyond previous bounds',
        'candidate':19,'bank':geometry['bank'],'radius_m':geometry['radius'],'onset_s':geometry['onset'],
        'profiles':profiles,'pilot_seed':1407,'heldout_seeds':seeds,'max_profiles':4,
        'nominal_gate':'unchanged: dynamic <=10mm, static5 >= dynamic+2mm AND >=1.2 times dynamic; all safe and contact-free',
        'pilot_gate':'unchanged risk/time/integral improvement, dynamic <30mm and >=5% RMSE improvement over static5; no zero-violation requirement',
        'heldout_cases':NAMES+['static35'],'headroom_enabled':False,
        'stop_after_this_refinement':True,'no_heldout_data_seen_when_refining':True}
    (a.output/'locked_plan.json').write_text(json.dumps(plan,indent=2))
    nominal=run_experiment(a.output/'nominal',geometry,[profile('nominal',[0]*6,geometry['onset'])],[1407])
    gate=nominal_gate(nominal)
    (a.output/'nominal_gate.json').write_text(json.dumps(gate,indent=2))
    if not gate['pass']:
        verdict={'status':'stopped_nominal_tracking_pressure_gate_failed','headroom_enabled':False}
        (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2));print('RADIUS_VERDICT',json.dumps(verdict),flush=True);return
    summaries=[];selected=None
    for pr in profiles:
        directory=a.output/'pilot'/pr['name']
        rows=run_experiment(directory,geometry,[pr],[1407]);gate=pilot_gate(rows)
        summaries.append({'family':'larger_circle','profile':pr,'directory':str(directory),'assessment':gate,'rows':rows})
        (a.output/'pilot_assessments.json').write_text(json.dumps(summaries,indent=2))
        print('RADIUS_PILOT_GATE',pr['name'],json.dumps(gate),flush=True)
        if gate['pass']:selected=pr;break
    from kroc.multijoint_margin_assessment import plot_pilot_summary
    plot_pilot_summary(summaries,a.output/'pilot_summary.png')
    if selected is None:
        verdict={'status':'stopped_no_pilot_tradeoff_after_radius_refinement','headroom_enabled':False,
            'not_an_impossibility_certificate':True}
        (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2));print('RADIUS_VERDICT',json.dumps(verdict),flush=True);return
    (a.output/'locked_selection.json').write_text(json.dumps({'profile':selected,'heldout_seeds':seeds},indent=2))
    run_experiment(a.output/'heldout',geometry,[selected],seeds,NAMES+['static35'])
    from kroc import audit_circle_validation as audit
    sys.argv=['audit',str(a.output/'heldout'),'--output',str(a.output/'audit'),
        '--mesh-dt','.005','--mesh-workers','6','--require-common-start','--cases',*NAMES,'static35']
    audit.main()
    from kroc.multijoint_margin_assessment import assess,plot_results
    result=assess(a.output/'heldout',a.output/'audit/audited_results.json')
    (a.output/'heldout_assessment.json').write_text(json.dumps(result,indent=2))
    plot_results(result,a.output/'tradeoff_results.png')
    from kroc import plot_circle_floor25 as plot
    sys.argv=['plot',str(a.output/'heldout'),'--output',str(a.output/'heldout_comparison.png'),'--cases',*NAMES,'static35']
    plot.main()
    verdict={'status':'outer_tradeoff_replicated' if result['primary_pass'] else 'stopped_heldout_not_replicated',
        'primary_pass':result['primary_pass'],'middle_static_dominates_dynamic':result['middle_static_dominates_dynamic'],
        'headroom_enabled':False,'not_a_formal_safety_or_significance_guarantee':True}
    (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2));print('RADIUS_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
