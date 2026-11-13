"""Bounded nominal 5% pressure pilot, then held-out validation in one process."""
import argparse
import json
from pathlib import Path
import sys
from kroc.benchmark_nominal_margin_pressure import main as benchmark


def invoke(screen, output, candidates, seeds):
    previous=sys.argv
    sys.argv=['benchmark_nominal_margin_pressure','--screen',str(screen),'--output',str(output),
        '--candidates',*map(str,candidates),'--seeds',*map(str,seeds),
        '--static-fractions','.025','.05','--period','8','--seconds','8']
    try:benchmark()
    finally:sys.argv=previous


def passes(rows):
    c={r['case']:r['metrics'] for r in rows}
    if set(c)!={'static25','directional25_5','static5'}:return False
    if any(m['self_collision_states'] or m['floor_collision_states'] or m['position_violation_steps'] for m in c.values()):return False
    baseline=max(c[n]['tracking_rmse_mm'] for n in ['static25','directional25_5'])
    return (baseline<10 and c['static5']['tracking_rmse_mm']-baseline>2
            and c['static5']['tracking_rmse_mm']>1.2*c['directional25_5']['tracking_rmse_mm'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    plan={'pilot_candidates':[63,61,53],'pilot_seed':207,'held_out_seeds':[307,317,327],
          'period_s':8,'headroom_enabled':False,'external_disturbance':False,
          'selection':'all actual states collision/position safe; low and dynamic RMSE <10mm; static5 worse than BOTH by >2mm and >20% versus dynamic',
          'candidate_selection':'moderate horizontal, strong horizontal, strong vertical local-IK pressure; no MPPI pilot results used'}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
    invoke(a.screen,a.output/'pilot',plan['pilot_candidates'],[plan['pilot_seed']])
    rows=json.loads((a.output/'pilot/metrics.json').read_text());selected=None
    for candidate in plan['pilot_candidates']:
        if passes([r for r in rows if r['candidate']==candidate]):selected=candidate;break
    result={**plan,'selected_candidate':selected,'status':'no_pilot_candidate'}
    if selected is not None:
        invoke(a.screen,a.output/'validation',[selected],plan['held_out_seeds'])
        rows=json.loads((a.output/'validation/metrics.json').read_text())
        result['pass_by_seed']={s:passes([r for r in rows if r['seed']==s]) for s in plan['held_out_seeds']}
        result['status']='held_out_pass' if all(result['pass_by_seed'].values()) else 'held_out_mixed'
    (a.output/'selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
