"""Bracket the observed phase-sensitive boundary, then lock held-out seeds."""
import argparse
import json
from pathlib import Path
import sys
from kroc import search_shape_margin_tradeoff as search
from kroc import benchmark_nominal_margin_pressure as benchmark


def passes(group):
    by={r['case']:r['metrics'] for r in group}
    low,dyn,high=[by[n] for n in ('static25','directional25_5','static5')]
    return (low['position_violation_max_rad']>1e-4 and low['test_joint_saturation_steps']>0
        and dyn['position_violation_max_rad']<=1e-6 and high['position_violation_max_rad']<=1e-6
        and all(m['first_collision_index'] is None for m in by.values())
        and dyn['tracking_rmse_mm']<=25.
        and high['tracking_rmse_mm']-dyn['tracking_rmse_mm']>=2.)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path('kroc/results');bank=root/'shape_margin_bank_20261113'
    plan={'candidate':22,'amplitudes_nm':[142.,144.,145.,146.],
        'onset_s':5.42,'ramp_s':.1,'sustained_to_end':True,'pilot_seed':807,
        'selection':'lowest amplitude passing full predeclared gate, not best tracking',
        'heldout_seeds':[1007,1017,1027],'heldout_cycles':1,'headroom_enabled':False,
        'full_gate':'low violation >1e-4 plus saturation; dynamic/high safe; all mesh-free @20ms; dynamic RMSE <=25mm and >=2mm better than high',
        'discovery_source':'margin_mechanism_probe_20261113/phase_rectangle at 147 Nm'}
    (a.output/'plan.json').write_text(json.dumps(plan,indent=2))
    pilot=a.output/'pilot'
    sys.argv=['search','--bank',str(bank),'--output',str(pilot),
        '--reuse-nominal',str(root/'shape_margin_tradeoff_20261113'),
        '--disturbed-candidates','22','--onset-mode','nominal-phase','--ramp','.1',
        '--amplitudes',*[str(v) for v in plan['amplitudes_nm']]]
    search.main()
    rows=json.loads((pilot/'disturbed_c22/metrics.json').read_text())
    eligible=[amp for amp in plan['amplitudes_nm'] if passes([r for r in rows if r['amplitude']==amp])]
    verdict={'eligible_pilot_amplitudes':eligible,'heldout_validated':False}
    if eligible:
        amp=min(eligible);heldout=a.output/'heldout'
        (a.output/'locked_selection.json').write_text(json.dumps({'amplitude':amp,**plan},indent=2))
        sys.argv=['benchmark','--screen',str(bank),'--candidates','22',
            '--static-fractions','.025','.05','--dynamic-max','.05',
            '--seeds',*[str(s) for s in plan['heldout_seeds']],
            '--output',str(heldout),'--use-banked-reference','--onset','5.42',
            '--ramp','.1','--amplitudes',str(amp)]
        benchmark.main()
        validation=json.loads((heldout/'metrics.json').read_text())
        paired={s:passes([r for r in validation if r['seed']==s]) for s in plan['heldout_seeds']}
        verdict.update(selected_amplitude=amp,heldout_paired_pass=paired,
            heldout_validated=all(paired.values()),dense_mesh_audit_still_required=True)
    verdict['status']='heldout_20ms_mesh_pass_pending_dense_audit' if verdict['heldout_validated'] else 'not_fully_verified'
    (a.output/'verdict.json').write_text(json.dumps(verdict,indent=2))
    print('WINDOW_VERDICT',json.dumps(verdict),flush=True)


if __name__=='__main__':main()
