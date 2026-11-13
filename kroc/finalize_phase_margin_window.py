"""Finalize tested claims from held-out data and offline mesh audit; no MD."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    a=p.parse_args();d=a.directory
    raw=json.loads((d/'verdict.json').read_text())
    audit=json.loads((d/'audit/audited_results.json').read_text())
    rows=json.loads((d/'heldout/metrics.json').read_text())
    seeds=sorted({r['seed'] for r in rows})
    names=['static25','directional25_5','static5'];cases={};paired={}
    for name in names:
        rr=[r for r in rows if r['case']==name]
        collision_free=audit['aggregate'][name]['mesh_collision_trajectories']==0
        peaks=[];margins=[]
        for r in rr:
            with np.load(d/'heldout'/f's0_seed{r["seed"]}_{name}.npz') as z:
                peaks.append(float(z['err_mm'].max()))
                margins.append([float(z['kroc_margin'][:,0,1,0].min()/(4*np.pi)*100),
                                float(z['kroc_margin'][:,0,1,0].max()/(4*np.pi)*100)])
        rms=[r['metrics']['tracking_rmse_mm'] for r in rr]
        cases[name]={**audit['aggregate'][name],'collision_free_dense_mesh':collision_free,
            'validated_contact_free_tracking_rmse_mean_mm':float(np.mean(rms)) if collision_free else None,
            'tracking_peak_mm_per_seed':dict(zip(seeds,peaks)),
            'risk_upper_margin_percent_range_per_seed':dict(zip(seeds,margins))}
    for seed in seeds:
        by={r['case']:r['metrics'] for r in rows if r['seed']==seed}
        paired[seed]={'low_violated':by['static25']['position_violation_max_rad']>1e-4,
            'dynamic_position_safe':by['directional25_5']['position_violation_max_rad']<=1e-6,
            'high_position_safe':by['static5']['position_violation_max_rad']<=1e-6,
            'dynamic_rmse_lower_than_high':by['directional25_5']['tracking_rmse_mm']<by['static5']['tracking_rmse_mm'],
            'dynamic_rmse_improvement_over_high_mm':by['static5']['tracking_rmse_mm']-by['directional25_5']['tracking_rmse_mm']}
    all_clear=all(c['collision_free_dense_mesh'] for c in cases.values())
    full=bool(raw['heldout_validated'] and all_clear)
    result={'status':'all_predeclared_heldout_gates_passed' if full else 'not_fully_reproduced',
        'full_predeclared_claim_verified':full,'seed_count':len(seeds),'seeds':seeds,
        'all_dense_mesh_collision_free':all_clear,'mesh_interpolation_dt_s':audit['mesh_sample_dt_s'],
        'mesh_sampling_is_not_continuous_safety_certificate':True,
        'headroom_enabled':False,'paired':paired,'cases':cases,
        'pilot_is_excluded_from_heldout_statistics':True,
        'normalization':'same fixed native 5%-span denominator for every case',
        'scope':'position constraints only; no claim of eliminating actuator saturation or uniformly minimizing tracking peaks',
        'not_a_formal_robustness_guarantee':True}
    (d/'final_verdict.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
