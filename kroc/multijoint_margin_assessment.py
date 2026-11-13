"""Paired risk/tracking assessment including an intermediate fixed-margin control."""
import json
from pathlib import Path
import numpy as np
from kroc.circle_risk_tradeoff_assessment import position_metrics

NAMES=['static25','directional25_5','static5','static35']


def dominates(a,b,tol=1e-8):
    keys=['mean_violation_step_fraction','worst_violation_rad',
          'mean_cumulative_violation_rad_s','tracking_rmse_mean_mm']
    return all(a[k]<=b[k]+tol for k in keys) and any(a[k]<b[k]-tol for k in keys)


def assess(directory,audit_file):
    directory=Path(directory);audit=json.loads(Path(audit_file).read_text())
    rows=json.loads((directory/'metrics.json').read_text());records=[]
    seeds=sorted({r['seed'] for r in rows});aggregate={}
    for row in rows:
        with np.load(directory/f's{row["scenario"]}_seed{row["seed"]}_{row["case"]}.npz') as z:
            risk=position_metrics(z['q'])
            records.append({'seed':row['seed'],'case':row['case'],**risk,
                'tracking_rmse_mm':float(np.sqrt(np.mean(z['err_mm']**2))),
                'tracking_peak_mm':float(z['err_mm'].max()),
                'saturation_duration_per_joint_s':row['metrics']['saturation_duration_per_joint_s']})
    assert len(records)==len(seeds)*len(NAMES)
    for name in NAMES:
        rr=[r for r in records if r['case']==name]
        def mean(k):return float(np.mean([r[k] for r in rr]))
        aggregate[name]={'seed_count':len(seeds),'violation_episodes':sum(r['violation_episode'] for r in rr),
            'mean_violation_step_fraction':mean('violation_step_fraction'),
            'worst_violation_rad':max(r['maximum_violation_rad'] for r in rr),
            'mean_cumulative_violation_rad_s':mean('cumulative_violation_rad_s'),
            'tracking_rmse_mean_mm':mean('tracking_rmse_mm'),
            'tracking_rmse_sd_mm':float(np.std([r['tracking_rmse_mm'] for r in rr],ddof=1)),
            'tracking_peak_worst_mm':max(r['tracking_peak_mm'] for r in rr),
            'mean_saturation_duration_per_joint_s':np.mean([r['saturation_duration_per_joint_s'] for r in rr],axis=0).tolist(),
            'mesh_collision_trajectories':audit['aggregate'][name]['mesh_collision_trajectories']}
    low,dyn,high,mid=[aggregate[n] for n in NAMES]
    clear=all(m['mesh_collision_trajectories']==0 for m in aggregate.values())
    gates={'less_violating_time_than_low':dyn['mean_violation_step_fraction']<low['mean_violation_step_fraction'],
        'smaller_worst_violation_than_low':dyn['worst_violation_rad']<low['worst_violation_rad'],
        'less_cumulative_violation_than_low':dyn['mean_cumulative_violation_rad_s']<low['mean_cumulative_violation_rad_s'],
        'lower_mean_tracking_rmse_than_high':dyn['tracking_rmse_mean_mm']<high['tracking_rmse_mean_mm'],
        'all_mesh_collision_free':clear}
    paired=[]
    for seed in seeds:
        by={r['case']:r for r in records if r['seed']==seed}
        paired.append({'seed':seed,'dynamic_minus_high_rmse_mm':by['directional25_5']['tracking_rmse_mm']-by['static5']['tracking_rmse_mm'],
            'dynamic_minus_mid_rmse_mm':by['directional25_5']['tracking_rmse_mm']-by['static35']['tracking_rmse_mm'],
            'dynamic_minus_low_cumulative_rad_s':by['directional25_5']['cumulative_violation_rad_s']-by['static25']['cumulative_violation_rad_s']})
    rng=np.random.default_rng(261114)
    def ci(key):
        values=np.array([r[key] for r in paired])
        boot=values[rng.integers(0,len(values),(20000,len(values)))].mean(axis=1)
        return np.quantile(boot,[.025,.975]).tolist()
    return {'primary_pass':all(gates.values()),'gates':gates,'seeds':seeds,'aggregate':aggregate,
        'per_seed':records,'paired':paired,'headroom_enabled':False,
        'middle_static_dominates_dynamic':dominates(mid,dyn) if clear else None,
        'dynamic_dominates_middle_static':dominates(dyn,mid) if clear else None,
        'pareto_comparison_is_sample_point_estimate_only':True,
        'bootstrap_dynamic_minus_high_rmse95_ci_mm':ci('dynamic_minus_high_rmse_mm'),
        'bootstrap_dynamic_minus_mid_rmse95_ci_mm':ci('dynamic_minus_mid_rmse_mm'),
        'bootstrap_small_sample_exploratory':True,'mesh_interpolation_s':audit['mesh_sample_dt_s'],
        'mesh_sampling_is_not_continuous_safety_certificate':True,
        'post_contact_tracking_not_validated':not clear,
        'zero_dynamic_violation_or_per_seed_dominance_not_required':True}


def plot_results(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Static 2.5%','Dynamic 2.5-5%','Static 5%','Static 3.5%'];rows=[]
    for name,label in zip(NAMES,labels):
        m=result['aggregate'][name]
        rows.append([label,f'{m["violation_episodes"]}/{m["seed_count"]}',
            f'{m["mean_violation_step_fraction"]*100:.2f}',f'{m["worst_violation_rad"]:.5f}',
            f'{m["mean_cumulative_violation_rad_s"]:.5f}',
            f'{m["tracking_rmse_mean_mm"]:.2f} +/- {m["tracking_rmse_sd_mm"]:.2f}',
            str(m['mesh_collision_trajectories'])])
    fig,ax=plt.subplots(figsize=(16,4));ax.axis('off')
    tab=ax.table(cellText=rows,colLabels=['Case','Violation\nepisodes','Violating\nsteps [%]',
        'Worst q excess\n[rad]','Mean cumulative\nexcess [rad s]','EE RMSE\nmean +/- SD [mm]','Mesh contact\ntrajectories'],
        cellLoc='center',loc='center',colWidths=[.16,.10,.12,.14,.16,.20,.12])
    tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2.1)
    title=f'Multi-joint sustained disturbance: {len(result["seeds"])} held-out seeds; noisy Post-DOB; no headroom'
    if not result['gates']['all_mesh_collision_free']:title+='\nDIAGNOSTIC: contact dynamics not modeled; full-episode tracking is not validated'
    ax.set_title(title,pad=18,color='black' if result['gates']['all_mesh_collision_free'] else 'crimson')
    fig.tight_layout();fig.savefig(output,dpi=170);plt.close(fig)


def plot_pilot_summary(summaries,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for item in summaries:
        by={r['case']:r['metrics'] for r in item['rows']}
        order=[by[n] for n in NAMES[:3]]
        vec=', '.join(f'J{i+1}:{v:g}' for i,v in enumerate(item['profile']['vector_nm']) if v)
        rows.append([item['profile']['name'],vec,
            ' / '.join(f'{m["tracking_rmse_mm"]:.2f}' for m in order),
            ' / '.join(f'{m["position_violation_max_rad"]:.4f}' for m in order),
            ' / '.join(f'{m["position_violation_integral_rad_s"]:.4f}' for m in order),
            ' / '.join('hit' if m['first_collision_index'] is not None else 'clear' for m in order),
            'pass' if item['assessment']['pass'] else 'no'])
    fig,ax=plt.subplots(figsize=(18,max(3.,1.8+.4*len(rows))));ax.axis('off')
    tab=ax.table(cellText=rows,colLabels=['Profile','Disturbance [Nm]','EE RMSE [mm]\nlow / dynamic / high',
        'Worst q excess [rad]\nlow / dynamic / high','Cumulative [rad s]\nlow / dynamic / high',
        'Mesh @20ms\nlow / dynamic / high','Pilot gate'],cellLoc='center',loc='center',
        colWidths=[.12,.21,.17,.18,.18,.12,.07])
    tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.8)
    ax.set_title('Exploratory pilot seed 1407; circle; noisy Post-DOB; no headroom\n'
        'All profiles retained. Contact rows are diagnostic only: post-contact physics are not modeled.',pad=18)
    fig.tight_layout();fig.savefig(output,dpi=170);plt.close(fig)
