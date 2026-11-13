"""All-joint position risk and paired tracking analysis, independent of MPPI."""
import json
from pathlib import Path
import numpy as np
from config import DT,Q_MAX,Q_MIN

NAMES=['static25','directional25_5','static5']


def comparison_gates(aggregate):
    low,dyn,high=[aggregate[n] for n in NAMES]
    return {'fewer_violation_episodes':dyn['violation_episodes']<low['violation_episodes'],
        'less_violating_time':dyn['mean_violation_step_fraction']<low['mean_violation_step_fraction'],
        'smaller_worst_violation':dyn['worst_violation_rad']<low['worst_violation_rad'],
        'less_cumulative_violation':dyn['mean_cumulative_violation_rad_s']<low['mean_cumulative_violation_rad_s'],
        'lower_mean_tracking_rmse_than_high':dyn['tracking_rmse_mean_mm']<high['tracking_rmse_mean_mm'],
        'all_mesh_collision_free':all(c['mesh_collision_trajectories']==0 for c in aggregate.values())}


def position_metrics(q,dt=DT,tolerance=1e-6):
    """q includes initial state; integrate positive excess over all six joints."""
    q=np.asarray(q,dtype=float)[1:]
    excess=np.maximum(np.maximum(np.asarray(Q_MIN)-q,q-np.asarray(Q_MAX)),0.)
    excess=np.where(excess>tolerance,excess,0.)
    step=(excess>0).any(axis=1)
    return {'violation_episode':bool(step.any()),'violating_steps':int(step.sum()),
        'violation_step_fraction':float(step.mean()),'violation_duration_s':float(step.sum()*dt),
        'maximum_violation_rad':float(excess.max()),
        'cumulative_violation_rad_s':float(excess.sum()*dt)}


def assess(directory,audit_file):
    directory=Path(directory);audit=json.loads(Path(audit_file).read_text())
    rows=json.loads((directory/'metrics.json').read_text())
    manifest=json.loads((directory/'manifest.json').read_text())
    assert manifest['arguments']['radii']==[.08] and manifest['arguments']['joint']==4
    assert not manifest['headroom_enabled']
    records=[];reference=disturbance=initial=None
    for row in rows:
        path=directory/f's{row["scenario"]}_seed{row["seed"]}_{row["case"]}.npz'
        with np.load(path) as z:
            assert str(z['compute_device'])=='cuda' and str(z['dob_compensation_mode'])=='output'
            if reference is None:
                reference=z['circle_reference'].copy();disturbance=z['d_true'].copy();initial=z['q'][0].copy()
            assert np.array_equal(z['circle_reference'],reference)
            assert np.array_equal(z['d_true'],disturbance)
            assert np.array_equal(z['q'][0],initial)
            risk=position_metrics(z['q'])
            records.append({'seed':row['seed'],'case':row['case'],**risk,
                'tracking_rmse_mm':float(np.sqrt(np.mean(z['err_mm']**2))),
                'tracking_peak_mm':float(z['err_mm'].max()),
                'wrist_saturation_s':row['metrics']['test_joint_saturation_steps']*DT})
    seeds=sorted({r['seed'] for r in records})
    assert len(records)==len(seeds)*3
    aggregate={}
    for name in NAMES:
        rr=[r for r in records if r['case']==name]
        def mean(key):return float(np.mean([r[key] for r in rr]))
        aggregate[name]={'seed_count':len(rr),'violation_episodes':sum(r['violation_episode'] for r in rr),
            'mean_violation_step_fraction':mean('violation_step_fraction'),
            'worst_violation_rad':max(r['maximum_violation_rad'] for r in rr),
            'mean_cumulative_violation_rad_s':mean('cumulative_violation_rad_s'),
            'mean_violation_duration_s':mean('violation_duration_s'),
            'tracking_rmse_mean_mm':mean('tracking_rmse_mm'),
            'tracking_rmse_sd_mm':float(np.std([r['tracking_rmse_mm'] for r in rr],ddof=1)),
            'worst_tracking_peak_mm':max(r['tracking_peak_mm'] for r in rr),
            'mean_wrist_saturation_s':mean('wrist_saturation_s'),
            'mesh_collision_trajectories':audit['aggregate'][name]['mesh_collision_trajectories']}
    gates=comparison_gates(aggregate)
    paired=[]
    for seed in seeds:
        by={r['case']:r for r in records if r['seed']==seed}
        paired.append({'seed':seed,'dynamic_minus_high_rmse_mm':by['directional25_5']['tracking_rmse_mm']-by['static5']['tracking_rmse_mm'],
            'dynamic_minus_low_integral_rad_s':by['directional25_5']['cumulative_violation_rad_s']-by['static25']['cumulative_violation_rad_s']})
    diff=np.array([r['dynamic_minus_high_rmse_mm'] for r in paired])
    rng=np.random.default_rng(260113)
    bootstrap=diff[rng.integers(0,len(diff),size=(20000,len(diff)))].mean(axis=1)
    result={'source':str(directory),'seeds':seeds,'primary_pass':all(gates.values()),'gates':gates,
        'aggregate':aggregate,'per_seed':records,'paired':paired,
        'paired_dynamic_minus_high_rmse_mean_mm':float(diff.mean()),
        'paired_rmse_bootstrap95_percentile_ci_mm':np.quantile(bootstrap,[.025,.975]).tolist(),
        'dynamic_tracking_wins_ties_losses_vs_high':[int((diff<-1e-6).sum()),int((np.abs(diff)<=1e-6).sum()),int((diff>1e-6).sum())],
        'bootstrap_small_sample_is_exploratory':True,'headroom_enabled':False,
        'position_violation_tolerance_rad':1e-6,
        'cumulative_definition':'dt * sum over time AND all joints of positive position-limit excess',
        'collision_audit':str(audit_file),'mesh_interpolation_s':audit['mesh_sample_dt_s'],
        'mesh_sampling_is_not_a_continuous_safety_certificate':True,
        'full_episode_tracking_is_valid_contact_free_physics':gates['all_mesh_collision_free']}
    return result


def plot_assessment(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Static 2.5%','Dynamic 2.5-5%','Static 5%'];rows=[]
    for name,label in zip(NAMES,labels):
        m=result['aggregate'][name]
        rows.append([label,f'{m["violation_episodes"]}/{m["seed_count"]}',
            f'{m["mean_violation_step_fraction"]*100:.2f}',f'{m["worst_violation_rad"]:.5f}',
            f'{m["mean_cumulative_violation_rad_s"]:.5f}',
            f'{m["tracking_rmse_mean_mm"]:.2f} +/- {m["tracking_rmse_sd_mm"]:.2f}',
            str(m['mesh_collision_trajectories'])])
    fig,ax=plt.subplots(figsize=(16,3.5));ax.axis('off')
    tab=ax.table(cellText=rows,colLabels=['Case','Violation\nepisodes','Violating\nsteps [%]',
        'Worst q excess\n[rad]','Mean cumulative\nexcess [rad s]','EE RMSE\nmean +/- SD [mm]','Mesh contact\ntrajectories'],
        cellLoc='center',loc='center',colWidths=[.16,.10,.12,.14,.16,.20,.12])
    tab.auto_set_font_size(False);tab.set_fontsize(11);tab.scale(1,2.1)
    title=f'Circle-only locked replication: {len(result["seeds"])} fresh seeds; noisy Post-DOB; no headroom'
    if not result['gates']['all_mesh_collision_free']:
        title+='\nDIAGNOSTIC ONLY: full-episode RMSE includes unmodeled post-contact motion'
    ax.set_title(title,pad=16,color='black' if result['gates']['all_mesh_collision_free'] else 'crimson')
    fig.tight_layout();fig.savefig(output,dpi=170);plt.close(fig)
