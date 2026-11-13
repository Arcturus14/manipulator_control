"""Combine locked held-out results with dense mesh audit; JSON/PNG only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import DT,Q_MIN,Q_MAX


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    a=p.parse_args();root=a.directory;output=root/'final/figures'
    selection=json.loads((root/'selection.json').read_text())
    audit=json.loads((output/'audited_results.json').read_text())
    assert audit['mesh_sample_dt_s']==.005 and audit['common_start_required']
    assert audit['seeds']==selection['plan']['final_seeds']
    manifest=json.loads((root/'final/manifest.json').read_text())
    assert not manifest['headroom_enabled']
    assert all(c['mode'] in ['fixed','directional'] and not c['output_compensated_headroom'] for c in manifest['cases'].values())
    names=['static25','directional25_5','static5'];verdicts={};table=[];prefix_metrics={}
    for seed in audit['seeds']:
        cases={r['case']:r['metrics'] for r in audit['per_seed'] if r['seed']==seed}
        tests=dict(selection['final_by_seed'][str(seed)])
        tests['all_collision_free']=all(not m['mesh_interpolated']['self_collision_states']
            and not m['mesh_interpolated']['floor_collision_states'] for m in cases.values())
        tests['actuator_equation_verified']=all(m['clamp_audit_max_nm']<1e-4 for m in cases.values())
        tests['common_start_verified']=all(m['initial_common_box_min_slack_rad']>0 for m in cases.values())
        tests['full_relation']=all(v for k,v in tests.items() if k not in ['complete','full_relation'])
        verdicts[seed]=tests
        contacts=[m['mesh_interpolated']['first_collision_index']*.005 for m in cases.values()
                  if m['mesh_interpolated']['first_collision_index'] is not None]
        cutoff=min(contacts) if contacts else selection['plan']['final_seconds']+1e-8
        prefix_metrics[seed]={'common_prefix_end_seconds':min(cutoff,selection['plan']['final_seconds']),
                             'terminated_by_contact':bool(contacts),'cases':{}}
        for case,m in cases.items():
            with np.load(root/'final'/f's0_seed{seed}_{case}.npz') as z:
                state_time=(np.arange(len(z['err_mm']))+1)*DT
                valid=state_time<cutoff
                q=z['q'][1:];distance=np.minimum(q-np.asarray(Q_MIN),np.asarray(Q_MAX)-q)
                ix=np.flatnonzero((distance<0).any(1))
                own_contact=m['mesh_interpolated']['first_collision_index']
                own_contact_time=own_contact*.005 if own_contact is not None else None
                own_valid=state_time<own_contact_time if own_contact_time is not None else np.ones(len(state_time),dtype=bool)
                prefix_metrics[seed]['cases'][case]={
                    'common_prefix_samples':int(valid.sum()),
                    'common_prefix_tracking_rmse_mm':float(np.sqrt(np.mean(z['err_mm'][valid]**2))) if valid.any() else None,
                    'first_position_violation_seconds':float(state_time[ix[0]]) if len(ix) else None,
                    'first_mesh_collision_seconds':own_contact_time,
                    'max_position_violation_before_contact_rad':float(np.maximum(-distance[own_valid],0).max()) if own_valid.any() else None,
                    'position_violation_before_own_collision':bool(len(ix) and (own_contact_time is None or state_time[ix[0]]<own_contact_time))}
        colliding=[c for c,m in cases.items() if m['mesh_interpolated']['self_collision_states'] or m['mesh_interpolated']['floor_collision_states']]
        table.append([seed,*[f'{cases[c]["position_violation_max_rad"]:.4f}' for c in names],
                      *[f'{cases[c]["tracking_rmse_mm"]:.2f}' for c in names],
                      '\n'.join({'static25':'Static 2.5%','directional25_5':'Dynamic','static5':'Static 5%'}[c] for c in colliding) or 'None',
                      'Yes' if tests['full_relation'] else 'No'])
    with np.load('kroc/results/nominal_margin5_search_20261113/validation/s0_seed307_static25.npz') as z:
        nominal_ref=z['circle_reference'].copy()
    with np.load(root/'final'/f's0_seed{audit["seeds"][0]}_static25.npz') as z:
        same_reference=np.array_equal(nominal_ref,z['circle_reference'][:len(nominal_ref)])
    assert same_reference
    success=[s for s,t in verdicts.items() if t['full_relation']]
    precontact_aggregate={}
    for case in names:
        ms=[prefix_metrics[s]['cases'][case] for s in audit['seeds']]
        rms=[m['common_prefix_tracking_rmse_mm'] for m in ms]
        precontact_aggregate[case]={
            'position_violation_before_contact_seeds':sum(m['position_violation_before_own_collision'] for m in ms),
            'worst_position_violation_before_contact_rad':max(m['max_position_violation_before_contact_rad'] for m in ms),
            'paired_common_prefix_rmse_mean_mm':float(np.mean(rms)),
            'paired_common_prefix_rmse_std_mm':float(np.std(rms,ddof=1)) if len(rms)>1 else 0.,
            'mesh_collision_trajectories':audit['aggregate'][case]['mesh_collision_trajectories']}
    result={'status':'verified_in_tested_scenario' if len(success)==len(audit['seeds']) else 'not_fully_verified',
        'scope':'finite simulated scenario, not a formal robust-safety guarantee',
        'selected_amplitude_nm':selection['selected_amplitude_nm'],'pilot_selection':selection['selection'],
        'final_duration_seconds':selection['plan']['final_seconds'],'external_disturbance_released':False,
        'headroom_enabled':False,'nominal_reference_prefix_identical':bool(same_reference),
        'seed_count':len(audit['seeds']),'full_relation_satisfied_seeds':success,
        'per_seed_verdict':verdicts,'aggregate':audit['aggregate'],
        'paired_precontact_prefix':prefix_metrics,
        'precontact_aggregate':precontact_aggregate,
        'full_episode_rmse_includes_invalid_postcontact_physics_when_collision_occurs':True,
        'audit':str(output/'audited_results.json'),
        'dense_mesh_sampling_seconds':.005,'continuous_collision_certificate':False}
    (root/'final_verdict.json').write_text(json.dumps(result,indent=2))
    fig,ax=plt.subplots(figsize=(17,2.4+.45*len(table)));ax.axis('off')
    tab=ax.table(cellText=table,colLabels=['Seed','Max viol.\nStatic 2.5% [rad]',
        'Max viol.\nDynamic [rad]','Max viol.\nStatic 5% [rad]',
        'RMSE\nStatic 2.5% [mm]','RMSE\nDynamic [mm]','RMSE\nStatic 5% [mm]',
        'Dense-mesh\ncolliding cases','All criteria\nsame seed?'],loc='center',cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,2.5)
    ax.set_title('Locked held-out final verification | sustained disturbance | 2 circles | no headroom\n'
        'All criteria: low static violates/saturates; dynamic/high static safe; dynamic lower RMSE; all cases collision-free\n'
        'Full-run RMSE is diagnostic only for colliding runs: post-contact dynamics are not modeled.',pad=15)
    fig.savefig(output/'final_seed_table.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
