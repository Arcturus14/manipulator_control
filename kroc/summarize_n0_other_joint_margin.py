"""Public selected-condition figures/JSON; no automatic markdown report."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import DT,Q_MAX
from kroc.search_n0_other_joint_estimator_margin import CASES,LABELS,COLORS

ROOT=Path('kroc/results/n0_other_joint_estimator_margin_20261118/final_refinement')


def main():
    result=json.loads((ROOT/'verdict.json').read_text())
    plan=json.loads((ROOT/'plan.json').read_text())
    for file,expected in plan['source_sha256'].items():
        assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected,file
    pairs=result['validation'];assert len(pairs)==3
    valid=[row for pair in pairs for row in pair['rows'] if row['eligible']]
    assert len(valid)==9,'Do not compare invalid trials'
    groups={}
    for case in CASES:
        rows=[r for r in valid if r['case']==case];m=[r['metrics'] for r in rows]
        groups[case]=dict(runs=len(rows),
            violation_runs=sum(x['max_position_excess_rad']>1e-6 for x in m),
            worst_position_excess_rad=max(x['max_position_excess_rad'] for x in m),
            mean_violation_duration_s=float(np.mean([x['violation_duration_s'] for x in m])),
            mean_excess_integral_rad_s=float(np.mean([x['excess_integral_rad_s'] for x in m])),
            mean_rmse_mm=float(np.mean([x['rmse_mm'] for x in m])),
            std_rmse_mm=float(np.std([x['rmse_mm'] for x in m],ddof=1)),
            min_upper_clearance_rad=min(x['risk_joint_upper_clearance_rad'] for x in m),
            saturation_steps=sum(x['saturation_steps'] for x in m),
            max_upper_margin_pct=max(x['margin_upper_max_pct'] for x in m),
            same_state_dhat_upper_extra_max_rad=max(x['same_state_dhat_upper_extra_max_rad'] for x in m))
    summary=dict(status=result['status'],scene=result['selected_scene'],
        amplitude_nm=result['selected_amplitude_nm'],heldout_seeds=[p['seed'] for p in pairs],
        primary_successes=sum(p['primary'] for p in pairs),
        secondary_successes=sum(p['secondary'] for p in pairs),groups=groups,
        validation=pairs,post_compensation=False,rollout_disturbance=False,
        time_window_s=[0.,6.],collision_check_dt_s=.005,
        no_saturation_mechanism_claim=True,no_closed_loop_velocity_only_ablation=True)
    (ROOT/'selected_condition_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    fig,axs=plt.subplots(2,3,figsize=(16,8),layout='constrained')
    all_end=[]
    q5_min=5.5
    for row in valid:
        with np.load(row['path']) as z:q5_min=min(q5_min,float(z['q'][:,4].min())-.03)
    for col,pair in enumerate(pairs):
        for row,label,color in zip(pair['rows'],LABELS,COLORS):
            with np.load(row['path']) as z:
                axs[0,col].plot(np.arange(301)*DT,z['q'][:,4],label=label,color=color)
                axs[1,col].plot((np.arange(300)+1)*DT,z['err_mm'],label=label,color=color)
                all_end.append(dict(seed=pair['seed'],case=row['case'],q5_end_rad=float(z['q'][-1,4]),
                    dq5_end_rad_s=float(z['qd'][-1,4]),q5_last1s_change_rad=float(z['q'][-1,4]-z['q'][-51,4])))
        axs[0,col].axhline(Q_MAX[4],color='k',ls='--',label='Actual upper limit = 2pi')
        axs[0,col].axhspan(Q_MAX[4],6.55,color='red',alpha=.05)
        axs[0,col].set(title=f'Heldout seed{pair["seed"]}',ylabel='Joint5 position [rad]',ylim=(q5_min,6.55))
        axs[1,col].set(ylabel='EE tracking error [mm]',ylim=(0,40))
        for ax in axs[:,col]:
            ax.set_xlabel('Time [s]');ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.suptitle('Frozen N0; DOB estimation only (NO output/rollout compensation); d5 -> -12.5Nm\n'
        'R=4cm / 6s circle; initial state inside all three tightened boxes; independent heldout seeds; no mesh contact')
    fig.savefig(ROOT/'heldout_comparison.png',dpi=160);plt.close(fig)
    (ROOT/'end_state_diagnostics.json').write_text(json.dumps(all_end,indent=2))
    # Complete 3D trajectories, not only their position-error scalar.
    pair=pairs[0]
    fig=plt.figure(figsize=(13,5),layout='constrained');ax=fig.add_subplot(121,projection='3d');xy=fig.add_subplot(122)
    for row,label,color in zip(pair['rows'],LABELS,COLORS):
        with np.load(row['path']) as z:
            ref=z['circle_reference'];ee=z['ee'][1:]
            if row==pair['rows'][0]:
                ax.plot(*ref.T,'k--',label='Reference circle');xy.plot(*ref[:,:2].T,'k--',label='Reference circle')
            ax.plot(*ee.T,color=color,label=label);xy.plot(*ee[:,:2].T,color=color,label=label)
    ax.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]',title=f'Actual EE paths, heldout seed{pair["seed"]}')
    xy.set(xlabel='X [m]',ylabel='Y [m]',title='XY projection',aspect='equal');xy.grid(alpha=.25)
    ax.legend(fontsize=8);xy.legend(fontsize=8)
    fig.savefig(ROOT/'heldout_ee_trajectory_seed8707.png',dpi=160);plt.close(fig)
    print(json.dumps(summary['groups'],indent=2))


if __name__=='__main__':main()
