"""Plot an individual scenario from search_postdob_floor25 artifacts."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import DT, Q_MAX, Q_MIN, TAU_MAX


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--scenario',type=int,default=0)
    a=p.parse_args()
    rows=[r for r in json.loads((a.directory/'metrics.json').read_text()) if r['scenario']==a.scenario]
    manifest=json.loads((a.directory/'manifest.json').read_text())
    j=manifest['arguments']['joint']; sign=rows[0]['sign']
    side=1 if sign==1 else 0
    limit=TAU_MAX[j]*manifest['arguments']['scale']
    fig,axes=plt.subplots(6,1,figsize=(11,17),sharex=True)
    names=['static25','directional25_5','static5']
    labels=['Static 2.5%','Directional 2.5-5%','Static 5%']
    colors=['tab:orange','tab:blue','tab:green']
    for name,label,color in zip(names,labels,colors):
        selected=[r for r in rows if r['case']==name]
        datasets=[np.load(a.directory/f's{a.scenario}_seed{r["seed"]}_{name}.npz') for r in selected]
        q=np.stack([d['q'][:,j] for d in datasets]);ts=np.arange(q.shape[1])*DT
        axes[0].plot(ts,q.mean(0),label=label,color=color)
        axes[0].fill_between(ts,q.min(0),q.max(0),color=color,alpha=.16)
        margins=np.stack([d['kroc_margin'][:,0,side,j] for d in datasets])*100/(Q_MAX[j]-Q_MIN[j])
        t=np.arange(margins.shape[1])*DT
        axes[1].plot(t,margins.mean(0),label=label,color=color)
        axes[1].fill_between(t,margins.min(0),margins.max(0),color=color,alpha=.16)
        d=datasets[0]
        axes[2].plot(t,d['d_hat_used'][:,j],label=label,color=color)
        axes[3].plot(t,d['u_mppi'][:,j]+d['d_hat_used'][:,j],label=label,color=color,alpha=.8)
        axes[4].plot(t,d['u_applied'][:,j],label=label,color=color,alpha=.8)
        err=np.stack([d['err_mm'] for d in datasets])
        axes[5].plot(t,err.mean(0),label=label,color=color)
        axes[5].fill_between(t,err.min(0),err.max(0),color=color,alpha=.12)
    axes[0].axhline(Q_MAX[j] if sign==1 else Q_MIN[j],color='red',ls='--',label='Physical limit')
    axes[0].axhline(datasets[0]['q'][0,j],color='gray',ls=':',label='Initial joint position')
    actual=datasets[0]['d_true'][:,j]
    if len(actual)==len(t)+1:
        actual=actual[1:]
    axes[2].plot(t,actual,color='black',ls='--',label='Actual disturbance')
    for ax in axes[3:5]:
        ax.axhline(limit,color='red',ls='--');ax.axhline(-limit,color='red',ls='--')
    for ax in axes:
        ax.axvline(.2,color='black',ls='--',lw=.8)
        ax.axvline(.2+rows[0]['duration']*DT,color='black',ls=':',lw=.8)
        ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=4)
    for ax,label in zip(axes,[f'q{j+1} [rad] (mean/range)','Risk-side margin [% span]','DOB estimate [Nm] (first seed)',
                              'Requested torque [Nm] (first seed)','Applied torque [Nm] (first seed)','EE error [mm] (mean/range)']):
        ax.set_ylabel(label)
    axes[-1].set_xlabel('Time [s]')
    fig.suptitle(f'Post-DOB, no headroom: {len(datasets)} seeds; amplitude {rows[0]["amplitude"]} Nm; duration {rows[0]["duration"]*DT:.2f} s')
    fig.tight_layout();fig.savefig(a.directory/f'comparison_s{a.scenario}.png',dpi=150);plt.close(fig)
    fields=['hard_violation_max_rad','risk_distance_min_rad','tracking_rmse_mm','test_joint_saturation_steps','margin_mean_fraction','compute_mean_ms']
    cell=[]
    for field in fields:
        row=[]
        for name in names:
            v=np.array([r['metrics'][field] for r in rows if r['case']==name])
            row.append(f'{v.mean():.4f} +/- {v.std(ddof=1):.4f}')
        cell.append(row)
    counts=[sum(r['metrics']['all_joint_violation_steps']>0 for r in rows if r['case']==n) for n in names]
    cell.insert(0,[f'{v}/{len(datasets)}' for v in counts]);fields.insert(0,'Seeds with position violation')
    fig,ax=plt.subplots(figsize=(12,3.8));ax.axis('off')
    tab=ax.table(cellText=cell,rowLabels=fields,colLabels=labels,loc='center',cellLoc='center')
    tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.8)
    fig.savefig(a.directory/f'table_s{a.scenario}.png',dpi=160,bbox_inches='tight');plt.close(fig)


if __name__=='__main__':
    main()
