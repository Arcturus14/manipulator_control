"""Visualize true joint state vs soft-cost onset and cycle-wise tracking RMSE."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import DT, Q_MIN, Q_MAX


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path)
    p.add_argument('--scenario',type=int,default=0)
    p.add_argument('--cases',nargs='+',default=['static25','directional25_5','static5'])
    p.add_argument('--joint',type=int,choices=range(6),help='Override displayed joint (zero-based).')
    a=p.parse_args();manifest=json.loads((a.directory/'manifest.json').read_text())
    period=manifest['arguments'].get('period',4.);j=manifest['arguments']['joint'] if a.joint is None else a.joint
    rows=[r for r in json.loads((a.directory/'metrics.json').read_text()) if r['scenario']==a.scenario]
    period=rows[0].get('period_seconds',period)
    colliding=any(r['metrics'].get('self_collision_states',0) or r['metrics'].get('floor_collision_states',0) for r in rows)
    styles={'static25':('Static 2.5%','tab:orange'),'directional25_5':('Dynamic 2.5-5%','tab:blue'),
            'directional25_75':('Dynamic 2.5-7.5%','tab:blue'),'static35':('Static 3.5%','tab:brown'),
            'static5':('Static 5%','tab:green'),'static75':('Static 7.5%','tab:purple'),'static10':('Static 10%','tab:red')}
    fig,axes=plt.subplots(1,2,figsize=(15,5));result={}
    width=.8/len(a.cases)
    for i,case in enumerate(a.cases):
        label,color=styles[case];rr=[r for r in rows if r['case']==case];states=[];margins=[];errors=[]
        for r in rr:
            with np.load(a.directory/f's{a.scenario}_seed{r["seed"]}_{case}.npz') as z:
                states.append(z['q'][:-1,j]);margins.append(z['kroc_margin'][:,0,1,j]);errors.append(z['err_mm'])
        q=np.array(states);m=np.array(margins);err=np.array(errors);t=np.arange(q.shape[1])*DT
        onset=Q_MAX[j]-m
        axes[0].plot(t,q.mean(0),color=color,label=label)
        axes[0].fill_between(t,q.min(0),q.max(0),color=color,alpha=.10)
        axes[0].plot(t,onset.mean(0),color=color,ls=':' if case.startswith('directional') else '--',
                     alpha=.7,label=label+' onset')
        count=round(q.shape[1]*DT/period);rms=[]
        for cycle in range(count):
            select=(t>=cycle*period)&(t<(cycle+1)*period)
            rms.append(np.sqrt(np.mean(err[:,select]**2,axis=1)))
        rms=np.array(rms)
        axes[1].bar(np.arange(count)+(i-(len(a.cases)-1)/2)*width,rms.mean(1),width=width,
                    yerr=rms.std(1,ddof=1) if len(rr)>1 else None,color=color,label=label,capsize=3)
        span=Q_MAX[j]-Q_MIN[j]
        result[case]={'seeds':[r['seed'] for r in rr],
            'cycle_rmse_mm_by_seed':rms.tolist(),'cycle_rmse_mean_mm':rms.mean(1).tolist(),
            'upper_soft_active_fraction_per_seed':(q>onset).mean(1).tolist(),
            'upper_margin_max_fraction':float(m.max()/span)}
    axes[0].axhline(Q_MAX[j],color='black',ls='-.',label='Physical upper limit')
    axes[0].set(xlabel='Time [s]',ylabel=f'q{j+1} [rad]',title='Solid: true state | dashed/dotted: soft-cost onset')
    axes[1].set(xticks=np.arange(count),xticklabels=[f'Cycle {i+1}' for i in range(count)],
                ylabel='EE tracking RMSE [mm]',title='Per-cycle RMSE (mean +/- seed SD)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    if colliding:
        fig.suptitle('DIAGNOSTIC ONLY: colliding runs included; post-contact dynamics are not modeled',color='crimson',fontsize=11)
        result['_includes_colliding_runs']=True
        result['_post_contact_metrics_are_not_validated_physics']=True
    fig.tight_layout();output=a.directory/'figures';output.mkdir(exist_ok=True)
    suffix=f'_j{j+1}' if a.joint is not None else ''
    fig.savefig(output/f'margin_pressure_s{a.scenario}{suffix}.png',dpi=170,bbox_inches='tight',pad_inches=.12);plt.close(fig)
    (output/f'cycle_metrics_s{a.scenario}{suffix}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
