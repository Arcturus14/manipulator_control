"""Plot circular tracking and aggregate scenario evidence without a report."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import DT,Q_MAX,Q_MIN,TAU_MAX


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory',type=Path);p.add_argument('--scenario',type=int,default=0)
    p.add_argument('--additional',action='append',default=[],metavar='DIRECTORY:SCENARIO',
                   help='Merge matching independent-seed data without copying trajectories.')
    p.add_argument('--output',type=Path)
    p.add_argument('--cases',nargs='+',default=['static25','directional25_5','static5'])
    p.add_argument('--trajectory-3d',action='store_true')
    p.add_argument('--seeds',type=int,nargs='+',help='Plot only explicitly listed seeds, without altering saved results.')
    a=p.parse_args();sources=[(a.directory,a.scenario)]
    sources += [(Path(x.rsplit(':',1)[0]),int(x.rsplit(':',1)[1])) for x in a.additional]
    rows=[]
    for directory,scenario in sources:
        for r in json.loads((directory/'metrics.json').read_text()):
            if r['scenario']==scenario and (a.seeds is None or r['seed'] in a.seeds):rows.append({**r,'_directory':directory})
    assert rows,'No matching trajectories'
    conditions=[(r['fraction'],r['radius'],r['pose'],r['amplitude'],r['onset'],r['ramp'],r['sign']) for r in rows]
    assert len(set(conditions))==1,'Only identical scenarios may be merged'
    assert len({(r['seed'],r['case']) for r in rows})==len(rows),'Duplicate seed/case'
    manifest=json.loads((a.directory/'manifest.json').read_text());j=manifest['arguments']['joint'];sgn=rows[0]['sign']
    style={'static25':('Static 2.5%','tab:orange'),'directional25_5':('Directional 2.5-5%','tab:blue'),
           'directional25_75':('Directional 2.5-7.5%','tab:blue'),'static35':('Static 3.5%','tab:brown'),
           'static5':('Static 5%','tab:green'),'static75':('Static 7.5%','tab:purple'),'static10':('Static 10%','tab:red')}
    names=a.cases;labels=[style[n][0] for n in names];colors=[style[n][1] for n in names]
    fig=plt.figure(figsize=(14,13));axes=[fig.add_subplot(3,2,i+1,**({'projection':'3d'} if i==0 and a.trajectory_3d else {})) for i in range(6)]
    summary={}
    for name,label,c in zip(names,labels,colors):
        rr=[r for r in rows if r['case']==name]
        zs=[np.load(r['_directory']/f's{r["scenario"]}_seed{r["seed"]}_{name}.npz') for r in rr];z=zs[0]
        q=np.stack([d['q'][:,j] for d in zs]);ts=np.arange(q.shape[1])*DT;t=ts[:-1]
        axes[0].plot(z['ee'][:,0],z['ee'][:,1],*([z['ee'][:,2]] if a.trajectory_3d else []),color=c,label=label)
        err=np.stack([d['err_mm'] for d in zs]);axes[1].plot(t,err.mean(0),color=c,label=label)
        axes[1].fill_between(t,err.min(0),err.max(0),color=c,alpha=.13)
        axes[2].plot(ts,q.mean(0),color=c,label=label);axes[2].fill_between(ts,q.min(0),q.max(0),color=c,alpha=.13)
        mar=np.stack([d['kroc_margin'][:,0,1 if sgn>0 else 0,j] for d in zs])*100/(Q_MAX[j]-Q_MIN[j])
        axes[3].plot(t,mar.mean(0),color=c,label=label);axes[3].fill_between(t,mar.min(0),mar.max(0),color=c,alpha=.13)
        axes[4].plot(t,z['u_mppi'][:,j]+z['d_hat_used'][:,j],color=c,label=label,alpha=.8)
        axes[5].plot(t,z['d_hat_used'][:,j],color=c,label=label)
        summary[name]={'violation_seeds':sum(r['metrics']['position_violation_steps']>0 for r in rr),'seeds':len(rr),
                       'per_seed':[{k:r[k] for k in ('seed','metrics')} for r in rr]}
        for field in rr[0]['metrics']:
            vals=[r['metrics'][field] for r in rr]
            if all(isinstance(v,(int,float)) for v in vals):
                summary[name][field]={'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1)) if len(vals)>1 else 0.,'min':float(min(vals)),'max':float(max(vals))}
    shape=rows[0].get('shape','circle')
    ref=z['circle_reference'];axes[0].plot(ref[:,0],ref[:,1],*([ref[:,2]] if a.trajectory_3d else []),color='k',ls='--',label=f'{shape.title()} reference')
    if a.trajectory_3d:axes[0].set_zlabel('z [m]')
    else:axes[0].axis('equal')
    axes[0].set(xlabel='x [m]',ylabel='y [m]',title=f'{"3D" if a.trajectory_3d else "XY"} trajectory (seed {rows[0]["seed"]})')
    axes[1].set(xlabel='Time [s]',ylabel='EE error [mm]',title='Tracking error: mean and seed range')
    axes[2].axhline(Q_MAX[j] if sgn>0 else Q_MIN[j],color='red',ls='--',label='Physical limit')
    axes[2].set(xlabel='Time [s]',ylabel=f'q{j+1} [rad]',title='Joint position: mean and seed range')
    axes[3].set(xlabel='Time [s]',ylabel='Margin [% full span]',title='Risk-side directional margin')
    lim=TAU_MAX[j]*manifest['arguments']['scale']
    axes[4].axhline(lim,color='red',ls='--');axes[4].axhline(-lim,color='red',ls='--')
    axes[4].set(xlabel='Time [s]',ylabel='Requested torque [Nm]',title=f'Before actuator clamp (seed {rows[0]["seed"]})')
    axes[5].plot(t,z['d_true'][1:,j],'k--',label='True disturbance')
    axes[5].set(xlabel='Time [s]',ylabel='Disturbance [Nm]',title='DOB estimate (first seed)')
    for ax in axes:
        ax.grid(alpha=.25);ax.legend(fontsize=8)
    if rows[0]['amplitude']:
        for ax in axes[1:]:ax.axvline(rows[0]['onset'],color='gray',ls=':')
    condition='sustained disturbance (no release)' if rows[0]['amplitude'] else 'no external disturbance'
    colliding=any(r['metrics'].get('first_collision_index') is not None for r in rows)
    title=f'{shape.title()} Post-DOB: no headroom; {len(zs)} seeds; {condition}'
    if colliding:
        title+='\nDIAGNOSTIC ONLY: colliding runs included; post-contact physics are not modeled'
        summary['_includes_colliding_runs']=True
        summary['_post_contact_metrics_are_not_validated_physics']=True
    fig.suptitle(title,color='crimson' if colliding else 'black')
    output=a.output or a.directory/f'circle_comparison_s{a.scenario}.png'
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.tight_layout();fig.savefig(output,dpi=160);plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2))


if __name__=='__main__':main()
