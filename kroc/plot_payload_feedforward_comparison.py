"""Static diagnostic figure for the fixed rounded-rectangle feedforward ablation."""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from config import DT, JOINT_ARMATURE
from kroc.view_payload_tcp import load_result


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    rows=json.loads((args.root/'paired_metrics.json').read_text())
    old=next(r for r in rows if r['seed']==6807 and r['variant']=='existing_residual_only')
    new=next(r for r in rows if r['seed']==6807 and r['variant']=='reflected_inertia_ff_on')
    a=load_result(Path(old['path']));b=load_result(Path(new['path']))
    n=len(a['u_mppi']);t=np.arange(n)*DT
    qref=a['q_reference'][:n]
    acc=np.gradient(np.gradient(qref,DT,axis=0,edge_order=2),DT,axis=0,edge_order=2)
    j2_reflected=acc[:,1]*JOINT_ARMATURE[1]
    fig,ax=plt.subplots(2,2,figsize=(13,8))
    short=t<=10.
    ax[0,0].plot(t[short],j2_reflected[short],label='J2 reflected inertia demand',lw=1)
    ax[0,0].plot(t[short],(a['u_mppi']-a['u_ff'])[short,1],label='Existing MPPI residual J2',lw=1)
    ax[0,0].set_ylabel('Torque [Nm]');ax[0,0].set_title('Residual supplies part of omitted feedforward')
    ax[0,0].legend(fontsize=8)
    ax[0,1].plot(t[short],a['u_requested'][short,1],label='Existing',lw=1)
    ax[0,1].plot(t[short],b['u_requested'][short,1],label='Reflected inertia FF',lw=1)
    ax[0,1].axhline(-150,color='red',ls='--',lw=.8)
    ax[0,1].set_ylabel('J2 requested torque [Nm]');ax[0,1].set_title('Nominal torque: seed 6807')
    ax[0,1].legend(fontsize=8)
    seeds=[6807,6817,6827];xx=np.arange(len(seeds));width=.35
    def values(variant,key):return [next(r[key] for r in rows if r['seed']==s and r['variant']==variant) for s in seeds]
    oldname='existing_residual_only';newname='reflected_inertia_ff_on'
    for axis,key,title in [(ax[1,0],'rmse_err_mm','TCP RMSE'),(ax[1,1],'torque_rate_rms','Applied torque-rate RMS')]:
        axis.bar(xx-width/2,values(oldname,key),width,label='Existing')
        axis.bar(xx+width/2,values(newname,key),width,label='Reflected inertia FF')
        axis.set_xticks(xx,seeds);axis.set_xlabel('Seed');axis.set_title(title)
        axis.set_ylabel('mm' if key=='rmse_err_mm' else 'Nm/s');axis.legend(fontsize=8)
    for axis in ax.flat:axis.grid(alpha=.25)
    ax[0,0].set_xlabel('Time [s]');ax[0,1].set_xlabel('Time [s]')
    fig.suptitle('Rounded rectangle, no disturbance, DOB off; all other settings fixed')
    fig.tight_layout();fig.savefig(args.root/'feedforward_comparison.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
