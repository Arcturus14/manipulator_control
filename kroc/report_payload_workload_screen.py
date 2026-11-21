"""Reproducible figures/CSV/assessment for frozen-N0 nominal-workload screen."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kroc.view_payload_tcp import load_result
from kroc.study_payload_torque_stability import write


def main():
    root=Path('kroc/results/payload_nominal_workload_20261121');root.mkdir(exist_ok=True)
    c=Path('kroc/results/payload_circle_workload_20261121/cuda');r=Path('kroc/results/payload_rectangle_workload_20261121/cuda')
    circle=json.loads((c/'pilot_metrics.json').read_text());rect=json.loads((r/'pilot_metrics.json').read_text())
    validation=json.loads((r/'secondary_validation.json').read_text());allrows=[]
    for typ,rows in [('circle',circle),('rectangle',rect),('rectangle_fixed_validation',validation['rows'])]:
        for row in rows:allrows.append(dict(row,kind=typ))
    columns=['kind','name','period_s','seed','rmse_err_mm','max_err_mm','steady_rmse_mm','steady_J2_negative_peak_nm','steady_J2_q99_abs_nm','steady_J2_fraction_below_minus120','steady_J2_fraction_below_minus130','mean_abs_j2_req_nm','clipping_steps','safe','ess_mean','torque_rate_rms','path']
    with (root/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(allrows)
    baseline=load_result(Path('kroc/results/payload_quarter_volume_20261121/20261121_211735/seed6807'))
    fig,ax=plt.subplots(figsize=(11,7))
    mapping=[]
    for typ,rows,color,marker in [('Circle',circle,'tab:blue','o'),('Rectangle',rect,'tab:orange','s')]:
        for i,row in enumerate(rows):
            x,y=row['steady_J2_negative_peak_nm'],row['rmse_err_mm'];failed=not row['safe'] or row['clipping_steps']>0
            ax.scatter(x,y,color='red' if failed else color,marker='x' if failed else marker,s=65,label=typ if i==0 else None)
            label=f'{typ[0]}{i+1}';mapping.append(f'{label}: {row["name"]}, {row["period_s"]:g}s')
            ax.annotate(label,(x,y),xytext=[(5,7),(5,-12),(-20,5)][i%3],textcoords='offset points',fontsize=9)
    bx=float(-baseline['u_requested'][:,1].min());by=float(np.sqrt(np.mean(baseline['err_mm']**2)))
    ax.scatter(bx,by,color='black',marker='*',s=120,label='Original20s circle');ax.annotate('B0',(bx,by),xytext=(5,6),textcoords='offset points')
    ax.axvline(130,color='gray',ls='--');ax.axhline(3,color='gray',ls='--');ax.set_yscale('log')
    ax.set(xlabel='Post-first-cycle negative J2 torque peak magnitude [Nm]',ylabel='Full-run TCP RMSE [mm]',title='Frozen N0: larger nominal torque versus tracking error\nRed crosses: actual nominal clipping or geometry/state failure')
    ax.grid(alpha=.25);ax.legend(fontsize=8,loc='upper left');fig.text(.77,.84,'Pilot cases (seed6807)\n\n'+'\n'.join(mapping),fontsize=8,va='top')
    fig.tight_layout(rect=(0,0,.77,1));fig.savefig(root/'workload_tradeoff.png',dpi=175);plt.close(fig)
    zs={str(row['seed']):load_result(Path(row['path'])) for row in validation['rows']}
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
    for seed,z in zs.items():
        n=len(z['u_requested']);t=np.arange(n)*.02
        axes[0].plot(t,z['u_requested'][:,1],label=f'seed {seed}',lw=.9)
        axes[1].plot(t,z['err_mm'],label=f'seed {seed}',lw=.8)
        axes[2].plot(t,z['mppi_weight_ess'],label=f'seed {seed}',lw=.6,alpha=.7)
    axes[0].axhline(-150,color='red',ls='--',label='Physical lower limit');axes[0].axhline(-130,color='gray',ls=':',label='Initial loading target')
    for ax,label in zip(axes,['J2 requested torque [Nm]','TCP error [mm]','ESS']):ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8,ncol=4)
    axes[-1].set_xlabel('Time [s]');fig.suptitle('Secondary trial: XZ rectangle24 x20cm, corner2cm, period5s; DOB/disturbance OFF')
    fig.tight_layout();fig.savefig(root/'rectangle_trial_timeseries.png',dpi=165);plt.close(fig)
    z=zs['6807'];n=len(z['u_requested']);fig,axes=plt.subplots(1,2,figsize=(12,5))
    ref=z['tcp_reference'];ee=z['ee']
    axes[0].plot(ref[:,0],ref[:,2],'k--',label='Reference');axes[0].plot(ee[:,0],ee[:,2],label='N0 TCP');axes[0].axis('equal')
    axes[0].set(xlabel='X [m]',ylabel='Z [m]',title='XZ rectangle: actual straight sides and C2 corner turns');axes[0].legend();axes[0].grid(alpha=.25)
    axes[1].plot(np.arange(len(baseline['u_mppi']))*.02,baseline['u_requested'][:,1],label='Original circle16cm,20s')
    axes[1].plot(np.arange(n)*.02,z['u_requested'][:,1],label='Rectangle24 x20cm,5s')
    axes[1].axhline(-150,color='red',ls='--',label='Physical lower limit');axes[1].set(xlabel='Time [s]',ylabel='J2 requested torque [Nm]',title='Higher corner peaks; not a uniform mean-load increase');axes[1].legend(fontsize=8);axes[1].grid(alpha=.25)
    fig.tight_layout();fig.savefig(root/'reference_and_nominal_torque.png',dpi=180);plt.close(fig)
    write(root/'assessment.json',dict(primary_target_met=False,primary_target='Safe,no nominal clipping,RMSE<=3mm and post-first-cycle negative J2>=130Nm for>=1% of steps.',
        circle_source=str(c),rectangle_source=str(r),secondary_validation=validation,
        actual_scope='N0 only,DOB/disturbance OFF. Existing controller,covariance,lambda,physical limits,5kg model unchanged. No N1/D0 research comparison or smaller-disturbance saturation claim.',
        interpretation='Circles raised torque but tracking/clipping worsened. C2 rectangles offer higher corner peaks with lower RMSE; fixed secondary trial does not meet primary130Nm loading target. Higher peaks must not be described as higher mean loading.',
        next_issue='Current rigid-body inverse-dynamics feedforward omits reflected armature although forward dynamics includes it. This is a possible fast-motion limitation; diagnosed, not changed or causally proven. See circle feedforward_audit.json.',
        GUI_trial_preset='configs/payload_cube_tcp_workload_rectangle_trial_n0.yaml',old_presets_and_defaults_preserved=True,
        generated=['summary.csv','workload_tradeoff.png','rectangle_trial_timeseries.png','reference_and_nominal_torque.png']))
    print('[OUTPUT]',root)

if __name__=='__main__':main()
