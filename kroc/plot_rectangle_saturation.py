"""Summarize the fixed N0 rounded-rectangle DOB saturation screen."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from config import DT
from kroc.view_payload_tcp import load_result


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    rows=json.loads((args.root/'runs.json').read_text())
    columns=['seed','amplitude_abs_nm','safe','rmse_err_mm','clip_steps_J2',
             'clip_J2_abs_max_nm','jclip_J2_nm_s','first_clip_s',
             'exclusively_compensation_induced_clip_steps_J2','nominal_alone_excess_steps_J2']
    with (args.root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for row in sorted(rows,key=lambda r:(r['seed'],r['amplitude_abs_nm'])):
            writer.writerow({k:row[k] for k in columns})
    fig,ax=plt.subplots(2,2,figsize=(13,9))
    for seed in sorted(set(r['seed'] for r in rows)):
        subset=sorted((r for r in rows if r['seed']==seed),key=lambda r:r['amplitude_abs_nm'])
        x=[r['amplitude_abs_nm'] for r in subset]
        for axis,key in [(ax[0,0],'clip_steps_J2'),(ax[0,1],'jclip_J2_nm_s'),(ax[1,0],'rmse_err_mm')]:
            axis.plot(x,[r[key] for r in subset],'o-',label=f'seed {seed}')
    for axis,title,ylabel in [(ax[0,0],'J2 clipped steps','steps / 1200'),
                              (ax[0,1],'Cumulative J2 clipping loss','Nm s'),
                              (ax[1,0],'TCP tracking RMSE','mm')]:
        axis.set_title(title);axis.set_xlabel('J2 disturbance magnitude [Nm]')
        axis.set_ylabel(ylabel);axis.grid(alpha=.3);axis.legend(fontsize=8)
    row=next(r for r in rows if r['seed']==6807 and r['amplitude_abs_nm']==30)
    z=load_result(Path(row['path']));n=len(z['u_requested']);t=np.arange(n)*DT
    view=(t>=4.4)&(t<=5.5)
    ax[1,1].plot(t[view],z['u_mppi'][view,1],label='MPPI nominal J2',lw=1)
    ax[1,1].plot(t[view],z['u_requested'][view,1],label='Requested after DOB',lw=1)
    ax[1,1].plot(t[view],z['u_applied'][view,1],label='Applied',lw=1)
    ax[1,1].axhline(-150,color='red',ls='--',lw=.8,label='J2 lower limit')
    ax[1,1].set_title('Compensation-induced clip, seed 6807, 30 Nm')
    ax[1,1].set_xlabel('Time [s]');ax[1,1].set_ylabel('J2 torque [Nm]')
    ax[1,1].grid(alpha=.3);ax[1,1].legend(fontsize=8)
    fig.suptitle('N0 rounded rectangle: empirical onset; no N1/D0 comparison')
    fig.tight_layout();fig.savefig(args.root/'saturation_screen.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
