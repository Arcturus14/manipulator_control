"""Figures and plain CSV tables for paper/, rebuilt from tracked evidence only.

 python -m kroc.make_paper_figures

No simulation is run. Inputs: kroc/evidence/{validation_20261122, realtime_20261123,
controls_20261123, saturation_sweep_20261123} and the final preset.
"""
import csv
import gzip
import json
from pathlib import Path

import numpy as np

from kroc import sweep_saturation as sweep
from kroc.evidence_common import CONFIG,EVIDENCE,TEST_SEEDS,load_config

PAPER=Path('paper');FIGURES=PAPER/'figures';DATA=PAPER/'data'
CASES=('N0','N1','D0');COLORS=sweep.COLORS
INK,MUTED,GRID_INK,SURFACE=sweep.INK,sweep.MUTED,sweep.GRID_INK,sweep.SURFACE


def write_csv(name,rows):
    path=DATA/name
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print('[DATA]',path,len(rows))


def tables():
    runs=json.loads((EVIDENCE/'validation_20261122/runs_kernel.json').read_text())
    names=dict(aligned_dob_off='no disturbance, DOB off',aligned_dob_on='no disturbance, DOB on',
               aligned_45='45 Nm',aligned_47p5='47.5 Nm',aligned_45_long='45 Nm, 40 s run (5-39 s)',
               aligned_45_lowfreq='45 Nm, 0.1 Hz +-5 % variation',
               aligned_45_unclamped='45 Nm, actuator clamp removed (non-physical diagnostic)')
    write_csv('final_runs.csv',[dict(condition=names[r['name']],case=r['case'],seed=r['seed'],
        rmse_mm=round(r['eval_rmse_mm'],4),max_error_mm=round(r['eval_max_mm'],4),
        q3_min_rad=round(r['q3_min_rad'],4),valid=r['valid']) for r in runs if r['name'] in names])
    stats=json.loads((EVIDENCE/'validation_20261122/statistics.json').read_text())
    write_csv('final_statistics.csv',[dict(comparison=c['comparison'],
        mean_difference_mm=round(c['mean_difference_mm'],4),ci95_half_width_mm=round(c['ci95_half_width_mm'],4),
        positive_seeds=c['positive_seeds'],wilcoxon_p=round(c['wilcoxon_p'],5),holm_p=round(c['holm_p'],5))
        for c in stats['comparisons'] if not c['comparison'].startswith(('original','DOB off original'))])
    summary=json.loads((EVIDENCE/'saturation_sweep_20261123/summary.json').read_text())
    rows=[]
    for t in summary['table']:
        row=dict(disturbance_nm=t['amplitude_abs_nm'],percent_of_torque_limit=round(t['percent_of_limit'],2))
        for c in CASES:
            row.update({f'{c}_rmse_mean_mm':round(t[c]['rmse_mean_mm'],4),f'{c}_rmse_std_mm':round(t[c]['rmse_std_mm'],4),
                        f'{c}_max_error_mean_mm':round(t[c]['max_mean_mm'],3),f'{c}_j_clip_nm_s':round(t[c]['j_clip_nm_s'],3),
                        f'{c}_clip_time_percent':round(t[c]['clip_percent'],2),f'{c}_valid_seeds':t[c]['valid']})
        for k in ('N1_better_than_N0','D0_better_than_N0','D0_better_than_N1'):
            row[f'{k}_seeds']=t[k]['seeds'];row[f'{k}_valid_pairs']=t[k]['valid_pairs']
        rows.append(row)
    write_csv('sweep_summary.csv',rows)
    write_csv('sweep_runs.csv',[dict(disturbance_nm=r['amplitude_abs_nm'],case=r['case'],seed=r['seed'],
        rmse_mm=round(r['rmse_mm'],4),max_error_mm=round(r['max_mm'],3),j_clip_nm_s=round(r['j_clip_nm_s'],3),
        clip_time_percent=round(r['clip_percent'],2),requested_j2_peak_nm=round(r['requested_j2_peak_nm'],2),
        q3_min_rad=round(r['q3_min_rad'],4),elbow_ok=r['elbow_ok'],limits_ok=r['limits_ok'],
        collision_free=r['collision_free'],valid=r['valid'],solve_mean_ms=round(r['solve_mean_ms'],3),
        solve_max_ms=round(r['solve_max_ms'],3)) for r in summary['runs']])
    levels=sweep.representative(summary['table'],len(TEST_SEEDS))
    with gzip.open(EVIDENCE/'saturation_sweep_20261123/timeseries.csv.gz','rt') as handle:
        reader=csv.DictReader(handle)
        keep=[r for r in reader if float(r['amplitude_abs_nm']) in levels and float(r['t_s'])<=15.0]
    write_csv('sweep_timeseries_selected_levels.csv',keep)
    realtime=json.loads((EVIDENCE/'realtime_20261123/summary.json').read_text())['table']
    write_csv('realtime_latency_summary.csv',[dict(backend=r['backend'],wait=r['wait'],case=r['case'],steps=r['steps'],
        **{f'{k}_{q}_ms':round(r[k][q],3) for k in ('wake','command','update_done') for q in ('mean','p99','max')},
        command_over_20ms=r['command']['over_20ms'],update_done_over_20ms=r['update_done']['over_20ms']) for r in realtime])
    causes=json.loads((EVIDENCE/'controls_20261123/causes.json').read_text())['table']
    label=dict(dob_off='DOB off',dob_off_noiseless='DOB off, no sensor noise',lpf6='DOB on, LPF 6 Hz (used)',
               lpf12='DOB on, LPF 12 Hz',lpf_off='DOB on, no LPF',lpf6_noiseless='DOB on, LPF 6 Hz, no sensor noise',
               lpf_off_noiseless='DOB on, no LPF, no sensor noise')
    write_csv('dob_no_disturbance_diagnostic.csv',[dict(condition=label[t['variant']],rmse_mean_mm=round(t['rmse_mean_mm'],4),
        rmse_std_mm=round(t['rmse_std_mm'],4),d_hat_j2_rms_nm=round(t['d_hat_j2_rms_nm'],4))
        for t in causes if t['variant'] in label])
    write_csv('ablation_first_input_smoothing_off.csv',[dict(case=t['case'],disturbance_nm=t['amplitude_abs_nm'],
        rmse_mean_mm=round(t['rmse_mean_mm'],4),rmse_std_mm=round(t['rmse_std_mm'],4),valid_seeds=t['valid'],seeds=t['runs'])
        for t in causes if t['variant']=='boundary_weight_zero'])
    sensitivity=json.loads((EVIDENCE/'controls_20261123/sensitivity.json').read_text())['table']
    write_csv('sensitivity_K_H_lambda.csv',[dict(setting=t['variant'],case=t['case'],disturbance_nm=t['amplitude_abs_nm'],
        rmse_mean_mm=round(t['rmse_mean_mm'],4),rmse_std_mm=round(t['rmse_std_mm'],4),valid_seeds=t['valid'],
        seeds=t['runs'],ess_median=round(t['ess_median'],2),eta_median=round(t['eta_median'],2),
        solve_mean_ms=round(t['solve_mean_ms'],3)) for t in sensitivity])
    return runs


def style(plt):
    plt.rcParams.update({'font.size':10,'axes.edgecolor':GRID_INK,'axes.labelcolor':INK,'xtick.color':MUTED,
        'ytick.color':MUTED,'axes.grid':True,'grid.color':GRID_INK,'grid.linewidth':.8,'axes.facecolor':SURFACE,
        'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})


def final_bars(runs,plt):
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    base=np.mean([r['eval_rmse_mm'] for r in runs if r['name']=='aligned_dob_on'])
    ax.axhline(base,color=MUTED,lw=1,label='no disturbance (DOB on)')
    for g,(name,label) in enumerate((('aligned_45','45 Nm (30 %)'),('aligned_47p5','47.5 Nm (31.7 %)'))):
        for i,case in enumerate(CASES):
            values=np.array([r['eval_rmse_mm'] for r in runs if r['name']==name and r['case']==case])
            x=g*1.4+(i-1)*.36
            ax.bar(x,values.mean(),width=.24,color=COLORS[case],label=case if g==0 else None,zorder=2)
            ax.errorbar(x,values.mean(),yerr=values.std(ddof=1),color=INK,lw=1,capsize=3,zorder=3)
            ax.scatter(np.full(len(values),x)+np.linspace(-.07,.07,len(values)),values,s=9,color=INK,alpha=.45,zorder=4,lw=0)
            ax.text(x,values.mean()+values.std(ddof=1)+.25,f'{values.mean():.2f}',ha='center',fontsize=9,color=INK)
    ax.set_xticks([0,1.4]);ax.set_xticklabels(['45 Nm (30 %)','47.5 Nm (31.7 %)'])
    ax.set_xlabel('J2 disturbance (share of 150 Nm limit)');ax.set_ylabel('TCP RMSE, 5–15 s [mm]')
    ax.grid(axis='x',visible=False);ax.set_ylim(0,None)
    ax.set_title('Bars: mean over 10 seeds, whiskers: sample SD, dots: seeds',fontsize=9,color=MUTED,loc='left')
    ax.legend(loc='upper left');fig.tight_layout();fig.savefig(FIGURES/'final_rmse_comparison.png',dpi=200);plt.close(fig)


def realtime(plt):
    fig,ax=plt.subplots(figsize=(6.4,3.8))
    series=[('kernel','sleep',c,COLORS[c],f'kernel, {c}','-') for c in CASES]
    series.append(('captured','sleep','N0',MUTED,'previous, N0','--'))
    for backend,wait,case,color,label,style_ in series:
        with gzip.open(EVIDENCE/f'realtime_20261123/{backend}_{wait}_{case}.csv.gz','rt') as handle:
            data=np.genfromtxt(handle,delimiter=',',skip_header=1)
        x=np.sort(data[:,5]);y=np.arange(1,len(x)+1)/len(x)
        ax.plot(x,y,color=color,lw=2,ls=style_,label=label)
    ax.axvline(20,color=INK,lw=1);ax.text(20.5,.05,'20 ms period',color=INK,fontsize=9)
    ax.set_xscale('log');ax.set_xlim(.8,40);ax.set_ylim(0,1.02)
    ax.set_xlabel('Control computation latency after waiting [ms]');ax.set_ylabel('Cumulative share of steps')
    ax.set_title('50 Hz release, 8,000 steps per line, release to DOB update (plant simulation excluded)',fontsize=9,color=MUTED,loc='left')
    ax.legend(loc='lower left',bbox_to_anchor=(.5,.25));fig.tight_layout();fig.savefig(FIGURES/'realtime_latency_cdf.png',dpi=200);plt.close(fig)


def setup(plt):
    import torch
    from config import DH_PARAMS,DT
    from dynamics.ur5_payload import CubePayload
    from kinematics.payload_tcp import PayloadTCP
    from kroc.payload_scene import make_reference
    config=load_config();payload=CubePayload(**config['payload']);scene=config['scene']
    tcp=PayloadTCP(payload,torch.device('cpu'),torch.float64);q=np.asarray(scene['initial_q'])
    reference=make_reference(tcp,q,round(scene['duration_s']/DT),DT,scene)
    transform=np.eye(4);points=[transform[:3,3].copy()]
    for angle,(a,d,alpha,offset) in zip(q,DH_PARAMS):
        c,s=np.cos(angle+offset),np.sin(angle+offset);ca,sa=np.cos(alpha),np.sin(alpha)
        transform=transform@np.array([[c,-s*ca,s*sa,a*c],[s,c*ca,-c*sa,a*s],[0,sa,ca,d],[0,0,0,1]])
        points.append(transform[:3,3].copy())
    points=np.array(points);cube=tcp.vertices(torch.as_tensor(q)[None])[0].numpy()
    fig=plt.figure(figsize=(5.6,4.6));ax=fig.add_subplot(projection='3d')
    ax.plot(*points.T,color=INK,lw=3,marker='o',ms=4,label='UR5 (initial posture)')
    ax.plot(*reference[int(3/DT):].T,color=COLORS['N0'],lw=1.5,label='TCP reference: YZ circle, r = 0.2 m, T = 2 s')
    for i in range(8):
        for j in range(i+1,8):
            if bin(i^j).count('1')==1:ax.plot(*cube[[i,j]].T,color=COLORS['N1'],lw=1.5)
    ax.plot([],[],color=COLORS['N1'],lw=1.5,label='5 kg payload cube (TCP = cube center)')
    ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.set_zlabel('Z [m]');ax.set_box_aspect((1,1,.8))
    ax.view_init(22,-58);ax.legend(loc='upper left',fontsize=8);fig.tight_layout()
    fig.savefig(FIGURES/'setup_robot_payload_trajectory.png',dpi=200);plt.close(fig)


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    runs=tables()
    sweep.figures(TEST_SEEDS,FIGURES)
    style(plt);final_bars(runs,plt);realtime(plt);setup(plt)
    print('[CONFIG]',CONFIG)


if __name__=='__main__':main()
