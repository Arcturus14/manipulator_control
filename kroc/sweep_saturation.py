"""N0/N1/D0 across J2 disturbance levels, from before saturation to invalid runs. CUDA only.

 python -m kroc.sweep_saturation             # simulate (resumable), audit, summarize, plot
 python -m kroc.sweep_saturation --summarize # rebuild tables from saved runs
 python -m kroc.sweep_saturation --figures   # redraw figures from saved runs

The disturbance level is raised until no case has a valid seed, so the range
always ends at constraint violation. A run is valid when it has no robot or
payload mesh collision, no joint position/velocity limit violation and keeps
q3 > 0.2 rad (elbow not flipped).

Metrics (seed 7207-7297, kernel backend, final preset otherwise unchanged):
 rmse/max      TCP error over 5 < t <= 15 s
 clip_percent  share of steps in that window where the J2 request exceeds 150 Nm
 j_clip        integral of |requested - applied| over all joints and the whole run [Nm s]
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
import subprocess
import sys

import numpy as np

from config import DT
from kroc.evidence_common import (EVIDENCE,TEST_SEEDS,load_config,provenance,
                                  simulate,write_json)

CASES=('N0','N1','D0')
GRID=(0.,35.,37.5,40.,41.,42.,43.,44.,45.,46.,47.,47.5,48.,49.,50.,51.,52.,53.,55.,57.5,60.)
EXTENSION_STEP,EXTENSION_LIMIT=2.5,80.
LIMIT=150.
RAW=Path('kroc/results/saturation_sweep_20261123')
OUT=EVIDENCE/'saturation_sweep_20261123'
WINDOW=slice(250,750)          # 5 < t <= 15 s; err_mm[k] is the error at t=(k+1) dt


def raw_path(case,amplitude,seed):
    return RAW/'raw'/f'{case}_a{amplitude:g}_seed{seed}.npz'


def run_group(case,amplitudes,seeds):
    for amplitude in amplitudes:
        for seed in seeds:
            path=raw_path(case,amplitude,seed)
            if path.exists():continue
            config=load_config();config['scene']['disturbance_amplitude_nm']=-amplitude
            z=simulate(config,seed,case)
            requested=z['u_mppi']+z['d_hat_used']
            path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,q=z['q'].astype(np.float32),err_mm=z['err_mm'].astype(np.float32),
                requested=requested.astype(np.float32),applied=z['u_applied'].astype(np.float32),
                d_true=z['d_true'][1:len(requested)+1].astype(np.float32),
                d_hat=z['d_hat_used'].astype(np.float32),
                solve_ms=(z['solve_t_raw'][10:]*1000).astype(np.float32),
                limits_ok=np.asarray(not bool(z['joint_position_limit_violated'].any()
                                              or z['joint_velocity_limit_violated'].any())))
        print('DONE',case,amplitude,flush=True)


def metrics(path):
    """Everything except the mesh audit, from one saved run."""
    with np.load(path) as z:
        error=z['err_mm'];requested=z['requested'];applied=z['applied']
        loss=np.abs(requested-applied).astype(float)
        q3=float(z['q'][:,2].min())
        return dict(rmse_mm=float(np.sqrt(np.mean(error[WINDOW].astype(float)**2))),
            max_mm=float(error[WINDOW].max()),
            clip_percent=float(np.mean(np.abs(requested[WINDOW,1])>LIMIT+1e-4)*100),
            j_clip_nm_s=float(loss.sum()*DT),
            first_clip_s=float(np.argmax(loss[:,1]>1e-4)*DT) if (loss[:,1]>1e-4).any() else None,
            requested_j2_peak_nm=float(np.abs(requested[WINDOW,1]).max()),
            q3_min_rad=q3,elbow_ok=q3>.2,limits_ok=bool(z['limits_ok']),
            solve_mean_ms=float(z['solve_ms'].mean()),solve_max_ms=float(z['solve_ms'].max()))


def audit(path):
    from dynamics.ur5_payload import CubePayload
    from kroc.audit_circle_validation import dense_mesh_check
    from kroc.payload_geometry import audit_cube_trajectory
    robot=dense_mesh_check((str(path),4))
    with np.load(path) as z:cube=audit_cube_trajectory(z['q'].astype(float),CubePayload(**load_config()['payload']))
    return not (robot['self_collision_states'] or robot['floor_collision_states']
                or cube['payload_collision_states'])


def amplitudes_on_disk():
    found={float(p.name.split('_a')[1].split('_seed')[0]) for p in (RAW/'raw').glob('*.npz')}
    return sorted(found)


def simulate_levels(levels,seeds):
    for case in CASES:
        subprocess.run([sys.executable,'-m','kroc.sweep_saturation','--group',case,
                        '--amplitudes',*map(str,levels),'--seeds',*map(str,seeds)],check=True)


def any_valid(level,seeds):
    """Elbow and limits only: the mesh audit is too slow to steer the sweep."""
    return any(metrics(raw_path(c,level,s))['elbow_ok'] and metrics(raw_path(c,level,s))['limits_ok']
               for c in CASES for s in seeds)


def summarize(seeds,workers):
    levels=amplitudes_on_disk()
    keys=[(c,a,s) for a in levels for c in CASES for s in seeds]
    cache=RAW/'audit.json';known=json.loads(cache.read_text()) if cache.exists() else {}
    missing=[k for k in keys if str(raw_path(*k)) not in known]
    with ProcessPoolExecutor(workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for k,free in zip(missing,pool.map(audit,[raw_path(*k) for k in missing])):
            known[str(raw_path(*k))]=free
    cache.write_text(json.dumps(known))
    runs=[]
    for case,amplitude,seed in keys:
        row=dict(case=case,amplitude_abs_nm=amplitude,seed=seed,**metrics(raw_path(case,amplitude,seed)))
        row['collision_free']=known[str(raw_path(case,amplitude,seed))]
        row['valid']=bool(row['collision_free'] and row['elbow_ok'] and row['limits_ok'])
        runs.append(row)
    pick=lambda c,a:sorted((r for r in runs if r['case']==c and r['amplitude_abs_nm']==a),key=lambda r:r['seed'])
    table=[]
    for amplitude in levels:
        item=dict(amplitude_abs_nm=amplitude,percent_of_limit=amplitude/LIMIT*100)
        for case in CASES:
            rows=pick(case,amplitude);value=lambda k:[r[k] for r in rows]
            item[case]=dict(valid=sum(value('valid')),seeds=len(rows),
                rmse_mean_mm=float(np.mean(value('rmse_mm'))),rmse_std_mm=float(np.std(value('rmse_mm'),ddof=1)),
                max_mean_mm=float(np.mean(value('max_mm'))),clip_percent=float(np.mean(value('clip_percent'))),
                j_clip_nm_s=float(np.mean(value('j_clip_nm_s'))),q3_min_rad=float(min(value('q3_min_rad'))))
        for better,worse in (('N1','N0'),('D0','N0'),('D0','N1')):
            pairs=[(b,w) for b,w in zip(pick(better,amplitude),pick(worse,amplitude)) if b['valid'] and w['valid']]
            item[f'{better}_better_than_{worse}']=dict(seeds=sum(b['rmse_mm']<w['rmse_mm'] for b,w in pairs),
                                                       valid_pairs=len(pairs))
        table.append(item)
    write_json(OUT/'summary.json',dict(definition=__doc__,seeds=seeds,levels=levels,table=table,runs=runs,
        provenance=provenance(Path('kroc/sweep_saturation.py'))))
    write_timeseries(levels,seeds)
    write_markdown(table)
    return table


def write_timeseries(levels,seeds):
    """Mean, min and max over the seeds for each case, level and step."""
    lines=['case,amplitude_abs_nm,t_s,statistic,ee_error_mm,requested_j2_nm,cumulative_clip_nm_s,d_true_j2_nm']
    for amplitude in levels:
        for case in CASES:
            series=[]
            for seed in seeds:
                with np.load(raw_path(case,amplitude,seed)) as z:
                    loss=np.abs(z['requested']-z['applied']).sum(1)
                    series.append(np.stack((z['err_mm'],z['requested'][:,1],np.cumsum(loss)*DT,z['d_true'][:,1]),1))
            series=np.stack(series)
            for name,values in (('mean',series.mean(0)),('min',series.min(0)),('max',series.max(0))):
                for k,row in enumerate(values):
                    lines.append(f'{case},{amplitude:g},{(k+1)*DT:.2f},{name},'+','.join(f'{x:.4f}' for x in row))
    (OUT/'timeseries.csv.gz').write_bytes(gzip.compress(('\n'.join(lines)+'\n').encode(),mtime=0))


def write_markdown(table):
    head=('| 외란 [Nm] | 한계 대비 | RMSE N0 / N1 / D0 [mm] | N1, D0 변화 | J_clip N0 / N1 / D0 [Nm s] '
          '| 포화 시간 N0 / N1 / D0 | 개선 seed N1<N0, D0<N0, D0<N1 | 유효 N0 / N1 / D0 |')
    lines=[head,'|'+'---:|'*8]
    for t in table:
        n0,n1,d0=(t[c] for c in CASES)
        change=lambda c:f"{(t[c]['rmse_mean_mm']/n0['rmse_mean_mm']-1)*100:+.1f}%"
        count=lambda k:f"{t[k]['seeds']}/{t[k]['valid_pairs']}"
        lines.append(f"| {t['amplitude_abs_nm']:g} | {t['percent_of_limit']:.1f}% "
            f"| {n0['rmse_mean_mm']:.2f} / {n1['rmse_mean_mm']:.2f} / {d0['rmse_mean_mm']:.2f} "
            f"| {change('N1')}, {change('D0')} "
            f"| {n0['j_clip_nm_s']:.1f} / {n1['j_clip_nm_s']:.1f} / {d0['j_clip_nm_s']:.1f} "
            f"| {n0['clip_percent']:.0f}% / {n1['clip_percent']:.0f}% / {d0['clip_percent']:.0f}% "
            f"| {count('N1_better_than_N0')}, {count('D0_better_than_N0')}, {count('D0_better_than_N1')} "
            f"| {n0['valid']} / {n1['valid']} / {d0['valid']} |")
    (OUT/'table.md').write_text('\n'.join(lines)+'\n\n개선 seed: 두 실행이 모두 유효한 seed 중 RMSE가 낮은 수 / 비교한 seed 수.\n'
        'RMSE와 J_clip은 무효 실행을 포함한 10 seed 평균이다. 유효가 10 미만인 행은 성능 비교에 쓰지 않는다.\n')
    print('\n'.join(lines))


COLORS=dict(N0='#2a78d6',N1='#eb6834',D0='#1baf7a')   # validated categorical slots 1-3
INK,MUTED,GRID_INK,SURFACE='#0b0b0b','#52514e','#e4e3df','#fcfcfb'


def figures(seeds,folder=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((OUT/'summary.json').read_text());table=summary['table']
    folder=Path(folder or RAW/'figures');folder.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.edgecolor':GRID_INK,'axes.labelcolor':INK,'xtick.color':MUTED,
        'ytick.color':MUTED,'axes.grid':True,'grid.color':GRID_INK,'grid.linewidth':.8,'axes.facecolor':SURFACE,
        'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False})
    data=np.genfromtxt(gzip.open(OUT/'timeseries.csv.gz','rt'),delimiter=',',names=True,dtype=None,encoding=None)
    series=lambda case,a,stat,col:data[col][(data['case']==case)&(np.isclose(data['amplitude_abs_nm'],a))&(data['statistic']==stat)]
    time=series('N0',table[0]['amplitude_abs_nm'],'mean','t_s')

    # 1. RMSE versus disturbance, valid seeds marked.
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    levels=[t['amplitude_abs_nm'] for t in table if t['amplitude_abs_nm']>0]
    for case in CASES:
        rows=[t for t in table if t['amplitude_abs_nm']>0]
        mean=np.array([t[case]['rmse_mean_mm'] for t in rows]);valid=np.array([t[case]['valid'] for t in rows])
        ax.plot(levels,mean,color=COLORS[case],lw=2,label=case,zorder=3)
        full=valid==len(seeds)
        ax.scatter(np.array(levels)[full],mean[full],s=36,color=COLORS[case],edgecolor=SURFACE,linewidth=1.5,zorder=4)
        ax.scatter(np.array(levels)[~full],mean[~full],s=36,facecolor=SURFACE,edgecolor=COLORS[case],linewidth=1.5,zorder=4)
    base=next(t for t in table if t['amplitude_abs_nm']==0)['N0']['rmse_mean_mm']
    ax.axhline(base,color=MUTED,lw=1);ax.text(levels[0],base*.86,'no disturbance',color=MUTED,fontsize=9,va='top')
    ax.set_yscale('log');ax.set_xlabel('J2 disturbance [Nm]');ax.set_ylabel('TCP RMSE, 5–15 s [mm]')
    ax.set_title('Filled: all 10 seeds valid   Open: at least one invalid seed',fontsize=9,color=MUTED,loc='left')
    ax.legend(loc='upper left');fig.tight_layout();fig.savefig(folder/'rmse_vs_disturbance.png',dpi=200);plt.close(fig)

    # 2. Disturbance profiles at representative levels.
    shown=representative(table,len(seeds))
    # Magnitude, not identity: a neutral ramp keeps the series hues for N0/N1/D0.
    fig,ax=plt.subplots(figsize=(5.6,3.4));grays=['#c9c8c2','#a3a29c','#7a7974','#52514e','#0b0b0b']
    for color,a in zip(grays,shown):
        ax.plot(time,series('N0',a,'mean','d_true_j2_nm'),color=color,lw=2,label=f'-{a:g} Nm')
    ax.axvline(2.5,color=GRID_INK,lw=1);ax.text(2.6,-2,'ramp end',color=MUTED,fontsize=9,va='top')
    ax.set_xlim(0,15);ax.set_xlabel('Time [s]');ax.set_ylabel('J2 disturbance d [Nm]')
    ax.legend(loc='upper right');fig.tight_layout();fig.savefig(folder/'disturbance_profiles.png',dpi=200);plt.close(fig)

    # 3. Time series at each representative level: mean line, min-max band over seeds.
    for a in shown:
        fig,axes=plt.subplots(1,3,figsize=(14,3.8))
        for case in CASES:
            for ax,column in zip(axes,('ee_error_mm','cumulative_clip_nm_s','requested_j2_nm')):
                ax.plot(time,series(case,a,'mean',column),color=COLORS[case],lw=2,label=case)
                ax.fill_between(time,series(case,a,'min',column),series(case,a,'max',column),color=COLORS[case],alpha=.12,lw=0)
        axes[2].axhline(-LIMIT,color=INK,lw=1);axes[2].text(15,-LIMIT,' limit',va='center',color=INK,fontsize=9)
        for ax,label in zip(axes,('TCP error [mm]','Cumulative clipping [Nm s]','Requested J2 torque [Nm]')):
            # The reference is stop-padded after 15.7 s; plot the evaluated span only.
            ax.set_xlim(0,15);ax.set_xlabel('Time [s]');ax.set_ylabel(label);ax.axvline(2.5,color=GRID_INK,lw=1)
        axes[2].set_xlim(2.4,15)
        row=next(t for t in table if t['amplitude_abs_nm']==a)
        valid=' / '.join(f"{c} {row[c]['valid']}/{len(seeds)}" for c in CASES)
        fig.suptitle(f'J2 disturbance -{a:g} Nm   line: mean, band: min–max over {len(seeds)} seeds   valid: {valid}',
                     fontsize=10,color=MUTED,x=.01,ha='left')
        axes[0].legend(loc='upper left');fig.tight_layout();fig.savefig(folder/f'timeseries_{a:g}Nm.png',dpi=200);plt.close(fig)
    print('[FIGURES]',folder,flush=True)


def representative(table,count):
    """Before saturation, first degradation, the main 45 Nm case, last all-valid and first all-invalid level."""
    levels=[t for t in table if t['amplitude_abs_nm']>0]
    before=[t['amplitude_abs_nm'] for t in levels if all(t[c]['clip_percent']==0 for c in CASES)]
    degraded=[t['amplitude_abs_nm'] for t in levels
              if t['N0']['rmse_mean_mm']>1.05*next(x for x in table if x['amplitude_abs_nm']==0)['N0']['rmse_mean_mm']]
    valid=[t['amplitude_abs_nm'] for t in levels if all(t[c]['valid']==count for c in CASES)]
    invalid=[t['amplitude_abs_nm'] for t in levels if all(t[c]['valid']==0 for c in CASES)]
    chosen=[before[-1] if before else levels[0]['amplitude_abs_nm'],degraded[0] if degraded else 45.,
            45.,valid[-1] if valid else 45.,invalid[0] if invalid else levels[-1]['amplitude_abs_nm']]
    return sorted(dict.fromkeys(chosen))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seeds',type=int,nargs='+',default=TEST_SEEDS)
    p.add_argument('--workers',type=int,default=20)
    p.add_argument('--summarize',action='store_true')
    p.add_argument('--figures',action='store_true')
    p.add_argument('--group',help=argparse.SUPPRESS)
    p.add_argument('--amplitudes',type=float,nargs='+',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.group:
        run_group(a.group,a.amplitudes,a.seeds);return
    if not (a.summarize or a.figures):
        simulate_levels(GRID,a.seeds)
        level=GRID[-1]
        while any_valid(level,a.seeds) and level<EXTENSION_LIMIT:
            level+=EXTENSION_STEP;simulate_levels([level],a.seeds)
    if not a.figures or a.summarize or not (OUT/'summary.json').exists():
        summarize(a.seeds,a.workers)
    figures(a.seeds)


if __name__=='__main__':main()
