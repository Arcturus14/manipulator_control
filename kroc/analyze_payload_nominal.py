"""Collect existing nominal payload results/figures without running control."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from kroc.view_payload_tcp import load_result,torque_metrics,save_outputs
from visualization.animator import animate_comparison


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--study',type=Path,required=True)
    args=p.parse_args();out=args.study/'analysis'
    selected=json.loads((args.study/'selected_scene.json').read_text());best=selected['name']
    baseline=load_result(args.baseline);chosen=load_result(args.study/best)
    out.mkdir(exist_ok=False)
    names=['baseline']+[r['name'] for r in json.loads((args.study/'rollout_metrics.json').read_text())]
    results={'baseline':baseline}
    for name in names[1:]:results[name]=load_result(args.study/name)
    rows=[]
    for name,z in results.items():
        metrics=torque_metrics(z)
        rows.append(dict(scene=name,seed=6807,**metrics))
    for seed in [6817,6827]:
        z=load_result(args.study/f'{best}_seed{seed}')
        rows.append(dict(scene=best,seed=seed,**torque_metrics(z)))
    (out/'summary.json').write_text(json.dumps(dict(baseline_source=str(args.baseline),study_source=str(args.study),
        selected_scene=best,results=rows,scope='N0 only: known payload 5kg, no disturbance, no DOB. Geometric/operating-point screen, not saturation or controller superiority evidence.'),indent=2))
    with (out/'joint_torques.csv').open('w') as handle:
        writer=csv.writer(handle);writer.writerow(['scene','seed','joint','req_min_nm','req_max_nm','req_rms_nm','peak_utilization','min_headroom_nm','EE_RMSE_mm'])
        for row in rows:
            for j in range(6):writer.writerow([row['scene'],row['seed'],j+1,*[row[k][j] for k in ['req_min_nm','req_max_nm','req_rms_nm','peak_utilization','min_headroom_nm']],row['rmse_err_mm']])
    save_outputs({'N0':chosen},out,torque=True,components=True,disturbance_error=False,csv=True)
    fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
    for j,axis in enumerate(axes.flat):
        for name,z in [('original pose',baseline),('extended YZ',chosen)]:
            axis.plot(np.arange(len(z['u_requested']))*.02,z['u_requested'][:,j],label=name,lw=1)
        axis.set_title(f'J{j+1} requested torque');axis.set_ylabel('Nm');axis.grid(alpha=.25)
    axes[0,0].legend();axes[-1,0].set_xlabel('Time [s]');axes[-1,1].set_xlabel('Time [s]')
    fig.suptitle('N0, 5 kg payload, no disturbance / DOB; same controller settings')
    fig.tight_layout();fig.savefig(out/'baseline_vs_extended_torques.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5),subplot_kw={'projection':'3d'})
    for axis,name,z in zip(axes,['Original pose','Extended YZ'],[baseline,chosen]):
        axis.plot(*z['tcp_reference'].T,'--',color='green',label='TCP reference')
        axis.plot(*z['ee'].T,color='tab:blue',label='object center')
        axis.set(xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]',title=name,
            xlim=(0,1),ylim=(-.1,.3),zlim=(0,.6));axis.set_box_aspect((1,.4,.6));axis.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'trajectory_locations.png',dpi=170);plt.close(fig)
    fig,animation=animate_comparison({'N0':chosen},chosen['tcp_reference'])
    for frame in [0,150,300]:
        animation._func(frame);fig.savefig(out/f'payload_animation_frame_{frame:04d}.png',dpi=150)
    animation._draw_was_started=True;plt.close(fig)
    print('[OUTPUT]',out)


if __name__=='__main__':main()
