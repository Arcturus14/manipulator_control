"""Payload GUI execution/replay, selectable plots, CSV and rigid-cube animation."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import yaml

CASES=('N0','N1','D0')
LIMITS=np.array([150.,150.,150.,28.,28.,28.])


def load_result(directory):
    with np.load(Path(directory)/'trajectory.npz',allow_pickle=False) as archive:
        result={key:archive[key] for key in archive.files}
    if 'payload_sample_ee' in result:
        result['_animation_samples']=[dict(ee=e,omega=w,ee_mean=m) for e,w,m in zip(
            result['payload_sample_ee'],result['payload_sample_omega'],result['payload_sample_mean'])]
    return result


def torque_metrics(result):
    req=result['u_requested'];loss=result['u_clamp_loss']
    solve=result['solve_t'];finite=solve[np.isfinite(solve)]
    return dict(rmse_err_mm=float(np.sqrt(np.mean(result['err_mm']**2))),
        max_err_mm=float(np.max(result['err_mm'])),
        mean_solve_ms=float(finite.mean()*1000) if len(finite) else None,
        worst_solve_ms=float(finite.max()*1000) if len(finite) else None,
        dob_rms_nm=float(np.sqrt(np.mean(result['d_hat_used']**2))),
        req_min_nm=req.min(0).tolist(),req_max_nm=req.max(0).tolist(),
        req_rms_nm=np.sqrt(np.mean(req**2,axis=0)).tolist(),
        peak_utilization=np.max(np.abs(req),axis=0).__truediv__(LIMITS).tolist(),
        min_headroom_nm=(LIMITS-np.max(np.abs(req),axis=0)).tolist(),
        clipping_steps=int(np.any(np.abs(loss)>1e-5,axis=1).sum()),
        J_clip_l2_nm_s=float(np.linalg.norm(loss,axis=1).sum()*.02))


def save_outputs(results,destination,torque=True,components=True,disturbance_error=True,csv=True):
    import matplotlib.pyplot as plt
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    metrics={key:torque_metrics(r) for key,r in results.items()}
    (destination/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False))
    for name,result in results.items():
        time=np.arange(len(result['u_requested']))*.02
        for enabled,kind in [(torque,'torque'),(components,'components'),(disturbance_error,'disturbance')]:
            if not enabled:continue
            fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
            for j,axis in enumerate(axes.flat):
                series=({'u_req':result['u_requested'],'u_app':result['u_applied']} if kind=='torque' else
                    {'u_mppi':result['u_mppi'],'compensation':result['u_requested']-result['u_mppi'],
                     'u_req':result['u_requested'],'u_app':result['u_applied']} if kind=='components' else
                    {'d_true':result['d_true'][1:len(time)+1],'d_hat_used':result['d_hat_used'],
                     '|d-d_hat_used|':np.abs(result['d_true'][1:len(time)+1]-result['d_hat_used'])})
                for label,values in series.items():axis.plot(time,values[:len(time),j],label=label,lw=1)
                if kind!='disturbance':
                    axis.axhline(LIMITS[j],color='r',ls='--');axis.axhline(-LIMITS[j],color='r',ls='--')
                axis.set_title(f'J{j+1}');axis.set_ylabel('Nm');axis.grid(alpha=.25)
            for axis in axes[-1]:axis.set_xlabel('Time [s]')
            axes[0,0].legend(fontsize=8);fig.suptitle(f'{name}: payload TCP / {kind}')
            fig.tight_layout();fig.savefig(destination/f'{name}_{kind}.png',dpi=160)
            plt.close(fig)
        if csv:
            columns=[time];headers=['time_s']
            for key in ('u_mppi','d_hat_used','d_true','u_requested','u_applied','u_clamp_loss','q','qd'):
                columns.extend(result[key][:len(time)].T)
                headers.extend(f'{key}_J{j+1}' for j in range(6))
            columns.append(result['err_mm']);headers.append('ee_error_mm')
            np.savetxt(destination/f'{name}_joints.csv',np.column_stack(columns),delimiter=',',header=','.join(headers),comments='')
    return metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=Path('configs/payload_cube_tcp.yaml'))
    p.add_argument('--case',choices=[*CASES,'N0 vs N1 vs D0'],default='N0')
    p.add_argument('--seed',type=int,default=6807)
    p.add_argument('--output-root',type=Path,default=Path('kroc/results/payload_gui'))
    p.add_argument('--replay',type=Path)
    for flag,default in [('show',True),('torque',True),('components',False),('disturbance-error',False),('csv',False),('capture-samples',False)]:
        p.add_argument('--'+flag,action=argparse.BooleanOptionalAction,default=default)
    args=p.parse_args();cfg=yaml.safe_load(args.config.read_text())
    cases=CASES if args.case=='N0 vs N1 vs D0' else [args.case]
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    destination=args.output_root/f'{stamp}_payload_seed{args.seed}';destination.mkdir(parents=True)
    results={}
    if args.replay:
        for case in cases:
            directory=args.replay/case if (args.replay/case/'trajectory.npz').exists() else args.replay
            resolved=json.loads((directory/'resolved_config.json').read_text())
            if resolved['case']!=case:raise ValueError(f'Replay case mismatch: requested {case}, saved {resolved["case"]}')
            results[case]=load_result(directory)
        (destination/'replay_source.json').write_text(json.dumps(dict(source=str(args.replay.resolve()),cases=list(cases)),indent=2))
    else:
        from kroc.run_payload_cube_tcp import run_payload
        for case in cases:
            results[case],_,_=run_payload(cfg,args.seed,case,output=destination/case,capture_samples=args.capture_samples)
    # The simulation imports batch modules that select Agg: switch only after running.
    import matplotlib
    matplotlib.use('Agg')
    if args.show:
        from visualization.backend import enable_gui_backend
        enable_gui_backend()
    metrics=save_outputs(results,destination,args.torque,args.components,args.disturbance_error,args.csv)
    from visualization.animator import animate_comparison
    import matplotlib.pyplot as plt
    reference=next(iter(results.values()))['tcp_reference']
    for result in results.values():np.testing.assert_allclose(result['tcp_reference'],reference)
    fig,animation=animate_comparison(results,reference)
    for frame in [0,len(next(iter(results.values()))['q'])-1]:
        animation._func(frame);fig.savefig(destination/f'animation_frame_{frame:04d}.png',dpi=130)
    animation._func(0)
    print(json.dumps(metrics,indent=2));print('[OUTPUT]',destination,flush=True)
    if args.show:plt.show()
    else:
        animation._draw_was_started=True;plt.close(fig)


if __name__=='__main__':main()
