"""Bounded opt-in posture/torque-noise study; GPU-only control, no ZIP."""
import copy
from datetime import datetime
import json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
import yaml
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.run_payload_cube_tcp import run_payload
from kroc.screen_payload_nominal import safe
from kroc.view_payload_tcp import load_result,torque_metrics,save_outputs

BASE=Path('kroc/results/payload_large_circle_20261121/20261121_174431')
VARIANTS={
 'diagnostic_pair_mask':dict(diagnostic_mask_base_upper_arm=True),
 'diagnostic_sensor_off':dict(diagnostic_sensor_noise_off=True),
 'posture500':dict(posture_weight=500.),
 'rate1':dict(boundary_weight=1.,smooth_weight=1.),
 'rate4':dict(boundary_weight=4.,smooth_weight=4.),
 'posture500_rate1':dict(posture_weight=500.,boundary_weight=1.,smooth_weight=1.)}


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False))


def metrics(z,v):
    m=torque_metrics(z);u=z['u_applied'];du=np.diff(u,axis=0)/.02
    hp=sosfiltfilt(butter(4,5,fs=50,btype='high',output='sos'),u,axis=0)
    q=z['q'];cross=np.flatnonzero(q[:-1,2]*q[1:,2]<0)*.02
    m.update(safe=safe(v),torque_rate_rms=float(np.sqrt(np.mean(du**2))),
        torque_rate_rms_per_joint=np.sqrt(np.mean(du**2,axis=0)).tolist(),
        torque_hp5_rms=float(np.sqrt(np.mean(hp**2))),
        torque_hp5_rms_per_joint=np.sqrt(np.mean(hp**2,axis=0)).tolist(),
        q3_zero_crossings_s=cross.tolist(),q3_min_rad=float(q[:,2].min()),
        max_abs_j2_req_nm=float(np.abs(z['u_requested'][:,1]).max()),
        mean_abs_j2_req_nm=float(np.abs(z['u_requested'][:,1]).mean()),
        ess_mean=float(z['mppi_weight_ess'].mean()),
        joint_reference_rmse_rad=np.sqrt(np.mean((q[:-1]-z['q_reference'][:len(q)-1])**2,axis=0)).tolist())
    return m


def main():
    cuda=cuda_preflight();cfg=yaml.safe_load(Path('configs/payload_cube_tcp_large_yz_n0.yaml').read_text())
    root=Path('kroc/results/payload_torque_stability_20261121')/datetime.now().strftime('%Y%m%d_%H%M%S');root.mkdir(parents=True)
    write(root/'plan.json',dict(cuda=cuda,variants=VARIANTS,pilot_seed=6807,heldout=[6817,6827],
        scope='N0 only, known 5kg, radius16cm period20s, DOB/disturbance OFF. Frozen baselines retained.',
        geometry='All physical limits and original collision costs retained for selectable candidates. Pair-mask and sensor-off runs diagnostic only.',
        posture_cost='w_q sum_j(q_j-q_ref_j)^2 per predicted stage; terminal 3*w_q, continuous existing TCP IK feedforward reference.',
        noise_metric='offline zero-phase Butterworth HP5Hz RMS; this filter is never applied to actuator commands.',
        selection='Exclude diagnostics. Pilot safe, no clipping, no J3 sign crossing, J2peak<=110Nm, mean|J2|>=80Nm, EE RMSE<=2mm, torque-rate<=baseline and HP5 RMS<=80%baseline. Minimize HP5 RMS. Fix then heldout and two-cycle run; do not reselect on heldout.'))
    # Confirm a disabled extension reproduces the original path exactly.
    a,_,_=run_payload(cfg,6807,'N0',steps=40,output=root/'zero_check_original')
    zero=copy.deepcopy(cfg);zero['controller']['experimental_stabilization']={}
    b,_,_=run_payload(zero,6807,'N0',steps=40,output=root/'zero_check_extension')
    for key in ['q','qd','u_mppi','u_applied','err_mm','d_hat_used']:np.testing.assert_array_equal(a[key],b[key])
    write(root/'zero_check.json',dict(bitwise_equal=True,keys=['q','qd','u_mppi','u_applied','err_mm','d_hat_used']))
    baseline={}
    for seed in [6807,6817,6827]:
        path=BASE/('r16cm_T20s' if seed==6807 else f'r16cm_T20s_seed{seed}')
        baseline[seed]=metrics(load_result(path),json.loads((path/'verification.json').read_text()))
    write(root/'baseline_metrics.json',baseline);rows={}
    for name,settings in VARIANTS.items():
        conf=copy.deepcopy(cfg);conf['controller']['experimental_stabilization']=settings
        z,v,path=run_payload(conf,6807,'N0',output=root/name)
        row=metrics(z,v);rows[name]=row;write(root/'pilot_metrics.json',rows)
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        print('PILOT',name,json.dumps(row),flush=True)
    original=baseline[6807]
    eligible=[name for name,r in rows.items() if not name.startswith('diagnostic_') and r['safe'] and r['clipping_steps']==0
        and not r['q3_zero_crossings_s'] and r['max_abs_j2_req_nm']<=110 and r['mean_abs_j2_req_nm']>=80
        and r['rmse_err_mm']<=2 and r['torque_rate_rms']<=original['torque_rate_rms'] and r['torque_hp5_rms']<=.8*original['torque_hp5_rms']]
    if not eligible:
        write(root/'selection.json',dict(selected=None,reason='No candidate passed predefined pilot criteria'));print('[OUTPUT]',root);return
    selected=min(eligible,key=lambda n:rows[n]['torque_hp5_rms']);conf=copy.deepcopy(cfg)
    conf['controller']['experimental_stabilization']=VARIANTS[selected]
    heldout={}
    for seed in [6817,6827]:
        z,v,path=run_payload(conf,seed,'N0',output=root/f'{selected}_seed{seed}')
        heldout[seed]=metrics(z,v);save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        write(root/'heldout_metrics.json',heldout)
    long=copy.deepcopy(conf);long['scene']['duration_s']=40.
    z,v,path=run_payload(long,6807,'N0',output=root/f'{selected}_two_cycles')
    long_metrics=metrics(z,v);save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
    valid=all(r['safe'] and r['clipping_steps']==0 and not r['q3_zero_crossings_s'] and r['max_abs_j2_req_nm']<=110 and r['rmse_err_mm']<=2
        and r['torque_hp5_rms']<=.8*baseline[seed]['torque_hp5_rms'] for seed,r in heldout.items())
    valid=valid and long_metrics['safe'] and not long_metrics['q3_zero_crossings_s'] and long_metrics['clipping_steps']==0 and long_metrics['rmse_err_mm']<=2 and long_metrics['max_abs_j2_req_nm']<=110
    preset=Path('configs/payload_cube_tcp_stabilized_n0.yaml')
    if valid:
        if preset.exists():raise FileExistsError(preset)
        preset.write_text('# Opt-in experimentally validated N0 candidate; frozen presets untouched.\n'+yaml.safe_dump(conf,sort_keys=False))
    write(root/'selection.json',dict(selected=selected,pilot=rows[selected],heldout=heldout,two_cycles=long_metrics,passes_all=bool(valid),preset=str(preset) if valid else None,
        limitation='N0 no-disturbance validation only. N0/N1/D0 disturbance comparison has not been repeated with these new common costs.'))
    # Include every diagnostic/candidate in the plotted and numerical record.
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,11),sharex=True)
    for name,path in [('baseline',BASE/'r16cm_T20s'),('pair mask (diagnostic)',root/'diagnostic_pair_mask'),(selected,root/selected)]:
        z=load_result(path);t=np.arange(len(z['u_requested']))*.02
        axes[0].plot(t,z['u_requested'][:,1],label=name,lw=1)
        axes[1].plot(t,z['q'][:-1,2],label=name,lw=1)
        axes[2].plot(t,z['err_mm'],label=name,lw=.8)
    for ax,label in zip(axes,['J2 torque [Nm]','J3 angle [rad]','TCP error [mm]']):ax.set_ylabel(label);ax.grid(alpha=.25);ax.legend(fontsize=8)
    axes[1].axhline(0,color='black',lw=.7);axes[-1].set_xlabel('Time [s]');fig.tight_layout();fig.savefig(root/'stability_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for name,path in [('baseline',BASE/'r16cm_T20s'),(selected,root/selected)]:
        z=load_result(path);t=np.arange(len(z['u_requested']))*.02
        for j,ax in enumerate(axes.flat):ax.plot(t,z['u_requested'][:,j],label=name,lw=.8)
    for j,ax in enumerate(axes.flat):ax.set_title(f'J{j+1}');ax.set_ylabel('Nm');ax.grid(alpha=.25)
    axes[0,0].legend();fig.tight_layout();fig.savefig(root/'all_joint_torques.png',dpi=170);plt.close(fig)
    print('[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
