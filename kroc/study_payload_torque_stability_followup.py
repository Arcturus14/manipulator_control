"""Predeclared second pilot stage after stage one failed all selection gates.

The independent candidate-validation seeds have NOT been evaluated in stage1.
"""
import copy
from datetime import datetime
import json
from pathlib import Path
import numpy as np
import yaml
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.run_payload_cube_tcp import run_payload
from kroc.study_payload_torque_stability import BASE,metrics,write
from kroc.view_payload_tcp import load_result,save_outputs

STAGE1=Path('kroc/results/payload_torque_stability_20261121/20261121_200925')
VARIANTS={'posture5000_rate4':dict(posture_weight=5000.,boundary_weight=4.,smooth_weight=4.),
          'posture5000_rate16':dict(posture_weight=5000.,boundary_weight=16.,smooth_weight=16.)}


def gate(r,b):
    return (r['safe'] and r['clipping_steps']==0 and not r['q3_zero_crossings_s'] and
        r['max_abs_j2_req_nm']<=110 and r['mean_abs_j2_req_nm']>=80 and r['rmse_err_mm']<=2 and
        r['torque_rate_rms']<=b['torque_rate_rms'] and r['torque_hp5_rms']<=.8*b['torque_hp5_rms'])


def main():
    cuda=cuda_preflight()
    assert json.loads((STAGE1/'selection.json').read_text())['selected'] is None
    assert not (STAGE1/'heldout_metrics.json').exists()
    root=Path('kroc/results/payload_torque_stability_20261121')/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_stage2');root.mkdir(parents=True)
    cfg=yaml.safe_load(Path('configs/payload_cube_tcp_large_yz_n0.yaml').read_text())
    baseline=json.loads((STAGE1/'baseline_metrics.json').read_text())
    write(root/'plan.json',dict(cuda=cuda,prior_pilots=str(STAGE1),variants=VARIANTS,pilot_seed=6807,
        heldout_seeds=[6817,6827],reason='Posture500 did not prevent elbow sign crossing; rate4 reduced HP5 by about20% but did not remove the transient.',
        unchanged='Same payload, physical/state bounds, collision penalties, sensor noise, covariance, lambda and DOB-OFF scene.',
        selection='Same stage1 gates, no relaxation. Minimize pilot HP5 among eligible, then freeze; no heldout reselection.',
        heldout_integrity='Stage1 selected nobody and ran no candidate on heldout seeds.'))
    rows={}
    for name,settings in VARIANTS.items():
        conf=copy.deepcopy(cfg);conf['controller']['experimental_stabilization']=settings
        z,v,path=run_payload(conf,6807,'N0',output=root/name)
        rows[name]=metrics(z,v);write(root/'pilot_metrics.json',rows)
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        print('PILOT',name,json.dumps(rows[name]),flush=True)
    eligible=[name for name,row in rows.items() if gate(row,baseline['6807'])]
    if not eligible:
        write(root/'selection.json',dict(selected=None,reason='No candidate passed unchanged pilot gates'));print('[OUTPUT]',root);return
    best=min(eligible,key=lambda name:rows[name]['torque_hp5_rms']);conf=copy.deepcopy(cfg)
    conf['controller']['experimental_stabilization']=VARIANTS[best]
    heldout={}
    for seed in [6817,6827]:
        z,v,path=run_payload(conf,seed,'N0',output=root/f'{best}_seed{seed}')
        heldout[str(seed)]=metrics(z,v);write(root/'heldout_metrics.json',heldout)
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
    long=copy.deepcopy(conf);long['scene']['duration_s']=40.
    z,v,path=run_payload(long,6807,'N0',output=root/f'{best}_two_cycles')
    long_result=metrics(z,v);save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
    valid=all(gate(row,baseline[seed]) for seed,row in heldout.items()) and gate(long_result,baseline['6807'])
    preset=Path('configs/payload_cube_tcp_stabilized_n0.yaml')
    if valid:
        if preset.exists():raise FileExistsError(preset)
        preset.write_text('# Optional N0 posture/noise candidate; original presets remain frozen.\n'+yaml.safe_dump(conf,sort_keys=False))
    write(root/'selection.json',dict(selected=best,pilot=rows[best],heldout=heldout,two_cycles=long_result,
        passes_all=bool(valid),preset=str(preset) if valid else None,
        scope='No-disturbance N0 only. N1/D0 disturbance superiority has not been validated with new common costs.'))
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,1,figsize=(13,11),sharex=True)
    for name,path in [('baseline',BASE/'r16cm_T20s'),(best,root/best)]:
        z=load_result(path);t=np.arange(len(z['u_requested']))*.02
        for ax,value in zip(axes,[z['u_requested'][:,1],z['q'][:-1,2],z['err_mm']]):ax.plot(t,value,label=name,lw=.9)
    for ax,label in zip(axes,['J2 requested torque [Nm]','J3 angle [rad]','TCP error [mm]']):ax.set_ylabel(label);ax.legend();ax.grid(alpha=.25)
    axes[1].axhline(0,color='black',lw=.7);axes[-1].set_xlabel('Time [s]');fig.tight_layout();fig.savefig(root/'stability_comparison.png',dpi=170);plt.close(fig)
    from scipy.signal import butter,sosfiltfilt
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for name,path in [('baseline',BASE/'r16cm_T20s'),(best,root/best)]:
        z=load_result(path);t=np.arange(len(z['u_applied']))*.02
        hp=sosfiltfilt(butter(4,5,fs=50,btype='high',output='sos'),z['u_applied'],axis=0);mask=(t>=4)&(t<=6)
        for j,ax in enumerate(axes.flat):ax.plot(t[mask],hp[mask,j],label=name,lw=.9)
    for j,ax in enumerate(axes.flat):ax.set_title(f'J{j+1}');ax.set_ylabel('Torque HP5 [Nm]');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Offline high-pass diagnostic only; commands are NOT filtered')
    fig.tight_layout();fig.savefig(root/'torque_noise_zoom.png',dpi=170);plt.close(fig)
    print('[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
