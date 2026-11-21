"""Fixed tradeoff candidate validation, separate from failed >=20% pilot gate.

Stage2 rate4 improved HP5 by19.2% (not20%), rate by22.7% and EE RMSE by25.5%.
Validate this fixed practical candidate; no re-tuning on independent seeds.
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
STAGE2=Path('kroc/results/payload_torque_stability_20261121/20261121_201719_stage2')
CANDIDATE='posture5000_rate4'


def acceptable(r,b):
    return bool(r['safe'] and r['clipping_steps']==0 and not r['q3_zero_crossings_s'] and r['max_abs_j2_req_nm']<=110
        and r['mean_abs_j2_req_nm']>=80 and r['rmse_err_mm']<=2 and r['torque_hp5_rms']<b['torque_hp5_rms'] and r['torque_rate_rms']<b['torque_rate_rms'])


def main():
    cuda=cuda_preflight();root=Path('kroc/results/payload_torque_stability_20261121')/(datetime.now().strftime('%Y%m%d_%H%M%S')+'_validation');root.mkdir(parents=True)
    cfg=yaml.safe_load((STAGE2/CANDIDATE/'resolved_config.json').read_text())
    # Only schema fields from the actual fixed candidate, not resolved run metadata.
    cfg={k:copy.deepcopy(cfg[k]) for k in ['schema_version','payload','scene','controller','output_root']}
    baseline=json.loads((STAGE1/'baseline_metrics.json').read_text())
    pilot=json.loads((STAGE2/'pilot_metrics.json').read_text())[CANDIDATE]
    write(root/'plan.json',dict(cuda=cuda,source=str(STAGE2/CANDIDATE),candidate=CANDIDATE,heldout=[6817,6827],
        note='Original20% HP5 pilot goal was NOT met. This is an explicit fixed practical-tradeoff validation, not a claim that the original gate passed.',
        criteria='safe, no clipping/J3 sign crossing, peak J2<=110Nm, mean|J2|>=80Nm, RMSE<=2mm, HP5 and torque-rate both lower than paired baseline. No reselection.',
        two_cycles_seed=6807,original_controller_presets_unchanged=True))
    heldout={}
    for seed in [6817,6827]:
        z,v,path=run_payload(cfg,seed,'N0',output=root/f'seed{seed}')
        heldout[str(seed)]=metrics(z,v);write(root/'heldout_metrics.json',heldout)
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
    long=copy.deepcopy(cfg);long['scene']['duration_s']=40.
    z,v,path=run_payload(long,6807,'N0',output=root/'two_cycles_seed6807')
    two=metrics(z,v);write(root/'two_cycles_metrics.json',two)
    save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
    valid=all(acceptable(row,baseline[seed]) for seed,row in heldout.items()) and acceptable(two,baseline['6807'])
    preset=Path('configs/payload_cube_tcp_stabilized_n0.yaml')
    if valid:
        if preset.exists():raise FileExistsError(preset)
        preset.write_text('# Optional practical N0 candidate, not a replacement for frozen research baselines.\n'+yaml.safe_dump(cfg,sort_keys=False))
    write(root/'assessment.json',dict(candidate=CANDIDATE,pilot=pilot,heldout=heldout,two_cycles=two,baseline=baseline,
        practical_checks_passed=valid,original_20pct_pilot_goal_passed=False,preset=str(preset) if valid else None,
        limitations='No-disturbance/DOB-OFF N0 only; N1/D0 performance and disturbance robustness are not validated for the new common costs.'))
    import matplotlib.pyplot as plt
    from scipy.signal import butter,sosfiltfilt
    results={'baseline':load_result(BASE/'r16cm_T20s'),'posture + rate':load_result(STAGE2/CANDIDATE)}
    fig,axes=plt.subplots(3,1,figsize=(13,11),sharex=True)
    for name,z in results.items():
        t=np.arange(len(z['u_requested']))*.02
        for ax,value in zip(axes,[z['u_requested'][:,1],z['q'][:-1,2],z['err_mm']]):ax.plot(t,value,label=name,lw=.9)
    for ax,label in zip(axes,['J2 requested torque [Nm]','J3 angle [rad]','TCP error [mm]']):ax.set_ylabel(label);ax.legend();ax.grid(alpha=.25)
    axes[1].axhline(0,color='black',lw=.7);axes[-1].set_xlabel('Time [s]');fig.tight_layout();fig.savefig(root/'stability_comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(13,9),sharex=True)
    for name,z in results.items():
        t=np.arange(len(z['u_applied']))*.02;hp=sosfiltfilt(butter(4,5,fs=50,btype='high',output='sos'),z['u_applied'],axis=0);mask=(t>=4)&(t<=6)
        for j,ax in enumerate(axes.flat):ax.plot(t[mask],hp[mask,j],label=name,lw=.9)
    for j,ax in enumerate(axes.flat):ax.set_title(f'J{j+1}');ax.set_ylabel('HP5 torque [Nm]');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Offline HP5 diagnostic only; actuator commands are not filtered')
    fig.tight_layout();fig.savefig(root/'torque_noise_zoom.png',dpi=170);plt.close(fig)
    from visualization.animator import animate_comparison
    z=results['posture + rate'];fig,anim=animate_comparison({'N0':z},z['tcp_reference']);anim._func(837)
    fig.savefig(root/'stabilized_pose_16p74s.png',dpi=150);anim._draw_was_started=True;plt.close(fig)
    print('[OUTPUT]',root,flush=True)


if __name__=='__main__':main()
