"""Audit raw CUDA circle experiments and summarize actual actuator/state behavior."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from config import DT,Q_MAX,Q_MIN,TAU_MAX,DQ_MAX
from kinematics.ur5_kinematics import fk_joints_np
from kroc.benchmark_postdob_small_directional_margin import stat


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    a=p.parse_args();manifest=json.loads((a.directory/'manifest.json').read_text())
    assert manifest['headroom_enabled'] is False
    assert all(c['mode'] in ('fixed','directional') and not c['output_compensated_headroom'] for c in manifest['cases'].values())
    rows=json.loads((a.directory/'metrics.json').read_text());out=[]
    j=manifest['arguments']['joint'];limit=np.array(TAU_MAX)*manifest['arguments']['scale']
    for r in rows:
        z=np.load(a.directory/f's{r["scenario"]}_seed{r["seed"]}_{r["case"]}.npz')
        assert str(z['compute_device'])=='cuda'
        assert str(z['dob_compensation_mode'])=='output'
        assert float(z['dob_acceleration_lpf_cutoff_hz'])==10. and float(z['dob_cutoff_hz'])==8.
        assert all(np.isfinite(z[k]).all() for k in ['q','qd','u_applied','u_mppi','d_hat_used','circle_reference'])
        assert np.max(abs(z['u_applied']-np.clip(z['u_mppi']+z['d_hat_used'],-limit,limit)))<1e-4
        ref=z['circle_reference'];center=ref.mean(0)
        radii=np.linalg.norm(ref-center,axis=1)
        assert np.max(abs(radii-r['radius']))<1e-6
        q=z['q'][1:];v=z['qd'][1:];t=(np.arange(len(q))+1)*DT
        distance=np.minimum(q-np.array(Q_MIN),np.array(Q_MAX)-q)
        violation=(distance<0).any(1);ix=np.flatnonzero(violation)
        applied=z['u_applied'];req=z['u_mppi']+z['d_hat_used']
        sat=np.isclose(abs(applied),limit,rtol=0,atol=1e-5)
        late=t>=max(6.,r['onset']+r['ramp']+1)
        ff=z['q_reference'][:len(ref)]
        ik_error=max(np.linalg.norm(fk_joints_np(qr)[-1]-target) for qr,target in zip(ff,ref))
        assert ik_error<1e-4
        m=dict(r['metrics'])
        m.update(first_position_violation_s=float(t[ix[0]]) if len(ix) else None,
            ik_max_fk_error_m=float(ik_error),
            velocity_violation_steps=int((abs(v)>DQ_MAX).any(1).sum()),
            late_test_joint_saturation_steps=int(sat[late,j].sum()),
            late_tracking_rmse_mm=float(np.sqrt(np.mean(z['err_mm'][late]**2))),
            late_q1_mean_rad=float(q[late,0].mean()),
            late_q3_mean_rad=float(q[late,2].mean()),
            requested_excess_before_first_violation_nm_s=float(np.maximum(abs(req[:ix[0]+1,j])-limit[j],0).sum()*DT) if len(ix) else None,
            position_violation_seeds=int(violation.any()))
        out.append({**r,'metrics':m})
    aggregates={}
    for s in sorted({r['scenario'] for r in out}):
        agg={}
        for name in manifest['cases']:
            rr=[r for r in out if r['scenario']==s and r['case']==name]
            agg[name]={f:stat([r['metrics'][f] for r in rr]) for f in rr[0]['metrics'] if all(r['metrics'][f] is not None for r in rr)}
            agg[name]['violation_seed_count']=sum(r['metrics']['position_violation_seeds'] for r in rr)
            agg[name]['seed_count']=len(rr)
        aggregates[str(s)]=agg
    result={'all_checks_passed':True,'manifest':manifest,'rows':out,'aggregate':aggregates}
    (a.directory/'audited_results.json').write_text(json.dumps(result,indent=2))
    cell=[];labels=[]
    names=list(manifest['cases'])
    for s,agg in aggregates.items():
        r=next(r for r in out if r['scenario']==int(s));prefix=f'd={-r["sign"]*r["amplitude"]:g} Nm'
        fields=[('Violation seeds',None),('RMSE [mm]','tracking_rmse_mm'),('Late RMSE [mm]','late_tracking_rmse_mm'),
                ('Worst clearance [rad]','position_minimum_distance_rad'),('Saturation steps','test_joint_saturation_steps'),
                ('Mean q margin [%]','margin_mean_fraction'),('Solve [ms]','compute_mean_ms')]
        for title,f in fields:
            vals=[]
            for name in names:
                m=agg[name]
                if f is None:val=f'{m["violation_seed_count"]}/{m["seed_count"]}'
                elif f=='position_minimum_distance_rad':val=f'{m[f]["min"]:.3f}'
                else:
                    scale=100 if f=='margin_mean_fraction' else 1
                    val=f'{m[f]["mean"]*scale:.2f} +/- {m[f]["sample_std"]*scale:.2f}'
                vals.append(val)
            cell.append([prefix+' | '+title,*vals])
    fig,ax=plt.subplots(figsize=(13,.45*len(cell)+1));ax.axis('off')
    tab=ax.table(cellText=cell,colLabels=['Condition / metric','Static 2.5%','Directional 2.5-5%','Static 5%'],loc='center',cellLoc='center',colWidths=[.37,.21,.21,.21])
    tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,1.65)
    fig.savefig(a.directory/'results_table.png',dpi=160,bbox_inches='tight');plt.close(fig)
    print(json.dumps({s:{n:{'violation_seeds':m['violation_seed_count'],'rmse_mm':m['tracking_rmse_mm']['mean']} for n,m in g.items()} for s,g in aggregates.items()},indent=2))


if __name__=='__main__':main()
