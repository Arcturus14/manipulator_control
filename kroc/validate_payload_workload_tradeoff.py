"""Secondary tradeoff reproducibility, not a relaxed primary success claim."""
import json
from pathlib import Path
import yaml
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from kroc.run_payload_cube_tcp import run_payload
from kroc.validate_payload_circle_workload import summarize
from kroc.study_payload_torque_stability import write
from kroc.view_payload_tcp import load_result,save_outputs
from kroc.screen_payload_rectangle_workload import ROOT

def main():
    cuda=cuda_preflight();root=ROOT/'cuda'
    item=next(r for r in json.loads((ROOT/'offline/ranked_candidates.json').read_text()) if r['name']=='xz12_r0.02' and r['period_s']==5.)
    cfg=item['config']
    write(root/'secondary_validation_plan.json',dict(cuda=cuda,name=item['name'],period_s=5.,seeds=[6807,6817,6827],
        scope='Both circle and rectangle primary gates failed. Fixed pilot compromise selected for reproducibility only; original130Nm/3mm target remains unmet. No controller tuning, no DOB/disturbance, no N1/D0 comparisons.',
        selection='Pilot XZ24x20cm,2cm corner,T5s: RMSE2.76mm and peak124.6Nm without clipping. Do not reselect based on heldout seeds.'))
    rows=[]
    for seed in [6807,6817,6827]:
        path=root/f'xz12_r0.02_T5_seed{seed}'
        if (path/'verification.json').exists():z=load_result(path);v=json.loads((path/'verification.json').read_text())
        else:z,v,path=run_payload(cfg,seed,'N0',output=path)
        row=summarize(z,v,5.);row.update(seed=seed,path=str(path));rows.append(row)
        write(root/'secondary_validation.json',dict(original_target_met=False,rows=rows))
        save_outputs({'N0':z},path,torque=True,components=True,disturbance_error=False,csv=True)
        print('SECONDARY',seed,json.dumps(row),flush=True)
    if all(r['safe'] and not r['clipping_steps'] for r in rows):
        preset=Path('configs/payload_cube_tcp_workload_rectangle_trial_n0.yaml')
        if preset.exists():raise FileExistsError(preset)
        preset.write_text('# Experimental tradeoff only; primary130Nm/3mm target NOT met. DOB/disturbance not yet validated.\n'+yaml.safe_dump(cfg,sort_keys=False))
    print('[OUTPUT]',root,flush=True)

if __name__=='__main__':main()
