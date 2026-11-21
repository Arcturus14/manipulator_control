"""Fixed-setting N0/N1 confirmation: 7 fresh primary pairs and 5 adjacent pairs.

Reuse the original three primary pairs without rerunning or selecting them.
No tuning, GUI changes, ZIP, or final paper report in this phase.
"""
import gc,hashlib,json,shutil
from pathlib import Path
import numpy as np
import torch
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from experiment_runner import run_mppi
from config import TAU_MAX,DT
from kroc.improve_n1_clipping_cost import ClippingLossN1,COVARIANCE,TEMPERATURE
from kroc.position_input_headroom_study import PositionInputTracker,metrics,usable,raised_cosine_load
from kroc.run_pre_input_baseline import load_preset
from kroc.run_input_headroom_study import run_options
from kroc.search_lowfreq_estimator_margin import geometry
from kroc.audit_circle_validation import dense_mesh_check
from kroc.final_input_headroom_checks import window_metrics
from kroc.compare_covariance_headroom import joint_metrics
from kroc.check_req_torque_filters import details

ROOT=Path('kroc/results/n1_final_fixed_validation_20261116')
PRIOR=Path('kroc/results/req_torque_filter_check_20261116')
REUSED=[6807,6817,6827]
PRIMARY_NEW=[6907,6917,6927,6937,6947,6957,6967]
ADJACENT=[7007,7017,7027,7037,7047]
LOADS={'primary133':-133.,'adjacent132p5':-132.5}

def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def audit(z,case,amplitude):
 assert str(z['compute_device'])=='cuda'
 assert float(z['mppi_temperature'])==64. and float(z['dob_cutoff_hz'])==3. and float(z['dob_acceleration_lpf_cutoff_hz'])==6.
 assert float(z['mppi_temporal_noise_rho'])==1. and float(z['mppi_mean_update_alpha'])==1.
 assert float(z['input_study_headroom_weight'])==0.
 assert not bool(z['input_study_fixed_clip']) and not bool(z['input_study_saturation_rollout'])
 assert not bool(z['nonphysical_unclamped_diagnostic'])
 assert str(z['input_study_case'])==case
 if case=='N1':assert float(z['n1_clipping_weight'])==1.
 for key in ['q','qd','err_mm','u_mppi','u_applied','d_hat_used','state_prediction_error']:assert np.isfinite(z[key]).all(),key
 np.testing.assert_array_equal(z['u_effective_pred'],z['u_mppi'])
 np.testing.assert_allclose(z['u_applied'],np.clip(z['u_mppi']+z['d_hat_used'],-np.array(TAU_MAX),TAU_MAX),atol=3e-6,rtol=1e-6)
 np.testing.assert_array_equal(z['mppi_active_noise_variance'],np.broadcast_to(np.asarray(COVARIANCE,np.float32),(300,6)))
 t=np.arange(300)*DT;phase=np.clip((t-.5)/2.,0.,1.);d=np.zeros((300,6));d[:,2]=amplitude*.5*(1-np.cos(np.pi*phase))
 np.testing.assert_allclose(z['d_true'][1:],d,atol=1e-5,rtol=1e-6)
 assert float(z['sensor_position_std_rad'])==.001 and float(z['sensor_velocity_std_rad_s'])==.01

def freeze(cuda):
 ROOT.mkdir(parents=True,exist_ok=True)
 prior=json.loads((PRIOR/'plan.json').read_text())
 assert prior['heldout']==REUSED
 assert sha('kroc/check_req_torque_filters.py')==prior['source_sha256']
 original=json.loads(Path('kroc/results/n1_clipping_cost_redesign_20261116/plan.json').read_text())
 files=dict(original['source_sha256'])
 for f,h in files.items():assert sha(f)==h,f
 for f in ['kroc/check_req_torque_filters.py',str(Path(__file__).relative_to(Path.cwd()))]:files[f]=sha(f)
 plan=dict(cuda=cuda,reused_primary=REUSED,new_primary=PRIMARY_NEW,adjacent_seeds=ADJACENT,loads=LOADS,
  variance=COVARIANCE,temperature=TEMPERATURE,dob_cutoff_hz=3.,acceleration_lpf_hz=6.,clipping_cost_weight=1.,
  base=load_preset(),source_sha256=files,intended_new_runs=24,intended_loaded_total_runs=30,
  selection='No tuning or scenario/seed selection. Primary10 pairs = original3 + fresh7. Adjacent5 fresh pairs. All outcomes retained.',
  analysis='Paired mean/median/worst RMSE changes and wins, Jclip, full/rise(.5-2.5)/plateau(2.5-6) RMSE; 95% paired bootstrap percentile interval for mean difference (N1-N0), 20000 resamples seed99401; descriptive small-sample interval, no universal guarantee.',
  failure='Keep failed state/contact runs in all statistics with flags. No successful-only subset. Report severe tracking regressions without retuning.',
  no_zip=True,final_report_deferred=True)
 target=ROOT/'plan.json'
 if target.exists():assert json.loads(target.read_text())==plan
 else:
  write(target,plan)
  for f in files:
   out=ROOT/'source'/Path(f).resolve().relative_to(Path.cwd());out.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,out)
 return plan

def metadata(path,load,seed,case,origin):
 mesh=dense_mesh_check((str(path),4))
 with np.load(path) as z:
  audit(z,case,LOADS[load]);full=metrics(z)
  row=dict(load=load,amplitude=LOADS[load],seed=seed,case=case,origin=origin,path=str(path),full_metrics=full,detail=details(z),joints=joint_metrics(z),mesh_5ms=mesh,
    windows={name:window_metrics(z,start,end) for name,start,end in [('before',0,.5),('rise',.5,2.5),('plateau',2.5,6.)]})
 row['safe']=usable(row);row['raw_sha256']=sha(path)
 return row

def run_one(load,seed,case):
 folder=ROOT/'raw'/load;folder.mkdir(parents=True,exist_ok=True);path=folder/f'{case}_seed{seed}.npz';meta=path.with_suffix('.json')
 if meta.exists():
  row=json.loads(meta.read_text());assert sha(path)==row['raw_sha256'];return row
 print('RUN',load,case,seed,flush=True)
 if not path.exists():
  ref,q0,_=geometry(.89);opts=run_options(load_preset(),seed,case,0.)
  opts.update(temperature=TEMPERATURE,noise_variance=COVARIANCE,tracker_class=ClippingLossN1 if case=='N1' else PositionInputTracker)
  opts['tracker_options'].update(headroom_fraction=.95,unclamped_diagnostic=False,dob_cutoff_hz=3.,dob_acceleration_lpf_cutoff_hz=6.)
  if case=='N1':opts['tracker_options']['clipping_weight']=1.
  z=run_mppi(ref,initial_q=q0,has_disturbance=True,disturbance_function=raised_cosine_load(2,LOADS[load]),**opts)
  z.update(circle_reference=ref);audit(z,case,LOADS[load]);np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')})
  del z;gc.collect();torch.cuda.empty_cache()
 row=metadata(path,load,seed,case,'new');write(meta,row)
 print('RESULT',load,case,seed,'RMSE',row['full_metrics']['rmse_mm'],'Jclip',row['full_metrics']['jclip_nm_s'],'safe',row['safe'],flush=True)
 return row

def main():
 cuda=cuda_preflight();print('CUDA',cuda,flush=True);freeze(cuda)
 rows=[];integrity=json.loads((PRIOR/'integrity.json').read_text())
 for seed in REUSED:
  for case in ['N0','N1']:
   path=PRIOR/'validation'/'loaded'/f'q3_a6_{case}_{seed}.npz'
   assert sha(path)==integrity['raw_sha256'][str(path.relative_to(PRIOR))]
   rows.append(metadata(path,'primary133',seed,case,'reused'))
 write(ROOT/'progress.json',rows)
 for load,seeds in [('primary133',PRIMARY_NEW),('adjacent132p5',ADJACENT)]:
  for seed in seeds:
   for case in ['N0','N1']:
    rows.append(run_one(load,seed,case));write(ROOT/'progress.json',rows)
 assert len(rows)==30
 write(ROOT/'completed.json',dict(new_cuda_runs=24,reused_runs=6,primary_pairs=10,adjacent_pairs=5,all_safe=all(r['safe'] for r in rows)))
 print('COMPLETE',ROOT,flush=True)
if __name__=='__main__':main()
