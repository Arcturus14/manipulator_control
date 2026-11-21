"""Frozen ESS-improved sampler: causal DOB Q/acceleration LPF ablation. No ZIP/default changes."""
import gc,hashlib,json
from pathlib import Path
import numpy as np
import torch
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from experiment_runner import run_mppi
from config import TAU_MAX
from kroc.improve_n1_clipping_cost import ClippingLossN1,COVARIANCE,TEMPERATURE
from kroc.position_input_headroom_study import PositionInputTracker,metrics,usable,raised_cosine_load
from kroc.run_pre_input_baseline import load_preset
from kroc.run_input_headroom_study import run_options
from kroc.search_lowfreq_estimator_margin import geometry
from kroc.audit_circle_validation import dense_mesh_check
from kroc.final_input_headroom_checks import window_metrics
ROOT=Path('kroc/results/req_torque_filter_check_20261116')
FILTERS={'q3_off':(3.,0.),'q2_off':(2.,0.),'q1_off':(1.,0.),'q3_a10':(3.,10.),'q3_a6':(3.,6.),'q2_a10':(2.,10.)}
PILOT=[6707,6717];HELDOUT=[6807,6817,6827]
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False))
def rate(a):return np.sqrt(np.mean((np.diff(a,axis=0)/.02)**2,axis=0))
def details(z):
 v=z['u_mppi'];c=z['d_hat_used'];req=v+c;truth=z['d_true'][:-1];mask=np.arange(len(c))*.02>=3.1
 dv=np.diff(v.astype(float),axis=0)/.02;dc=np.diff(c.astype(float),axis=0)/.02
 a=z['dob_acceleration_raw'];m=z['measured_state'];true=np.concatenate([z['q'],z['qd']],axis=1)
 return dict(request_rate=float(np.sqrt(np.mean((dv+dc)**2))),nominal_rate=float(np.sqrt(np.mean(dv**2))),compensation_rate=float(np.sqrt(np.mean(dc**2))),
  request_rate_per_joint=rate(req).tolist(),nominal_rate_per_joint=rate(v).tolist(),compensation_rate_per_joint=rate(c).tolist(),
  cross_rate_squared_per_joint=np.mean(2*dv*dc,axis=0).tolist(),
  estimate_rmse_nm=float(np.sqrt(np.mean((c-truth)**2))),estimate_j3_rmse_nm=float(np.sqrt(np.mean((c[:,2]-truth[:,2])**2))),
  steady_estimate_rmse_nm=float(np.sqrt(np.mean((c[mask]-truth[mask])**2))),steady_estimate_bias_per_joint=np.mean(c[mask]-truth[mask],axis=0).tolist(),
  acceleration_noise_rms_per_joint=np.sqrt(np.mean(((np.diff(m[:,6:],axis=0)-np.diff(true[:,6:],axis=0))/.02)**2,axis=0)).tolist(),
  steady_request_rate=float(np.sqrt(np.mean((np.diff(req[mask],axis=0)/.02)**2))))
def run(phase,profile,flt,seed,controller='N0',no_sensor=False):
 folder=ROOT/phase/profile;folder.mkdir(parents=True,exist_ok=True)
 path=folder/f'{flt}_{controller}_{seed}{"_no_sensor" if no_sensor else ""}.npz';meta=path.with_suffix('.json')
 if meta.exists():return json.loads(meta.read_text())
 q,a=FILTERS[flt];amp=0. if profile=='nominal' else -133.
 print('RUN',phase,profile,flt,seed,controller,no_sensor,flush=True)
 if not path.exists():
  ref,q0,_=geometry(.89);opts=run_options(load_preset(),seed,controller,0.)
  opts.update(temperature=TEMPERATURE,noise_variance=COVARIANCE,tracker_class=PositionInputTracker if controller=='N0' else ClippingLossN1)
  opts['tracker_options'].update(dob_cutoff_hz=q,dob_acceleration_lpf_cutoff_hz=a,headroom_fraction=.95,unclamped_diagnostic=False)
  if controller=='N1':opts['tracker_options']['clipping_weight']=1.
  if no_sensor:opts['tracker_options']['sensor_noise'].update(position_std_rad=0.,velocity_std_rad_s=0.)
  z=run_mppi(ref,initial_q=q0,has_disturbance=bool(amp),disturbance_function=raised_cosine_load(2,amp),**opts)
  z.update(circle_reference=ref)
  assert str(z['compute_device'])=='cuda'
  assert not bool(z['input_study_fixed_clip']) and not bool(z['input_study_saturation_rollout'])
  np.testing.assert_allclose(z['u_applied'],np.clip(z['u_mppi']+z['d_hat_used'],-np.array(TAU_MAX),TAU_MAX),atol=3e-6,rtol=1e-6)
  assert float(z['dob_cutoff_hz'])==q and float(z['dob_acceleration_lpf_cutoff_hz'])==a
  np.savez_compressed(path,**{k:v for k,v in z.items() if not k.startswith('_')});del z;gc.collect();torch.cuda.empty_cache()
 mesh=dense_mesh_check((str(path),4))
 with np.load(path) as z:
  row=dict(phase=phase,profile=profile,filter=flt,seed=seed,controller=controller,no_sensor=no_sensor,path=str(path),
   full_metrics=metrics(z),detail=details(z),mesh_5ms=mesh,windows={n:window_metrics(z,s,e) for n,s,e in [('rise',.5,2.5),('plateau',2.5,6.)]})
 row['safe']=usable(row);write(meta,row)
 print('RESULT',flt,profile,controller,seed,'rmse',row['full_metrics']['rmse_mm'],'reqrate',row['detail']['request_rate'],'safe',row['safe'],flush=True)
 return row

def select(rows):
 result=[]
 for name in FILTERS:
  if name=='q3_off':continue
  info={'filter':name,'all_safe':all(r['safe'] for r in rows if r['filter']==name)}
  for profile in ['nominal','loaded']:
   base=[r for r in rows if r['filter']=='q3_off' and r['profile']==profile];cur=[r for r in rows if r['filter']==name and r['profile']==profile]
   info[profile]={k:float(np.mean([r[group][k] for r in cur])/np.mean([r[group][k] for r in base])) for group,k in [('full_metrics','rmse_mm'),('detail','request_rate'),('detail','estimate_rmse_nm')]}
  info['eligible']=info['all_safe'] and all(info[p]['rmse_mm']<=1.05 and info[p]['request_rate']<=.85 for p in ['nominal','loaded']) and info['loaded']['estimate_rmse_nm']<=1.10
  result.append(info)
 pool=[r for r in result if r['eligible']]
 selected=min(pool or result,key=lambda r: sum(r[p]['rmse_mm'] for p in ['nominal','loaded'])+.25*sum(r[p]['request_rate'] for p in ['nominal','loaded']))
 return dict(selected=selected,candidates=result,gate_passed=bool(pool),frozen_before_heldout=True)

def main():
 cuda=cuda_preflight();ROOT.mkdir(parents=True,exist_ok=True)
 source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 plan=dict(cuda=cuda,filters=FILTERS,pilot=PILOT,heldout=HELDOUT,variance=COVARIANCE,temperature=TEMPERATURE,source_sha256=source,
  selection='N0 pilot only; all safe; both profiles RMSE<=1.05 baseline and request rate<=.85; loaded estimate RMSE<=1.10. Rank sum tracking ratios +.25 sum rate ratios. If none eligible validate best diagnostic candidate without success claim.',
  profiles={'nominal':0.,'loaded':-133.},validation='Selected vs baseline, nominal and loaded, N0 and revised N1, three fresh paired seeds; no postvalidation retuning.',no_zip=True,no_defaults_changed=True)
 if (ROOT/'plan.json').exists():assert json.loads((ROOT/'plan.json').read_text())==plan
 else:write(ROOT/'plan.json',plan)
 rows=[]
 for profile in ['nominal','loaded']:
  for seed in PILOT:
   for flt in FILTERS:rows.append(run('pilot',profile,flt,seed));write(ROOT/'pilot.json',rows)
 selected=select(rows);sp=ROOT/'selection.json'
 if sp.exists():assert json.loads(sp.read_text())==selected
 else:write(sp,selected)
 print('SELECTION',selected,flush=True)
 rows=[]
 for profile in ['nominal','loaded']:
  for seed in HELDOUT:
   for controller in ['N0','N1']:
    for flt in ['q3_off',selected['selected']['filter']]:rows.append(run('validation',profile,flt,seed,controller));write(ROOT/'validation.json',rows)
 for flt in ['q3_off',selected['selected']['filter']]:rows.append(run('sensor_diagnostic','nominal',flt,PILOT[0],no_sensor=True));write(ROOT/'sensor_diagnostic.json',rows[-2:])
 write(ROOT/'completed.json',dict(pilot=24,validation=24,sensor_diagnostic=2,total=50))
if __name__=='__main__':main()
