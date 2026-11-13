"""GPU-only collision fusion regression/timing; JSON evidence, no MD report.

Run on each target machine with the same --input and batch size. Compilation
is excluded. The reference backend retains the original 26-DOP equations.
"""
import argparse
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import Q_MAX,Q_MIN
from kroc.self_collision_cost import PolytopeCollisionCost


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path(
        'kroc/results/circle_collision_cost_boundary_pilot/s1_seed7_directional25_5.npz'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=200)
    a=parser.parse_args();metadata=cuda_preflight()
    if a.repeats<1:parser.error('repeats must be positive')
    a.output.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda');torch.set_float32_matmul_precision('high')
    z=np.load(a.input);samples=136
    state=torch.tensor(np.column_stack((z['q'][1:1+samples],z['qd'][1:1+samples])),
                       device=device,dtype=torch.float32)
    if len(state)!=samples:raise ValueError('input needs at least 136 true states')
    costs={b:PolytopeCollisionCost(device,torch.float32,backend=b)
           for b in ('reference','fused','triton')}
    options={'triton.cudagraphs':False,'triton.cudagraph_trees':False}
    functions={b:torch.compile(c.__call__,options=options) for b,c in costs.items()}
    # Include broad physically bounded joint configurations, recorded states,
    # and permutations; never substitute CPU closed-loop benchmark results.
    torch.manual_seed(20261112)
    lo=torch.tensor(Q_MIN,device=device);hi=torch.tensor(Q_MAX,device=device)
    random_q=lo+torch.rand((8*samples,6),device=device)*(hi-lo)
    validation=torch.cat((torch.cat((random_q,torch.zeros_like(random_q)),1),state),0)
    errors={'cost_max_abs':0.,'cost_max_relative_to_clamp1':0.,'gap_max_abs_m':0.,'floor_max_abs_m':0.,
            'unambiguous_threshold_disagreements':0,'states':len(validation)}
    seps={b:torch.compile(costs[b].separations,options=options) for b in ('fused','triton')}
    for b,sep in seps.items():
      for chunk in validation.split(samples):
        expected=functions['reference'](chunk);actual=functions[b](chunk)
        torch.testing.assert_close(actual,expected,rtol=2e-4,atol=.05)
        rp,rf=costs['reference'].separations_reference(chunk[:,:6]);fp,ff=sep(chunk[:,:6])
        torch.testing.assert_close(fp,rp,rtol=0,atol=3e-6)
        torch.testing.assert_close(ff,rf,rtol=0,atol=3e-6)
        errors['cost_max_abs']=max(errors['cost_max_abs'],float((actual-expected).abs().max()))
        errors['cost_max_relative_to_clamp1']=max(errors['cost_max_relative_to_clamp1'],
            float(((actual-expected).abs()/expected.abs().clamp_min(1)).max()))
        errors['gap_max_abs_m']=max(errors['gap_max_abs_m'],float((fp-rp).abs().max()))
        errors['floor_max_abs_m']=max(errors['floor_max_abs_m'],float((ff-rf).abs().max()))
        errors['unambiguous_threshold_disagreements']+=int((((fp<.005)!=(rp<.005))&((rp-.005).abs()>3e-6)).sum())
    assert errors['unambiguous_threshold_disagreements']==0
    timings={b:[] for b in functions}
    # Alternate order across rounds to limit clock/temperature/order bias.
    for round_index in range(4):
        for b in (list(functions) if round_index%2==0 else list(functions)[::-1]):
            f=functions[b]
            for _ in range(5):f(state)
            torch.cuda.synchronize();begin=time.perf_counter()
            for _ in range(a.repeats):f(state)
            torch.cuda.synchronize()
            timings[b].append((time.perf_counter()-begin)*1000/a.repeats)
    result={'cuda':metadata,'input':str(a.input),'batch_size':samples,
            'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),
            'source_sha256':{name:hashlib.sha256(Path(name).read_bytes()).hexdigest()
                             for name in ('kroc/self_collision_cost.py','kroc/collision_kernels.py')},
            'repeats_per_round':a.repeats,'compilation_excluded':True,
            'geometry_unchanged':True,'headroom_enabled':False,
            'validation':errors,'call_ms_rounds':timings,
            'call_ms_median':{b:float(np.median(t)) for b,t in timings.items()}}
    for b,f in functions.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as p:
            for _ in range(10):f(state)
            torch.cuda.synchronize()
        result.setdefault('profile',{})[b]={'bmm_calls_per_10_evaluations':sum(
            e.count for e in p.key_averages() if e.key=='aten::bmm')}
    result['speedup']={b:result['call_ms_median']['reference']/t
                       for b,t in result['call_ms_median'].items()}
    (a.output/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
