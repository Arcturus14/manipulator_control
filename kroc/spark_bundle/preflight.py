"""Integrity, native dependencies and a real CUDA/Triton smoke test."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def verify_bundle(root):
    manifest=json.loads((root/'bundle_manifest.json').read_text())
    for name,expected in manifest['files_sha256'].items():
        path=root/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
            raise RuntimeError(f'Bundle file missing or modified: {name}')
    return len(manifest['files_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--require-arm64',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parent
    verified=verify_bundle(root)
    if a.require_arm64 and platform.machine().lower() not in {'aarch64','arm64'}:
        raise SystemExit('ARM64 required; do not run an x86 container through emulation.')
    source=root/'source';sys.path.insert(0,str(source));os.chdir(source)
    import numpy as np
    import torch
    import triton
    from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
    from kroc.self_collision_cost import PolytopeCollisionCost
    import fcl
    cuda=cuda_preflight()
    q=np.array([.92*2*np.pi,-2,.9*np.pi,-2,0,0],dtype=np.float32)
    state=torch.tensor(np.tile(np.r_[q,np.zeros(6)],(136,1)),device='cuda',dtype=torch.float32)
    reference=PolytopeCollisionCost('cuda',torch.float32,backend='reference')
    optimized=PolytopeCollisionCost('cuda',torch.float32,backend='triton')
    compiled=torch.compile(optimized.__call__,options={
        'triton.cudagraphs':False,'triton.cudagraph_trees':False})
    expected=reference(state);actual=compiled(state);torch.cuda.synchronize()
    torch.testing.assert_close(actual,expected,rtol=2e-4,atol=.05)
    probe=torch.eye(6,device='cuda').expand(136,6,6)
    torch.linalg.solve(probe,torch.ones(136,6,1,device='cuda'))
    torch.cuda.synchronize()
    result={'status':'passed','architecture':platform.machine(),'python':sys.version,
            'cuda':cuda,'gpu_capability':list(torch.cuda.get_device_capability()),
            'verified_files':verified,'collision_smoke_max_abs':float((actual-expected).abs().max()),
            'packages':{n:importlib.metadata.version(n) for n in (
                'torch','triton','pytorch-mppi','numpy','scipy','matplotlib','trimesh','python-fcl','PyYAML')}}
    smi=subprocess.run(['nvidia-smi'],capture_output=True,text=True,check=False)
    result['nvidia_smi']=smi.stdout or smi.stderr
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
