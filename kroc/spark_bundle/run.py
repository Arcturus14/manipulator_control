"""Run the self-contained snapshot; outputs stay inside this bundle."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['check','smoke','micro','validate','closedloop','all'])
    p.add_argument('--require-arm64',action='store_true')
    p.add_argument('--seeds',nargs='+',type=int,default=[7])
    p.add_argument('--repeats',type=int,default=200)
    a=p.parse_args();root=Path(__file__).resolve().parent;source=root/'source'
    if not (source/'config.py').is_file():raise SystemExit('Source snapshot missing. Transfer the complete bundle.')
    tag=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]
    output=root/'results'/f'{tag}_{a.mode}';output.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env['PYTHONPATH']=str(source)
    env['MPLBACKEND']='Agg';env['MPLCONFIGDIR']=str(root/'runtime'/'matplotlib')
    env['XDG_CACHE_HOME']=str(root/'runtime'/'cache')
    (root/'runtime'/'matplotlib').mkdir(parents=True,exist_ok=True)
    def run(name,args):
        cmd=[sys.executable,*map(str,args)]
        print('RUN',name,flush=True)
        with (output/f'{name}.log').open('w') as log:
            subprocess.run(cmd,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('PASS',name,flush=True)
    try:
        check=[root/'preflight.py','--output',output/'environment.json']
        if a.require_arm64:check+=['--require-arm64']
        run('preflight',check)
        if a.mode in {'smoke','all'}:
            run('unit_tests',['-m','unittest','kroc.test_self_collision_cost',
                'kroc.test_circle_reference','tests.test_dob_acceleration_filter',
                'tests.test_kroc_constraint_scenarios'])
        if a.mode in {'micro','all'}:
            run('micro',['-m','kroc.benchmark_collision_fusion','--repeats',a.repeats,
                         '--output',output/'micro'])
        if a.mode in {'validate','all'}:
            run('solve_validation',['-m','kroc.validate_collision_solve',
                                    '--output',output/'solve_validation'])
        if a.mode in {'closedloop','all'}:
            directories=[]
            for name,backend,compiled in [('reference','reference',False),
                                           ('triton','triton',False),('fullgraph','triton',True)]:
                directory=output/name;directories.append(directory)
                command=['-m','kroc.search_circle_floor25','--candidate-file',
                    'kroc/results/collision_free_circle_screen_outward/candidates.json',
                    '--poses','0','--joint','2','--fractions','.90','--base-fraction','.92',
                    '--radii','.04','--outward','--collision-check','--collision-weight','50000',
                    '--collision-backend',backend,'--amplitudes','135','--onsets','1.0',
                    '--ramps','.2','--seconds','8','--seeds',*a.seeds,'--output',directory]
                if compiled:command+=['--compile-rollout-loop']
                run(name,command)
            run('summary',['-m','kroc.summarize_collision_fusion','--inputs',*directories,
                           '--output',output/'comparison'])
        (output/'run_status.json').write_text(json.dumps({'status':'passed','mode':a.mode,
            'seeds':a.seeds,'output':str(output)},indent=2))
    except subprocess.CalledProcessError as exc:
        (output/'run_status.json').write_text(json.dumps({'status':'failed',
            'returncode':exc.returncode,'command':exc.cmd},indent=2))
        print(f'FAILED: see logs under {output}',file=sys.stderr)
        raise SystemExit(exc.returncode)
    print('RESULTS',output,flush=True)


if __name__=='__main__':main()
