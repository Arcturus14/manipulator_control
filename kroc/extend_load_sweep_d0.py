"""Add frozen D0 to every completed N0/N1 load and seed, without retuning."""
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from experiment_runner import run_mppi
from kroc.audit_circle_validation import dense_mesh_check
from kroc.optimize_final_mppi import check, options
from kroc.position_input_headroom_study import raised_cosine_load
from kroc.search_lowfreq_estimator_margin import geometry
from kroc.sweep_final_n1_load import ROOT as BASE, SEEDS, sha, summarize, write

ROOT = Path('kroc/results/final_three_way_load_sweep_20261117')


def freeze(cuda):
    prior = json.loads((BASE/'summary.json').read_text())
    assert prior['completed'] and prior['source_integrity_passed']
    files = dict(json.loads((BASE/'plan.json').read_text())['source_sha256'])
    for name, digest in files.items():
        assert sha(name) == digest, name
    files['kroc/extend_load_sweep_d0.py'] = sha(__file__)
    loads = [g['amplitude_nm'] for g in prior['groups']]
    plan = dict(cuda=cuda, loads_nm=loads, seeds=SEEDS, cases=['N0', 'N1', 'D0'],
        source_sha256=files, base_progress_sha256=sha(BASE/'progress.json'),
        fixed='Same frozen final settings/scenario as N0/N1 sweep. D0=N0 objective plus compensation-shifted sample/sequence bounds; no soft input penalty.',
        backend='margin_capture', reuse='All N0/N1; existing five primary D0 pairs at133Nm, after raw hash and settings checks.',
        failure='Keep all complete outcomes including unsafe136/138Nm. No larger load, no controller tuning. Stop on nonfinite or integrity error.',
        dob_plot=dict(amplitude_nm=133., representative_seed=SEEDS[0],
            timing='d_true[k+1] is d applied on interval[k*dt,(k+1)*dt). d_hat_used[k] is causal pre-step estimate. d_hat[k+1] is updated at interval end.',
            metrics='10/50/90% crossing apparent lag; rise-only analytic-profile shift fit, not a pure time-delay identification.'),
        no_zip=True, no_gui_changes=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT/'plan.json'
    if target.exists():
        old = json.loads(target.read_text())
        for key in plan:
            if key != 'cuda':
                assert plan[key] == old[key], key
    else:
        write(target, plan)
        for name in files:
            dest = ROOT/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, dest)
    return plan


def run_one(amplitude, seed):
    folder = ROOT/'raw'/f'a{amplitude:g}'
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder/f'D0_{seed}.json'
    if meta.exists():
        row = json.loads(meta.read_text())
        assert sha(row['path']) == row['raw_sha256']
        return row
    prior = Path('kroc/results/final_mppi_optimization_20261117/margin_comparison_primary')/f'a-133_D0_margin_capture_{seed}.npz'
    if amplitude == 133 and prior.exists():
        old = json.loads(prior.with_suffix('.json').read_text())
        assert sha(prior) == old['raw_sha256']
        with np.load(prior) as archive:
            row = summarize(archive, amplitude, seed, 'D0', prior, old['mesh_5ms'], 'reused_verified_D0')
    else:
        print('RUN D0', amplitude, seed, flush=True)
        ref, q0, _ = geometry(.89)
        z = run_mppi(ref, initial_q=q0, has_disturbance=True,
            disturbance_function=raised_cosine_load(2, -amplitude),
            **options(seed, 'D0', 'margin_capture'))
        check(z, 'D0', True)
        z.update(circle_reference=ref)
        path = folder/f'D0_{seed}.npz'
        np.savez_compressed(path, **{k:v for k,v in z.items() if not k.startswith('_')})
        row = summarize(z, amplitude, seed, 'D0', path, dense_mesh_check((str(path), 4)), 'new_cuda_D0')
        del z
        gc.collect()
        torch.cuda.empty_cache()
    write(meta, row)
    print('RESULT D0', amplitude, seed, 'RMSE', row['metrics']['rmse_mm'],
          'Jclip', row['metrics']['jclip_nm_s'], 'safe', row['safe'], flush=True)
    return row


def aggregate(rows, loads):
    groups = []
    for amplitude in loads:
        runs = [r for r in rows if r['amplitude_nm'] == amplitude]
        if len(runs) != 30:
            continue
        group = dict(amplitude_nm=amplitude, n=10, cases={}, comparisons={})
        for case in ['N0', 'N1', 'D0']:
            selected = [r for r in runs if r['case'] == case]
            assert sorted(r['seed'] for r in selected) == sorted(SEEDS)
            group['cases'][case] = dict(safe_count=sum(r['safe'] for r in selected),
                mean={k:float(np.mean([r['metrics'][k] for r in selected]))
                      for k,v in selected[0]['metrics'].items() if isinstance(v,(int,float))})
        for a,b in [('N0','N1'), ('N0','D0'), ('N1','D0')]:
            pairs = []
            for seed in SEEDS:
                r0,r1=[next(r for r in runs if r['case']==c and r['seed']==seed) for c in [a,b]]
                m0,m1=r0['metrics'],r1['metrics']
                pairs.append(dict(seed=seed, both_safe=r0['safe'] and r1['safe'],
                    rmse_difference_mm=m1['rmse_mm']-m0['rmse_mm'],
                    rmse_change_pct=100*(m1['rmse_mm']/m0['rmse_mm']-1),
                    jclip_difference_nm_s=m1['jclip_nm_s']-m0['jclip_nm_s']))
            ma,mb=[group['cases'][c]['mean'] for c in [a,b]]
            group['comparisons'][a+'_to_'+b] = dict(pairs=pairs,
                all_pairs_safe=all(p['both_safe'] for p in pairs),
                mean_rmse_change_pct=100*(mb['rmse_mm']/ma['rmse_mm']-1),
                median_paired_change_pct=float(np.median([p['rmse_change_pct'] for p in pairs])),
                rmse_wins=sum(p['rmse_difference_mm']<0 for p in pairs),
                practical_1pct_wins=sum(p['rmse_change_pct']<=-1 for p in pairs))
        groups.append(group)
    return groups


def main():
    cuda = cuda_preflight()
    print('CUDA', cuda, flush=True)
    plan = freeze(cuda)
    rows = json.loads((BASE/'progress.json').read_text())
    for r in rows:
        assert sha(r['path']) == r['raw_sha256']
    for amplitude in plan['loads_nm']:
        for seed in SEEDS:
            rows.append(run_one(amplitude, seed))
            write(ROOT/'progress.json', rows)
        groups = aggregate(rows, plan['loads_nm'])
        write(ROOT/'summary.json', dict(groups=groups, completed=False))
        g=groups[-1]
        print('LEVEL', amplitude,
            {c:(g['cases'][c]['mean']['rmse_mm'],g['cases'][c]['safe_count']) for c in ['N0','N1','D0']},flush=True)
    for name,digest in plan['source_sha256'].items():
        assert sha(name) == digest, name
    assert len(rows)==210
    write(ROOT/'summary.json',dict(groups=aggregate(rows,plan['loads_nm']),completed=True,
        total_runs=210, new_D0_cuda_runs=sum(r['origin']=='new_cuda_D0' for r in rows),
        reused_D0_runs=sum(r['origin']=='reused_verified_D0' for r in rows),
        reused_N0_N1_runs=140, source_integrity_passed=True))
    print('COMPLETE',ROOT,flush=True)


if __name__=='__main__':
    main()
