"""Frozen N0/N1 paired load sweep. No retuning, D0, GUI changes, or ZIP."""
import gc
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from config import DT
from experiment_runner import run_mppi
from kroc.audit_circle_validation import dense_mesh_check
from kroc.final_input_headroom_checks import window_metrics
from kroc.optimize_final_mppi import check, options
from kroc.position_input_headroom_study import metrics, raised_cosine_load
from kroc.search_lowfreq_estimator_margin import geometry

ROOT = Path('kroc/results/final_n1_load_sweep_20261117')
SEEDS = [6807, 6817, 6827, 6907, 6917, 6927, 6937, 6947, 6957, 6967]
LOADS = [132.5, 133., 133.5, 134., 135., 136., 138., 140., 150., 170.]


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze(cuda):
    ROOT.mkdir(parents=True, exist_ok=True)
    chosen = json.loads(Path('kroc/results/final_mppi_optimization_20261117/selected_execution.json').read_text())
    files = dict(chosen['source_sha256'])
    for path, digest in files.items():
        assert sha(path) == digest, path
    for path in ['kroc/sweep_final_n1_load.py', 'kroc/optimize_final_mppi.py',
                 'controllers/mppi_dob_controller.py', 'kroc/input_headroom_study.py',
                 'kroc/position_input_headroom_study.py', 'config.py', 'experiment_runner.py']:
        files[path] = sha(path)
    plan = dict(seeds=SEEDS, loads_nm=LOADS, cases=['N0', 'N1'], backend='margin_capture',
                source_sha256=files, cuda=cuda,
                scenario='J3 negative raised cosine .5--2.5s, held to6s; same4cm/6s circle',
                fixed='Final N0/N1: lambda64, variance[9,9,9,.3136,.3136,.3136], K136,H10,dt.02,Q3,accelLPF6,N1w1, headroom0',
                stop_rule='After a complete level, stop larger loads if either case has >=5/10 unsafe runs. Stop immediately on nonfinite/integrity error.',
                selection='No tuning, seed filtering, or claiming independent validation. Same historical primary10 seeds at every load. All unsafe rows retained; unsafe mean is not evidence of valid tracking superiority.',
                analysis='Full/plateau RMSE,Jclip,clipping RMS/max/duty/duration,per-seed differences. Strict wins and >=1% practical gains separately. No universal clipping threshold inferred.',
                reuse='At133Nm reuse existing five primary margin_capture pairs only if raw hash, backend hash, and settings audits pass.',
                no_zip=True, no_gui_changes=True)
    path = ROOT/'plan.json'
    if path.exists():
        old = json.loads(path.read_text())
        for key in plan:
            if key != 'cuda':
                assert old[key] == plan[key], key
    else:
        write(path, plan)
        for name in files:
            dest = ROOT/'source'/name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(name, dest)


def summarize(z, amplitude, seed, case, path, mesh, origin):
    check(z, case, True)
    n = len(z['u_mppi'])
    assert n == 300
    t = np.arange(n)*DT
    d = -amplitude*.5*(1-np.cos(np.pi*np.clip((t-.5)/2, 0, 1)))
    np.testing.assert_allclose(z['d_true'][1:, 2], d, atol=2e-5, rtol=1e-6)
    loss = z['u_mppi']+z['d_hat_used']-z['u_applied']
    m = metrics(z)
    m.update(clip_vector_rms_nm=float(np.sqrt(np.mean(np.sum(loss**2, axis=1)))),
             clip_j3_rms_nm=float(np.sqrt(np.mean(loss[:, 2]**2))),
             plateau_clip_vector_rms_nm=float(np.sqrt(np.mean(np.sum(loss[t>=2.5]**2, axis=1)))))
    safe = (mesh['first_collision_index'] is None and m['position_violation_steps'] == 0
            and m['velocity_violation_steps'] == 0)
    return dict(amplitude_nm=amplitude, seed=seed, case=case, path=str(path),
                raw_sha256=sha(path), origin=origin, metrics=m, safe=safe, mesh_5ms=mesh,
                windows={name: window_metrics(z, a, b) for name, a, b in
                         [('rise', .5, 2.5), ('plateau', 2.5, 6.)]})


def run_one(amplitude, seed, case):
    folder = ROOT/'raw'/f'a{amplitude:g}'
    folder.mkdir(parents=True, exist_ok=True)
    meta = folder/f'{case}_{seed}.json'
    if meta.exists():
        row = json.loads(meta.read_text())
        assert sha(row['path']) == row['raw_sha256']
        return row
    prior = Path('kroc/results/final_mppi_optimization_20261117/margin_comparison_primary')/f'a-133_{case}_margin_capture_{seed}.npz'
    if amplitude == 133 and prior.exists():
        old = json.loads(prior.with_suffix('.json').read_text())
        assert sha(prior) == old['raw_sha256']
        with np.load(prior) as archive:
            row = summarize(archive, amplitude, seed, case, prior, old['mesh_5ms'], 'reused_verified_margin_capture')
    else:
        print('RUN', amplitude, seed, case, flush=True)
        ref, q0, _ = geometry(.89)
        opts = options(seed, case, 'margin_capture')
        z = run_mppi(ref, initial_q=q0, has_disturbance=True,
                     disturbance_function=raised_cosine_load(2, -amplitude), **opts)
        z.update(circle_reference=ref)
        check(z, case, True)
        path = folder/f'{case}_{seed}.npz'
        np.savez_compressed(path, **{k: v for k, v in z.items() if not k.startswith('_')})
        mesh = dense_mesh_check((str(path), 4))
        row = summarize(z, amplitude, seed, case, path, mesh, 'new_cuda')
        del z
        gc.collect()
        torch.cuda.empty_cache()
    write(meta, row)
    print('RESULT', amplitude, seed, case, 'RMSE', row['metrics']['rmse_mm'],
          'Jclip', row['metrics']['jclip_nm_s'], 'safe', row['safe'], flush=True)
    return row


def aggregate(rows):
    groups = []
    for a in LOADS:
        level = [r for r in rows if r['amplitude_nm'] == a]
        if len(level) != 2*len(SEEDS):
            continue
        cases = {c: sorted([r for r in level if r['case'] == c], key=lambda r: r['seed']) for c in ['N0', 'N1']}
        group = dict(amplitude_nm=a, n=len(SEEDS), all_safe=all(r['safe'] for r in level), cases={})
        for c, runs in cases.items():
            group['cases'][c] = dict(safe_count=sum(r['safe'] for r in runs),
                mean={k: float(np.mean([r['metrics'][k] for r in runs])) for k in runs[0]['metrics']
                      if isinstance(runs[0]['metrics'][k], (int, float))},
                median_rmse_mm=float(np.median([r['metrics']['rmse_mm'] for r in runs])))
        pairs = []
        for n0, n1 in zip(cases['N0'], cases['N1']):
            assert n0['seed'] == n1['seed']
            m0, m1 = n0['metrics'], n1['metrics']
            pairs.append(dict(seed=n0['seed'], both_safe=n0['safe'] and n1['safe'],
                n0_rmse_mm=m0['rmse_mm'], n1_rmse_mm=m1['rmse_mm'],
                n0_jclip_nm_s=m0['jclip_nm_s'], n1_jclip_nm_s=m1['jclip_nm_s'],
                n0_clip_rms_nm=m0['clip_vector_rms_nm'],
                rmse_change_pct=100*(m1['rmse_mm']/m0['rmse_mm']-1),
                jclip_change_pct=100*(m1['jclip_nm_s']/max(m0['jclip_nm_s'], 1e-12)-1)))
        group['pairs'] = pairs
        m0, m1 = [group['cases'][c]['mean'] for c in ['N0', 'N1']]
        group.update(rmse_change_pct=100*(m1['rmse_mm']/m0['rmse_mm']-1),
            jclip_change_pct=100*(m1['jclip_nm_s']/max(m0['jclip_nm_s'], 1e-12)-1),
            rmse_wins=sum(p['rmse_change_pct'] < 0 for p in pairs),
            practical_1pct_rmse_wins=sum(p['rmse_change_pct'] <= -1 for p in pairs),
            simultaneous_wins=sum(p['rmse_change_pct'] < 0 and p['jclip_change_pct'] < 0 for p in pairs),
            median_paired_rmse_change_pct=float(np.median([p['rmse_change_pct'] for p in pairs])))
        groups.append(group)
    return groups


def main():
    cuda = cuda_preflight()
    print('CUDA', cuda, flush=True)
    freeze(cuda)
    rows = []
    stop = None
    for amplitude in LOADS:
        for seed in SEEDS:
            for case in ['N0', 'N1']:
                rows.append(run_one(amplitude, seed, case))
                write(ROOT/'progress.json', rows)
        groups = aggregate(rows)
        write(ROOT/'summary.json', dict(groups=groups, stop_reason=None))
        g = groups[-1]
        print('LEVEL', amplitude, 'RMSE%', g['rmse_change_pct'], 'Jclip%', g['jclip_change_pct'],
              'wins', g['rmse_wins'], 'safe', [g['cases'][c]['safe_count'] for c in ['N0', 'N1']], flush=True)
        if any(g['cases'][c]['safe_count'] <= len(SEEDS)//2 for c in ['N0', 'N1']):
            stop = f'Safety escalation stop at{amplitude:g}Nm: >=5/10 unsafe in at least one case.'
            break
    for path, digest in json.loads((ROOT/'plan.json').read_text())['source_sha256'].items():
        assert sha(path) == digest, path
    write(ROOT/'summary.json', dict(groups=aggregate(rows), stop_reason=stop,
        completed=True, total_runs=len(rows), new_cuda_runs=sum(r['origin']=='new_cuda' for r in rows),
        reused_runs=sum(r['origin']!='new_cuda' for r in rows), source_integrity_passed=True))
    print('COMPLETE', ROOT, stop, flush=True)


if __name__ == '__main__':
    main()
