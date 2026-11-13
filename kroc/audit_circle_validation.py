"""Audit/plot paired circular experiments; JSON and figures only, no MD report.

Mesh checks use interpolated joint samples, not continuous collision detection.
They run offline and must never be included in the controller timing claim.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import DT, Q_MAX, Q_MIN, TAU_MAX, TIMING_WARMUP_STEPS

NAMES = ['static25', 'directional25_5', 'static5']
LABELS = ['Static 2.5%', 'Directional 2.5-5%', 'Static 5%']
COLORS = ['tab:orange', 'tab:blue', 'tab:green']
_WORKER_GEOMETRY = None


def dense_mesh_check(task):
    """Independent offline worker; never creates a CUDA context."""
    global _WORKER_GEOMETRY
    if _WORKER_GEOMETRY is None:
        from kroc.collision_geometry import Geometry
        _WORKER_GEOMETRY = Geometry()
    path, subdivisions = task
    with np.load(path) as z:
        q=z['q'].copy()
    alpha=np.arange(subdivisions)/subdivisions
    dense=(q[:-1,None]*(1-alpha[None,:,None])+q[1:,None]*alpha[None,:,None]).reshape(-1,6)
    dense=np.concatenate([dense,q[-1:]])
    return _WORKER_GEOMETRY.trajectory(dense)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--scenario', type=int, default=0)
    p.add_argument('--additional', action='append', default=[], metavar='DIRECTORY:SCENARIO')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mesh-dt', type=float, default=0.)
    p.add_argument('--mesh-workers',type=int,default=1)
    p.add_argument('--require-common-start', action='store_true')
    p.add_argument('--cases', nargs='+', default=NAMES)
    p.add_argument('--common-fraction', type=float, default=.05)
    a = p.parse_args()
    if a.mesh_workers<1:p.error('--mesh-workers must be positive')
    styles = {'static25':('Static 2.5%','tab:orange'), 'directional25_5':('Directional 2.5-5%','tab:blue'),
              'directional25_75':('Directional 2.5-7.5%','tab:blue'), 'static35':('Static 3.5%','tab:brown'),
              'static5':('Static 5%','tab:green'), 'static75':('Static 7.5%','tab:purple'), 'static10':('Static 10%','tab:red')}
    names=a.cases; labels=[styles[n][0] for n in names]; colors=[styles[n][1] for n in names]
    sources = [(a.directory, a.scenario)] + [
        (Path(x.rsplit(':', 1)[0]), int(x.rsplit(':', 1)[1])) for x in a.additional]
    a.output.mkdir(parents=True, exist_ok=True)
    geom = None
    if a.mesh_dt:
        from kroc.collision_geometry import Geometry
        geom = Geometry()
        subdivisions = round(DT / a.mesh_dt)
        assert subdivisions >= 1 and np.isclose(subdivisions * a.mesh_dt, DT)
    records, series, seen = [], {}, set()
    ref0 = d0 = condition0 = manifest0 = None
    span = np.asarray(Q_MAX) - np.asarray(Q_MIN)
    for directory, scenario in sources:
        manifest = json.loads((directory / 'manifest.json').read_text())
        assert not manifest['headroom_enabled']
        if manifest0 is None:
            manifest0 = manifest
        else:
            for field in ['cases', 'sensor_noise', 'dob', 'collision_cost', 'reference']:
                assert manifest[field] == manifest0[field], field
        j = manifest['arguments']['joint']
        limit = np.asarray(TAU_MAX) * manifest['arguments']['scale']
        for row in json.loads((directory / 'metrics.json').read_text()):
            if row['scenario'] != scenario:
                continue
            key = (row['seed'], row['case'])
            assert key not in seen
            seen.add(key)
            condition = [row[k] for k in ['fraction', 'radius', 'pose', 'amplitude', 'onset', 'ramp', 'sign']]
            if condition0 is None:
                condition0 = condition
            assert condition == condition0
            path = directory / f's{scenario}_seed{row["seed"]}_{row["case"]}.npz'
            with np.load(path) as z:
                assert str(z['compute_device']) == 'cuda'
                assert str(z['dob_compensation_mode']) == 'output'
                q = z['q'].copy()
                assert all(np.isfinite(z[k]).all() for k in ['q', 'qd', 'u_mppi', 'u_applied', 'd_hat_used', 'err_mm'])
                req = z['u_mppi'] + z['d_hat_used']
                clamp_error = float(np.max(np.abs(z['u_applied'] - np.clip(req, -limit, limit))))
                assert clamp_error < 1e-4
                if ref0 is None:
                    ref0, d0 = z['circle_reference'].copy(), z['d_true'].copy()
                assert np.array_equal(z['circle_reference'], ref0)
                assert np.array_equal(z['d_true'], d0)
                common = np.minimum(q[0] - (np.asarray(Q_MIN) + a.common_fraction * span),
                                    np.asarray(Q_MAX) - a.common_fraction * span - q[0])
                if a.require_common_start:
                    assert common.min() > 0
                assert np.max(np.abs(z['qd'][0])) < 1e-8
                ms = z['solve_t_raw'][TIMING_WARMUP_STEPS:].copy() * 1000
                distance = np.minimum(q - np.asarray(Q_MIN), np.asarray(Q_MAX) - q)
                sat = np.isclose(np.abs(z['u_applied']), limit, atol=1e-5, rtol=0)
                metrics = dict(row['metrics'])
                metrics.update(initial_common_box_min_slack_rad=float(common.min()),
                               clamp_audit_max_nm=clamp_error,
                               timing_samples=len(ms), compute_mean_ms=float(ms.mean()),
                               compute_p95_ms=float(np.percentile(ms, 95)),
                               compute_max_ms=float(ms.max()),
                               deadline_miss_rate=float(np.mean(ms > DT * 1000)))
                if geom is not None and a.mesh_workers==1:
                    alpha = np.arange(subdivisions) / subdivisions
                    dense = (q[:-1, None] * (1-alpha[None, :, None]) +
                             q[1:, None] * alpha[None, :, None]).reshape(-1, 6)
                    dense = np.concatenate([dense, q[-1:]])
                    metrics['mesh_interpolated'] = geom.trajectory(dense)
                records.append({'seed': key[0], 'case': key[1], 'source': str(path), 'metrics': metrics})
                series[key] = {'ms': ms, 'sat': sat[:, j], 'distance': distance.min(1),
                               'requested': req[:, j], 'applied': z['u_applied'][:, j].copy()}
                print(key, 'violation=', metrics['position_violation_max_rad'],
                      'RMSE=', metrics['tracking_rmse_mm'], flush=True)
    if geom is not None and a.mesh_workers>1:
        tasks=[(r['source'],subdivisions) for r in records]
        with ProcessPoolExecutor(max_workers=a.mesh_workers,
                mp_context=multiprocessing.get_context('spawn')) as pool:
            for r,check in zip(records,pool.map(dense_mesh_check,tasks)):
                r['metrics']['mesh_interpolated']=check
                print('DENSE_MESH',r['seed'],r['case'],check,flush=True)
    seeds = sorted({r['seed'] for r in records})
    assert seen == {(s, c) for s in seeds for c in names}, 'Missing paired cases'
    aggregate = {}
    table = []
    for case, label in zip(names, labels):
        rr = [r['metrics'] for r in records if r['case'] == case]
        ms = np.concatenate([series[(s, case)]['ms'] for s in seeds])
        agg = {'position_violation_seeds': sum(m['position_violation_steps'] > 0 for m in rr),
               'seed_count': len(seeds),
               'worst_position_violation_rad': max(m['position_violation_max_rad'] for m in rr),
               'tracking_rmse_mean_mm': float(np.mean([m['tracking_rmse_mm'] for m in rr])),
               'tracking_rmse_std_mm': float(np.std([m['tracking_rmse_mm'] for m in rr], ddof=1)) if len(rr)>1 else 0.,
               'saturation_duration_mean_s': float(np.mean([m['test_joint_saturation_steps'] * DT for m in rr])),
               'compute_mean_ms': float(ms.mean()), 'compute_p95_ms': float(np.percentile(ms, 95)),
               'compute_max_ms': float(ms.max()), 'within_20ms_percent': float(np.mean(ms <= 20) * 100)}
        if geom is not None:
            agg['mesh_collision_trajectories'] = sum(
                bool(m['mesh_interpolated']['self_collision_states'] or m['mesh_interpolated']['floor_collision_states']) for m in rr)
        aggregate[case] = agg
        table.append([label, f'{agg["position_violation_seeds"]}/{len(seeds)}',
                      f'{agg["worst_position_violation_rad"]:.4f}',
                      f'{agg["tracking_rmse_mean_mm"]:.2f} +/- {agg["tracking_rmse_std_mm"]:.2f}',
                      f'{agg["saturation_duration_mean_s"]:.2f}', f'{agg["compute_mean_ms"]:.2f}',
                      f'{agg["compute_p95_ms"]:.2f}', f'{agg["within_20ms_percent"]:.2f}'])
    result = {'sources': [str(d) + ':' + str(s) for d, s in sources], 'seeds': seeds,
              'headroom_enabled': False, 'common_start_required': a.require_common_start,
              'common_start_fraction': a.common_fraction,
              'mesh_sample_dt_s': a.mesh_dt or None,
              'mesh_workers':a.mesh_workers,
              'mesh_interpolation_is_not_continuous_certificate': True,
              'timing_excludes_offline_mesh_check': True,
              'timing_warmup_excluded_steps': TIMING_WARMUP_STEPS,
              'aggregate': aggregate, 'per_seed': records}
    paired = {}
    dynamic_case=next(n for n in names if n.startswith('directional'))
    upper_case='static75' if dynamic_case=='directional25_75' else 'static5'
    for seed in seeds:
        cases = {r['case']: r['metrics'] for r in records if r['seed'] == seed}
        paired[seed] = {
            'low_static_violated': cases['static25']['position_violation_steps'] > 0,
            'dynamic_position_safe': cases[dynamic_case]['position_violation_steps'] == 0,
            'high_static_position_safe': cases[upper_case]['position_violation_steps'] == 0,
            'dynamic_rmse_lower_than_high_static':
                cases[dynamic_case]['tracking_rmse_mm'] < cases[upper_case]['tracking_rmse_mm'],
            'dynamic_minus_high_static_rmse_mm':
                cases[dynamic_case]['tracking_rmse_mm'] - cases[upper_case]['tracking_rmse_mm'],
        }
    result['paired_seed_comparison'] = paired
    contains_collision=any(r['metrics'].get('first_collision_index') is not None
        or r['metrics'].get('mesh_interpolated',{}).get('first_collision_index') is not None for r in records)
    result['contains_colliding_runs']=contains_collision
    result['post_contact_metrics_are_validated_physics']=False
    (a.output / 'audited_results.json').write_text(json.dumps(result, indent=2))
    fig, ax = plt.subplots(figsize=(16, 3.2)); ax.axis('off')
    if contains_collision:
        fig.suptitle('DIAGNOSTIC: colliding runs included; full-episode RMSE contains unmodeled post-contact motion',color='crimson',fontsize=11)
    tab = ax.table(cellText=table, colLabels=['Case', 'Violation seeds', 'Max violation\n[rad]',
        'Tracking RMSE\nmean +/- SD [mm]', f'J{j+1} saturated\nmean [s]',
        'Solve mean\n[ms]', 'Solve p95\n[ms]', '<=20ms\n[%]'], loc='center', cellLoc='center',
        colWidths=[.16, .11, .11, .19, .12, .1, .1, .11])
    tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2.4)
    ax.set_title(f'Circular Post-DOB | seeds {seeds} | no headroom\n'
                 'Saturation: actual actuator clamp; timing: warm controller, offline mesh excluded', pad=15)
    fig.savefig(a.output / 'results_table.png', dpi=160, bbox_inches='tight'); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.5))
    first = seeds[0]
    for case, label, color in zip(names, labels, colors):
        data = [series[(s, case)] for s in seeds]
        ms = np.stack([d['ms'] for d in data]); tms = (np.arange(ms.shape[1])+TIMING_WARMUP_STEPS)*DT
        axes[0, 0].plot(tms, ms.mean(0), color=color, label=label)
        axes[0, 0].fill_between(tms, ms.min(0), ms.max(0), color=color, alpha=.13)
        sat = np.stack([d['sat'] for d in data]); t = np.arange(sat.shape[1])*DT
        axes[0, 1].plot(t, sat.mean(0), color=color, label=label)
        d = series[(first, case)]
        axes[1, 0].plot(t, d['requested'], color=color, label=label)
        axes[1, 1].plot(t, d['applied'], color=color, label=label)
    axes[0, 0].axhline(20, color='red', ls='--', label='20 ms deadline')
    axes[0, 0].set(ylabel='MPPI solve [ms]', title='Warm solve: seed mean and range')
    axes[0, 1].set(ylabel='Saturated seed fraction', title=f'Actual J{j+1} saturation (not eliminated by margin)')
    for ax in axes[1]:
        ax.axhline(limit[j], color='red', ls='--'); ax.axhline(-limit[j], color='red', ls='--')
        ax.set_ylabel('Torque [Nm]')
    axes[1, 0].set_title(f'Requested u_MPPI + d_hat | seed {first}')
    axes[1, 1].set_title(f'Applied clamp(u_MPPI + d_hat) | seed {first}')
    for ax in axes.ravel():
        ax.set_xlabel('Time [s]'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(a.output / 'runtime_actuator.png', dpi=160); plt.close(fig)
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
