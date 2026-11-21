"""GPU-only execution optimization, followed by a fixed dynamic-bound comparison.

Keep the final N0/N1 controller equations/configuration frozen. No tuning or ZIP.
All outcomes are retained; calibration must pass before the comparator runs.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from controllers.fast_final_mppi import FastFinalTracker
from controllers.final_n0_controller import FinalN0Tracker
from experiment_runner import run_mppi
from kroc.audit_circle_validation import dense_mesh_check
from kroc.check_final_controller_speed import timing
from kroc.improve_n1_clipping_cost import ClippingLossN1, COVARIANCE, TEMPERATURE
from kroc.position_input_headroom_study import metrics, raised_cosine_load
from kroc.run_input_headroom_study import run_options
from kroc.run_pre_input_baseline import load_preset
from kroc.search_lowfreq_estimator_margin import geometry


ROOT = Path('kroc/results/final_mppi_optimization_20261117')
CALIBRATION_SEEDS = [6807, 6937]
PRIMARY_SEEDS = [6907, 6917, 6927, 6937, 6947]
ADJACENT_SEEDS = [7007, 7017, 7027, 7037, 7047]
VARIANTS = ['reference', 'auxiliary', 'fused', 'deferred']
CONTROL_KEYS = ['q', 'qd', 'u_mppi', 'u_applied', 'd_hat_used', 'err_mm']


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False))


def options(seed, case, variant):
    opts = run_options(load_preset(), seed, case if case != 'D0' else 'N0', 0.)
    opts.update(temperature=TEMPERATURE, noise_variance=COVARIANCE)
    opts['tracker_options'].update(headroom_fraction=.95, unclamped_diagnostic=False,
        dob_cutoff_hz=3., dob_acceleration_lpf_cutoff_hz=6.)
    if variant == 'reference':
        opts['tracker_class'] = ClippingLossN1 if case == 'N1' else FinalN0Tracker
        if case == 'N1':
            opts['tracker_options']['clipping_weight'] = 1.
    else:
        opts['tracker_class'] = FastFinalTracker
        opts['tracker_options'].update(fuse_costs=variant != 'auxiliary',
            compile_auxiliary=True, deferred_diagnostics=variant in {'deferred', 'loop', 'graph', 'manual', 'full_capture', 'margin_capture'})
        if variant in {'manual', 'full_capture', 'margin_capture'}:
            from controllers.captured_final_mppi import CapturedFinalTracker
            opts['tracker_class'] = CapturedFinalTracker
            if variant == 'full_capture':
                from controllers.fully_captured_final_mppi import FullyCapturedFinalTracker
                opts['tracker_class'] = FullyCapturedFinalTracker
            if variant == 'margin_capture':
                from controllers.fixed_margin_captured_mppi import FixedMarginCapturedTracker
                opts['tracker_class'] = FixedMarginCapturedTracker
        if variant in {'loop', 'graph'}:
            from controllers.graph_final_mppi import GraphFinalTracker
            opts['tracker_class'] = GraphFinalTracker
            opts['tracker_options']['cuda_graphs'] = variant == 'graph'
        if case == 'D0':
            from controllers.dynamic_bound_mppi import DynamicBoundTracker, DynamicBoundGraphTracker
            opts['tracker_class'] = DynamicBoundGraphTracker if variant in {'loop', 'graph'} else DynamicBoundTracker
            if variant == 'manual':
                from controllers.captured_dynamic_bound_mppi import DynamicBoundCapturedTracker
                opts['tracker_class'] = DynamicBoundCapturedTracker
            if variant == 'full_capture':
                from controllers.fully_captured_dynamic_bound_mppi import DynamicBoundFullyCapturedTracker
                opts['tracker_class'] = DynamicBoundFullyCapturedTracker
            if variant == 'margin_capture':
                from controllers.fixed_margin_dynamic_bound_mppi import DynamicBoundFixedMarginCapturedTracker
                opts['tracker_class'] = DynamicBoundFixedMarginCapturedTracker
    return opts


def check(z, case, use_dob):
    from config import TAU_MAX
    assert str(z['compute_device']) == 'cuda'
    assert float(z['mppi_temperature']) == 64.
    assert float(z['dob_cutoff_hz']) == 3.
    assert float(z['dob_acceleration_lpf_cutoff_hz']) == 6.
    assert float(z['input_study_headroom_weight']) == 0.
    assert not bool(z['input_study_saturation_rollout'])
    assert not bool(z['nonphysical_unclamped_diagnostic'])
    for key in CONTROL_KEYS + ['state_prediction_error']:
        assert np.isfinite(z[key]).all(), key
    comp = z['d_hat_used'] if use_dob else 0.
    np.testing.assert_array_equal(z['u_effective_pred'], z['u_mppi'])
    np.testing.assert_allclose(z['u_applied'], np.clip(z['u_mppi']+comp, -np.array(TAU_MAX), TAU_MAX), atol=3e-6, rtol=1e-6)
    np.testing.assert_array_equal(z['mppi_active_noise_variance'], np.broadcast_to(np.asarray(COVARIANCE, np.float32), (300, 6)))
    if case == 'N1':
        assert float(z['n1_clipping_weight']) == 1.
    if case == 'D0':
        assert bool(z['dynamic_input_bounds'])
        assert np.max(np.abs(z['u_applied']-z['u_mppi']-comp)) <= 3.1e-5
        bounds = z['dynamic_nominal_input_bounds']
        np.testing.assert_allclose(bounds[:, 0], -np.asarray(TAU_MAX)-comp, atol=1e-6)
        np.testing.assert_allclose(bounds[:, 1], np.asarray(TAU_MAX)-comp, atol=1e-6)
        assert np.all(z['u_mppi']>=bounds[:, 0]-3e-5)
        assert np.all(z['u_mppi']<=bounds[:, 1]+3e-5)


def run_one(phase, seed, case, variant, amplitude=-133., use_dob=True):
    directory = ROOT / phase
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'a{amplitude:g}_{case}_{variant}_{seed}.npz'
    impl = Path('controllers/fast_final_mppi.py')
    source = impl.read_bytes()
    if variant in {'loop', 'graph'}:
        source += Path('controllers/graph_final_mppi.py').read_bytes()
    if case == 'D0':
        source += Path('controllers/dynamic_bound_mppi.py').read_bytes()
    if variant in {'manual', 'full_capture', 'margin_capture'}:
        source += Path('controllers/captured_final_mppi.py').read_bytes()
        if case == 'D0':
            source += Path('controllers/captured_dynamic_bound_mppi.py').read_bytes()
    if variant == 'full_capture':
        source += Path('controllers/fully_captured_final_mppi.py').read_bytes()
        if case == 'D0':
            source += Path('controllers/fully_captured_dynamic_bound_mppi.py').read_bytes()
    if variant == 'margin_capture':
        source += Path('controllers/fixed_margin_captured_mppi.py').read_bytes()
        if case == 'D0':
            source += Path('controllers/fixed_margin_dynamic_bound_mppi.py').read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    meta_path = path.with_suffix('.json')
    if path.exists():
        row = json.loads(meta_path.read_text())
        assert row['fast_source_sha256'] == digest, 'Stale optimized controller cache'
        with np.load(path) as loaded:
            z = dict(loaded)
        check(z, case, use_dob)
        return row, z
    print('RUN', phase, seed, case, variant, amplitude, use_dob, flush=True)
    ref, q0, _ = geometry(.89)
    opts = options(seed, case, variant)
    opts['use_dob'] = use_dob
    z = run_mppi(ref, initial_q=q0, has_disturbance=bool(amplitude),
                 disturbance_function=raised_cosine_load(2, amplitude), **opts)
    check(z, case, use_dob)
    z.update(circle_reference=ref)
    np.savez_compressed(path, **{k: v for k, v in z.items() if not k.startswith('_')})
    mesh = dense_mesh_check((str(path), 4))
    m = metrics(z)
    row = dict(phase=phase, seed=seed, case=case, variant=variant, amplitude=amplitude,
               use_dob=use_dob, path=str(path), metrics=m,
               solve=timing(z['solve_t_raw']), cycle=timing(z['control_cycle_t_raw']),
               diagnostic_export_ms=float(z.get('diagnostic_export_time_s', 0.))*1000,
               mesh_5ms=mesh, fast_source_sha256=digest,
               raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    row['safe'] = (mesh['first_collision_index'] is None and m['position_violation_steps'] == 0
                   and m['velocity_violation_steps'] == 0)
    write(meta_path, row)
    print('RESULT', seed, case, variant, 'RMSE', m['rmse_mm'], 'Jclip', m['jclip_nm_s'],
          'solve', row['solve']['mean_ms'], 'cycle', row['cycle']['mean_ms'], 'safe', row['safe'], flush=True)
    gc.collect()
    torch.cuda.empty_cache()
    return row, z


def compare_arrays(a, b):
    keys = [k for k in a if k in b and isinstance(a[k], np.ndarray) and isinstance(b[k], np.ndarray)
            and a[k].shape == b[k].shape and a[k].dtype.kind in 'biufc'
            and not any(word in k for word in ('time', 'solve_t', 'cycle_t', 'elapsed'))]
    return dict(bitwise_control=all(np.array_equal(a[k], b[k]) for k in CONTROL_KEYS),
                max_control_differences={k: float(np.max(np.abs(a[k]-b[k]))) for k in CONTROL_KEYS},
                differing_numeric_keys=[k for k in keys if not np.array_equal(a[k], b[k], equal_nan=True)])


def calibrate():
    rows = []
    comparisons = []
    for seed in CALIBRATION_SEEDS:
        for case in ['N0', 'N1']:
            reference = None
            fused = None
            for variant in VARIANTS:
                row, z = run_one('calibration', seed, case, variant)
                rows.append(row)
                if variant == 'reference':
                    reference = z
                else:
                    comparison = compare_arrays(reference, z)
                    comparison.update(seed=seed, case=case, variant=variant)
                    comparison['rmse_relative_change'] = row['metrics']['rmse_mm']/metrics(reference)['rmse_mm']-1
                    comparison['jclip_relative_change'] = row['metrics']['jclip_nm_s']/metrics(reference)['jclip_nm_s']-1
                    if variant == 'deferred':
                        comparison['versus_fused'] = compare_arrays(fused, z)
                    comparisons.append(comparison)
                if variant == 'fused':
                    fused = z
                write(ROOT/'calibration_progress.json', dict(rows=rows, comparisons=comparisons))
    # Additional equivalence check without an external disturbance or observer.
    for variant in ['reference', 'deferred']:
        row, z = run_one('nominal_calibration', 7107, 'N0', variant, amplitude=0., use_dob=False)
        rows.append(row)
        if variant == 'reference':
            reference = z
        else:
            comparison = compare_arrays(reference, z)
            comparison.update(seed=7107, case='N0_DOB_OFF', variant=variant,
                rmse_relative_change=row['metrics']['rmse_mm']/metrics(reference)['rmse_mm']-1)
            comparisons.append(comparison)
    passed = (all(r['safe'] for r in rows)
              and all(abs(c['rmse_relative_change']) <= .01 for c in comparisons)
              and all(abs(c.get('jclip_relative_change', 0.)) <= .05 for c in comparisons)
              and all(c['versus_fused']['bitwise_control'] for c in comparisons if 'versus_fused' in c))
    result = dict(passed=passed, rows=rows, comparisons=comparisons,
        acceptance='All runs safe; per-run RMSE within 1%, Jclip within 5%; deferred diagnostics bitwise control-equivalent to fused immediate diagnostics. No retuning.')
    write(ROOT/'calibration.json', result)
    if not passed:
        raise RuntimeError('Optimization equivalence gate failed; comparator must not run')
    print('CALIBRATION PASSED', flush=True)


def comparison():
    gate = json.loads((ROOT/'calibration.json').read_text())
    assert gate['passed'], 'Optimization calibration must pass first'
    variant = 'deferred'
    graph_path = ROOT/'graph_check.json'
    if graph_path.exists():
        graph = json.loads(graph_path.read_text())
        if graph.get('passed'):
            baseline = np.mean([r['cycle']['mean_ms'] for r in gate['rows']
                                if r['variant']=='deferred' and r['phase']=='calibration'])
            candidates = {v: np.mean([r['cycle']['mean_ms'] for r in graph['rows'] if r['variant']==v])
                          for v in {r['variant'] for r in graph['rows']}}
            best = min(candidates, key=candidates.get)
            if candidates[best] < .98*baseline:
                variant = best
    selection = dict(variant=variant, selection='Runtime only after equivalence gate; use a horizon candidate only if mean cycle is at least 2% faster than deferred.',
        dynamic_source_sha256=hashlib.sha256(Path('controllers/dynamic_bound_mppi.py').read_bytes()).hexdigest(),
        graph_source_sha256=hashlib.sha256(Path('controllers/graph_final_mppi.py').read_bytes()).hexdigest())
    selection_path = ROOT/'execution_choice.json'
    if selection_path.exists():
        assert json.loads(selection_path.read_text()) == selection
    else:
        write(selection_path, selection)
    rows = []
    for label, amplitude, seeds in [('primary', -133., PRIMARY_SEEDS), ('adjacent', -132.5, ADJACENT_SEEDS)]:
        for seed in seeds:
            for case in ['N0', 'N1', 'D0']:
                row, _ = run_one('comparison_'+label, seed, case, variant, amplitude)
                rows.append(row)
                write(ROOT/'comparison.json', dict(rows=rows))


def graph_check(capture=False):
    gate = json.loads((ROOT/'calibration.json').read_text())
    assert gate['passed']
    rows = []
    comparisons = []
    for seed in CALIBRATION_SEEDS:
        for case in ['N0', 'N1']:
            _, reference = run_one('calibration', seed, case, 'reference')
            for variant in (['loop', 'graph'] if capture else ['loop']):
                row, z = run_one('graph_check', seed, case, variant)
                c = compare_arrays(reference, z)
                c.update(seed=seed, case=case, variant=variant,
                    rmse_relative_change=row['metrics']['rmse_mm']/metrics(reference)['rmse_mm']-1,
                    jclip_relative_change=row['metrics']['jclip_nm_s']/metrics(reference)['jclip_nm_s']-1)
                rows.append(row)
                comparisons.append(c)
                write(ROOT/'graph_check.json', dict(rows=rows, comparisons=comparisons))
                if not row['safe'] or abs(c['rmse_relative_change'])>.01 or abs(c['jclip_relative_change'])>.05:
                    raise RuntimeError('Compiled-horizon equivalence gate failed')
    write(ROOT/'graph_check.json', dict(passed=True, rows=rows, comparisons=comparisons))


def explicit_capture_check(variant='manual'):
    assert json.loads((ROOT/'calibration.json').read_text())['passed']
    rows = []
    comparisons = []
    phase = {'manual': 'explicit_capture', 'full_capture': 'full_capture', 'margin_capture': 'margin_capture'}[variant]
    target = ROOT/(phase+'.json')
    try:
        for seed in CALIBRATION_SEEDS:
            for case in ['N0', 'N1', 'D0']:
                reference_phase = 'calibration' if case != 'D0' else 'manual_reference'
                _, reference = run_one(reference_phase, seed, case, 'deferred')
                row, z = run_one(phase, seed, case, variant)
                c = compare_arrays(reference, z)
                c.update(seed=seed, case=case)
                if variant=='margin_capture':
                    for key in [k for k in reference if k.startswith('kroc_')]:
                        np.testing.assert_allclose(z[key], reference[key], rtol=5e-5, atol=5e-5)
                rows.append(row)
                comparisons.append(c)
                write(target, dict(rows=rows, comparisons=comparisons))
                if not row['safe'] or not c['bitwise_control']:
                    raise RuntimeError('Explicit graph replay is not bitwise control-equivalent')
        _, reference = run_one('nominal_calibration', 7107, 'N0', 'deferred', 0., False)
        row, z = run_one(phase+'_nominal', 7107, 'N0', variant, 0., False)
        c = compare_arrays(reference, z)
        c.update(seed=7107, case='N0_DOB_OFF')
        rows.append(row)
        comparisons.append(c)
        if not row['safe'] or not c['bitwise_control']:
            raise RuntimeError('Explicit graph nominal equivalence failed')
    except Exception as error:
        write(target, dict(passed=False, rows=rows, comparisons=comparisons,
            failure=f'{type(error).__name__}: {error}'))
        raise
    write(target, dict(passed=True, rows=rows, comparisons=comparisons,
        acceptance='All seven N0/N1/D0/nominal runs safe and bitwise control-equivalent to deferred. Sampling outside graph; no retuning.'))


def explicit_comparison(variant='manual'):
    gate_name = {'manual': 'explicit_capture', 'full_capture': 'full_capture', 'margin_capture': 'margin_capture'}[variant]
    gate = json.loads((ROOT/(gate_name+'.json')).read_text())
    assert gate['passed']
    rows = []
    prefix = {'manual': 'explicit_comparison', 'full_capture': 'full_comparison', 'margin_capture': 'margin_comparison'}[variant]
    for label, amplitude, seeds in [('primary', -133., PRIMARY_SEEDS), ('adjacent', -132.5, ADJACENT_SEEDS)]:
        for seed in seeds:
            for case in ['N0', 'N1', 'D0']:
                row, z = run_one(prefix+'_'+label, seed, case, variant, amplitude)
                _, reference = run_one('comparison_'+label, seed, case, 'deferred', amplitude)
                c = compare_arrays(reference, z)
                row = dict(row, versus_deferred=c)
                rows.append(row)
                write(ROOT/(prefix+'.json'), dict(rows=rows))
                if not c['bitwise_control']:
                    raise RuntimeError('An existing-scenario replay changed the control result')


def selected_comparison():
    candidates = []
    for variant, filename in [('manual', 'explicit_capture.json'), ('full_capture', 'full_capture.json'),
                              ('margin_capture', 'margin_capture.json')]:
        path = ROOT/filename
        if not path.exists():
            continue
        gate = json.loads(path.read_text())
        if not gate.get('passed'):
            continue
        group = [r for r in gate['rows'] if r['case'] in {'N0', 'N1'} and r['use_dob']]
        assert len(group)==4
        candidates.append(dict(variant=variant, cycle_mean_ms=float(np.mean([r['cycle']['mean_ms'] for r in group]))))
    assert candidates and candidates[0]['variant']=='manual'
    choice = candidates[0]
    for candidate in candidates[1:]:
        if candidate['cycle_mean_ms'] < .98*choice['cycle_mean_ms']:
            choice = candidate
    sources = ['controllers/fast_final_mppi.py', 'controllers/captured_final_mppi.py',
               'controllers/dynamic_bound_mppi.py', 'controllers/captured_dynamic_bound_mppi.py']
    if choice['variant']=='margin_capture':
        sources += ['controllers/fixed_margin_captured_mppi.py', 'controllers/fixed_margin_dynamic_bound_mppi.py']
    if choice['variant']=='full_capture':
        sources += ['controllers/fully_captured_final_mppi.py', 'controllers/fully_captured_dynamic_bound_mppi.py']
    selection = dict(variant=choice['variant'], candidates=candidates,
        criterion='All seven equivalence/safety checks pass; retain the simpler explicit replay unless another candidate improves mean cycle >=2%. No control retuning.',
        source_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in sources},
        no_gui_default_change=True)
    path = ROOT/'selected_execution.json'
    if path.exists():
        assert json.loads(path.read_text())==selection
    else:
        write(path, selection)
    explicit_comparison(choice['variant'])
    filename = {'manual': 'explicit_comparison', 'full_capture': 'full_comparison', 'margin_capture': 'margin_comparison'}[choice['variant']]
    comparison = json.loads((ROOT/(filename+'.json')).read_text())
    assert len(comparison['rows'])==30
    write(ROOT/'final_comparison.json', dict(**comparison, selection=selection,
        all_bitwise_control=all(r['versus_deferred']['bitwise_control'] for r in comparison['rows'])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['calibration', 'graph', 'comparison', 'explicit', 'explicit_comparison', 'full', 'full_comparison', 'margin', 'margin_comparison', 'selected_comparison'], required=True)
    parser.add_argument('--capture', action='store_true', help='Also test CUDA graph capture after the compiled-loop check')
    args = parser.parse_args()
    cuda = cuda_preflight()
    ROOT.mkdir(parents=True, exist_ok=True)
    plan = dict(cuda=cuda, calibration_seeds=CALIBRATION_SEEDS, primary_seeds=PRIMARY_SEEDS,
        adjacent_seeds=ADJACENT_SEEDS, primary_amplitude=-133., adjacent_amplitude=-132.5,
        comparison='N0/N1 and D0 on the same optimized implementation. D0=N0 cost with compensation-shifted sample/sequence projection; no soft input penalty; nominal rollout unchanged.',
        purpose='Implementation equivalence and runtime; followed by existing-scenario comparator, not independent validation or hyperparameter tuning.',
        source_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
            ['controllers/fast_final_mppi.py', 'controllers/mppi_dob_controller.py', 'kroc/input_headroom_study.py', 'kroc/improve_n1_clipping_cost.py']},
        no_zip=True, no_gui_default_change=True)
    plan_path = ROOT/'plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text())['source_sha256'] == plan['source_sha256'], 'Source changed after freeze'
    else:
        write(plan_path, plan)
    if args.phase == 'graph':
        graph_check(args.capture)
    else:
        {'calibration': calibrate, 'comparison': comparison, 'explicit': explicit_capture_check,
         'explicit_comparison': explicit_comparison,
         'full': lambda: explicit_capture_check('full_capture'),
         'full_comparison': lambda: explicit_comparison('full_capture'),
         'margin': lambda: explicit_capture_check('margin_capture'),
         'margin_comparison': lambda: explicit_comparison('margin_capture'),
         'selected_comparison': selected_comparison}[args.phase]()


if __name__ == '__main__':
    main()
