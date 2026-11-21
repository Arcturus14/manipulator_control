"""Isolated GPU speed check; does not change frozen controllers or GUI defaults.

Same final N0/N1 equations, K/H, sampling, costs and DOB filters. Full auxiliary
compilation may change floating point rounding; retain every closed-loop result.
Run as a module. Compilation/warmup is outside the recorded control loop.
"""
import argparse
import gc
import json
import types
from pathlib import Path

import numpy as np
import torch

from benchmarks.benchmark_dob_aware_nominal_shift import cuda_preflight
from controllers.research_baseline_controller import asynchronous_forward_dynamics
from experiment_runner import run_mppi
from kroc.improve_n1_clipping_cost import ClippingLossN1, COVARIANCE, TEMPERATURE
from kroc.position_input_headroom_study import PositionInputTracker, metrics, raised_cosine_load
from kroc.run_input_headroom_study import run_options
from kroc.run_pre_input_baseline import load_preset
from kroc.search_lowfreq_estimator_margin import geometry
from kroc.validate_n1_final_fixed import audit

ROOT = Path('kroc/results/final_controller_speed_20261117')


class SpeedMixin:
    def __init__(self, *args, speed_variant='reference', **kwargs):
        self.speed_variant = speed_variant
        super().__init__(*args, **kwargs)
        if speed_variant not in {'reference', 'async', 'async_aux'}:
            raise ValueError(speed_variant)
        if speed_variant != 'reference':
            self.dynamics.forward_dynamics = types.MethodType(
                asynchronous_forward_dynamics, self.dynamics)
        if speed_variant == 'async_aux':
            options = {'triton.cudagraphs': False, 'triton.cudagraph_trees': False}
            for field in ('mass_matrix_and_bias', 'plant_step_effective', 'rollout_effective'):
                setattr(self.dynamics, field, torch.compile(getattr(self.dynamics, field), options=options))
            self.dob.update = torch.compile(self.dob.update, options=options)

    def _warm_up_controller(self, state, feedforward, sampling_center_mode):
        super()._warm_up_controller(state, feedforward, sampling_center_mode)
        if self.speed_variant == 'async_aux':
            self.dynamics.mass_matrix_and_bias(state[None, :6], state[None, 6:])
            torque = feedforward[0][None, :]
            self.dynamics.plant_step_effective(state[None, :], torque)
            self.dynamics.rollout_effective(state[None, :], torque)
            fields = ('d_hat', 'residual', 'tau_model', 'raw_acceleration', 'filtered_acceleration')
            saved = {key: getattr(self.dob, key).clone() for key in fields}
            self.dob.update(torque[0], state[:6], state[6:], state[:6], state[6:])
            for key, value in saved.items():
                setattr(self.dob, key, value)
            torch.cuda.synchronize(self.device)


class SpeedN0(SpeedMixin, PositionInputTracker):
    pass


class SpeedN1(SpeedMixin, ClippingLossN1):
    pass


def timing(values):
    values = np.asarray(values)[20:] * 1000
    return dict(mean_ms=float(values.mean()), p95_ms=float(np.percentile(values, 95)),
                max_ms=float(values.max()), over20_fraction=float((values > 20).mean()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[6807, 6937])
    parser.add_argument('--variants', nargs='+', default=['reference', 'async', 'async_aux'])
    args = parser.parse_args()
    cuda = cuda_preflight()
    ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        for case in ('N0', 'N1'):
            reference = None
            for variant in args.variants:
                path = ROOT / f'{variant}_{case}_{seed}.npz'
                if not path.exists():
                    print('RUN', seed, case, variant, flush=True)
                    ref, q0, _ = geometry(.89)
                    opts = run_options(load_preset(), seed, case, 0.)
                    opts.update(temperature=TEMPERATURE, noise_variance=COVARIANCE,
                                tracker_class=SpeedN1 if case == 'N1' else SpeedN0)
                    opts['tracker_options'].update(speed_variant=variant, headroom_fraction=.95,
                        unclamped_diagnostic=False, dob_cutoff_hz=3., dob_acceleration_lpf_cutoff_hz=6.)
                    if case == 'N1':
                        opts['tracker_options']['clipping_weight'] = 1.
                    z = run_mppi(ref, initial_q=q0, has_disturbance=True,
                                 disturbance_function=raised_cosine_load(2, -133.), **opts)
                    audit(z, case, -133.)
                    np.savez_compressed(path, **{k: v for k, v in z.items() if not k.startswith('_')})
                    del z
                    gc.collect()
                    torch.cuda.empty_cache()
                with np.load(path) as stored:
                    z = dict(stored)
                audit(z, case, -133.)
                if variant == 'reference':
                    reference = z
                row = dict(seed=seed, case=case, variant=variant, path=str(path),
                           metrics=metrics(z), solve=timing(z['solve_t']),
                           cycle=timing(z['control_cycle_t_raw']))
                if reference is not None:
                    row['bitwise_reference'] = all(np.array_equal(z[k], reference[k])
                        for k in ('q', 'qd', 'u_mppi', 'u_applied', 'd_hat_used'))
                    row['max_applied_difference_nm'] = float(np.max(np.abs(z['u_applied'] - reference['u_applied'])))
                rows.append(row)
                (ROOT / 'summary.json').write_text(json.dumps(dict(cuda=cuda, rows=rows,
                    warmup_steps_excluded=20, hard_realtime_guarantee=False,
                    note='Auxiliary compilation can change rounding and closed-loop trajectories. No GUI/default change.'), indent=2))
                print('RESULT', json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
