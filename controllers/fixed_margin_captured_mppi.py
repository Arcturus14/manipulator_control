"""Cache invariant fixed state bounds and replay their unchanged diagnostics."""
import numpy as np
import torch

from controllers.captured_final_mppi import CapturedFinalTracker, CapturedFunction
from controllers.fast_final_mppi import COMPILE_OPTIONS


class FixedMarginCapturedTracker(CapturedFinalTracker):
    MARGIN_KEYS = ('margin', 'd_hat_margin', 'a_d_hat', 'a_d_hat_raw', 'acceleration_deadband',
                   'headroom_positive', 'headroom_negative', 'gain_lower', 'gain_upper',
                   'center_horizon', 'headroom_center_horizon')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cfg = self.margins.config
        if cfg.mode != 'fixed':
            raise ValueError('Only fixed, invariant bounds can use this implementation')
        self._margin_deadband = torch.tensor(cfg.acceleration_deadband_rad_s2,
                                             device=self.device, dtype=self.dtype)
        self._fixed_diagnostics = torch.compile(self._fixed_diagnostics_value, options=COMPILE_OPTIONS)

    def _fixed_diagnostics_value(self, state, center, d_hat):
        cfg = self.margins.config
        dyn = self.dynamics
        clipped = d_hat.clamp(-dyn.u_max*cfg.disturbance_clip_fraction,
                              dyn.u_max*cfg.disturbance_clip_fraction)
        mass, _ = dyn.mass_matrix_and_bias(state[:6], state[6:])
        raw = -torch.linalg.solve_ex(mass, clipped[:, None], check_errors=False)[0].squeeze(-1)
        acceleration = torch.sign(raw)*torch.relu(raw.abs()-self._margin_deadband)
        headroom_center = center+d_hat[None] if cfg.output_compensated_headroom else center
        positive = (dyn.u_max-headroom_center).amin(0)
        negative = (headroom_center-dyn.u_min).amin(0)
        safe = dyn.u_max*cfg.headroom_safe_fraction
        lower = 1+cfg.headroom_gain*torch.relu(1-positive/safe)
        upper = 1+cfg.headroom_gain*torch.relu(1-negative/safe)
        return (self.margins.margin, clipped, acceleration, raw, self._margin_deadband,
                positive, negative, lower, upper, center, headroom_center)

    def _warm_up_controller(self, state, feedforward, sampling_center_mode):
        super()._warm_up_controller(state, feedforward, sampling_center_mode)
        self._margin_graph = CapturedFunction(self._fixed_diagnostics,
            (state, feedforward[self.step:self.step+self.horizon], self.dob.d_hat), self.device)

    def _prepare_solve(self, state, sampling_center):
        # FastFinalTracker.run resets/applies fixed bounds before every
        # episode. Since fixed margins never change, repeated bound copies
        # are idempotent and unnecessary. Keep the original exact buffers.
        values = self._margin_graph(state, sampling_center, self.dob.d_hat)
        self.margin_history.append({key: value.detach() for key, value in zip(self.MARGIN_KEYS, values)})
        self._record_applied_bounds()

    def export_sensor_data(self):
        result = super().export_sensor_data()
        result['fast_final_fixed_margin_graph'] = np.asarray(True)
        return result
