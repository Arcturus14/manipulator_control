"""Also replay the unchanged stateful DOB update through explicit state inputs."""
import numpy as np

from controllers.captured_final_mppi import CapturedFinalTracker, CapturedFunction


class FullyCapturedFinalTracker(CapturedFinalTracker):
    DOB_FIELDS = ('d_hat', 'residual', 'tau_model', 'raw_acceleration', 'filtered_acceleration')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._original_dob_update = self.dob.update

    def _warm_up_controller(self, state, feedforward, sampling_center_mode):
        self.dob.update = self._original_dob_update
        super()._warm_up_controller(state, feedforward, sampling_center_mode)
        saved = {key: getattr(self.dob, key) for key in self.DOB_FIELDS}

        def update(applied, q_previous, velocity_previous, q_current, velocity_current,
                   estimate, filtered_acceleration):
            self.dob.d_hat = estimate
            self.dob.filtered_acceleration = filtered_acceleration
            self._original_dob_update(applied, q_previous, velocity_previous,
                                      q_current, velocity_current)
            return tuple(getattr(self.dob, key) for key in self.DOB_FIELDS)

        try:
            self._dob_graph = CapturedFunction(update, (feedforward[0], state[:6], state[6:],
                state[:6], state[6:], saved['d_hat'], saved['filtered_acceleration']), self.device)
        finally:
            for key, value in saved.items():
                setattr(self.dob, key, value)
        self.dob.update = self._replay_dob_update

    def _replay_dob_update(self, applied, q_previous, velocity_previous, q_current, velocity_current):
        values = self._dob_graph(applied, q_previous, velocity_previous, q_current, velocity_current,
                                self.dob.d_hat, self.dob.filtered_acceleration)
        for key, value in zip(self.DOB_FIELDS, values):
            setattr(self.dob, key, value)
        return self.dob.d_hat

    def export_sensor_data(self):
        result = super().export_sensor_data()
        result['fast_final_explicit_dob_graph'] = np.asarray(True)
        return result
