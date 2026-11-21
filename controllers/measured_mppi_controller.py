"""GUI-compatible measured-state MPPI; no dynamic state-margin machinery."""
import numpy as np
from controllers.mppi_dob_controller import UR5Tracker
from kroc.sensor_noise import SensorNoiseConfig, StateMeasurementNoise


class MeasuredStateTracker(UR5Tracker):
    def __init__(self,*args,sensor_noise=None,physical_limit_penalty_only=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.sensor_noise_config=SensorNoiseConfig(**(sensor_noise or {}))
        self.sensor_model=StateMeasurementNoise(self.device,self.dtype,self.sensor_noise_config)
        self.physical_limit_penalty_only=bool(physical_limit_penalty_only)
        self.measured_state_history=[]
        if self.physical_limit_penalty_only:
            # Keep the native nonzero cost normalizers; only move the onset.
            cost=self.tracking_cost
            cost.q_soft_min.copy_(cost.q_min)
            cost.q_soft_max.copy_(cost.q_max)
            cost.dq_soft_max.copy_(cost.dq_max)

    def run(self,*args,**kwargs):
        self.sensor_model.reset();self.measured_state_history.clear()
        return super().run(*args,**kwargs)

    def _measure_state(self,true_state):
        measured=self.sensor_model.measure(true_state)
        self.measured_state_history.append(measured.detach().clone())
        return measured

    def export_sensor_data(self):
        import torch
        cfg=self.sensor_noise_config
        return {'measured_state':torch.stack(self.measured_state_history).cpu().numpy(),
            'sensor_position_std_rad':np.asarray(cfg.position_std_rad),
            'sensor_velocity_std_rad_s':np.asarray(cfg.velocity_std_rad_s),
            'sensor_noise_seed':np.asarray(cfg.seed),
            'physical_limit_penalty_only':np.asarray(self.physical_limit_penalty_only),
            'q_soft_min':self.tracking_cost.q_soft_min.cpu().numpy().copy(),
            'q_soft_max':self.tracking_cost.q_soft_max.cpu().numpy().copy(),
            'dq_soft_max':self.tracking_cost.dq_soft_max.cpu().numpy().copy()}
