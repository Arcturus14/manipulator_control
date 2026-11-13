"""Measurement-noise experiment support; physical state remains untouched."""
from dataclasses import asdict, dataclass
import math

import torch

from controllers.mppi_dob_controller import UR5Tracker


@dataclass(frozen=True)
class SensorNoiseConfig:
    position_std_rad: float = 1e-3
    velocity_std_rad_s: float = 1e-2
    seed: int = 10007

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(values[name]) for name in
                   ("position_std_rad", "velocity_std_rad_s")):
            raise ValueError("sensor noise standard deviations must be finite")
        if self.position_std_rad < 0 or self.velocity_std_rad_s < 0:
            raise ValueError("sensor noise standard deviations must be nonnegative")
        if self.seed < 0:
            raise ValueError("sensor noise seed must be nonnegative")


class StateMeasurementNoise:
    """Independent joint-position and joint-velocity Gaussian measurements."""

    def __init__(self, device, dtype, config: SensorNoiseConfig):
        self.config = config
        self.device, self.dtype = device, dtype
        self.generator = torch.Generator(device=device)
        self.std = torch.tensor(
            [config.position_std_rad] * 6
            + [config.velocity_std_rad_s] * 6,
            device=device, dtype=dtype,
        )
        self.enabled = bool(
            config.position_std_rad or config.velocity_std_rad_s
        )
        self.reset()

    def reset(self):
        self.generator.manual_seed(self.config.seed)

    def measure(self, true_state):
        if not self.enabled:
            return true_state.clone()
        noise = torch.randn(
            true_state.shape, generator=self.generator,
            device=self.device, dtype=self.dtype,
        ) * self.std
        return true_state + noise


class SensorNoiseTracker(UR5Tracker):
    """Post-DOB tracker with a separate true and measured state."""

    def __init__(self, *args, sensor_noise=None, torque_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < torque_scale <= 1:
            raise ValueError("torque scale must be in (0, 1]")
        self.dynamics.u_max.mul_(torque_scale)
        self.dynamics.u_min.mul_(torque_scale)
        self.sensor_noise_config = SensorNoiseConfig(**(sensor_noise or {}))
        self.sensor_model = StateMeasurementNoise(
            self.device, self.dtype, self.sensor_noise_config
        )
        self.measured_state_history = []

    def run(self, *args, **kwargs):
        if (not kwargs.get("use_dob")
                or kwargs.get("dob_compensation_mode") != "output"):
            raise ValueError("sensor-noise comparison requires Post-DOB")
        self.sensor_model.reset()
        self.measured_state_history.clear()
        return super().run(*args, **kwargs)

    def _measure_state(self, true_state):
        measured = self.sensor_model.measure(true_state)
        self.measured_state_history.append(measured.detach().clone())
        return measured

    def export_sensor_data(self):
        measured = torch.stack(self.measured_state_history).cpu().numpy()
        return {
            "measured_state": measured,
            "sensor_position_std_rad": self.sensor_noise_config.position_std_rad,
            "sensor_velocity_std_rad_s": self.sensor_noise_config.velocity_std_rad_s,
            "sensor_noise_seed": self.sensor_noise_config.seed,
        }
