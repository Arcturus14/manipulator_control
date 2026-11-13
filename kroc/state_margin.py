"""Joint-wise DOB margins, frozen over each solve; no safety guarantee."""
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from controllers.mppi_dob_controller import UR5Tracker
from kroc.sensor_noise import SensorNoiseConfig, StateMeasurementNoise


@dataclass(frozen=True)
class MarginConfig:
    mode: str = "fixed"
    reaction_time_s: float = 0.10
    position_gain: float = 1.0
    velocity_gain: float = 1.0
    min_fraction: float = 0.025
    fixed_fraction: float = 0.05
    max_fraction: float = 0.10
    min_internal_fraction: float = 0.50
    headroom_safe_fraction: float = 0.20
    headroom_gain: float = 1.0
    disturbance_clip_fraction: float = 1.0
    beta_up: float = 1.0
    beta_down: float = 0.10
    acceleration_deadband_rad_s2: tuple[float, ...] = (0.0,) * 6
    penalty_kind: str = "hinge"
    softplus_sharpness: float = 8.0
    output_compensated_headroom: bool = False

    def __post_init__(self):
        import math
        values = asdict(self)
        scalar_values = (v for k, v in values.items()
                         if k not in {"mode", "penalty_kind",
                                      "acceleration_deadband_rad_s2"})
        if any(not math.isfinite(v) for v in scalar_values):
            raise ValueError("margin parameters must be finite")
        if (len(self.acceleration_deadband_rad_s2) != 6
                or any(not math.isfinite(v) or v < 0
                       for v in self.acceleration_deadband_rad_s2)):
            raise ValueError("acceleration deadband must contain six nonnegative values")
        if self.mode not in {"fixed", "directional", "headroom"}:
            raise ValueError("unknown margin mode")
        if self.penalty_kind not in {"hinge", "softplus"}:
            raise ValueError("unknown state-limit penalty")
        if not 0 <= self.min_fraction <= self.fixed_fraction <= self.max_fraction < 0.5:
            raise ValueError("require 0 <= min <= fixed <= max < 0.5")
        if not 0 < self.min_internal_fraction < 1:
            raise ValueError("invalid minimum internal interval")
        if 2 * self.min_fraction > 1 - self.min_internal_fraction:
            raise ValueError("minimum margins leave insufficient internal interval")
        if not 0 < self.beta_down <= self.beta_up <= 1:
            raise ValueError("invalid asymmetric filter coefficients")
        if min(self.reaction_time_s, self.headroom_safe_fraction,
               self.disturbance_clip_fraction,
               self.softplus_sharpness) <= 0:
            raise ValueError("time, headroom and clipping must be positive")
        if min(self.position_gain, self.velocity_gain, self.headroom_gain) < 0:
            raise ValueError("gains must be nonnegative")


class StateMargins:
    """Fraction parameters refer to full physical position/velocity span."""

    def __init__(self, dynamics, cost, config: MarginConfig):
        self.dynamics, self.cost, self.config = dynamics, cost, config
        self.span = torch.stack((cost.q_max - cost.q_min, 2 * cost.dq_max))
        # shape (state kind q/dq, side lower/upper, joint)
        self.margin = self.span[:, None, :].expand(-1, 2, -1).clone()
        self.reset()

    def reset(self):
        config = self.config
        initial = config.fixed_fraction if config.mode == "fixed" else config.min_fraction
        self.margin.copy_(self.span[:, None, :] * initial)
        self._protect()
        self.apply()

    def _protect(self):
        cfg = self.config
        minimum = self.span[:, None, :] * cfg.min_fraction
        self.margin.copy_(torch.maximum(minimum, torch.minimum(
            self.margin, self.span[:, None, :] * cfg.max_fraction)))
        # Scale only the excess over the required minimum, preserving it.
        extra = self.margin - minimum
        budget = self.span * (1 - cfg.min_internal_fraction - 2 * cfg.min_fraction)
        scale = (budget / extra.sum(dim=1).clamp_min(1e-12)).clamp(max=1)
        self.margin.copy_(minimum + extra * scale[:, None, :])

    def apply(self):
        cost = self.cost
        cost.q_soft_min.copy_(cost.q_min + self.margin[0, 0])
        cost.q_soft_max.copy_(cost.q_max - self.margin[0, 1])
        cost.dq_soft_max.copy_(cost.dq_max - self.margin[1, 1])
        # Existing cost is replaced by an asymmetric equivalent below.

    def update(self, state, center, d_hat):
        cfg, dyn = self.config, self.dynamics
        clipped = d_hat.clamp(-dyn.u_max * cfg.disturbance_clip_fraction,
                             dyn.u_max * cfg.disturbance_clip_fraction)
        mass, _ = dyn.mass_matrix_and_bias(state[:6], state[6:])
        acceleration_raw = -torch.linalg.solve(
            mass, clipped.unsqueeze(-1)
        ).squeeze(-1)
        deadband = torch.as_tensor(
            cfg.acceleration_deadband_rad_s2,
            device=acceleration_raw.device, dtype=acceleration_raw.dtype,
        )
        # Soft threshold: remove the calibrated nominal-noise floor without a
        # discontinuous jump when a genuine disturbance crosses the threshold.
        acceleration = torch.sign(acceleration_raw) * torch.relu(
            acceleration_raw.abs() - deadband
        )
        # Use UNCLAMPED U_ff + R. Negative reserve means an over-limit plan.
        headroom_center = center
        if cfg.output_compensated_headroom:
            # Post-DOB applies d_hat after the MPPI action.  Use the command
            # that would reach the actuator, not only the internal sampling
            # center, when the experiment tests saturation-aware tightening.
            headroom_center = center + d_hat.detach()[None, :]
        positive = (dyn.u_max - headroom_center).amin(dim=0)
        negative = (headroom_center - dyn.u_min).amin(dim=0)
        safe = dyn.u_max * cfg.headroom_safe_fraction
        gain_lower = 1 + cfg.headroom_gain * torch.relu(1 - positive / safe)
        gain_upper = 1 + cfg.headroom_gain * torch.relu(1 - negative / safe)
        if cfg.mode != "fixed":
            a = torch.stack((torch.relu(-acceleration), torch.relu(acceleration)))
            v = torch.stack((torch.relu(-state[6:]), torch.relu(state[6:])))
            target = self.span[:, None, :] * cfg.min_fraction + torch.stack((
                cfg.position_gain * (v * cfg.reaction_time_s
                                     + 0.5 * a * cfg.reaction_time_s ** 2),
                cfg.velocity_gain * a * cfg.reaction_time_s,
            ))
            if cfg.mode == "headroom":
                target = target * torch.stack((gain_lower, gain_upper))[None, :, :]
            # Project target BEFORE filtering. Convex updates can still violate
            # sum limits with asymmetric rates, so project the final state too.
            previous = self.margin.clone()
            self.margin.copy_(target)
            self._protect()
            beta = torch.where(self.margin > previous, cfg.beta_up, cfg.beta_down)
            self.margin.copy_(previous + beta * (self.margin - previous))
            self._protect()
        self.apply()
        return {
            "margin": self.margin.clone(), "d_hat_margin": clipped,
            "a_d_hat": acceleration, "a_d_hat_raw": acceleration_raw,
            "acceleration_deadband": deadband,
            "headroom_positive": positive,
            "headroom_negative": negative,
            "gain_lower": gain_lower, "gain_upper": gain_upper,
            "center_horizon": center.clone(),
            "headroom_center_horizon": headroom_center.clone(),
        }

    def cost_value(self, state):
        cost = self.cost
        q, dq = state[..., :6], state[..., 6:]
        q_low = (cost.q_soft_min - q) / cost.q_lower_margin
        q_high = (q - cost.q_soft_max) / cost.q_upper_margin
        dq_low = (-cost.dq_max + self.margin[1, 0] - dq) / cost.dq_margin
        dq_high = (dq - cost.dq_soft_max) / cost.dq_margin
        if self.config.penalty_kind == "softplus":
            sharpness = self.config.softplus_sharpness
            q_low, q_high, dq_low, dq_high = (
                F.softplus(value * sharpness) / sharpness
                for value in (q_low, q_high, dq_low, dq_high)
            )
        else:
            q_low, q_high, dq_low, dq_high = (
                torch.relu(value)
                for value in (q_low, q_high, dq_low, dq_high)
            )
        return (cost.joint_position_limit_weight * (q_low.square() + q_high.square()).sum(-1)
                + cost.joint_velocity_limit_weight * (dq_low.square() + dq_high.square()).sum(-1))


class KrocTracker(UR5Tracker):
    def __init__(self, *args, margin_config=None, torque_scale=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 < torque_scale <= 1:
            raise ValueError("torque scale must be in (0, 1]")
        # MPPI bounds and TrackingCost share these tensors with dynamics.
        self.dynamics.u_max.mul_(torque_scale)
        self.dynamics.u_min.mul_(torque_scale)
        self.margin_config = MarginConfig(**(margin_config or {}))
        self.margins = StateMargins(self.dynamics, self.tracking_cost, self.margin_config)
        self.tracking_cost.state_limit_cost = self.margins.cost_value
        self.margin_history = []

    def run(self, *args, **kwargs):
        if not kwargs.get("use_dob") or kwargs.get("dob_compensation_mode") not in {
                "output", "rollout_model_only", "rollout", "estimate_only"}:
            raise ValueError(
                "KRoC margin comparison requires a supported DOB architecture"
            )
        self.margin_history.clear()
        self.margins.reset()
        return super().run(*args, **kwargs)

    def _prepare_solve(self, state, sampling_center):
        # Margin generation is architecture-independent: post-DOB does not put
        # d_hat in the rollout model, but it must still shape the state margin.
        # The torque-headroom reference remains the unclamped U_ff + R center;
        # d_hat is intentionally not added to it.
        values = self.margins.update(state, sampling_center, self.dob.d_hat)
        # Keep GPU tensors until the episode ends; no per-step CPU transfer.
        self.margin_history.append({k: v.detach().clone() for k, v in values.items()})

    def export_margin_data(self):
        return {f"kroc_{key}": torch.stack([row[key] for row in self.margin_history]).cpu().numpy()
                for key in self.margin_history[0]}


class NoiseAwareKrocTracker(KrocTracker):
    """Dynamic-margin tracker with measured/true state separation."""

    def __init__(self, *args, sensor_noise=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sensor_noise_config = SensorNoiseConfig(**(sensor_noise or {}))
        self.sensor_model = StateMeasurementNoise(
            self.device, self.dtype, self.sensor_noise_config
        )
        self.measured_state_history = []

    def run(self, *args, **kwargs):
        if kwargs.get("dob_compensation_mode") not in {
                "output", "rollout_model_only", "rollout", "estimate_only"}:
            raise ValueError(
                "noise-aware margin experiment requires a supported DOB "
                "architecture"
            )
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
