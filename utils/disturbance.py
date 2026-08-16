"""External joint-torque disturbance generators."""
from __future__ import annotations

import math

import torch

from config import D_AMP, D_FREQ, DT


def sinusoidal_disturbance(step: int, device: torch.device, dtype: torch.dtype,
                           dt: float = DT) -> torch.Tensor:
    time = step * dt
    values = [amplitude * math.sin(2 * math.pi * frequency * time)
              for amplitude, frequency in zip(D_AMP, D_FREQ)]
    return torch.tensor(values, dtype=dtype, device=device)
