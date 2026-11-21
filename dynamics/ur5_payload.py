"""Opt-in rigid cube on link6; original UR5Dynamics and global config untouched."""
from dataclasses import asdict, dataclass
import math

import torch

from dynamics.ur5_dynamics import UR5Dynamics, _skew


@dataclass(frozen=True)
class CubePayload:
    mass_kg: float = 5.
    side_m: float = .10
    center_in_flange_m: tuple[float, float, float] | None = None

    def __post_init__(self):
        if not math.isfinite(self.mass_kg) or not 0 <= self.mass_kg <= 5:
            raise ValueError('Total rigid payload mass must be in[0,5]kg; no gripper is included')
        if not math.isfinite(self.side_m) or self.side_m <= 0:
            raise ValueError('Cube side must be finite and positive')
        center = self.center_in_flange_m
        if center is None:
            center = (0., 0., self.side_m/2)
        if len(center) != 3 or not all(math.isfinite(x) for x in center):
            raise ValueError('Payload center must contain three finite flange-frame coordinates')
        object.__setattr__(self, 'center_in_flange_m', tuple(float(x) for x in center))

    @property
    def inertia_com_kg_m2(self):
        return self.mass_kg*self.side_m**2/6

    def to_dict(self):
        return dict(asdict(self), inertia_com_diag_kg_m2=[self.inertia_com_kg_m2]*3,
                    gripper_mass_kg=0., tcp='cube center', attachment='rigid; cube axes aligned to link6 DH frame')


class PayloadUR5Dynamics(UR5Dynamics):
    """Merge payload spatial inertia into link6, affecting M,C,g consistently.

    Existing RNEA, reflected armature, damping, plant RK4, and rollout integrator
    are inherited. The TCP/COM is expressed in the existing last-DH-link frame.
    A cube has isotropic COM inertia; its fixed rotation does not change it.
    """
    def __init__(self, device, dtype, payload: CubePayload, **kwargs):
        super().__init__(device, dtype, **kwargs)
        self.payload = payload
        r = torch.tensor(payload.center_in_flange_m, device=device, dtype=dtype)
        cross = _skew(r)
        eye = torch.eye(3, device=device, dtype=dtype)
        m = payload.mass_kg
        upper = torch.cat((payload.inertia_com_kg_m2*eye+m*cross@cross.T, m*cross), -1)
        lower = torch.cat((m*cross.T, m*eye), -1)
        self.payload_spatial_inertia = torch.cat((upper, lower), -2)
        if m:
            self.spatial_inertia = self.spatial_inertia.clone()
            self.spatial_inertia[-1] += self.payload_spatial_inertia

    def metadata(self):
        return self.payload.to_dict()
