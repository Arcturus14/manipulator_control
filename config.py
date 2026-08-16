"""UR5 MPPI simulation configuration.

All tunable values live here so the model, costs, and controller do not depend
on one another's implementation details.
"""
from __future__ import annotations

import math

import numpy as np

# UR5 standard DH parameters: (a, d, alpha, theta_offset)
DH_PARAMS = (
    (0.0, 0.089159, math.pi / 2, 0.0),
    (-0.425, 0.0, 0.0, 0.0),
    (-0.39225, 0.0, 0.0, 0.0),
    (0.0, 0.10915, math.pi / 2, 0.0),
    (0.0, 0.09465, -math.pi / 2, 0.0),
    (0.0, 0.0823, 0.0, 0.0),
)

# Joint-space limits and viscous friction
I_EFF = (3.70, 8.40, 2.30, 1.20, 1.40, 0.30)
# Reflected motor/gear inertia added to the rigid-link mass matrix. This is
# essential for a geared manipulator and prevents unrealistically large wrist
# acceleration when only the small link inertia is considered.
JOINT_ARMATURE = I_EFF
B_DAMP = (0.12, 0.12, 0.10, 0.08, 0.08, 0.06)
TAU_MAX = (150.0, 150.0, 150.0, 28.0, 28.0, 28.0)
DQ_MAX = 6.0

# UR5 rigid-body inertial parameters in the modified-DH link frames.
# Values follow the public UR5/ur_description inertial model; inertia tensors
# are expressed at each center of mass in kg*m^2.
LINK_MASSES = (3.7000, 8.3930, 2.2750, 1.2190, 1.2190, 0.1879)
LINK_COM = (
    (0.0, -0.02561, 0.00193),
    (0.21250, 0.0, 0.11336),
    (0.15000, 0.0, 0.02650),
    (0.0, -0.00180, 0.01634),
    (0.0, 0.00180, 0.01634),
    (0.0, 0.0, -0.001159),
)
LINK_INERTIA_DIAG = (
    (0.0102675, 0.0102675, 0.0066600),
    (0.2268910, 0.2268910, 0.0151074),
    (0.0494433, 0.0494433, 0.0040950),
    (0.0027000, 0.0027000, 0.0022000),
    (0.0027000, 0.0027000, 0.0022000),
    (0.0001200, 0.0001200, 0.0001000),
)
GRAVITY = (0.0, 0.0, -9.81)
MASS_MATRIX_REGULARIZATION = 1e-5

# Simulation
DT = 0.02
T_SIM = 10.0
N_STEPS = int(T_SIM / DT)

# Ignore one-time Torch/CUDA compilation and cache warm-up in realtime metrics.
TIMING_WARMUP_STEPS = 10

# MPPI
HORIZON = 10
N_SAMPLES = 136
MPPI_SEED = 7
LAMBDA = 0.2
NOISE_VAR = (16.0, 28.0, 16.0, 3.2, 3.2, 1.0)
HORIZON_OBS = 10
N_SAMPLES_OBS = 136

# MPPI cost weights
MPPI_W_POS = 960_000.0
MPPI_W_TERM = 2_400_000.0
W_ACT = 0.0005
W_VEL = 5.0
W_SMOOTH = 0.25
W_SMOOTH_OBS = 0.025

# SQP position cost weights are kept separate so MPPI tuning does not change
# the deterministic controller used as the comparison baseline.
W_POS = 120_000.0
W_TERM = 300_000.0

# Static obstacles
OBSTACLES = (
    {"center": (0.492, 0.092), "r_obs": 0.008},
    {"center": (0.308, -0.092), "r_obs": 0.008},
)
W_EXP = 5e3
K_EXP = 1.5
OBS_MARGIN = 0.05
SOFT_ZONE = 0.03

# Dynamic crossing obstacles
CROSS_SAFE_RADIUS = 0.056
CROSS_EXP_W = 160.0
CROSS_EXP_K = 38.0
CROSS_EXP_ZONE = 0.012
CROSS_RADIUS = 0.030
CROSS_MARGIN = 0.026

# Reference trajectory and disturbance
TRAJ_CENTER = np.array([0.40, 0.0, 0.35])
TRAJ_RADIUS = 0.13
D_AMP = (15.0, -12.0, 10.0, 3.0, -3.0, 1.5)
D_FREQ = (1.0, 1.5, 0.8, 1.2, 0.9, 1.1)
DOB_ALPHA = 40.0
