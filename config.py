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

# MPPI compilation is explicitly completed before the control loop, so every
# control step is included in realtime metrics.
TIMING_WARMUP_STEPS = 0

# MPPI
HORIZON = 10
N_SAMPLES = 136
MPPI_SEED = 7
LAMBDA = 0.2
NOISE_VAR = (16.0, 28.0, 16.0, 3.2, 3.2, 1.0)
HORIZON_OBS = 10
N_SAMPLES_OBS = 136
SQP_MAX_ITER = 5
ILQR_MAX_ITER = 2

# MPPI cost weights
MPPI_W_POS = 960_000.0
MPPI_W_TERM = 2_400_000.0
W_ACT = 0.0005
W_VEL = 5.0
W_SMOOTH = 0.25
W_SMOOTH_OBS = 0.025

# iLQR uses the same cost terms as MPPI but keeps optimizer-specific weights.
ILQR_W_POS = 1_200_000.0
ILQR_W_TERM = 4_800_000.0
ILQR_W_ACT = 0.0005
ILQR_W_VEL = 1.0
ILQR_W_SMOOTH = 0.001
ILQR_W_SMOOTH_OBS = 0.0001

# MPPI static-cube barrier, tuned for the 20 ms control period
MPPI_STATIC_HORIZON = 9
MPPI_STATIC_SAMPLES = 112
MPPI_STATIC_EXP_WEIGHT = 5_000.0
ILQR_STATIC_EXP_WEIGHT = 1_500.0
MPPI_STATIC_EXP_GAIN = 500.0
MPPI_STATIC_MARGIN = 0.002
MPPI_STATIC_ZONE = 0.004

# SQP uses the same mathematical cost terms as MPPI.  These weights remain
# separate because the deterministic local optimizer and the sampling-based
# optimizer generally need different numerical scaling.
SQP_W_POS = 960_000.0
SQP_W_TERM = 2_400_000.0
SQP_W_ACT = 0.0005
SQP_W_VEL = 5.0
SQP_W_SMOOTH = 0.0001
SQP_W_SMOOTH_OBS = 0.0001
SQP_STATIC_EXP_WEIGHT = 8_250.0
SQP_CROSS_EXP_WEIGHT = 8_500.0

# Static obstacles generated from the selected reference trajectory
STATIC_OBSTACLE_SIZE = 0.010
STATIC_OBSTACLE_COUNT = 4
# Dynamic trajectory-chasing cube
# MPPI and SQP keep separate barrier weights because their optimizers have
# different numerical scaling; the gain, zone, margin, and geometry are shared.
CROSS_EXP_W = 1_400.0
ILQR_CROSS_EXP_WEIGHT = 700.0
CROSS_EXP_K = 38.0
CROSS_EXP_ZONE = 0.012
CROSS_MARGIN = 0.026
DYNAMIC_OBSTACLE_SIZE = 0.010
DYNAMIC_OBSTACLE_SPEED_RATIO = 4.6
DYNAMIC_OBSTACLE_PHASE_OFFSET = 0.5

# Reference trajectory and disturbance
TRAJ_CENTER = np.array([0.40, 0.0, 0.35])
TRAJ_RADIUS = 0.13
D_AMP = (15.0, -12.0, 10.0, 3.0, -3.0, 1.5)
D_FREQ = (1.0, 1.5, 0.8, 1.2, 0.9, 1.1)
DOB_ALPHA = 40.0
