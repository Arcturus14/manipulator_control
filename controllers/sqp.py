"""acados Full-SQP controller sharing the UR5 model and runtime with MPPI.

The prediction model, simulated plant, and DOB nominal torque all use

    M(q) qdd + C(q, qd) qd + g(q) + B qd = tau.

CasADi RNEA builds the SQP prediction model; the plant uses the same numerical
parameters through :class:`dynamics.ur5_dynamics.UR5Dynamics`.
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings
from pathlib import Path

# CasADi is installed in the mppi environment, while acados_template belongs to
# the shared source checkout. Append paths so the test environment keeps its
# own NumPy/Torch versions.
_CASADI_SITE = "/home/arclab-04/miniconda3/envs/mppi/lib/python3.10/site-packages"
_ACADOS_PYTHON = "/home/arclab-04/acados/interfaces/acados_template"
for _path in (_CASADI_SITE, _ACADOS_PYTHON):
    if _path not in sys.path:
        sys.path.append(_path)

os.environ.setdefault("ACADOS_SOURCE_DIR", "/home/arclab-04/acados")

import casadi as ca
import numpy as np
import torch
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver

from config import (B_DAMP, CROSS_MARGIN, CROSS_RADIUS, DH_PARAMS, DQ_MAX, DT,
                    GRAVITY, HORIZON, JOINT_ARMATURE, LINK_COM,
                    LINK_INERTIA_DIAG, LINK_MASSES,
                    MASS_MATRIX_REGULARIZATION, STATIC_OBSTACLE_SIZE, TAU_MAX,
                    TIMING_WARMUP_STEPS, W_ACT, W_POS, W_TERM)
from dynamics.ur5_dynamics import UR5Dynamics
from kinematics.ur5_kinematics import fk_batch, ik_np, ik_trajectory_np
from observers.dob import TorchDisturbanceObserver
from utils.disturbance import sinusoidal_disturbance

SQP_HORIZON = HORIZON
SQP_MAX_ITER = 5
SQP_W_POS = W_POS
SQP_W_VEL = 2_000.0
SQP_W_ACT = W_ACT
SQP_W_TERM = W_TERM
SQP_MIN_OBSTACLE_SLOTS = 2

# SQP-specific obstacle barriers.  Keeping these local is intentional: MPPI
# uses sampling-specific barrier gains, while Gauss-Newton SQP needs a steep
# residual before the warm-started trajectory reaches an obstacle.
SQP_STATIC_EXP_WEIGHT = 100.0
SQP_STATIC_EXP_GAIN = 500.0
SQP_STATIC_MARGIN = 0.001
SQP_STATIC_ZONE = 0.004
SQP_STATIC_GUIDANCE_OFFSET = 0.012
SQP_STATIC_GUIDANCE_HALF_WIDTH = 0.020
SQP_CROSS_EXP_WEIGHT = 300.0
SQP_CROSS_EXP_GAIN = 55.0
SQP_CROSS_EXP_ZONE = 0.015

_ROOT = Path(__file__).resolve().parents[1]
_ACADOS_OUT = _ROOT / "c_generated_code_ur5_sqp"
_ACADOS_JSON = "ur5_sqp_acados_ocp.json"


def _ca_skew(vector):
    x, y, z = vector[0], vector[1], vector[2]
    return ca.vertcat(
        ca.horzcat(0, -z, y),
        ca.horzcat(z, 0, -x),
        ca.horzcat(-y, x, 0),
    )


def _ca_motion_cross(motion, other):
    angular, linear = motion[:3], motion[3:]
    return ca.vertcat(
        ca.cross(angular, other[:3]),
        ca.cross(angular, other[3:]) + ca.cross(linear, other[:3]),
    )


def _ca_force_cross(motion, force):
    angular, linear = motion[:3], motion[3:]
    return ca.vertcat(
        ca.cross(angular, force[:3]) + ca.cross(linear, force[3:]),
        ca.cross(angular, force[3:]),
    )


def _ca_dh_motion_transform(theta, a: float, d: float, alpha: float):
    ct, st = ca.cos(theta), ca.sin(theta)
    caa, saa = math.cos(alpha), math.sin(alpha)
    rotation = ca.vertcat(
        ca.horzcat(ct, -st * caa, st * saa),
        ca.horzcat(st, ct * caa, -ct * saa),
        ca.horzcat(0, saa, caa),
    )
    translation = ca.vertcat(a * ct, a * st, d)
    rotation_t = rotation.T
    zero = ca.SX.zeros(3, 3)
    return ca.vertcat(
        ca.horzcat(rotation_t, zero),
        ca.horzcat(-rotation_t @ _ca_skew(translation), rotation_t),
    )


def _spatial_inertias() -> list[ca.DM]:
    result = []
    for mass, com_values, diagonal in zip(LINK_MASSES, LINK_COM, LINK_INERTIA_DIAG):
        com = np.asarray(com_values, dtype=float)
        cross = np.array([[0, -com[2], com[1]], [com[2], 0, -com[0]],
                          [-com[1], com[0], 0]], dtype=float)
        inertia_com = np.diag(diagonal)
        result.append(ca.DM(np.block([
            [inertia_com + mass * cross @ cross.T, mass * cross],
            [mass * cross.T, mass * np.eye(3)],
        ])))
    return result


_SPATIAL_INERTIAS = _spatial_inertias()
_JOINT_AXES = []
for _a, _d, _alpha, _offset in DH_PARAMS:
    _sa, _ca = math.sin(_alpha), math.cos(_alpha)
    _JOINT_AXES.append(ca.DM([0, _sa, _ca, 0, _a * _ca, -_a * _sa]))


def _ca_inverse_dynamics(q, velocity, acceleration, gravity_enabled: bool):
    parent_velocity = ca.SX.zeros(6, 1)
    gravity = ca.DM(GRAVITY)
    parent_acceleration = ca.vertcat(ca.SX.zeros(3, 1), -gravity) \
        if gravity_enabled else ca.SX.zeros(6, 1)
    transforms, forces = [], []
    for index, (a, d, alpha, offset) in enumerate(DH_PARAMS):
        transform = _ca_dh_motion_transform(q[index] + offset, a, d, alpha)
        axis = _JOINT_AXES[index]
        joint_velocity = axis * velocity[index]
        link_velocity = transform @ parent_velocity + joint_velocity
        link_acceleration = (transform @ parent_acceleration
                             + axis * acceleration[index]
                             + _ca_motion_cross(link_velocity, joint_velocity))
        momentum = _SPATIAL_INERTIAS[index] @ link_velocity
        force = (_SPATIAL_INERTIAS[index] @ link_acceleration
                 + _ca_force_cross(link_velocity, momentum))
        transforms.append(transform)
        forces.append(force)
        parent_velocity, parent_acceleration = link_velocity, link_acceleration

    torque = ca.SX.zeros(6, 1)
    accumulated = forces[-1]
    for index in range(5, -1, -1):
        if index < 5:
            accumulated = forces[index] + transforms[index + 1].T @ accumulated
        torque[index] = ca.dot(_JOINT_AXES[index], accumulated)
    return torque


def _ca_mass_matrix_and_bias(q, velocity):
    zero = ca.SX.zeros(6, 1)
    bias = _ca_inverse_dynamics(q, velocity, zero, gravity_enabled=True)
    columns = []
    for index in range(6):
        basis = ca.SX.zeros(6, 1)
        basis[index] = 1
        columns.append(_ca_inverse_dynamics(q, zero, basis, gravity_enabled=False))
    mass_matrix = ca.horzcat(*columns)
    mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
    mass_matrix += ca.diag(ca.DM(JOINT_ARMATURE))
    mass_matrix += MASS_MATRIX_REGULARIZATION * ca.SX.eye(6)
    return mass_matrix, bias


def _casadi_fk(q):
    transform = ca.SX.eye(4)
    for index, (a, d, alpha, offset) in enumerate(DH_PARAMS):
        theta = q[index] + offset
        ct, st = ca.cos(theta), ca.sin(theta)
        caa, saa = math.cos(alpha), math.sin(alpha)
        local = ca.vertcat(
            ca.horzcat(ct, -st*caa, st*saa, a*ct),
            ca.horzcat(st, ct*caa, -ct*saa, a*st),
            ca.horzcat(0, saa, caa, d),
            ca.horzcat(0, 0, 0, 1),
        )
        transform = transform @ local
    return transform[:3, 3]


def _sphere_obstacle_residual(ee, center, enabled):
    distance = ca.sqrt(ca.sumsqr(ee - center) + 1e-8)
    outer_radius = CROSS_RADIUS + CROSS_MARGIN + SQP_CROSS_EXP_ZONE
    penetration = ca.fmax(outer_radius - distance, 0)
    return (enabled * math.sqrt(SQP_CROSS_EXP_WEIGHT)
            * (ca.exp(0.5 * SQP_CROSS_EXP_GAIN * penetration) - 1))


def _box_obstacle_residual(ee, center, enabled):
    outside = ca.fmax(ca.fabs(ee - center) - STATIC_OBSTACLE_SIZE / 2.0, 0)
    distance = ca.sqrt(ca.sumsqr(outside) + 1e-8)
    penetration = ca.fmax(SQP_STATIC_MARGIN + SQP_STATIC_ZONE - distance, 0)
    return (enabled * math.sqrt(SQP_STATIC_EXP_WEIGHT)
            * (ca.exp(0.5 * SQP_STATIC_EXP_GAIN * penetration) - 1))


def _static_avoidance_reference(waypoints: np.ndarray,
                                centers: np.ndarray) -> np.ndarray:
    """Add smooth lateral seeds that break SQP's head-on obstacle symmetry.

    A barrier alone has no lateral gradient on a path through a cube center;
    local SQP can therefore settle at a stop in front of the cube.  The bump is
    used only as the optimizer reference.  Tracking metrics remain relative to
    the requested, unmodified trajectory.
    """
    guided = np.asarray(waypoints, dtype=float).copy()
    segment_length = np.linalg.norm(np.diff(guided, axis=0), axis=1)
    arc_length = np.concatenate(([0.0], np.cumsum(segment_length)))
    for center in np.asarray(centers):
        obstacle_index = int(np.argmin(np.linalg.norm(guided - center, axis=1)))
        before = max(0, obstacle_index - 2)
        after = min(len(guided) - 1, obstacle_index + 2)
        tangent = guided[after] - guided[before]
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-9:
            continue
        tangent /= tangent_norm
        normal = np.cross(tangent, np.array([0.0, 0.0, 1.0]))
        if np.linalg.norm(normal) < 1e-6:
            normal = np.cross(tangent, np.array([1.0, 0.0, 0.0]))
        normal /= np.linalg.norm(normal)
        distance = np.abs(arc_length - arc_length[obstacle_index])
        active = distance < SQP_STATIC_GUIDANCE_HALF_WIDTH
        phase = distance[active] / SQP_STATIC_GUIDANCE_HALF_WIDTH
        bump = SQP_STATIC_GUIDANCE_OFFSET * 0.5 * (1.0 + np.cos(np.pi * phase))
        guided[active] += bump[:, None] * normal
    return guided


def build_ur5_acados_ocp(
        x0: np.ndarray, obstacle_slots: int = SQP_MIN_OBSTACLE_SLOTS) -> AcadosOcp:
    q = ca.SX.sym("q", 6)
    velocity = ca.SX.sym("dq", 6)
    state = ca.vertcat(q, velocity)
    torque = ca.SX.sym("tau", 6)
    state_dot = ca.SX.sym("xdot", 12)
    # Each slot contains XYZ center, dynamic-sphere enable, static-box enable.
    obstacle_parameters = ca.SX.sym("obstacles", 5 * obstacle_slots)

    mass_matrix, bias = _ca_mass_matrix_and_bias(q, velocity)
    acceleration = ca.solve(
        mass_matrix,
        torque - bias - ca.DM(B_DAMP) * velocity,
    )
    dynamics = ca.vertcat(velocity, acceleration)
    ee = _casadi_fk(q)
    jacobian = ca.jacobian(ee, q)
    ee_velocity = jacobian @ velocity
    obstacle_terms = []
    for slot in range(obstacle_slots):
        offset = 5 * slot
        center = obstacle_parameters[offset:offset + 3]
        obstacle_terms.append(
            _sphere_obstacle_residual(ee, center, obstacle_parameters[offset + 3])
            + _box_obstacle_residual(ee, center, obstacle_parameters[offset + 4])
        )
    obstacle_terms = ca.vertcat(*obstacle_terms)

    model = AcadosModel()
    model.name = "ur5_nonlinear_sqp_obstacle_modes"
    model.x, model.u, model.xdot = state, torque, state_dot
    model.p = obstacle_parameters
    model.f_expl_expr = dynamics
    model.f_impl_expr = state_dot - dynamics
    model.cost_y_expr = ca.vertcat(ee, ee_velocity, torque, obstacle_terms)
    model.cost_y_expr_e = ca.vertcat(ee, obstacle_terms)

    ocp = AcadosOcp()
    ocp.model = model
    ocp.solver_options.N_horizon = SQP_HORIZON
    ocp.solver_options.tf = SQP_HORIZON * DT
    ocp.parameter_values = np.zeros(5 * obstacle_slots)

    ocp.cost.cost_type = "NONLINEAR_LS"
    ocp.cost.cost_type_e = "NONLINEAR_LS"
    ocp.cost.W = np.diag(
        [SQP_W_POS] * 3 + [SQP_W_VEL] * 3 + [SQP_W_ACT] * 6
        + [1.0] * obstacle_slots
    )
    ocp.cost.W_e = np.diag(
        [SQP_W_TERM] * 3 + [1.0] * obstacle_slots
    )
    ocp.cost.yref = np.zeros(12 + obstacle_slots)
    ocp.cost.yref_e = np.zeros(3 + obstacle_slots)

    ocp.constraints.idxbx_0 = np.arange(12)
    # Stage-0 bounds are replaced with the measured state before every solve.
    # Use a canonical generated value so a different trajectory start does not
    # invalidate otherwise identical acados generated code.
    canonical_x0 = np.zeros_like(x0)
    ocp.constraints.lbx_0 = canonical_x0
    ocp.constraints.ubx_0 = canonical_x0
    ocp.constraints.idxbu = np.arange(6)
    ocp.constraints.lbu = -np.asarray(TAU_MAX)
    ocp.constraints.ubu = np.asarray(TAU_MAX)
    lower = np.full(12, -1e9)
    upper = np.full(12, 1e9)
    lower[6:], upper[6:] = -DQ_MAX, DQ_MAX
    ocp.constraints.idxbx = np.arange(12)
    ocp.constraints.lbx, ocp.constraints.ubx = lower, upper
    ocp.constraints.idxbx_e = np.arange(12)
    ocp.constraints.lbx_e, ocp.constraints.ubx_e = lower, upper

    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 1
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.nlp_solver_max_iter = SQP_MAX_ITER
    ocp.solver_options.tol = 1e-4
    ocp.solver_options.qp_tol = 1e-4
    ocp.solver_options.print_level = 0
    ocp.code_gen_options.code_export_directory = str(_ACADOS_OUT)
    ocp.code_gen_options.json_file = str(_ACADOS_OUT / _ACADOS_JSON)
    return ocp


class UR5SQPTracker:
    def __init__(self, waypoints: np.ndarray,
                 dynamic_paths: np.ndarray | None = None,
                 static_centers: np.ndarray | None = None):
        if dynamic_paths is not None and static_centers is not None:
            raise ValueError("dynamic and static obstacles are mutually exclusive")
        self.original_count = len(waypoints)
        optimization_waypoints = (
            _static_avoidance_reference(waypoints, static_centers)
            if static_centers is not None else np.asarray(waypoints)
        )
        self.waypoints = np.vstack((np.asarray(waypoints), np.tile(
            waypoints[-1:], (SQP_HORIZON + 2, 1)
        )))
        self.optimization_waypoints = np.vstack((
            optimization_waypoints, np.tile(
                optimization_waypoints[-1:], (SQP_HORIZON + 2, 1)
            )
        ))
        reference_velocity = np.gradient(
            optimization_waypoints, DT, axis=0, edge_order=2
        )
        self.reference_velocity = np.vstack((reference_velocity, np.tile(
            reference_velocity[-1:], (SQP_HORIZON + 2, 1)
        )))
        self.dynamic_paths = dynamic_paths
        self.static_centers = static_centers
        requested_slots = (
            len(dynamic_paths) if dynamic_paths is not None else
            len(static_centers) if static_centers is not None else 0
        )
        self.obstacle_slots = max(SQP_MIN_OBSTACLE_SLOTS, requested_slots)
        self.obstacle_mode = (
            "dynamic" if dynamic_paths is not None else
            "static" if static_centers is not None else "false"
        )
        self.plant = UR5Dynamics(torch.device("cpu"), torch.float64)
        self.dob = TorchDisturbanceObserver(self.plant, DT)

    def _build_solver(self, initial_state: np.ndarray) -> AcadosOcpSolver:
        print("[SQP] acados nonlinear OCP 준비 중 (호환 캐시 자동 재사용)...")
        start = time.perf_counter()
        # A changed OCP makes acados_template read the previous cache JSON for
        # compatibility.  Different acados versions can leave harmless legacy
        # fields in that JSON and emit one warning per unknown field.  Limit
        # suppression to those cache-schema warnings; solver/build warnings
        # with any other message remain visible.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Attribute .* not in dictionary\.",
                category=UserWarning,
                module=r"acados_template(?:\..*)?",
            )
            solver = AcadosOcpSolver(
                build_ur5_acados_ocp(initial_state, self.obstacle_slots),
                generate=False,
                build=False,
                check_reuse_possible=True,
                verbose=False,
            )
        cache_status = "재생성/컴파일" if solver.generated else "캐시 재사용"
        print(f"[SQP] solver ready ({cache_status}): "
              f"{time.perf_counter() - start:.2f} s")
        return solver

    def _feedforward(self, q0: np.ndarray):
        joint = ik_trajectory_np(
            self.optimization_waypoints[:self.original_count], q0
        )
        velocity = np.gradient(joint, DT, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, DT, axis=0, edge_order=2)
        q_t = torch.as_tensor(joint, dtype=torch.float64)
        velocity_t = torch.as_tensor(velocity, dtype=torch.float64)
        acceleration_t = torch.as_tensor(acceleration, dtype=torch.float64)
        torque = self.plant.inverse_dynamics(q_t, velocity_t, acceleration_t)
        torque += self.plant.damping * velocity_t
        return joint, velocity, np.clip(torque.numpy(), -np.asarray(TAU_MAX), np.asarray(TAU_MAX))

    def _obstacle_parameter(self, simulation_step: int, horizon_step: int) -> np.ndarray:
        parameters = np.zeros(5 * self.obstacle_slots)
        if self.dynamic_paths is not None:
            path_index = min(
                simulation_step + horizon_step,
                self.dynamic_paths.shape[1] - 1,
            )
            for slot, path in enumerate(self.dynamic_paths):
                offset = 5 * slot
                parameters[offset:offset + 3] = path[path_index]
                parameters[offset + 3] = 1.0
        elif self.static_centers is not None:
            for slot, center in enumerate(self.static_centers):
                offset = 5 * slot
                parameters[offset:offset + 3] = center
                parameters[offset + 4] = 1.0
        return parameters

    def run(self, q0: np.ndarray, has_disturbance: bool = False,
            use_dob: bool = False) -> dict:
        state_np = np.concatenate((q0, np.zeros(6)))
        solver = self._build_solver(state_np)
        q_reference, velocity_reference, feedforward = self._feedforward(q0)
        for stage in range(SQP_HORIZON + 1):
            reference_index = min(stage, len(q_reference) - 1)
            solver.set(stage, "x", np.concatenate((
                q_reference[reference_index], velocity_reference[reference_index]
            )))
            if stage < SQP_HORIZON:
                solver.set(stage, "u", feedforward[reference_index])

        state = torch.as_tensor(state_np, dtype=torch.float64)
        self.dob.reset()
        q_history = [q0.copy()]
        ee_history = [fk_batch(state[:6].unsqueeze(0))[0].numpy()]
        control_history, error_history = [], []
        estimate_history = [np.zeros(6)]
        disturbance_history = [np.zeros(6)]
        residual_history = [np.zeros(6)]
        model_torque_history = [np.zeros(6)]
        solve_times_raw = []

        print(f"SQP simulation: {self.original_count} steps "
              f"(N={SQP_HORIZON}, max_iter={SQP_MAX_ITER}, "
              f"DOB={use_dob}, disturbance={has_disturbance}, "
              f"obstacle={self.obstacle_mode})")
        for simulation_step in range(self.original_count):
            state_np = state.numpy()
            solver.set(0, "lbx", state_np)
            solver.set(0, "ubx", state_np)
            for stage in range(SQP_HORIZON):
                reference_index = simulation_step + stage
                yref = np.concatenate((self.optimization_waypoints[reference_index],
                                       self.reference_velocity[reference_index],
                                       np.zeros(6 + self.obstacle_slots)))
                solver.set(stage, "yref", yref)
                solver.set(stage, "p", self._obstacle_parameter(simulation_step, stage))
            solver.set(SQP_HORIZON, "yref", np.concatenate((
                self.optimization_waypoints[simulation_step + SQP_HORIZON],
                np.zeros(self.obstacle_slots)
            )))
            solver.set(SQP_HORIZON, "p", self._obstacle_parameter(
                simulation_step, SQP_HORIZON
            ))

            start = time.perf_counter()
            status = solver.solve()
            solve_time = time.perf_counter() - start
            solve_times_raw.append(solve_time)
            if status not in (0, 2):
                print(f"  [WARN] acados status={status}, step={simulation_step}")
            sqp_torque = torch.as_tensor(solver.get(0, "u"), dtype=torch.float64)
            commanded = sqp_torque + self.dob.d_hat if use_dob else sqp_torque
            applied = commanded.clamp(self.plant.u_min, self.plant.u_max)
            true_disturbance = sinusoidal_disturbance(
                simulation_step, torch.device("cpu"), torch.float64
            ) if has_disturbance else torch.zeros(6, dtype=torch.float64)
            effective = (applied - true_disturbance).clamp(
                self.plant.u_min, self.plant.u_max
            )
            previous = state.clone()
            state = self.plant.plant_step(
                state.unsqueeze(0), effective.unsqueeze(0)
            )[0]
            if use_dob:
                self.dob.update(applied, previous[:6], previous[6:],
                                state[:6], state[6:])

            for stage in range(SQP_HORIZON):
                solver.set(stage, "x", solver.get(stage + 1, "x"))
                if stage < SQP_HORIZON - 1:
                    solver.set(stage, "u", solver.get(stage + 1, "u"))

            ee = fk_batch(state[:6].unsqueeze(0))[0]
            error_mm = torch.linalg.vector_norm(
                ee - torch.as_tensor(self.waypoints[simulation_step + 1])
            ).item() * 1e3
            q_history.append(state[:6].numpy().copy())
            ee_history.append(ee.numpy().copy())
            control_history.append(applied.numpy().copy())
            error_history.append(error_mm)
            estimate_history.append(self.dob.d_hat.numpy().copy())
            disturbance_history.append(true_disturbance.numpy().copy())
            residual_history.append(self.dob.residual.numpy().copy())
            model_torque_history.append(self.dob.tau_model.numpy().copy())
            if (simulation_step + 1) % 50 == 0:
                print(f"  step {simulation_step + 1:>3}/{self.original_count}  "
                      f"EE err={error_mm:.1f} mm  solve={solve_time*1e3:.2f} ms")

        solve_times_raw = np.asarray(solve_times_raw)
        solve_times = solve_times_raw.copy()
        solve_times[:min(TIMING_WARMUP_STEPS, len(solve_times))] = np.nan
        return {
            "q": np.asarray(q_history), "ee": np.asarray(ee_history),
            "u": np.asarray(control_history),
            "err_mm": np.asarray(error_history),
            "err": np.asarray(error_history) / 1e3,
            "d_hat": np.asarray(estimate_history),
            "d_true": np.asarray(disturbance_history),
            "dob_residual": np.asarray(residual_history),
            "dob_tau_model": np.asarray(model_torque_history),
            "solve_t": solve_times, "solve_t_raw": solve_times_raw,
            "elapsed": np.asarray(np.sum(solve_times_raw)),
            "has_disturbance": np.asarray(has_disturbance),
            "use_dob": np.asarray(use_dob),
            "use_obstacle": np.asarray(self.obstacle_mode != "false"),
            "obstacle_mode": np.asarray(self.obstacle_mode),
            "controller": np.asarray("sqp"),
            "dynamics_model": np.asarray("M(q)qdd+C(q,qd)qd+g(q)+Bqd=tau"),
            "sqp_static_exp_weight": np.asarray(SQP_STATIC_EXP_WEIGHT),
            "sqp_static_exp_gain": np.asarray(SQP_STATIC_EXP_GAIN),
            "sqp_static_margin": np.asarray(SQP_STATIC_MARGIN),
            "sqp_static_zone": np.asarray(SQP_STATIC_ZONE),
            "sqp_cross_exp_weight": np.asarray(SQP_CROSS_EXP_WEIGHT),
            "sqp_cross_exp_gain": np.asarray(SQP_CROSS_EXP_GAIN),
            "sqp_cross_exp_zone": np.asarray(SQP_CROSS_EXP_ZONE),
            **({"sqp_guidance_waypoints": self.optimization_waypoints[
                :self.original_count
            ]} if self.static_centers is not None else {}),
            **({"cross_paths": self.dynamic_paths} if self.dynamic_paths is not None else {}),
            **({"static_obstacle_centers": self.static_centers}
               if self.static_centers is not None else {}),
        }


def run_sqp(reference: np.ndarray, has_disturbance: bool = False,
            use_dob: bool = False,
            dynamic_paths: np.ndarray | None = None,
            static_centers: np.ndarray | None = None) -> dict:
    hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    q0 = ik_np(reference[0], hint)
    return UR5SQPTracker(
        reference, dynamic_paths=dynamic_paths, static_centers=static_centers
    ).run(
        q0, has_disturbance=has_disturbance, use_dob=use_dob
    )
