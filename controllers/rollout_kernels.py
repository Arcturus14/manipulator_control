"""Generated Triton kernels for the payload MPPI rollout, costs and DOB.

Equations, weights and indices are those of PayloadStabilizedTracker. The
speed-up comes from evaluation only:

- dynamics, kinematics and analytic costs are one kernel over K x H
- collision geometry is evaluated in link frames, without world vertices
- a bounding-box lower bound skips the vertices of a collision pair only
  when its penalty is exactly zero
- the terminal stage reuses the collision cost already computed for x(H)

Sources are generated for the robot/cost constants at warm-up and compiled by
Triton for the installed GPU, so no GTX-1650-specific binary is involved.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os

import torch
import triton

from config import W_ACT, W_VEL
from dynamics.scalar_codegen import (Graph, RobotModel, dot, flange_pose,
                                     mesh_frames, total, unary)
from utils.torch_runtime import configure_compilation_cache

HEADER = "import triton\nimport triton.language as tl\n\n\n"
BLOCK = 8        # samples per program
PLANES = 32      # 26 planes, padded by repetition
VERTICES = 64    # 48 vertices, padded by repetition
# Skip vertices only when the penalty is zero with room for float32 rounding.
BROAD_PHASE_SLACK = 1e-4


def _load_kernels(source: str, *names: str):
    """triton.jit reads source text, so generated code has to live in a file."""
    digest = hashlib.sha256(source.encode()).hexdigest()[:20]
    folder = configure_compilation_cache() / "kernels"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"mppi_{digest}.py"
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(source)
    os.replace(temporary, path)
    spec = importlib.util.spec_from_file_location(f"mppi_kernels_{digest}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [getattr(module, name) for name in names]


def _indent(lines, level=1):
    return "\n".join("    "*level + line for line in lines)


def _pad(tensor: torch.Tensor, size: int) -> torch.Tensor:
    """Repeat the first entry; leaves every min/max over the axis unchanged."""
    count = tensor.shape[1]
    if count > size:
        raise ValueError("Collision geometry exceeds the generated kernel size")
    return torch.cat((tensor, tensor[:, :1].expand(-1, size-count, *tensor.shape[2:])), 1)


def _rollout_source(tracker) -> str:
    cost, horizon = tracker.tracking_cost, tracker.horizon
    model = RobotModel(tracker.dynamics)
    graph = Graph()
    six = range(6)
    inputs = lambda prefix, count=6: [graph.input(f"{prefix}{i}") for i in range(count)]
    q, velocity, action = inputs("q"), inputs("w"), inputs("u")
    before, compensation = inputs("pu"), inputs("cp")
    reference, terminal_reference = inputs("r", 3), inputs("e", 3)
    posture, terminal_posture = inputs("g"), inputs("h")
    last, difference_weight = graph.input("last"), graph.input("edge")
    square = lambda value: value*value

    next_q, next_velocity = model.semi_implicit_step(q, velocity, action)
    rotation, position = flange_pose(next_q)
    center = [float(v) for v in tracker.tcp.center.cpu()]
    tcp = [p + dot(row, center) for p, row in zip(position, rotation)]

    values = lambda tensor: tensor.cpu().tolist()
    q_low, q_high = values(cost.q_soft_min), values(cost.q_soft_max)
    speed_low = values(-cost.dq_max + tracker.margins.margin[1, 0])
    speed_high = values(cost.dq_soft_max)
    low_scale, high_scale = values(cost.q_lower_margin), values(cost.q_upper_margin)
    speed_scale = values(cost.dq_margin.expand(6))
    q_weight = float(cost.joint_position_limit_weight)
    speed_weight = float(cost.joint_velocity_limit_weight)
    hinge = lambda value: square(unary("relu", value))
    limits = (
        q_weight*total(hinge((q_low[i] - next_q[i])*(1./low_scale[i]))
                       + hinge((next_q[i] - q_high[i])*(1./high_scale[i])) for i in six)
        + speed_weight*total(hinge((speed_low[i] - next_velocity[i])*(1./speed_scale[i]))
                             + hinge((next_velocity[i] - speed_high[i])*(1./speed_scale[i]))
                             for i in six))
    value = (float(cost.position_weight)*total(square(a - b) for a, b in zip(tcp, reference))
             + W_ACT*total(square(a) for a in action)
             + W_VEL*total(square(v) for v in next_velocity)
             + limits
             + difference_weight*total(square(a - b) for a, b in zip(action, before)))
    end = (float(cost.terminal_weight)*total(square(a - b)
                                             for a, b in zip(tcp, terminal_reference))
           + limits)
    weight = float(tracker.posture_weight)
    if weight:
        value = value + weight*total(square(a - b) for a, b in zip(next_q, posture))
        end = end + 3*weight*total(square(a - b) for a, b in zip(next_q, terminal_posture))
    if tracker.clipping_weight:
        low, high = values(tracker.dynamics.u_min), values(tracker.dynamics.u_max)
        requested = [a + c for a, c in zip(action, compensation)]
        value = value + float(tracker.clipping_weight)*total(
            hinge(r - high[i]) + hinge(low[i] - r) for i, r in enumerate(requested))

    outputs = {f"nq{i}": next_q[i] for i in six}
    outputs.update({f"nw{i}": next_velocity[i] for i in six})
    outputs["stage"] = value + last*end
    stores = [(f"STATES + at*12 + {i}", f"nq{i}") for i in six]
    stores += [(f"STATES + at*12 + {6+i}", f"nw{i}") for i in six]
    identity = [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.]
    stores += [(f"FRAMES + at*96 + {i}", repr(v)) for i, v in enumerate(identity)]
    # 8 poses per stage: base, six mesh links, then the flange.
    poses = (mesh_frames(next_q, tracker.collision_geometry_cost.origin_coefficients)
             + [(rotation, position)])
    for link, (matrix, translation) in enumerate(poses, start=1):
        for i, item in enumerate(matrix[0] + matrix[1] + matrix[2] + list(translation)):
            outputs[f"f{link}_{i}"] = item
            stores.append((f"FRAMES + at*96 + {12*link+i}", f"f{link}_{i}"))

    setup = [f"q{i} = zero + tl.load(X + {i})" for i in six]
    setup += [f"w{i} = zero + tl.load(X + {6+i})" for i in six]
    setup += [f"pu{i} = zero + tl.load(PREVIOUS + {i})" for i in six]
    setup += [f"cp{i} = tl.load(COMPENSATION + {i})" for i in six]
    setup += [f"e{i} = tl.load(REFERENCE + {horizon*3+i})" for i in range(3)]
    setup += [f"h{i} = tl.load(POSTURE + {horizon*6+i})" for i in six]

    offset = int(cost.reference_offset)
    body = [f"at = k*{horizon} + t",
            f"last = (t == {horizon-1}).to(tl.float32)",
            f"edge = tl.where(t == 0, {float(tracker.boundary_weight)!r}, "
            f"{float(cost.smooth_weight)/max(horizon-1, 1)!r})"]
    body += [f"u{i} = tl.load(ACTIONS + at*6 + {i}, mask=keep, other=0.0)" for i in six]
    body += [f"r{i} = tl.load(REFERENCE + (t + {offset})*3 + {i})" for i in range(3)]
    body += [f"g{i} = tl.load(POSTURE + (t + {offset})*6 + {i})" for i in six]
    body += graph.emit(outputs)
    body += [f"tl.store({address}, zero + ({item}), mask=keep)" for address, item in stores]
    body += ["cost = cost + stage"]
    body += [f"q{i} = zero + (nq{i})" for i in six]
    body += [f"w{i} = zero + (nw{i})" for i in six]
    body += [f"pu{i} = u{i}" for i in six]

    return f"""@triton.jit
def rollout(X, ACTIONS, REFERENCE, POSTURE, PREVIOUS, COMPENSATION,
            STATES, FRAMES, COST):
    k = tl.program_id(0)*{BLOCK} + tl.arange(0, {BLOCK})
    keep = k < {tracker.n_samples}
    zero = tl.zeros([{BLOCK}], dtype=tl.float32)
    cost = zero
{_indent(setup)}
    for t in range(0, {horizon}):
{_indent(body, 2)}
    tl.store(COST + k, cost, mask=keep)
"""


def _collision_source(tracker) -> str:
    robot, cube = tracker.collision_geometry_cost, tracker.cube_cost
    horizon = tracker.horizon
    threshold = repr(float(robot.clearance) + BROAD_PHASE_SLACK)
    cube_threshold = repr(float(cube.clearance) + BROAD_PHASE_SLACK)
    half = repr(float(cube.tcp.payload.side_m)/2)
    center = [float(v) for v in cube.tcp.center.cpu()]

    def pose(name, link):
        return ([f"{name}{i} = tl.load(FRAMES + at*96 + ({link})*12 + {i})" for i in range(9)]
                + [f"{name}t{i} = tl.load(FRAMES + at*96 + ({link})*12 + {9+i})"
                   for i in range(3)])

    def relative(left, right, origin):
        """A = left^T right and s = left^T (origin - left origin)."""
        lines = [f"a{r}{c} = " + " + ".join(f"{left}{3*m+r}*{right}{3*m+c}" for m in range(3))
                 for r in range(3) for c in range(3)]
        return lines + [f"s{r} = " + " + ".join(f"{left}{3*m+r}*({origin[m]} - {left}t{m})"
                                               for m in range(3))
                        for r in range(3)]

    def vector(name, base, link, size):
        return [f"{name}{i} = tl.load({base} + ({link})*{size*3} + index{size}*3 + {i})"
                for i in range(3)]

    pair = ["i = tl.load(I + p)", "j = tl.load(J + p)"]
    pair += pose("ri", "i") + pose("rj", "j") + relative("ri", "rj", ["rjt0", "rjt1", "rjt2"])
    pair += [f"back{r} = -(a0{r}*s0 + a1{r}*s1 + a2{r}*s2)" for r in range(3)]
    for side, source, target, entry, shift in (
            ("ab", "i", "j", lambda m, c: f"a{m}{c}", "s"),
            ("ba", "j", "i", lambda m, c: f"a{c}{m}", "back")):
        pair += vector(f"n{side}", "NORMALS", source, PLANES)
        pair += [f"offset_{side} = tl.load(OFFSETS + ({source})*{PLANES} + index{PLANES}) + "
                 + " + ".join(f"n{side}{m}*{shift}{m}" for m in range(3))]
        pair += [f"w{side}{c} = " + " + ".join(f"n{side}{m}*{entry(m, c)}" for m in range(3))
                 for c in range(3)]
        pair += [f"lower_{side} = tl.max(offset_{side} + "
                 + " + ".join(f"w{side}{c}*tl.load(BOX + ({target})*6 + {c})" for c in range(3))
                 + " - ("
                 + " + ".join(f"tl.abs(w{side}{c})*tl.load(BOX + ({target})*6 + {3+c})"
                              for c in range(3))
                 + "), axis=0)"]
    pair += [f"if tl.maximum(lower_ab, lower_ba) < {threshold}:"]
    for side, target in (("ab", "j"), ("ba", "i")):
        narrow = vector(f"v{side}", "VERTS", target, VERTICES)
        narrow += [f"signed_{side} = offset_{side}[:, None] + "
                   + " + ".join(f"w{side}{c}[:, None]*v{side}{c}[None, :]" for c in range(3)),
                   f"gap_{side} = tl.max(tl.min(signed_{side}, axis=1), axis=0)"]
        pair += ["    " + line for line in narrow]
    pair += [f"    excess = tl.maximum({float(robot.clearance)!r} - "
             "tl.maximum(gap_ab, gap_ba), 0.0)*20.0",
             "    robot_penalty = robot_penalty + excess*excess"]

    floor = pose("rl", "l")
    floor += ["lowest = rlt2 + "
              + " + ".join(f"rl{6+c}*tl.load(BOX + l*6 + {c})" for c in range(3)) + " - ("
              + " + ".join(f"tl.abs(rl{6+c})*tl.load(BOX + l*6 + {3+c})" for c in range(3))
              + ")",
              f"if lowest < {threshold}:"]
    narrow = vector("vf", "VERTS", "l", VERTICES)
    narrow += ["height = rlt2 + tl.min("
               + " + ".join(f"rl{6+c}*vf{c}" for c in range(3)) + ", axis=0)",
               f"excess = tl.maximum({float(robot.clearance)!r} - height, 0.0)*20.0",
               "robot_penalty = robot_penalty + excess*excess"]
    floor += ["    " + line for line in narrow]

    payload = pose("re", 7)
    payload += [f"c{m} = ret{m} + "
                + " + ".join(f"re{3*m+c}*({center[c]!r})" for c in range(3)) for m in range(3)]
    payload += [f"bottom = c2 - {half}*(tl.abs(re6) + tl.abs(re7) + tl.abs(re8))",
                f"excess = tl.maximum({float(cube.clearance)!r} - bottom, 0.0)*20.0",
                "cube_penalty = cube_penalty + excess*excess"]

    # Links 0..5: wrist3 carries the cube and is excluded, as in the reference.
    link = pose("rc", "l") + relative("rc", "re", ["c0", "c1", "c2"])
    link += vector("nc", "NORMALS", "l", PLANES)
    link += ["middle = " + " + ".join(f"nc{m}*s{m}" for m in range(3)),
             f"extent = {half}*("
             + " + ".join("tl.abs(" + " + ".join(f"nc{m}*a{m}{c}" for m in range(3)) + ")"
                          for c in range(3)) + ")",
             f"smallest = tl.load(RANGE + l*{2*PLANES} + index{PLANES})",
             f"largest = tl.load(RANGE + l*{2*PLANES} + {PLANES} + index{PLANES})",
             "separation = tl.max(tl.maximum(middle - extent - largest, "
             "smallest - middle - extent), axis=0)",
             f"if separation < {cube_threshold}:"]
    narrow = vector("vc", "VERTS", "l", VERTICES)
    for c in range(3):
        narrow += [f"projected{c} = " + " + ".join(f"a{m}{c}*vc{m}" for m in range(3)),
                   f"middle{c} = " + " + ".join(f"a{m}{c}*s{m}" for m in range(3)),
                   f"separation = tl.maximum(separation, tl.maximum("
                   f"middle{c} - {half} - tl.max(projected{c}, axis=0), "
                   f"tl.min(projected{c}, axis=0) - middle{c} - {half}))"]
    narrow += [f"excess = tl.maximum({float(cube.clearance)!r} - separation, 0.0)*20.0",
               "cube_penalty = cube_penalty + excess*excess"]
    link += ["    " + line for line in narrow]

    return f"""@triton.jit
def collision(FRAMES, VERTS, NORMALS, OFFSETS, BOX, RANGE, I, J, OUT):
    at = tl.program_id(0)
    index{PLANES} = tl.arange(0, {PLANES})
    index{VERTICES} = tl.arange(0, {VERTICES})
    robot_penalty = 0.0
    cube_penalty = 0.0
    for p in range(0, {len(robot.i)}):
{_indent(pair, 2)}
    for l in range(1, 7):
{_indent(floor, 2)}
{_indent(payload)}
    for l in range(0, 6):
{_indent(link, 2)}
    tl.store(OUT + at, {float(robot.weight)!r}*robot_penalty
             + {float(cube.weight)!r}*cube_penalty)


@triton.jit
def total_cost(ROLLOUT, COLLISION, COST):
    k = tl.program_id(0)
    t = tl.arange(0, 16)
    stages = tl.load(COLLISION + k*{horizon} + t, mask=t < {horizon}, other=0.0)
    # The terminal collision cost is that of the final stage: both see x(H).
    end = tl.load(COLLISION + k*{horizon} + {horizon-1})
    tl.store(COST + k, tl.load(ROLLOUT + k) + tl.sum(stages, axis=0) + end)
"""


def _observer_source(dob, dynamics) -> str:
    model = RobotModel(dynamics)
    graph = Graph()
    q, velocity, acceleration = ([graph.input(f"{prefix}{i}") for i in range(6)]
                                 for prefix in ("q", "w", "a"))
    torque = model.inverse_dynamics(model.link_transforms(q), velocity, acceleration,
                                    [0.0]*3 + [-g for g in model.gravity])
    outputs = {f"model{i}": (torque[i]
                             + (model.armature[i] + model.regularization)*acceleration[i]
                             + model.damping[i]*velocity[i]) for i in range(6)}
    lines = []
    for i in range(6):
        lines += [f"q{i} = 0.5*(tl.load(Q0 + {i}) + tl.load(Q1 + {i}))",
                  f"w{i} = 0.5*(tl.load(V0 + {i}) + tl.load(V1 + {i}))",
                  f"raw{i} = (tl.load(V1 + {i}) - tl.load(V0 + {i}))*{1./dob.dt!r}"]
        beta = dob.acceleration_lpf_retention
        lines.append(f"a{i} = raw{i}" if beta is None else
                     f"a{i} = {beta!r}*tl.load(FILTERED + {i}) + {1.-beta!r}*raw{i}")
    lines += graph.emit(outputs)
    for i in range(6):
        lines += [f"residual{i} = tl.load(APPLIED + {i}) - model{i}",
                  f"old{i} = tl.load(ESTIMATE + {i})"]
        beta = dob.exact_lpf_retention
        lines.append(f"new{i} = old{i} + {dob.dt*dob.alpha!r}*(residual{i} - old{i})"
                     if beta is None else
                     f"new{i} = {beta!r}*old{i} + {1.-beta!r}*residual{i}")
        for row, name in enumerate(("new", "residual", "model", "raw", "a")):
            lines.append(f"tl.store(OUT + {6*row+i}, {name}{i})")
    return f"""@triton.jit
def observe(APPLIED, Q0, V0, Q1, V1, ESTIMATE, FILTERED, OUT):
{_indent(lines)}
"""


class RolloutKernels:
    """Fixed-shape replacement for MPPI._compute_rollout_costs_single."""

    def __init__(self, tracker):
        robot, cube = tracker.collision_geometry_cost, tracker.cube_cost
        if tracker.device.type != "cuda" or tracker.dtype != torch.float32:
            raise ValueError("The kernel backend requires CUDA float32")
        if robot is None or cube is None:
            raise ValueError("The kernel backend requires robot and payload collision costs")
        if tracker.margins.config.mode != "fixed" or tracker.margins.config.penalty_kind != "hinge":
            raise ValueError("The kernel backend requires fixed hinge state margins")
        if tracker.study["fixed_clip"] or tracker.study["saturation_rollout"] or tracker.headroom_weight:
            raise ValueError("The kernel backend supports the final N0/N1/D0 equations only")
        if tracker.obstacle_cost is not None or not 1 <= tracker.horizon <= 16:
            raise ValueError("Unsupported obstacle or horizon for the kernel backend")
        self.tracker = tracker
        self.rollout, self.collision, self.total = _load_kernels(
            HEADER + _rollout_source(tracker) + "\n\n" + _collision_source(tracker),
            "rollout", "collision", "total_cost")
        self.vertices = _pad(robot.vertices, VERTICES).contiguous()
        self.normals = _pad(robot.normals, PLANES).contiguous()
        self.offsets = _pad(robot.offsets, PLANES).contiguous()
        low, high = robot.vertices.amin(1), robot.vertices.amax(1)
        # Slightly enlarged boxes keep the broad phase a true lower bound.
        self.box = torch.cat(((low+high)/2, (high-low)/2 + 1e-6), 1).contiguous()
        projection = self.normals @ robot.vertices.transpose(1, 2)
        self.range = torch.cat((projection.amin(-1), projection.amax(-1)), 1).contiguous()
        self.i, self.j = robot.i.int().contiguous(), robot.j.int().contiguous()
        samples, horizon = tracker.n_samples, tracker.horizon
        empty = lambda *shape: torch.empty(shape, device=tracker.device, dtype=tracker.dtype)
        self.states = empty(1, samples, horizon, 12)
        self.frames = empty(samples, horizon, 96)
        self.rollout_cost = empty(samples)
        self.collision_cost = empty(samples, horizon)
        self.cost = empty(samples)

    def __call__(self, actions: torch.Tensor):
        tracker = self.tracker
        samples, horizon = tracker.n_samples, tracker.horizon
        if actions.shape != (samples, horizon, 6) or tracker.ctrl.state.shape != (12,):
            raise ValueError("Kernel rollout shape changed after warm-up")
        actions = actions.contiguous()
        self.rollout[((samples + BLOCK-1)//BLOCK,)](
            tracker.ctrl.state.contiguous(), actions,
            tracker.tracking_cost.reference_window, tracker.posture_window,
            tracker.tracking_cost.previous_action, tracker._compensation_snapshot,
            self.states, self.frames, self.rollout_cost)
        self.collision[(samples*horizon,)](
            self.frames, self.vertices, self.normals, self.offsets,
            self.box, self.range, self.i, self.j, self.collision_cost, num_warps=1)
        self.total[(samples,)](self.rollout_cost, self.collision_cost, self.cost)
        return self.cost.clone(), self.states, actions[None]


class ObserverKernel:
    """Replacement for TorchDisturbanceObserver.update."""

    def __init__(self, dob, dynamics):
        self.dob = dob
        self.kernel, = _load_kernels(HEADER + _observer_source(dob, dynamics), "observe")

    def __call__(self, applied, q_previous, velocity_previous, q_current, velocity_current):
        dob = self.dob
        output = applied.new_empty((5, 6))
        self.kernel[(1,)](*(value.contiguous() for value in (
            applied, q_previous, velocity_previous, q_current, velocity_current,
            dob.d_hat, dob.filtered_acceleration)), output)
        (dob.d_hat, dob.residual, dob.tau_model,
         dob.raw_acceleration, dob.filtered_acceleration) = output.unbind(0)
        return dob.d_hat
