"""Symbolic scalar form of the existing RNEA, mass matrix and kinematics.

The equations of UR5Dynamics are traced once on symbolic scalars. Structural
zeros fold away and shared terms are computed once, so a batched forward
dynamics step becomes straight-line arithmetic inside one GPU kernel instead
of seven RNEA passes of 6x6 matrix products. Nothing is approximated: the
only differences from UR5Dynamics are float32 rounding and dropping constants
below 1e-12 (cos(pi/2) = 6e-17).
"""
from __future__ import annotations

import math

from config import DH_PARAMS

ZERO = 1e-12


class Graph:
    """Expression DAG with constant folding and common-subexpression reuse."""

    def __init__(self):
        self.nodes = []
        self.known = {}

    def node(self, operation, *arguments) -> "Scalar":
        key = (operation, arguments)
        if key not in self.known:
            self.known[key] = len(self.nodes)
            self.nodes.append(key)
        return Scalar(self, self.known[key])

    def input(self, name: str) -> "Scalar":
        return self.node("input", name)

    def emit(self, assignments: dict[str, "Scalar | float"],
             functions: str = "tl") -> list[str]:
        """Python/Triton statements that evaluate every named output."""
        needed = set()
        pending = [v.index for v in assignments.values() if isinstance(v, Scalar)]
        while pending:
            index = pending.pop()
            if index in needed:
                continue
            needed.add(index)
            operation, arguments = self.nodes[index]
            if operation != "input":
                pending += [value for kind, value in arguments if kind == "node"]

        def text(argument):
            kind, value = argument
            if kind == "const":
                return f"({value!r})"
            operation, arguments = self.nodes[value]
            return arguments[0] if operation == "input" else f"v{value}"

        lines = []
        for index in sorted(needed):
            operation, arguments = self.nodes[index]
            if operation == "input":
                continue
            operands = [text(a) for a in arguments]
            if operation in "+-*/":
                expression = f" {operation} ".join(operands)
            elif operation == "neg":
                expression = f"-{operands[0]}"
            elif operation == "relu":
                expression = f"{functions}.maximum({operands[0]}, 0.0)"
            else:
                expression = f"{functions}.{operation}({operands[0]})"
            lines.append(f"v{index} = {expression}")
        for name, value in assignments.items():
            lines.append(f"{name} = {text(_argument(value))}")
        return lines


def _argument(value):
    return (("node", value.index) if isinstance(value, Scalar)
            else ("const", float(value)))


class Scalar:
    __slots__ = ("graph", "index")

    def __init__(self, graph: Graph, index: int):
        self.graph, self.index = graph, index

    def __add__(self, other):
        if not isinstance(other, Scalar) and abs(other) < ZERO:
            return self
        return self.graph.node("+", *sorted((_argument(self), _argument(other))))

    __radd__ = __add__

    def __sub__(self, other):
        if not isinstance(other, Scalar) and abs(other) < ZERO:
            return self
        if isinstance(other, Scalar) and other.index == self.index:
            return 0.0
        return self.graph.node("-", _argument(self), _argument(other))

    def __rsub__(self, other):
        if abs(other) < ZERO:
            return -self
        return self.graph.node("-", _argument(other), _argument(self))

    def __mul__(self, other):
        if not isinstance(other, Scalar):
            if abs(other) < ZERO:
                return 0.0
            if abs(other - 1.0) < ZERO:
                return self
            if abs(other + 1.0) < ZERO:
                return -self
        return self.graph.node("*", *sorted((_argument(self), _argument(other))))

    __rmul__ = __mul__

    def __truediv__(self, other):
        return self.graph.node("/", _argument(self), _argument(other))

    def __rtruediv__(self, other):
        if abs(other) < ZERO:
            return 0.0
        return self.graph.node("/", _argument(other), _argument(self))

    def __neg__(self):
        operation, arguments = self.graph.nodes[self.index]
        if operation == "neg":
            return Scalar(self.graph, arguments[0][1])
        return self.graph.node("neg", _argument(self))


def unary(operation: str, value):
    if isinstance(value, Scalar):
        return value.graph.node(operation, _argument(value))
    if operation == "relu":
        return max(value, 0.0)
    return getattr(math, operation)(value)


def total(values):
    result = 0.0
    for value in values:
        result = result + value
    return result


def dot(left, right):
    return total(a * b for a, b in zip(left, right))


def cross(a, b):
    return [a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]]


def add(*vectors):
    return [total(values) for values in zip(*vectors)]


def matmul(left, right):
    columns = list(zip(*right))
    return [[dot(row, column) for column in columns] for row in left]


class RobotModel:
    """Constants read from an existing (optionally payload) UR5Dynamics."""

    def __init__(self, dynamics):
        self.inertia = dynamics.spatial_inertia.cpu().tolist()
        self.axes = dynamics.joint_axes.cpu().tolist()
        self.gravity = dynamics.gravity.cpu().tolist()
        self.armature = dynamics.armature.cpu().tolist()
        self.damping = dynamics.damping.cpu().tolist()
        self.regularization = float(dynamics.mass_regularization)
        self.dt = float(dynamics.dt)

    @staticmethod
    def link_transforms(q):
        """Parent-to-child motion transforms, as in _dh_motion_transform."""
        result = []
        for angle, (a, d, alpha, offset) in zip(q, DH_PARAMS):
            c, s = unary("cos", angle + offset), unary("sin", angle + offset)
            ca, sa = math.cos(alpha), math.sin(alpha)
            transpose = [[c, s, 0.0],
                         [-s*ca, c*ca, sa],
                         [s*sa, -c*sa, ca]]
            x, y, z = a*c, a*s, d
            skew = [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
            lower = matmul(transpose, skew)
            result.append([row + [0.0]*3 for row in transpose]
                          + [[-1.0*v for v in row] + transpose[i]
                             for i, row in enumerate(lower)])
        return result

    def inverse_dynamics(self, transforms, velocity, acceleration, base):
        parent_velocity, parent_acceleration = [0.0]*6, list(base)
        forces = []
        for i, (x, axis) in enumerate(zip(transforms, self.axes)):
            joint = [s*velocity[i] for s in axis]
            v = add([dot(row, parent_velocity) for row in x], joint)
            motion_cross = (cross(v[:3], joint[:3])
                            + add(cross(v[:3], joint[3:]), cross(v[3:], joint[:3])))
            a = add([dot(row, parent_acceleration) for row in x],
                    [s*acceleration[i] for s in axis], motion_cross)
            momentum = [dot(row, v) for row in self.inertia[i]]
            force_cross = (add(cross(v[:3], momentum[:3]), cross(v[3:], momentum[3:]))
                           + cross(v[:3], momentum[3:]))
            forces.append(add([dot(row, a) for row in self.inertia[i]], force_cross))
            parent_velocity, parent_acceleration = v, a
        torque = [0.0]*6
        accumulated = forces[-1]
        for i in range(5, -1, -1):
            if i < 5:
                accumulated = add(forces[i], [dot(column, accumulated)
                                              for column in zip(*transforms[i+1])])
            torque[i] = dot(accumulated, self.axes[i])
        return torque

    def mass_and_bias(self, q, velocity):
        x = self.link_transforms(q)
        bias = self.inverse_dynamics(
            x, velocity, [0.0]*6, [0.0]*3 + [-g for g in self.gravity])
        columns = [self.inverse_dynamics(
            x, [0.0]*6, [float(i == j) for i in range(6)], [0.0]*6) for j in range(6)]
        mass = [[0.0]*6 for _ in range(6)]
        for i in range(6):
            for j in range(i, 6):
                value = (columns[i][i] + self.armature[i] + self.regularization if i == j
                         else 0.5*(columns[j][i] + columns[i][j]))
                mass[i][j] = mass[j][i] = value
        return mass, bias

    def semi_implicit_step(self, q, velocity, torque):
        """Same transition as UR5Dynamics.rollout_effective."""
        mass, bias = self.mass_and_bias(q, velocity)
        acceleration = solve_spd(mass, [
            u - h - b*v for u, h, b, v in zip(torque, bias, self.damping, velocity)])
        next_velocity = [v + self.dt*a for v, a in zip(velocity, acceleration)]
        next_q = [p + self.dt*v for p, v in zip(q, next_velocity)]
        return next_q, next_velocity


def solve_spd(matrix, rhs):
    """LDL^T without pivoting; the regularized mass matrix is SPD."""
    n = len(rhs)
    lower = [[0.0]*n for _ in range(n)]
    diagonal, inverse, y = [], [], []
    for i in range(n):
        for j in range(i):
            lower[i][j] = (matrix[i][j] - total(
                lower[i][k]*lower[j][k]*diagonal[k] for k in range(j))) * inverse[j]
        pivot = matrix[i][i] - total(lower[i][k]*lower[i][k]*diagonal[k] for k in range(i))
        diagonal.append(pivot)
        inverse.append(1.0/pivot)
        y.append(rhs[i] - total(lower[i][k]*y[k] for k in range(i)))
    x = [0.0]*n
    for i in range(n-1, -1, -1):
        x[i] = y[i]*inverse[i] - total(lower[k][i]*x[k] for k in range(i+1, n))
    return x


def flange_pose(q):
    """Rotation and position of the last standard-DH frame, as in fk_batch."""
    rotation = [[float(i == j) for j in range(3)] for i in range(3)]
    position = [0.0]*3
    for angle, (a, d, alpha, offset) in zip(q, DH_PARAMS):
        c, s = unary("cos", angle + offset), unary("sin", angle + offset)
        ca, sa = math.cos(alpha), math.sin(alpha)
        position = add(position, [dot(row, [a*c, a*s, d]) for row in rotation])
        rotation = matmul(rotation, [[c, -s*ca, s*sa], [s, c*ca, -c*sa], [0.0, sa, ca]])
    return rotation, position


def mesh_frames(q, origins):
    """URDF link frames, as in PolytopeCollisionCost.world (base excluded)."""
    rotation = [[float(i == j) for j in range(3)] for i in range(3)]
    position, frames = [0.0]*3, []
    for angle, origin in zip(q, origins):
        position = add(position, [dot(row, [origin[3], origin[7], origin[11]])
                                  for row in rotation])
        fixed = matmul(rotation, [list(origin[0:3]), list(origin[4:7]), list(origin[8:11])])
        c, s = unary("cos", angle), unary("sin", angle)
        rotation = [[x*c + y*s, -x*s + y*c, z] for x, y, z in fixed]
        frames.append((rotation, position))
    return frames
