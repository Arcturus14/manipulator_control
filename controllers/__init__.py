"""Optimal controllers.

Deterministic controllers are loaded lazily because importing their numerical
stacks is unnecessary for MPPI-only executions.
"""

from .mppi_dob_controller import UR5Tracker

__all__ = ["UR5Tracker", "UR5ILQRTracker", "run_ilqr",
           "UR5SQPTracker", "run_sqp"]


def __getattr__(name: str):
    if name in {"UR5ILQRTracker", "run_ilqr"}:
        from .ilqr import UR5ILQRTracker, run_ilqr

        return {"UR5ILQRTracker": UR5ILQRTracker, "run_ilqr": run_ilqr}[name]
    if name in {"UR5SQPTracker", "run_sqp"}:
        from .sqp import UR5SQPTracker, run_sqp

        return {"UR5SQPTracker": UR5SQPTracker, "run_sqp": run_sqp}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
