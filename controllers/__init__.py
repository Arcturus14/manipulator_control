"""Optimal controllers.

The SQP module is loaded lazily because importing acados/CasADi is unnecessary
for MPPI-only executions.
"""

from .mppi_dob_controller import UR5Tracker

__all__ = ["UR5Tracker", "UR5SQPTracker", "run_sqp"]


def __getattr__(name: str):
    if name in {"UR5SQPTracker", "run_sqp"}:
        from .sqp import UR5SQPTracker, run_sqp

        return {"UR5SQPTracker": UR5SQPTracker, "run_sqp": run_sqp}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
