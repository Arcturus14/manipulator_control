"""Parse the user-facing ``config.txt`` execution settings."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

from config import DT
from utils.trajectory import TRAJECTORIES


CONFIG_KEYS = {
    "mode",
    "comparison_type",
    "controller",
    "use_dob",
    "use_disturbance",
    "use_obstacles",
    "t_sim",
    "static_obstacle_count",
    "trajectory",
}


def parse_boolean(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "yes", "y", "1", "on"}:
        return True
    if normalized in {"false", "no", "n", "0", "off"}:
        return False
    raise ValueError(f"{key} must be true/false, got: {value!r}")


def parse_obstacle_mode(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"dynamic", "static"}:
        return normalized
    if normalized in {"false", "none", "off", "no", "0"}:
        return "false"
    raise ValueError(
        "use_obstacles must be Dynamic, Static, or False, "
        f"got: {value!r}"
    )


def parse_simulation_steps(value: str) -> tuple[float, int]:
    try:
        duration = float(value)
    except ValueError as error:
        raise ValueError(f"t_sim must be a number, got: {value!r}") from error
    if duration <= 0 or not math.isfinite(duration):
        raise ValueError("t_sim must be a finite positive number")
    step_count = round(duration / DT)
    if step_count < 3:
        raise ValueError(f"t_sim must contain at least 3 control steps (DT={DT})")
    if not math.isclose(duration, step_count * DT, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"t_sim must be an integer multiple of DT={DT}")
    return duration, step_count


def parse_obstacle_count(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise ValueError(
            f"static_obstacle_count must be an integer, got: {value!r}"
        ) from error
    if count < 1:
        raise ValueError("static_obstacle_count must be at least 1")
    return count


def load_execution_config(path: Path) -> dict:
    """Load and strictly validate one runtime configuration file."""
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{line_number}: expected key = value")
        key, value = (part.strip() for part in line.split("=", 1))
        key = key.lower()
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        values[key] = value

    missing = CONFIG_KEYS - values.keys()
    unknown = values.keys() - CONFIG_KEYS
    if missing:
        raise ValueError(f"Missing config keys: {', '.join(sorted(missing))}")
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(sorted(unknown))}")

    mode = values["mode"].lower()
    comparison_type = values["comparison_type"].lower()
    controller = values["controller"].lower()
    if mode not in {"single", "comparison"}:
        raise ValueError("mode must be 'single' or 'comparison'")
    if comparison_type not in {"controller", "dob"}:
        raise ValueError("comparison_type must be 'controller' or 'dob'")
    if controller not in {"mppi", "sqp"}:
        raise ValueError("controller must be 'mppi' or 'sqp'")
    trajectory = values["trajectory"].lower()
    if trajectory not in TRAJECTORIES:
        available = ", ".join(sorted(TRAJECTORIES))
        raise ValueError(f"trajectory must be one of: {available}")

    t_sim, n_steps = parse_simulation_steps(values["t_sim"])
    static_obstacle_count = parse_obstacle_count(
        values["static_obstacle_count"]
    )
    if static_obstacle_count >= n_steps:
        raise ValueError(
            "static_obstacle_count must be smaller than the number of "
            f"simulation steps ({n_steps})"
        )
    return {
        "mode": mode,
        "comparison_type": comparison_type,
        "controller": controller,
        "trajectory": trajectory,
        "use_dob": parse_boolean(values["use_dob"], "use_dob"),
        "has_disturbance": parse_boolean(
            values["use_disturbance"], "use_disturbance"
        ),
        "obstacle_mode": parse_obstacle_mode(values["use_obstacles"]),
        "t_sim": t_sim,
        "n_steps": n_steps,
        "static_obstacle_count": static_obstacle_count,
    }


def configuration_name(controller: str, has_disturbance: bool, use_dob: bool,
                       obstacle_mode: str, trajectory: str) -> str:
    state = lambda enabled: "on" if enabled else "off"
    return (f"{controller}_dist_{state(has_disturbance)}_dob_{state(use_dob)}_"
            f"obs_{obstacle_mode}_{trajectory}_results.npz")


def _main() -> None:
    parser = argparse.ArgumentParser(description="Read an execution config field")
    parser.add_argument("config", type=Path)
    parser.add_argument("--field", choices=("mode",), required=True)
    args = parser.parse_args()
    print(load_execution_config(args.config.resolve())[args.field])


if __name__ == "__main__":
    _main()
