"""Layered YAML configuration for the unified command-line entry point."""
from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
TRAJECTORY_NAMES = {"circle", "infinity", "rectangle", "complex_3d_wave", "multiscale_3d"}


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level YAML value must be a mapping")
    return data


def _merge(base: dict[str, Any], override: dict[str, Any], prefix: str = "") -> None:
    for key, value in override.items():
        name = f"{prefix}.{key}" if prefix else key
        if key not in base:
            raise ValueError(f"Unknown configuration key: {name}")
        if isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"{name} must be a mapping")
            _merge(base[key], value, name)
        else:
            base[key] = value


def load_config(path: Path | None, overrides: dict[str, Any]) -> dict[str, Any]:
    """Return defaults < YAML file < CLI overrides."""
    config = copy.deepcopy(_read_yaml(DEFAULT_CONFIG))
    selected = DEFAULT_CONFIG if path is None else path.resolve()
    if selected != DEFAULT_CONFIG.resolve():
        _merge(config, _read_yaml(selected))
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        target = config
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown configuration key: {dotted_key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration key: {dotted_key}")
        target[parts[-1]] = value
    validate_config(config)
    return config


def validate_config(cfg: dict[str, Any]) -> None:
    from numbers import Real

    dt = cfg["simulation"]["dt"]
    duration = cfg["experiment"]["duration"]
    if not isinstance(dt, Real) or dt <= 0:
        raise ValueError("simulation.dt must be positive")
    if not isinstance(duration, Real) or duration <= 0:
        raise ValueError("experiment.duration must be positive")
    steps = round(duration / dt)
    if steps < 3 or not math.isclose(duration, steps * dt, abs_tol=1e-9):
        raise ValueError("experiment.duration must be an integer multiple of simulation.dt")
    if cfg["controller"]["type"] not in {"mppi", "sqp"}:
        raise ValueError("controller.type must be mppi or sqp")
    if cfg["experiment"]["trajectory"] not in TRAJECTORY_NAMES:
        raise ValueError(f"experiment.trajectory must be one of: {', '.join(sorted(TRAJECTORY_NAMES))}")
    if cfg["controller"]["horizon"] < 1 or cfg["controller"]["samples"] < 1:
        raise ValueError("controller.horizon and controller.samples must be positive")
    if len(cfg["controller"]["noise_variance"]) != 6:
        raise ValueError("controller.noise_variance must contain six values")
    if cfg["obstacle"]["mode"] not in {"none", "static", "dynamic"}:
        raise ValueError("obstacle.mode must be none, static, or dynamic")
    if cfg["comparison"]["type"] not in {"controller", "dob"}:
        raise ValueError("comparison.type must be controller or dob")
    if cfg["benchmark"]["type"] not in {"baseline", "static", "dynamic"}:
        raise ValueError("benchmark.type must be baseline, static, or dynamic")
    unknown_trajectories = set(cfg["benchmark"]["trajectories"]) - TRAJECTORY_NAMES
    if unknown_trajectories:
        raise ValueError(f"Unknown benchmark trajectories: {', '.join(sorted(unknown_trajectories))}")
    if len(cfg["robot"]["torque_limit"]) != 6 or len(cfg["robot"]["damping"]) != 6:
        raise ValueError("robot torque_limit and damping must contain six values")
    if cfg["obstacle"]["count"] < 1:
        raise ValueError("obstacle.count must be positive")
    if cfg["obstacle"]["mode"] == "static" and cfg["obstacle"]["count"] >= steps:
        raise ValueError("obstacle.count must be smaller than the step count in static mode")


def apply_to_legacy_constants(cfg: dict[str, Any]) -> None:
    """Apply YAML values before importing modules that bind config constants."""
    import config as legacy

    robot, sim = cfg["robot"], cfg["simulation"]
    ctrl, costs = cfg["controller"], cfg["cost"]
    obs, experiment = cfg["obstacle"], cfg["experiment"]
    legacy.TAU_MAX = tuple(float(v) for v in robot["torque_limit"])
    legacy.DQ_MAX = float(robot["velocity_limit"])
    legacy.B_DAMP = tuple(float(v) for v in robot["damping"])
    legacy.DT = float(sim["dt"])
    legacy.T_SIM = float(experiment["duration"])
    legacy.N_STEPS = round(legacy.T_SIM / legacy.DT)
    legacy.HORIZON = legacy.HORIZON_OBS = int(ctrl["horizon"])
    legacy.N_SAMPLES = legacy.N_SAMPLES_OBS = int(ctrl["samples"])
    legacy.LAMBDA = float(ctrl["temperature"])
    legacy.MPPI_SEED = int(ctrl["seed"])
    legacy.NOISE_VAR = tuple(float(v) for v in ctrl["noise_variance"])
    legacy.SQP_MAX_ITER = int(ctrl["sqp_max_iterations"])

    mppi, sqp = costs["mppi"], costs["sqp"]
    legacy.MPPI_W_POS, legacy.MPPI_W_TERM = float(mppi["position"]), float(mppi["terminal"])
    legacy.W_VEL, legacy.W_ACT = float(mppi["velocity"]), float(mppi["action"])
    legacy.W_SMOOTH = float(mppi["smoothness"])
    legacy.W_SMOOTH_OBS = float(mppi["obstacle_smoothness"])
    legacy.SQP_W_POS, legacy.SQP_W_TERM = float(sqp["position"]), float(sqp["terminal"])
    legacy.SQP_W_VEL, legacy.SQP_W_ACT = float(sqp["velocity"]), float(sqp["action"])
    legacy.SQP_W_SMOOTH = float(sqp["smoothness"])
    legacy.SQP_W_SMOOTH_OBS = float(sqp["obstacle_smoothness"])

    static, dynamic = obs["static"], obs["dynamic"]
    legacy.STATIC_OBSTACLE_COUNT = int(obs["count"])
    legacy.STATIC_OBSTACLE_SIZE = float(static["size"])
    legacy.MPPI_STATIC_EXP_WEIGHT = float(static["mppi_weight"])
    legacy.SQP_STATIC_EXP_WEIGHT = float(static["sqp_weight"])
    legacy.MPPI_STATIC_EXP_GAIN = float(static["gain"])
    legacy.MPPI_STATIC_MARGIN = float(static["margin"])
    legacy.MPPI_STATIC_ZONE = float(static["zone"])
    legacy.DYNAMIC_OBSTACLE_SIZE = float(dynamic["size"])
    legacy.DYNAMIC_OBSTACLE_SPEED_RATIO = float(dynamic["speed_ratio"])
    legacy.DYNAMIC_OBSTACLE_PHASE_OFFSET = float(dynamic["phase_offset"])
    legacy.CROSS_EXP_W = float(dynamic["mppi_weight"])
    legacy.SQP_CROSS_EXP_WEIGHT = float(dynamic["sqp_weight"])
    legacy.CROSS_EXP_K = float(dynamic["gain"])
    legacy.CROSS_MARGIN = float(dynamic["margin"])
    legacy.CROSS_EXP_ZONE = float(dynamic["zone"])


def runtime_settings(cfg: dict[str, Any], command: str) -> dict[str, Any]:
    experiment, obstacle = cfg["experiment"], cfg["obstacle"]
    return {
        "mode": "single" if command == "run" else "comparison",
        "comparison_type": cfg["comparison"]["type"],
        "controller": cfg["controller"]["type"],
        "trajectory": experiment["trajectory"],
        "use_dob": bool(experiment["dob"]),
        "has_disturbance": bool(experiment["disturbance"]),
        "obstacle_mode": "false" if obstacle["mode"] == "none" else obstacle["mode"],
        "t_sim": float(experiment["duration"]),
        "n_steps": round(experiment["duration"] / cfg["simulation"]["dt"]),
        "static_obstacle_count": int(obstacle["count"]),
    }


def dump_config(cfg: dict[str, Any], path: Path) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
