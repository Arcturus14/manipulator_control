"""Shared controller runners used by the unified CLI and benchmarks."""
from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import matplotlib
import numpy as np

# Configure the persistent workspace-local compilation cache before importing
# the compiled MPPI controller implementation.
from utils.torch_runtime import configure_compilation_cache

configure_compilation_cache()


class _SmallGpuTemplateWarningFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Not enough SMs to use max_autotune_gemm mode" not in record.getMessage()


logging.getLogger("torch._inductor.utils").addFilter(
    _SmallGpuTemplateWarningFilter()
)

import torch


def _prefer_matplotlib_toolkits() -> None:
    toolkit_dir = Path(matplotlib.__file__).resolve().parent.parent / "mpl_toolkits"
    if toolkit_dir.is_dir():
        import mpl_toolkits
        if str(toolkit_dir) not in mpl_toolkits.__path__:
            mpl_toolkits.__path__.insert(0, str(toolkit_dir))


_prefer_matplotlib_toolkits()
import matplotlib.pyplot as plt

from config import (CROSS_EXP_K, CROSS_EXP_W, CROSS_EXP_ZONE, CROSS_MARGIN,
                    DT, DYNAMIC_OBSTACLE_PHASE_OFFSET, DYNAMIC_OBSTACLE_SIZE,
                    DYNAMIC_OBSTACLE_SPEED_RATIO, HORIZON_OBS, LAMBDA, MPPI_SEED,
                    MPPI_MEAN_UPDATE_ALPHA,
                    MPPI_STATIC_EXP_GAIN, MPPI_STATIC_EXP_WEIGHT,
                    MPPI_STATIC_HORIZON, MPPI_STATIC_MARGIN,
                    MPPI_STATIC_SAMPLES, MPPI_STATIC_ZONE, MPPI_W_POS,
                    MPPI_W_TERM, NOISE_VAR, N_STEPS, TIMING_WARMUP_STEPS,
                    TRAJ_CENTER, TRAJ_RADIUS)
from controllers.mppi_dob_controller import UR5Tracker
from costs.obstacle_cost import ObstacleCost
from kinematics.ur5_kinematics import ik_np
from utils.trajectory import (TRAJECTORIES, chasing_obstacle_paths,
                              static_obstacle_centers)
from visualization.plotters import (LABELS, compute_metrics, plot_3d_trajectory,
                                    plot_disturbance_estimation_error,
                                    plot_joint_torques, plot_result_dashboard,
                                    plot_timeseries)

PLOT_FILES = {
    "trajectory": "final_trajectory.png",
    "timeseries": "timeseries.png",
    "dashboard": "metrics_dashboard.png",
    "torque": "joint_torques.png",
    "disturbance_error": "disturbance_estimation_error.png",
}


def save_plots(results: dict, reference: np.ndarray, *,
               dynamic_paths: np.ndarray | None = None,
               static_centers: np.ndarray | None = None,
               output_dir: Path,
               include_torque: bool = False,
               include_disturbance_error: bool = False) -> dict[str, plt.Figure]:
    """Save result figures in the run's timestamped output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "trajectory": plot_3d_trajectory(
            results, reference, dynamic_paths=dynamic_paths,
            static_centers=static_centers,
        ),
        "timeseries": plot_timeseries(results),
        "dashboard": plot_result_dashboard(results),
    }
    if include_torque:
        figures["torque"] = plot_joint_torques(results)
    if include_disturbance_error:
        figures["disturbance_error"] = plot_disturbance_estimation_error(results)
    for name, figure in figures.items():
        destination = output_dir / PLOT_FILES[name]
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        print(f"[PLOT] {destination}")
    return figures


def waypoints(name: str = "circle", n_steps: int = N_STEPS) -> np.ndarray:
    return TRAJECTORIES[name](n_steps, TRAJ_CENTER, TRAJ_RADIUS)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initial_configuration(reference: np.ndarray, target_device: torch.device) -> torch.Tensor:
    hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    return torch.tensor(ik_np(reference[0], hint), dtype=torch.float64, device=target_device)


def run_mppi(reference: np.ndarray, has_disturbance: bool = True,
             use_dob: bool = False,
             dynamic_paths: np.ndarray | None = None,
             static_centers: np.ndarray | None = None, *,
             seed: int = MPPI_SEED,
             n_samples: int | None = None,
             horizon: int | None = None,
             temperature: float = LAMBDA,
             mean_update_alpha: float = MPPI_MEAN_UPDATE_ALPHA,
             noise_variance: tuple[float, ...] | list[float] = NOISE_VAR,
             position_weight: float = MPPI_W_POS,
             terminal_weight: float = MPPI_W_TERM,
             smooth_weight: float | None = None,
             capture_samples: bool = False,
             sample_count: int = 30,
             static_exp_weight: float = MPPI_STATIC_EXP_WEIGHT,
             static_exp_gain: float = MPPI_STATIC_EXP_GAIN,
             static_margin: float = MPPI_STATIC_MARGIN,
             static_zone: float = MPPI_STATIC_ZONE,
             dynamic_exp_weight: float = CROSS_EXP_W,
             dynamic_exp_gain: float = CROSS_EXP_K,
             dynamic_margin: float = CROSS_MARGIN,
             dynamic_zone: float = CROSS_EXP_ZONE) -> dict:
    # MPPI is stochastic. A fixed seed makes controller tuning comparisons
    # reproducible while leaving the sampled-control algorithm unchanged.
    torch.manual_seed(seed)
    target_device = device()
    simulation_dtype = torch.float32
    obstacle = None
    default_horizon = (MPPI_STATIC_HORIZON
                       if static_centers is not None else HORIZON_OBS)
    controller_horizon = default_horizon if horizon is None else horizon
    controller_samples = (MPPI_STATIC_SAMPLES
                          if static_centers is not None and n_samples is None
                          else n_samples)
    if dynamic_paths is not None or static_centers is not None:
        obstacle = ObstacleCost(
            controller_horizon, target_device, simulation_dtype,
            dynamic_paths=dynamic_paths,
            static_centers=static_centers,
            static_exp_weight=static_exp_weight,
            static_exp_gain=static_exp_gain,
            static_margin=static_margin,
            static_zone=static_zone,
            dynamic_size=DYNAMIC_OBSTACLE_SIZE,
            dynamic_margin=dynamic_margin,
            exp_weight=dynamic_exp_weight,
            exp_gain=dynamic_exp_gain,
            exp_zone=dynamic_zone,
        )
    tracker = UR5Tracker(
        reference, target_device, dtype=simulation_dtype,
        obstacle_cost=obstacle, n_samples=controller_samples,
        horizon=controller_horizon if obstacle is not None else horizon,
        temperature=temperature, mean_update_alpha=mean_update_alpha,
        noise_variance=noise_variance,
        position_weight=position_weight,
        terminal_weight=terminal_weight, smooth_weight=smooth_weight,
    )
    start = time.perf_counter()
    q, ee, control, error_mm, sample_history, estimate, actual = tracker.run(
        initial_configuration(reference, target_device),
        has_disturbance=has_disturbance,
        use_dob=use_dob,
        capture_samples=capture_samples,
        n_show=sample_count,
    )
    elapsed = time.perf_counter() - start
    solve_times_raw = np.asarray(tracker.solve_time_history)
    solve_times = (solve_times_raw.copy() if len(solve_times_raw) == len(control)
                   else np.full(len(control), elapsed / max(len(control), 1)))
    solve_times[:min(TIMING_WARMUP_STEPS, len(solve_times))] = np.nan
    result = {
        "q": q, "ee": ee, "u": control,
        "err": error_mm / 1e3, "err_mm": error_mm,
        "d_hat": estimate, "d_true": actual,
        "dob_residual": np.asarray(tracker.dob_residual_history),
        "dob_tau_model": np.asarray(tracker.dob_model_torque_history),
        "solve_t": solve_times,
        "solve_t_raw": solve_times_raw,
        "elapsed": np.asarray(elapsed),
        "has_disturbance": np.asarray(has_disturbance),
        "use_dob": np.asarray(use_dob),
        "use_obstacle": np.asarray(obstacle is not None),
        "obstacle_mode": np.asarray(
            "dynamic" if dynamic_paths is not None else
            "static" if static_centers is not None else "false"
        ),
        "mppi_seed": np.asarray(seed),
        "mppi_horizon": np.asarray(tracker.horizon),
        "mppi_samples": np.asarray(tracker.n_samples),
        "mppi_temperature": np.asarray(temperature),
        "mppi_mean_update_alpha": np.asarray(mean_update_alpha),
        "mppi_noise_variance": np.asarray(noise_variance),
        "mppi_position_weight": np.asarray(position_weight),
        "mppi_terminal_weight": np.asarray(terminal_weight),
        "mppi_smooth_weight": np.asarray(tracker.tracking_cost.smooth_weight),
        "static_exp_weight": np.asarray(static_exp_weight),
        "static_exp_gain": np.asarray(static_exp_gain),
        "static_margin": np.asarray(static_margin),
        "static_zone": np.asarray(static_zone),
        "dynamic_exp_weight": np.asarray(dynamic_exp_weight),
        "dynamic_exp_gain": np.asarray(dynamic_exp_gain),
        "dynamic_margin": np.asarray(dynamic_margin),
        "dynamic_zone": np.asarray(dynamic_zone),
        "dynamic_obstacle_size": np.asarray(DYNAMIC_OBSTACLE_SIZE),
        "dynamic_obstacle_speed_ratio": np.asarray(
            DYNAMIC_OBSTACLE_SPEED_RATIO
        ),
        "dynamic_obstacle_phase_offset": np.asarray(
            DYNAMIC_OBSTACLE_PHASE_OFFSET
        ),
        "controller": np.asarray("mppi"),
        "dynamics_model": np.asarray("M(q)qdd+C(q,qd)qd+g(q)+Bqd=tau"),
        **({"cross_paths": dynamic_paths} if dynamic_paths is not None else {}),
        **({"static_obstacle_centers": static_centers}
           if static_centers is not None else {}),
    }
    # Animation-only data is intentionally transient.  main._save_npz omits
    # underscore-prefixed fields so enabling the interactive visualization
    # does not change the persisted result schema or static plots.
    if sample_history is not None:
        result["_animation_samples"] = sample_history
    return result


def _result_key(controller: str, use_dob: bool) -> str:
    return f"{controller.upper()}_DOB" if use_dob else controller.upper()


def _comparison_cases(settings: dict) -> list[tuple[str, bool]]:
    if settings["comparison_type"] == "controller":
        return [(controller, settings["use_dob"])
                for controller in settings["comparison_controllers"]]
    return [(settings["controller"], False),
            (settings["controller"], True)]


def _run_controller(controller: str, reference: np.ndarray,
                    has_disturbance: bool, use_dob: bool,
                    dynamic_paths: np.ndarray | None,
                    static_centers: np.ndarray | None,
                    mppi_options: dict | None = None,
                    ilqr_options: dict | None = None,
                    capture_mppi_samples: bool = False) -> dict:
    if controller == "mppi":
        return run_mppi(reference, has_disturbance=has_disturbance,
                        use_dob=use_dob, dynamic_paths=dynamic_paths,
                        static_centers=static_centers,
                        capture_samples=capture_mppi_samples,
                        **(mppi_options or {}))
    if controller == "sqp":
        from controllers.sqp import run_sqp
        return run_sqp(reference, has_disturbance=has_disturbance,
                       use_dob=use_dob, dynamic_paths=dynamic_paths,
                       static_centers=static_centers)
    if controller == "ilqr":
        from controllers.ilqr import run_ilqr
        return run_ilqr(reference, has_disturbance=has_disturbance,
                        use_dob=use_dob, dynamic_paths=dynamic_paths,
                        static_centers=static_centers,
                        **(ilqr_options or {}))
    raise ValueError(f"unsupported controller: {controller}")


def run_comparison(settings: dict, reference: np.ndarray
                   ) -> tuple[dict, np.ndarray | None, np.ndarray | None]:
    """Run exactly two cases under identical environment conditions."""
    obstacle_mode = settings["obstacle_mode"]
    paths = (chasing_obstacle_paths(reference, HORIZON_OBS + 2)
             if obstacle_mode == "dynamic" else None)
    static_centers = (static_obstacle_centers(
        reference, settings["static_obstacle_count"]
    )
                      if obstacle_mode == "static" else None)
    results = {}
    for controller, use_dob in _comparison_cases(settings):
        key = _result_key(controller, use_dob)
        print("\n[RUN] "
              f"controller={controller.upper()}, "
              f"disturbance={'ON' if settings['has_disturbance'] else 'OFF'}, "
              f"DOB={'ON' if use_dob else 'OFF'}, "
              f"obstacle={obstacle_mode.upper()}, "
              f"T_SIM={settings['t_sim']:g}s, "
              f"static_count={settings['static_obstacle_count']}, "
              f"trajectory={settings['trajectory']}")
        data = _run_controller(
            controller, reference,
            has_disturbance=settings["has_disturbance"],
            use_dob=use_dob,
            dynamic_paths=paths,
            static_centers=static_centers,
            mppi_options=settings.get("mppi_options"),
            ilqr_options=settings.get("ilqr_options"),
            capture_mppi_samples=bool(settings.get("capture_mppi_samples", False)),
        )
        results[key] = data
    return results, paths, static_centers
