"""Run two configured UR5 cases and render side-by-side comparisons."""
from __future__ import annotations

import argparse
import logging
import math
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
import numpy as np

# This module is also an independent entry point, so configure the temporary
# cache here as well. Repeated calls in the same process reuse one directory.
from utils.torch_runtime import configure_temporary_inductor_cache

configure_temporary_inductor_cache()


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
                    CROSS_RADIUS, DT, HORIZON_OBS, LAMBDA, MPPI_SEED,
                    MPPI_W_POS, MPPI_W_TERM, N_STEPS, TIMING_WARMUP_STEPS,
                    TRAJ_CENTER, TRAJ_RADIUS)
from controllers.mppi_dob_controller import UR5Tracker
from costs.obstacle_cost import ObstacleCost
from kinematics.ur5_kinematics import ik_np
from utils.execution_config import (configuration_name,
                                    load_execution_config)
from utils.trajectory import TRAJECTORIES, crossing_obstacle_paths
from visualization.animator import animate_comparison
from visualization.plotters import (LABELS, compute_metrics, plot_3d_trajectory,
                                    plot_result_dashboard, plot_timeseries)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
NPZ_DIR = DATA_DIR / "npz"
PNG_ARCHIVE_DIR = DATA_DIR / "png"
LEGACY_RESULT_NAMES = (
    "MPPI_circle_results.npz",
    "MPPI_DOB_circle_results.npz",
    "MPPI_cross_results.npz",
    "MPPI_DOB_cross_results.npz",
)

PLOT_FILES = {
    "trajectory": "final_trajectory.png",
    "timeseries": "timeseries_ee_error.png",
    "dashboard": "metrics_dashboard.png",
}


def prepare_data_directories() -> None:
    """Create storage directories and migrate legacy root-level NPZ files."""
    NPZ_DIR.mkdir(parents=True, exist_ok=True)
    PNG_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for name in LEGACY_RESULT_NAMES:
        legacy = ROOT / name
        if not legacy.exists():
            continue
        target = NPZ_DIR / name
        if target.exists():
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target = NPZ_DIR / f"legacy_{stamp}_{legacy.name}"
        shutil.move(str(legacy), str(target))
        print(f"[MIGRATE] {legacy} -> {target}")


def archive_previous_plots() -> Path | None:
    """Move current top-level PNG artifacts into a timestamped archive."""
    current_plots = sorted(DATA_DIR.glob("*.png"))
    if not current_plots:
        return None
    archive = PNG_ARCHIVE_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    archive.mkdir(parents=True)
    for source in current_plots:
        shutil.move(str(source), str(archive / source.name))
    print(f"[ARCHIVE] 이전 PNG {len(current_plots)}개: {archive}")
    return archive


def save_latest_plots(results: dict, reference: np.ndarray,
                      show_obstacles: bool) -> dict[str, plt.Figure]:
    """Archive old plots and save all figures for the latest result under data/."""
    archive_previous_plots()
    figures = {
        "trajectory": plot_3d_trajectory(results, reference, show_obstacles=show_obstacles),
        "timeseries": plot_timeseries(results),
        "dashboard": plot_result_dashboard(results),
    }
    for name, figure in figures.items():
        destination = DATA_DIR / PLOT_FILES[name]
        figure.savefig(destination, dpi=180, bbox_inches="tight")
        print(f"[PLOT] {destination}")
    return figures


def waypoints(name: str = "circle") -> np.ndarray:
    return TRAJECTORIES[name](N_STEPS, TRAJ_CENTER, TRAJ_RADIUS)


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initial_configuration(reference: np.ndarray, target_device: torch.device) -> torch.Tensor:
    hint = np.array([math.pi, -1.2, 1.2, -1.5, -1.57, 0.0])
    return torch.tensor(ik_np(reference[0], hint), dtype=torch.float64, device=target_device)


def run_mppi(reference: np.ndarray, has_disturbance: bool = True,
             use_dob: bool = False,
             dynamic_paths: np.ndarray | None = None, *,
             seed: int = MPPI_SEED,
             n_samples: int | None = None,
             temperature: float = LAMBDA,
             position_weight: float = MPPI_W_POS,
             terminal_weight: float = MPPI_W_TERM) -> dict:
    # MPPI is stochastic. A fixed seed makes controller tuning comparisons
    # reproducible while leaving the sampled-control algorithm unchanged.
    torch.manual_seed(seed)
    target_device = device()
    simulation_dtype = torch.float32
    obstacle = None
    if dynamic_paths is not None:
        obstacle = ObstacleCost(
            HORIZON_OBS, target_device, simulation_dtype,
            dynamic_paths=dynamic_paths,
            safe_radius=CROSS_RADIUS + CROSS_MARGIN,
            exp_weight=CROSS_EXP_W,
            exp_gain=CROSS_EXP_K,
            exp_zone=CROSS_EXP_ZONE,
        )
    tracker = UR5Tracker(
        reference, target_device, dtype=simulation_dtype,
        obstacle_cost=obstacle, n_samples=n_samples,
        temperature=temperature, position_weight=position_weight,
        terminal_weight=terminal_weight,
    )
    start = time.perf_counter()
    q, ee, control, error_mm, _, estimate, actual = tracker.run(
        initial_configuration(reference, target_device),
        has_disturbance=has_disturbance,
        use_dob=use_dob,
    )
    elapsed = time.perf_counter() - start
    solve_times_raw = np.asarray(tracker.solve_time_history)
    solve_times = (solve_times_raw.copy() if len(solve_times_raw) == len(control)
                   else np.full(len(control), elapsed / max(len(control), 1)))
    solve_times[:min(TIMING_WARMUP_STEPS, len(solve_times))] = np.nan
    return {
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
        "use_obstacle": np.asarray(dynamic_paths is not None),
        "mppi_seed": np.asarray(seed),
        "mppi_temperature": np.asarray(temperature),
        "mppi_position_weight": np.asarray(position_weight),
        "mppi_terminal_weight": np.asarray(terminal_weight),
        "controller": np.asarray("mppi"),
        "dynamics_model": np.asarray("M(q)qdd+C(q,qd)qd+g(q)+Bqd=tau"),
        **({"cross_paths": dynamic_paths} if dynamic_paths is not None else {}),
    }


def _result_key(controller: str, use_dob: bool) -> str:
    return f"{controller.upper()}_DOB" if use_dob else controller.upper()


def _comparison_cases(settings: dict) -> list[tuple[str, bool]]:
    if settings["comparison_type"] == "controller":
        return [("mppi", settings["use_dob"]),
                ("sqp", settings["use_dob"])]
    return [(settings["controller"], False),
            (settings["controller"], True)]


def _run_controller(controller: str, reference: np.ndarray,
                    has_disturbance: bool, use_dob: bool,
                    dynamic_paths: np.ndarray | None) -> dict:
    if controller == "mppi":
        return run_mppi(reference, has_disturbance=has_disturbance,
                        use_dob=use_dob, dynamic_paths=dynamic_paths)
    from controllers.sqp import run_sqp

    return run_sqp(reference, has_disturbance=has_disturbance,
                   use_dob=use_dob, dynamic_paths=dynamic_paths)


def run_comparison(settings: dict, reference: np.ndarray
                   ) -> tuple[dict, np.ndarray | None]:
    """Run exactly two cases under identical environment conditions."""
    paths = (crossing_obstacle_paths(
        N_STEPS + HORIZON_OBS + 2, TRAJ_CENTER
    ) if settings["use_obstacle"] else None)
    results = {}
    for controller, use_dob in _comparison_cases(settings):
        key = _result_key(controller, use_dob)
        print("\n[RUN] "
              f"controller={controller.upper()}, "
              f"disturbance={'ON' if settings['has_disturbance'] else 'OFF'}, "
              f"DOB={'ON' if use_dob else 'OFF'}, "
              f"obstacle={'ON' if settings['use_obstacle'] else 'OFF'}, "
              f"trajectory={settings['trajectory']}")
        data = _run_controller(
            controller, reference,
            has_disturbance=settings["has_disturbance"],
            use_dob=use_dob,
            dynamic_paths=paths,
        )
        destination = NPZ_DIR / configuration_name(
            controller, settings["has_disturbance"], use_dob,
            settings["use_obstacle"], settings["trajectory"]
        )
        np.savez(destination, **data)
        print(f"[SAVE] {key}: {destination}")
        results[key] = data
    return results, paths


def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two UR5 control cases")
    parser.add_argument(
        "--config", type=Path, default=ROOT / "config.txt",
        help="key=value execution config (default: ./config.txt)",
    )
    return parser.parse_args()


def main() -> None:
    args = command_line()
    settings = load_execution_config(args.config.resolve())
    if settings["mode"] != "comparison":
        raise SystemExit("run_comparison.py requires mode=comparison in config.txt")
    print(f"[CONFIG] {args.config.resolve()}")
    prepare_data_directories()
    reference = waypoints(settings["trajectory"])
    results, paths = run_comparison(settings, reference)
    print("\n결과 요약")
    for key, data in results.items():
        metrics = compute_metrics(data)
        print(f"  {LABELS.get(key, key):<12} "
              f"mean={metrics['mean_err_mm']:.2f} mm, "
              f"RMSE={metrics['rmse_err_mm']:.2f} mm, "
              f"Std={metrics['std_err_mm']:.2f} mm, "
              f"MeanT={metrics['mean_solve_ms']:.2f} ms, "
              f"WorstT={metrics['worst_solve_ms']:.2f} ms")

    saved_figures = save_latest_plots(
        results, reference, show_obstacles=settings["use_obstacle"]
    )
    figure, anim = animate_comparison(results, reference, moving_obstacle_paths=paths)
    globals().update(_LIVE_FIGURE=figure, _LIVE_ANIMATION=anim,
                     _SAVED_FIGURES=saved_figures)
    plt.show()


if __name__ == "__main__":
    main()
