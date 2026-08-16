"""Run one configurable UR5 MPPI or SQP simulation.

The user can independently enable plant disturbance, DOB compensation, and
dynamic crossing obstacles. Results are stored under ``data/``.
"""
from __future__ import annotations

import logging
import argparse
from pathlib import Path

# Configure this before run_comparison imports torch. The temporary directory
# remains available during the process and is deleted on normal shutdown.
from utils.torch_runtime import configure_temporary_inductor_cache

configure_temporary_inductor_cache()


class _SmallGpuTemplateWarningFilter(logging.Filter):
    """Hide a harmless Inductor template-selection warning on smaller GPUs."""

    def filter(self, record: logging.LogRecord) -> bool:
        return "Not enough SMs to use max_autotune_gemm mode" not in record.getMessage()


logging.getLogger("torch._inductor.utils").addFilter(
    _SmallGpuTemplateWarningFilter()
)

import matplotlib.pyplot as plt
import numpy as np

from config import HORIZON_OBS, N_STEPS, TRAJ_CENTER
from run_comparison import (NPZ_DIR, compute_metrics, prepare_data_directories,
                            run_mppi, save_latest_plots, waypoints)
from utils.execution_config import (configuration_name,
                                    load_execution_config)
from utils.trajectory import TRAJECTORIES, crossing_obstacle_paths
from visualization.animator import animate_comparison


def ask_boolean(prompt: str, default: bool) -> bool:
    """Read a Korean yes/no answer, accepting Enter as the supplied default."""
    suffix = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "1", "예", "ㅇ"}:
            return True
        if answer in {"n", "no", "0", "아니오", "ㄴ"}:
            return False
        print("  y 또는 n으로 입력해 주세요.")


def command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one UR5 controller simulation")
    parser.add_argument("--config", type=Path, help="key=value execution config")
    return parser.parse_args()


def main() -> None:
    args = command_line()
    prepare_data_directories()
    if args.config:
        settings = load_execution_config(args.config.resolve())
        if settings["mode"] != "single":
            raise SystemExit("run.py requires mode=single in config.txt")
        print(f"[CONFIG] {args.config.resolve()}")
    else:
        print("\nUR5 단일 제어기 시뮬레이션 설정")
        controller = input("제어기 선택 [mppi/sqp] (default=mppi): ").strip().lower() or "mppi"
        if controller not in {"mppi", "sqp"}:
            raise SystemExit("제어기는 mppi 또는 sqp여야 합니다.")
        settings = {
            "controller": controller,
            "has_disturbance": ask_boolean("외란을 적용할까요?", False),
            "use_dob": ask_boolean("DOB 보상을 사용할까요?", False),
            "use_obstacle": ask_boolean("움직이는 crossing 장애물을 사용할까요?", False),
        }
        choices = "/".join(TRAJECTORIES)
        trajectory = input(
            f"Reference 궤적 [{choices}] (default=circle): "
        ).strip().lower() or "circle"
        if trajectory not in TRAJECTORIES:
            raise SystemExit(f"궤적은 {choices} 중 하나여야 합니다.")
        settings["trajectory"] = trajectory

    controller = settings["controller"]
    has_disturbance = settings["has_disturbance"]
    use_dob = settings["use_dob"]
    use_obstacle = settings["use_obstacle"]
    trajectory = settings["trajectory"]

    reference = waypoints(trajectory)
    paths = (crossing_obstacle_paths(N_STEPS + HORIZON_OBS + 2, TRAJ_CENTER)
             if use_obstacle else None)
    print("\n[RUN] "
          f"controller={controller.upper()}, "
          f"disturbance={'ON' if has_disturbance else 'OFF'}, "
          f"DOB={'ON' if use_dob else 'OFF'}, "
          f"obstacle={'ON' if use_obstacle else 'OFF'}, "
          f"trajectory={trajectory}")
    if controller == "mppi":
        data = run_mppi(reference, has_disturbance=has_disturbance,
                        use_dob=use_dob, dynamic_paths=paths)
    else:
        from controllers.sqp import run_sqp
        data = run_sqp(reference, has_disturbance=has_disturbance,
                       use_dob=use_dob, dynamic_paths=paths)
    destination = NPZ_DIR / configuration_name(
        controller, has_disturbance, use_dob, use_obstacle, trajectory
    )
    np.savez(destination, **data)
    print(f"[SAVE] {destination}")

    key = f"{controller.upper()}_DOB" if use_dob else controller.upper()
    results = {key: data}
    metrics = compute_metrics(data)
    print("\n결과 요약")
    print(f"  Mean error={metrics['mean_err_mm']:.2f} mm")
    print(f"  RMSE={metrics['rmse_err_mm']:.2f} mm")
    print(f"  Std={metrics['std_err_mm']:.2f} mm")
    print(f"  MeanT={metrics['mean_solve_ms']:.2f} ms")
    print(f"  WorstT={metrics['worst_solve_ms']:.2f} ms")

    saved_figures = save_latest_plots(results, reference, show_obstacles=use_obstacle)
    figure, animation = animate_comparison(
        results, reference, moving_obstacle_paths=paths
    )
    globals().update(_LIVE_FIGURE=figure, _LIVE_ANIMATION=animation,
                     _SAVED_FIGURES=saved_figures)
    plt.show()


if __name__ == "__main__":
    main()
