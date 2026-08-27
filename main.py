"""Unified CLI for simulations, comparisons, and MPPI benchmarks."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from utils.yaml_config import (apply_runtime_config, dump_config,
                               load_config, runtime_settings)


ROOT = Path(__file__).resolve().parent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "compare", "benchmark"):
        sub = subparsers.add_parser(command)
        sub.add_argument("--config", type=Path)
        sub.add_argument("--trajectory")
        sub.add_argument("--seed", type=int)
        sub.add_argument("--duration", type=float)
        sub.add_argument("--controller", choices=("mppi", "sqp"))
        sub.add_argument("--obstacle", choices=("none", "static", "dynamic"))
        sub.add_argument("--disturbance", action=argparse.BooleanOptionalAction,
                         default=None)
        sub.add_argument("--dob", action=argparse.BooleanOptionalAction,
                         default=None)
        sub.add_argument("--show", action=argparse.BooleanOptionalAction,
                         default=None)
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "experiment.trajectory": args.trajectory,
        "controller.seed": args.seed,
        "experiment.duration": args.duration,
        "controller.type": args.controller,
        "obstacle.mode": args.obstacle,
        "experiment.disturbance": args.disturbance,
        "experiment.dob": args.dob,
        "output.show": args.show,
    }


def _output_directory(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["output"]["root"])
    if not root.is_absolute():
        root = ROOT / root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = root / stamp
    destination.mkdir(parents=True)
    return destination


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _save_npz(results: dict[str, dict], destination: Path, single: bool) -> None:
    if single:
        payload = next(iter(results.values()))
    else:
        payload = {
            f"{case.lower()}__{name}": value
            for case, data in results.items() for name, value in data.items()
        }
    np.savez(destination / "results.npz", **payload)


def _mppi_options(cfg: dict[str, Any]) -> dict[str, Any]:
    controller, cost = cfg["controller"], cfg["cost"]["mppi"]
    return {
        "seed": int(controller["seed"]),
        "n_samples": int(controller["samples"]),
        "horizon": int(controller["horizon"]),
        "temperature": float(controller["temperature"]),
        "position_weight": float(cost["position"]),
        "terminal_weight": float(cost["terminal"]),
        "smooth_weight": float(
            cost["obstacle_smoothness"]
            if cfg["obstacle"]["mode"] != "none" else cost["smoothness"]
        ),
    }


def _run_or_compare(command: str, cfg: dict[str, Any], output: Path) -> None:
    from experiment_runner import (LABELS, compute_metrics, run_comparison,
                                   run_mppi, save_plots, waypoints)
    from utils.trajectory import chasing_obstacle_paths, static_obstacle_centers

    settings = runtime_settings(cfg, command)
    settings["mppi_options"] = _mppi_options(cfg)
    reference = waypoints(settings["trajectory"], settings["n_steps"])
    if command == "compare":
        results, paths, centers = run_comparison(settings, reference)
    else:
        mode = settings["obstacle_mode"]
        paths = chasing_obstacle_paths(reference, cfg["controller"]["horizon"] + 2) \
            if mode == "dynamic" else None
        centers = static_obstacle_centers(reference, settings["static_obstacle_count"]) \
            if mode == "static" else None
        controller = settings["controller"]
        if controller == "mppi":
            data = run_mppi(
                reference, settings["has_disturbance"], settings["use_dob"],
                paths, centers, **settings["mppi_options"],
            )
        else:
            from controllers.sqp import run_sqp
            data = run_sqp(
                reference, settings["has_disturbance"], settings["use_dob"],
                paths, centers,
            )
        key = f"{controller.upper()}_DOB" if settings["use_dob"] else controller.upper()
        results = {key: data}

    metrics = {key: compute_metrics(data) for key, data in results.items()}
    _save_npz(results, output, single=command == "run")
    (output / "metrics.json").write_text(
        json.dumps(_jsonable(metrics), indent=2) + "\n", encoding="utf-8"
    )
    save_plots(results, reference, dynamic_paths=paths,
               static_centers=centers, output_dir=output)
    print("\n결과 요약")
    for key, values in metrics.items():
        print(f"  {LABELS.get(key, key):<12} RMSE={values['rmse_err_mm']:.2f} mm, "
              f"MeanT={values['mean_solve_ms']:.2f} ms")
    if cfg["output"]["show"]:
        import matplotlib.pyplot as plt
        from visualization.animator import animate_comparison
        figure, animation = animate_comparison(
            results, reference, moving_obstacle_paths=paths,
            static_obstacle_centers=centers,
        )
        globals().update(_LIVE_FIGURE=figure, _LIVE_ANIMATION=animation)
        plt.show()


def _benchmark(cfg: dict[str, Any], output: Path) -> None:
    bench = cfg["benchmark"]
    kind = bench["type"]
    if kind == "baseline":
        from benchmark_mppi import run_benchmark
        run_benchmark(
            seeds=bench["seeds"], trajectory=bench["trajectories"][0],
            cases=bench["cases"], output=output / "metrics.json",
        )
    elif kind == "static":
        from benchmark_static_mppi import run_benchmark
        run_benchmark(
            seeds=bench["seeds"], trajectories=bench["trajectories"],
            cases=bench["cases"], output=output / "metrics.json",
            save_result=output / "results.npz" if bench["save_result"] else None,
        )
    elif kind == "dynamic":
        from benchmark_dynamic_mppi import run_benchmark
        run_benchmark(
            seeds=bench["seeds"], trajectories=bench["trajectories"],
            weights=bench["weights"], output=output / "metrics.json",
        )
    else:
        raise ValueError("benchmark.type must be baseline, static, or dynamic")


def main() -> None:
    args = _parser().parse_args()
    cfg = load_config(args.config, _overrides(args))
    if args.command == "benchmark":
        if args.trajectory is not None:
            cfg["benchmark"]["trajectories"] = [args.trajectory]
        if args.seed is not None:
            cfg["benchmark"]["seeds"] = [args.seed]
    apply_runtime_config(cfg)
    if not cfg["output"]["show"]:
        import matplotlib
        matplotlib.use("Agg")
    output = _output_directory(cfg)
    dump_config(cfg, output / "config.yaml")
    print(f"[CONFIG] {args.config.resolve() if args.config else 'configs/default.yaml'}")
    print(f"[OUTPUT] {output}")
    if args.command == "benchmark":
        _benchmark(cfg, output)
    else:
        _run_or_compare(args.command, cfg, output)


if __name__ == "__main__":
    main()
