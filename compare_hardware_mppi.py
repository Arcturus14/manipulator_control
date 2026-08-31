"""Compare two JSON reports produced by benchmark_hardware_mppi.py."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("benchmark") != "mppi_hardware_tracking":
        raise ValueError(f"{path} is not an MPPI hardware benchmark report")
    return report


def _label(report: dict[str, Any]) -> str:
    env = report["environment"]
    return f"{env.get('gpu_name', 'unknown GPU')} @ {env.get('hostname', 'unknown host')}"


def compare(first_path: Path, second_path: Path) -> dict[str, Any]:
    first, second = _load(first_path), _load(second_path)
    first_conditions = copy.deepcopy(first["conditions"])
    second_conditions = copy.deepcopy(second["conditions"])
    for conditions in (first_conditions, second_conditions):
        software = conditions.get("software", {})
        if "python" in software:
            software["python"] = ".".join(software["python"].split(".")[:2])
    if first_conditions != second_conditions:
        raise ValueError(
            "Benchmark conditions differ; use reports with identical trajectory, "
            "seeds, controller parameters, and timing policy."
        )
    a, b = first["aggregate"], second["aggregate"]
    ratio = a["mean_solve_ms"] / b["mean_solve_ms"]
    return {
        "benchmark": "mppi_hardware_comparison",
        "conditions": first["conditions"],
        "first": {"source": str(first_path), "label": _label(first), **a},
        "second": {"source": str(second_path), "label": _label(second), **b},
        "second_vs_first": {
            "speedup": ratio,
            "mean_solve_time_reduction_pct": (1.0 - 1.0 / ratio) * 100.0,
            "p95_speedup": a["p95_solve_ms"] / b["p95_solve_ms"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path, help="First machine metrics.json")
    parser.add_argument("second", type=Path, help="Second machine metrics.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = compare(args.first, args.second)
    rendered = json.dumps(result, indent=2) + "\n"
    print(rendered, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"[SAVE] {args.output}")


if __name__ == "__main__":
    main()
