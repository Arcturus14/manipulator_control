"""Content-addressed result reuse for the unified experiment CLI."""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


MANIFEST_NAME = "run_manifest.json"
SCHEMA_VERSION = 1

# These files define numerical execution rather than result rendering or CLI
# persistence.  A behavior-changing edit therefore creates a new result even
# when the YAML parameters remain unchanged.
NUMERICAL_SOURCE_PATHS = (
    "config.py",
    "experiment_runner.py",
    "controllers",
    "costs",
    "dynamics",
    "kinematics",
    "observers",
    "utils/disturbance.py",
    "utils/trajectory.py",
)

BENCHMARK_SOURCE = {
    "baseline": "benchmark_mppi.py",
    "static": "benchmark_static_mppi.py",
    "dynamic": "benchmark_dynamic_mppi.py",
    "hardware": "benchmark_hardware_mppi.py",
    "mean_update": "benchmark_mean_update_mppi.py",
}


def result_root(cfg: dict[str, Any], repository_root: Path) -> Path:
    root = Path(cfg["output"]["root"])
    return root if root.is_absolute() else repository_root / root


def _uses_mppi_dob(command: str, cfg: dict[str, Any]) -> bool:
    controller = cfg["controller"]["type"]
    if command == "run":
        return controller == "mppi" and bool(cfg["experiment"]["dob"])
    if command == "compare":
        comparison = cfg["comparison"]
        if comparison["type"] == "dob":
            return controller == "mppi"
        return ("mppi" in comparison["controllers"]
                and bool(cfg["experiment"]["dob"]))
    return cfg["benchmark"]["type"] in {"static", "dynamic", "mean_update"}


def parameter_spec(command: str, cfg: dict[str, Any], *,
                   legacy: bool = False) -> dict[str, Any]:
    """Return only settings that can change computed numerical results."""
    normalized = copy.deepcopy(cfg)
    normalized.pop("output", None)
    normalized.get("benchmark", {}).pop("save_result", None)
    dob_mode = "not_applicable"
    if _uses_mppi_dob(command, normalized):
        # The main execution path preserves the original post-optimizer DOB
        # compensation. DOB-aware rollout is opt-in in dedicated experiment
        # scripts and must not change the GUI/CLI default behavior.
        dob_mode = "output"
    return {
        "schema_version": SCHEMA_VERSION,
        "command": command,
        "dob_compensation_mode": dob_mode,
        "config": normalized,
    }


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_files(repository_root: Path, command: str,
                  cfg: dict[str, Any]) -> list[Path]:
    files: set[Path] = set()
    for relative in NUMERICAL_SOURCE_PATHS:
        path = repository_root / relative
        if path.is_dir():
            files.update(path.rglob("*.py"))
        elif path.is_file():
            files.add(path)
    if command == "benchmark":
        relative = BENCHMARK_SOURCE.get(cfg["benchmark"]["type"])
        if relative:
            files.add(repository_root / relative)
    return sorted(files)


def source_fingerprint(repository_root: Path, command: str,
                       cfg: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for path in _source_files(repository_root, command, cfg):
        digest.update(path.relative_to(repository_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def environment_spec() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "pytorch_mppi": importlib.metadata.version("pytorch-mppi"),
        "device": "cuda" if cuda_available else "cpu",
        "device_name": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if cuda_available else platform.processor() or platform.machine()
        ),
    }


def build_manifest(command: str, cfg: dict[str, Any],
                   repository_root: Path) -> dict[str, Any]:
    parameters = parameter_spec(command, cfg)
    environment = environment_spec()
    source_hash = source_fingerprint(repository_root, command, cfg)
    parameter_hash = _canonical_hash(parameters)
    execution_hash = _canonical_hash({
        "source": source_hash,
        "environment": environment,
    })
    fingerprint = _canonical_hash({
        "parameters": parameter_hash,
        "execution": execution_hash,
    })
    return {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "parameter_fingerprint": parameter_hash,
        "execution_fingerprint": execution_hash,
        "command": command,
        "created_at": datetime.now().astimezone().isoformat(),
        "environment": environment,
        "numerical_source_fingerprint": source_hash,
    }


def _result_complete(directory: Path, command: str) -> bool:
    required = [directory / "config.yaml", directory / "metrics.json"]
    if command != "benchmark":
        required.append(directory / "results.npz")
    return all(path.is_file() for path in required)


def _infer_legacy_command(metrics_path: Path) -> str | None:
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metrics, dict):
        return None
    if "benchmark" in metrics:
        return "benchmark"
    if metrics and all(isinstance(value, dict) for value in metrics.values()):
        return "run" if len(metrics) == 1 else "compare"
    return None


def _legacy_parameter_fingerprint(directory: Path,
                                  command: str) -> str | None:
    if _infer_legacy_command(directory / "metrics.json") != command:
        return None
    try:
        cfg = yaml.safe_load(
            (directory / "config.yaml").read_text(encoding="utf-8")
        )
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(cfg, dict):
        return None
    try:
        return _canonical_hash(parameter_spec(command, cfg, legacy=True))
    except (KeyError, TypeError):
        return None


def find_cached_result(root: Path, command: str, cfg: dict[str, Any],
                       manifest: dict[str, Any]) -> Path | None:
    """Return a complete matching result without modifying prior artifacts."""
    if not root.is_dir():
        return None
    current_parameter_hash = manifest["parameter_fingerprint"]
    directories = sorted(
        (path for path in root.iterdir() if path.is_dir()), reverse=True,
    )
    for directory in directories:
        manifest_path = directory / MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (existing.get("fingerprint") == manifest["fingerprint"]
                and _result_complete(directory, command)):
            return directory

    # Preserve and reuse complete timestamp results produced before manifests.
    # Their historical DOB convention is the same output compensation used by
    # the main GUI/CLI execution path.
    for directory in directories:
        if (directory / MANIFEST_NAME).exists():
            continue
        if not _result_complete(directory, command):
            continue
        if _legacy_parameter_fingerprint(directory, command) == current_parameter_hash:
            return directory
    return None


def write_manifest(directory: Path, manifest: dict[str, Any]) -> None:
    destination = directory / MANIFEST_NAME
    destination.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
