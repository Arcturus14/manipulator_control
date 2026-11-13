"""Audit baseline/compiled closed-loop MPPI timing result."""
import argparse
import json
from pathlib import Path

import numpy as np

from config import DT, TIMING_WARMUP_STEPS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "metrics.json").read_text())
    expected = len(manifest["arguments"]["seeds"]) * 2
    if len(rows) != expected:
        raise AssertionError(f"incomplete experiment: {len(rows)}/{expected}")
    checks = {"runs": len(rows), "cuda_runs": 0,
              "compile_flag_checks": 0, "metric_recomputations": 0}
    for row in rows:
        path = root / f"seed{row['seed']}_{row['condition']}.npz"
        with np.load(path, allow_pickle=False) as data:
            assert data["compute_device"].item() == "cuda"
            checks["cuda_runs"] += 1
            expected_flag = row["condition"] == "compiled_rollout"
            assert data["mppi_compile_rollout_loop"].item() == expected_flag
            checks["compile_flag_checks"] += 1
            error = data["err_mm"]
            rmse = float(np.sqrt(np.mean(error ** 2)))
            np.testing.assert_allclose(
                rmse, row["metrics"]["tracking_rmse_mm"], atol=1e-12
            )
            times = data["solve_t_raw"][TIMING_WARMUP_STEPS:] * 1e3
            np.testing.assert_allclose(
                times.mean(), row["metrics"]["solve_mean_ms"], atol=1e-12
            )
            assert row["metrics"]["deadline_miss_rate"] == float(
                (times > DT * 1e3).mean()
            )
            checks["metric_recomputations"] += 1
    payload = {"status": "passed", "checks": checks}
    (root / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
