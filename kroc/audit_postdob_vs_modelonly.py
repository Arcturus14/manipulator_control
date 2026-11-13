"""Audit raw Post-DOB versus model-only dynamic-margin benchmark data."""
import argparse
import json
from pathlib import Path

import numpy as np

from config import DQ_MAX, Q_MAX, Q_MIN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "metrics.json").read_text())
    expected = (len(manifest["arguments"]["seeds"])
                * len(manifest["arguments"]["scenarios"])
                * len(manifest["architectures"]))
    if len(rows) != expected:
        raise AssertionError(f"incomplete experiment: {len(rows)}/{expected}")
    counts = {
        "runs": len(rows), "expected_runs": expected,
        "cuda_runs": 0, "unclamped_state_runs": 0,
        "margin_d_hat_checks": 0, "headroom_checks": 0,
        "margin_interval_checks": 0, "raw_violation_checks": 0,
    }
    spans = np.stack((np.asarray(Q_MAX) - np.asarray(Q_MIN),
                      np.full(6, 2 * DQ_MAX)))
    architecture_nonzero = {name: False for name in manifest["architectures"]}
    for row in rows:
        stem = (f"{row['scenario']}_seed{row['seed']}_"
                f"{row['architecture']}_{row['margin_config']['mode']}")
        with np.load(root / f"{stem}.npz", allow_pickle=False) as data:
            if data["compute_device"].item() != "cuda":
                raise AssertionError("non-CUDA result")
            counts["cuda_runs"] += 1
            np.testing.assert_array_equal(data["qd_raw"], data["qd"])
            assert not data["state_clamp_enabled"].item()
            counts["unclamped_state_runs"] += 1

            cfg = row["margin_config"]
            limits = data["physical_torque_limit"]
            expected_d = np.clip(
                data["d_hat_used"],
                -limits * cfg["disturbance_clip_fraction"],
                limits * cfg["disturbance_clip_fraction"],
            )
            np.testing.assert_allclose(
                data["kroc_d_hat_margin"], expected_d, atol=2e-5
            )
            if row["scenario"] != "nominal" and np.any(np.abs(expected_d) > 1e-4):
                architecture_nonzero[row["architecture"]] = True
            counts["margin_d_hat_checks"] += 1

            center = data["kroc_center_horizon"]
            positive = (limits - center).min(axis=1)
            negative = (center + limits).min(axis=1)
            np.testing.assert_allclose(
                data["kroc_headroom_positive"], positive, atol=2e-5
            )
            np.testing.assert_allclose(
                data["kroc_headroom_negative"], negative, atol=2e-5
            )
            safe = limits * cfg["headroom_safe_fraction"]
            expected_lower = 1 + cfg["headroom_gain"] * np.maximum(
                1 - positive / safe, 0
            )
            expected_upper = 1 + cfg["headroom_gain"] * np.maximum(
                1 - negative / safe, 0
            )
            np.testing.assert_allclose(
                data["kroc_gain_lower"], expected_lower, atol=2e-6
            )
            np.testing.assert_allclose(
                data["kroc_gain_upper"], expected_upper, atol=2e-6
            )
            counts["headroom_checks"] += 1

            margin = data["kroc_margin"]
            assert np.all(
                margin >= spans[None, :, None, :] * cfg["min_fraction"] - 1e-6
            )
            assert np.all(
                margin <= spans[None, :, None, :] * cfg["max_fraction"] + 1e-6
            )
            assert np.all(
                margin.sum(axis=2)
                <= spans[None] * (1 - cfg["min_internal_fraction"]) + 1e-6
            )
            counts["margin_interval_checks"] += 1

            q, dq = data["q"][1:], data["qd_raw"][1:]
            q_steps = np.any(
                (q < np.asarray(Q_MIN)) | (q > np.asarray(Q_MAX)), axis=1
            ).sum()
            dq_steps = np.any(np.abs(dq) > DQ_MAX, axis=1).sum()
            assert q_steps == row["metrics"]["position_violation_steps"]
            assert dq_steps == row["metrics"]["velocity_violation_steps"]
            counts["raw_violation_checks"] += 1

    if not all(architecture_nonzero.values()):
        raise AssertionError(
            f"dynamic margin did not receive nonzero d_hat: {architecture_nonzero}"
        )
    payload = {
        "status": "passed", "checks": counts,
        "nonzero_margin_d_hat_by_architecture": architecture_nonzero,
    }
    (root / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
