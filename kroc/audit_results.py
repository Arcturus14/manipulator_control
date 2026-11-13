"""Audit saved raw data and create mean/median/worst scalar statistics."""
import argparse
import json
from pathlib import Path

import numpy as np

from config import Q_MAX, Q_MIN, DQ_MAX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    rows = json.loads((root / "metrics.json").read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    expected = (len(manifest["arguments"]["seeds"])
                * len(manifest["arguments"]["caps"])
                * len(manifest["arguments"]["scenarios"]) * 3)
    if len(rows) != expected:
        raise AssertionError(f"incomplete experiment: {len(rows)}/{expected}")
    counts = {"runs": len(rows), "expected_runs": expected,
              "unclamped_state_runs": 0, "direction_headroom_checks": 0,
              "margin_interval_checks": 0, "raw_violation_checks": 0}
    summaries = {}
    for row in rows:
        scenario, cap, seed, mode = (row[k] for k in ("scenario", "cap", "seed", "mode"))
        stem = f"{scenario}_cap{cap:.3f}_seed{seed}_{mode}"
        with np.load(root / f"{stem}.npz", allow_pickle=False) as data:
            if data["compute_device"].item() != "cuda":
                raise AssertionError("non-CUDA result")
            np.testing.assert_array_equal(data["qd_raw"], data["qd"])
            assert not data["state_clamp_enabled"].item()
            counts["unclamped_state_runs"] += 1
            limits = data["physical_torque_limit"]
            center = data["kroc_center_horizon"]
            np.testing.assert_allclose(data["kroc_headroom_positive"], (limits - center).min(axis=1), atol=1e-5)
            np.testing.assert_allclose(data["kroc_headroom_negative"], (center + limits).min(axis=1), atol=1e-5)
            cfg = row["margin_config"]
            for side, reserve in (("lower", "positive"), ("upper", "negative")):
                expected_gain = 1 + cfg["headroom_gain"] * np.maximum(
                    1 - data[f"kroc_headroom_{reserve}"] / (limits * cfg["headroom_safe_fraction"]), 0)
                np.testing.assert_allclose(data[f"kroc_gain_{side}"], expected_gain, atol=2e-6)
            counts["direction_headroom_checks"] += 1
            margin = data["kroc_margin"]
            span = np.stack((np.asarray(Q_MAX) - np.asarray(Q_MIN), np.full(6, 2 * DQ_MAX)))
            assert np.all(margin >= span[None, :, None, :] * cfg["min_fraction"] - 1e-6)
            assert np.all(margin <= span[None, :, None, :] * cap + 1e-6)
            assert np.all(margin.sum(axis=2) <= span[None] * (1 - cfg["min_internal_fraction"]) + 1e-6)
            if mode == "fixed":
                np.testing.assert_allclose(margin, np.broadcast_to(span[None, :, None] * cfg["fixed_fraction"], margin.shape), atol=1e-6)
            counts["margin_interval_checks"] += 1
            q, dq = data["q"][1:], data["qd_raw"][1:]
            q_steps = np.any((q < np.asarray(Q_MIN)) | (q > np.asarray(Q_MAX)), axis=1).sum()
            dq_steps = np.any(np.abs(dq) > DQ_MAX, axis=1).sum()
            assert q_steps == row["metrics"]["position_violation_steps"]
            assert dq_steps == row["metrics"]["velocity_violation_steps"]
            counts["raw_violation_checks"] += 1
            row["metrics"]["position_margin_variation_l1_rad"] = float(np.abs(np.diff(margin[:, 0], axis=0)).sum())
            row["metrics"]["velocity_margin_variation_l1_rad_s"] = float(np.abs(np.diff(margin[:, 1], axis=0)).sum())
            row["metrics"]["headroom_gain_active_fraction"] = float(np.mean(
                (data["kroc_gain_lower"] > 1 + 1e-6) | (data["kroc_gain_upper"] > 1 + 1e-6)))
            row["metrics"]["margin_ceiling_fraction"] = float(np.mean(
                np.isclose(margin, span[None, :, None, :] * cap, atol=1e-6, rtol=0)))
            row["metrics"].pop("margin_variation_l1", None)
        key = f"{scenario}/cap{cap:.3f}/{mode}"
        summaries.setdefault(key, []).append(row["metrics"])
    aggregate = {}
    for key, metrics in summaries.items():
        aggregate[key] = {"runs": len(metrics), "statistics": {}}
        for name, value in metrics[0].items():
            if not isinstance(value, (int, float)):
                continue
            values = np.asarray([m[name] for m in metrics])
            aggregate[key]["statistics"][name] = {
                "mean": float(values.mean()), "median": float(np.median(values)),
                "min": float(values.min()), "max": float(values.max()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.,
            }
    (root / "audit.json").write_text(json.dumps({"checks": counts, "status": "passed"}, indent=2) + "\n")
    (root / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n")
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
