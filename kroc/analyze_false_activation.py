"""Summarize adaptive-margin activation in zero-disturbance KRoC runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from config import DQ_MAX, Q_MAX, Q_MIN


def summarize(path: Path) -> dict:
    with np.load(path) as data:
        margin = np.asarray(data["kroc_margin"])
        acceleration = np.asarray(data["kroc_a_d_hat"])
    span = np.stack((np.asarray(Q_MAX) - np.asarray(Q_MIN),
                     np.full(6, 2.0 * DQ_MAX)))
    baseline = 0.05 * span[:, None, :]
    excess = np.maximum(margin - baseline, 0.0)
    active = excess > 1e-5
    sign = np.sign(acceleration)
    flips = np.sum(sign[1:] * sign[:-1] < 0, axis=0)
    return {
        "file": path.name,
        "step_activation_rate": float(np.any(active, axis=(1, 2)).mean()),
        "channel_activation_rate": float(active.mean()),
        "position_max_excess_rad": float(excess[:, 0].max()),
        "velocity_max_excess_rad_s": float(excess[:, 1].max()),
        "position_excess_mean_rad": float(excess[:, 0].mean()),
        "velocity_excess_mean_rad_s": float(excess[:, 1].mean()),
        "a_dhat_abs_p95_rad_s2": float(np.percentile(np.abs(acceleration), 95)),
        "a_dhat_abs_p99_rad_s2": float(np.percentile(np.abs(acceleration), 99)),
        "a_dhat_sign_flips_by_joint": flips.astype(int).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    paths = sorted(args.result_dir.glob("nominal_*_directional.npz"))
    if not paths:
        parser.error("no nominal directional result files found")
    rows = [summarize(path) for path in paths]
    keys = [key for key in rows[0] if key != "file" and "by_joint" not in key]
    aggregate = {
        key: {"mean": float(np.mean([row[key] for row in rows])),
              "min": float(np.min([row[key] for row in rows])),
              "max": float(np.max([row[key] for row in rows]))}
        for key in keys
    }
    payload = {"definition": "true disturbance is zero; excess is above 5% baseline",
               "runs": rows, "aggregate": aggregate}
    output = args.result_dir / "false_activation.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    print("SAVED", output)


if __name__ == "__main__":
    main()
