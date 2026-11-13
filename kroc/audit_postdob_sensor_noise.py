"""Audit clean/noisy Post-DOB raw results and measurement separation."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from config import DQ_MAX, DT, Q_MAX, Q_MIN, TAU_MAX


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    manifest = json.loads((root / "manifest.json").read_text())
    rows = json.loads((root / "metrics.json").read_text())
    expected = (len(manifest["arguments"]["seeds"])
                * len(manifest["arguments"]["scenarios"])
                * len(manifest["conditions"]))
    if len(rows) != expected:
        raise AssertionError(f"incomplete experiment: {len(rows)}/{expected}")
    checks = {
        "runs": len(rows), "expected_runs": expected, "cuda_runs": 0,
        "true_state_checks": 0, "measurement_checks": 0,
        "dynamic_margin_disabled_checks": 0, "metric_recomputations": 0,
        "acceleration_filter_checks": 0,
        "dob_filter_update_checks": 0,
    }
    for row in rows:
        stem = f"{row['scenario']}_seed{row['seed']}_{row['condition']}"
        with np.load(root / f"{stem}.npz", allow_pickle=False) as data:
            assert data["compute_device"].item() == "cuda"
            checks["cuda_runs"] += 1
            true = np.concatenate((data["q"], data["qd"]), axis=1)
            np.testing.assert_array_equal(data["true_state"], true)
            assert not data["state_clamp_enabled"].item()
            checks["true_state_checks"] += 1
            assert not data["dynamic_state_margin_enabled"].item()
            assert not any(key.startswith("kroc_margin") for key in data.files)
            checks["dynamic_margin_disabled_checks"] += 1

            error = data["measured_state"] - true
            q_rms = float(np.sqrt(np.mean(error[:, :6] ** 2)))
            dq_rms = float(np.sqrt(np.mean(error[:, 6:] ** 2)))
            np.testing.assert_allclose(
                q_rms, row["metrics"]["realized_position_noise_rms_rad"],
                atol=1e-12,
            )
            np.testing.assert_allclose(
                dq_rms, row["metrics"]["realized_velocity_noise_rms_rad_s"],
                atol=1e-12,
            )
            if row["condition"] == "clean":
                np.testing.assert_array_equal(data["measured_state"], true)
            else:
                target_q = row["sensor_noise"]["position_std_rad"]
                target_dq = row["sensor_noise"]["velocity_std_rad_s"]
                assert abs(q_rms / target_q - 1) < .1
                assert abs(dq_rms / target_dq - 1) < .1
            checks["measurement_checks"] += 1

            q, dq = data["q"][1:], data["qd"][1:]
            q_steps = np.any(
                (q < np.asarray(Q_MIN)) | (q > np.asarray(Q_MAX)), axis=1
            ).sum()
            dq_steps = np.any(np.abs(dq) > DQ_MAX, axis=1).sum()
            assert q_steps == row["metrics"]["position_violation_steps"]
            assert dq_steps == row["metrics"]["velocity_violation_steps"]
            d_error = data["d_hat"][1:] - data["d_true"][1:]
            d_rmse = float(np.sqrt(np.mean(d_error ** 2)))
            np.testing.assert_allclose(
                d_rmse, row["metrics"]["dob_estimation_rmse_nm"], atol=1e-12
            )
            limits = np.asarray(TAU_MAX) * manifest["scenarios"][row["scenario"]]["torque_scale"]
            saturation = np.isclose(
                np.abs(data["u_applied"]), limits, rtol=0, atol=1e-5
            ).any(1).sum()
            assert saturation == row["metrics"]["saturation_steps"]
            if "dob_acceleration_lpf_cutoff_hz" in row:
                cutoff = row["dob_acceleration_lpf_cutoff_hz"]
                np.testing.assert_allclose(
                    data["dob_acceleration_lpf_cutoff_hz"].item(), cutoff,
                    rtol=0, atol=0,
                )
                raw = data["dob_acceleration_raw"]
                filtered = data["dob_acceleration_filtered"]
                if cutoff == 0:
                    np.testing.assert_array_equal(filtered, raw)
                else:
                    beta = math.exp(-2.0 * math.pi * cutoff * DT)
                    expected_filtered = np.zeros_like(raw)
                    for index in range(1, len(raw)):
                        expected_filtered[index] = (
                            beta * expected_filtered[index - 1]
                            + (1.0 - beta) * raw[index]
                        )
                    np.testing.assert_allclose(
                        filtered, expected_filtered, rtol=2e-6, atol=2e-6
                    )
                checks["acceleration_filter_checks"] += 1
            if "dob_alpha" in row:
                np.testing.assert_allclose(
                    data["dob_alpha"].item(), row["dob_alpha"], rtol=0, atol=0
                )
            if "dob_cutoff_hz" in row:
                expected_cutoff = row["dob_cutoff_hz"]
                actual_cutoff = data["dob_cutoff_hz"].item()
                if expected_cutoff is None:
                    assert np.isnan(actual_cutoff)
                    assert data["dob_filter_discretization"].item() == "legacy_euler"
                else:
                    np.testing.assert_allclose(
                        actual_cutoff, expected_cutoff, rtol=0, atol=0
                    )
                    assert data["dob_filter_discretization"].item() == "exact_zoh"
                    beta = np.exp(-2.0 * np.pi * expected_cutoff * DT)
                    expected_hat = np.zeros_like(data["d_hat"])
                    for index in range(1, len(expected_hat)):
                        expected_hat[index] = (
                            beta * expected_hat[index - 1]
                            + (1.0 - beta) * data["dob_residual"][index]
                        )
                    np.testing.assert_allclose(
                        data["d_hat"], expected_hat, rtol=3e-6, atol=3e-5
                    )
                checks["dob_filter_update_checks"] += 1
            checks["metric_recomputations"] += 1
    payload = {"status": "passed", "checks": checks}
    (root / "audit.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
