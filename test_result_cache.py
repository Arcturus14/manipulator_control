from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from utils.result_cache import (MANIFEST_NAME, _canonical_hash,
                                find_cached_result, parameter_spec)


def _config(*, dob: bool = False) -> dict:
    return {
        "robot": {"torque_limit": [1] * 6, "velocity_limit": 1,
                  "damping": [0] * 6},
        "simulation": {"dt": 0.02},
        "controller": {"type": "mppi", "seed": 7},
        "cost": {"mppi": {}},
        "experiment": {"duration": 1.0, "trajectory": "circle",
                       "disturbance": dob, "dob": dob},
        "obstacle": {"mode": "none"},
        "comparison": {"type": "controller",
                       "controllers": ["mppi", "sqp"]},
        "benchmark": {"type": "hardware", "save_result": False},
        "output": {"root": "results", "show": False,
                   "torque_plot": False},
    }


class ResultCacheTests(unittest.TestCase):
    def test_output_options_and_raw_result_toggle_do_not_change_parameters(self):
        first = _config()
        second = copy.deepcopy(first)
        second["output"].update(show=True, torque_plot=True, root="elsewhere")
        second["benchmark"]["save_result"] = True
        self.assertEqual(parameter_spec("run", first),
                         parameter_spec("run", second))

    def test_main_dob_mode_preserves_output_compensation(self):
        cfg = _config(dob=True)
        self.assertEqual(
            parameter_spec("run", cfg, legacy=True)["dob_compensation_mode"],
            "output",
        )
        self.assertEqual(
            parameter_spec("run", cfg)["dob_compensation_mode"], "output"
        )

    def test_manifest_match_is_reused(self):
        cfg = _config()
        spec = parameter_spec("run", cfg)
        parameter_hash = _canonical_hash(spec)
        manifest = {
            "fingerprint": "same",
            "parameter_fingerprint": parameter_hash,
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "result"
            result.mkdir()
            (result / "config.yaml").write_text("x: 1\n", encoding="utf-8")
            (result / "metrics.json").write_text("{}\n", encoding="utf-8")
            (result / "results.npz").write_bytes(b"npz")
            (result / MANIFEST_NAME).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(
                find_cached_result(root, "run", cfg, manifest), result
            )

    def test_complete_legacy_result_is_reused(self):
        cfg = _config()
        manifest = {
            "fingerprint": "new",
            "parameter_fingerprint": _canonical_hash(
                parameter_spec("run", cfg)
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "legacy"
            result.mkdir()
            (result / "config.yaml").write_text(
                yaml.safe_dump(cfg), encoding="utf-8"
            )
            (result / "metrics.json").write_text(
                json.dumps({"MPPI": {"rmse_err_mm": 1.0}}),
                encoding="utf-8",
            )
            (result / "results.npz").write_bytes(b"npz")
            self.assertEqual(
                find_cached_result(root, "run", cfg, manifest), result
            )


if __name__ == "__main__":
    unittest.main()
