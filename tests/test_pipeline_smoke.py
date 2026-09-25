from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pilot.er_common import (
    FEATURE_NAMES,
    PairText,
    available_memory_bytes,
    blocking_keys,
    current_rss_bytes,
    disk_free_bytes,
    pair_features,
    peak_rss_bytes,
)


class PipelineSmokeTests(unittest.TestCase):
    def test_frozen_model_matches_feature_schema(self) -> None:
        model = json.loads((ROOT / "pilot/frozen_pilot_model.json").read_text(encoding="utf-8"))
        self.assertEqual(model["feature_names"], FEATURE_NAMES)
        self.assertEqual(len(model["weights"]), len(FEATURE_NAMES))
        self.assertEqual(len(model["feature_mean"]), len(FEATURE_NAMES))
        self.assertEqual(len(model["feature_std"]), len(FEATURE_NAMES))
        self.assertEqual(model["decision_policy"]["max_predictions_per_query"], 5)

    def test_blocking_and_feature_shapes(self) -> None:
        keys = blocking_keys("Acme Foods, Inc.", "12 Main Street", "US")
        self.assertTrue(keys)
        left = PairText.make("Acme Foods, Inc.", "12 Main Street", "US")
        right = PairText.make("Acme Food", "12 Main St", "US")
        vector = pair_features(left, right)
        self.assertEqual(vector.shape, (len(FEATURE_NAMES),))
        self.assertTrue(bool((vector == vector).all()))

    def test_cross_platform_resource_helpers(self) -> None:
        self.assertGreater(disk_free_bytes(ROOT), 0)
        for value in (available_memory_bytes(), current_rss_bytes(), peak_rss_bytes()):
            self.assertGreaterEqual(value, 0)

    def test_inference_cli_help(self) -> None:
        result = subprocess.run(
            [sys.executable, "pilot/stream_infer.py", "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--output-dir", result.stdout)
        self.assertIn("--index-synchronous", result.stdout)


if __name__ == "__main__":
    unittest.main()
