from __future__ import annotations

import contextlib
import io
import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pilot.er_common import (
    FEATURE_NAMES,
    PairText,
    available_memory_bytes,
    blocking_keys,
    canonical_target_fields,
    current_rss_bytes,
    disk_free_bytes,
    pair_features,
    peak_rss_bytes,
)
from pilot.run_pilot import QueryRecord, pair_features_to_blob, select_block_postings
from pilot.stream_infer import acquire_run_lock
from utils.validate_submission import validate as validate_submission


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

    def test_run_lock_acquisition_and_contention(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / ".run.lock"
            with lock_path.open("a+", encoding="utf-8") as owner:
                self.assertTrue(acquire_run_lock(owner))
                with lock_path.open("a+", encoding="utf-8") as contender:
                    self.assertFalse(acquire_run_lock(contender))
            with lock_path.open("a+", encoding="utf-8") as after_release:
                self.assertTrue(acquire_run_lock(after_release))

    def test_module_style_pilot_feature_import(self) -> None:
        left = PairText.make("Acme Foods", "12 Main Street", "US")
        right = PairText.make("Acme Food", "12 Main St", "US")
        self.assertEqual(len(pair_features_to_blob(left, right)), len(FEATURE_NAMES) * 4)

    def test_blocking_selection_uses_validation_labels_only(self) -> None:
        queries = [
            QueryRecord(0, "S1-0", "Validation", "", "US", "validation"),
            QueryRecord(1, "S1-1", "Test", "", "US", "test"),
        ]
        validation_key = "1|us|nfull|validation"
        test_key = "1|us|nfull|pilot_test"
        key_index = {validation_key: [(0, 1)], test_key: [(1, 1)]}
        counts = {validation_key: 1_000, test_key: 1}
        true_keys = {(0, "S2-0"): {validation_key}, (1, "S2-1"): {test_key}}
        truth = {0: {"S2-0"}, 1: {"S2-1"}}
        with contextlib.redirect_stdout(io.StringIO()):
            _active, _retained, report = select_block_postings(
                queries, key_index, counts, true_keys, truth, 10_000
            )
        cap_100 = next(
            trial for trial in report["trials"]
            if trial["block_cap"] == 100 and trial["query_cap"] == 250
        )
        self.assertEqual(cap_100["expected_true_pair_recall"], 0.0)
        self.assertEqual(report["selection_split"], "validation")

    def test_candidate_subset_check_direction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            matching = root / "matching_results.tsv"
            candidate = root / "candidate_pairs.tsv"
            test_dir = root / "test"
            test_dir.mkdir()
            matching.write_text(
                "source1_entity_id\tmatched_entity_ids\nS1-1\tS2-1\n",
                encoding="utf-8",
                newline="",
            )
            (test_dir / "test_source1.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-1\tA\tB\tUS\n",
                encoding="utf-8",
                newline="",
            )
            (test_dir / "test_source2.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
                "S2-1\tA\tB\tUS\nS2-2\tOther\tC\tUS\n",
                encoding="utf-8",
                newline="",
            )
            (test_dir / "test_source3.tsv").write_text(
                "entity_id\tbusiness_name\tbusiness_address\tcountry\n",
                encoding="utf-8",
                newline="",
            )
            candidate.write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-1,S2-2\n",
                encoding="utf-8",
                newline="",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                errors, warnings = validate_submission(
                    str(matching), str(candidate), str(test_dir), check_ids=True
                )
            self.assertEqual(errors, [])
            self.assertEqual(warnings, [])

            candidate.write_text(
                "source1_entity_id\tcandidate_entity_ids\nS1-1\tS2-2\n",
                encoding="utf-8",
                newline="",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                errors, warnings = validate_submission(
                    str(matching), str(candidate), str(test_dir), check_ids=True
                )
            self.assertEqual(errors, [])
            self.assertEqual(len(warnings), 1)
            self.assertIn("not present in candidate_pairs.tsv", warnings[0])

    def test_tiny_end_to_end_and_missing_result_database(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pipeline données ") as directory:
            root = Path(directory)
            data = root / "données test"
            work = root / "work output"
            output = root / "final output"
            data.mkdir()
            rows = {
                1: [
                    "S1-1\tAcme Foods\t12 Main Street\tUS",
                    "S1-2\tBeta Clinic\t99 Park Road\tIndia",
                ],
                2: [
                    "S2-1\tAcme Food\t12 Main St\tUS",
                    "S2-2\tBeta Clinic\t99 Park Rd\tIndia",
                ],
                3: ["S3-1\tAcme Foods Incorporated\t12 Main Street\tUS"],
            }
            for source, lines in rows.items():
                path = data / f"test_source{source}.tsv"
                encoding = "utf-8-sig" if source == 1 else "utf-8"
                with path.open("w", encoding=encoding, newline="\n") as handle:
                    handle.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                    handle.write("\n".join(lines) + "\n")

            command = [
                sys.executable,
                str(ROOT / "pilot/stream_infer.py"),
                "--data-root", str(data),
                "--work-dir", str(work),
                "--output-dir", str(output),
                "--model", str(ROOT / "pilot/frozen_pilot_model.json"),
                "--mode", "full",
                "--queries", "0",
                "--index-batch", "2",
                "--block-cap", "100",
                "--query-posting-cap", "100",
                "--safety-free-gib", "0",
            ]
            first = subprocess.run(
                command, cwd=root, check=False, capture_output=True, text=True
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            connection = sqlite3.connect(work / "index.sqlite")
            try:
                stored_target = connection.execute(
                    "SELECT name,address,country FROM targets WHERE id='S2-1'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(
                stored_target,
                canonical_target_fields("Acme Food", "12 Main St", "US"),
            )
            self.assertEqual(sorted(path.name for path in output.iterdir()),
                             ["candidate_pairs.tsv", "matching_results.tsv"])
            for name in ("matching_results.tsv", "candidate_pairs.tsv"):
                payload = (output / name).read_bytes()
                self.assertFalse(payload.startswith(b"\xef\xbb\xbf"))
                self.assertNotIn(b"\r\n", payload)
                self.assertEqual(len(payload.splitlines()), 3)

            validation = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "utils/validate_submission.py"),
                    "--matching", str(output / "matching_results.tsv"),
                    "--candidate", str(output / "candidate_pairs.tsv"),
                    "--test-dir", str(data),
                    "--check-ids",
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validation.returncode, 0, validation.stdout + validation.stderr)
            self.assertIn("PASS", validation.stdout)

            documentation = root / "Documentation_template.md"
            documentation.write_text(
                "# Synthetic methodology\n\nEnd-to-end fixture documentation.\n",
                encoding="utf-8",
            )
            archive_path = root / "team submission.zip"
            package = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/build_submission.py"),
                    "--team-name", "Test Team",
                    "--documentation", str(documentation),
                    "--matching", str(output / "matching_results.tsv"),
                    "--candidate", str(output / "candidate_pairs.tsv"),
                    "--test-dir", str(data),
                    "--output-zip", str(archive_path),
                    "--check-ids",
                    "--compression-level", "0",
                ],
                cwd=root,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(package.returncode, 0, package.stdout + package.stderr)
            with zipfile.ZipFile(archive_path) as archive:
                names = set(archive.namelist())
                self.assertIsNone(archive.testzip())
            self.assertIn("output/matching_results.tsv", names)
            self.assertIn("output/candidate_pairs.tsv", names)
            self.assertIn("code/business_entity_resolution/src/pilot/stream_infer.py", names)
            self.assertIn("Documentation_template.md", names)
            self.assertFalse(any(name.startswith(".github/") for name in names))

            (work / "results.sqlite").unlink()
            unsafe_resume = subprocess.run(
                command, cwd=root, check=False, capture_output=True, text=True
            )
            self.assertNotEqual(unsafe_resume.returncode, 0)
            self.assertIn("results database", unsafe_resume.stderr)

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
