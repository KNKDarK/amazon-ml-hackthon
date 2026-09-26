#!/usr/bin/env python3
"""Tests for utils/validate_submission.py.

Each published rule gets a passing case and a failing case, using a small
synthetic test set so the suite is fast and does not depend on the multi-GB
challenge data. Warnings are asserted to stay warnings: they must never turn a
run into a failure.
"""
from __future__ import annotations

import io
import contextlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import validate_submission as vs  # noqa: E402

MATCH_H = "source1_entity_id\tmatched_entity_ids"
CAND_H = "source1_entity_id\tcandidate_entity_ids"


def make_test_dir(base: Path, s1, s2, s3) -> Path:
    d = base / "test"
    d.mkdir(parents=True, exist_ok=True)
    for name, ids in (("1", s1), ("2", s2), ("3", s3)):
        with (d / f"test_source{name}.tsv").open("w", encoding="utf-8", newline="\n") as f:
            f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for i in ids:
                f.write(f"S{name}-{i}\tCorp {i}\t{i} Road\tUS\n")
    return d


def write_result(path: Path, rows, header=MATCH_H, newline="\n",
                 encoding="utf-8", bom=False) -> Path:
    data = header + newline + newline.join(rows) + newline
    raw = data.encode(encoding)
    if bom:
        raw = b"\xef\xbb\xbf" + raw
    path.write_bytes(raw)
    return path


class ValidatorTestCase(unittest.TestCase):
    S1 = ["1", "2", "3", "4"]
    S2 = ["10", "11"]
    S3 = ["20"]

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.test_dir = make_test_dir(self.base, self.S1, self.S2, self.S3)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def good_matching(self) -> Path:
        return write_result(
            self.base / "matching_results.tsv",
            ["S1-1\tS2-10", "S1-2\t", "S1-3\tS3-20", "S1-4\tS2-10,S3-20"],
        )

    def good_candidates(self) -> Path:
        return write_result(
            self.base / "candidate_pairs.tsv",
            ["S1-1\tS2-10,S2-11", "S1-2\t", "S1-3\tS3-20", "S1-4\tS2-10,S2-11,S3-20"],
            header=CAND_H,
        )

    def run_validate(self, matching=None, candidate=None, check_ids=False):
        matching = matching or self.good_matching()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            errors, warnings = vs.validate(
                matching, candidate, self.test_dir, check_ids=check_ids
            )
        return errors, warnings


class HappyPathTests(ValidatorTestCase):
    def test_clean_submission_passes(self) -> None:
        errors, _ = self.run_validate(candidate=self.good_candidates())
        self.assertEqual(errors, [])

    def test_empty_predictions_are_allowed(self) -> None:
        m = write_result(self.base / "m.tsv", ["S1-1\t", "S1-2\t", "S1-3\t", "S1-4\t"])
        errors, _ = self.run_validate(matching=m)
        self.assertEqual(errors, [])

    def test_blank_lines_are_ignored(self) -> None:
        p = self.base / "m.tsv"
        p.write_bytes((MATCH_H + "\nS1-1\tS2-10\n\nS1-2\t\n\nS1-3\t\nS1-4\t\n").encode())
        errors, _ = self.run_validate(matching=p)
        self.assertEqual(errors, [])

    def test_utf8_bom_is_tolerated(self) -> None:
        p = self.good_matching()
        p.write_bytes(b"\xef\xbb\xbf" + p.read_bytes())
        errors, _ = self.run_validate(matching=p)
        self.assertEqual(errors, [])

    def test_crlf_line_endings_are_tolerated(self) -> None:
        p = write_result(self.base / "m.tsv",
                         ["S1-1\tS2-10", "S1-2\t", "S1-3\t", "S1-4\t"],
                         newline="\r\n")
        errors, _ = self.run_validate(matching=p)
        self.assertEqual(errors, [])


class BlockingRuleTests(ValidatorTestCase):
    def assertFails(self, fragment, **kw):
        errors, _ = self.run_validate(**kw)
        self.assertTrue(errors, f"expected a blocking error mentioning {fragment!r}")
        self.assertTrue(any(fragment in e for e in errors),
                        f"none of {errors!r} mention {fragment!r}")

    def test_missing_matching_file(self) -> None:
        self.assertFails("File not found", matching=self.base / "nope.tsv")

    def test_empty_file(self) -> None:
        p = self.base / "m.tsv"
        p.write_bytes(b"")
        self.assertFails("is empty", matching=p)

    def test_comma_separated_header_is_called_out(self) -> None:
        p = write_result(self.base / "m.tsv", ["S1-1,S2-10"], header="source1_entity_id,matched_entity_ids")
        self.assertFails("COMMA-separated", matching=p)

    def test_wrong_header(self) -> None:
        p = write_result(self.base / "m.tsv", ["S1-1\tS2-10"], header="source1_entity_id\tpreds")
        self.assertFails("unexpected header", matching=p)

    def test_duplicate_s1_row(self) -> None:
        p = write_result(self.base / "m.tsv",
                         ["S1-1\tS2-10", "S1-2\t", "S1-3\t", "S1-4\t", "S1-1\tS2-11"])
        self.assertFails("duplicate source1_entity_id", matching=p)

    def test_repeated_id_inside_one_list(self) -> None:
        p = write_result(self.base / "m.tsv",
                         ["S1-1\tS2-10,S2-10", "S1-2\t", "S1-3\t", "S1-4\t"])
        self.assertFails("repeated ID inside", matching=p)

    def test_self_match_is_rejected(self) -> None:
        p = write_result(self.base / "m.tsv",
                         ["S1-1\tS1-1", "S1-2\t", "S1-3\t", "S1-4\t"])
        self.assertFails("self-matches", matching=p)

    def test_id_without_s2_s3_prefix(self) -> None:
        p = write_result(self.base / "m.tsv",
                         ["S1-1\tS9-99", "S1-2\t", "S1-3\t", "S1-4\t"])
        self.assertFails("without an S2-/S3- prefix", matching=p)

    def test_missing_required_s1_entity(self) -> None:
        p = write_result(self.base / "m.tsv", ["S1-1\tS2-10", "S1-2\t", "S1-3\t"])
        self.assertFails("required S1 entity(ies) missing", matching=p)

    def test_s1_id_not_in_test_set(self) -> None:
        p = write_result(self.base / "m.tsv",
                         ["S1-1\tS2-10", "S1-2\t", "S1-3\t", "S1-4\t", "S1-99\t"])
        self.assertFails("not in the test set", matching=p)

    def test_malformed_row_without_tab(self) -> None:
        p = self.base / "m.tsv"
        p.write_bytes((MATCH_H + "\nS1-1\tS2-10\nS1-2\nS1-3\t\nS1-4\t\n").encode())
        self.assertFails("malformed row (no tab)", matching=p)

    def test_missing_test_source1(self) -> None:
        errors, _ = self.run_validate(matching=self.good_matching(),
                                      candidate=None)
        # now point at a directory with no source1
        empty = self.base / "empty"
        empty.mkdir()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            errors, _ = vs.validate(self.good_matching(), None, empty, check_ids=False)
        self.assertTrue(any("Test source1 file not found" in e for e in errors))


class CheckIdsTests(ValidatorTestCase):
    def test_unknown_s2_id_is_blocking_with_check_ids(self) -> None:
        p = write_result(self.base / "m.tsv",
                         ["S1-1\tS2-999", "S1-2\t", "S1-3\t", "S1-4\t"])
        errors, _ = self.run_validate(matching=p, check_ids=True)
        self.assertTrue(any("not in the test Source-2/3" in e for e in errors),
                        f"got {errors!r}")

    def test_unknown_id_is_not_checked_without_the_flag(self) -> None:
        p = write_result(self.base / "m.tsv",
                         ["S1-1\tS2-999", "S1-2\t", "S1-3\t", "S1-4\t"])
        errors, warnings = self.run_validate(matching=p, check_ids=False)
        self.assertEqual(errors, [])
        self.assertTrue(any("ID-existence check is OFF" in w for w in warnings))

    def test_candidate_unknown_id_is_blocking_with_check_ids(self) -> None:
        c = write_result(self.base / "c.tsv",
                         ["S1-1\tS2-999", "S1-2\t", "S1-3\t", "S1-4\t"],
                         header=CAND_H)
        errors, _ = self.run_validate(candidate=c, check_ids=True)
        self.assertTrue(any("not in the test Source-2/3" in e for e in errors))


class WarningOnlyTests(ValidatorTestCase):
    def test_matched_not_in_candidates_is_a_warning_not_a_failure(self) -> None:
        c = write_result(self.base / "c.tsv",
                         ["S1-1\tS2-10", "S1-2\t", "S1-3\tS3-20", "S1-4\t"],
                         header=CAND_H)
        errors, warnings = self.run_validate(candidate=c)
        self.assertEqual(errors, [], "must not be a blocking error")
        self.assertTrue(any("not present in candidate_pairs.tsv" in w for w in warnings),
                        f"expected the subset warning, got {warnings!r}")

    def test_empty_candidate_row_with_matches_is_still_flagged(self) -> None:
        """Regression: an empty candidate list used to skip the subset check.

        S1-4 predicts two matches and has an empty candidate list, which is the
        most serious provenance failure. It must still raise the warning.
        """
        c = write_result(self.base / "c.tsv",
                         ["S1-1\tS2-10", "S1-2\t", "S1-3\tS3-20", "S1-4\t"],
                         header=CAND_H)
        errors, warnings = self.run_validate(candidate=c)
        self.assertEqual(errors, [])
        offender_warnings = [w for w in warnings if "not present in candidate_pairs.tsv" in w]
        self.assertEqual(len(offender_warnings), 1, warnings)
        self.assertIn("S1-4", offender_warnings[0])

    def test_empty_candidate_row_without_matches_is_not_flagged(self) -> None:
        """The fix must not create false positives for genuinely empty rows."""
        c = self.good_candidates()
        errors, warnings = self.run_validate(candidate=c)
        self.assertEqual(errors, [])
        self.assertFalse([w for w in warnings if "not present in candidate_pairs.tsv" in w],
                         warnings)

    def test_absent_candidate_file_is_a_warning(self) -> None:
        errors, warnings = self.run_validate(candidate=self.base / "missing.tsv")
        self.assertEqual(errors, [])
        self.assertTrue(any("not found" in w and "candidate_pairs" in w for w in warnings))

    def test_candidate_header_mismatch_is_blocking(self) -> None:
        c = write_result(self.base / "c.tsv",
                         ["S1-1\tS2-10", "S1-2\t", "S1-3\t", "S1-4\t"],
                         header=MATCH_H)  # wrong header on purpose
        errors, _ = self.run_validate(candidate=c)
        self.assertTrue(any("unexpected header" in e for e in errors))


class CliTests(ValidatorTestCase):
    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(ROOT / "utils/validate_submission.py"), *args],
            capture_output=True, text=True, cwd=str(ROOT),
        )

    def test_exit_zero_on_clean_submission(self) -> None:
        r = self.run_cli("--matching", str(self.good_matching()),
                         "--candidate", str(self.good_candidates()),
                         "--test-dir", str(self.test_dir), "--check-ids")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS", r.stdout)

    def test_exit_one_on_blocking_error(self) -> None:
        bad = write_result(self.base / "bad.tsv", ["S1-1\tS2-10"])
        r = self.run_cli("--matching", str(bad), "--test-dir", str(self.test_dir))
        self.assertEqual(r.returncode, 1)
        self.assertIn("FAIL", r.stdout)

    def test_non_utf8_reports_actionable_message(self) -> None:
        p = self.base / "m.tsv"
        p.write_bytes((MATCH_H + "\nS1-1\tS2-10\n").encode("utf-8")
                      + b"S1-2\tS2-10\xff\xfe\n")
        r = self.run_cli("--matching", str(p), "--test-dir", str(self.test_dir))
        self.assertEqual(r.returncode, 1)
        self.assertIn("UTF-8", r.stdout)

    def test_score_only_without_truth_fails(self) -> None:
        r = self.run_cli("--score-only", "--matching", str(self.good_matching()))
        self.assertEqual(r.returncode, 1)
        self.assertIn("--ground-truth", r.stdout)


class KnownGapTests(ValidatorTestCase):
    def test_row_order_is_not_verified_by_the_validator(self) -> None:
        """Documents a real gap: order is required by the rules but unchecked.

        The published format requires the S1 order to be preserved, but
        validate() compares sets of IDs and never looks at line order. A
        shuffled submission therefore passes here. This test records the
        current behaviour so the gap is visible rather than silent; it is not
        an endorsement.
        """
        shuffled = write_result(
            self.base / "m.tsv",
            ["S1-3\t", "S1-1\tS2-10", "S1-4\t", "S1-2\t"],
        )
        errors, _ = self.run_validate(matching=shuffled)
        self.assertEqual(errors, [],
                         "documents that order is currently unchecked")


if __name__ == "__main__":
    unittest.main(verbosity=2)
