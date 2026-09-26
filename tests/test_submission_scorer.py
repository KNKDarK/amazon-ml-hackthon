#!/usr/bin/env python3
"""Tests for the validator's macro F0.5 scorer.

The expected numbers are worked out by hand from the metric definition in
``pilot.er_common.macro_f05`` so a regression in the arithmetic is caught, and
the scorer is checked against that same reference implementation.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "utils"))

from utils import validate_submission as vs  # noqa: E402
from pilot import er_common  # noqa: E402


def write_pairs(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1, ids in rows:
            f.write(f"{s1}\t{ids}\n")


class ScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def score(self, pred_rows, truth_rows) -> dict:
        pred = self.dir / "pred.tsv"
        truth = self.dir / "truth.tsv"
        write_pairs(pred, pred_rows)
        write_pairs(truth, truth_rows)
        return vs.score_against_truth(pred, truth)

    def test_perfect_predictions_score_one(self) -> None:
        rows = [("S1-1", "S2-1,S3-1"), ("S1-2", "S2-2"), ("S1-3", "")]
        s = self.score(rows, rows)
        self.assertAlmostEqual(s["macro_f05"], 1.0, places=9)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (3, 0, 0))

    def test_hand_computed_mixed_case(self) -> None:
        # S1-1: truth {a,b}, pred {a,b}      -> P=1, R=1  -> F0.5 = 1.25/1.25 = 1
        # S1-2: truth {c},   pred {c,d,e}    -> tp=1 fp=2 fn=0 -> P=1/3, R=1
        #                                      F0.5 = 1.25*(1/3) / (0.25*(1/3) + 1)
        #                                           = 0.416667 / 1.083333 = 0.3846154
        # S1-3: truth {},    pred {}         -> 1.0
        # S1-4: truth {f},   pred {}         -> 0.0  (no true match retrieved)
        # S1-5: truth {},    pred {g}        -> 0.0  (singleton penalised)
        # macro = (1 + 0.3846154 + 1 + 0 + 0) / 5 = 2.3846154 / 5 = 0.4769231
        # totals: tp = 2+1 = 3
        #         fp = 0 (S1-1) + 2 (S1-2) + 1 (S1-5, predicting for a singleton) = 3
        #         fn = 1 (S1-4, a true match that was not retrieved)
        # predicted ids = 2 + 3 + 0 + 0 + 1 = 6 -> micro P = 3/6
        # tp + fn      = 3 + 1 = 4              -> micro R = 3/4
        s = self.score(
            [("S1-1", "a,b"), ("S1-2", "c,d,e"), ("S1-3", ""),
             ("S1-4", ""), ("S1-5", "g")],
            [("S1-1", "a,b"), ("S1-2", "c"), ("S1-3", ""),
             ("S1-4", "f"), ("S1-5", "")],
        )
        self.assertAlmostEqual(s["macro_f05"], 2.3846153846153847 / 5, places=9)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (3, 3, 1))
        self.assertAlmostEqual(s["micro_precision"], 3 / 6, places=9)
        self.assertAlmostEqual(s["micro_recall"], 3 / 4, places=9)

    def test_empty_truth_and_empty_prediction_scores_one(self) -> None:
        s = self.score([("S1-1", "")], [("S1-1", "")])
        self.assertAlmostEqual(s["macro_f05"], 1.0, places=9)

    def test_predicting_for_a_singleton_scores_zero(self) -> None:
        s = self.score([("S1-1", "x")], [("S1-1", "")])
        self.assertAlmostEqual(s["macro_f05"], 0.0, places=9)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (0, 1, 0))

    def test_all_wrong_scores_zero(self) -> None:
        s = self.score([("S1-1", "x"), ("S1-2", "y")],
                       [("S1-1", "a"), ("S1-2", "b")])
        self.assertAlmostEqual(s["macro_f05"], 0.0, places=9)

    def test_duplicate_ids_do_not_inflate_counts(self) -> None:
        s = self.score([("S1-1", "a,a,a")], [("S1-1", "a")])
        self.assertAlmostEqual(s["macro_f05"], 1.0, places=9)
        self.assertEqual((s["tp"], s["fp"], s["fn"]), (1, 0, 0))

    def test_matches_reference_implementation(self) -> None:
        """Cross-check against er_common.macro_f05 on pseudo-random data."""
        import random
        rng = random.Random(20260926)
        truth_rows, pred_rows = [], []
        for i in range(400):
            qt = {f"S2-{rng.randrange(50)}" for _ in range(rng.randrange(0, 3))}
            qp = {f"S2-{rng.randrange(50)}" for _ in range(rng.randrange(0, 3))}
            truth_rows.append((f"S1-{i}", ",".join(sorted(qt))))
            pred_rows.append((f"S1-{i}", ",".join(sorted(qp))))
        s = self.score(pred_rows, truth_rows)

        query_ids = list(range(len(truth_rows)))
        truth_map = {i: set(v.split(",")) - {""} for i, (_, v) in enumerate(truth_rows)}
        pred_map = {i: set(v.split(",")) - {""} for i, (_, v) in enumerate(pred_rows)}
        ref = er_common.macro_f05(query_ids, truth_map, pred_map)

        self.assertAlmostEqual(s["macro_f05"], ref["macro_f05"], places=12)
        self.assertEqual(s["tp"], ref["tp"])
        self.assertEqual(s["fp"], ref["fp"])
        self.assertEqual(s["fn"], ref["fn"])
        self.assertAlmostEqual(s["micro_precision"], ref["micro_precision"], places=12)
        self.assertAlmostEqual(s["micro_recall"], ref["micro_recall"], places=12)

    def test_missing_truth_file_reports_no_score(self) -> None:
        pred = self.dir / "pred.tsv"
        write_pairs(pred, [("S1-1", "a")])
        self.assertIsNone(vs.report_score(pred, self.dir / "nope.tsv"))

    def test_bad_truth_header_is_reported_not_raised(self) -> None:
        pred = self.dir / "pred.tsv"
        write_pairs(pred, [("S1-1", "a")])
        truth = self.dir / "truth.tsv"
        truth.write_text("wrong\theader\nS1-1\ta\n", encoding="utf-8")
        self.assertIsNone(vs.report_score(pred, truth))


if __name__ == "__main__":
    unittest.main(verbosity=2)
