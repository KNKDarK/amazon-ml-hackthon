#!/usr/bin/env python3
"""Verify pilot artifacts using bounded SQLite queries and streaming TSV reads."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def read_id_map(path: Path, id_column: str, list_column: str) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames != [id_column, list_column]:
            raise AssertionError(f"Unexpected header in {path}: {reader.fieldnames}")
        for row in reader:
            entity_id = row[id_column]
            assert entity_id not in result, f"Duplicate ID in {path}: {entity_id}"
            result[entity_id] = {
                value for value in row[list_column].split(",") if value
            }
    return result


def entity_f05(truth_count: int, predicted_count: int, tp: int) -> float:
    if truth_count == 0:
        return 1.0 if predicted_count == 0 else 0.0
    if tp == 0:
        return 0.0
    precision = tp / predicted_count
    recall = tp / truth_count
    return 1.25 * precision * recall / (0.25 * precision + recall)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("work_dir", type=Path)
    parser.add_argument("--sample-size", type=int, default=10_000)
    args = parser.parse_args()
    work = args.work_dir

    truth = read_id_map(work / "pilot_labels.tsv", "source1_entity_id", "matched_entity_ids")
    candidates = read_id_map(work / "pilot_candidate_pairs.tsv", "source1_entity_id", "candidate_entity_ids")
    matches = read_id_map(work / "pilot_matching_results.tsv", "source1_entity_id", "matched_entity_ids")
    assert len(truth) == len(candidates) == len(matches) == args.sample_size
    assert truth.keys() == candidates.keys() == matches.keys()

    split_by_id: Dict[str, str] = {}
    with (work / "pilot_queries.tsv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            split_by_id[row["entity_id"]] = row["split"]
    assert len(split_by_id) == args.sample_size

    for s1_id, predicted in matches.items():
        assert not (predicted - candidates[s1_id]), f"Prediction outside candidates: {s1_id}"
        assert all(value.startswith(("S2-", "S3-")) for value in predicted)
    for values in candidates.values():
        assert all(value.startswith(("S2-", "S3-")) for value in values)

    candidate_db = sqlite3.connect(work / "pilot_candidates.sqlite")
    feature_db = sqlite3.connect(work / "pilot_features.sqlite")
    score_db = sqlite3.connect(work / "pilot_scores.sqlite")
    candidate_count = candidate_db.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    feature_count, bad_feature_size = feature_db.execute(
        "SELECT COUNT(*), SUM(LENGTH(features) != ?) FROM pair_features",
        (31 * 4,),
    ).fetchone()
    score_count, bad_score_range = score_db.execute(
        "SELECT COUNT(*), SUM(score < 0 OR score > 1 OR score IS NULL) FROM pair_scores"
    ).fetchone()
    assert candidate_count == feature_count == score_count
    assert bad_feature_size == 0
    assert bad_score_range == 0

    qrow_by_id: Dict[str, int] = {}
    with (work / "pilot_queries.tsv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            qrow_by_id[row["entity_id"]] = int(row["qrow"])

    true_pairs = 0
    true_hits = 0
    for s1_id, actual in truth.items():
        qrow = qrow_by_id[s1_id]
        true_pairs += len(actual)
        for target_id in actual:
            hit = candidate_db.execute(
                "SELECT 1 FROM candidates WHERE qrow=? AND target_id=?",
                (qrow, target_id),
            ).fetchone()
            true_hits += int(hit is not None)
    recall = true_hits / true_pairs if true_pairs else 1.0

    scores: List[float] = []
    for split in ("validation", "test"):
        ids = [s1_id for s1_id, value in split_by_id.items() if value == split]
        values = [entity_f05(len(truth[s1_id]), len(matches[s1_id]),
                             len(truth[s1_id] & matches[s1_id])) for s1_id in ids]
        scores.append(sum(values) / len(values))
        if split == "test":
            test_f05 = scores[-1]

    report = json.loads((work / "pilot_report.json").read_text(encoding="utf-8"))
    assert math.isclose(recall, report["candidate"]["true_pair_recall"], rel_tol=0, abs_tol=1e-12)
    assert math.isclose(test_f05, report["evaluation"]["test"]["macro_f05"], rel_tol=0, abs_tol=1e-12)
    assert candidate_count == report["candidate"]["candidate_count"]

    print("PASS")
    print(f"rows={args.sample_size:,}")
    print(f"candidates={candidate_count:,}")
    print(f"features={feature_count:,}")
    print(f"scores={score_count:,}")
    print(f"blocking_recall={recall:.9f}")
    print(f"validation_macro_f05={scores[0]:.9f}")
    print(f"test_macro_f05={test_f05:.9f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
