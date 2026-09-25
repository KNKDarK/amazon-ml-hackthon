#!/usr/bin/env python3
"""Summarize production blocking-cap replays without touching the frozen model.

The replay directories are produced by ``pilot/stream_infer.py``.  This script
only reads their TSV/JSON artifacts and the frozen pilot labels; it never trains
or writes a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

TEST_S1_ROWS = 1_732_544
TEST_TARGET_ROWS = 4_887_273 + 5_082_316
PILOT_TARGET_ROWS = 5_034_616 + 5_285_603


def read_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)


def read_list_file(path: Path, key: str, value: str) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {}
    for row in read_rows(path):
        entity_id = row[key]
        if entity_id in result:
            raise ValueError(f"duplicate {key}: {entity_id}")
        values = [item for item in row[value].split(",") if item]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate target ID in {path}: {entity_id}")
        result[entity_id] = set(values)
    return result


def percentile(values: Sequence[int], percent: float) -> float:
    """The pilot's linear-interpolation percentile convention."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100.0
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(ordered[low])
    return float(ordered[low] * (high - position) + ordered[high] * (position - low))


def file_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot-work", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--production-index-bytes", type=int, default=11_309_927_940,
                        help="conservative full production-test index estimate from the bounded preflight")
    parser.add_argument("--production-index-seconds", type=float, default=8028.0,
                        help="full production-test index-build estimate from the bounded preflight")
    args = parser.parse_args()

    labels = read_list_file(args.pilot_work / "pilot_labels.tsv", "source1_entity_id", "matched_entity_ids")
    queries = [row for row in read_rows(args.pilot_work / "pilot_queries.tsv") if row["split"] == "validation"]
    query_ids = [row["entity_id"] for row in queries]
    query_id_set = set(query_ids)
    if len(query_ids) != len(query_id_set):
        raise ValueError("duplicate validation query IDs")

    original = read_list_file(args.pilot_work / "pilot_candidate_pairs.tsv", "source1_entity_id", "candidate_entity_ids")
    original_counts = [len(original.get(entity_id, set())) for entity_id in query_ids]
    original_true = sum(len(labels[entity_id]) for entity_id in query_ids)
    original_hits = sum(len(labels[entity_id] & original.get(entity_id, set())) for entity_id in query_ids)

    results: Dict[str, Dict[str, object]] = {}
    for cap in (100, 500, 1000):
        cap_root = args.root / f"cap_{cap}"
        candidate_path = cap_root / "output" / "candidate_pairs.tsv"
        report_path = cap_root / "preflight_report.json"
        if not candidate_path.is_file() or not report_path.is_file():
            raise FileNotFoundError(f"incomplete cap_{cap}: {candidate_path} / {report_path}")
        candidates = read_list_file(candidate_path, "source1_entity_id", "candidate_entity_ids")
        if set(candidates) != query_id_set:
            missing = sorted(query_id_set - set(candidates))[:3]
            extra = sorted(set(candidates) - query_id_set)[:3]
            raise ValueError(f"cap_{cap} query coverage mismatch; missing={missing}, extra={extra}")
        counts = [len(candidates[entity_id]) for entity_id in query_ids]
        total = sum(counts)
        true_total = sum(len(labels[entity_id]) for entity_id in query_ids)
        hits = sum(len(labels[entity_id] & candidates[entity_id]) for entity_id in query_ids)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        expected_profile = {
            "target_sample_rate": 1,
            "query_stride": 1,
            "block_frequency_cap_full": cap,
            "query_posting_cap_full": cap,
        }
        actual_profile = {key: report.get(key) for key in expected_profile}
        if actual_profile != expected_profile:
            raise ValueError(f"cap_{cap} profile mismatch: {actual_profile!r} != {expected_profile!r}")
        indexed_rows = sum(int(value) for value in report.get("indexed_target_rows_by_source", {}).values())
        scanned_rows = int(report.get("index_source_rows_scanned", 0))
        if indexed_rows > PILOT_TARGET_ROWS or scanned_rows != PILOT_TARGET_ROWS:
            raise ValueError(f"cap_{cap} did not scan the full pilot target corpus")
        index_bytes = sum(file_bytes(cap_root / name) for name in ("index.sqlite", "index.sqlite-wal", "index.sqlite-shm"))
        result_bytes = sum(file_bytes(cap_root / name) for name in ("results.sqlite", "results.sqlite-wal", "results.sqlite-shm"))
        candidate_tsv_bytes = file_bytes(cap_root / "output" / "candidate_pairs.tsv")
        matching_tsv_bytes = file_bytes(cap_root / "output" / "matching_results.tsv")
        query_scale = TEST_S1_ROWS / len(query_ids)
        target_scale = TEST_TARGET_ROWS / PILOT_TARGET_ROWS
        work_scale = query_scale * target_scale
        projected_pairs = int(round(total * work_scale))
        # The replay index intentionally omits postings for keys absent from
        # the validation queries.  Use the separately measured conservative
        # full-production index estimate for full-test storage/runtime.
        projected_index = int(args.production_index_bytes)
        projected_results = int(round(result_bytes * work_scale))
        projected_candidate_tsv = int(round(candidate_tsv_bytes * work_scale))
        projected_matching_tsv = int(round(matching_tsv_bytes * query_scale))
        projected_peak_output = 2 * (projected_candidate_tsv + projected_matching_tsv)
        projected_peak_working = projected_index + projected_results + projected_peak_output
        index_seconds = float(report.get("index_seconds") or 0.0)
        query_seconds = float(report.get("query_seconds") or 0.0)
        query_safety_factor = 1.25
        projected_runtime = args.production_index_seconds + query_seconds * work_scale * query_safety_factor
        results[str(cap)] = {
            "cap": cap,
            "configuration": {
                "block_frequency_cap": cap,
                "query_posting_cap": cap,
                "target_sample_rate": 1,
                "query_count": len(query_ids),
                "query_split": "validation",
            },
            "candidate_metrics": {
                "true_links": true_total,
                "true_link_candidate_hits": hits,
                "true_link_candidate_recall": hits / true_total if true_total else 1.0,
                "total_candidate_pairs": total,
                "average_candidates_per_s1": total / len(query_ids),
                "p95_candidates_per_s1": percentile(counts, 95),
                "median_candidates_per_s1": percentile(counts, 50),
                "max_candidates_per_s1": max(counts),
                "queries_with_candidates": sum(count > 0 for count in counts),
            },
            "measured_resources": {
                "index_bytes_replay_equivalent": index_bytes,
                "index_seconds_replay_builder": index_seconds,
                "results_sqlite_bytes": result_bytes,
                "candidate_pairs_tsv_bytes": candidate_tsv_bytes,
                "matching_results_tsv_bytes": matching_tsv_bytes,
                "index_seconds": index_seconds,
                "query_and_output_seconds": query_seconds,
                "run_seconds": float(report.get("run_seconds") or 0.0),
                "peak_rss_bytes": int(report.get("peak_rss_bytes") or 0),
                "indexed_target_rows": indexed_rows,
                "index_source_rows_scanned": scanned_rows,
                "throughput_queries_per_second": float(report.get("throughput_queries_per_second") or 0.0),
            },
            "full_test_projection": {
                "method": "linear S1 query scaling times target-row scaling; index uses target-row scaling; output peak includes temp plus final TSV",
                "test_s1_rows": TEST_S1_ROWS,
                "test_target_rows": TEST_TARGET_ROWS,
                "pilot_target_rows": PILOT_TARGET_ROWS,
                "query_scale": query_scale,
                "target_scale": target_scale,
                "combined_work_scale": work_scale,
                "projected_candidate_pairs": projected_pairs,
                "production_index_bytes_basis": args.production_index_bytes,
                "production_index_seconds_basis": args.production_index_seconds,
                "query_safety_factor": query_safety_factor,
                "projected_index_bytes": projected_index,
                "projected_results_sqlite_bytes": projected_results,
                "projected_candidate_pairs_tsv_bytes": projected_candidate_tsv,
                "projected_matching_results_tsv_bytes": projected_matching_tsv,
                "projected_peak_output_bytes": projected_peak_output,
                "projected_peak_working_bytes": projected_peak_working,
                "projected_runtime_seconds": projected_runtime,
                "projected_runtime_hours": projected_runtime / 3600.0,
            },
        }

    payload = {
        "model": {
            "path": str(args.model),
            "sha256": sha256(args.model),
        },
        "pilot": {
            "validation_queries": len(query_ids),
            "validation_true_links": original_true,
            "original_pilot_candidate_hits_on_validation": original_hits,
            "original_pilot_validation_recall": original_hits / original_true if original_true else 1.0,
            "original_pilot_validation_pairs": sum(original_counts),
            "original_pilot_validation_average_candidates": sum(original_counts) / len(query_ids),
            "original_pilot_validation_p95_candidates": percentile(original_counts, 95),
        },
        "caps": results,
    }
    with (args.root / "cap_validation.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
