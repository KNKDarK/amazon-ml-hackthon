#!/usr/bin/env python3
"""Prove the parallel blocking passes are byte-identical to the serial ones.

Builds a real subset of the training corpus, runs measure_block_frequencies and
persist_candidates at workers=1 and at workers=10, and diffs every observable:
the frequency counters, the true-key sets, the candidate SQLite contents, and
the reported statistics.

The subset is only a few MiB, which is below the production
``parallel_scan.MIN_CHUNK_BYTES`` floor of 4 MiB. Left alone, that floor makes
``plan_byte_ranges`` return a *single* range no matter how many workers are
requested, so the "parallel" run would execute as one in-process task and this
check would compare serial against serial while reporting PASS. The floor is
therefore lowered here -- only in this parent process, and only for range
planning -- so the 40k-row subset splits into many real chunks. Ranges are
planned in the parent and handed to the pool as explicit byte offsets, so
workers are unaffected. ``_assert_really_chunked`` fails the run outright if the
subset ever stops splitting.
"""
from __future__ import annotations

import csv
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pilot.er_common import MemoryMonitor, blocking_keys, stable_u64  # noqa: E402
from pilot import parallel_scan  # noqa: E402
from pilot import run_pilot as rp  # noqa: E402

ROWS = 40_000

# Small enough that a 40k-row subset still yields many chunks, large enough that
# every chunk still holds thousands of rows.
TEST_MIN_CHUNK_BYTES = 256 * 1024
CHUNKS_PER_WORKER = 4
PARALLEL_WORKERS = 10
MIN_RANGES = 4


def _planned_range_count(path: Path, workers: int) -> int:
    return len(parallel_scan.plan_byte_ranges(path, workers * CHUNKS_PER_WORKER))


def _assert_really_chunked(train: Path) -> int:
    """Fail loudly unless the subset actually splits into several ranges."""
    parallel_scan.MIN_CHUNK_BYTES = TEST_MIN_CHUNK_BYTES
    counts = {
        path.name: _planned_range_count(path, PARALLEL_WORKERS)
        for path in sorted(train.glob("train_source[23].tsv"))
    }
    if not counts:
        raise SystemExit("no source2/source3 subset found; cannot verify chunking")
    smallest = min(counts.values())
    if smallest < MIN_RANGES:
        parallel_scan.MIN_CHUNK_BYTES = parallel_scan.MIN_CHUNK_BYTES
        raise SystemExit(
            f"REFUSING to report PASS: the corpus subset only splits into "
            f"{smallest} range(s) at workers={PARALLEL_WORKERS} ({counts}), so the "
            f"parallel path would never run. Raise ROWS or lower "
            f"TEST_MIN_CHUNK_BYTES so each source yields >= {MIN_RANGES} chunks."
        )
    return smallest



def build_subset(dest: Path, source: Path, rows: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8-sig", newline="") as handle, \
            dest.open("w", encoding="utf-8", newline="\n") as out:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_NONE,
                            lineterminator="\n")
        writer.writerow(next(reader))
        for index, record in enumerate(reader):
            if index >= rows:
                break
            writer.writerow(record)


def labelled_source1_subset(dest: Path, source: Path, truth: Path, rows: int) -> set[str]:
    """Write the first ``rows`` S1 rows that actually carry ground-truth labels.

    ``sample_queries`` requires a label for every sampled S1 row, so the subset
    has to be drawn from labelled rows rather than a blind prefix.
    """
    wanted: set[str] = set()
    with truth.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for index, (s1_id, _matches) in enumerate(reader):
            if index >= rows * 5:
                break
            wanted.add(s1_id)

    kept: set[str] = set()
    with source.open("r", encoding="utf-8-sig", newline="") as handle, \
            dest.open("w", encoding="utf-8", newline="\n") as out:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_NONE,
                            lineterminator="\n")
        writer.writerow(next(reader))
        for record in reader:
            if record[0] in wanted and len(kept) < rows:
                writer.writerow(record)
                kept.add(record[0])
    return kept


def write_truth_subset(dest: Path, truth: Path, wanted: set[str]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with truth.open("r", encoding="utf-8-sig", newline="") as handle, \
            dest.open("w", encoding="utf-8", newline="\n") as out:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        writer = csv.writer(out, delimiter="\t", quoting=csv.QUOTE_NONE,
                            lineterminator="\n")
        writer.writerow(next(reader))
        for record in reader:
            if record[0] in wanted:
                writer.writerow(record)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="parallel_check_"))
    data_root = tmp / "dataset"
    train = data_root / "train"
    real = ROOT / "dataset" / "train"
    train.mkdir(parents=True, exist_ok=True)
    print(f"building {ROWS:,}-row subset in {tmp}")
    kept = labelled_source1_subset(train / "train_source1.tsv", real / "train_source1.tsv",
                                   real / "train_ground_truth.tsv", ROWS)
    write_truth_subset(train / "train_ground_truth.tsv", real / "train_ground_truth.tsv", kept)
    build_subset(train / "train_source2.tsv", real / "train_source2.tsv", ROWS)
    build_subset(train / "train_source3.tsv", real / "train_source3.tsv", ROWS)

    ranges = _assert_really_chunked(train)
    print(f"verified real chunking: >= {ranges} byte ranges per source at "
          f"workers={PARALLEL_WORKERS} "
          f"(serial workers=1 would use {_planned_range_count(train / 'train_source2.tsv', 1)})")

    # Every sampled S1 row is its own query, so no ground-truth file matching is
    # needed for the comparison itself; labels are synthesised for the
    # last-writer-wins path in persist_candidates.
    work = tmp / "work"
    work.mkdir(parents=True, exist_ok=True)
    queries, truth, _ = rp.sample_queries(data_root, work, min(ROWS, 5000))
    key_index = rp.build_query_key_index(queries)
    print(f"queries={len(queries)} keys={len(key_index):,}")

    monitor = MemoryMonitor().start()
    results = {}
    for workers in (1, PARALLEL_WORKERS):
        wd = tmp / f"work_w{workers}"
        wd.mkdir(parents=True, exist_ok=True)
        counts, true_keys, freq_stats = rp.measure_block_frequencies(
            data_root, queries, truth, key_index, monitor, workers
        )
        active_index, retained, _ = rp.select_block_postings(
            queries, key_index, counts, true_keys, truth, 5_000_000, None, ROWS * 2
        )
        del retained
        cand_path, cand_stats = rp.persist_candidates(
            data_root, wd, queries, truth, active_index, monitor, 10_000, workers
        )
        rows = sqlite3.connect(cand_path).execute(
            "SELECT qrow,target_id,source,name_norm,address_norm,country,block_mask "
            "FROM candidates ORDER BY qrow,target_id"
        ).fetchall()
        results[workers] = {
            "counts": dict(counts),
            "true_keys": {k: sorted(v) for k, v in true_keys.items()},
            "candidates": rows,
            "cand_count": cand_stats["candidate_count"],
            "recall": cand_stats["true_pair_recall"],
            "by_source": cand_stats["candidates_by_source"],
            "masks": cand_stats["block_mask_counts"],
            "posting_matches": cand_stats["active_posting_matches"],
            "keys_generated": cand_stats["average_generated_keys_per_target"],
        }
        print(f"  workers={workers:>2}: candidates={len(rows):,} "
              f"recall={cand_stats['true_pair_recall']:.6f}")
    monitor.stop()

    serial, parallel = results[1], results[PARALLEL_WORKERS]
    failures = []
    for field in ("counts", "true_keys", "candidates", "cand_count", "recall",
                  "by_source", "masks", "posting_matches", "keys_generated"):
        same = serial[field] == parallel[field]
        print(f"  {field:<18} {'MATCH' if same else 'DIFFER'}")
        if not same:
            failures.append(field)
            if field in ("counts", "true_keys"):
                a, b = serial[field], parallel[field]
                diff = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
                print(f"      {len(diff)} differing keys, e.g. {diff[:5]}")
            elif field == "candidates":
                print(f"      serial={len(serial[field])} parallel={len(parallel[field])}")

    shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        print(f"\nFAIL: {failures}")
        return 1
    print(f"\nPASS: workers=1 and workers={PARALLEL_WORKERS} are byte-identical "
          f"across {ranges}+ real chunks per source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
