#!/usr/bin/env python3
"""End-to-end CLI check: --workers 10 must reproduce --workers 1 exactly.

Runs the real ``run_pilot.py`` twice over a corpus subset built from the actual
training data and diffs every artifact, ignoring only the fields that are
inherently non-deterministic (wall time, RSS, available memory).

The subset is a few MiB, below the production ``MIN_CHUNK_BYTES`` floor, so the
child is given a lowered floor via the environment; otherwise it would plan a
single range and the "parallel" run would execute serially in one task while
still reporting PASS. ``assert_really_chunked`` fails the run outright if the
subset ever stops splitting.
"""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pilot import parallel_scan as ps  # noqa: E402
from check_parallel_equivalence import (  # noqa: E402
    CHUNKS_PER_WORKER,
    MIN_RANGES,
    PARALLEL_WORKERS,
    ROWS,
    TEST_MIN_CHUNK_BYTES,
    build_subset,
    labelled_source1_subset,
    write_truth_subset,
)

# Fields that legitimately differ between two runs of the same pipeline.
VOLATILE = {
    "seconds", "phase_seconds", "memory", "peak_process_rss_bytes",
    "peak_process_rss_mib", "minimum_system_mem_available_bytes",
    "minimum_system_mem_available_mib", "start_available_memory_bytes",
    "start_available_memory_gib", "parallelism", "worker", "workers",
    "requested_workers", "ram_budget_bytes", "sqlite_cache_mib",
    "native_thread_caps", "projection", "blocking_stats", "generated_at",
    "elapsed", "duration", "throughput", "peak_work_bytes",
    # Physical SQLite size tracks B-tree page layout, which depends on batch
    # boundaries and insertion order, not on stored content. Chunked parallel
    # inserts pack marginally tighter (measured: 33 pages / 0.33% smaller on a
    # 2,977,349-row table). The row-level dumps below are the real assertion.
    "candidate_sqlite_bytes", "candidate_sqlite_logical_page_bytes",
    "feature_sqlite_bytes", "feature_sqlite_logical_page_bytes",
    "score_sqlite_bytes",
}


def strip(value):
    if isinstance(value, dict):
        return {k: strip(v) for k, v in value.items() if k not in VOLATILE}
    if isinstance(value, list):
        return [strip(v) for v in value]
    return value


def table_dump(path: Path, table: str) -> list:
    connection = sqlite3.connect(path)
    try:
        return connection.execute(
            f"SELECT * FROM {table} ORDER BY 1,2"
        ).fetchall()
    finally:
        connection.close()


def run(work: Path, data_root: Path, workers: int) -> dict:
    command = [
        sys.executable, str(ROOT / "pilot/run_pilot.py"),
        "--data-root", str(data_root),
        "--work-dir", str(work),
        "--sample-size", str(min(ROWS, 10_000)),
        "--selection-cap", "500",
        "--max-projected-postings", "50000000",
        "--negative-per-query", "30",
        "--batch-size", "10000",
        "--workers", str(workers),
    ]
    env = dict(os.environ)
    # Make the subset actually split. The production floor is 4 MiB and the
    # subset is only a few MiB, so without this the child plans a single range
    # and "workers=10" runs serially in one task -- the comparison below would
    # pass while proving nothing about the parallel path.
    env[ps.MIN_CHUNK_BYTES_ENV] = str(TEST_MIN_CHUNK_BYTES)
    result = subprocess.run(command, capture_output=True, text=True, cwd=ROOT, env=env)
    if result.returncode != 0:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        raise SystemExit(f"workers={workers} exited {result.returncode}")
    return {}


def assert_really_chunked(train: Path, workers: int) -> int:
    """Refuse to report PASS unless the child would have used several ranges."""
    # The parent must plan with the same floor the child will use, otherwise
    # this guard measures the production 4 MiB floor and always sees one range.
    previous = os.environ.get(ps.MIN_CHUNK_BYTES_ENV)
    os.environ[ps.MIN_CHUNK_BYTES_ENV] = str(TEST_MIN_CHUNK_BYTES)
    try:
        counts = {
            path.name: len(ps.plan_byte_ranges(path, workers * CHUNKS_PER_WORKER))
            for path in sorted(train.glob("train_source[23].tsv"))
        }
    finally:
        if previous is None:
            os.environ.pop(ps.MIN_CHUNK_BYTES_ENV, None)
        else:
            os.environ[ps.MIN_CHUNK_BYTES_ENV] = previous
    if not counts:
        raise SystemExit("no source2/source3 subset found; cannot verify chunking")
    smallest = min(counts.values())
    if smallest < MIN_RANGES:
        raise SystemExit(
            f"REFUSING to report PASS: the subset only splits into {smallest} "
            f"range(s) at workers={workers} ({counts}), so the parallel path never "
            f"ran. Raise ROWS or lower TEST_MIN_CHUNK_BYTES so each source yields "
            f">= {MIN_RANGES} chunks."
        )
    return smallest


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="cli_check_"))
    data_root = tmp / "dataset"
    train = data_root / "train"
    train.mkdir(parents=True, exist_ok=True)
    real = ROOT / "dataset" / "train"

    print(f"building {ROWS:,}-row subset")
    kept = labelled_source1_subset(train / "train_source1.tsv",
                                   real / "train_source1.tsv",
                                   real / "train_ground_truth.tsv", ROWS)
    write_truth_subset(train / "train_ground_truth.tsv",
                       real / "train_ground_truth.tsv", kept)
    build_subset(train / "train_source2.tsv", real / "train_source2.tsv", ROWS)
    build_subset(train / "train_source3.tsv", real / "train_source3.tsv", ROWS)

    ranges = assert_really_chunked(train, PARALLEL_WORKERS)
    print(f"verified real chunking: >= {ranges} byte ranges per source at "
          f"workers={PARALLEL_WORKERS} (child floor {TEST_MIN_CHUNK_BYTES} B)")

    reports = {}
    for workers in (1, PARALLEL_WORKERS):
        work = tmp / f"w{workers}"
        print(f"--- run_pilot.py --workers {workers} ---")
        run(work, data_root, workers)
        report = json.loads((work / "pilot_report.json").read_text())
        print(f"    macro F0.5={report['evaluation']['test']['macro_f05']:.6f} "
              f"candidates={report['candidate']['candidate_count']:,} "
              f"recall={report['candidate']['true_pair_recall']:.6f}")
        reports[workers] = (work, strip(report))

    failures = []
    serial, parallel = reports[1], reports[PARALLEL_WORKERS]

    def compare(label, left, right) -> None:
        same = left == right
        print(f"  {label:<34} {'MATCH' if same else 'DIFFER'}")
        if not same:
            failures.append(label)

    print("\nartifacts:")
    for name in ("candidate_stats.json", "blocking_stats.json",
                 "pilot_features.stats.json", "pilot_scores.stats.json"):
        a = strip(json.loads((serial[0] / name).read_text()))
        b = strip(json.loads((parallel[0] / name).read_text()))
        compare(name, a, b)

    print("\nreport sections:")
    for key in ("sample_summary", "blocking", "candidate", "feature",
                "training", "score", "evaluation", "queries"):
        compare(f"report.{key}", serial[1][key], parallel[1][key])

    print("\nmodel + output:")
    compare("model.json",
            strip(json.loads((serial[0] / "model.json").read_text())),
            strip(json.loads((parallel[0] / "model.json").read_text())))
    for name in ("pilot_queries.tsv", "pilot_labels.tsv", "pilot_features.tsv"):
        if (serial[0] / name).exists() and (parallel[0] / name).exists():
            compare(name, (serial[0] / name).read_bytes(),
                    (parallel[0] / name).read_bytes())

    print("\nsqlite contents:")
    for db, table in (("pilot_candidates.sqlite", "candidates"),
                      ("pilot_features.sqlite", "pair_features"),
                      ("pilot_scores.sqlite", "pair_scores")):
        if (serial[0] / db).exists() and (parallel[0] / db).exists():
            compare(f"{db}:{table}", table_dump(serial[0] / db, table),
                    table_dump(parallel[0] / db, table))

    if failures:
        print(f"\nFAIL: {failures}")
        return 1
    print(f"\nPASS: full CLI pipeline is identical at workers=1 and "
          f"workers={PARALLEL_WORKERS} across {ranges}+ real chunks per source")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
