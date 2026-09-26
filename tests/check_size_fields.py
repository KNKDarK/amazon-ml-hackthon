#!/usr/bin/env python3
"""Pin down which candidate_stats.json field differs between workers=1 and 10.

The full CLI check showed identical SQLite *contents* but a differing stats
file, so this isolates the physical-size fields that depend on B-tree page
layout rather than on logical content.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_parallel_equivalence import (  # noqa: E402
    ROWS,
    build_subset,
    labelled_source1_subset,
    write_truth_subset,
)
from pilot import run_pilot as rp  # noqa: E402


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="size_check_"))
    data_root = tmp / "dataset"
    train = data_root / "train"
    train.mkdir(parents=True, exist_ok=True)
    real = ROOT / "dataset" / "train"
    kept = labelled_source1_subset(train / "train_source1.tsv",
                                   real / "train_source1.tsv",
                                   real / "train_ground_truth.tsv", ROWS)
    write_truth_subset(train / "train_ground_truth.tsv",
                       real / "train_ground_truth.tsv", kept)
    build_subset(train / "train_source2.tsv", real / "train_source2.tsv", ROWS)
    build_subset(train / "train_source3.tsv", real / "train_source3.tsv", ROWS)

    work = tmp / "w"
    work.mkdir(parents=True, exist_ok=True)
    queries, truth, _ = rp.sample_queries(data_root, work, min(ROWS, 10_000))
    key_index = rp.build_query_key_index(queries)
    monitor = rp.MemoryMonitor()

    reports = {}
    for workers in (1, 10):
        counts, true_keys, _ = rp.measure_block_frequencies(
            data_root, queries, truth, key_index, monitor, workers)
        active_index, retained, _ = rp.select_block_postings(
            queries, key_index, counts, true_keys, truth, 50_000_000, 500, ROWS * 2)
        del retained
        wd = tmp / f"w{workers}"
        wd.mkdir(parents=True, exist_ok=True)
        path, stats = rp.persist_candidates(
            data_root, wd, queries, truth, active_index, monitor, 10_000, workers)
        rows = sqlite3.connect(path).execute(
            "SELECT qrow,target_id,source,name_norm,address_norm,country,block_mask "
            "FROM candidates ORDER BY qrow,target_id").fetchall()
        connection = sqlite3.connect(path)
        pages = connection.execute("PRAGMA page_count").fetchone()[0]
        size = connection.execute("PRAGMA page_size").fetchone()[0]
        freelist = connection.execute("PRAGMA freelist_count").fetchone()[0]
        connection.close()
        reports[workers] = (stats, rows, pages, size, freelist, path.stat().st_size)

    a, b = reports[1], reports[10]
    print(f"logical rows identical: {a[1] == b[1]}  ({len(a[1]):,} rows)")
    print(f"{'field':<42}{'workers=1':>16}{'workers=10':>16}")
    for label, ia, ib in (
        ("candidate_sqlite_bytes", a[5], b[5]),
        ("candidate_sqlite_logical_page_bytes", a[2] * a[3], b[2] * b[3]),
        ("page_count", a[2], b[2]),
        ("page_size", a[3], b[3]),
        ("freelist_count", a[4], b[4]),
    ):
        flag = "" if ia == ib else "   <-- DIFFERS"
        print(f"{label:<42}{ia:>16,}{ib:>16,}{flag}")

    def walk(x, y, path=""):
        if isinstance(x, dict) and isinstance(y, dict):
            for k in sorted(set(x) | set(y)):
                walk(x.get(k), y.get(k), f"{path}.{k}")
        elif x != y:
            print(f"  stats diff {path}: w1={x!r} w10={y!r}")
    print("\nstats JSON differences:")
    walk(a[0], b[0])

    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
