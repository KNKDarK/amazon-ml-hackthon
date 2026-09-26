#!/usr/bin/env python3
"""Memory-safe merge for sharded worker results.

Assumes the corrected workers use disjoint S1 query shards. Streams candidate
and matched target IDs from each worker SQLite database, performs a 16-way
qid merge, and writes exactly one row for every S1 entity in test_source1.tsv.
It does NOT load the candidate corpus into Python memory.
"""

from __future__ import annotations

import argparse
import heapq
import sqlite3
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--work-dir",
        type=Path,
        default=Path("artifacts/full_inference_20260926_sharded16_sharedindex"),
    )
    p.add_argument("--test-s1", type=Path, default=Path("dataset/test/test_source1.tsv"))
    p.add_argument("--output-dir", type=Path, default=Path("output"))
    return p.parse_args()


def iter_s1_ids(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\r\n")
        if header.split("\t")[:1] != ["entity_id"]:
            raise SystemExit(f"Unexpected S1 header in {path}: {header!r}")
        for line in f:
            line = line.rstrip("\r\n")
            if not line:
                continue
            yield line.split("\t", 1)[0].strip()


def open_worker_dbs(work_dir: Path):
    dbs = []
    for worker_dir in sorted(work_dir.glob("run_worker_*"), key=lambda p: int(p.name.split("_")[-1])):
        db_path = worker_dir / "results.sqlite"
        if not db_path.exists():
            continue
        conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        conn.execute("PRAGMA query_only=ON")
        dbs.append((worker_dir.name, conn))
    if not dbs:
        raise SystemExit(f"No worker results.sqlite found under {work_dir}")
    return dbs


def worker_stream(conn: sqlite3.Connection, table: str):
    """Yield (qid, source1_id, [target_ids]) one qid at a time."""
    query = (
        f"SELECT q.qid, q.id, p.target_id "
        f"FROM queries q LEFT JOIN {table} p ON p.qid=q.qid "
        f"ORDER BY q.qid, p.target_id"
    )
    cur = conn.execute(query)
    current_qid = None
    current_id = None
    values = []

    for qid, source1_id, target_id in cur:
        if current_qid is not None and qid != current_qid:
            yield current_qid, current_id, values
            values = []
        current_qid = qid
        current_id = source1_id
        if target_id is not None:
            values.append(target_id)

    if current_qid is not None:
        yield current_qid, current_id, values


def merge_worker_streams(streams):
    """K-way merge. Raises on duplicate qids across workers."""
    heap = []
    for worker_idx, stream in enumerate(streams):
        try:
            item = next(stream)
        except StopIteration:
            continue
        heapq.heappush(heap, (item[0], worker_idx, item[1], item[2]))

    last_qid = 0
    while heap:
        qid, worker_idx, source1_id, targets = heapq.heappop(heap)
        if qid == last_qid:
            raise RuntimeError(
                f"Duplicate qid {qid} found across worker databases; "
                "the workers are not uniquely sharded."
            )
        if qid < last_qid:
            raise RuntimeError("Worker qids are not globally ordered")
        last_qid = qid
        yield qid, source1_id, targets

        try:
            item = next(streams[worker_idx])
        except StopIteration:
            continue
        heapq.heappush(heap, (item[0], worker_idx, item[1], item[2]))


def write_submission(
    *,
    work_dir: Path,
    test_s1: Path,
    output_path: Path,
    table: str,
    header: str,
):
    dbs = open_worker_dbs(work_dir)
    streams = [worker_stream(conn, table) for _, conn in dbs]
    merged = merge_worker_streams(streams)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")

    expected_qid = 1
    emitted = 0
    matched_rows = 0
    result = next(merged, None)

    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as out:
            out.write(header)
            for s1_id in iter_s1_ids(test_s1):
                if result is not None and result[0] < expected_qid:
                    raise RuntimeError(
                        f"Worker result qid {result[0]} is behind expected qid {expected_qid}"
                    )

                if result is not None and result[0] == expected_qid:
                    result_qid, result_s1_id, targets = result
                    if result_s1_id != s1_id:
                        raise RuntimeError(
                            f"S1 ID mismatch at qid {expected_qid}: "
                            f"dataset={s1_id!r}, worker={result_s1_id!r}"
                        )
                    values = sorted(set(targets))
                    if values:
                        matched_rows += 1
                    out.write(f"{s1_id}\t{','.join(values)}\n")
                    result = next(merged, None)
                else:
                    out.write(f"{s1_id}\t\n")

                emitted += 1
                expected_qid += 1

            if result is not None:
                raise RuntimeError(
                    f"Worker databases contain qid {result[0]} beyond the end "
                    f"of test_source1.tsv (which ended at qid {emitted})."
                )

            out.flush()

        tmp.replace(output_path)
    finally:
        for _, conn in dbs:
            conn.close()
        if tmp.exists():
            # If replace did not happen because of an exception, leave no partial tmp.
            try:
                tmp.unlink()
            except OSError:
                pass

    print(
        f"{table}: wrote {emitted:,} S1 rows; "
        f"{matched_rows:,} rows with non-empty predictions -> {output_path}"
    )
    return emitted, matched_rows


def main():
    a = parse_args()
    s1_count = sum(1 for _ in iter_s1_ids(a.test_s1))
    print(f"Required S1 rows: {s1_count:,}")

    matching_path = a.output_dir / "matching_results.tsv"
    candidates_path = a.output_dir / "candidate_pairs.tsv"

    matching_rows, _ = write_submission(
        work_dir=a.work_dir,
        test_s1=a.test_s1,
        output_path=matching_path,
        table="pairs",
        header="source1_entity_id\tmatched_entity_ids\n",
    )
    candidate_rows, _ = write_submission(
        work_dir=a.work_dir,
        test_s1=a.test_s1,
        output_path=candidates_path,
        table="candidates",
        header="source1_entity_id\tcandidate_entity_ids\n",
    )

    if matching_rows != s1_count or candidate_rows != s1_count:
        raise SystemExit(
            f"Coverage error: expected {s1_count:,} rows in each output, "
            f"got matching={matching_rows:,}, candidates={candidate_rows:,}"
        )

    print("Streaming merge completed successfully.")


if __name__ == "__main__":
    main()
