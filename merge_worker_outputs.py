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
    p.add_argument(
        "--expect-workers",
        type=int,
        default=0,
        help="required number of run_worker_* directories. 0 infers the count "
             "from what is on disk, which cannot detect a worker that never ran.",
    )
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


def worker_index(worker_dir: Path) -> int:
    """Return the numeric suffix of a ``run_worker_<n>`` directory name."""
    tail = worker_dir.name.rsplit("_", 1)[-1]
    if not tail.isdigit():
        raise SystemExit(f"Cannot read a worker number from {worker_dir.name!r}")
    return int(tail)


def discover_worker_dirs(work_dir: Path, expect_workers: int) -> list[Path]:
    """Return every shard directory, failing if any is missing or incomplete.

    A worker directory without a readable ``results.sqlite`` used to be skipped
    silently, which produced a submission with blank rows for that whole shard
    and no error anywhere. Every expected shard must now be present.
    """
    if not work_dir.is_dir():
        raise SystemExit(f"Shard work directory does not exist: {work_dir}")

    found: dict[int, Path] = {}
    for worker_dir in sorted(work_dir.glob("run_worker_*")):
        if not worker_dir.is_dir():
            continue
        index = worker_index(worker_dir)
        if index in found:
            raise SystemExit(
                f"Two shard directories claim worker {index}: "
                f"{found[index].name} and {worker_dir.name}"
            )
        found[index] = worker_dir

    if not found:
        raise SystemExit(f"No run_worker_* directories found under {work_dir}")

    if expect_workers and len(found) != expect_workers:
        expected = set(range(expect_workers))
        missing = sorted(expected - set(found))
        extra = sorted(set(found) - expected)
        raise SystemExit(
            f"Expected {expect_workers} shard(s) under {work_dir}, found "
            f"{len(found)}. Missing worker index(es): {missing or 'none'}; "
            f"unexpected: {extra or 'none'}. Refusing to write a partial submission."
        )

    # 0..n-1 with no gaps, so offsets and directory numbering agree.
    gaps = sorted(set(range(len(found))) - set(found))
    if gaps:
        raise SystemExit(
            f"Shard numbering under {work_dir} is not contiguous from 0; "
            f"missing worker index(es): {gaps}"
        )

    incomplete = [
        d.name for _, d in sorted(found.items())
        if not (d / "results.sqlite").is_file()
        or (d / "results.sqlite").stat().st_size == 0
    ]
    if incomplete:
        raise SystemExit(
            f"Shard(s) with no usable results.sqlite: {incomplete}. A worker most "
            f"likely crashed or was never launched; refusing to write a partial "
            f"submission."
        )
    return [found[i] for i in sorted(found)]


def verify_shard_plan(worker_dirs: list[Path], s1_count: int) -> list[sqlite3.Connection]:
    """Open each shard read-only after checking the shards cannot overlap.

    Guards against two workers claiming the same ``query_offset`` (which would
    duplicate every S1 row they both processed) and against shards that do not
    jointly cover the file. Runs before any output is written.
    """
    import json

    offsets: dict[int, str] = {}
    strides: set[int] = set()
    for worker_dir in worker_dirs:
        checkpoint = worker_dir / "checkpoint.json"
        if not checkpoint.is_file():
            raise SystemExit(f"Shard {worker_dir.name} has no checkpoint.json")
        try:
            state = json.loads(checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Shard {worker_dir.name} has an unreadable checkpoint: {exc}") from exc
        profile = state.get("identity", {}).get("profile") or state.get("profile") or {}
        offset, stride = profile.get("query_offset"), profile.get("query_stride")
        if offset is None or stride is None:
            raise SystemExit(
                f"Shard {worker_dir.name} checkpoint records no query_offset/"
                f"query_stride; cannot prove the shards are disjoint"
            )
        if offset in offsets:
            raise SystemExit(
                f"Shards {offsets[offset]} and {worker_dir.name} both use "
                f"query_offset={offset}; their S1 rows would be duplicated in the "
                f"submission"
            )
        offsets[offset] = worker_dir.name
        strides.add(stride)

    if len(strides) != 1:
        raise SystemExit(f"Shards disagree on query_stride: {sorted(strides)}")
    stride = strides.pop()
    if sorted(offsets) != list(range(stride)):
        raise SystemExit(
            f"Shard offsets {sorted(offsets)} do not cover 0..{stride - 1} exactly; "
            f"the shards cannot partition the S1 file"
        )

    # Ownership is exact by construction: every qi has precisely one owner.
    expected = [0] * stride
    for offset in range(stride):
        expected[offset] = len(range(offset, s1_count, stride))
    print(
        f"Shard plan OK: {len(worker_dirs)} worker(s), stride={stride}, "
        f"offsets={sorted(offsets)}, S1 rows={s1_count:,}; "
        f"per-shard rows={[expected[o] for o in sorted(offsets)]}"
    )

    offset_of = {name: offset for offset, name in offsets.items()}
    return [(offset_of[d.name], stride) for d in worker_dirs]


def verify_qid_coverage(
    worker_dirs: list[Path],
    plan: list[tuple[int, int]],
    s1_count: int,
) -> None:
    """Require the shards to cover every S1 row exactly once, with no overlap.

    Done as SQL aggregates rather than by materialising every qid, so the check
    stays memory-safe at 1.7M rows.

    For a shard with offset ``o`` and stride ``n`` the owned rows are exactly
    ``{qi+1 : qi % n == o}``. If every row it holds is in range, has the right
    residue, is unique (qid is the primary key) and the count matches the size
    of that residue class, then its rows are *exactly* that class. Distinct
    offsets are distinct residue classes, so the shards are disjoint and their
    union is the whole file.
    """
    total = 0
    for worker_dir, (offset, stride) in zip(worker_dirs, plan):
        conn = sqlite3.connect(
            f"file:{(worker_dir / 'results.sqlite').resolve()}?mode=ro", uri=True
        )
        try:
            conn.execute("PRAGMA query_only=ON")
            count, lo, hi = conn.execute(
                "SELECT count(*), min(qid), max(qid) FROM queries"
            ).fetchone()
            count = int(count or 0)
            expected = len(range(offset, s1_count, stride))
            if count != expected:
                raise SystemExit(
                    f"{worker_dir.name} (query_offset={offset}) holds {count:,} rows "
                    f"but should hold {expected:,}; it is incomplete, or was run "
                    f"with a different --query-stride"
                )
            if count:
                stray = conn.execute(
                    "SELECT count(*) FROM queries WHERE qid < 1 OR qid > ? "
                    "OR (qid - 1) % ? != ?",
                    (s1_count, stride, offset),
                ).fetchone()[0]
                if stray:
                    raise SystemExit(
                        f"{worker_dir.name} (query_offset={offset}) contains "
                        f"{int(stray):,} row(s) outside its shard (qid must satisfy "
                        f"(qid-1) % {stride} == {offset}); it would duplicate or "
                        f"miss S1 rows"
                    )
                if (int(lo) - 1) % stride != offset or (int(hi) - 1) % stride != offset:
                    raise SystemExit(
                        f"{worker_dir.name} (query_offset={offset}) spans qid "
                        f"{int(lo)}..{int(hi)}, which is not a single residue "
                        f"class mod {stride}"
                    )
            total += count
        finally:
            conn.close()
    if total != s1_count:
        raise SystemExit(
            f"Shards hold {total:,} rows in total but the S1 file has {s1_count:,}; "
            f"refusing to write a partial submission"
        )
    print(
        f"Coverage OK: {total:,} S1 rows partitioned across {len(worker_dirs)} shard(s), "
        f"each row in exactly one shard."
    )


def open_worker_dbs(work_dir: Path):
    dbs = []
    for worker_dir in sorted(work_dir.glob("run_worker_*"), key=worker_index):
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
    dbs: list[Path] | None = None,
):
    # Each call opens and closes its own read-only handles: write_submission
    # closes what it opens, so a caller must never share connections with it.
    targets = dbs if dbs is not None else list(work_dir.glob("run_worker_*"))
    handles = []
    for worker_dir in sorted(targets, key=worker_index):
        if not worker_dir.is_dir():
            continue
        conn = sqlite3.connect(
            f"file:{(worker_dir / 'results.sqlite').resolve()}?mode=ro", uri=True
        )
        conn.execute("PRAGMA query_only=ON")
        handles.append((worker_dir.name, conn))
    if not handles:
        raise SystemExit(f"No worker results.sqlite found under {work_dir}")
    dbs = handles
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

    # Validate the shard set before writing anything. A duplicate offset or a
    # shard that never ran would otherwise be written out as blank or repeated
    # rows with no error, because the row-count check below only compares S1
    # rows written against S1 rows read.
    worker_dirs = discover_worker_dirs(a.work_dir, a.expect_workers)
    plan = verify_shard_plan(worker_dirs, s1_count)
    verify_qid_coverage(worker_dirs, plan, s1_count)

    matching_path = a.output_dir / "matching_results.tsv"
    candidates_path = a.output_dir / "candidate_pairs.tsv"

    matching_rows, _ = write_submission(
        work_dir=a.work_dir,
        test_s1=a.test_s1,
        output_path=matching_path,
        table="pairs",
        header="source1_entity_id\tmatched_entity_ids\n",
        dbs=worker_dirs,
    )
    candidate_rows, _ = write_submission(
        work_dir=a.work_dir,
        test_s1=a.test_s1,
        output_path=candidates_path,
        table="candidates",
        header="source1_entity_id\tcandidate_entity_ids\n",
        dbs=worker_dirs,
    )

    if matching_rows != s1_count or candidate_rows != s1_count:
        raise SystemExit(
            f"Coverage error: expected {s1_count:,} rows in each output, "
            f"got matching={matching_rows:,}, candidates={candidate_rows:,}"
        )

    print("Streaming merge completed successfully.")


if __name__ == "__main__":
    main()
