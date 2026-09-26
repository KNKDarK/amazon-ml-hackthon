#!/usr/bin/env python3
"""Tests for the sharded-merge duplicate/coverage guards.

The merger previously skipped any worker directory without a results.sqlite and
then compared S1 rows written against S1 rows read, which holds even when an
entire shard is missing. These tests pin down that a missing shard, a repeated
query_offset, a disagreeing stride or an incomplete shard all fail loudly
instead of producing a partial or duplicated submission.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import merge_worker_outputs as mw  # noqa: E402

S1_ROWS = 40


def write_s1(path: Path, rows: int = S1_ROWS) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for i in range(rows):
            f.write(f"S1-{i}\tCorp {i}\t{i} Main St\tUS\n")


def build_shard(
    work_dir: Path,
    index: int,
    offset: int,
    stride: int,
    s1_rows: int = S1_ROWS,
    qids: list[int] | None = None,
    profile_stride: int | None = None,
    with_checkpoint: bool = True,
    empty_db: bool = False,
) -> Path:
    """Create one ``run_worker_<index>`` shard directory."""
    worker = work_dir / f"run_worker_{index}"
    worker.mkdir(parents=True, exist_ok=True)

    if with_checkpoint:
        state = {
            "version": 2,
            "identity": {
                "profile": {
                    "target_sample_rate": 1,
                    "query_stride": stride if profile_stride is None else profile_stride,
                    "query_offset": offset,
                    "block_cap": 500,
                    "query_posting_cap": 500,
                }
            },
        }
        (worker / "checkpoint.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    db_path = worker / "results.sqlite"
    if empty_db:
        db_path.write_bytes(b"")
        return worker

    if qids is None:
        qids = [qi + 1 for qi in range(offset, s1_rows, stride)]

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """CREATE TABLE IF NOT EXISTS queries(qid INTEGER PRIMARY KEY,id TEXT UNIQUE,
                  name TEXT,address TEXT,country TEXT);
           CREATE TABLE IF NOT EXISTS pairs(qid INTEGER,target_id TEXT,score REAL,
                  PRIMARY KEY(qid,target_id)) WITHOUT ROWID;
           CREATE TABLE IF NOT EXISTS candidates(qid INTEGER,target_id TEXT,
                  PRIMARY KEY(qid,target_id)) WITHOUT ROWID;"""
    )
    for qid in qids:
        conn.execute("INSERT OR IGNORE INTO queries VALUES(?,?,?,?,?)",
                     (qid, f"S1-{qid - 1}", "n", "a", "US"))
        conn.execute("INSERT OR IGNORE INTO candidates VALUES(?,?)", (qid, f"S2-{qid}"))
    conn.commit()
    conn.close()
    return worker


def build_complete_set(work_dir: Path, workers: int = 4, s1_rows: int = S1_ROWS) -> None:
    for w in range(workers):
        build_shard(work_dir, w, w, workers, s1_rows=s1_rows)


class GuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name)
        self.work = self.base / "sharded"
        self.work.mkdir()
        self.s1 = self.base / "test_source1.tsv"
        write_s1(self.s1)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    # ---- discover_worker_dirs -------------------------------------------
    def test_complete_set_is_accepted(self) -> None:
        build_complete_set(self.work)
        dirs = mw.discover_worker_dirs(self.work, 4)
        self.assertEqual([d.name for d in dirs],
                         ["run_worker_0", "run_worker_1", "run_worker_2", "run_worker_3"])

    def test_missing_worker_is_rejected(self) -> None:
        build_complete_set(self.work)
        with self.assertRaises(SystemExit) as ctx:
            mw.discover_worker_dirs(self.work, 6)
        self.assertIn("Expected 6 shard", str(ctx.exception))

    def test_empty_results_db_is_rejected(self) -> None:
        build_complete_set(self.work)
        (self.work / "run_worker_2" / "results.sqlite").write_bytes(b"")
        with self.assertRaises(SystemExit) as ctx:
            mw.discover_worker_dirs(self.work, 4)
        self.assertIn("no usable results.sqlite", str(ctx.exception))

    def test_non_contiguous_numbering_is_rejected(self) -> None:
        build_shard(self.work, 0, 0, 4)
        build_shard(self.work, 1, 1, 4)
        build_shard(self.work, 3, 3, 4)   # 2 is missing
        with self.assertRaises(SystemExit) as ctx:
            mw.discover_worker_dirs(self.work, 0)
        self.assertIn("not contiguous", str(ctx.exception))

    def test_unnumbered_worker_dir_is_rejected(self) -> None:
        build_complete_set(self.work)
        (self.work / "run_worker_extra").mkdir()
        with self.assertRaises(SystemExit) as ctx:
            mw.discover_worker_dirs(self.work, 0)
        self.assertIn("Cannot read a worker number", str(ctx.exception))

    # ---- verify_shard_plan ----------------------------------------------
    def test_duplicate_offset_is_rejected(self) -> None:
        build_shard(self.work, 0, 0, 4)
        build_shard(self.work, 1, 0, 4)   # same offset as worker 0
        dirs = mw.discover_worker_dirs(self.work, 0)
        with self.assertRaises(SystemExit) as ctx:
            mw.verify_shard_plan(dirs, S1_ROWS)
        self.assertIn("both use query_offset=0", str(ctx.exception))

    def test_disagreeing_stride_is_rejected(self) -> None:
        build_shard(self.work, 0, 0, 4)
        build_shard(self.work, 1, 1, 4)
        build_shard(self.work, 2, 2, 4)
        build_shard(self.work, 3, 3, 8, profile_stride=8)   # disagrees
        dirs = mw.discover_worker_dirs(self.work, 0)
        with self.assertRaises(SystemExit) as ctx:
            mw.verify_shard_plan(dirs, S1_ROWS)
        self.assertIn("disagree on query_stride", str(ctx.exception))

    def test_offsets_not_covering_full_stride_is_rejected(self) -> None:
        # stride 4 but only offsets 0,1,2 present -> row 3 unowned
        build_shard(self.work, 0, 0, 4)
        build_shard(self.work, 1, 1, 4)
        build_shard(self.work, 2, 2, 4)
        dirs = mw.discover_worker_dirs(self.work, 0)
        with self.assertRaises(SystemExit) as ctx:
            mw.verify_shard_plan(dirs, S1_ROWS)
        self.assertIn("do not cover 0..3", str(ctx.exception))

    def test_missing_checkpoint_is_rejected(self) -> None:
        build_complete_set(self.work)
        (self.work / "run_worker_1" / "checkpoint.json").unlink()
        dirs = mw.discover_worker_dirs(self.work, 4)
        with self.assertRaises(SystemExit) as ctx:
            mw.verify_shard_plan(dirs, S1_ROWS)
        self.assertIn("no checkpoint.json", str(ctx.exception))

    # ---- verify_qid_coverage --------------------------------------------
    def test_complete_coverage_passes(self) -> None:
        build_complete_set(self.work)
        dirs = mw.discover_worker_dirs(self.work, 4)
        plan = mw.verify_shard_plan(dirs, S1_ROWS)
        mw.verify_qid_coverage(dirs, plan, S1_ROWS)   # must not raise

    def test_incomplete_shard_is_rejected(self) -> None:
        # worker 2 is missing two of the qids it owns
        build_shard(self.work, 0, 0, 4)
        build_shard(self.work, 1, 1, 4)
        build_shard(self.work, 2, 2, 4, qids=[3, 7])          # 11 and 15 missing
        build_shard(self.work, 3, 3, 4)
        dirs = mw.discover_worker_dirs(self.work, 4)
        plan = mw.verify_shard_plan(dirs, S1_ROWS)
        with self.assertRaises(SystemExit) as ctx:
            mw.verify_qid_coverage(dirs, plan, S1_ROWS)
        self.assertIn("should hold", str(ctx.exception))

    def test_right_count_wrong_residue_is_rejected(self) -> None:
        """Correct row count, but the rows are not this shard's residue class.

        This is the case that would silently duplicate another shard's rows.
        """
        build_shard(self.work, 0, 0, 4)
        # offset 1 must own qids 2,6,10,...; give it 10 rows that are not.
        build_shard(self.work, 1, 1, 4, qids=[qi + 1 for qi in range(10)])
        build_shard(self.work, 2, 2, 4)
        build_shard(self.work, 3, 3, 4)
        dirs = mw.discover_worker_dirs(self.work, 4)
        plan = mw.verify_shard_plan(dirs, S1_ROWS)
        with self.assertRaises(SystemExit) as ctx:
            mw.verify_qid_coverage(dirs, plan, S1_ROWS)
        self.assertIn("outside its shard", str(ctx.exception))

    def test_overlapping_qids_across_shards_are_rejected(self) -> None:
        # worker 1 claims offset 1 but holds too few rows
        build_shard(self.work, 0, 0, 4)
        build_shard(self.work, 1, 1, 4, qids=[1, 2, 3, 4])
        build_shard(self.work, 2, 2, 4)
        build_shard(self.work, 3, 3, 4)
        dirs = mw.discover_worker_dirs(self.work, 4)
        plan = mw.verify_shard_plan(dirs, S1_ROWS)
        with self.assertRaises(SystemExit):
            mw.verify_qid_coverage(dirs, plan, S1_ROWS)

    def test_out_of_range_qid_is_rejected(self) -> None:
        build_complete_set(self.work)
        # push worker 3 past the end of the file
        conn = sqlite3.connect(self.work / "run_worker_3" / "results.sqlite")
        conn.execute("INSERT INTO queries VALUES(?,?,?,?,?)", (9999, "S1-9998", "n", "a", "US"))
        conn.commit()
        conn.close()
        dirs = mw.discover_worker_dirs(self.work, 4)
        plan = mw.verify_shard_plan(dirs, S1_ROWS)
        with self.assertRaises(SystemExit):
            mw.verify_qid_coverage(dirs, plan, S1_ROWS)

    # ---- end to end through the CLI -------------------------------------
    def _run_cli(self, *extra: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(ROOT / "merge_worker_outputs.py"),
             "--work-dir", str(self.work),
             "--test-s1", str(self.s1),
             "--output-dir", str(self.base / "out"),
             "--expect-workers", "4", *extra],
            capture_output=True, text=True, cwd=str(ROOT),
        )

    def test_cli_happy_path_writes_both_files(self) -> None:
        build_complete_set(self.work)
        result = self._run_cli()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = self.base / "out"
        for name, header in (("matching_results.tsv", "source1_entity_id\tmatched_entity_ids"),
                            ("candidate_pairs.tsv", "source1_entity_id\tcandidate_entity_ids")):
            lines = (out / name).read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0], header)
            self.assertEqual(len(lines) - 1, S1_ROWS)
        self.assertIn("Coverage OK", result.stdout)

    def test_cli_missing_shard_writes_nothing(self) -> None:
        build_complete_set(self.work)
        (self.work / "run_worker_3" / "results.sqlite").unlink()
        result = self._run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("no usable results.sqlite", result.stdout + result.stderr)
        # nothing may be written when validation fails
        self.assertFalse((self.base / "out" / "matching_results.tsv").exists())
        self.assertFalse((self.base / "out" / "candidate_pairs.tsv").exists())

    def test_cli_duplicate_offset_writes_nothing(self) -> None:
        build_shard(self.work, 0, 0, 4)
        build_shard(self.work, 1, 0, 4)
        build_shard(self.work, 2, 2, 4)
        build_shard(self.work, 3, 3, 4)
        result = self._run_cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("both use query_offset=0", result.stdout + result.stderr)
        self.assertFalse((self.base / "out" / "matching_results.tsv").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
