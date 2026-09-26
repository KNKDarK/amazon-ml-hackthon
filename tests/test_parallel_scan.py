#!/usr/bin/env python3
"""Unit tests for the parallel scan primitives: sizing and byte-range splitting."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pilot import parallel_scan as ps  # noqa: E402


class SizingTests(unittest.TestCase):
    def test_explicit_budget_wins(self) -> None:
        self.assertEqual(ps.ram_budget_bytes(8 * 1024 ** 3, 2.0), 2 * 1024 ** 3)

    def test_auto_budget_is_a_fraction_of_available(self) -> None:
        # Simulates a 7.5 GiB-free laptop: 60% of it, and 10 workers must fit.
        available = int(7.5 * 1024 ** 3)
        budget = ps.ram_budget_bytes(available, 0.0)
        self.assertAlmostEqual(budget / available, 0.60, places=2)
        self.assertEqual(ps.resolve_workers(10, budget), 10)

    def test_tight_budget_clamps_workers(self) -> None:
        # 1 GiB cannot host 10 workers at 320 MiB each.
        self.assertEqual(ps.resolve_workers(10, 1024 ** 3), 3)

    def test_cpu_ceiling_clamps_absurd_request(self) -> None:
        import os
        budget = 64 * 1024 ** 3
        self.assertEqual(ps.resolve_workers(9999, budget), max(1, os.cpu_count() or 1))

    def test_unknown_availability_falls_back_to_one_worker_budget(self) -> None:
        # Zero means "kernel would not tell us"; must not plan a big pool.
        self.assertEqual(ps.ram_budget_bytes(0, 0.0), ps.BYTES_PER_WORKER)

    def test_rejects_bad_worker_count(self) -> None:
        with self.assertRaises(ValueError):
            ps.resolve_workers(0, 1024 ** 3)


class ByteRangeTests(unittest.TestCase):
    def _write(self, path: Path, lines: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
            for line in lines:
                handle.write(line + "\n")

    def test_ranges_tile_the_file_without_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.tsv"
            self._write(path, [f"S2-{i}\tName {i}\t12 Main Street\tUS" for i in range(20000)])
            total = path.stat().st_size
            for chunks in (1, 3, 7, 40):
                ranges = ps.plan_byte_ranges(path, chunks)
                self.assertEqual(ranges[0][0], 0, chunks)
                self.assertEqual(ranges[-1][1], total, chunks)
                for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:]):
                    self.assertEqual(prev_end, next_start, f"gap/overlap at {chunks}")

    def test_chunked_reads_reproduce_the_whole_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.tsv"
            rows = [f"S2-{i}\tCorp {i} Ltd\t{i} Park Road\tIndia" for i in range(5000)]
            self._write(path, rows)
            total = path.stat().st_size
            reference = [
                (r["entity_id"], r["business_name"], r["business_address"], r["country"])
                for r in ps.iter_chunk_rows(path, 0, total)
            ]
            self.assertEqual(len(reference), 5000)

            # The production floor is 4 MiB and this file is only ~200 KiB, so
            # plan_byte_ranges would collapse every request to a *single* range
            # and this test would pass without ever splitting a file. Lower the
            # floor so the splits are real, and assert they happened.
            original = ps.MIN_CHUNK_BYTES
            ps.MIN_CHUNK_BYTES = 8 * 1024
            try:
                for chunks in (2, 5, 16, 33):
                    ranges = ps.plan_byte_ranges(path, chunks)
                    self.assertGreater(
                        len(ranges), 1,
                        f"chunks={chunks} collapsed to one range; the test would be vacuous",
                    )
                    merged = []
                    for start, end in ranges:
                        merged.extend(ps.iter_chunk_rows(path, start, end))
                    got = [(r["entity_id"], r["business_name"],
                            r["business_address"], r["country"]) for r in merged]
                    # A row lost or duplicated at a range boundary is the exact
                    # regression this guards: no gaps, no double counting, same
                    # order as the serial read.
                    self.assertEqual(got, reference, f"chunks={chunks}")
            finally:
                ps.MIN_CHUNK_BYTES = original

    def test_no_row_is_lost_or_duplicated_at_any_split_point(self) -> None:
        """Regression: one row used to vanish at every internal boundary."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.tsv"
            self._write(path, [f"S2-{i}\tCorp {i} Ltd\t{i} Park Road\tIndia"
                               for i in range(3000)])
            total = path.stat().st_size
            reference = [r["entity_id"] for r in ps.iter_chunk_rows(path, 0, total)]

            original = ps.MIN_CHUNK_BYTES
            ps.MIN_CHUNK_BYTES = 1  # chunk size == 1 byte, so splits land everywhere
            try:
                for chunks in (2, 3, 4, 7, 11, 32, 100):
                    ranges = ps.plan_byte_ranges(path, chunks)
                    merged = []
                    for start, end in ranges:
                        merged.extend(r["entity_id"] for r in ps.iter_chunk_rows(path, start, end))
                    self.assertEqual(
                        merged, reference,
                        f"chunks={chunks}: expected {len(reference)} rows in order, "
                        f"got {len(merged)} (lost {len(reference) - len(merged)}, "
                        f"duplicated {len(merged) - len(set(merged))})",
                    )
            finally:
                ps.MIN_CHUNK_BYTES = original

    def test_split_points_onto_record_boundaries_keep_every_row(self) -> None:
        """A boundary landing exactly on a record start must not swallow a row."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.tsv"
            rows = [f"S2-{i}\tCorp {i} Ltd\t{i} Park Road\tIndia" for i in range(500)]
            self._write(path, rows)
            total = path.stat().st_size
            reference = [r["entity_id"] for r in ps.iter_chunk_rows(path, 0, total)]

            # Real record-start offsets, so every interior boundary below sits
            # exactly on one. The old reader seeked to `start` and consumed a
            # whole line whenever that happened.
            starts = []
            with path.open("rb") as handle:
                offset = 0
                for raw in handle:
                    starts.append(offset)
                    offset += len(raw)
            self.assertEqual(starts[0], 0)
            self.assertEqual(offset, total)
            interior = starts[1:-1]
            self.assertGreater(len(interior), 20)

            for cut in (1, 2, 3, 7, 13, len(interior) // 2, len(interior) - 1):
                bounds = [(0, interior[cut - 1]), (interior[cut - 1], total)]
                merged = []
                for start, end in bounds:
                    merged.extend(r["entity_id"] for r in ps.iter_chunk_rows(path, start, end))
                self.assertEqual(merged, reference, f"cut at record start #{cut}")

            # And a multi-way split on record starts.
            picks = [interior[i] for i in range(0, len(interior), max(1, len(interior) // 9))]
            bounds = list(zip([0] + picks, picks + [total]))
            merged = []
            for start, end in bounds:
                merged.extend(r["entity_id"] for r in ps.iter_chunk_rows(path, start, end))
            self.assertEqual(merged, reference, f"{len(bounds)} aligned ranges")

    def test_empty_file_yields_no_ranges(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.tsv"
            path.write_bytes(b"")
            self.assertEqual(ps.plan_byte_ranges(path, 8), [])

    def test_bom_is_tolerated_only_at_offset_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.tsv"
            with path.open("w", encoding="utf-8-sig", newline="\n") as handle:
                handle.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                handle.write("S2-1\tAcme\t12 Main Street\tUS\n")
            rows = list(ps.iter_chunk_rows(path, 0, path.stat().st_size))
            self.assertEqual(rows[0]["entity_id"], "S2-1")

    def test_missing_trailing_newline_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.tsv"
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
                handle.write("S2-1\tAcme\t12 Main Street\tUS\n")
                handle.write("S2-2\tBeta\t99 Park Road\tIndia")  # no final newline
            rows = list(ps.iter_chunk_rows(path, 0, path.stat().st_size))
            self.assertEqual([r["entity_id"] for r in rows], ["S2-1", "S2-2"])

    def test_schema_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "t.tsv"
            path.write_text("a\tb\tc\td\n1\t2\t3\t4\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                list(ps.iter_chunk_rows(path, 0, path.stat().st_size))


class ThreadEnvTests(unittest.TestCase):
    def test_caps_are_applied_and_existing_values_win(self) -> None:
        import os
        from pilot import thread_env
        for name in thread_env.THREAD_ENV_VARS:
            os.environ.pop(name, None)
        caps = thread_env.cap_native_threads(1)
        self.assertEqual(set(caps.values()), {"1"})
        os.environ["OMP_NUM_THREADS"] = "4"
        self.assertEqual(thread_env.cap_native_threads(1)["OMP_NUM_THREADS"], "4")
        os.environ.pop("OMP_NUM_THREADS", None)

    def test_rejects_zero_threads(self) -> None:
        from pilot import thread_env
        with self.assertRaises(ValueError):
            thread_env.cap_native_threads(0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
