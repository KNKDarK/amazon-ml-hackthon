#!/usr/bin/env python3
"""Process-parallel full-corpus scanning for the two blocking passes.

``measure_block_frequencies`` and ``persist_candidates`` each stream the whole
``train_source2.tsv`` + ``train_source3.tsv`` pair (~1 GiB, ~10M rows) and call
``blocking_keys`` on every single row.  That call is pure Python -- regular
expressions, ``unicodedata`` folding and set arithmetic -- so it is GIL-bound
and OS threads would simply serialise it again.  Parallelism therefore has to be
process based.

Each worker opens the TSV itself and owns a byte range, so both the TSV parsing
and the key generation scale with the pool instead of leaving the CSV decode
bottlenecked in the parent.

Determinism
-----------
The parent submits chunks in ascending file order and merges results in that
same order, and every quantity the pipeline records downstream is order
independent:

* ``counts`` and the row/posting counters are integer sums.
* ``true_keys`` is a union of per-pair key sets.
* candidate rows are written into a ``WITHOUT ROWID`` table keyed by
  ``(qrow, target_id)`` with ``ON CONFLICT ... block_mask = a | b``; OR is
  commutative and associative, so row order cannot change the stored mask.
* ``retrieved_true_masks`` is last-writer-wins; because chunks are merged in
  order within a source and sources are still processed S2 then S3, "last"
  means the same row it did in the serial implementation.

Byte-range splitting is safe here because the reader is ``csv.QUOTE_NONE`` over
a strictly four-column TSV, so records can never span or contain a newline.
Each worker drops the fragment before its start offset and, unless it owns the
tail of the file, the record that straddles its end offset.
"""

from __future__ import annotations

import csv
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Sequence, Set, Tuple

try:
    from pilot.er_common import (
        blocking_keys,
        canonical_target_fields,
    )
except ModuleNotFoundError:  # direct ``python pilot/parallel_scan.py`` execution
    from er_common import (  # type: ignore[no-redef]
        blocking_keys,
        canonical_target_fields,
    )

# Matches the schema check in ``er_common.iter_tsv``.
EXPECTED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]

# Smallest byte range worth handing to a worker. Below this the per-task
# pickling overhead cancels out the parallelism.
MIN_CHUNK_BYTES = 4 * 1024 * 1024

# Diagnostic override for the floor above. A small corpus subset is smaller than
# MIN_CHUNK_BYTES, so plan_byte_ranges collapses it to a single range and the
# parallel path is bypassed no matter how many workers are requested; the
# equivalence checks set this so the subset really splits. Unset or invalid
# means "use MIN_CHUNK_BYTES".
MIN_CHUNK_BYTES_ENV = "PARALLEL_SCAN_MIN_CHUNK_BYTES"

# Tasks kept in flight beyond the pool size, so a slow chunk never starves the
# idle workers while the parent is busy merging.
PIPELINE_DEPTH = 3

# Resident bytes assumed per worker process (interpreter + NumPy + the pickled
# index). Used to derive a worker count that respects the RAM budget.
BYTES_PER_WORKER = 320 * 1024 * 1024

# Fraction of currently-free memory the run is allowed to plan against.
DEFAULT_RAM_FRACTION = 0.60


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------

def ram_budget_bytes(
    available_bytes: int,
    explicit_gib: float = 0.0,
    fraction: float = DEFAULT_RAM_FRACTION,
) -> int:
    """Return the RAM this run may plan against.

    An explicit ``--ram-budget-gib`` wins.  Otherwise take ``fraction`` of what
    the kernel currently reports as available, so the pool shrinks on a busy
    laptop instead of driving the desktop into swap.
    """
    if explicit_gib and explicit_gib > 0:
        return int(explicit_gib * 1024 ** 3)
    if available_bytes <= 0:
        # Availability unknown: fall back to a single worker rather than
        # guessing a pool size against memory we cannot see.
        return BYTES_PER_WORKER
    return int(available_bytes * fraction)


def resolve_workers(requested: int, budget_bytes: int) -> int:
    """Clamp the requested worker count to CPU and RAM ceilings.

    Hyperthread siblings are cheap here because the work is regex/memory bound
    rather than saturating an FPU, so the logical CPU count is the ceiling.
    """
    if requested < 1:
        raise ValueError("requested workers must be >= 1")
    cpu_ceiling = max(1, os.cpu_count() or 1)
    ram_ceiling = max(1, budget_bytes // BYTES_PER_WORKER)
    return max(1, min(requested, cpu_ceiling, ram_ceiling))


# ---------------------------------------------------------------------------
# Byte-range TSV reading
# ---------------------------------------------------------------------------

def min_chunk_bytes() -> int:
    """Return the effective chunk floor, honouring the diagnostic override."""
    raw = os.environ.get(MIN_CHUNK_BYTES_ENV)
    if raw:
        try:
            value = int(raw)
        except ValueError:
            return MIN_CHUNK_BYTES
        if value > 0:
            return value
    return MIN_CHUNK_BYTES


def plan_byte_ranges(path: Path, chunks: int) -> List[Tuple[int, int]]:
    """Split ``path`` into ``chunks`` ascending byte ranges covering the file.

    Boundaries are not line aligned; the reader assigns each record to the range
    holding its first byte, so ranges must be produced in ascending order and
    must tile the whole file without gaps or overlap.
    """
    total = path.stat().st_size
    if chunks < 1:
        raise ValueError("chunks must be >= 1")
    if total == 0:
        return []
    size = max(min_chunk_bytes(), -(-total // chunks))  # ceil-div
    ranges: List[Tuple[int, int]] = []
    start = 0
    while start < total:
        end = min(start + size, total)
        ranges.append((start, end))
        start = end
    return ranges


def _iter_range_lines(
    handle,
    start: int,
    end: int,
) -> Iterator[str]:
    """Yield every record whose *first byte* lies in ``[start, end)``.

    Each record is therefore yielded by exactly one range: the one holding its
    first byte, which is unique because the planned ranges tile the file
    contiguously.

    Two boundary details matter and both were previously wrong, so that one row
    was silently lost per internal boundary:

    * A record starting before ``end`` but extending past it still belongs to
      *this* range. The next range seeks to ``end`` and discards this record's
      tail, so yielding it here yields it exactly once. Discarding it here as
      well dropped it from both ranges.
    * ``start`` may already sit on a record boundary, in which case the
      preceding ``readline()`` would swallow an entire record. The fragment is
      discarded only when the byte before ``start`` is not a newline.
    """
    handle.seek(start)
    if start > 0:
        handle.seek(start - 1)
        if handle.read(1) != b"\n":
            handle.seek(start)
            handle.readline()  # tail of the record owned by the previous range
    while True:
        position = handle.tell()
        if position >= end:
            return
        raw = handle.readline()
        if not raw:
            return
        line = raw.decode("utf-8")
        if position == 0 and line.startswith("\ufeff"):
            line = line[1:]  # tolerate the BOM utf-8-sig used by iter_tsv
        yield line


def iter_chunk_rows(path: Path, start: int, end: int) -> Iterator[Dict[str, str]]:
    """Stream one byte range of a strict four-column TSV as dictionaries.

    Mirrors ``er_common.iter_tsv`` field for field, including its schema check,
    which runs on whichever range owns offset zero.
    """
    with path.open("rb") as handle:
        lines = _iter_range_lines(handle, start, end)
        if start == 0:
            reader = csv.DictReader(lines, dialect=csv.excel_tab, quoting=csv.QUOTE_NONE)
            if reader.fieldnames != EXPECTED_COLUMNS:
                raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames!r}")
        else:
            reader = csv.DictReader(
                lines,
                dialect=csv.excel_tab,
                quoting=csv.QUOTE_NONE,
                fieldnames=list(EXPECTED_COLUMNS),
            )
        yield from reader


# ---------------------------------------------------------------------------
# Worker state
# ---------------------------------------------------------------------------
# Module globals rather than arguments: the index is pickled once per worker
# through ``initializer`` instead of once per chunk.

_FREQ_INDEX: Dict[str, List[Tuple[int, int]]] = {}
_FREQ_TARGETS: Dict[str, int] = {}

_CAND_INDEX: Dict[str, List[Tuple[int, int]]] = {}
_CAND_TARGETS: Dict[str, int] = {}


def _init_frequency_state(key_index: Mapping[str, List[Tuple[int, int]]],
                          target_to_query: Mapping[str, int]) -> None:
    global _FREQ_INDEX, _FREQ_TARGETS
    _FREQ_INDEX = dict(key_index)
    _FREQ_TARGETS = dict(target_to_query)


def _init_candidate_state(active_index: Mapping[str, List[Tuple[int, int]]],
                          target_to_query: Mapping[str, int]) -> None:
    global _CAND_INDEX, _CAND_TARGETS
    _CAND_INDEX = dict(active_index)
    _CAND_TARGETS = dict(target_to_query)


# ---------------------------------------------------------------------------
# Chunk workers
# ---------------------------------------------------------------------------

def frequency_chunk(task: Tuple[str, int, int]) -> Tuple[Dict[str, int], Set[Tuple[int, str, str]], int, int, int]:
    """Count posting frequencies and record true-pair hits for one byte range.

    Returns ``(counts, true_keys, rows, matched_rows, generated_postings)``.
    Only keys that actually hit the query index are returned, so the result is
    proportional to the number of matches rather than to the number of rows.
    """
    path, start, end = task
    key_index = _FREQ_INDEX
    target_to_query = _FREQ_TARGETS
    counts: Dict[str, int] = {}
    true_keys: Set[Tuple[int, str, str]] = set()
    rows = matched_rows = generated_postings = 0
    for row in iter_chunk_rows(Path(path), start, end):
        rows += 1
        target_id = row["entity_id"]
        keys = blocking_keys(row["business_name"], row["business_address"], row["country"])
        generated_postings += len(keys)
        row_matched = False
        positive_qrow = target_to_query.get(target_id)
        for key in keys:
            entries = key_index.get(key)
            if not entries:
                continue
            counts[key] = counts.get(key, 0) + 1
            row_matched = True
            if positive_qrow is not None and any(qrow == positive_qrow for qrow, _ in entries):
                true_keys.add((positive_qrow, target_id, key))
        matched_rows += int(row_matched)
    return counts, true_keys, rows, matched_rows, generated_postings


def candidate_chunk(task: Tuple[str, int, int, int]) -> Tuple[List[Tuple[int, str, int, str, str, str, int]], Dict[Tuple[int, str], int], int, int, int]:
    """Match one byte range against the active index and emit candidate rows.

    Returns ``(records, retrieved_true_masks, rows, generated_keys, matches)``.
    Only rows that matched at least one query key are returned, which is a small
    fraction of the corpus, so IPC volume stays proportional to the candidate
    count instead of the row count.
    """
    path, start, end, source_number = task
    active_index = _CAND_INDEX
    target_to_query = _CAND_TARGETS
    records: List[Tuple[int, str, int, str, str, str, int]] = []
    true_masks: Dict[Tuple[int, str], int] = {}
    rows = generated_keys = matches = 0
    for row in iter_chunk_rows(Path(path), start, end):
        rows += 1
        keys = blocking_keys(row["business_name"], row["business_address"], row["country"])
        generated_keys += len(keys)
        pair_masks: Dict[int, int] = {}
        for key in keys:
            entries = active_index.get(key)
            if not entries:
                continue
            matches += len(entries)
            for qrow, bit in entries:
                pair_masks[qrow] = pair_masks.get(qrow, 0) | bit
        if not pair_masks:
            continue
        name_norm, address_norm, target_country = canonical_target_fields(
            row["business_name"], row["business_address"], row["country"]
        )
        target_id = row["entity_id"]
        positive_qrow = target_to_query.get(target_id)
        for qrow, mask in pair_masks.items():
            records.append((qrow, target_id, source_number, name_norm,
                            address_norm, target_country, mask))
            if qrow == positive_qrow:
                true_masks[(qrow, target_id)] = mask
    return records, true_masks, rows, generated_keys, matches


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_chunks(
    tasks: Sequence[Any],
    worker: Callable[[Any], Any],
    initializer: Callable[..., None],
    initargs: Tuple[Any, ...],
    workers: int,
    progress: Callable[[int, int], None] | None = None,
) -> Iterator[Tuple[int, Any]]:
    """Execute ``tasks`` in order, streaming results to ``on_result`` as they land.

    ``workers <= 1`` runs inline in this process with no pool and no pickling, so
    the tiny end-to-end tests and single-core machines keep the exact serial
    behaviour.  Otherwise a pool is opened with a bounded submission window.

    Yields ``(position, result)`` in ascending task order.  The pool stays open
    for the whole generator, so the caller must fully consume it.
    """
    if workers <= 1:
        initializer(*initargs)
        for position, task in enumerate(tasks):
            result = worker(task)
            yield position, result
            if progress is not None:
                progress(position + 1, len(tasks))
        return

    pending: deque = deque()
    remaining = iter(tasks)
    with ProcessPoolExecutor(max_workers=workers, initializer=initializer,
                             initargs=initargs) as pool:
        def submit_next() -> bool:
            task = next(remaining, None)
            if task is None:
                return False
            pending.append(pool.submit(worker, task))
            return True

        for _ in range(workers * PIPELINE_DEPTH):
            if not submit_next():
                break
        position = 0
        while pending:
            result = pending.popleft().result()
            submit_next()
            yield position, result
            position += 1
            if progress is not None:
                progress(position, len(tasks))
