#!/usr/bin/env python3
"""Bounded end-to-end entity-resolution pilot on 10,000 training S1 rows.

The script is intentionally pilot-only.  It never reads the test files and does
not create cross-source dense matrices.  Target TSVs are streamed twice: once to
measure block frequencies, then once to persist deduplicated candidates.  Pair
features are generated in bounded batches and saved as compact float32 BLOBs.
"""

from __future__ import annotations

import argparse
import csv
import gc
import heapq
import json
import math
import os
import sqlite3
import struct
import sys
import time
import zlib
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from pilot.er_common import (
        BLOCK_BIT,
        BLOCK_SCHEMES,
        FEATURE_NAMES,
        MemoryMonitor,
        PairText,
        acquire_run_lock,
        canonical_target_fields,
        adapt_batch,
        available_memory_bytes,
        blocking_keys,
        disk_free_bytes,
        iter_tsv,
        macro_f05,
        normalize_country,
        pair_features,
        percentile,
        reservoir_sample_rows,
        stable_u64,
    )
except ModuleNotFoundError:  # direct ``python pilot/run_pilot.py`` execution
    from er_common import (
        BLOCK_BIT,
        BLOCK_SCHEMES,
        FEATURE_NAMES,
        MemoryMonitor,
        PairText,
        acquire_run_lock,
        canonical_target_fields,
        adapt_batch,
        available_memory_bytes,
        blocking_keys,
        disk_free_bytes,
        iter_tsv,
        macro_f05,
        normalize_country,
        pair_features,
        percentile,
        reservoir_sample_rows,
        stable_u64,
    )


@dataclass(slots=True)
class QueryRecord:
    qrow: int
    entity_id: str
    business_name: str
    business_address: str
    country: str
    split: str


@dataclass
class PhaseTimer:
    phases: Dict[str, float]

    def start(self) -> float:
        return time.perf_counter()

    def finish(self, name: str, started: float) -> None:
        elapsed = time.perf_counter() - started
        self.phases[name] = self.phases.get(name, 0.0) + elapsed
        log(f"phase {name} completed in {elapsed:.2f}s")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def configure_sqlite(path: Path, cache_mib: int = 128) -> sqlite3.Connection:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA page_size=32768")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute(f"PRAGMA cache_size=-{cache_mib * 1024}")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    return connection


def sample_queries(data_root: Path, work_dir: Path, sample_size: int) -> Tuple[List[QueryRecord], Dict[int, Set[str]], Dict[int, str]]:
    train_dir = data_root / "train"
    source1_path = train_dir / "train_source1.tsv"
    log(f"sampling {sample_size:,} rows uniformly from {source1_path.name}")
    sampled = reservoir_sample_rows(source1_path, sample_size)
    queries: List[QueryRecord] = []
    for qrow, row in enumerate(sampled):
        bucket = stable_u64(row["entity_id"], "split-v1") % 20
        split = "train" if bucket < 14 else "validation" if bucket < 17 else "test"
        queries.append(QueryRecord(
            qrow=qrow,
            entity_id=row["entity_id"],
            business_name=row["business_name"],
            business_address=row["business_address"],
            country=normalize_country(row["country"]),
            split=split,
        ))

    qid_to_qrow = {query.entity_id: query.qrow for query in queries}
    truth: Dict[int, Set[str]] = {}
    with (train_dir / "train_ground_truth.tsv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        header = next(reader)
        if header != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"Unexpected ground-truth header: {header!r}")
        for s1_id, raw_matches in reader:
            qrow = qid_to_qrow.get(s1_id)
            if qrow is not None:
                truth[qrow] = {item for item in raw_matches.split(",") if item}
    if len(truth) != sample_size:
        raise ValueError(f"Only found labels for {len(truth):,}/{sample_size:,} sampled S1 rows")

    query_path = work_dir / "pilot_queries.tsv"
    label_path = work_dir / "pilot_labels.tsv"
    with query_path.open("w", encoding="utf-8", newline="") as qhandle, \
            label_path.open("w", encoding="utf-8", newline="") as lhandle:
        qwriter = csv.writer(qhandle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        lwriter = csv.writer(lhandle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        qwriter.writerow(["qrow", "entity_id", "business_name", "business_address", "country", "split"])
        lwriter.writerow(["source1_entity_id", "matched_entity_ids"])
        for query in queries:
            qwriter.writerow([query.qrow, query.entity_id, query.business_name, query.business_address, query.country, query.split])
            lwriter.writerow([query.entity_id, ",".join(sorted(truth[query.qrow]))])

    split_counts = Counter(query.split for query in queries)
    link_counts = Counter(
        target_id.split("-", 1)[0]
        for matches in truth.values()
        for target_id in matches
    )
    log(f"sample: {dict(split_counts)}; truth links={sum(map(len, truth.values())):,} {dict(link_counts)}; "
        f"singletons={sum(not value for value in truth.values()):,}")
    return queries, truth, {query.qrow: query.split for query in queries}


def build_query_key_index(queries: Sequence[QueryRecord]) -> Dict[str, List[Tuple[int, int]]]:
    index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for query in queries:
        for key in blocking_keys(query.business_name, query.business_address, query.country):
            index[key].append((query.qrow, BLOCK_BIT[scheme_from_key(key)]))
    return dict(index)


KEY_SCHEME_CODE = {
    "name_exact": "nfull",
    "name_prefix": "npre",
    "name_token": "ntok",
    "name_soundex": "nsnd",
    "address_numeric": "anum",
    "address_house_postal": "ahp",
    "address_token": "atok",
    "address_bigram": "abg",
}


def scheme_from_key(key: str) -> str:
    # Key format: integer | country | scheme-code | payload...
    code = key.split("|", 3)[2]
    for scheme, expected_code in KEY_SCHEME_CODE.items():
        if code == expected_code:
            return scheme
    raise ValueError(f"Unknown blocking key: {key!r}")


def measure_block_frequencies(
    data_root: Path,
    queries: Sequence[QueryRecord],
    truth: Mapping[int, Set[str]],
    key_index: Mapping[str, List[Tuple[int, int]]],
    monitor: MemoryMonitor,
) -> Tuple[Dict[str, int], Dict[Tuple[int, str], Set[str]], Dict[str, object]]:
    counts: Dict[str, int] = {key: 0 for key in key_index}
    true_keys: Dict[Tuple[int, str], Set[str]] = defaultdict(set)
    target_to_query: Dict[str, int] = {}
    for qrow, matches in truth.items():
        for target_id in matches:
            previous = target_to_query.setdefault(target_id, qrow)
            if previous != qrow:
                raise ValueError(f"Target {target_id} labels multiple sampled S1 rows")

    source_stats: Dict[str, object] = {}
    started = time.perf_counter()
    for source_number, name in ((2, "train_source2.tsv"), (3, "train_source3.tsv")):
        path = data_root / "train" / name
        rows = matched_rows = generated_postings = 0
        pass_started = time.perf_counter()
        for row in iter_tsv(path):
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
                counts[key] += 1
                row_matched = True
                if positive_qrow is not None and any(qrow == positive_qrow for qrow, _ in entries):
                    true_keys[(positive_qrow, target_id)].add(key)
            matched_rows += int(row_matched)
            if rows % 500_000 == 0:
                available_gib = available_memory_bytes() / 1024 ** 3
                log(f"frequency pass {name}: {rows:,}; process RSS={monitor.peak_rss / 1024**2:.1f} MiB; "
                    f"MemAvailable={available_gib:.2f} GiB")
                if available_gib and available_gib < 2.0:
                    log("WARNING: available RAM is below 2 GiB; forcing GC before continuing")
                    gc.collect()
        source_stats[f"S{source_number}"] = {
            "rows": rows,
            "matched_query_rows": matched_rows,
            "generated_key_postings": generated_postings,
            "average_keys_per_row": generated_postings / rows,
            "seconds": time.perf_counter() - pass_started,
        }
        log(f"frequency pass {name}: complete ({source_stats[f'S{source_number}']})")

    total_true_pairs = sum(len(value) for value in truth.values())
    per_scheme: Dict[str, Dict[str, int]] = {}
    for scheme in BLOCK_SCHEMES:
        bit = BLOCK_BIT[scheme]
        scheme_keys = [key for key in key_index if BLOCK_BIT[scheme_from_key(key)] == bit]
        nonzero = sum(counts[key] > 0 for key in scheme_keys)
        postings = sum(counts[key] for key in scheme_keys)
        pair_hits = 0
        for (qrow, _target), keys in true_keys.items():
            if any(BLOCK_BIT[scheme_from_key(key)] == bit for key in keys):
                pair_hits += 1
        per_scheme[scheme] = {
            "query_keys": len(scheme_keys),
            "query_keys_nonzero": nonzero,
            "query_postings_before_frequency_cap": postings,
            "raw_true_pair_hits_before_cap": pair_hits,
            "raw_true_pair_recall_before_cap": pair_hits / total_true_pairs if total_true_pairs else 1.0,
        }

    result: Dict[str, object] = {
        "target_rows_scanned": sum(int(source_stats[key]["rows"]) for key in source_stats),  # type: ignore[index]
        "true_pairs": total_true_pairs,
        "sources": source_stats,
        "schemes_before_cap": per_scheme,
    }
    log(f"frequency pass complete in {time.perf_counter() - started:.2f}s; "
        f"raw union true-pair recall={len(true_keys) / total_true_pairs:.6f}")
    return counts, dict(true_keys), result


def select_block_postings(
    queries: Sequence[QueryRecord],
    key_index: Mapping[str, List[Tuple[int, int]]],
    counts: Mapping[str, int],
    true_keys: Mapping[Tuple[int, str], Set[str]],
    truth: Mapping[int, Set[str]],
    max_projected_postings: int,
    fixed_cap: int | None = None,
    target_row_total: int | None = None,
) -> Tuple[Dict[str, List[Tuple[int, int]]], Dict[int, Set[str]], Dict[str, object]]:
    # Sort each query's postings from most selective (low target frequency) to least.
    by_query: List[List[Tuple[int, str, int]]] = [[] for _ in queries]
    seen: Set[Tuple[int, str]] = set()
    for key, entries in key_index.items():
        frequency = counts.get(key, 0)
        for qrow, bit in entries:
            pair = (qrow, key)
            if pair in seen:
                continue
            seen.add(pair)
            by_query[qrow].append((frequency, key, bit))
    for postings in by_query:
        postings.sort(key=lambda item: (item[0], item[2], item[1]))

    configurations = (
        [(fixed_cap, fixed_cap)]
        if fixed_cap is not None
        else [
            (block_cap, query_cap)
            for block_cap in (100, 250, 500, 1000, 2000, 5000, 10000)
            for query_cap in (250, 500, 1000, 2000, 5000, 10000)
        ]
    )
    trials: List[Dict[str, float | int]] = []
    chosen: Tuple[int, int, List[Set[str]], int, int, float, float] | None = None
    selection_qids = {query.qrow for query in queries if query.split == "validation"}
    if not selection_qids:
        raise ValueError("blocking selection requires at least one validation query")
    total_true = sum(len(truth.get(qrow, set())) for qrow in selection_qids)
    matched_query_total = sum(bool(truth.get(qrow)) for qrow in selection_qids)

    for block_cap, query_cap in configurations:
        retained: List[Set[str]] = []
        projected = 0
        active_entries = 0
        for postings in by_query:
            kept: Set[str] = set()
            used = 0
            for frequency, key, _bit in postings:
                if frequency <= 0 or frequency > block_cap:
                    continue
                if used + frequency > query_cap:
                    continue
                kept.add(key)
                used += frequency
                active_entries += 1
            retained.append(kept)
            projected += used
        pair_hits = sum(
            1 for pair, keys in true_keys.items()
            if pair[0] in selection_qids
            and any(key in retained[pair[0]] for key in keys)
        )
        query_hits = len({
            pair[0] for pair, keys in true_keys.items()
            if pair[0] in selection_qids
            and any(key in retained[pair[0]] for key in keys)
        })
        recall = pair_hits / total_true if total_true else 1.0
        trial = {
            "block_cap": block_cap,
            "query_cap": query_cap,
            "projected_raw_postings": projected,
            "active_query_postings": active_entries,
            "expected_true_pair_recall": recall,
            "expected_true_query_recall": query_hits / matched_query_total if matched_query_total else 1.0,
        }
        trials.append(trial)
        feasible = projected <= max_projected_postings
        better = chosen is None
        if chosen is not None:
            _, _, _, chosen_projected, _, chosen_recall, _ = chosen
            chosen_feasible = chosen_projected <= max_projected_postings
            if feasible != chosen_feasible:
                better = feasible
            elif feasible:
                better = recall > chosen_recall + 1e-12 or (
                    abs(recall - chosen_recall) <= 1e-12 and projected < chosen_projected
                )
            else:
                better = recall > chosen_recall
        if better:
            chosen = (block_cap, query_cap, retained, projected, active_entries, recall, query_hits / len(selection_qids))

    if chosen is None:
        raise RuntimeError("No blocking configuration selected")
    block_cap, query_cap, retained, projected, active_entries, recall, query_recall = chosen
    if fixed_cap is not None and projected > max_projected_postings:
        raise ValueError(
            f"fixed cap {fixed_cap}/{fixed_cap} projects {projected:,} raw postings, "
            f"above the {max_projected_postings:,} limit"
        )
    active_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for qrow, keys in enumerate(retained):
        for key in keys:
            for original_qrow, bit in key_index[key]:
                if original_qrow == qrow:
                    active_index[key].append((qrow, bit))
                    break

    per_scheme: Dict[str, Dict[str, float | int]] = {}
    for scheme in BLOCK_SCHEMES:
        bit = BLOCK_BIT[scheme]
        keys = [key for key in active_index if BLOCK_BIT[scheme_from_key(key)] == bit]
        postings = sum(counts[key] for key in keys)
        hits = 0
        for (qrow, _target), pair_keys in true_keys.items():
            if qrow in selection_qids and any(
                key in retained[qrow] and BLOCK_BIT[scheme_from_key(key)] == bit
                for key in pair_keys
            ):
                hits += 1
        per_scheme[scheme] = {
            "active_query_keys": len(keys),
            "active_postings": postings,
            "expected_true_pair_hits": hits,
            "expected_true_pair_recall": hits / total_true if total_true else 1.0,
        }

    result: Dict[str, object] = {
        "selection_objective": (
            "predeclared matched cap; validation labels used only for reporting"
            if fixed_cap is not None
            else "maximize validation true-pair recall subject to projected raw postings cap"
        ),
        "selection_split": "validation",
        "selection_query_count": len(selection_qids),
        "max_projected_postings": max_projected_postings,
        "selected_block_frequency_cap": block_cap,
        "selected_postings_per_query_cap": query_cap,
        "projected_raw_postings": projected,
        "active_query_postings": active_entries,
        "expected_true_pair_recall": recall,
        "expected_true_query_recall": query_recall,
        "expected_reduction_ratio": 1.0 - projected / (
            len(queries) * int(target_row_total or source_row_total_for_projection())
        ),
        "schemes_after_cap": per_scheme,
        "trials": trials,
    }
    log(f"selected block cap={block_cap:,}, query posting cap={query_cap:,}; "
        f"projected postings={projected:,}; expected pair recall={recall:.6f}")
    return dict(active_index), retained, result


def source_row_total_for_projection() -> int:
    """Fallback for the audited challenge corpus; production passes the measured total."""
    return 10_320_219


def persist_candidates(
    data_root: Path,
    work_dir: Path,
    queries: Sequence[QueryRecord],
    truth: Mapping[int, Set[str]],
    active_index: Mapping[str, List[Tuple[int, int]]],
    monitor: MemoryMonitor,
    initial_batch_size: int,
) -> Tuple[Path, Dict[str, object]]:
    path = work_dir / "pilot_candidates.sqlite"
    connection = configure_sqlite(path)
    connection.execute("""
        CREATE TABLE candidates (
            qrow INTEGER NOT NULL,
            target_id TEXT NOT NULL,
            source INTEGER NOT NULL,
            name_norm TEXT NOT NULL,
            address_norm TEXT NOT NULL,
            country TEXT NOT NULL,
            block_mask INTEGER NOT NULL,
            PRIMARY KEY (qrow, target_id)
        ) WITHOUT ROWID
    """)
    insert_sql = """
        INSERT INTO candidates(qrow,target_id,source,name_norm,address_norm,country,block_mask)
        VALUES(?,?,?,?,?,?,?)
        ON CONFLICT(qrow,target_id) DO UPDATE SET
            block_mask=(candidates.block_mask | excluded.block_mask)
    """
    target_to_query: Dict[str, int] = {
        target_id: qrow for qrow, matches in truth.items() for target_id in matches
    }
    retrieved_true_masks: Dict[Tuple[int, str], int] = {}
    batch: List[Tuple[int, str, int, str, str, str, int]] = []
    batch_size = initial_batch_size
    inserted_since_commit = 0
    total_rows = total_posting_matches = total_generated_keys = 0
    source_stats: Dict[str, Dict[str, int | float]] = {}
    started = time.perf_counter()

    for source_number, name in ((2, "train_source2.tsv"), (3, "train_source3.tsv")):
        source_started = time.perf_counter()
        source_rows = source_candidates = source_matches = 0
        for row in iter_tsv(data_root / "train" / name):
            source_rows += 1
            total_rows += 1
            keys = blocking_keys(row["business_name"], row["business_address"], row["country"])
            total_generated_keys += len(keys)
            pair_masks: Dict[int, int] = {}
            for key in keys:
                entries = active_index.get(key)
                if not entries:
                    continue
                source_matches += len(entries)
                total_posting_matches += len(entries)
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
                batch.append((qrow, target_id, source_number, name_norm, address_norm,
                              target_country, mask))
                source_candidates += 1
                if qrow == positive_qrow:
                    retrieved_true_masks[(qrow, target_id)] = mask
            if len(batch) >= batch_size:
                connection.executemany(insert_sql, batch)
                inserted_since_commit += len(batch)
                batch.clear()
                if inserted_since_commit >= 250_000:
                    connection.commit()
                    inserted_since_commit = 0
                    gc.collect()
            if source_rows % 250_000 == 0:
                available_gib = available_memory_bytes() / 1024 ** 3
                log(f"candidate pass {name}: {source_rows:,} targets; batch={batch_size:,}; "
                    f"process RSS={monitor.peak_rss / 1024**2:.1f} MiB; MemAvailable={available_gib:.2f} GiB")
                batch_size = adapt_batch(batch_size, minimum_batch=1000)
                if available_gib and available_gib < 2.0:
                    if batch:
                        connection.executemany(insert_sql, batch)
                        batch.clear()
                    gc.collect()
                    log("WARNING: reduced SQLite batch due to MemAvailable < 2 GiB")
        if batch:
            connection.executemany(insert_sql, batch)
            batch.clear()
        connection.commit()
        source_stats[f"S{source_number}"] = {
            "rows": source_rows,
            "candidate_rows": source_candidates,
            "active_posting_matches": source_matches,
            "seconds": time.perf_counter() - source_started,
        }

    candidate_count = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    qcounts = [row[0] for row in connection.execute(
        "SELECT COUNT(*) FROM candidates GROUP BY qrow"
    )]
    covered = set(row[0] for row in connection.execute("SELECT DISTINCT qrow FROM candidates"))
    coverage = np.zeros(len(queries), dtype=np.int64)
    for qrow in covered:
        coverage[qrow] = 1
    counts_by_source = {
        str(source): count for source, count in connection.execute(
            "SELECT source,COUNT(*) FROM candidates GROUP BY source"
        )
    }
    mask_counts = {
        str(mask): count for mask, count in connection.execute(
            "SELECT block_mask,COUNT(*) FROM candidates GROUP BY block_mask ORDER BY block_mask"
        )
    }
    page_count = connection.execute("PRAGMA page_count").fetchone()[0]
    page_size = connection.execute("PRAGMA page_size").fetchone()[0]
    connection.close()

    total_true = sum(len(value) for value in truth.values())
    true_hits = len(retrieved_true_masks)
    true_queries = len({qrow for qrow, _ in retrieved_true_masks})
    scheme_hits = Counter()
    for mask in retrieved_true_masks.values():
        for scheme, bit in BLOCK_BIT.items():
            if mask & bit:
                scheme_hits[scheme] += 1
    total_queries_with_candidates = len(qcounts)
    stats: Dict[str, object] = {
        "candidate_count": candidate_count,
        "candidate_count_per_s1_mean": candidate_count / len(queries),
        "candidate_count_per_s1_median": percentile(qcounts, 50),
        "candidate_count_per_s1_p90": percentile(qcounts, 90),
        "candidate_count_per_s1_p99": percentile(qcounts, 99),
        "candidate_count_per_s1_max": max(qcounts, default=0),
        "s1_rows_with_candidates": total_queries_with_candidates,
        "s1_rows_without_candidates": len(queries) - total_queries_with_candidates,
        "singleton_s1_rows_without_candidates": sum(
            not truth.get(qrow) and not coverage[qrow] for qrow in range(len(queries))
        ),
        "candidates_by_source": counts_by_source,
        "block_mask_counts": mask_counts,
        "true_pair_count": total_true,
        "true_pair_recall": true_hits / total_true if total_true else 1.0,
        "true_query_recall_with_any_match": true_queries / sum(bool(v) for v in truth.values())
        if any(truth.values()) else 1.0,
        "true_pair_hits_by_scheme": dict(scheme_hits),
        "theoretical_comparisons_without_blocking": len(queries) * total_rows,
        "candidate_reduction_ratio": 1.0 - candidate_count / (len(queries) * total_rows),
        "target_rows_scanned": total_rows,
        "active_posting_matches": total_posting_matches,
        "average_generated_keys_per_target": total_generated_keys / total_rows,
        "sources": source_stats,
        "candidate_sqlite_bytes": path.stat().st_size,
        "candidate_sqlite_logical_page_bytes": page_count * page_size,
        "seconds": time.perf_counter() - started,
    }
    write_json(work_dir / "candidate_stats.json", stats)
    log(f"candidate generation complete: {candidate_count:,} pairs, recall={stats['true_pair_recall']:.6f}, "
        f"mean={stats['candidate_count_per_s1_mean']:.2f}, size={path.stat().st_size / 1024**2:.1f} MiB")
    return path, stats


def build_feature_database(
    candidate_db: Path,
    feature_db: Path,
    queries: Sequence[QueryRecord],
    initial_batch_size: int,
    monitor: MemoryMonitor,
) -> Tuple[Path, Dict[str, object]]:
    if feature_db.exists():
        feature_db.unlink()
    source = sqlite3.connect(candidate_db)
    target = configure_sqlite(feature_db)
    columns = ",".join(f"f{index}" for index in range(len(FEATURE_NAMES)))
    target.execute(f"""
        CREATE TABLE pair_features (
            qrow INTEGER NOT NULL,
            target_id TEXT NOT NULL,
            features BLOB NOT NULL,
            PRIMARY KEY(qrow,target_id)
        ) WITHOUT ROWID
    """)
    insert_sql = "INSERT INTO pair_features(qrow,target_id,features) VALUES(?,?,?)"
    query_text = {
        query.qrow: PairText.make(query.business_name, query.business_address, query.country)
        for query in queries
    }
    cursor = source.execute("""
        SELECT qrow,target_id,name_norm,address_norm,country
        FROM candidates ORDER BY qrow,target_id
    """)
    batch: List[Tuple[int, str, bytes]] = []
    batch_size = initial_batch_size
    rows = 0
    started = time.perf_counter()
    for qrow, target_id, name_norm, address_norm, country in cursor:
        left = query_text[qrow]
        right = PairText.make(name_norm, address_norm, country)
        vector = pair_features_to_blob(left, right)
        batch.append((qrow, target_id, vector))
        rows += 1
        if len(batch) >= batch_size:
            target.executemany(insert_sql, batch)
            batch.clear()
            if rows % (batch_size * 20) == 0:
                target.commit()
                gc.collect()
            if rows % 100_000 == 0:
                available_gib = available_memory_bytes() / 1024 ** 3
                log(f"feature batches: {rows:,}; process RSS={monitor.peak_rss / 1024**2:.1f} MiB; "
                    f"MemAvailable={available_gib:.2f} GiB")
                batch_size = adapt_batch(batch_size, minimum_batch=250)
        if rows % 100_000 == 0:
            available_gib = available_memory_bytes() / 1024 ** 3
            if available_gib and available_gib < 2.0:
                gc.collect()
                log("WARNING: feature batch reduced because MemAvailable < 2 GiB")
                batch_size = adapt_batch(batch_size, minimum_batch=250)
    if batch:
        target.executemany(insert_sql, batch)
    target.commit()
    target.execute("PRAGMA optimize")
    page_count = target.execute("PRAGMA page_count").fetchone()[0]
    page_size = target.execute("PRAGMA page_size").fetchone()[0]
    target.close()
    source.close()
    stats: Dict[str, object] = {
        "rows": rows,
        "feature_count": len(FEATURE_NAMES),
        "float32_bytes_per_row": len(FEATURE_NAMES) * 4,
        "feature_sqlite_bytes": feature_db.stat().st_size,
        "feature_sqlite_logical_page_bytes": page_count * page_size,
        "bytes_per_candidate_including_key": feature_db.stat().st_size / rows if rows else 0.0,
        "seconds": time.perf_counter() - started,
    }
    write_json(feature_db.with_suffix(".stats.json"), stats)
    log(f"feature generation complete: {rows:,} vectors -> {feature_db.stat().st_size / 1024**2:.1f} MiB "
        f"in {stats['seconds']:.1f}s")
    return feature_db, stats


def pair_features_to_blob(left: PairText, right: PairText) -> bytes:
    return pair_features(left, right).astype("<f4", copy=False).tobytes()


def unpack_features(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4", count=len(FEATURE_NAMES))


def train_compact_classifier(
    candidate_db: Path,
    feature_db: Path,
    queries: Sequence[QueryRecord],
    truth: Mapping[int, Set[str]],
    negative_per_query: int,
    monitor: MemoryMonitor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    connection = sqlite3.connect(candidate_db)
    connection.execute("ATTACH DATABASE ? AS feat", (str(feature_db),))
    cursor = connection.execute("""
        SELECT c.qrow,c.target_id,c.source,c.name_norm,c.address_norm,c.country,f.features
        FROM candidates c
        JOIN feat.pair_features f ON f.qrow=c.qrow AND f.target_id=c.target_id
        ORDER BY c.qrow,c.target_id
    """)
    query_text = {
        query.qrow: PairText.make(query.business_name, query.business_address, query.country)
        for query in queries
    }
    split = {query.qrow: query.split for query in queries}
    features: List[np.ndarray] = []
    labels: List[int] = []
    selected_positive = selected_negative = 0
    covered_train_queries = 0
    current_qrow: int | None = None
    positives: List[Tuple[int, bytes, int]] = []
    negative_heap: List[Tuple[int, str, bytes, int]] = []

    def finish(qrow: int) -> None:
        nonlocal selected_positive, selected_negative, covered_train_queries
        if qrow is None:
            return
        rows = [(1, blob, source) for _rank, blob, source in positives]
        rows.extend((0, blob, source) for _rank, _target, blob, source in negative_heap)
        for label, blob, _source in rows:
            features.append(unpack_features(blob).copy())
            labels.append(label)
            selected_positive += label
            selected_negative += 1 - label
        covered_train_queries += 1

    for qrow, target_id, source, name_norm, address_norm, country, blob in cursor:
        if qrow != current_qrow:
            if current_qrow is not None and split[current_qrow] == "train":
                finish(current_qrow)
            current_qrow = qrow
            positives = []
            negative_heap = []
        if split[qrow] != "train":
            continue
        label = int(target_id in truth[qrow])
        if label:
            positives.append((source, blob, source))
        else:
            rank = zlib.crc32(target_id.encode("ascii"))
            item = (rank, target_id, blob, source)
            if len(negative_heap) < negative_per_query:
                heapq.heappush(negative_heap, item)
            elif rank < negative_heap[0][0]:
                heapq.heapreplace(negative_heap, item)
    if current_qrow is not None and split[current_qrow] == "train":
        finish(current_qrow)
    connection.close()

    if not features:
        raise RuntimeError("No training features were generated")
    X = np.vstack(features).astype(np.float32, copy=False)
    y = np.asarray(labels, dtype=np.int8)
    del features, labels
    gc.collect()

    mean = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = X.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    mean[0] = 0.0
    std[0] = 1.0
    Z = (X - mean) / std
    Z[:, 0] = 1.0

    positives_count = int(y.sum())
    negatives_count = int(len(y) - positives_count)
    sample_weight = np.where(y == 1, negatives_count / max(1, positives_count), 1.0).astype(np.float32)
    sample_weight /= sample_weight.mean()
    weights = np.zeros(Z.shape[1], dtype=np.float32)
    first_moment = np.zeros_like(weights)
    second_moment = np.zeros_like(weights)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    batch_size = 4096
    epochs = 12
    learning_rate = 0.03
    l2 = 1e-4
    started = time.perf_counter()
    generator = np.random.default_rng(20260925)
    for epoch in range(epochs):
        order = generator.permutation(len(Z))
        epoch_loss = 0.0
        seen_weight = 0.0
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            xb = Z[index]
            yb = y[index].astype(np.float32)
            wb = sample_weight[index]
            logits = np.clip(xb @ weights, -20.0, 20.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            error = (probabilities - yb) * wb
            gradient = (xb.T @ error) / len(index)
            penalty = np.zeros_like(weights)
            penalty[1:] = l2 * weights[1:]
            gradient += penalty
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient * gradient
            corrected_first = first_moment / (1.0 - beta1 ** (epoch + 1))
            corrected_second = second_moment / (1.0 - beta2 ** (epoch + 1))
            weights -= learning_rate * corrected_first / (np.sqrt(corrected_second) + epsilon)
            loss = np.maximum(logits, 0) - logits * yb + np.log1p(np.exp(-np.abs(logits)))
            epoch_loss += float((loss * wb).sum())
            seen_weight += float(wb.sum())
            if int(start / batch_size) % 100 == 0:
                available_gib = available_memory_bytes() / 1024 ** 3
                if available_gib and available_gib < 2.0:
                    gc.collect()
                    log("WARNING: low MemAvailable during classifier training; forcing GC")
        log(f"classifier epoch {epoch + 1}/{epochs}: weighted loss={epoch_loss / max(seen_weight, 1e-9):.6f}, "
            f"process RSS={monitor.peak_rss / 1024**2:.1f} MiB")

    scores = Z @ weights
    auc = binary_auc(scores, y)
    stats: Dict[str, object] = {
        "model": "NumPy mini-batch Adam logistic regression",
        "feature_count": len(FEATURE_NAMES),
        "training_rows": len(y),
        "training_positive_rows": positives_count,
        "training_negative_rows": negatives_count,
        "negative_candidates_per_train_query": negative_per_query,
        "train_queries_with_candidate": covered_train_queries,
        "class_balanced_positive_weight": negatives_count / max(1, positives_count),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "l2": l2,
        "training_auc": auc,
        "seconds": time.perf_counter() - started,
    }
    return weights, mean, std, stats


def binary_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels == 1]
    negative = scores[labels == 0]
    if not len(positive) or not len(negative):
        return 0.5
    order = np.argsort(np.concatenate([negative, positive]), kind="mergesort")
    ranks = np.empty(len(order), dtype=np.float64)
    sorted_scores = np.concatenate([negative, positive])[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = ranks[len(negative):].sum()
    return float((positive_rank_sum - len(positive) * (len(positive) + 1) / 2) /
                 (len(positive) * len(negative)))


def score_all_candidates(
    feature_db: Path,
    score_db: Path,
    weights: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    initial_batch_size: int,
    monitor: MemoryMonitor,
) -> Tuple[Path, Dict[str, object]]:
    if score_db.exists():
        score_db.unlink()
    source = sqlite3.connect(feature_db)
    target = configure_sqlite(score_db)
    target.execute("""
        CREATE TABLE pair_scores (
            qrow INTEGER NOT NULL,
            target_id TEXT NOT NULL,
            score REAL NOT NULL,
            PRIMARY KEY(qrow,target_id)
        ) WITHOUT ROWID
    """)
    cursor = source.execute("SELECT qrow,target_id,features FROM pair_features ORDER BY qrow,target_id")
    batch: List[Tuple[int, str, float]] = []
    batch_size = initial_batch_size
    rows = 0
    started = time.perf_counter()
    for qrow, target_id, blob in cursor:
        vector = unpack_features(blob)
        vector = (vector - mean) / std
        vector[0] = 1.0
        score = float(np.clip(vector @ weights, -30.0, 30.0))
        logistic = 1.0 / (1.0 + math.exp(-score))
        batch.append((qrow, target_id, logistic))
        rows += 1
        if len(batch) >= batch_size:
            target.executemany("INSERT INTO pair_scores(qrow,target_id,score) VALUES(?,?,?)", batch)
            batch.clear()
            if rows % (batch_size * 20) == 0:
                target.commit()
                gc.collect()
        if rows % 100_000 == 0:
            available_gib = available_memory_bytes() / 1024 ** 3
            log(f"scoring: {rows:,}; process RSS={monitor.peak_rss / 1024**2:.1f} MiB; "
                f"MemAvailable={available_gib:.2f} GiB")
            batch_size = adapt_batch(batch_size, minimum_batch=500)
    if batch:
        target.executemany("INSERT INTO pair_scores(qrow,target_id,score) VALUES(?,?,?)", batch)
    target.commit()
    target.close()
    source.close()
    stats = {
        "rows": rows,
        "score_sqlite_bytes": score_db.stat().st_size,
        "seconds": time.perf_counter() - started,
    }
    write_json(score_db.with_suffix(".stats.json"), stats)
    log(f"scoring complete: {rows:,} rows in {stats['seconds']:.1f}s")
    return score_db, stats


def load_scored_data(score_db: Path, qids: Set[int]) -> Dict[int, List[Tuple[str, float, int]]]:
    connection = sqlite3.connect(score_db)
    result: Dict[int, List[Tuple[str, float, int]]] = defaultdict(list)
    for qrow, target_id, score in connection.execute(
        "SELECT qrow,target_id,score FROM pair_scores ORDER BY qrow,target_id"
    ):
        if qrow in qids:
            result[qrow].append((target_id, score, 0))
    connection.close()
    return result


def entity_f05(true_count: int, predicted_count: int, true_positive: int) -> float:
    if true_count == 0:
        return 1.0 if predicted_count == 0 else 0.0
    if true_positive == 0:
        return 0.0
    precision = true_positive / predicted_count
    recall = true_positive / true_count
    return 1.25 * precision * recall / (0.25 * precision + recall)


def optimize_policy(
    validation: Mapping[int, Sequence[Tuple[str, float, int]]],
    truth: Mapping[int, Set[str]],
    validation_qids: Sequence[int],
) -> Tuple[Dict[str, float | int], List[Dict[str, float | int]]]:
    if not validation_qids:
        raise ValueError("policy optimization requires validation queries")
    prepared: Dict[int, Tuple[np.ndarray, np.ndarray, int]] = {}
    for qrow in validation_qids:
        rows = sorted(validation.get(qrow, ()), key=lambda item: (-item[1], item[0]))
        actual = truth[qrow]
        labels = np.fromiter((int(target in actual) for target, _score, _ in rows), dtype=np.int8)
        cumulative = np.concatenate(([0], np.cumsum(labels, dtype=np.int64)))
        prepared[qrow] = (
            np.asarray([-score for _target, score, _ in rows], dtype=np.float64),
            cumulative,
            len(actual),
        )

    coarse = np.linspace(0.05, 0.995, 190).round(5)
    fine = np.linspace(0.95, 0.9999, 500).round(4)
    thresholds = sorted(set([0.0, 0.999, *coarse.tolist(), *fine.tolist()]))
    caps = [1, 2, 3, 4, 5, 8, 12, 20, 50, 100, 1_000_000]
    trials: List[Dict[str, float | int]] = []
    best: Dict[str, float | int] | None = None
    best_key = (-1.0, -math.inf)
    for threshold in thresholds:
        for cap in caps:
            macro = 0.0
            total_tp = total_pred = total_true = 0
            for qrow in validation_qids:
                negative_scores, cumulative, true_count = prepared[qrow]
                eligible = int(np.searchsorted(negative_scores, -threshold, side="right"))
                predicted = min(cap, eligible)
                true_positive = int(cumulative[predicted])
                macro += entity_f05(true_count, predicted, true_positive)
                total_tp += true_positive
                total_pred += predicted
                total_true += true_count
            result = {
                "threshold": float(threshold),
                "max_predictions_per_query": cap,
                "macro_f05": macro / len(validation_qids),
                "micro_precision": total_tp / total_pred if total_pred else 1.0,
                "micro_recall": total_tp / total_true if total_true else 1.0,
            }
            trials.append(result)
            key = (float(result["macro_f05"]), float(result["micro_precision"]))
            if key > best_key:
                best_key = key
                best = result
    assert best is not None
    trials.sort(key=lambda item: (float(item["macro_f05"]), float(item["micro_precision"])), reverse=True)
    return best, trials[:20]


def apply_policy(
    rows: Sequence[Tuple[str, float, int]],
    truth: Mapping[int, int],
    qrow: int,
    threshold: float,
    cap: int,
) -> Tuple[List[str], List[str]]:
    ordered = sorted(rows, key=lambda item: (-item[1], item[0]))
    candidates = [target for target, score, _ in ordered]
    predicted = [target for target, score, _ in ordered if score >= threshold][:cap]
    truth_ids = truth[qrow]
    return candidates, predicted


def evaluate_and_write_predictions(
    score_db: Path,
    work_dir: Path,
    queries: Sequence[QueryRecord],
    truth: Mapping[int, Set[str]],
    policy: Mapping[str, float | int],
) -> Dict[str, object]:
    by_qrow = {query.qrow: query for query in queries}
    split_qids: Dict[str, List[int]] = defaultdict(list)
    for query in queries:
        split_qids[query.split].append(query.qrow)
    threshold = float(policy["threshold"])
    cap = int(policy["max_predictions_per_query"])
    validation = load_scored_data(score_db, set(split_qids["validation"]))
    validation_policy = dict(policy)
    validation_policy.update(macro_f05(
        split_qids["validation"],
        truth,
        {qrow: apply_policy(validation.get(qrow, ()), truth, qrow, threshold, cap)[1]
         for qrow in split_qids["validation"]},
    ))
    del validation

    connection = sqlite3.connect(score_db)
    cursor = connection.execute("SELECT qrow,target_id,score FROM pair_scores ORDER BY qrow,target_id")
    grouped: Dict[int, List[Tuple[str, float, int]]] = defaultdict(list)
    for qrow, target_id, score in cursor:
        grouped[qrow].append((target_id, score, 0))
    connection.close()

    all_predictions: Dict[int, List[str]] = {}
    candidate_file = work_dir / "pilot_candidate_pairs.tsv"
    matching_file = work_dir / "pilot_matching_results.tsv"
    detail_file = work_dir / "pilot_predictions.tsv"
    with candidate_file.open("w", encoding="utf-8", newline="") as chandle, \
            matching_file.open("w", encoding="utf-8", newline="") as mhandle, \
            detail_file.open("w", encoding="utf-8", newline="") as dhandle:
        cwriter = csv.writer(chandle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        mwriter = csv.writer(mhandle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        dwriter = csv.writer(dhandle, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        cwriter.writerow(["source1_entity_id", "candidate_entity_ids"])
        mwriter.writerow(["source1_entity_id", "matched_entity_ids"])
        dwriter.writerow(["qrow", "source1_entity_id", "split", "candidate_count", "prediction_count", "max_candidate_score", "true_count"])
        for qrow in range(len(queries)):
            query = by_qrow[qrow]
            rows = grouped.get(qrow, [])
            candidates, predicted = apply_policy(rows, truth, qrow, threshold, cap)
            all_predictions[qrow] = predicted
            cwriter.writerow([query.entity_id, ",".join(candidates)])
            mwriter.writerow([query.entity_id, ",".join(predicted)])
            max_score = max((score for _target, score, _ in rows), default=0.0)
            dwriter.writerow([qrow, query.entity_id, query.split, len(candidates), len(predicted), f"{max_score:.8f}", len(truth[qrow])])

    evaluation: Dict[str, object] = {
        "policy": policy,
        "validation": validation_policy,
        "test": macro_f05(split_qids["test"], truth, all_predictions),
        "all_10000_descriptive_including_train": macro_f05([q.qrow for q in queries], truth, all_predictions),
        "split_query_counts": {name: len(values) for name, values in split_qids.items()},
    }
    write_json(work_dir / "evaluation.json", evaluation)
    return evaluation


def make_model_artifact(
    path: Path,
    weights: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    training_stats: Mapping[str, object],
    policy: Mapping[str, object],
) -> None:
    write_json(path, {
        "model_type": "compact_cpu_logistic_regression",
        "feature_names": FEATURE_NAMES,
        "weights": weights.tolist(),
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "training": training_stats,
        "decision_policy": policy,
    })


def build_projection(
    work_dir: Path,
    queries: Sequence[QueryRecord],
    candidate_stats: Mapping[str, object],
    feature_stats: Mapping[str, object],
    timer: PhaseTimer,
    monitor: MemoryMonitor,
) -> Dict[str, object]:
    test_s1 = 1_732_544
    test_targets = 4_887_273 + 5_082_316
    train_targets = int(candidate_stats["target_rows_scanned"])
    query_scale = test_s1 / len(queries)
    target_scale = test_targets / train_targets
    combined_scale = query_scale * target_scale
    average_candidates = float(candidate_stats["candidate_count_per_s1_mean"])
    projected_candidates = int(round(average_candidates * test_s1))
    candidate_bytes_per_row = (
        int(candidate_stats["candidate_sqlite_bytes"]) / max(1, int(candidate_stats["candidate_count"]))
    )
    feature_bytes_per_row = float(feature_stats["bytes_per_candidate_including_key"])
    projected_candidate_db = projected_candidates * candidate_bytes_per_row
    projected_feature_db = projected_candidates * feature_bytes_per_row
    estimated_text_output = projected_candidates * 12 + test_s1 * 24
    average_target_keys = float(candidate_stats["average_generated_keys_per_target"])
    projected_index_postings = int(average_target_keys * test_targets)
    # Conservative SQLite B-tree estimate: 48 logical/physical bytes per posting.
    projected_index_db = projected_index_postings * 48
    free_bytes = disk_free_bytes(work_dir)
    pilot_compute_seconds = float(timer.phases.get("total_wall", sum(timer.phases.values())))
    projection = {
        "method": "linear scaling by S1 query count and S2+S3 target rows; conservative 48 bytes/index posting",
        "test_s1_rows": test_s1,
        "test_s2_s3_rows": test_targets,
        "query_scale": query_scale,
        "target_scale": target_scale,
        "combined_compute_scale": combined_scale,
        "projected_candidate_pairs": projected_candidates,
        "projected_candidates_per_s1": average_candidates,
        "projected_candidate_sqlite_bytes": projected_candidate_db,
        "projected_feature_sqlite_bytes": projected_feature_db,
        "projected_candidate_pairs_tsv_bytes": estimated_text_output,
        "projected_full_inverted_index_postings": projected_index_postings,
        "projected_full_inverted_index_bytes": projected_index_db,
        "projected_working_artifacts_bytes": (
            projected_candidate_db + projected_feature_db + projected_index_db + estimated_text_output
        ),
        "filesystem_free_bytes_at_report": free_bytes,
        "pilot_compute_seconds": pilot_compute_seconds,
        "naive_linear_runtime_seconds": pilot_compute_seconds * combined_scale,
        "target_working_memory_gib": 4.0,
        "notes": [
            "Projection is linear and does not claim speedup from a future full index build.",
            "Pilot feature BLOBs are intentionally uncompressed; full scale should use float16 or compressed Parquet shards and delete scored shards.",
            "A full index is not built by this pilot; its size is estimated from measured average generated keys.",
        ],
    }
    write_json(work_dir / "full_scale_projection.json", projection)
    return projection


def write_report(work_dir: Path, payload: Mapping[str, object]) -> None:
    candidate = payload["candidate"]
    evaluation = payload["evaluation"]
    blocking = payload["blocking"]
    memory = payload["resources"]["memory"]
    projection = payload["projection"]
    report = f"""# 10K Training S1 Pilot Report

## Scope

- Pilot queries: **{len(payload['queries']):,} uniformly sampled training S1 records**.
- Targets streamed: **{candidate['target_rows_scanned']:,}** training S2/S3 records.
- Test data processed: **0**.
- Primary score: macro F0.5 on the deterministic held-out pilot test split.
- No dense cross-source matrix was constructed.

## Blocking and candidates

- True-pair blocking recall: **{100.0 * float(candidate['true_pair_recall']):.4f}%**.
- Candidate pairs: **{int(candidate['candidate_count']):,}** ({float(candidate['candidate_count_per_s1_mean']):.2f}/S1).
- Candidate reduction ratio: **{100.0 * float(candidate['candidate_reduction_ratio']):.6f}%**.
- S1 rows with at least one candidate: **{int(candidate['s1_rows_with_candidates']):,}**.
- Selected frequency cap: **{int(blocking['selected_block_frequency_cap']):,}** target rows/block.
- Selected per-query posting cap: **{int(blocking['selected_postings_per_query_cap']):,}** raw postings/query.

## Matching model

- Model: compact CPU L2 logistic regression over {len(FEATURE_NAMES)} incremental name/address features.
- Validation macro F0.5: **{100.0 * float(evaluation['validation']['macro_f05']):.4f}%**.
- Held-out test macro F0.5: **{100.0 * float(evaluation['test']['macro_f05']):.4f}%**.
- Policy: score threshold **{float(evaluation['policy']['threshold']):.5f}**, at most **{int(evaluation['policy']['max_predictions_per_query']):,}** prediction(s)/S1.

## Resources

- Pilot wall time: **{float(payload['resources']['phase_seconds']['total_wall']):.1f}s**.
- Peak process RSS: **{float(memory['peak_process_rss_mib']):.1f} MiB**.
- Minimum system `MemAvailable`: **{float(memory['minimum_system_mem_available_mib']):.1f} MiB**.
- Pilot candidate DB: **{int(candidate['candidate_sqlite_bytes']) / 1024**2:.1f} MiB**.
- Pilot feature DB: **{int(payload['feature']['feature_sqlite_bytes']) / 1024**2:.1f} MiB**.

## Full-scale linear projection

- Candidate pairs: **{int(projection['projected_candidate_pairs']):,}**.
- Naive scaled compute time: **{float(projection['naive_linear_runtime_seconds']) / 3600.0:.2f} h**.
- Working artifacts: **{float(projection['projected_working_artifacts_bytes']) / 1024**3:.2f} GiB**.
- Full inverted-index estimate: **{float(projection['projected_full_inverted_index_bytes']) / 1024**3:.2f} GiB**.
- Projection assumes a **4 GiB** working-memory target and remains bounded batches; it is not authorization to run full scale.

See `pilot_report.json` for exact per-phase, per-scheme, and projection details.
"""
    with (work_dir / "REPORT.md").open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(report)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--work-dir", type=Path, default=PROJECT_ROOT / "artifacts/pilot_10k")
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--max-projected-postings", type=int, default=2_000_000)
    parser.add_argument(
        "--selection-cap",
        type=int,
        default=None,
        help="predeclare one matched block/query posting cap; labels remain report-only",
    )
    parser.add_argument("--negative-per-query", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=10_000)
    args = parser.parse_args(argv)
    for name in ("sample_size", "max_projected_postings", "negative_per_query", "batch_size"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.selection_cap is not None and args.selection_cap <= 0:
        parser.error("--selection-cap must be positive")
    return args


def main() -> int:
    args = parse_args()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.data_root = args.data_root.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.work_dir / ".run.lock").open("a+", encoding="utf-8")
    if not acquire_run_lock(lock_handle):
        raise SystemExit(f"a pilot process already holds {args.work_dir / '.run.lock'}")
    try:
        return run_pipeline(args)
    finally:
        lock_handle.close()


def run_pipeline(args: argparse.Namespace) -> int:
    if args.sample_size != 10_000:
        log("WARNING: this run differs from the requested 10,000-record pilot")
    monitor = MemoryMonitor().start()
    timer = PhaseTimer(phases={})
    overall_started = time.perf_counter()

    started = timer.start()
    queries, truth, _split = sample_queries(args.data_root, args.work_dir, args.sample_size)
    timer.finish("sample", started)

    started = timer.start()
    key_index = build_query_key_index(queries)
    log(f"query inverted keys: {len(key_index):,} unique postings keys")
    timer.finish("build_query_index", started)

    started = timer.start()
    key_counts, true_keys, frequency_stats = measure_block_frequencies(
        args.data_root, queries, truth, key_index, monitor
    )
    timer.finish("measure_block_frequencies", started)

    started = timer.start()
    active_index, retained_by_query, selection_stats = select_block_postings(
        queries, key_index, key_counts, true_keys, truth,
        args.max_projected_postings, args.selection_cap,
        int(frequency_stats["target_rows_scanned"]),
    )
    del retained_by_query
    timer.finish("select_block_postings", started)

    blocking_stats = {
        "frequency_pass": frequency_stats,
        "selection": selection_stats,
        "index_key_count": len(key_index),
    }
    write_json(args.work_dir / "blocking_stats.json", blocking_stats)

    started = timer.start()
    candidate_db, candidate_stats = persist_candidates(
        args.data_root, args.work_dir, queries, truth, active_index, monitor, args.batch_size
    )
    timer.finish("persist_candidates", started)
    del active_index, key_index, key_counts, true_keys
    gc.collect()

    started = timer.start()
    feature_db, feature_stats = build_feature_database(
        candidate_db, args.work_dir / "pilot_features.sqlite", queries, args.batch_size, monitor
    )
    timer.finish("generate_features", started)

    started = timer.start()
    weights, mean, std, training_stats = train_compact_classifier(
        candidate_db, feature_db, queries, truth, args.negative_per_query, monitor
    )
    timer.finish("train_classifier", started)

    started = timer.start()
    score_db, score_stats = score_all_candidates(
        feature_db, args.work_dir / "pilot_scores.sqlite", weights, mean, std,
        args.batch_size, monitor
    )
    timer.finish("score_candidates", started)

    started = timer.start()
    validation_qids = [query.qrow for query in queries if query.split == "validation"]
    validation_data = load_scored_data(score_db, set(validation_qids))
    # attach labels only for training-policy optimization
    for qrow, rows in validation_data.items():
        actual = truth[qrow]
        validation_data[qrow] = [(target, score, int(target in actual)) for target, score, _ in rows]
    policy, policy_trials = optimize_policy(validation_data, truth, validation_qids)
    del validation_data
    evaluation = evaluate_and_write_predictions(score_db, args.work_dir, queries, truth, policy)
    evaluation["policy_trials"] = policy_trials
    write_json(args.work_dir / "evaluation.json", evaluation)
    make_model_artifact(
        args.work_dir / "model.json", weights, mean, std, training_stats, policy
    )
    timer.finish("evaluate_and_write", started)

    monitor.stop()
    timer.phases["total_wall"] = time.perf_counter() - overall_started
    projection = build_projection(
        args.work_dir, queries, candidate_stats, feature_stats, timer, monitor
    )
    payload: Dict[str, object] = {
        "queries": [
            {"qrow": q.qrow, "entity_id": q.entity_id, "split": q.split}
            for q in queries
        ],
        "sample_summary": {
            "query_count": len(queries),
            "split_counts": dict(Counter(q.split for q in queries)),
            "singleton_queries": sum(not value for value in truth.values()),
            "true_pairs": sum(len(value) for value in truth.values()),
        },
        "blocking": selection_stats,
        "candidate": candidate_stats,
        "feature": feature_stats,
        "training": training_stats,
        "score": score_stats,
        "evaluation": evaluation,
        "resources": {
            "phase_seconds": timer.phases,
            "memory": monitor.as_dict(),
            "start_available_memory_bytes": available_memory_bytes(),
        },
        "projection": projection,
    }
    write_json(args.work_dir / "pilot_report.json", payload)
    write_report(args.work_dir, payload)
    log(f"pilot complete: held-out macro F0.5={evaluation['test']['macro_f05']:.6f}; "
        f"blocking recall={candidate_stats['true_pair_recall']:.6f}; "
        f"peak RSS={monitor.peak_rss / 1024**2:.1f} MiB; total={timer.phases['total_wall']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
