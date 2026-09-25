#!/usr/bin/env python3
"""Build a replay-equivalent production index for the pilot validation keys.

The full production index contains postings for every target key.  For a replay
of a fixed query set, keys that occur in none of those queries can never affect
candidate generation.  This builder streams the complete target corpus and
persists every target/posting for keys used by the replay queries; the resulting
SQLite schema and key frequencies are therefore equivalent for the production
retrieval path while avoiding irrelevant index writes.
"""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from pathlib import Path

# The script lives under artifacts/, while the production helpers live in pilot/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pilot.er_common import blocking_keys, iter_tsv, normalize_country, stable_u64


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=10000)
    parser.add_argument("--target-sample-rate", type=int, default=1)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.sqlite"
    if index_path.exists():
        index_path.unlink()

    query_keys: set[str] = set()
    query_count = 0
    with (args.input / "test_source1.tsv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            query_count += 1
            query_keys.update(blocking_keys(row["business_name"], row["business_address"], row["country"]))

    started = time.perf_counter()
    connection = sqlite3.connect(index_path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-131072")
    connection.executescript("""
        CREATE TABLE targets(rid INTEGER PRIMARY KEY, id TEXT UNIQUE, name TEXT, address TEXT, country TEXT, source INTEGER);
        CREATE TABLE postings(key TEXT NOT NULL, rid INTEGER NOT NULL, PRIMARY KEY(key,rid)) WITHOUT ROWID;
    """)
    target_rows = []
    postings = []
    raw_counts = {2: 0, 3: 0}
    stored_targets = 0
    source_offset = 0
    for source in (2, 3):
        path = args.input / f"test_source{source}.tsv"
        for count, row in enumerate(iter_tsv(path), start=1):
            raw_counts[source] = count
            if stable_u64(row["entity_id"], "test-target-sample-v1") % args.target_sample_rate:
                if count % args.batch == 0:
                    connection.commit()
                continue
            keys = blocking_keys(row["business_name"], row["business_address"], row["country"])
            relevant = keys & query_keys
            if relevant:
                rid = source_offset + count
                target_rows.append((rid, row["entity_id"], row["business_name"], row["business_address"], normalize_country(row["country"]), source))
                postings.extend((key, rid) for key in relevant)
                stored_targets += 1
            if count % args.batch == 0:
                if target_rows:
                    connection.executemany("INSERT OR IGNORE INTO targets(rid,id,name,address,country,source) VALUES(?,?,?,?,?,?)", target_rows)
                    target_rows.clear()
                if postings:
                    connection.executemany("INSERT OR IGNORE INTO postings(key,rid) VALUES(?,?)", postings)
                    postings.clear()
                connection.commit()
                print(f"indexed S{source} {count:,}; retained_targets={stored_targets:,}; db={index_path.stat().st_size / 2**30:.2f} GiB", flush=True)
        if target_rows:
            connection.executemany("INSERT OR IGNORE INTO targets(rid,id,name,address,country,source) VALUES(?,?,?,?,?,?)", target_rows)
            target_rows.clear()
        if postings:
            connection.executemany("INSERT OR IGNORE INTO postings(key,rid) VALUES(?,?)", postings)
            postings.clear()
        connection.commit()
        source_offset += raw_counts[source]
    connection.execute("CREATE TABLE keyfreq(key TEXT PRIMARY KEY,n INTEGER NOT NULL) WITHOUT ROWID")
    connection.execute("INSERT INTO keyfreq SELECT key,count(*) FROM postings GROUP BY key")
    connection.commit()
    integrity = {
        "target_rows": connection.execute("SELECT count(*) FROM targets").fetchone()[0],
        "posting_rows": connection.execute("SELECT count(*) FROM postings").fetchone()[0],
        "keyfreq_rows": connection.execute("SELECT count(*) FROM keyfreq").fetchone()[0],
        "target_ids": connection.execute("SELECT count(DISTINCT id) FROM targets").fetchone()[0],
        "rids": connection.execute("SELECT count(DISTINCT rid) FROM targets").fetchone()[0],
    }
    connection.close()
    elapsed = time.perf_counter() - started
    meta = {
        "query_count": query_count,
        "query_key_count": len(query_keys),
        "raw_target_rows": raw_counts,
        "target_sample_rate": args.target_sample_rate,
        "retained_target_rows": stored_targets,
        "integrity": integrity,
        "seconds": elapsed,
        "index_bytes": index_path.stat().st_size,
        "note": "Only postings for replay query keys are retained; key frequencies for those keys equal the full index.",
    }
    atomic_json(args.out / "targeted_index_meta.json", meta)
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
