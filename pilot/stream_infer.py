#!/usr/bin/env python3
"""Disk-backed, restartable test inference with safe multi-process sharding.

This version fixes three issues important for the Windows multi-worker setup:
1. --query-stride/--query-offset are enforced in FULL mode, so workers process
   disjoint S1 rows.
2. The per-workdir lock is real on both POSIX and Windows, preventing two
   processes from writing the same results.sqlite/checkpoint.json.
3. --no-output lets worker processes skip per-worker TSV generation; a separate
   streaming merge can build the final submission without huge RAM usage.
"""
from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import platform
import sqlite3
import time
from pathlib import Path

import numpy as np
from er_common import (
    FEATURE_NAMES,
    PairText,
    blocking_keys,
    iter_tsv,
    normalize_country,
    pair_features,
    stable_u64,
)

IS_WINDOWS = platform.system() == "Windows"

if not IS_WINDOWS:
    import fcntl
else:
    import msvcrt

try:
    import resource
except ImportError:
    resource = None

VERSION = 2
POSTING_CAP = 100
QUERY_POSTING_CAP = 100


def get_free_disk_space(path: Path) -> int:
    """Return free bytes on the filesystem containing path."""
    if IS_WINDOWS:
        import ctypes

        free_bytes = ctypes.c_ulonglong(0)
        ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
            ctypes.c_wchar_p(str(path.resolve())),
            ctypes.byref(free_bytes),
            None,
            None,
        )
        if not ok:
            raise OSError("GetDiskFreeSpaceExW failed")
        return int(free_bytes.value)

    st = os.statvfs(path)
    return int(st.f_bavail * st.f_frsize)


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass

    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.1 * (attempt + 1))


def db(path: Path, cache=64, synchronous="FULL", journal_mode="WAL"):
    c = sqlite3.connect(path)
    c.execute(f"PRAGMA journal_mode={journal_mode}")
    c.execute(f"PRAGMA synchronous={synchronous}")
    c.execute("PRAGMA temp_store=FILE")
    c.execute(f"PRAGMA cache_size=-{cache * 1024}")
    c.execute("PRAGMA wal_autocheckpoint=10000")
    return c


def rss() -> int:
    """Best-effort RSS in bytes; Windows fallback returns 0."""
    if resource is None:
        return 0
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if IS_WINDOWS:
        return 0
    return int(value * 1024)


def work_bytes(path: Path) -> int:
    total = 0
    for p in path.iterdir():
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def acquire_run_lock(lock_path: Path):
    """Acquire an exclusive per-workdir lock and keep its handle open.

    On Windows this uses msvcrt.locking instead of the previous dummy flock
    implementation. A second process targeting the same work directory will
    now fail before touching the SQLite result database.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)

        if IS_WINDOWS:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise SystemExit(
                    f"an inference process already holds {lock_path}"
                ) from exc
        else:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SystemExit(
                    f"an inference process already holds {lock_path}"
                ) from exc

        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n".encode("ascii"))
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass
        return handle
    except BaseException:
        handle.close()
        raise


<<<<<<< HEAD
def release_run_lock(handle) -> None:
    if handle is None:
        return
    try:
        handle.seek(0)
        if IS_WINDOWS:
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
        else:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        try:
            handle.close()
        except OSError:
            pass


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("dataset/test"))
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--model", type=Path, default=Path("pilot/frozen_pilot_model.json"))
    p.add_argument(
        "--queries",
        type=int,
        default=2500,
        help="bounded preflight query count; 0 means all S1 in full mode",
    )
    p.add_argument("--mode", choices=("preflight", "full"), default="preflight")
    p.add_argument("--index-batch", type=int, default=5000)
    p.add_argument(
        "--target-sample-rate",
        type=int,
        default=1,
        help="index each Nth deterministic target hash; use 20 for preflight",
    )
    p.add_argument(
        "--query-stride",
        type=int,
        default=1,
        help="number of S1 shards; process only rows where qi %% stride == offset",
    )
    p.add_argument(
        "--query-offset",
        type=int,
        default=0,
        help="zero-based shard offset in [0, query-stride-1]",
    )
    p.add_argument("--block-cap", type=int, default=POSTING_CAP)
    p.add_argument("--query-posting-cap", type=int, default=QUERY_POSTING_CAP)
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="directory for TSV outputs",
    )
    p.add_argument(
        "--no-output",
        action="store_true",
        help="skip TSV generation; recommended for parallel workers",
    )
    p.add_argument(
        "--index-synchronous",
        choices=("FULL", "NORMAL", "OFF"),
        default="FULL",
        help="SQLite synchronous mode while building the target index",
    )
    return p.parse_args()


def main():
    a = parse_args()

    if a.query_stride < 1:
        raise SystemExit("--query-stride must be >= 1")
    if not 0 <= a.query_offset < a.query_stride:
        raise SystemExit("--query-offset must satisfy 0 <= offset < query-stride")
    if a.queries < 0:
        raise SystemExit("--queries must be >= 0")
    if a.target_sample_rate < 1:
        raise SystemExit("--target-sample-rate must be >= 1")

    a.work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = a.output_dir or a.work_dir
    if not a.no_output:
        output_dir.mkdir(parents=True, exist_ok=True)

    lock_handle = acquire_run_lock(a.work_dir / ".run.lock")
    atexit.register(release_run_lock, lock_handle)

    paths = {n: a.data_root / f"test_source{n}.tsv" for n in (1, 2, 3)}
    for path in paths.values():
        if not path.exists():
            raise SystemExit(f"missing dataset file: {path}")

    fingerprint = {
        str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths.values()
    }
    state_path = a.work_dir / "checkpoint.json"
    index_path = a.work_dir / "index.sqlite"
    result_path = a.work_dir / "results.sqlite"

    profile = {
        "target_sample_rate": a.target_sample_rate,
        "query_stride": a.query_stride,
        "query_offset": a.query_offset,
        "block_cap": a.block_cap,
        "query_posting_cap": a.query_posting_cap,
    }
=======
def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=PROJECT_ROOT/'dataset/test')
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,default=PROJECT_ROOT/'pilot/frozen_pilot_model.json')
    p.add_argument('--queries',type=int,default=2500,help='bounded preflight query count; 0 means all S1')
    p.add_argument('--mode',choices=('preflight','full'),default='preflight')
    p.add_argument('--index-batch',type=int,default=5000)
    p.add_argument('--target-sample-rate',type=int,default=1,help='index each Nth deterministic target hash; use 20 for preflight')
    p.add_argument('--query-stride',type=int,default=1,help='number of S1 shards; process only rows where qi %% stride == offset')
    p.add_argument('--query-offset',type=int,default=0,help='zero-based shard offset in [0, query-stride-1]')
    p.add_argument('--block-cap',type=int,default=POSTING_CAP)
    p.add_argument('--query-posting-cap',type=int,default=QUERY_POSTING_CAP)
    p.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'output',
                   help='directory for final matching_results.tsv and candidate_pairs.tsv (default: repository output)')
    p.add_argument('--no-output',action='store_true',
                   help='skip TSV generation; recommended for parallel shard workers, which are merged separately')
    p.add_argument('--index-only',action='store_true',
                   help='build and commit the target index, then exit without any query work; '
                        'leaves a self-contained index.sqlite for a sharded query run')
    p.add_argument('--index-synchronous',choices=('FULL','NORMAL','OFF'),default='FULL',
                   help='SQLite synchronous mode while building the target index')
    p.add_argument('--safety-free-gib',type=float,default=20.0,
                   help='minimum free space to preserve; use 0 only for tiny test fixtures')
    args=p.parse_args(argv)
    if args.queries < 0:
        p.error('--queries must be non-negative')
    for name in ('index_batch','target_sample_rate','query_stride','block_cap','query_posting_cap'):
        if getattr(args,name) <= 0:
            p.error(f'--{name.replace("_", "-")} must be positive')
    if args.safety_free_gib < 0:
        p.error('--safety-free-gib must be non-negative')
    if args.mode == 'full' and args.queries != 0:
        p.error('--queries must be 0 in full mode (0 means every S1 row)')
    if not 0 <= args.query_offset < args.query_stride:
        p.error('--query-offset must be zero-based and satisfy 0 <= offset < query-stride')
    return args

def main():
    a=parse_args()
    a.work_dir=a.work_dir.expanduser().resolve()
    a.data_root=a.data_root.expanduser().resolve()
    a.model=a.model.expanduser().resolve()
    a.output_dir=(a.output_dir or a.work_dir).expanduser().resolve()
    a.work_dir.mkdir(parents=True,exist_ok=True)
    output_dir=a.output_dir
    # An index-only run never renders TSVs, so it must not create or lock the
    # output directory either.
    write_output=not (a.no_output or a.index_only)
    if write_output:
        output_dir.mkdir(parents=True,exist_ok=True)
    lock_handle=(a.work_dir/'.run.lock').open('a+', encoding='utf-8')
    if not acquire_run_lock(lock_handle):
        raise SystemExit(f'an inference process already holds {a.work_dir}/.run.lock')
    output_lock_handle=None
    if write_output and output_dir != a.work_dir:
        output_lock_handle=(output_dir/'.output.lock').open('a+', encoding='utf-8')
        if not acquire_run_lock(output_lock_handle):
            raise SystemExit(f'an inference process already writes {output_dir}')
    paths={n:a.data_root/f'test_source{n}.tsv' for n in (1,2,3)}
    missing_inputs=[str(path) for path in paths.values() if not path.is_file()]
    if missing_inputs:
        raise SystemExit('missing input file(s): ' + ', '.join(missing_inputs))
    if not a.model.is_file():
        raise SystemExit(f'model file not found: {a.model}')
    try:
        with a.model.open(encoding='utf-8') as model_handle:
            model=json.load(model_handle)
        means=np.asarray(model['feature_mean'],dtype=np.float32)
        std=np.asarray(model['feature_std'],dtype=np.float32)
        weights=np.asarray(model['weights'],dtype=np.float32)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f'invalid model artifact {a.model}: {exc}') from exc
    if (model.get('feature_names') != FEATURE_NAMES
            or means.shape != (len(FEATURE_NAMES),)
            or std.shape != (len(FEATURE_NAMES),)
            or weights.shape != (len(FEATURE_NAMES),)
            or not np.isfinite(means).all()
            or not np.isfinite(std).all()
            or not np.isfinite(weights).all()
            or (std < 0).any()):
        raise SystemExit('model feature schema incompatibility')
    decision_policy=model.get('decision_policy', {})
    if (not isinstance(decision_policy.get('threshold'), (int, float))
            or not 0.0 <= float(decision_policy['threshold']) <= 1.0
            or not isinstance(decision_policy.get('max_predictions_per_query'), int)
            or decision_policy['max_predictions_per_query'] < 1):
        raise SystemExit('model decision policy is invalid')
>>>>>>> eaac310 (1)

    if state_path.exists():
        with state_path.open("r", encoding="utf-8") as f:
            state = json.load(f)
    else:
        state = {
            "version": VERSION,
            "fingerprint": fingerprint,
            "profile": profile,
            "index_complete": False,
            "query_cursor": 0,
        }
        atomic_json(state_path, state)

    if (
        state.get("version") != VERSION
        or state.get("fingerprint") != fingerprint
        or state.get("profile") != profile
    ):
        raise SystemExit(
            "checkpoint/data/profile version mismatch; use a fresh work directory "
            "for the corrected sharded run"
        )

    t0 = time.time()
    c = db(
        index_path,
        synchronous=a.index_synchronous,
        journal_mode="OFF" if a.index_synchronous == "OFF" else "WAL",
    )
    c.executescript(
        """CREATE TABLE IF NOT EXISTS targets(
               rid INTEGER PRIMARY KEY,
               id TEXT UNIQUE,
               name TEXT,
               address TEXT,
               country TEXT,
               source INTEGER
           );
           CREATE TABLE IF NOT EXISTS postings(
               key TEXT NOT NULL,
               rid INTEGER NOT NULL,
               PRIMARY KEY(key,rid)
           ) WITHOUT ROWID;"""
    )

    if not state["index_complete"]:

        def flush_index_batch(src, count, target_rows, posts):
            if target_rows:
                c.executemany(
                    "INSERT OR IGNORE INTO targets(rid,id,name,address,country,source) "
                    "VALUES(?,?,?,?,?,?)",
                    target_rows,
                )
            if posts:
                c.executemany("INSERT OR IGNORE INTO postings VALUES(?,?)", posts)
            c.commit()
            state[f"source{src}_rows"] = count
            state["peak_work_bytes"] = max(
                state.get("peak_work_bytes", 0), work_bytes(a.work_dir)
            )
            atomic_json(state_path, state)
            free = get_free_disk_space(a.work_dir)
            if free < 20 * 2**30:
                raise SystemExit(
                    f"safety stop: free disk below 20 GiB ({free / 2**30:.2f} GiB)"
                )
            if count % a.index_batch == 0:
                print(
                    f"indexed S{src} {count:,}; "
                    f"db={index_path.stat().st_size / 2**30:.2f} GiB; "
                    f"rss={rss() / 2**20:.0f} MiB",
                    flush=True,
                )

        for src in (2, 3):
            path = paths[src]
            done = int(state.get(f"source{src}_rows", 0))
            count = 0
            target_rows = []
            posts = []
            source_offset = 0 if src == 2 else int(state.get("source2_rows", 0))

            for row in iter_tsv(path):
                count += 1
                if count <= done:
                    continue
                if stable_u64(row["entity_id"], "test-target-sample-v1") % a.target_sample_rate:
                    if count % a.index_batch == 0:
                        flush_index_batch(src, count, target_rows, posts)
                        target_rows.clear()
                        posts.clear()
                    continue

                rid = source_offset + count
                target_rows.append(
                    (
                        rid,
                        row["entity_id"],
                        row["business_name"],
                        row["business_address"],
                        normalize_country(row["country"]),
                        src,
                    )
                )
<<<<<<< HEAD
                posts.extend(
                    (key, rid)
                    for key in blocking_keys(
                        row["business_name"],
                        row["business_address"],
                        row["country"],
                    )
                )
                if count % a.index_batch == 0:
                    flush_index_batch(src, count, target_rows, posts)
                    target_rows.clear()
                    posts.clear()

            flush_index_batch(src, count, target_rows, posts)

        c.execute(
            "CREATE TABLE IF NOT EXISTS keyfreq(" 
            "key TEXT PRIMARY KEY,n INTEGER NOT NULL) WITHOUT ROWID"
        )
        c.execute(
            "INSERT OR REPLACE INTO keyfreq SELECT key,count(*) FROM postings GROUP BY key"
        )
        c.commit()
        state["index_complete"] = True
        state["index_seconds"] = time.time() - t0
        atomic_json(state_path, state)

    indexed_target_counts = {
        src: c.execute(
            "SELECT count(*) FROM targets WHERE source=?", (src,)
        ).fetchone()[0]
        for src in (2, 3)
    }
    c.close()

    with a.model.open("r", encoding="utf-8") as f:
        model = json.load(f)
    means = np.asarray(model["feature_mean"], dtype=np.float32)
    std = np.asarray(model["feature_std"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    if model["feature_names"] != FEATURE_NAMES or len(weights) != len(FEATURE_NAMES):
        raise SystemExit("model feature schema incompatibility")

    r = db(result_path)
    r.executescript(
        """CREATE TABLE IF NOT EXISTS queries(
               qid INTEGER PRIMARY KEY,
               id TEXT UNIQUE,
               name TEXT,
               address TEXT,
               country TEXT
           );
           CREATE TABLE IF NOT EXISTS pairs(
               qid INTEGER,
               target_id TEXT,
               score REAL,
               PRIMARY KEY(qid,target_id)
           ) WITHOUT ROWID;
           CREATE TABLE IF NOT EXISTS candidates(
               qid INTEGER,
               target_id TEXT,
               PRIMARY KEY(qid,target_id)
           ) WITHOUT ROWID;
           CREATE TABLE IF NOT EXISTS completed(qid INTEGER PRIMARY KEY);"""
    )

    query_limit = a.queries if a.mode == "preflight" else 0
    qstart = int(state.get("query_cursor", 0))
    rows = 0
    cand_n = 0
    score_n = 0
    qtime = time.time()
    last_qi = qstart - 1

    ix = sqlite3.connect(index_path)
    ix.execute("PRAGMA cache_size=-65536")

    with paths[1].open(encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        if rd.fieldnames != ["entity_id", "business_name", "business_address", "country"]:
            raise SystemExit("S1 header mismatch")

        for qi, row in enumerate(rd):
            last_qi = qi

            if a.mode == "preflight" and state.get("preflight_complete"):
                break
            if qi < qstart:
                continue

            # IMPORTANT: this applies in BOTH preflight and full modes.
            if a.query_stride > 1 and qi % a.query_stride != a.query_offset:
                continue

            if query_limit and rows >= query_limit:
                break

            qid = qi + 1
            if r.execute("SELECT 1 FROM completed WHERE qid=?", (qid,)).fetchone():
                continue

            rows += 1
            name = row["business_name"]
            address = row["business_address"]
            country = normalize_country(row["country"])
            r.execute(
                "INSERT OR IGNORE INTO queries VALUES(?,?,?,?,?)",
                (qid, row["entity_id"], name, address, country),
            )

            ids = {}
            posting_cap = max(1, a.block_cap // a.target_sample_rate)
            query_posting_cap = max(1, a.query_posting_cap // a.target_sample_rate)
            keys = sorted(blocking_keys(name, address, country))
            raw_rids = []

=======
                target_rows.append((rid,row['entity_id'],target_name,target_address,target_country,src))
                posts.extend((key,rid) for key in blocking_keys(row['business_name'],row['business_address'],row['country']))
                if count%a.index_batch==0:
                    flush_index_batch(src,count,target_rows,posts); target_rows.clear(); posts.clear()
            flush_index_batch(src,count,target_rows,posts); target_rows.clear(); posts.clear()
        c.execute('CREATE TABLE IF NOT EXISTS keyfreq(key TEXT PRIMARY KEY,n INTEGER NOT NULL) WITHOUT ROWID')
        c.execute('INSERT OR REPLACE INTO keyfreq SELECT key,count(*) FROM postings GROUP BY key'); c.commit()
        state['index_complete']=True; state['index_seconds']=time.time()-t0; atomic_json(state_path,state)
    indexed_target_counts={src:c.execute('SELECT count(*) FROM targets WHERE source=?',(src,)).fetchone()[0] for src in (2,3)}
    if a.index_only:
        # Fold the WAL back into the main database *before* closing. A shard
        # launcher copies index.sqlite alone, never the -wal sidecar, so any
        # page still living in the WAL would be missing from every worker's
        # copy and each worker would silently miss those candidates.
        try:
            c.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        except sqlite3.OperationalError:
            pass
    c.close()
    if a.index_only:
        if not state.get('index_complete'):
            raise SystemExit('--index-only requested but the target index is not complete')
        report={'index_only':True,'index_target_rows':sum(indexed_target_counts.values()),
                'indexed_target_rows_by_source':indexed_target_counts,
                'index_source_rows_scanned':sum(int(state.get(f'source{i}_rows',0)) for i in (2,3)),
                'index_seconds':state.get('index_seconds'),
                'index_path':str(index_path),
                'index_bytes':index_path.stat().st_size,
                'index_self_contained':not (index_path.with_name(index_path.name+'-wal').exists()),
                'state_identity':identity}
        atomic_json(a.work_dir/'preflight_report.json',report)
        lock_handle.close()
        if not report['index_self_contained']:
            raise SystemExit('index still has an uncheckpointed -wal; not safe to copy per worker')
        print(json.dumps(report,indent=2))
        return
    r=db(result_path); r.executescript('''CREATE TABLE IF NOT EXISTS queries(qid INTEGER PRIMARY KEY,id TEXT UNIQUE,name TEXT,address TEXT,country TEXT);
    CREATE TABLE IF NOT EXISTS pairs(qid INTEGER,target_id TEXT,score REAL,PRIMARY KEY(qid,target_id)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS candidates(qid INTEGER,target_id TEXT,PRIMARY KEY(qid,target_id)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS completed(qid INTEGER PRIMARY KEY);''')
    query_limit=a.queries if a.mode=='preflight' else 0; qstart=int(state.get('query_cursor',0)); rows=0; cand_n=score_n=0; qtime=time.time()
    last_qi=qstart-1
    ix=sqlite3.connect(index_path); ix.execute('PRAGMA cache_size=-65536')
    with paths[1].open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        if rd.fieldnames!=['entity_id','business_name','business_address','country']: raise SystemExit('S1 header mismatch')
        for qi,row in enumerate(rd):
            last_qi=qi
            if a.mode=='preflight' and state.get('preflight_complete'): break
            if qi<qstart: continue
            # The shard filter applies in BOTH preflight and full mode so that N
            # workers partition S1 into disjoint, complete shards.
            if a.query_stride>1 and qi%a.query_stride!=a.query_offset: continue
            if query_limit and rows>=query_limit: break
            qid=qi+1
            if r.execute('SELECT 1 FROM completed WHERE qid=?',(qid,)).fetchone(): continue
            rows+=1
            name,address,country=row['business_name'],row['business_address'],normalize_country(row['country'])
            r.execute('INSERT OR IGNORE INTO queries VALUES(?,?,?,?,?)',(qid,row['entity_id'],name,address,country))
            ids={}
            posting_cap=max(1,a.block_cap//a.target_sample_rate); query_posting_cap=max(1,a.query_posting_cap//a.target_sample_rate)
            keys=sorted(blocking_keys(name,address,country))
            raw_rids=[]
>>>>>>> eaac310 (1)
            if keys:
                marks = ",".join("?" for _ in keys)
                freq_rows = ix.execute(
                    f"SELECT key,n FROM keyfreq WHERE key IN ({marks}) ORDER BY n,key",
                    keys,
                )
                used = 0
                for key, freq in freq_rows:
                    if freq > posting_cap or freq > query_posting_cap - used:
                        continue
                    raw_rids.extend(
                        r0[0]
                        for r0 in ix.execute(
                            "SELECT rid FROM postings WHERE key=? ORDER BY rid", (key,)
                        )
                    )
                    used += freq

                if raw_rids:
                    unique_rids = list(dict.fromkeys(raw_rids))
                    target_marks = ",".join("?" for _ in unique_rids)
                    for tid, tn, ta, tc in ix.execute(
                        f"SELECT id,name,address,country FROM targets "
                        f"WHERE rid IN ({target_marks})",
                        unique_rids,
                    ):
                        ids[tid] = (tn, ta, tc)

            left = PairText.make(name, address, country)
            vectors = []
            targets = []
            for tid, (tn, ta, tc) in ids.items():
                vectors.append(pair_features(left, PairText.make(tn, ta, tc)))
                targets.append((qid, tid))
                cand_n += 1

            if targets:
                matrix = np.asarray(vectors, dtype=np.float32)
                matrix = (matrix - means) / np.where(std == 0, 1, std)
                matrix[:, 0] = 1.0
                logits = np.clip(matrix @ weights, -30, 30)
                probs = 1.0 / (1.0 + np.exp(-logits))
                feat = probs.tolist()

                r.executemany(
                    "INSERT OR IGNORE INTO candidates VALUES(?,?)", targets
                )
                ranked = sorted(
                    (
                        (float(sc), tid)
                        for (_q, tid), sc in zip(targets, feat)
                        if sc >= model["decision_policy"]["threshold"]
                    ),
                    reverse=True,
                )
                accepted = [
                    (qid, tid, sc)
                    for sc, tid in ranked[: int(model["decision_policy"]["max_predictions_per_query"])]
                ]
                r.executemany(
                    "INSERT OR IGNORE INTO pairs VALUES(?,?,?)", accepted
                )
                score_n += len(accepted)

            r.execute("INSERT OR IGNORE INTO completed VALUES(?)", (qid,))

            if rows % 100 == 0:
                r.commit()
                state["query_cursor"] = qi + 1
                state["peak_work_bytes"] = max(
                    state.get("peak_work_bytes", 0), work_bytes(a.work_dir)
                )
                atomic_json(state_path, state)
                free = get_free_disk_space(a.work_dir)
                if free < 20 * 2**30:
                    raise SystemExit(
                        f"safety stop: free disk below 20 GiB ({free / 2**30:.2f} GiB)"
                    )

            if rows % 250 == 0:
                print(
                    f"queries {rows:,}; candidates {cand_n:,}; "
                    f"RSS {rss() / 2**20:.0f} MiB; "
                    f"shard {a.query_offset}/{a.query_stride}",
                    flush=True,
                )

    ix.close()
    r.commit()

    if a.mode == "preflight":
        state["preflight_complete"] = True
    else:
        state["query_cursor"] = max(
            state.get("query_cursor", 0), last_qi + 1
        )
    atomic_json(state_path, state)

    candidate_count_total = r.execute(
        "SELECT count(*) FROM candidates"
    ).fetchone()[0]
    query_count_total = r.execute("SELECT count(*) FROM queries").fetchone()[0]

    output_sizes = {}
    output_paths = {
        "candidate_pairs.tsv": output_dir / "candidate_pairs.tsv",
        "matching_results.tsv": output_dir / "matching_results.tsv",
    }

    if not a.no_output:
        output_reserve = 2 * (
            candidate_count_total * 20 + query_count_total * 100
        )
        work_free = get_free_disk_space(a.work_dir)
        output_free = get_free_disk_space(output_dir)
        if work_free < 20 * 2**30:
            raise SystemExit(
                "safety stop before TSV generation: work directory needs 20 GiB "
                f"reserve; have {work_free / 2**30:.2f} GiB"
            )
        if output_free < 20 * 2**30 + output_reserve:
            raise SystemExit(
                "safety stop before TSV generation: need "
                f"{output_reserve / 2**30:.2f} GiB output space plus 20 GiB reserve; "
                f"have {output_free / 2**30:.2f} GiB"
            )

        for table, col, path, header in [
            (
                "candidates",
                "target_id",
                output_paths["candidate_pairs.tsv"],
                "source1_entity_id\tcandidate_entity_ids\n",
            ),
            (
                "pairs",
                "target_id",
                output_paths["matching_results.tsv"],
                "source1_entity_id\tmatched_entity_ids\n",
            ),
        ]:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf8", newline="\n") as out:
                out.write(header)
                current = None
                eid = ""
                vals = []
                cursor = r.execute(
                    f"SELECT q.qid,q.id,p.{col} "
                    f"FROM queries q LEFT JOIN {table} p ON p.qid=q.qid "
                    f"ORDER BY q.qid,p.{col}"
                )
                for qid, row_id, target_id in cursor:
                    if current is not None and qid != current:
                        out.write(eid + "\t" + ",".join(vals) + "\n")
                        vals = []
                    current = qid
                    eid = row_id
                    if target_id is not None:
                        vals.append(target_id)
                if current is not None:
                    out.write(eid + "\t" + ",".join(vals) + "\n")
                out.flush()
                os.fsync(out.fileno())
            os.replace(tmp, path)

        output_sizes = {
            p.name: p.stat().st_size
            for p in output_dir.iterdir()
            if p.is_file()
        }

    duration = time.time() - t0
    work_sizes = {
        p.name: p.stat().st_size for p in a.work_dir.iterdir() if p.is_file()
    }
    s1_rows_scanned = last_qi + 1 if last_qi >= 0 else 0
    projected_candidates = cand_n * a.target_sample_rate * (
        s1_rows_scanned / max(1, rows)
    )
    work_free = get_free_disk_space(a.work_dir)
    output_free = get_free_disk_space(output_dir) if not a.no_output else work_free

    report = {
        "mode": a.mode,
        "queries_processed_this_run": rows,
        "candidate_pairs_this_run": cand_n,
        "estimated_full_candidate_pairs_from_sample": projected_candidates,
        "threshold_matches_this_run": score_n,
        "target_sample_rate": a.target_sample_rate,
        "query_stride": a.query_stride,
        "query_offset": a.query_offset,
        "query_posting_cap_full": a.query_posting_cap,
        "block_frequency_cap_full": a.block_cap,
        "index_target_rows": sum(indexed_target_counts.values()),
        "indexed_target_rows_by_source": indexed_target_counts,
        "index_source_rows_scanned": sum(
            int(state.get(f"source{i}_rows", 0)) for i in (2, 3)
        ),
        "index_seconds": state.get("index_seconds"),
        "run_seconds": duration,
        "query_seconds": time.time() - qtime,
        "peak_rss_bytes": rss(),
        "peak_work_files_bytes": max(
            state.get("peak_work_bytes", 0), work_bytes(a.work_dir)
        ),
        "work_files_bytes": work_sizes,
        "output_dir": str(output_dir),
        "output_files": output_sizes,
        "output_generation_skipped": bool(a.no_output),
        "model_features_compatible": True,
        "restartability": "completed query IDs committed transactionally; outputs atomically regenerated",
        "duplicates": "UNIQUE/PRIMARY KEY on source and target pair",
        "throughput_queries_per_second": rows / max(1, time.time() - qtime),
        "disk_free_bytes": min(work_free, output_free),
    }
    atomic_json(a.work_dir / "preflight_report.json", report)
    r.close()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
