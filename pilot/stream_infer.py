#!/usr/bin/env python3
"""Disk-backed, restartable test inference and bounded corpus preflight.

The index stores normalized blocking keys and target row numbers. Feature vectors
exist only in a bounded per-query batch; completed query results are committed to
SQLite and rendered to the requested output directory after inference.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sqlite3, sys, time
from pathlib import Path
import numpy as np
try:
    from pilot.er_common import (FEATURE_NAMES, PairText, acquire_run_lock,
        blocking_keys, canonical_target_fields, disk_free_bytes, iter_tsv,
        normalize_country, pair_features, peak_rss_bytes, stable_u64)
except ModuleNotFoundError:  # direct ``python pilot/stream_infer.py`` execution
    from er_common import (FEATURE_NAMES, PairText, acquire_run_lock,
        blocking_keys, canonical_target_fields, disk_free_bytes, iter_tsv,
        normalize_country, pair_features, peak_rss_bytes, stable_u64)

VERSION = 2
POSTING_CAP = 100
QUERY_POSTING_CAP = 100
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def atomic_json(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(obj,f,indent=2,sort_keys=True,ensure_ascii=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)

def db(path, cache=64, synchronous='FULL', journal_mode='WAL'):
    """Open SQLite with bounded cache and a portable busy timeout.

    Some local/network filesystems cannot provide SQLite WAL shared-memory
    semantics. Fall back to rollback journalling there while keeping the same
    transactional resume guarantees.
    """
    c=sqlite3.connect(path, timeout=60.0)
    c.execute('PRAGMA busy_timeout=60000')
    try:
        mode=c.execute(f'PRAGMA journal_mode={journal_mode}').fetchone()[0]
        if journal_mode == 'WAL' and str(mode).lower() != 'wal':
            c.execute('PRAGMA journal_mode=DELETE')
    except sqlite3.OperationalError:
        if journal_mode == 'WAL':
            c.execute('PRAGMA journal_mode=DELETE')
        else:
            c.close()
            raise
    c.execute(f'PRAGMA synchronous={synchronous}')
    c.execute('PRAGMA temp_store=FILE')
    c.execute(f'PRAGMA cache_size=-{cache*1024}')
    c.execute('PRAGMA wal_autocheckpoint=10000')
    return c

def rss(): return peak_rss_bytes()
def work_bytes(path): return sum(p.stat().st_size for p in path.iterdir() if p.is_file())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def portable_input_fingerprint(paths) -> dict:
    """Return logical-name/content fingerprints independent of checkout paths."""
    return {
        path.name: {'size': path.stat().st_size, 'sha256': sha256_file(path)}
        for path in sorted(paths, key=lambda item: item.name)
    }


def legacy_input_fingerprint(paths) -> dict:
    return {
        str(path): [path.stat().st_size, path.stat().st_mtime_ns]
        for path in paths
    }


def legacy_fingerprint_matches(stored, paths) -> bool:
    """Compare version-1 size/mtime values without depending on path syntax."""
    if not isinstance(stored, dict) or len(stored) != 3:
        return False
    expected = sorted((int(item[0]), int(item[1])) for item in legacy_input_fingerprint(paths).values())
    try:
        actual = sorted((int(item[0]), int(item[1])) for item in stored.values())
    except (TypeError, ValueError, IndexError):
        return False
    return actual == expected


def validate_database(path: Path, required_tables: set[str], label: str) -> None:
    if not path.is_file():
        raise SystemExit(f'checkpoint requires {label} database, but it is missing: {path}')
    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        tables = {str(row[0]) for row in rows}
        missing = sorted(required_tables - tables)
        if missing:
            raise SystemExit(
                f'{label} database is missing required table(s): {", ".join(missing)}'
            )
    finally:
        connection.close()


def canonicalize_legacy_targets(path: Path) -> int:
    """Convert version-1 raw target fields to the model's canonical form."""
    connection = db(path)
    updated = 0
    try:
        cursor = connection.execute('SELECT rid,name,address,country FROM targets')
        while True:
            rows = cursor.fetchmany(10_000)
            if not rows:
                break
            changes = []
            for rid, name, address, country in rows:
                target_name, target_address, target_country = canonical_target_fields(
                    name, address, country
                )
                changes.append(
                    (target_name, target_address, target_country, rid)
                )
            connection.executemany(
                'UPDATE targets SET name=?,address=?,country=? WHERE rid=?', changes
            )
            updated += len(changes)
        connection.commit()
        try:
            connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        except sqlite3.OperationalError:
            pass
    finally:
        connection.close()
    return updated


def parse_args(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=PROJECT_ROOT/'dataset/test')
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,default=PROJECT_ROOT/'pilot/frozen_pilot_model.json')
    p.add_argument('--queries',type=int,default=2500,help='bounded preflight query count; 0 means all S1')
    p.add_argument('--mode',choices=('preflight','full'),default='preflight')
    p.add_argument('--index-batch',type=int,default=5000)
    p.add_argument('--target-sample-rate',type=int,default=1,help='index each Nth deterministic target hash; use 20 for preflight')
    p.add_argument('--query-stride',type=int,default=1,help='process every Nth S1 input row')
    p.add_argument('--block-cap',type=int,default=POSTING_CAP)
    p.add_argument('--query-posting-cap',type=int,default=QUERY_POSTING_CAP)
    p.add_argument('--output-dir',type=Path,default=PROJECT_ROOT/'output',
                   help='directory for final matching_results.tsv and candidate_pairs.tsv (default: repository output)')
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
    if args.mode == 'full' and args.query_stride != 1:
        p.error('--query-stride must be 1 in full mode')
    return args

def main():
    a=parse_args()
    a.work_dir=a.work_dir.expanduser().resolve()
    a.data_root=a.data_root.expanduser().resolve()
    a.model=a.model.expanduser().resolve()
    a.output_dir=(a.output_dir or a.work_dir).expanduser().resolve()
    a.work_dir.mkdir(parents=True,exist_ok=True)
    output_dir=a.output_dir
    output_dir.mkdir(parents=True,exist_ok=True)
    lock_handle=(a.work_dir/'.run.lock').open('a+', encoding='utf-8')
    if not acquire_run_lock(lock_handle):
        raise SystemExit(f'an inference process already holds {a.work_dir}/.run.lock')
    output_lock_handle=None
    if output_dir != a.work_dir:
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

    state_path=a.work_dir/'checkpoint.json'
    index_path=a.work_dir/'index.sqlite'
    result_path=a.work_dir/'results.sqlite'
    profile={'target_sample_rate':a.target_sample_rate,'query_stride':a.query_stride,
             'block_cap':a.block_cap,'query_posting_cap':a.query_posting_cap}
    identity={
        'version':VERSION,
        'inputs':portable_input_fingerprint(paths.values()),
        'model_sha256':sha256_file(a.model),
        'pipeline_code_sha256':sha256_file(Path(__file__).resolve()),
        'feature_code_sha256':sha256_file(Path(__file__).with_name('er_common.py')),
        'mode':a.mode,
        'query_limit':a.queries if a.mode=='preflight' else 0,
        'profile':profile,
    }
    if state_path.exists():
        try:
            with state_path.open(encoding='utf-8') as handle:
                state=json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f'cannot read checkpoint {state_path}: {exc}') from exc
        if state.get('version')==1:
            # Version-1 checkpoints did not bind model/content identity. They
            # can be migrated safely only before query results exist; the target
            # index itself depends solely on the validated data/profile.
            legacy_ok=(legacy_fingerprint_matches(state.get('fingerprint'), paths.values())
                       and state.get('profile')==profile
                       and int(state.get('query_cursor',0))==0
                       and not state.get('preflight_complete',False)
                       and (not result_path.exists() or result_path.stat().st_size==0))
            if not legacy_ok:
                raise SystemExit(
                    'legacy checkpoint has query results or mismatched inputs/profile; '
                    'use a new work directory'
                )
            converted_targets = 0
            if index_path.is_file():
                validate_database(index_path, {'targets','postings'}, 'index')
                converted_targets = canonicalize_legacy_targets(index_path)
            state={
                'version':VERSION,
                'identity':identity,
                'index_complete':bool(state.get('index_complete',False)),
                'query_cursor':0,
                'source2_rows':int(state.get('source2_rows',0)),
                'source3_rows':int(state.get('source3_rows',0)),
                'peak_work_bytes':int(state.get('peak_work_bytes',0)),
            }
            atomic_json(state_path,state)
            print(
                f'migrated a query-free legacy index checkpoint to portable state v2; '
                f'canonicalized {converted_targets:,} target rows',
                flush=True,
            )
        elif state.get('version')!=VERSION or state.get('identity')!=identity:
            raise SystemExit(
                'checkpoint identity mismatch (data, model, feature code, mode, or profile); '
                'use a new work directory'
            )
    else:
        state={'version':VERSION,'identity':identity,'index_complete':False,'query_cursor':0}

    if state.get('index_complete'):
        validate_database(index_path, {'targets','postings','keyfreq'}, 'index')
    elif int(state.get('source2_rows',0))+int(state.get('source3_rows',0)) > 0:
        validate_database(index_path, {'targets','postings'}, 'index')
    if int(state.get('query_cursor',0)) > 0 or state.get('preflight_complete'):
        validate_database(result_path, {'queries','pairs','candidates','completed'}, 'results')
    if not state_path.exists():
        atomic_json(state_path,state)

    safety_reserve=int(a.safety_free_gib*(1024**3))
    t0=time.time(); c=db(index_path, synchronous=a.index_synchronous,
                           journal_mode='OFF' if a.index_synchronous=='OFF' else 'WAL')
    c.executescript('''CREATE TABLE IF NOT EXISTS targets(rid INTEGER PRIMARY KEY, id TEXT UNIQUE, name TEXT, address TEXT, country TEXT, source INTEGER);
    CREATE TABLE IF NOT EXISTS postings(key TEXT NOT NULL, rid INTEGER NOT NULL, PRIMARY KEY(key,rid)) WITHOUT ROWID;''')
    if not state['index_complete']:
        def flush_index_batch(src, count, target_rows, posts):
            if target_rows:
                c.executemany('INSERT OR IGNORE INTO targets(rid,id,name,address,country,source) VALUES(?,?,?,?,?,?)', target_rows)
            if posts:
                c.executemany('INSERT OR IGNORE INTO postings VALUES(?,?)', posts)
            c.commit()
            state[f'source{src}_rows']=count
            state['peak_work_bytes']=max(state.get('peak_work_bytes',0),work_bytes(a.work_dir)); atomic_json(state_path,state)
            free=disk_free_bytes(a.work_dir)
            if free<safety_reserve: raise SystemExit(f'safety stop: free disk below {a.safety_free_gib:.2f} GiB ({free/2**30:.2f} GiB)')
            if count%a.index_batch==0:
                print(f'indexed S{src} {count:,}; db={index_path.stat().st_size/2**30:.2f} GiB; rss={rss()/2**20:.0f} MiB',flush=True)
        for src in (2,3):
            path=paths[src]; done=int(state.get(f'source{src}_rows',0)); count=0
            target_rows=[]; posts=[]
            # IDs are source-prefixed and unique.  A raw row ordinal gives us
            # a deterministic RID without a per-row INSERT/SELECT round trip;
            # this preserves the postings schema while making index build
            # batchable.  The S3 offset is the completed S2 raw row count.
            source_offset=0 if src==2 else int(state.get('source2_rows',0))
            for row in iter_tsv(path):
                count+=1
                if count<=done: continue
                if stable_u64(row['entity_id'],'test-target-sample-v1') % a.target_sample_rate:
                    if count%a.index_batch==0:
                        flush_index_batch(src,count,target_rows,posts); target_rows.clear(); posts.clear()
                    continue
                rid=source_offset+count
                target_name,target_address,target_country=canonical_target_fields(
                    row['business_name'],row['business_address'],row['country']
                )
                target_rows.append((rid,row['entity_id'],target_name,target_address,target_country,src))
                posts.extend((key,rid) for key in blocking_keys(row['business_name'],row['business_address'],row['country']))
                if count%a.index_batch==0:
                    flush_index_batch(src,count,target_rows,posts); target_rows.clear(); posts.clear()
            flush_index_batch(src,count,target_rows,posts); target_rows.clear(); posts.clear()
        c.execute('CREATE TABLE IF NOT EXISTS keyfreq(key TEXT PRIMARY KEY,n INTEGER NOT NULL) WITHOUT ROWID')
        c.execute('INSERT OR REPLACE INTO keyfreq SELECT key,count(*) FROM postings GROUP BY key'); c.commit()
        state['index_complete']=True; state['index_seconds']=time.time()-t0; atomic_json(state_path,state)
    indexed_target_counts={src:c.execute('SELECT count(*) FROM targets WHERE source=?',(src,)).fetchone()[0] for src in (2,3)}
    c.close()
    r=db(result_path); r.executescript('''CREATE TABLE IF NOT EXISTS queries(qid INTEGER PRIMARY KEY,id TEXT UNIQUE,name TEXT,address TEXT,country TEXT);
    CREATE TABLE IF NOT EXISTS pairs(qid INTEGER,target_id TEXT,score REAL,PRIMARY KEY(qid,target_id)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS candidates(qid INTEGER,target_id TEXT,PRIMARY KEY(qid,target_id)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS completed(qid INTEGER PRIMARY KEY);''')
    query_limit=a.queries if a.mode=='preflight' else 0; qstart=int(state.get('query_cursor',0)); rows=0; cand_n=score_n=0; qtime=time.time()
    ix=sqlite3.connect(index_path); ix.execute('PRAGMA cache_size=-65536')
    with paths[1].open(encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f,delimiter='\t',quoting=csv.QUOTE_NONE)
        if rd.fieldnames!=['entity_id','business_name','business_address','country']: raise SystemExit('S1 header mismatch')
        for qi,row in enumerate(rd):
            if a.mode=='preflight' and state.get('preflight_complete'): break
            if qi<qstart: continue
            if a.mode=='preflight' and qi%a.query_stride: continue
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
            if keys:
                marks=','.join('?' for _ in keys)
                freq_rows=ix.execute(f'SELECT key,n FROM keyfreq WHERE key IN ({marks}) ORDER BY n,key',keys)
                used=0
                for key,freq in freq_rows:
                    if freq>posting_cap or freq>query_posting_cap-used: continue
                    raw_rids.extend(r[0] for r in ix.execute('SELECT rid FROM postings WHERE key=? ORDER BY rid',(key,)))
                    used+=freq
                if raw_rids:
                    unique_rids=list(dict.fromkeys(raw_rids)); target_marks=','.join('?' for _ in unique_rids)
                    for tid,tn,ta,tc in ix.execute(f'SELECT id,name,address,country FROM targets WHERE rid IN ({target_marks})',unique_rids):
                        ids[tid]=(tn,ta,tc)
            left=PairText.make(name,address,country); vectors=[]; targets=[]
            for tid,(tn,ta,tc) in ids.items():
                vectors.append(pair_features(left,PairText.make(tn,ta,tc)))
                targets.append((qid,tid)); cand_n+=1
            if targets:
                matrix=np.asarray(vectors,dtype=np.float32); matrix=(matrix-means)/np.where(std==0,1,std); matrix[:,0]=1.0
                logits=np.clip(matrix@weights,-30,30); probs=1.0/(1.0+np.exp(-logits)); feat=probs.tolist()
                r.executemany('INSERT OR IGNORE INTO candidates VALUES(?,?)',targets)
                ranked=sorted(
                    ((float(sc),tid) for (_q,tid),sc in zip(targets,feat)
                     if sc>=decision_policy['threshold']),
                    key=lambda item: (-item[0], item[1]),
                )
                accepted=[(qid,tid,sc) for sc,tid in ranked[:int(decision_policy['max_predictions_per_query'])]]
                r.executemany('INSERT OR IGNORE INTO pairs VALUES(?,?,?)',accepted); score_n+=len(accepted)
            r.execute('INSERT OR IGNORE INTO completed VALUES(?)',(qid,))
            if rows % 100 == 0:
                r.commit()
                state['query_cursor']=qi+1; state['peak_work_bytes']=max(state.get('peak_work_bytes',0),work_bytes(a.work_dir)); atomic_json(state_path,state)
                free=disk_free_bytes(a.work_dir)
                if free<safety_reserve: raise SystemExit(f'safety stop: free disk below {a.safety_free_gib:.2f} GiB ({free/2**30:.2f} GiB)')
            if rows%250==0: print(f'queries {rows:,}; candidates {cand_n:,}; RSS {rss()/2**20:.0f} MiB',flush=True)
    ix.close(); r.commit()
    if a.mode=='full':
        expected_s1_rows=sum(1 for _row in iter_tsv(paths[1]))
        completed_s1_rows=int(r.execute('SELECT count(*) FROM completed').fetchone()[0])
        stored_query_rows=int(r.execute('SELECT count(*) FROM queries').fetchone()[0])
        if completed_s1_rows != expected_s1_rows or stored_query_rows != expected_s1_rows:
            raise SystemExit(
                f'full inference coverage check failed: completed {completed_s1_rows:,}, '
                f'stored {stored_query_rows:,}, expected {expected_s1_rows:,} S1 rows'
            )
        orphan_matches=int(r.execute(
            'SELECT count(*) FROM pairs p LEFT JOIN candidates c '
            'ON c.qid=p.qid AND c.target_id=p.target_id '
            'WHERE c.qid IS NULL'
        ).fetchone()[0])
        if orphan_matches:
            raise SystemExit(f'internal consistency check failed: {orphan_matches:,} matches are not candidates')
    if a.mode=='preflight': state['preflight_complete']=True
    else: state['query_cursor']=max(state.get('query_cursor',0),qi+1 if 'qi' in locals() else 0)
    atomic_json(state_path,state)
    # Final TSVs are written outside the resumable work state by default when
    # --output-dir is supplied.  Each file is still generated via a temporary
    # file in its destination directory and atomically replaced.
    candidate_count_total=r.execute('SELECT count(*) FROM candidates').fetchone()[0]
    query_count_total=r.execute('SELECT count(*) FROM queries').fetchone()[0]
    output_reserve=2*(candidate_count_total*20+query_count_total*100)
    work_free=disk_free_bytes(a.work_dir)
    output_free=disk_free_bytes(output_dir)
    if work_free<safety_reserve:
        raise SystemExit(f'safety stop before TSV generation: work directory needs {a.safety_free_gib:.2f} GiB reserve; have {work_free/2**30:.2f} GiB')
    if output_free<safety_reserve+output_reserve:
        raise SystemExit(f'safety stop before TSV generation: need {output_reserve/2**30:.2f} GiB output space plus {a.safety_free_gib:.2f} GiB reserve; have {output_free/2**30:.2f} GiB')
    output_paths={
        'candidate_pairs.tsv': output_dir/'candidate_pairs.tsv',
        'matching_results.tsv': output_dir/'matching_results.tsv',
    }
    for table,col,path,header in [('candidates','target_id',output_paths['candidate_pairs.tsv'],'source1_entity_id\tcandidate_entity_ids\n'),('pairs','target_id',output_paths['matching_results.tsv'],'source1_entity_id\tmatched_entity_ids\n')]:
        tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('w',encoding='utf-8',newline='\n') as out:
            out.write(header)
            current=None; eid=''; vals=[]
            cursor=r.execute(f'SELECT q.qid,q.id,p.{col} FROM queries q LEFT JOIN {table} p ON p.qid=q.qid ORDER BY q.qid,p.{col}')
            for qid,row_id,target_id in cursor:
                if current is not None and qid!=current:
                    out.write(eid+'\t'+','.join(vals)+'\n'); vals=[]
                current=qid; eid=row_id
                if target_id is not None: vals.append(target_id)
            if current is not None: out.write(eid+'\t'+','.join(vals)+'\n')
            out.flush(); os.fsync(out.fileno())
        os.replace(tmp,path)
    duration=time.time()-t0
    work_sizes={p.name:p.stat().st_size for p in a.work_dir.iterdir() if p.is_file()}
    output_sizes={p.name:p.stat().st_size for p in output_dir.iterdir() if p.is_file()}
    s1_rows_scanned=(qi+1) if 'qi' in locals() else 0
    projected_candidates=cand_n*a.target_sample_rate*(s1_rows_scanned/max(1,rows))
    report={'mode':a.mode,'queries_processed_this_run':rows,'candidate_pairs_this_run':cand_n,'estimated_full_candidate_pairs_from_sample':projected_candidates,'threshold_matches_this_run':score_n,'cumulative_completed_queries':int(r.execute('SELECT count(*) FROM completed').fetchone()[0]),'cumulative_candidate_pairs':candidate_count_total,'cumulative_threshold_matches':int(r.execute('SELECT count(*) FROM pairs').fetchone()[0]),'target_sample_rate':a.target_sample_rate,'query_stride':a.query_stride,'query_posting_cap_full':a.query_posting_cap,'block_frequency_cap_full':a.block_cap,'index_target_rows':sum(indexed_target_counts.values()),'indexed_target_rows_by_source':indexed_target_counts,'index_source_rows_scanned':sum(int(state.get(f'source{i}_rows',0)) for i in (2,3)),'index_seconds':state.get('index_seconds'),'run_seconds':duration,'query_seconds':time.time()-qtime,'peak_rss_bytes':rss(),'peak_work_files_bytes':max(state.get('peak_work_bytes',0),work_bytes(a.work_dir)),'work_files_bytes':work_sizes,'output_dir':str(output_dir),'output_files':output_sizes,'model_features_compatible':True,'restartability':'completed query IDs committed transactionally; outputs atomically regenerated','duplicates':'UNIQUE/PRIMARY KEY on source and target pair','throughput_queries_per_second':rows/max(1,time.time()-qtime),'disk_free_bytes':min(work_free,output_free),'runtime':{'python':sys.version.split()[0],'numpy':np.__version__,'sqlite':sqlite3.sqlite_version,'platform':sys.platform},'state_identity':identity}
    atomic_json(a.work_dir/'preflight_report.json',report)
    r.close()
    lock_handle.close()
    if output_lock_handle is not None:
        output_lock_handle.close()
        (output_dir/'.output.lock').unlink(missing_ok=True)
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
