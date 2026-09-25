#!/usr/bin/env python3
"""Disk-backed, restartable test inference and bounded corpus preflight.

The index stores normalized blocking keys and target row numbers. Feature vectors
exist only in a bounded per-query batch; completed query results are committed to
SQLite and rendered to the requested output directory after inference.
"""
from __future__ import annotations
import argparse, csv, fcntl, json, os, sqlite3, time, resource
from pathlib import Path
import numpy as np
from er_common import (FEATURE_NAMES, PairText, blocking_keys,
    iter_tsv, normalize_country, pair_features, stable_u64)

VERSION = 1
POSTING_CAP = 100
QUERY_POSTING_CAP = 100

def atomic_json(path, obj):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w') as f: json.dump(obj,f,indent=2,sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)

def db(path, cache=64, synchronous='FULL', journal_mode='WAL'):
    c=sqlite3.connect(path); c.execute(f'PRAGMA journal_mode={journal_mode}'); c.execute(f'PRAGMA synchronous={synchronous}')
    c.execute('PRAGMA temp_store=FILE'); c.execute(f'PRAGMA cache_size=-{cache*1024}')
    c.execute('PRAGMA wal_autocheckpoint=10000')
    return c

def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
def work_bytes(path): return sum(p.stat().st_size for p in path.iterdir() if p.is_file())

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--data-root',type=Path,default=Path('dataset/test'))
    p.add_argument('--work-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,default=Path('artifacts/pilot_10k_restart_20260925/model.json'))
    p.add_argument('--queries',type=int,default=2500,help='bounded preflight query count; 0 means all S1')
    p.add_argument('--mode',choices=('preflight','full'),default='preflight')
    p.add_argument('--index-batch',type=int,default=5000)
    p.add_argument('--target-sample-rate',type=int,default=1,help='index each Nth deterministic target hash; use 20 for preflight')
    p.add_argument('--query-stride',type=int,default=1,help='process every Nth S1 input row')
    p.add_argument('--block-cap',type=int,default=POSTING_CAP)
    p.add_argument('--query-posting-cap',type=int,default=QUERY_POSTING_CAP)
    p.add_argument('--output-dir',type=Path,default=Path('output'),
                   help='directory for final matching_results.tsv and candidate_pairs.tsv (default: output)')
    p.add_argument('--index-synchronous',choices=('FULL','NORMAL','OFF'),default='FULL',
                   help='SQLite synchronous mode while building the target index')
    return p.parse_args()

def main():
    a=parse_args(); a.work_dir.mkdir(parents=True,exist_ok=True)
    output_dir=a.output_dir or a.work_dir
    output_dir.mkdir(parents=True,exist_ok=True)
    lock_handle=(a.work_dir/'.run.lock').open('a+')
    try: fcntl.flock(lock_handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit(f'an inference process already holds {a.work_dir}/.run.lock')
    lock_handle.seek(0); lock_handle.truncate(); lock_handle.write(f'{os.getpid()}\n'); lock_handle.flush(); os.fsync(lock_handle.fileno())
    paths={n:a.data_root/f'test_source{n}.tsv' for n in (1,2,3)}
    fingerprint={str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in paths.values()}
    state_path=a.work_dir/'checkpoint.json'; index_path=a.work_dir/'index.sqlite'; result_path=a.work_dir/'results.sqlite'
    profile={'target_sample_rate':a.target_sample_rate,'query_stride':a.query_stride,
             'block_cap':a.block_cap,'query_posting_cap':a.query_posting_cap}
    state=json.load(state_path.open()) if state_path.exists() else {'version':VERSION,'fingerprint':fingerprint,'profile':profile,'index_complete':False,'query_cursor':0}
    if state.get('version')!=VERSION or state.get('fingerprint')!=fingerprint or state.get('profile')!=profile: raise SystemExit('checkpoint/data/profile version mismatch')
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
            free=os.statvfs(a.work_dir).f_bavail*os.statvfs(a.work_dir).f_frsize
            if free<20*2**30: raise SystemExit(f'safety stop: free disk below 20 GiB ({free/2**30:.2f} GiB)')
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
                target_rows.append((rid,row['entity_id'],row['business_name'],row['business_address'],normalize_country(row['country']),src))
                posts.extend((key,rid) for key in blocking_keys(row['business_name'],row['business_address'],row['country']))
                if count%a.index_batch==0:
                    flush_index_batch(src,count,target_rows,posts); target_rows.clear(); posts.clear()
            flush_index_batch(src,count,target_rows,posts); target_rows.clear(); posts.clear()
        c.execute('CREATE TABLE IF NOT EXISTS keyfreq(key TEXT PRIMARY KEY,n INTEGER NOT NULL) WITHOUT ROWID')
        c.execute('INSERT OR REPLACE INTO keyfreq SELECT key,count(*) FROM postings GROUP BY key'); c.commit()
        state['index_complete']=True; state['index_seconds']=time.time()-t0; atomic_json(state_path,state)
    indexed_target_counts={src:c.execute('SELECT count(*) FROM targets WHERE source=?',(src,)).fetchone()[0] for src in (2,3)}
    c.close()
    model=json.load(a.model.open()); means=np.asarray(model['feature_mean'],dtype=np.float32); std=np.asarray(model['feature_std'],dtype=np.float32); weights=np.asarray(model['weights'],dtype=np.float32)
    if model['feature_names']!=FEATURE_NAMES or len(weights)!=len(FEATURE_NAMES): raise SystemExit('model feature schema incompatibility')
    r=db(result_path); r.executescript('''CREATE TABLE IF NOT EXISTS queries(qid INTEGER PRIMARY KEY,id TEXT UNIQUE,name TEXT,address TEXT,country TEXT);
    CREATE TABLE IF NOT EXISTS pairs(qid INTEGER,target_id TEXT,score REAL,PRIMARY KEY(qid,target_id)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS candidates(qid INTEGER,target_id TEXT,PRIMARY KEY(qid,target_id)) WITHOUT ROWID;
    CREATE TABLE IF NOT EXISTS completed(qid INTEGER PRIMARY KEY);''')
    query_limit=a.queries if a.mode=='preflight' else 0; qstart=int(state.get('query_cursor',0)); rows=0; cand_n=score_n=0; qtime=time.time()
    ix=sqlite3.connect(index_path); ix.execute('PRAGMA cache_size=-65536')
    with paths[1].open(encoding='utf-8',newline='') as f:
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
                ranked=sorted(((float(sc),tid) for (_q,tid),sc in zip(targets,feat) if sc>=model['decision_policy']['threshold']),reverse=True)
                accepted=[(qid,tid,sc) for sc,tid in ranked[:int(model['decision_policy']['max_predictions_per_query'])]]
                r.executemany('INSERT OR IGNORE INTO pairs VALUES(?,?,?)',accepted); score_n+=len(accepted)
            r.execute('INSERT OR IGNORE INTO completed VALUES(?)',(qid,))
            if rows % 100 == 0:
                r.commit()
                state['query_cursor']=qi+1; state['peak_work_bytes']=max(state.get('peak_work_bytes',0),work_bytes(a.work_dir)); atomic_json(state_path,state)
                free=os.statvfs(a.work_dir).f_bavail*os.statvfs(a.work_dir).f_frsize
                if free<20*2**30: raise SystemExit(f'safety stop: free disk below 20 GiB ({free/2**30:.2f} GiB)')
            if rows%250==0: print(f'queries {rows:,}; candidates {cand_n:,}; RSS {rss()/2**20:.0f} MiB',flush=True)
    ix.close(); r.commit()
    if a.mode=='preflight': state['preflight_complete']=True
    else: state['query_cursor']=max(state.get('query_cursor',0),qi+1 if 'qi' in locals() else 0)
    atomic_json(state_path,state)
    # Final TSVs are written outside the resumable work state by default when
    # --output-dir is supplied.  Each file is still generated via a temporary
    # file in its destination directory and atomically replaced.
    candidate_count_total=r.execute('SELECT count(*) FROM candidates').fetchone()[0]
    query_count_total=r.execute('SELECT count(*) FROM queries').fetchone()[0]
    output_reserve=2*(candidate_count_total*20+query_count_total*100)
    work_free=os.statvfs(a.work_dir).f_bavail*os.statvfs(a.work_dir).f_frsize
    output_free=os.statvfs(output_dir).f_bavail*os.statvfs(output_dir).f_frsize
    if work_free<20*2**30:
        raise SystemExit(f'safety stop before TSV generation: work directory needs 20 GiB reserve; have {work_free/2**30:.2f} GiB')
    if output_free<20*2**30+output_reserve:
        raise SystemExit(f'safety stop before TSV generation: need {output_reserve/2**30:.2f} GiB output space plus 20 GiB reserve; have {output_free/2**30:.2f} GiB')
    output_paths={
        'candidate_pairs.tsv': output_dir/'candidate_pairs.tsv',
        'matching_results.tsv': output_dir/'matching_results.tsv',
    }
    for table,col,path,header in [('candidates','target_id',output_paths['candidate_pairs.tsv'],'source1_entity_id\tcandidate_entity_ids\n'),('pairs','target_id',output_paths['matching_results.tsv'],'source1_entity_id\tmatched_entity_ids\n')]:
        tmp=path.with_suffix(path.suffix+'.tmp')
        with tmp.open('w',encoding='utf8') as out:
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
    report={'mode':a.mode,'queries_processed_this_run':rows,'candidate_pairs_this_run':cand_n,'estimated_full_candidate_pairs_from_sample':projected_candidates,'threshold_matches_this_run':score_n,'target_sample_rate':a.target_sample_rate,'query_stride':a.query_stride,'query_posting_cap_full':a.query_posting_cap,'block_frequency_cap_full':a.block_cap,'index_target_rows':sum(indexed_target_counts.values()),'indexed_target_rows_by_source':indexed_target_counts,'index_source_rows_scanned':sum(int(state.get(f'source{i}_rows',0)) for i in (2,3)),'index_seconds':state.get('index_seconds'),'run_seconds':duration,'query_seconds':time.time()-qtime,'peak_rss_bytes':rss(),'peak_work_files_bytes':max(state.get('peak_work_bytes',0),work_bytes(a.work_dir)),'work_files_bytes':work_sizes,'output_dir':str(output_dir),'output_files':output_sizes,'model_features_compatible':True,'restartability':'completed query IDs committed transactionally; outputs atomically regenerated','duplicates':'UNIQUE/PRIMARY KEY on source and target pair','throughput_queries_per_second':rows/max(1,time.time()-qtime),'disk_free_bytes':min(work_free,output_free)}
    atomic_json(a.work_dir/'preflight_report.json',report); r.close(); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
