#!/usr/bin/env python3
"""Export intermediate matching_results.tsv and candidate_pairs.tsv from results.sqlite."""
import os
import sqlite3
from pathlib import Path

work_dir = Path("artifacts/full_inference_20260925_cap500")
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)

db_path = work_dir / "results.sqlite"
if not db_path.exists():
    raise SystemExit(f"Error: {db_path} does not exist.")

conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)

output_paths = {
    'candidate_pairs.tsv': output_dir / 'candidate_pairs.tsv',
    'matching_results.tsv': output_dir / 'matching_results.tsv',
}

print("Exporting intermediate TSVs from SQLite...")

for table, col, path, header in [
    ('candidates', 'target_id', output_paths['candidate_pairs.tsv'], 'source1_entity_id\tcandidate_entity_ids\n'),
    ('pairs', 'target_id', output_paths['matching_results.tsv'], 'source1_entity_id\tmatched_entity_ids\n')
]:
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf8') as out:
        out.write(header)
        current = None
        eid = ''
        vals = []
        cursor = conn.execute(
            f'SELECT q.qid, q.id, p.{col} FROM queries q LEFT JOIN {table} p ON p.qid=q.qid ORDER BY q.qid, p.{col}'
        )
        for qid, row_id, target_id in cursor:
            if current is not None and qid != current:
                out.write(eid + '\t' + ','.join(vals) + '\n')
                vals = []
            current = qid
            eid = row_id
            if target_id is not None:
                vals.append(target_id)
        if current is not None:
            out.write(eid + '\t' + ','.join(vals) + '\n')
            out.flush()
    os.replace(tmp, path)
    print(f"  - Generated {path.name} ({path.stat().st_size / 1024 / 1024:.2f} MB)")

conn.close()
print("Export complete!")