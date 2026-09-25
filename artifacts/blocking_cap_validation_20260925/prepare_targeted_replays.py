#!/usr/bin/env python3
from __future__ import annotations
import json
import os
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
index_dir = root / "targeted_index"
index_path = index_dir / "index.sqlite"
meta = json.loads((index_dir / "targeted_index_meta.json").read_text(encoding="utf-8"))
if not index_path.is_file():
    raise SystemExit(f"missing targeted index: {index_path}")
input_dir = root / "input"
paths = {n: input_dir / f"test_source{n}.tsv" for n in (1, 2, 3)}
fingerprint = {str(p.relative_to(Path.cwd())): [p.stat().st_size, p.stat().st_mtime_ns] for p in paths.values()}
# The production script is launched from student_resource, so the relative
# fingerprint above matches its data-root paths.
partial = root / "partial_full_index"
if (root / "cap_100").exists() and not partial.exists():
    (root / "cap_100").rename(partial)
for cap in (100, 500, 1000):
    work = root / f"cap_{cap}"
    work.mkdir(parents=True, exist_ok=True)
    for name in ("results.sqlite", "results.sqlite-wal", "results.sqlite-shm", "checkpoint.json", "preflight_report.json"):
        path = work / name
        if path.exists() or path.is_symlink():
            path.unlink()
    link = work / "index.sqlite"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(index_path.resolve())
    state = {
        "version": 1,
        "fingerprint": fingerprint,
        "profile": {"target_sample_rate": 1, "query_stride": 1, "block_cap": cap, "query_posting_cap": cap},
        "index_complete": True,
        "query_cursor": 0,
        "source2_rows": 5_034_616,
        "source3_rows": 5_285_603,
        "index_seconds": meta["seconds"],
        "replay_index": "targeted query-key equivalent",
    }
    (work / "checkpoint.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(work)
