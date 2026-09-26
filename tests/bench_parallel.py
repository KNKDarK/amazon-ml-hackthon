#!/usr/bin/env python3
"""Measure the speedup curve of the parallel frequency pass on real data.

Runs the real ``measure_block_frequencies`` work over a slice of the production
corpus at several worker counts, reporting wall time, rows/second, and the peak
RSS of the whole process tree so the RAM budget can be checked.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pilot import run_pilot as rp  # noqa: E402

SLICE_MB = 200
CONFIGS = [1, 2, 4, 6, 8, 10, 12]


def tree_rss_bytes() -> int:
    out = subprocess.run(
        ["ps", "-eo", "rss=,pid=,ppid="], capture_output=True, text=True
    ).stdout.splitlines()
    me = str(__import__("os").getpid())
    rss = {int(p): int(r) * 1024 for r, p, _ in
           (line.split() for line in out if len(line.split()) == 3)}
    kids = set()
    for _r, pid, ppid in (line.split() for line in out if len(line.split()) == 3):
        if ppid == me:
            kids.add(int(pid))
    total = rss.get(int(me), 0)
    for pid in kids:
        total += rss.get(pid, 0)
    return total


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="bench_"))
    data_root = tmp / "dataset"
    train = data_root / "train"
    train.mkdir(parents=True, exist_ok=True)
    real = ROOT / "dataset" / "train"

    keep = SLICE_MB * 1024 * 1024
    for name in ("train_source2.tsv", "train_source3.tsv"):
        src = real / name
        with src.open("rb") as fin, (train / name).open("wb") as fout:
            data = fin.read(keep)
            cut = data.rfind(b"\n")
            fout.write(data[: cut + 1])
        print(f"{name}: {(train / name).stat().st_size / 1024**2:.0f} MiB")

    # Link the remaining inputs rather than copying.
    for name in ("train_source1.tsv", "train_ground_truth.tsv"):
        (train / name).symlink_to(real / name)

    work = tmp / "work"
    work.mkdir(parents=True, exist_ok=True)
    queries, truth, _ = rp.sample_queries(data_root, work, 10_000)
    key_index = rp.build_query_key_index(queries)
    print(f"queries={len(queries):,} keys={len(key_index):,}\n")

    print(f"{'workers':>7} {'seconds':>9} {'rows/s':>10} {'speedup':>8} {'tree RSS':>10}")
    print("-" * 50)
    baseline = None
    rows_seen = 0
    for workers in CONFIGS:
        stop = threading.Event()
        peak = [0]

        def watch() -> None:
            while not stop.wait(0.25):
                peak[0] = max(peak[0], tree_rss_bytes())

        watcher = threading.Thread(target=watch, daemon=True)
        start_available = rp.available_memory_bytes()
        budget = rp.ram_budget_bytes(start_available, 0.0)
        effective = rp.resolve_workers(workers, budget)
        watcher.start()
        began = time.perf_counter()
        counts, true_keys, stats = rp.measure_block_frequencies(
            data_root, queries, truth, key_index, rp.MemoryMonitor(), effective
        )
        elapsed = time.perf_counter() - began
        stop.set()
        watcher.join(timeout=1)
        rows_seen = sum(int(s["rows"]) for s in stats["sources"].values())
        if baseline is None:
            baseline = elapsed
        print(f"{effective:>7} {elapsed:>9.1f} {rows_seen / elapsed:>10,.0f} "
              f"{baseline / elapsed:>7.2f}x {peak[0] / 1024**3:>9.2f}G")

    print(f"\nrows scanned per configuration: {rows_seen:,}")
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
