#!/usr/bin/env python3
"""Launch a safe 16-way sharded inference run on Windows.

Each worker owns exactly one query shard:
    worker 0 -> qi %% 16 == 0
    worker 1 -> qi %% 16 == 1
    ...
    worker 15 -> qi %% 16 == 15

The master target index is copied once per worker so workers only read their own
SQLite index file. Worker outputs are disabled; the final submission should be
built by a streaming merge from the worker results.sqlite files.
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

NUM_WORKERS = 16
PYTHON_EXE = r"C:\Users\Aum Namaha\Documents\finalenv\Scripts\python.exe"
MASTER_DIR = Path("artifacts/full_inference_20260925_cap500")
RUN_DIR = Path("artifacts/full_inference_20260926_sharded16")
OUTPUT_DIR = Path("output")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--master-dir", type=Path, default=MASTER_DIR)
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def prepare_worker_state(worker_dir: Path, worker_id: int, workers: int, master_state: dict):
    worker_dir.mkdir(parents=True, exist_ok=True)

    worker_profile = {
        "target_sample_rate": 1,
        "query_stride": workers,
        "query_offset": worker_id,
        "block_cap": 500,
        "query_posting_cap": 500,
    }

    worker_state = {
        "version": 2,
        "fingerprint": master_state.get("fingerprint"),
        "profile": worker_profile,
        "index_complete": True,
        "query_cursor": 0,
        "source2_rows": master_state.get("source2_rows"),
        "source3_rows": master_state.get("source3_rows"),
        "index_seconds": master_state.get("index_seconds", 0),
    }

    with (worker_dir / "checkpoint.json").open("w", encoding="utf-8") as f:
        json.dump(worker_state, f, indent=2, sort_keys=True)
        f.write("\n")


def copy_master_index(master_index: Path, worker_index: Path):
    if worker_index.exists():
        if worker_index.stat().st_size != master_index.stat().st_size:
            raise SystemExit(
                f"Existing worker index has different size: {worker_index}. "
                "Use a fresh run directory."
            )
        return

    print(f"  Copying index -> {worker_index}")
    shutil.copy2(master_index, worker_index)


def main():
    a = parse_args()

    if a.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if not Path(PYTHON_EXE).exists():
        raise SystemExit(f"Python executable not found: {PYTHON_EXE}")

    master_index = a.master_dir / "index.sqlite"
    master_checkpoint = a.master_dir / "checkpoint.json"
    stream_script = Path("pilot/stream_infer.py")

    if not master_index.exists() or not master_checkpoint.exists():
        raise SystemExit(f"Missing completed master index/checkpoint in {a.master_dir}")
    if not stream_script.exists():
        raise SystemExit(f"Missing inference script: {stream_script}")

    with master_checkpoint.open("r", encoding="utf-8") as f:
        master_state = json.load(f)

    a.run_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    log_handles = []

    print(f"=== Spawning {a.workers} NON-OVERLAPPING workers ===")
    print(f"Master index : {master_index}")
    print(f"Run directory: {a.run_dir}")
    print(f"Shard rule   : qi % {a.workers} == worker_id")

    for worker_id in range(a.workers):
        worker_dir = a.run_dir / f"run_worker_{worker_id}"
        worker_dir.mkdir(parents=True, exist_ok=True)

        result_db = worker_dir / "results.sqlite"
        checkpoint = worker_dir / "checkpoint.json"
        worker_index = worker_dir / "index.sqlite"

        if a.resume and (result_db.exists() or checkpoint.exists()):
            # Keep the existing checkpoint/index/results for this worker.
            # stream_infer.py's real lock prevents a second live process.
            if checkpoint.exists():
                print(f"Worker {worker_id:02d}: resuming existing state")
            else:
                raise SystemExit(
                    f"Worker {worker_id:02d} has results.sqlite but no checkpoint.json"
                )
            if not worker_index.exists():
                copy_master_index(master_index, worker_index)
        else:
            if any(worker_dir.iterdir()):
                raise SystemExit(
                    f"Worker directory is non-empty: {worker_dir}\n"
                    "Use a fresh --run-dir or pass --resume explicitly."
                )
            copy_master_index(master_index, worker_index)
            prepare_worker_state(worker_dir, worker_id, a.workers, master_state)

        worker_output_dir = worker_dir / "worker_output"
        worker_output_dir.mkdir(parents=True, exist_ok=True)
        log_file = worker_dir / "run.log"

        cmd = [
            PYTHON_EXE,
            "-u",
            str(stream_script),
            "--data-root",
            "dataset/test",
            "--work-dir",
            str(worker_dir),
            "--output-dir",
            str(worker_output_dir),
            "--model",
            "pilot/frozen_pilot_model.json",
            "--mode",
            "full",
            "--queries",
            "0",
            "--target-sample-rate",
            "1",
            "--query-stride",
            str(a.workers),
            "--query-offset",
            str(worker_id),
            "--index-batch",
            "25000",
            "--index-synchronous",
            "OFF",
            "--block-cap",
            "500",
            "--query-posting-cap",
            "500",
            "--no-output",
        ]

        print(f"Worker {worker_id:02d} launched -> Log: {log_file}")
        log_handle = log_file.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
        processes.append((worker_id, proc))
        log_handles.append(log_handle)
        time.sleep(0.15)

    print("\nAll workers launched. Waiting for completion...\n")

    try:
        remaining = set(range(len(processes)))
        while remaining:
            for idx in list(remaining):
                worker_id, proc = processes[idx]
                rc = proc.poll()
                if rc is not None:
                    print(f"Worker {worker_id:02d} exited with code {rc}")
                    remaining.remove(idx)
            if remaining:
                time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping workers...")
        for _, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in processes:
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise SystemExit(130)
    finally:
        for handle in log_handles:
            try:
                handle.close()
            except OSError:
                pass

    failed = []
    for worker_id, proc in processes:
        if proc.returncode != 0:
            failed.append((worker_id, proc.returncode))

    if failed:
        print("\nWorkers with non-zero exit codes:")
        for worker_id, rc in failed:
            print(f"  Worker {worker_id:02d}: {rc}")
        raise SystemExit(1)

    print("\nParallel processing finished successfully.")


if __name__ == "__main__":
    main()
