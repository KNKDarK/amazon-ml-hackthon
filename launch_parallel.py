#!/usr/bin/env python3
"""Launch 16 non-overlapping S1 shards against one shared read-only SQLite index.

Important design points for Windows:
- Each worker gets a disjoint S1 shard: qi % workers == worker_id.
- The completed master index is opened read-only by every worker.
- No per-worker index.sqlite copies are created, so disk usage is not multiplied by 16.
- Each worker has its own results.sqlite/checkpoint.json/run.log.
- Worker TSV generation is disabled; use the streaming merge after inference.
"""

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

NUM_WORKERS = 16
PYTHON_EXE = r"C:\Users\Aum Namaha\Documents\finalenv\Scripts\python.exe"
MASTER_DIR = Path("artifacts/full_inference_20260925_cap500")
RUN_DIR = Path("artifacts/full_inference_20260926_sharded16_sharedindex")
OUTPUT_DIR = Path("output")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--master-dir", type=Path, default=MASTER_DIR)
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def prepare_worker_state(
    worker_dir: Path,
    worker_id: int,
    workers: int,
    master_state: dict,
    master_index: Path,
):
    profile = {
        "target_sample_rate": 1,
        "query_stride": workers,
        "query_offset": worker_id,
        "block_cap": 500,
        "query_posting_cap": 500,
        "index_read_only": True,
        "index_path": str(master_index.resolve()),
    }

    state = {
        "version": 3,
        "fingerprint": master_state.get("fingerprint"),
        "profile": profile,
        "index_complete": True,
        "query_cursor": 0,
        "source2_rows": master_state.get("source2_rows"),
        "source3_rows": master_state.get("source3_rows"),
        "index_seconds": master_state.get("index_seconds", 0),
    }

    with (worker_dir / "checkpoint.json").open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.write("\n")


def main():
    a = parse_args()

    if a.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if not Path(PYTHON_EXE).exists():
        raise SystemExit(f"Python executable not found: {PYTHON_EXE}")

    master_index = (a.master_dir / "index.sqlite").resolve()
    master_checkpoint = (a.master_dir / "checkpoint.json").resolve()
    stream_script = Path("pilot/stream_infer.py").resolve()

    if not master_index.exists() or not master_checkpoint.exists():
        raise SystemExit(f"Missing master index/checkpoint in {a.master_dir}")
    if not stream_script.exists():
        raise SystemExit(f"Missing inference script: {stream_script}")

    # The run uses the master index in read-only mode. It should already be complete
    # and must not be deleted/rebuilt while workers are running.
    with master_checkpoint.open("r", encoding="utf-8") as f:
        master_state = json.load(f)
    if not master_state.get("index_complete"):
        raise SystemExit("Master checkpoint does not say index_complete=true")

    # Warn early if the disk is already extremely low. The workers themselves also
    # enforce a 20 GiB safety reserve during inference.
    disk = shutil.disk_usage(a.run_dir.resolve().parent)
    print(f"Free disk before launch: {disk.free / 2**30:.2f} GiB")
    if disk.free < 20 * 2**30:
        raise SystemExit("Less than 20 GiB free; free disk space before launching inference.")

    a.run_dir.mkdir(parents=True, exist_ok=True)

    processes = []
    log_handles = []

    print(f"=== Spawning {a.workers} NON-OVERLAPPING workers ===")
    print(f"Master index : {master_index} (SHARED READ-ONLY)")
    print(f"Run directory: {a.run_dir.resolve()}")
    print(f"Shard rule   : qi % {a.workers} == worker_id")
    print("Index copies : 0")

    try:
        for worker_id in range(a.workers):
            worker_dir = a.run_dir / f"run_worker_{worker_id}"
            worker_dir.mkdir(parents=True, exist_ok=True)

            result_db = worker_dir / "results.sqlite"
            checkpoint = worker_dir / "checkpoint.json"

            if a.resume and (result_db.exists() or checkpoint.exists()):
                if not checkpoint.exists():
                    raise SystemExit(
                        f"Worker {worker_id:02d} has results.sqlite but no checkpoint.json"
                    )
                print(f"Worker {worker_id:02d}: resuming existing state")
            else:
                if any(worker_dir.iterdir()):
                    raise SystemExit(
                        f"Worker directory is non-empty: {worker_dir}\n"
                        "Use a fresh --run-dir or pass --resume explicitly."
                    )
                prepare_worker_state(
                    worker_dir,
                    worker_id,
                    a.workers,
                    master_state,
                    master_index,
                )

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
                "FULL",
                "--index-path",
                str(master_index),
                "--index-read-only",
                "--block-cap",
                "500",
                "--query-posting-cap",
                "500",
                "--no-output",
            ]

            print(f"Worker {worker_id:02d} launched -> Log: {log_file}")
            log_handle = log_file.open("a", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT)
            processes.append((worker_id, proc))
            log_handles.append(log_handle)
            time.sleep(0.15)

    except Exception:
        for _, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in processes:
            try:
                proc.wait(timeout=10)
            except Exception:
                if proc.poll() is None:
                    proc.kill()
        raise
    finally:
        # Keep handles open while children run; close only after wait below.
        pass

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

    failed = [(worker_id, proc.returncode) for worker_id, proc in processes if proc.returncode != 0]
    if failed:
        print("\nWorkers with non-zero exit codes:")
        for worker_id, rc in failed:
            print(f"  Worker {worker_id:02d}: {rc}")
        raise SystemExit(1)

    print("\nParallel processing finished successfully.")


if __name__ == "__main__":
    main()
