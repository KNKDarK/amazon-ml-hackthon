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
import sqlite3
import subprocess
import time
from pathlib import Path

NUM_WORKERS = 16
# Default to the interpreter running this script, so the launcher works on any
# platform and inside CI. Override with --python-exe for a specific venv.
PYTHON_EXE = sys.executable
MASTER_DIR = Path("artifacts/full_inference_20260925_cap500")
RUN_DIR = Path("artifacts/full_inference_20260926_sharded16_sharedindex")
OUTPUT_DIR = Path("output")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=NUM_WORKERS)
    p.add_argument("--master-dir", type=Path, default=MASTER_DIR)
    p.add_argument("--run-dir", type=Path, default=RUN_DIR)
    p.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--model", type=Path, default=MODEL_PATH)
    p.add_argument("--python-exe", type=Path, default=Path(PYTHON_EXE))
    p.add_argument(
        "--safety-free-gib",
        type=float,
        default=20.0,
        help="total free space to preserve across the whole pool; each worker is "
             "given total/workers so N workers do not each demand the full reserve",
    )
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def prepare_worker_state(
    worker_dir: Path,
    worker_id: int,
    workers: int,
    master_state: dict,
<<<<<<< HEAD
    master_index: Path,
):
    profile = {
        "target_sample_rate": 1,
=======
    data_root: Path,
    model_path: Path,
):
    worker_dir.mkdir(parents=True, exist_ok=True)

    # Build the worker checkpoint with the very identity that stream_infer.py
    # validates on startup. Seeding a weaker hand-rolled fingerprint would
    # silently drop the model/feature-code binding from the resume check.
    worker_profile = {
        "target_sample_rate": TARGET_SAMPLE_RATE,
>>>>>>> 5baaf91 (Make the sharded launcher portable and safe to actually run)
        "query_stride": workers,
        "query_offset": worker_id,
        "block_cap": 500,
        "query_posting_cap": 500,
        "index_read_only": True,
        "index_path": str(master_index.resolve()),
    }
<<<<<<< HEAD
=======
    identity = build_identity(
        {n: data_root / f"test_source{n}.tsv" for n in (1, 2, 3)},
        model_path,
        "full",
        0,
        worker_profile,
    )
>>>>>>> 5baaf91 (Make the sharded launcher portable and safe to actually run)

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


<<<<<<< HEAD
=======
def verify_index_copy(master_index: Path, worker_index: Path) -> None:
    """Fail loudly if the copied index is not a faithful, self-contained copy.

    Only ``index.sqlite`` is copied, never the ``-wal``/``-shm`` sidecars. If the
    master still had uncheckpointed WAL pages, the copy would open fine but hold
    fewer rows, and every worker would silently return a subset of the true
    candidates. Comparing row counts against the master turns that into a loud
    failure at launch instead of a quietly wrong submission.
    """

    def counts(path: Path) -> tuple[int, int]:
        connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        try:
            check = connection.execute("PRAGMA quick_check").fetchone()[0]
            if str(check).lower() != "ok":
                raise SystemExit(f"{path} failed quick_check: {check}")
            targets = connection.execute("SELECT count(*) FROM targets").fetchone()[0]
            postings = connection.execute("SELECT count(*) FROM postings").fetchone()[0]
            return int(targets), int(postings)
        except sqlite3.Error as exc:
            raise SystemExit(f"{path} is not a usable index copy: {exc}") from exc
        finally:
            connection.close()

    expected = counts(master_index)
    actual = counts(worker_index)
    if actual != expected:
        raise SystemExit(
            f"Worker index {worker_index} does not match the master: "
            f"targets/postings {actual} != {expected}. The master index likely has "
            f"uncheckpointed WAL pages; checkpoint it (or rerun the master to "
            f"completion) before sharding."
        )


def copy_master_index(master_index: Path, worker_index: Path):
    if worker_index.exists():
        if worker_index.stat().st_size != master_index.stat().st_size:
            raise SystemExit(
                f"Existing worker index has different size: {worker_index}. "
                "Use a fresh run directory."
            )
        verify_index_copy(master_index, worker_index)
        return

    print(f"  Copying index -> {worker_index}")
    shutil.copy2(master_index, worker_index)
    verify_index_copy(master_index, worker_index)


>>>>>>> 5baaf91 (Make the sharded launcher portable and safe to actually run)
def main():
    a = parse_args()

    if a.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if a.safety_free_gib < 0:
        raise SystemExit("--safety-free-gib must be non-negative")
    if not a.python_exe.exists():
        raise SystemExit(f"Python executable not found: {a.python_exe}")

    # stream_infer enforces its own free-space floor per process. Left at the
    # default, N workers would each demand the full reserve, so 16 workers would
    # need 320 GiB of headroom to start. The reserve is a property of the
    # filesystem, not of each writer, so split the documented total across the
    # pool and keep the aggregate guarantee intact.
    per_worker_safety_gib = a.safety_free_gib / a.workers

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
<<<<<<< HEAD
    print("Index copies : 0")
=======
    print(f"Safety floor : {per_worker_safety_gib:.3f} GiB per worker "
          f"({a.safety_free_gib:.1f} GiB preserved across the pool)")
>>>>>>> 5baaf91 (Make the sharded launcher portable and safe to actually run)

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
<<<<<<< HEAD
=======
            if not worker_index.exists():
                copy_master_index(master_index, worker_index)
        else:
            if any(worker_dir.iterdir()):
                raise SystemExit(
                    f"Worker directory is non-empty: {worker_dir}\n"
                    "Use a fresh --run-dir or pass --resume explicitly."
                )
            copy_master_index(master_index, worker_index)
            prepare_worker_state(
                worker_dir, worker_id, a.workers, master_state, a.data_root, a.model
            )
>>>>>>> 5baaf91 (Make the sharded launcher portable and safe to actually run)

            worker_output_dir = worker_dir / "worker_output"
            worker_output_dir.mkdir(parents=True, exist_ok=True)
            log_file = worker_dir / "run.log"

<<<<<<< HEAD
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
=======
        cmd = [
            str(a.python_exe),
            "-u",
            str(stream_script),
            "--data-root",
            str(a.data_root),
            "--work-dir",
            str(worker_dir),
            "--output-dir",
            str(worker_output_dir),
            "--model",
            str(a.model),
            "--mode",
            "full",
            "--queries",
            "0",
            "--target-sample-rate",
            str(TARGET_SAMPLE_RATE),
            "--query-stride",
            str(a.workers),
            "--query-offset",
            str(worker_id),
            "--index-batch",
            "25000",
            "--index-synchronous",
            "OFF",
            "--block-cap",
            str(BLOCK_CAP),
            "--query-posting-cap",
            str(QUERY_POSTING_CAP),
            "--safety-free-gib",
            str(per_worker_safety_gib),
            "--no-output",
        ]
>>>>>>> 5baaf91 (Make the sharded launcher portable and safe to actually run)

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
