# Linux runbook

The production pipeline is portable across mainstream Linux distributions on
local filesystems. It uses Python and NumPy rather than GNU-specific runtime
commands. The optional shell/systemd helpers are not required.

Supported baselines include Debian/Ubuntu, Fedora/RHEL, Arch, openSUSE, and
Alpine Linux, provided Python 3.12+ and a local filesystem with reliable SQLite
locking are available.

## Prerequisites

- Git
- Python 3.12 or newer
- A local ext4, XFS, Btrfs, or equivalent work filesystem
- The challenge test TSVs under `dataset/test/`

Do not place a live work directory on NFS, SMB, a network-mounted folder, or an
actively synchronized cloud directory. SQLite WAL mode requires correct local
file locking.

## Clone and install

```bash
git clone https://github.com/KNKDarK/amazon-ml-hackthon.git
cd amazon-ml-hackthon
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements_lock.txt
.venv/bin/python -m pip check
```

A virtual environment avoids PEP 668 externally-managed-environment errors on
Debian, Ubuntu, Fedora, and similar distributions.

## Verify the checkout

```bash
.venv/bin/python -m compileall -q pilot utils tests tools
.venv/bin/python -m unittest discover -s tests -v
```

This includes a tiny end-to-end build, scoring, output, validation, lock
contention, feature-parity, and unsafe-resume test.

## Resource check

```bash
df -h .
free -h
```

The measured cap-500 plan requires about 29 GiB peak working storage plus a 20
GiB safety reserve. The pipeline stops before crossing that reserve. Keep the
dataset, work directory, and output directory on a filesystem with sufficient
free space.

## Run in the foreground

```bash
mkdir -p artifacts/full_inference_cap500 output
.venv/bin/python -u pilot/stream_infer.py \
  --data-root dataset/test \
  --work-dir artifacts/full_inference_cap500 \
  --output-dir output \
  --model pilot/frozen_pilot_model.json \
  --mode full \
  --queries 0 \
  --target-sample-rate 1 \
  --query-stride 1 \
  --index-batch 5000 \
  --index-synchronous FULL \
  --block-cap 500 \
  --query-posting-cap 500
```

The same command works in a foreground terminal, an SSH session with
`nohup`, a process supervisor, or a systemd service. Only the optional launcher
under `pilot/systemd/` is Linux-specific.

## Run the training pilot

```bash
mkdir -p artifacts/pilot_10k_supervised
WORKERS=6 RAM_BUDGET_GIB=0 \
  pilot/launch_supervised_pilot.sh
```

Or invoke the pipeline directly:

```bash
python -u pilot/run_pilot.py \
  --data-root dataset \
  --work-dir artifacts/pilot_10k_supervised \
  --sample-size 10000 \
  --selection-cap 500 \
  --negative-per-query 30 \
  --workers 6 \
  --ram-budget-gib 0
```

### Workers and the RAM budget

The two phases that dominate the run — `measure_block_frequencies` and
`persist_candidates` — each stream the full `train_source2.tsv` +
`train_source3.tsv` pair and call `blocking_keys()` on every row. That work is
pure Python, so it is **GIL-bound and does not benefit from OS threads**;
`--workers` therefore starts worker *processes*. Each worker reads its own byte
range of the TSV, so both the parsing and the key generation scale with the
pool.

- `--workers N` requests N processes. The value is clamped to the logical CPU
  count and to the RAM ceiling, and the resolved count is logged before any
  heavy allocation. `--workers 1` forces the original serial path, which runs
  in-process with no pool and no pickling.
- **The default is 6, this machine's physical core count.** It is deliberately
  not the 12 logical CPUs: SMT siblings share one core's execution units and add
  no throughput to GIL-bound pure Python, only contention. Measured on a 200 MiB
  corpus slice (`python tests/bench_parallel.py`, 4,358,974 rows scanned):

  | workers | seconds | rows/s | speedup |
  |--------:|--------:|-------:|--------:|
  | 1 | 706.9 | 6,167 | 1.00x |
  | 2 | 353.6 | 12,327 | 2.00x |
  | 4 | 203.9 | 21,375 | 3.47x |
  | 6 | 163.5 | 26,656 | 4.32x |
  | 8 | 164.9 | 26,435 | 4.29x |
  | 10 | 162.9 | 26,764 | 4.34x |
  | 11 | 152.3 | 28,612 | 4.64x |

  Scaling is near-linear to 2 workers, reaches the knee at 6, and is flat from
  8 onward. Tree RSS stays at 0.44 GiB throughout, so this ceiling is core
  count and not memory. Raise `WORKERS` only on a machine with more *physical*
  cores.
- `--ram-budget-gib 0` (the default) auto-sizes to **60% of `MemAvailable`** at
  startup, so the pool shrinks rather than driving the machine into swap. Pass a
  positive value to pin a hard ceiling. The budget also sizes the SQLite page
  caches, which is why the run now uses far more of the 14 GiB than the previous
  370 MiB peak RSS.
- Native BLAS/OpenMP runtimes are pinned to one thread per worker, before NumPy
  is imported. Six workers each opening a 12-thread math pool would oversubscribe
  the CPU. `pilot/thread_env.py` sets `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
  `MKL_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`,
  `BLIS_NUM_THREADS`, and `GOTO_NUM_THREADS`, leaving any value you already
  exported untouched.

Results are **bit-identical to the serial implementation**: chunks are merged in
ascending file order, all recorded aggregates are sums or unions, and candidate
rows land in a `WITHOUT ROWID` table whose conflict handler ORs the block mask.
`tests/check_parallel_equivalence.py` asserts this on a real corpus subset, and
`tests/bench_parallel.py` measures the speedup curve.

Both checks refuse to report `PASS` unless the corpus subset really splits into
several byte ranges. The production floor is 4 MiB and a test subset is smaller
than that, so `plan_byte_ranges` would otherwise return a *single* range for any
worker count — the "parallel" run would execute as one in-process task and the
comparison would silently degrade to serial-versus-serial. The checks lower the
floor through `PARALLEL_SCAN_MIN_CHUNK_BYTES` (a diagnostic override, unset in
normal runs) and assert the resulting range count first.

Assigning each record to the range holding its **first byte** is what makes the
split lossless. Two boundary cases matter, and both are covered by
`tests/test_parallel_scan.py`:

- A record starting before a range's `end` but extending past it belongs to that
  range; the next range discards only its tail. Discarding it from both sides
  loses one row per boundary.
- A range `start` may already sit on a record boundary, so the leading fragment
  is skipped only when the preceding byte is not a newline.

## Resume after interruption

Repeat the identical command with the same `--work-dir`, `--model`, and blocking
options. The process lock prevents a second writer. The checkpoint validates
content hashes and state before resuming. Do not edit or combine work
directories from different models or profiles.

If the directory is moved, make sure `index.sqlite`, `index.sqlite-wal`,
`index.sqlite-shm`, `results.sqlite`, `results.sqlite-wal`,
`results.sqlite-shm`, and `checkpoint.json` remain together. Prefer simply
rerunning the original command on the original local path.

## Validate final output

```bash
.venv/bin/python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

Do not package or upload until the validator prints `PASS` with no warnings.

## Optional systemd pilot supervision

`pilot/systemd/` contains a training-pilot helper template. Before installing
it, copy the unit to `~/.config/systemd/user/`, adjust its repository path and
Python environment, create its log directory, then run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now student-resource-pilot-10k.service
journalctl --user -u student-resource-pilot-10k.service -f
```

The systemd files are not used by the full production inference command.
