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
