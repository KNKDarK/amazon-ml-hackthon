# macOS runbook

The production Python path is tested on both GitHub `macos-latest` runners and is
supported on Apple Silicon and Intel Macs. SQLite, process locking, Unicode
normalization, and output newlines are handled by the same cross-platform code.

## Prerequisites

- macOS on Apple Silicon or Intel
- Python 3.12+ (`python3.12 --version`)
- Git
- About 50 GiB free on the work/output volume
- The challenge test TSVs under `dataset/test/`

Use a local APFS or HFS+ volume. Do not place the live work directory in iCloud
Drive, Dropbox, an SMB/NFS mount, or another synchronized/network folder.

## Clone and install

```bash
git clone https://github.com/KNKDarK/amazon-ml-hackthon.git
cd amazon-ml-hackthon
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements_lock.txt
.venv/bin/python -m pip check
```

Place the challenge files at:

```text
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

## Verify the checkout

```bash
.venv/bin/python -m compileall -q pilot utils tests tools
.venv/bin/python -m unittest discover -s tests -v
```

The suite covers a tiny end-to-end run, lock contention, canonical feature
parity, deterministic output, and unsafe-resume detection.

## Check resources

```bash
df -h .
vm_stat
```

The measured cap-500 plan requires about 29 GiB peak working storage plus a 20
GiB safety reserve. The pipeline stops before crossing that reserve.

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

## Run detached

```bash
nohup .venv/bin/python -u pilot/stream_infer.py \
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
  --query-posting-cap 500 \
  > artifacts/full_inference_cap500/run.log 2>&1 &
echo $!
```

Monitor with:

```bash
tail -f artifacts/full_inference_cap500/run.log
```

If the process stops, repeat the identical command against the same work
directory. The state and lock checks prevent unsafe concurrent or mismatched
resume. Keep SQLite sidecar files with their database if the directory is moved.

## Validate the final files

```bash
.venv/bin/python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

Build the required archive only after `PASS` with no warnings:

```bash
.venv/bin/python tools/build_submission.py \
  --team-name TEAM_NAME \
  --check-ids
```

The shell/systemd helpers under `pilot/` are Linux-only conveniences and are not
part of the macOS production path.
