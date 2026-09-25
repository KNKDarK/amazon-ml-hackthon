# macOS runbook

The production Python pipeline is supported on macOS as well as Windows 11 and
Linux. GitHub Actions runs the smoke/schema checks on `macos-latest`.

## Prerequisites

- macOS on Apple Silicon or Intel
- Python 3.12 (`python3.12 --version`)
- Git
- The challenge test TSVs placed under `dataset/test/`

The raw dataset is not stored in GitHub. The frozen model is tracked at
`pilot/frozen_pilot_model.json`.

## Clone and install

```bash
git clone https://github.com/KNKDarK/amazon-ml-hackthon.git
cd amazon-ml-hackthon
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Copy the challenge files to:

```text
dataset/test/test_source1.tsv
dataset/test/test_source2.tsv
dataset/test/test_source3.tsv
```

## Check resources

```bash
df -h .
vm_stat
```

The recommended cap 500 run needs approximately 29 GiB peak working storage and
can take 32–43 hours. Use one work directory per concurrent run.

## Run in the foreground

```bash
mkdir -p artifacts/full_inference_20260925_cap500 output
.venv/bin/python -u pilot/stream_infer.py \
  --data-root dataset/test \
  --work-dir artifacts/full_inference_20260925_cap500 \
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
mkdir -p artifacts/full_inference_20260925_cap500 output
nohup .venv/bin/python -u pilot/stream_infer.py \
  --data-root dataset/test \
  --work-dir artifacts/full_inference_20260925_cap500 \
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
  > artifacts/full_inference_20260925_cap500/run.log 2>&1 &
echo $!
```

Monitor with:

```bash
tail -f artifacts/full_inference_20260925_cap500/run.log
```

If the process stops, rerun the same command with the same `--work-dir` to
resume from the checkpoint. The `.run.lock` file prevents duplicate workers.

## Validate the final files

```bash
.venv/bin/python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The Linux `systemd` files and `launch_supervised_pilot.sh` are optional Linux
supervision helpers. They are not required on macOS or Windows; invoke the
Python command directly as shown above.
