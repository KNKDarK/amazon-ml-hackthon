# Windows 11 runbook

The production code is cross-platform. It uses `shutil.disk_usage` and Win32
memory/locking fallbacks on Windows; GitHub Actions also runs the smoke tests on
`windows-latest`.

## Prerequisites

- Windows 11 64-bit
- Git for Windows
- Python 3.12 64-bit (`py -3.12 --version`)
- The challenge test TSVs placed under `dataset\test\`

The raw dataset is not stored in GitHub. The frozen model is tracked at
`pilot\frozen_pilot_model.json`.

## Clone and install

```powershell
git clone https://github.com/KNKDarK/amazon-ml-hackthon.git
cd amazon-ml-hackthon
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy the challenge files to:

```text
dataset\test\test_source1.tsv
dataset\test\test_source2.tsv
dataset\test\test_source3.tsv
```

## Check resources

```powershell
Get-PSDrive -Name C
Get-CimInstance Win32_OperatingSystem |
  Select-Object Caption, Version, FreePhysicalMemory
```

The recommended cap 500 run needs approximately 29 GiB peak working storage and
can take 32–43 hours. Do not launch two processes against the same work
directory.

## Run in the foreground

```powershell
New-Item -ItemType Directory -Force artifacts\full_inference_20260925_cap500, output | Out-Null
.\.venv\Scripts\python.exe -u pilot\stream_infer.py `
  --data-root dataset\test `
  --work-dir artifacts\full_inference_20260925_cap500 `
  --output-dir output `
  --model pilot\frozen_pilot_model.json `
  --mode full `
  --queries 0 `
  --target-sample-rate 1 `
  --query-stride 1 `
  --index-batch 5000 `
  --index-synchronous FULL `
  --block-cap 500 `
  --query-posting-cap 500
```

## Run detached and monitor

```powershell
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$args = @(
  '-u', 'pilot\stream_infer.py',
  '--data-root', 'dataset\test',
  '--work-dir', 'artifacts\full_inference_20260925_cap500',
  '--output-dir', 'output',
  '--model', 'pilot\frozen_pilot_model.json',
  '--mode', 'full',
  '--queries', '0',
  '--target-sample-rate', '1',
  '--query-stride', '1',
  '--index-batch', '5000',
  '--index-synchronous', 'FULL',
  '--block-cap', '500',
  '--query-posting-cap', '500'
)
$p = Start-Process `
  -FilePath $python `
  -ArgumentList $args `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput 'artifacts\full_inference_20260925_cap500\run.log' `
  -RedirectStandardError 'artifacts\full_inference_20260925_cap500\run.err.log' `
  -PassThru
$p.Id
```

```powershell
Get-Content artifacts\full_inference_20260925_cap500\run.log -Wait
```

If the process stops, run the same command again with the same `--work-dir` to
resume from the checkpoint. The `.run.lock` file prevents duplicate workers.

## Validate the final files

```powershell
.\.venv\Scripts\python.exe utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir dataset\test
```
