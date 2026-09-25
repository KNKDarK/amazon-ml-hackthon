# Windows 11 runbook

The production path is native Python and is tested on `windows-latest` in GitHub
Actions. It supports Windows file locking, Win32 memory checks, long local
paths, and deterministic UTF-8/LF output.

## Prerequisites

- Windows 11 64-bit
- Git for Windows
- Python 3.12 64-bit (`py -3.12 --version`)
- About 50 GiB free on the work/output volume for the measured cap-500 plan plus
  its 20 GiB safety reserve
- The challenge test TSVs under `dataset\test\`

Use a local NTFS volume. Do not run the work directory from OneDrive, Dropbox,
SMB, or another synchronized/network filesystem.

## Clone and install

```powershell
git clone https://github.com/KNKDarK/amazon-ml-hackthon.git
cd amazon-ml-hackthon
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements_lock.txt
.\.venv\Scripts\python.exe -m pip check
```

Place the challenge files at:

```text
dataset\test\test_source1.tsv
dataset\test\test_source2.tsv
dataset\test\test_source3.tsv
```

The raw data and large generated outputs are intentionally not committed.

## Verify the checkout

```powershell
.\.venv\Scripts\python.exe -m compileall -q pilot utils tests tools
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The test suite includes a tiny full run, output validation, BOM/LF checks,
Windows-compatible lock contention, canonical feature parity, and missing-state
detection.

## Check the correct volume

```powershell
New-Item -ItemType Directory -Force artifacts, output | Out-Null
$workPath = (Resolve-Path artifacts).Path
$driveLetter = (Split-Path $workPath -Qualifier).TrimEnd('\').TrimEnd(':')
Get-PSDrive -Name $driveLetter
Get-CimInstance Win32_OperatingSystem |
  Select-Object Caption, Version, FreePhysicalMemory
```

The pipeline preserves 20 GiB free and stops rather than filling the disk.

## Run in the foreground

```powershell
.\.venv\Scripts\python.exe -u pilot\stream_infer.py `
  --data-root dataset\test `
  --work-dir artifacts\full_inference_cap500 `
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

Do not use a nonzero query stride in full mode; the program rejects it and
performs an explicit full-S1 coverage assertion before writing final files.

## Run detached and monitor

```powershell
$python = (Resolve-Path .\.venv\Scripts\python.exe).Path
$arguments = @(
  '-u', 'pilot\stream_infer.py',
  '--data-root', 'dataset\test',
  '--work-dir', 'artifacts\full_inference_cap500',
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
$process = Start-Process `
  -FilePath $python `
  -ArgumentList $arguments `
  -WorkingDirectory (Get-Location) `
  -RedirectStandardOutput 'artifacts\full_inference_cap500\run.log' `
  -RedirectStandardError 'artifacts\full_inference_cap500\run.err.log' `
  -PassThru
$process.Id
```

```powershell
Get-Content artifacts\full_inference_cap500\run.log -Wait
```

If the process stops, run the identical command again with the same work
directory. `.run.lock` uses the Windows `msvcrt` byte-range lock and prevents a
second writer. The state validator also refuses to resume if the model, input
contents, feature code, mode, or profile differs.

## Validate the final files

```powershell
.\.venv\Scripts\python.exe utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir dataset\test `
  --check-ids
```

Wait for `PASS` with no warnings. Build the final archive only after validation:

```powershell
.\.venv\Scripts\python.exe tools\build_submission.py `
  --team-name TEAM_NAME `
  --check-ids
```
