#!/usr/bin/env python3
"""Generate a portable reproducibility manifest without host-specific paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]

STATIC_FILES = (
    ".github/workflows/ci.yml",
    ".gitattributes",
    "CONTRIBUTING.md",
    "Documentation_template.md",
    "LICENSE",
    "LINUX_RUNBOOK.md",
    "MACOS_RUNBOOK.md",
    "MODEL_LICENSE.md",
    "README.md",
    "SECURITY.md",
    "WINDOWS_RUNBOOK.md",
    "error_analysis.md",
    "pilot/DATASET_INVENTORY.md",
    "pilot/FULL_SCALE_ARCHITECTURE_PROPOSAL.md",
    "pilot/README.md",
    "pilot/analyze_cap_validation.py",
    "pilot/er_common.py",
    "pilot/frozen_pilot_model.json",
    "pilot/run_pilot.py",
    "pilot/stream_infer.py",
    "pilot/verify_artifacts.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements_lock.txt",
    "tests/test_pipeline_smoke.py",
    "tools/build_submission.py",
    "utils/validate_submission.py",
)

DATASET_FILES = (
    "dataset/train/train_source1.tsv",
    "dataset/train/train_source2.tsv",
    "dataset/train/train_source3.tsv",
    "dataset/train/train_ground_truth.tsv",
    "dataset/test/test_source1.tsv",
    "dataset/test/test_source2.tsv",
    "dataset/test/test_source3.tsv",
)

PILOT_FILES = (
    "artifacts/pilot_10k_cap500_clean_20260925/REPORT.md",
    "artifacts/pilot_10k_cap500_clean_20260925/blocking_stats.json",
    "artifacts/pilot_10k_cap500_clean_20260925/candidate_stats.json",
    "artifacts/pilot_10k_cap500_clean_20260925/evaluation.json",
    "artifacts/pilot_10k_cap500_clean_20260925/model.json",
    "artifacts/pilot_10k_cap500_clean_20260925/pilot_report.json",
    "artifacts/pilot_10k_cap500_clean_20260925/production_score_parity.json",
    "artifacts/blocking_cap_validation_20260925/REPORT.md",
    "artifacts/blocking_cap_validation_20260925/cap_validation.json",
)

OPTIONAL_OUTPUTS = (
    "output/matching_results.tsv",
    "output/candidate_pairs.tsv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(relative: str) -> dict[str, object]:
    path = ROOT / relative
    return {"bytes": path.stat().st_size, "sha256": sha256_file(path)}


def git(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def available_files(paths: Iterable[str]) -> dict[str, dict[str, object]]:
    return {relative: record(relative) for relative in paths if (ROOT / relative).is_file()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "baseline_manifest.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model = json.loads((ROOT / "pilot/frozen_pilot_model.json").read_text(encoding="utf-8"))
    evaluation = json.loads(
        (ROOT / "artifacts/pilot_10k_cap500_clean_20260925/evaluation.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "schema_version": 2,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": "amazon-ml-hackthon",
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "remote": git("remote", "get-url", "origin"),
            "working_tree_clean": not bool(git("status", "--porcelain")),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": __import__("numpy").__version__,
            "sqlite": sqlite3.sqlite_version,
            "platform": platform.system(),
            "machine": platform.machine(),
        },
        "model": {
            "sha256": sha256_file(ROOT / "pilot/frozen_pilot_model.json"),
            "type": model.get("model_type"),
            "feature_count": len(model.get("weights", [])),
            "decision_policy": model.get("decision_policy"),
            "validation": evaluation.get("validation"),
            "pilot_test_report_only": evaluation.get("test"),
        },
        "files": {
            "source_and_documentation": available_files(STATIC_FILES),
            "pilot_evidence": available_files(PILOT_FILES),
            "dataset": available_files(DATASET_FILES),
            "final_outputs": available_files(OPTIONAL_OUTPUTS),
        },
        "excluded": [
            "raw datasets from Git",
            "generated SQLite databases and WAL sidecars",
            "logs and run artifacts",
            "final multi-gigabyte outputs and submission ZIP",
        ],
    }
    output = args.output.expanduser().resolve()
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
