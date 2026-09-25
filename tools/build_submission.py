#!/usr/bin/env python3
"""Build and verify the challenge's required cross-platform submission ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.validate_submission import validate  # noqa: E402

SOURCE_FILES = {
    "pilot/__init__.py": "code/business_entity_resolution/src/pilot/__init__.py",
    "pilot/er_common.py": "code/business_entity_resolution/src/pilot/er_common.py",
    "pilot/run_pilot.py": "code/business_entity_resolution/src/pilot/run_pilot.py",
    "pilot/stream_infer.py": "code/business_entity_resolution/src/pilot/stream_infer.py",
    "pilot/verify_artifacts.py": "code/business_entity_resolution/src/pilot/verify_artifacts.py",
    "pilot/analyze_cap_validation.py": "code/business_entity_resolution/src/pilot/analyze_cap_validation.py",
    "pilot/frozen_pilot_model.json": "code/business_entity_resolution/src/pilot/frozen_pilot_model.json",
    "utils/__init__.py": "code/business_entity_resolution/src/utils/__init__.py",
    "utils/validate_submission.py": "code/business_entity_resolution/src/utils/validate_submission.py",
}

REQUIRED_OUTPUTS = (
    "output/matching_results.tsv",
    "output/candidate_pairs.tsv",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_team_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
    if not cleaned:
        raise ValueError("team name must contain at least one letter or number")
    return cleaned


def code_readme(team_name: str) -> str:
    return f"""# Business Entity Resolution — {team_name}

This directory is the self-contained source package for the ML Challenge 2026
submission. It uses Python 3.12+ and NumPy; challenge data are not embedded.

## Install

```text
python -m venv .venv
# Windows: .venv\\Scripts\\python -m pip install -r requirements.txt
# macOS/Linux: .venv/bin/python -m pip install -r requirements.txt
```

## Run

Pass the challenge test directory explicitly. Paths may contain spaces.

```text
python src/pilot/stream_infer.py \\
  --data-root /path/to/dataset/test \\
  --work-dir /path/to/local-work-directory \\
  --output-dir /path/to/output \\
  --model src/pilot/frozen_pilot_model.json \\
  --mode full --queries 0 --target-sample-rate 1 --query-stride 1 \\
  --index-batch 5000 --index-synchronous FULL \\
  --block-cap 500 --query-posting-cap 500
```

Use a local NTFS, APFS, or ext4 filesystem. Do not put the work directory on
SMB/NFS or an actively synchronized cloud folder. If a run is interrupted,
repeat the identical command against the same work directory; the checkpoint
binds the input contents, model, feature code, mode, and blocking profile.

## Validate

```text
python src/utils/validate_submission.py \\
  --matching /path/to/output/matching_results.tsv \\
  --candidate /path/to/output/candidate_pairs.tsv \\
  --test-dir /path/to/dataset/test \\
  --check-ids
```

The output files use UTF-8 without a BOM, tab separators, exact headers, and LF
line endings. The final archive includes the already-generated files under
`output/`; the command above regenerates them.
"""


def validate_inputs(
    matching: Path,
    candidate: Path,
    test_dir: Path,
    documentation: Path,
    check_ids: bool,
) -> None:
    for path in (matching, candidate, documentation):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not test_dir.is_dir():
        raise FileNotFoundError(test_dir)
    for source in SOURCE_FILES:
        path = ROOT / source
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.is_symlink():
            raise ValueError(f"submission source must not be a symlink: {source}")
    documentation_text = documentation.read_text(encoding="utf-8")
    if re.search(r"\[(?:Your Team Name|List all team members|Date|Describe|Provide)", documentation_text):
        raise ValueError("documentation still contains template placeholders")

    errors, warnings = validate(
        str(matching), str(candidate), str(test_dir), check_ids=check_ids
    )
    if warnings:
        raise ValueError("submission validator returned blocking warnings: " + " | ".join(warnings))
    if errors:
        raise ValueError("submission validator failed: " + " | ".join(errors))


def zip_info(name: str, timestamp: tuple[int, int, int, int, int, int]) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=timestamp)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    return info


def add_bytes(
    archive: zipfile.ZipFile,
    name: str,
    payload: bytes,
    timestamp: tuple[int, int, int, int, int, int],
) -> None:
    archive.writestr(zip_info(name, timestamp), payload)


def add_file(
    archive: zipfile.ZipFile,
    source: Path,
    name: str,
    timestamp: tuple[int, int, int, int, int, int],
) -> None:
    with source.open("rb") as input_handle, archive.open(
        zip_info(name, timestamp), "w", force_zip64=True
    ) as output_handle:
        shutil.copyfileobj(input_handle, output_handle, length=8 * 1024 * 1024)


def build_manifest(
    team_name: str,
    documentation: Path,
    matching: Path,
    candidate: Path,
    timestamp: tuple[int, int, int, int, int, int],
) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for source, archive_name in sorted(SOURCE_FILES.items()):
        path = ROOT / source
        files.append({
            "path": archive_name,
            "source": source,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    for relative, path in zip(REQUIRED_OUTPUTS, (matching, candidate)):
        files.append({
            "path": relative,
            "source": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    files.append({
        "path": "Documentation_template.md",
        "source": str(documentation.relative_to(ROOT)) if documentation.is_relative_to(ROOT) else str(documentation),
        "bytes": documentation.stat().st_size,
        "sha256": sha256_file(documentation),
    })
    model = json.loads((ROOT / "pilot/frozen_pilot_model.json").read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "team_name": team_name,
        "built_at_utc": datetime(*timestamp[:6], tzinfo=timezone.utc).isoformat(),
        "model": {
            "type": model.get("model_type"),
            "license": "MIT",
            "feature_count": len(model.get("weights", [])),
            "decision_policy": model.get("decision_policy"),
        },
        "excluded": [".github", "dataset", "artifacts", "virtual environments", "CI cache"],
        "files": files,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--documentation", type=Path, default=ROOT / "Documentation_template.md")
    parser.add_argument("--matching", type=Path, default=ROOT / "output/matching_results.tsv")
    parser.add_argument("--candidate", type=Path, default=ROOT / "output/candidate_pairs.tsv")
    parser.add_argument("--test-dir", type=Path, default=ROOT / "dataset/test")
    parser.add_argument("--output-zip", type=Path, default=None)
    parser.add_argument("--check-ids", action="store_true")
    parser.add_argument("--compression-level", type=int, choices=range(0, 10), default=6)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    team_name = safe_team_name(args.team_name)
    documentation = args.documentation.expanduser().resolve()
    matching = args.matching.expanduser().resolve()
    candidate = args.candidate.expanduser().resolve()
    test_dir = args.test_dir.expanduser().resolve()
    output_zip = (
        args.output_zip.expanduser().resolve()
        if args.output_zip
        else ROOT / "dist" / f"{team_name}_submission.zip"
    )
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    temporary_zip = output_zip.with_suffix(output_zip.suffix + ".tmp")

    validate_inputs(matching, candidate, test_dir, documentation, args.check_ids)
    timestamp = (2026, 9, 25, 0, 0, 0)
    readme = code_readme(team_name).encode("utf-8")
    requirements = b"# Reproducibility-pinned runtime\nnumpy==2.5.3\n"
    license_text = (ROOT / "LICENSE").read_bytes()
    model_license = (ROOT / "MODEL_LICENSE.md").read_bytes()
    manifest = build_manifest(
        team_name, documentation, matching, candidate, timestamp
    )
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")

    try:
        with zipfile.ZipFile(
            temporary_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=args.compression_level,
            allowZip64=True,
        ) as archive:
            add_file(archive, matching, "output/matching_results.tsv", timestamp)
            add_file(archive, candidate, "output/candidate_pairs.tsv", timestamp)
            for source, archive_name in sorted(SOURCE_FILES.items()):
                add_file(archive, ROOT / source, archive_name, timestamp)
            add_bytes(archive, "code/business_entity_resolution/README.md", readme, timestamp)
            add_bytes(archive, "code/business_entity_resolution/requirements.txt", requirements, timestamp)
            add_bytes(archive, "code/business_entity_resolution/LICENSE", license_text, timestamp)
            add_bytes(archive, "code/business_entity_resolution/MODEL_LICENSE.md", model_license, timestamp)
            add_file(archive, documentation, "Documentation_template.md", timestamp)
            add_bytes(archive, "SUBMISSION_MANIFEST.json", manifest_bytes, timestamp)
        os.replace(temporary_zip, output_zip)
    except BaseException:
        temporary_zip.unlink(missing_ok=True)
        raise

    expected = {
        "output/matching_results.tsv",
        "output/candidate_pairs.tsv",
        "Documentation_template.md",
        "SUBMISSION_MANIFEST.json",
        "code/business_entity_resolution/README.md",
        "code/business_entity_resolution/requirements.txt",
    }
    with zipfile.ZipFile(output_zip, "r") as archive:
        names = set(archive.namelist())
        missing = sorted(expected - names)
        if missing:
            output_zip.unlink(missing_ok=True)
            raise ValueError("built archive is missing: " + ", ".join(missing))
        bad = archive.testzip()
        if bad:
            output_zip.unlink(missing_ok=True)
            raise ValueError(f"built archive has a corrupt member: {bad}")
        if any(name.startswith((".github/", "dataset/", "artifacts/")) for name in names):
            output_zip.unlink(missing_ok=True)
            raise ValueError("built archive contains an excluded directory")

    print(json.dumps({
        "status": "PASS",
        "zip": str(output_zip),
        "bytes": output_zip.stat().st_size,
        "sha256": sha256_file(output_zip),
        "members": len(names),
        "id_check": bool(args.check_ids),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
