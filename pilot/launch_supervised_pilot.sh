#!/usr/bin/env bash
# Optional Linux launcher for the training-only 10K pilot.
# The Python process has its own cross-platform work-directory lock; this wrapper
# adds systemd-friendly status/exit-code reporting.

set -euo pipefail
set -o pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-"$ROOT/dataset"}
WORK_DIR=${WORK_DIR:-"$ROOT/artifacts/pilot_10k_supervised"}
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
SELECTION_CAP=${SELECTION_CAP:-500}
MAX_PROJECTED_POSTINGS=${MAX_PROJECTED_POSTINGS:-5000000}
SERVICE_ID=${SERVICE_ID:-student-resource-pilot-10k.service}

readonly ROOT DATA_ROOT WORK_DIR PYTHON SELECTION_CAP MAX_PROJECTED_POSTINGS SERVICE_ID
readonly PILOT_SCRIPT="$ROOT/pilot/run_pilot.py"
readonly LOCK_FILE="$WORK_DIR/launcher.lock"
readonly PID_FILE="$WORK_DIR/launcher.pid"
readonly EXIT_FILE="$WORK_DIR/exit_code"
readonly STATUS_FILE="$WORK_DIR/launcher_status.json"

if [[ ! -x "$PYTHON" ]]; then
    printf 'Python interpreter is not executable: %s\n' "$PYTHON" >&2
    exit 64
fi
if ! command -v flock >/dev/null 2>&1; then
    printf 'Required Linux utility not found: flock\n' >&2
    exit 69
fi

mkdir -p -- "$WORK_DIR"
cd -- "$ROOT"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
    printf 'Duplicate pilot refused: %s is already locked\n' "$LOCK_FILE" >&2
    exit 75
fi

now() {
    "$PYTHON" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())'
}

write_status() {
    local state=$1
    local exit_code=$2
    local pilot_pid=$3
    local started_at=$4
    local ended_at=$5
    local temporary="${STATUS_FILE}.tmp.$$"
    "$PYTHON" - "$temporary" "$state" "$exit_code" "$pilot_pid" "$started_at" \
        "$ended_at" "$SERVICE_ID" "${INVOCATION_ID:-}" "$ROOT" "$DATA_ROOT" "$WORK_DIR" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "service_id": sys.argv[7],
    "invocation_id": sys.argv[8] or None,
    "state": sys.argv[2],
    "exit_code": int(sys.argv[3]) if sys.argv[3] else None,
    "pilot_pid": int(sys.argv[4]) if sys.argv[4] else None,
    "started_at": sys.argv[5],
    "ended_at": sys.argv[6],
    "working_directory": sys.argv[9],
    "data_root": sys.argv[10],
    "work_dir": sys.argv[11],
    "scope": "training-source-1-sample-10000-only",
}
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

started_at=$(now)
printf '%s\n' "$$" > "$PID_FILE"
write_status starting "" "" "$started_at" "$started_at"
printf '%s starting %s (wrapper_pid=%s invocation_id=%s)\n' \
    "$started_at" "$SERVICE_ID" "$$" "${INVOCATION_ID:-unknown}" >&2

"$PYTHON" -u "$PILOT_SCRIPT" \
    --data-root "$DATA_ROOT" \
    --work-dir "$WORK_DIR" \
    --sample-size 10000 \
    --selection-cap "$SELECTION_CAP" \
    --max-projected-postings "$MAX_PROJECTED_POSTINGS" \
    --negative-per-query 30 \
    --batch-size 10000 &
pilot_pid=$!
printf '%s\n' "$pilot_pid" > "$PID_FILE"

set +e
wait "$pilot_pid"
rc=$?
set -e
ended_at=$(now)
printf '%s\n' "$rc" > "${EXIT_FILE}.tmp.$$"
mv -f -- "${EXIT_FILE}.tmp.$$" "$EXIT_FILE"
write_status exited "$rc" "$pilot_pid" "$started_at" "$ended_at"
printf '%s pilot exited: service=%s pid=%s exit_code=%s\n' \
    "$ended_at" "$SERVICE_ID" "$pilot_pid" "$rc" >&2
exit "$rc"
