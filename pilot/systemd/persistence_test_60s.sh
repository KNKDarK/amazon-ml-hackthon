#!/usr/bin/env bash
# Optional Linux systemd-user persistence check. It never reads challenge data.
set -euo pipefail

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)}
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
TEST_DIR=${TEST_DIR:-"$ROOT/artifacts/pilot_supervision_test"}
SERVICE_ID=${SERVICE_ID:-student-resource-pilot-persistence-test.service}
readonly ROOT PYTHON TEST_DIR SERVICE_ID
readonly EXIT_FILE="$TEST_DIR/exit_code"
readonly STATUS_FILE="$TEST_DIR/status.json"

if [[ ! -x "$PYTHON" ]]; then
    printf 'Python interpreter is not executable: %s\n' "$PYTHON" >&2
    exit 64
fi
mkdir -p -- "$TEST_DIR"
now() {
    "$PYTHON" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).isoformat())'
}

started_at=$(now)
parent_pid=$PPID
parent_command=$(ps -p "$parent_pid" -o comm= 2>/dev/null | tr -d '[:space:]' || true)
printf '%s persistence test started: service=%s shell_pid=%s parent_pid=%s parent=%s\n' \
    "$started_at" "$SERVICE_ID" "$$" "$parent_pid" "${parent_command:-unknown}" >&2

sleep 60
rc=$?
ended_at=$(now)
printf '%s\n' "$rc" > "${EXIT_FILE}.tmp.$$"
mv -f -- "${EXIT_FILE}.tmp.$$" "$EXIT_FILE"
"$PYTHON" - "$STATUS_FILE.tmp.$$" "$SERVICE_ID" "$rc" "$$" "$parent_pid" \
    "${parent_command:-unknown}" "$started_at" "$ended_at" "${INVOCATION_ID:-}" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "service_id": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "shell_pid": int(sys.argv[4]),
    "parent_pid": int(sys.argv[5]),
    "parent_command": sys.argv[6],
    "started_at": sys.argv[7],
    "ended_at": sys.argv[8],
    "invocation_id": sys.argv[9] or None,
    "duration_requested_seconds": 60,
    "accessed_pilot_data": False,
}
temp = path.with_name(path.name + f".tmp.{os.getpid()}")
temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temp.replace(path)
PY
printf '%s persistence test finished: service=%s exit_code=%s\n' \
    "$ended_at" "$SERVICE_ID" "$rc" >&2
exit "$rc"
