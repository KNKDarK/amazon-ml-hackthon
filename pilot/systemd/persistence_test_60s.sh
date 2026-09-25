#!/usr/bin/env bash
# Harmless systemd-user persistence check. It never reads pilot or test data.
set -u

readonly TEST_DIR=/home/knk/ml/student_resource/artifacts/pilot_supervision_test
readonly EXIT_FILE=/home/knk/ml/student_resource/artifacts/pilot_supervision_test/exit_code
readonly STATUS_FILE=/home/knk/ml/student_resource/artifacts/pilot_supervision_test/status.json
readonly SERVICE_ID=student-resource-pilot-persistence-test-20260925.service

mkdir -p -- "$TEST_DIR"
started_at=$(date --iso-8601=seconds)
parent_pid=$PPID
parent_command=$(/usr/bin/ps -p "$parent_pid" -o comm= 2>/dev/null | tr -d '[:space:]' || true)
printf '%s persistence test started: service=%s shell_pid=%s parent_pid=%s parent=%s\n' \
    "$started_at" "$SERVICE_ID" "$$" "$parent_pid" "${parent_command:-unknown}" >&2

/usr/bin/sleep 60
rc=$?
ended_at=$(date --iso-8601=seconds)
printf '%s\n' "$rc" > "${EXIT_FILE}.tmp.$$"
mv -f -- "${EXIT_FILE}.tmp.$$" "$EXIT_FILE"

/usr/bin/python3 - "$STATUS_FILE.tmp.$$" "$SERVICE_ID" "$rc" "$$" "$parent_pid" \
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
final_path = path.parent / "status.json"
temp.replace(final_path)
PY
printf '%s persistence test finished: service=%s exit_code=%s\n' \
    "$ended_at" "$SERVICE_ID" "$rc" >&2
exit "$rc"
