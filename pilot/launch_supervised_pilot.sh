#!/usr/bin/env bash
# Persistent launcher for the approved 10K pilot only.
# The systemd user unit owns this process; closing an interactive terminal does not stop it.

set -u
set -o pipefail

<<<<<<< HEAD
readonly ROOT=/home/knk/ml/student_resource
readonly DATA_ROOT=/home/knk/ml/student_resource/dataset
readonly WORK_DIR=/home/knk/ml/student_resource/artifacts/pilot_10k_restart_20260925
readonly PILOT_SCRIPT=/home/knk/ml/student_resource/pilot/run_pilot.py
readonly PYTHON=/usr/bin/python3
readonly LOCK_FILE=/home/knk/ml/student_resource/artifacts/pilot_10k_restart_20260925/launcher.lock
readonly PID_FILE=/home/knk/ml/student_resource/artifacts/pilot_10k_restart_20260925/launcher.pid
readonly EXIT_FILE=/home/knk/ml/student_resource/artifacts/pilot_10k_restart_20260925/exit_code
readonly STATUS_FILE=/home/knk/ml/student_resource/artifacts/pilot_10k_restart_20260925/launcher_status.json
readonly SERVICE_ID=student-resource-pilot-10k.service
=======
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-"$ROOT/dataset"}
WORK_DIR=${WORK_DIR:-"$ROOT/artifacts/pilot_10k_supervised"}
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
SELECTION_CAP=${SELECTION_CAP:-500}
MAX_PROJECTED_POSTINGS=${MAX_PROJECTED_POSTINGS:-5000000}
SERVICE_ID=${SERVICE_ID:-student-resource-pilot-10k.service}

# Worker processes for the two full-corpus blocking passes. Those passes are pure
# Python and GIL-bound, so this is process parallelism, not thread parallelism.
# run_pipeline clamps the request to the CPU and RAM ceilings; 1 forces the
# serial path. RAM_BUDGET_GIB=0 auto-sizes to 60% of MemAvailable.
#
# 6 is the physical core count of this machine (12 logical with SMT). Measured on
# a 200 MiB corpus slice: 6 workers reached 4.32x and 8/10/11 stayed flat at
# ~4.3x, because SMT siblings add no throughput to GIL-bound pure Python. Raise
# WORKERS on a machine with more physical cores.
WORKERS=${WORKERS:-6}
RAM_BUDGET_GIB=${RAM_BUDGET_GIB:-0}
NATIVE_THREADS=${NATIVE_THREADS:-1}

# Each worker runs a 30-wide dot product, so a native BLAS/OpenMP pool per worker
# only oversubscribes the CPU. Pin it before the interpreter starts NumPy.
export OMP_NUM_THREADS="$NATIVE_THREADS"
export OPENBLAS_NUM_THREADS="$NATIVE_THREADS"
export MKL_NUM_THREADS="$NATIVE_THREADS"
export NUMEXPR_NUM_THREADS="$NATIVE_THREADS"
export VECLIB_MAXIMUM_THREADS="$NATIVE_THREADS"
export BLIS_NUM_THREADS="$NATIVE_THREADS"
export GOTO_NUM_THREADS="$NATIVE_THREADS"

readonly ROOT DATA_ROOT WORK_DIR PYTHON SELECTION_CAP MAX_PROJECTED_POSTINGS SERVICE_ID
readonly WORKERS RAM_BUDGET_GIB NATIVE_THREADS
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
>>>>>>> 5f56dee (Process-parallel full-corpus blocking passes)

mkdir -p -- "$WORK_DIR"
cd "$ROOT" || exit 70

# All supervised launches share this lock.  A second systemd start or manual
# invocation exits immediately instead of touching the candidate database.
exec 9>"$LOCK_FILE"
if ! /usr/bin/flock -n 9; then
    printf '%s duplicate pilot refused: lock is already held\n' "$(date --iso-8601=seconds)" >&2
    exit 75
fi

# Also refuse to overlap the legacy terminal-bound launcher, which did not use
# this lock.  This check is intentionally done before starting Python.
legacy_pids=$(/usr/bin/pgrep -f -- 'pilot/run_pilot.py' || true)
for legacy_pid in $legacy_pids; do
    legacy_cmdline=""
    if [ -r "/proc/$legacy_pid/cmdline" ]; then
        legacy_cmdline=$(/usr/bin/tr '\0' ' ' < "/proc/$legacy_pid/cmdline")
    fi
    case "$legacy_cmdline" in
        *"--work-dir $WORK_DIR"*|*"--work-dir artifacts/pilot_10k_restart_20260925"*)
            printf '%s duplicate pilot refused: legacy run_pilot.py process is active (pid=%s)\n' \
                "$(date --iso-8601=seconds)" "$legacy_pid" >&2
            exit 75
            ;;
    esac
done

write_status() {
    local state=$1
    local exit_code=$2
    local pilot_pid=$3
    local started_at=$4
    local ended_at
    local tmp
    ended_at=$(date --iso-8601=seconds)
    tmp="${STATUS_FILE}.tmp.$$"
    /usr/bin/python3 - "$tmp" "$state" "$exit_code" "$pilot_pid" "$started_at" "$ended_at" "$SERVICE_ID" "${INVOCATION_ID:-}" <<'PY'
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
    "working_directory": "/home/knk/ml/student_resource",
    "data_root": "/home/knk/ml/student_resource/dataset",
    "work_dir": "/home/knk/ml/student_resource/artifacts/pilot_10k_restart_20260925",
    "scope": "training-source-1-sample-10000-only",
}
temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(path)
PY
}

started_at=$(date --iso-8601=seconds)
printf '%s\n' "$$" > "$PID_FILE"
write_status starting "" "" "$started_at" "$started_at"

printf '%s starting %s (wrapper_pid=%s invocation_id=%s)\n' \
    "$started_at" "$SERVICE_ID" "$$" "${INVOCATION_ID:-unknown}" >&2

/usr/bin/python3 "$PILOT_SCRIPT" \
    --data-root "$DATA_ROOT" \
    --work-dir "$WORK_DIR" \
    --sample-size 10000 \
    --max-projected-postings 2000000 \
    --negative-per-query 30 \
    --batch-size 10000 \
    --workers "$WORKERS" \
    --ram-budget-gib "$RAM_BUDGET_GIB" &
pilot_pid=$!
printf '%s\n' "$pilot_pid" > "$PID_FILE"

set +e
wait "$pilot_pid"
rc=$?
set -e
ended_at=$(date --iso-8601=seconds)
printf '%s\n' "$rc" > "${EXIT_FILE}.tmp.$$"
mv -f -- "${EXIT_FILE}.tmp.$$" "$EXIT_FILE"
write_status exited "$rc" "$pilot_pid" "$started_at" "$ended_at"
printf '%s pilot exited: service=%s pid=%s exit_code=%s\n' \
    "$ended_at" "$SERVICE_ID" "$pilot_pid" "$rc" >&2
exit "$rc"
