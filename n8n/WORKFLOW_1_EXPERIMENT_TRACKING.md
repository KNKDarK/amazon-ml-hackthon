# Workflow 1 — Experiment Tracking

## Current status

- Local inactive draft: `n8n/workflow_1_experiment_tracking.json`
- Source/generator: `n8n/workflow_1_experiment_tracking.mjs`
- Synthetic test: `n8n/workflow_1_experiment_tracking.test.mjs`
- Synthetic request: `n8n/samples/workflow_1_experiment_report.synthetic.json`
- Created in n8n as inactive on 2026-09-25.
- n8n workflow ID: `W5Lie7xICsA2p3kF`
- n8n name: `ML Team - 1 Experiment Tracking (Inactive Draft)`
- The authenticated Webhook node is intentionally waiting for a user-created Header Auth credential; no secret is stored in the workflow.
- No existing n8n workflow was changed, activated, or deleted.
- `https://github.com/KNKDarK/amazon-ml-hackthon.git` was inspected read-only and is currently empty.

## Trigger

- Node: `Authenticated Experiment Webhook`
- Type: core `Webhook` 2.1
- Method: `POST`
- Path: `ml-experiment-status-v1`
- Authentication: n8n `Header Auth` credential
- Response mode: `Using Respond to Webhook Node`
- Binary input: disabled
- The JSON contains a credential **reference placeholder**, never a secret.
- Intended production URL after activation: `http://<self-hosted-n8n>/webhook/<generated-id>/ml-experiment-status-v1`

The Header Auth credential must be created/bound in n8n before this draft can be activated.

## Main path

1. `Authenticated Experiment Webhook`
2. `Validate and Normalize`
3. `Payload Valid?`
   - true → `Get Experiment State`
   - false → `Respond Validation Error`
4. `Get Experiment State`
5. `Decide Event Handling`
6. `Should Persist?`
   - true → `Persist Experiment State` → `Build Accepted Response` → `Respond Accepted`
   - false → `Build Acknowledgement` → `Respond Duplicate Stale or Conflict`

The webhook and data-table paths process one compact item. No LLM, vector store, training, dataset, leaderboard, geocoding, or business-identity node is present.

## Payload schema v1

Maximum serialized request size: **16 KiB**. Binary is rejected. Unknown fields are rejected. Business record fields such as `business_name`, `business_address`, `entity_id`, `records`, and `rows` are rejected recursively.

```json
{
  "schema_version": 1,
  "event_id": "evt-synthetic-job-001-v2",
  "event_version": 2,
  "job_id": "job-synthetic-001",
  "occurred_at": "2026-09-25T10:05:00Z",
  "status": "completed",
  "runner_slot": "B",
  "ram_gb": 16,
  "experiment": {
    "name": "synthetic-logreg-v1",
    "code_revision": "abc1234",
    "config_hash": "<64 lowercase hex characters>",
    "dataset_fingerprint": "<64 lowercase hex characters>"
  },
  "metrics": {
    "f05": 0.8125,
    "precision": 0.84,
    "recall": 0.79,
    "candidate_count": 1200,
    "runtime_seconds": 4.2,
    "peak_rss_mb": 310
  },
  "artifacts": [
    {
      "kind": "report",
      "ref": "artifact:synthetic-report-001",
      "sha256": "<64 lowercase hex characters>",
      "bytes": 2048
    }
  ]
}
```

Rules:

- `event_id` and `job_id`: 3–80 safe identifier characters.
- `event_version`: positive integer; lower/equal versions are stale unless they are the exact last event.
- `status`: `queued`, `running`, `blocked`, `completed`, `failed`, or `cancelled`.
- Runner mapping is fixed: `A=8 GB`, `B=16 GB`, `C=16 GB`, `D=32 GB`.
- A `completed` event must include `metrics.f05`.
- `metrics` contains aggregate scalar metrics only.
- `artifacts` contains at most eight opaque metadata references; it cannot contain file contents or absolute/file URLs.
- No names, addresses, entity IDs, matched ID lists, raw rows, samples, or datasets are accepted.

## Idempotency outcomes

The current state row is read by `job_id`, then one of these outcomes is returned:

| Outcome | HTTP | State change |
|---|---:|---|
| `created` | 202 | Insert first state for the job |
| `updated` | 202 | Upsert only when `event_version` is newer |
| `duplicate` | 200 | None; same `event_id` and same canonical payload |
| `stale` | 200 | None; version is not newer |
| `conflict` | 409 | None; same `event_id` was reused with a different payload |

This provides deterministic retry handling for normal webhook retries. n8n Data Tables do not expose a unique constraint, so two truly simultaneous first writes for the same new `job_id` remain a store-level race. If concurrent first writes become common, use a transactional self-hosted store with a unique `job_id` constraint while retaining this payload contract.

## Stored tables

Only the following n8n Data Tables are needed; all three were created in the Personal project:

- `ml_experiment_state` — `IhyUkUj6xfvpbpIr`
- `ml_workflow_failures` — `5P7HWVZ95vJkuPtg`
- `ml_failure_notifications` — `VEWokLGO0k5pZLYZ`

### `ml_experiment_state`

One current row per `job_id` with schema version, last event ID/version, timestamps, status, runner slot/RAM, experiment fingerprints, aggregate metrics, compact artifact metadata, canonical event fingerprint, and payload byte count. Metric `-1` means not reported.

### `ml_workflow_failures`

One sanitized row per failed execution: workflow/execution IDs, timestamp, failed node, and error type. It excludes request headers, credentials, input payloads, and free-form error messages.

### `ml_failure_notifications`

One durable, sanitized failure-notification outbox row per failed execution. `delivery_state` starts as `pending`.

The optional external alert node is disabled and points to a non-routable `.invalid` placeholder. A real self-hosted notification endpoint and Header Auth credential must be explicitly approved/configured before that node is enabled. The internal outbox is the safe default notification record.

## Error handling

- Validation failures return structured 400/413/415/422 responses without echoing untrusted values.
- Data persistence and runtime failures are caught by the same workflow's `Workflow Failure Trigger`.
- The error path extracts only safe identifiers, node name, and error type.
- Failure summary and notification-outbox writes continue independently so one optional table failure does not suppress the other attempt.
- The disabled external alert has a 5-second timeout and two bounded attempts when later approved.
- Main execution timeout: 30 seconds.
- n8n success and error execution payload retention: disabled.
- Compact audit history is kept in the three Data Tables instead of verbose n8n execution data.

The static validator emits defensive warnings that Code nodes can throw and that a response node cannot handle an unexpected trigger-level failure. The Code nodes avoid throwing for expected input errors, and the Error Trigger sanitizes unexpected failures. Unexpected failures still produce an n8n-level error and the compact failure path runs.

## Synthetic test results

Command:

```bash
node n8n/workflow_1_experiment_tracking.test.mjs
```

Result:

```text
PASS workflow 1 synthetic tests
state_rows=1
validated=created,duplicate,conflict,updated,stale
rejected=oversized,raw_data,unknown_field,ram_mismatch,missing_metric,bad_artifact
failure_sanitization=PASS
```

The test uses fabricated metadata only. It verifies that an incoming authorization header and a fabricated error message cannot propagate into stored event/notification output.

## Static n8n validation

Strict validation result:

- 17 total nodes
- 16 enabled; one optional external alert intentionally disabled
- 2 triggers: Webhook and Error Trigger
- 14 valid connections
- 0 invalid connections
- 50 expressions validated
- 0 errors
- 7 non-blocking defensive warnings described above

## Four-workflow plan

1. **Experiment tracking** — this draft; compact job/event state and metrics.
2. **Task coordination** — authenticated task-status webhook, `task_id` plus `job_id`, slot-based ownership, due/blocker summaries, and idempotent updates. No personal identity lookup.
3. **Validation and quality gates** — accepts only scalar validation summaries and gate decisions; returns a compact gate report. It never reads raw records or launches training.
4. **Submission readiness** — accepts validator/check summaries and returns `ready`, `not_ready`, or `human_review_required`; it never uploads to a leaderboard and never starts heavy work.

Workflow 2 has not been built.
