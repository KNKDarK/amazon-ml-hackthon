import { writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const validateAndNormalizeCode = String.raw`
const MAX_PAYLOAD_BYTES = 16 * 1024;
const ALLOWED_TOP_LEVEL = new Set([
  'schema_version',
  'event_id',
  'event_version',
  'job_id',
  'occurred_at',
  'status',
  'runner_slot',
  'ram_gb',
  'experiment',
  'metrics',
  'artifacts',
]);
const FORBIDDEN_KEYS = new Set([
  'business_name',
  'business_address',
  'entity_id',
  'entity_ids',
  'matched_entity_ids',
  'records',
  'rows',
  'raw_data',
  'dataset_rows',
]);

const STATUSES = new Set(['queued', 'running', 'blocked', 'completed', 'failed', 'cancelled']);
const RUNNER_RAM = { A: 8, B: 16, C: 16, D: 32 };
const METRIC_KEYS = new Set([
  'f05',
  'precision',
  'recall',
  'candidate_count',
  'runtime_seconds',
  'peak_rss_mb',
]);
const ARTIFACT_KEYS = new Set(['kind', 'ref', 'sha256', 'bytes']);
const ARTIFACT_KINDS = new Set(['report', 'model', 'code', 'config']);
const SAFE_ID = /^[A-Za-z0-9][A-Za-z0-9._:-]{2,79}$/;
const SHA256 = /^[a-f0-9]{64}$/;
const ARTIFACT_REF = /^(artifact|git|run):[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,159}$/;

function isPlainObject(value) {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function rejection(status, reason, message) {
  return [{
    json: {
      valid: false,
      http_status: status,
      response: {
        schema_version: 1,
        outcome: 'rejected',
        reason,
        message,
      },
    },
  }];
}

function containsForbiddenKey(value, depth = 0) {
  if (depth > 5 || value === null || typeof value !== 'object') return false;
  if (Array.isArray(value)) return value.some((entry) => containsForbiddenKey(entry, depth + 1));
  return Object.entries(value).some(([key, entry]) =>
    FORBIDDEN_KEYS.has(key.toLowerCase()) || containsForbiddenKey(entry, depth + 1),
  );
}

function canonicalize(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(canonicalize).join(',') + ']';
  return '{' + Object.keys(value).sort().map((key) =>
    JSON.stringify(key) + ':' + canonicalize(value[key]),
  ).join(',') + '}';
}

if (items.length !== 1) return rejection(400, 'one_request_required', 'Send exactly one JSON object per request.');
if (items[0].binary && Object.keys(items[0].binary).length > 0) {
  return rejection(415, 'binary_not_accepted', 'Send metadata JSON only; do not send files or datasets.');
}

const body = items[0].json?.body;
if (!isPlainObject(body)) {
  return rejection(400, 'json_object_required', 'Request body must be one JSON object.');
}

let payloadText;
try {
  payloadText = JSON.stringify(body);
} catch {
  return rejection(400, 'invalid_json', 'Request body must be valid JSON.');
}

const payloadBytes = new TextEncoder().encode(payloadText).length;
if (payloadBytes > MAX_PAYLOAD_BYTES) {
  return rejection(413, 'payload_too_large', 'Payload exceeds the 16 KiB limit.');
}
if (containsForbiddenKey(body)) {
  return rejection(422, 'raw_business_data_forbidden', 'Business records and raw row data are not accepted.');
}

const topLevelKeys = Object.keys(body);
const unknownKeys = topLevelKeys.filter((key) => !ALLOWED_TOP_LEVEL.has(key));
if (unknownKeys.length > 0) {
  return rejection(422, 'unexpected_fields', 'Payload contains fields outside schema version 1.');
}

const requiredKeys = [
  'schema_version',
  'event_id',
  'event_version',
  'job_id',
  'occurred_at',
  'status',
  'runner_slot',
  'ram_gb',
  'experiment',
];
if (requiredKeys.some((key) => !Object.prototype.hasOwnProperty.call(body, key))) {
  return rejection(422, 'missing_required_fields', 'One or more required schema version 1 fields are missing.');
}
if (body.schema_version !== 1) {
  return rejection(422, 'unsupported_schema_version', 'Only schema_version 1 is supported.');
}
if (!SAFE_ID.test(body.event_id) || !SAFE_ID.test(body.job_id)) {
  return rejection(422, 'invalid_identifier', 'event_id and job_id must use 3-80 safe identifier characters.');
}
if (!Number.isInteger(body.event_version) || body.event_version < 1 || body.event_version > 1000000) {
  return rejection(422, 'invalid_event_version', 'event_version must be an integer from 1 through 1000000.');
}
if (typeof body.occurred_at !== 'string' || body.occurred_at.length > 40 ||
    !/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,3})?(?:Z|[+-]\d{2}:\d{2})$/.test(body.occurred_at) ||
    !Number.isFinite(Date.parse(body.occurred_at))) {
  return rejection(422, 'invalid_occurred_at', 'occurred_at must be an ISO-8601 timestamp with a timezone.');
}
if (!STATUSES.has(body.status)) {
  return rejection(422, 'invalid_status', 'status is not in the allowed enum.');
}
if (!Object.prototype.hasOwnProperty.call(RUNNER_RAM, body.runner_slot) ||
    Number(body.ram_gb) !== RUNNER_RAM[body.runner_slot]) {
  return rejection(422, 'runner_ram_mismatch', 'runner_slot and ram_gb must match A=8, B=16, C=16, D=32.');
}

if (!isPlainObject(body.experiment)) {
  return rejection(422, 'invalid_experiment', 'experiment must be an object.');
}
const experimentKeys = Object.keys(body.experiment).sort();
const expectedExperimentKeys = ['code_revision', 'config_hash', 'dataset_fingerprint', 'name'];
if (canonicalize(experimentKeys) !== canonicalize(expectedExperimentKeys)) {
  return rejection(422, 'invalid_experiment_fields', 'experiment requires name, code_revision, config_hash, and dataset_fingerprint only.');
}
if (typeof body.experiment.name !== 'string' || body.experiment.name.length < 1 || body.experiment.name.length > 100 ||
    typeof body.experiment.code_revision !== 'string' || body.experiment.code_revision.length < 1 || body.experiment.code_revision.length > 64 ||
    typeof body.experiment.config_hash !== 'string' || !SHA256.test(body.experiment.config_hash) ||
    typeof body.experiment.dataset_fingerprint !== 'string' || !SHA256.test(body.experiment.dataset_fingerprint)) {
  return rejection(422, 'invalid_experiment_metadata', 'Experiment metadata has an invalid value or length.');
}

let metrics;
if (Object.prototype.hasOwnProperty.call(body, 'metrics')) {
  if (!isPlainObject(body.metrics)) return rejection(422, 'invalid_metrics', 'metrics must be an object.');
  if (Object.keys(body.metrics).some((key) => !METRIC_KEYS.has(key))) {
    return rejection(422, 'unexpected_metric_fields', 'metrics contains an unsupported field.');
  }
  metrics = {};
  for (const key of Object.keys(body.metrics).sort()) {
    const value = body.metrics[key];
    if (!Number.isFinite(value) || value < 0) {
      return rejection(422, 'invalid_metric', 'Metric values must be finite and non-negative.');
    }
    if (['f05', 'precision', 'recall'].includes(key) && value > 1) {
      return rejection(422, 'invalid_metric', 'f05, precision, and recall must be between 0 and 1.');
    }
    if (['candidate_count', 'peak_rss_mb'].includes(key) && !Number.isInteger(value)) {
      return rejection(422, 'invalid_metric', 'candidate_count and peak_rss_mb must be integers.');
    }
    metrics[key] = value;
  }
}
if (body.status === 'completed' && (!metrics || !Object.prototype.hasOwnProperty.call(metrics, 'f05'))) {
  return rejection(422, 'completed_metric_required', 'A completed event must include metrics.f05.');
}

let artifacts = [];
if (Object.prototype.hasOwnProperty.call(body, 'artifacts')) {
  if (!Array.isArray(body.artifacts) || body.artifacts.length > 8) {
    return rejection(422, 'invalid_artifacts', 'artifacts must be an array with at most 8 entries.');
  }
  artifacts = body.artifacts.map((artifact) => {
    if (!isPlainObject(artifact) ||
        canonicalize(Object.keys(artifact).sort()) !== canonicalize([...ARTIFACT_KEYS].sort())) {
      return null;
    }
    if (!ARTIFACT_KINDS.has(artifact.kind) ||
        typeof artifact.ref !== 'string' || !ARTIFACT_REF.test(artifact.ref) ||
        typeof artifact.sha256 !== 'string' || !SHA256.test(artifact.sha256) ||
        !Number.isInteger(artifact.bytes) || artifact.bytes < 0 || artifact.bytes > 1000000000000) {
      return null;
    }
    return {
      kind: artifact.kind,
      ref: artifact.ref,
      sha256: artifact.sha256,
      bytes: artifact.bytes,
    };
  });
  if (artifacts.some((artifact) => artifact === null)) {
    return rejection(422, 'invalid_artifact_metadata', 'Artifact entries contain invalid or unsupported metadata.');
  }
}

const event = {
  schema_version: 1,
  event_id: body.event_id,
  event_version: body.event_version,
  job_id: body.job_id,
  occurred_at: body.occurred_at,
  status: body.status,
  runner_slot: body.runner_slot,
  ram_gb: RUNNER_RAM[body.runner_slot],
  experiment_name: body.experiment.name,
  code_revision: body.experiment.code_revision,
  config_hash: body.experiment.config_hash,
  dataset_fingerprint: body.experiment.dataset_fingerprint,
  f05: metrics?.f05 ?? -1,
  precision: metrics?.precision ?? -1,
  recall: metrics?.recall ?? -1,
  candidate_count: metrics?.candidate_count ?? -1,
  runtime_seconds: metrics?.runtime_seconds ?? -1,
  peak_rss_mb: metrics?.peak_rss_mb ?? -1,
  artifacts_json: JSON.stringify(artifacts),
  payload_bytes: payloadBytes,
};
event.event_fingerprint = canonicalize(event);

return [{ json: { valid: true, http_status: 202, event } }];
`;

const decideEventHandlingCode = `
const incoming = $('Validate and Normalize').first().json.event;
const existing = $input.first()?.json ?? {};
const hasExisting = existing.job_id === incoming.job_id;
const now = new Date().toISOString();

let action = 'ack';
let outcome;
let httpStatus = 200;
let reason = null;

if (!hasExisting) {
  action = 'persist';
  outcome = 'created';
  httpStatus = 202;
} else if (existing.last_event_id === incoming.event_id) {
  if (existing.event_fingerprint === incoming.event_fingerprint) {
    outcome = 'duplicate';
    reason = 'event_already_processed';
  } else {
    outcome = 'conflict';
    httpStatus = 409;
    reason = 'event_id_payload_mismatch';
  }
} else if (incoming.event_version <= Number(existing.last_event_version || 0)) {
  outcome = 'stale';
  reason = 'event_version_not_newer';
} else {
  action = 'persist';
  outcome = 'updated';
  httpStatus = 202;
}

const response = {
  schema_version: 1,
  outcome,
  job_id: incoming.job_id,
  event_id: incoming.event_id,
  event_version: incoming.event_version,
};
if (reason) response.reason = reason;
if (hasExisting) {
  response.current_event_id = existing.last_event_id || null;
  response.current_event_version = Number(existing.last_event_version || 0);
  response.current_status = existing.status || null;
}

return [{
  json: {
    action,
    outcome,
    http_status: httpStatus,
    response,
    row: action === 'persist' ? { ...incoming, received_at: now } : null,
  },
}];
`;

const buildAcceptedResponseCode = `
const decision = $('Decide Event Handling').first().json;
return [{
  json: {
    http_status: decision.http_status,
    response: {
      schema_version: 1,
      outcome: decision.outcome,
      job_id: decision.response.job_id,
      event_id: decision.response.event_id,
      event_version: decision.response.event_version,
      status: decision.row.status,
      received_at: decision.row.received_at,
    },
  },
}];
`;

const buildAcknowledgementCode = `
const decision = $input.first().json;
return [{ json: { http_status: decision.http_status, response: decision.response } }];
`;

const sanitizeFailureCode = `
const source = $input.first()?.json ?? {};
const execution = source.execution ?? source.workflowExecution ?? {};
const workflow = execution.workflow ?? source.workflow ?? {};
const error = source.error ?? {};

function safeToken(value, fallback, maxLength) {
  const text = String(value ?? fallback).replace(/[^A-Za-z0-9._:@/-]/g, '_').slice(0, maxLength);
  return text || fallback;
}

const executionId = safeToken(execution.id ?? source.executionId, 'unknown-execution', 80);
const workflowId = safeToken(workflow.id ?? execution.workflowId, 'unknown-workflow', 80);
const workflowName = safeToken(workflow.name ?? execution.workflowName, 'ML Team Workflow', 100);
const failedNode = safeToken(source.lastNodeExecuted ?? execution.lastNodeExecuted, 'unknown-node', 80);
const errorType = safeToken(error.name ?? error.constructor?.name, 'WorkflowError', 80);
const failedAt = new Date().toISOString();
const failureKey = workflowId + ':' + executionId;

return [{
  json: {
    failure: {
      failure_key: failureKey,
      failed_at: failedAt,
      workflow_id: workflowId,
      workflow_name: workflowName,
      execution_id: executionId,
      failed_node: failedNode,
      error_type: errorType,
    },
    notification: {
      schema_version: 1,
      severity: 'error',
      alert_key: failureKey,
      created_at: failedAt,
      workflow_name: workflowName,
      execution_id: executionId,
      failed_node: failedNode,
      error_type: errorType,
      delivery_state: 'pending',
    },
  },
}];
`;

const stateSchema = [
  ['schema_version', 'number'],
  ['job_id', 'string'],
  ['last_event_id', 'string'],
  ['last_event_version', 'number'],
  ['occurred_at', 'date'],
  ['received_at', 'date'],
  ['status', 'string'],
  ['runner_slot', 'string'],
  ['ram_gb', 'number'],
  ['experiment_name', 'string'],
  ['code_revision', 'string'],
  ['config_hash', 'string'],
  ['dataset_fingerprint', 'string'],
  ['f05', 'number'],
  ['precision', 'number'],
  ['recall', 'number'],
  ['candidate_count', 'number'],
  ['runtime_seconds', 'number'],
  ['peak_rss_mb', 'number'],
  ['artifacts_json', 'string'],
  ['event_fingerprint', 'string'],
  ['payload_bytes', 'number'],
];

const failureSchema = [
  ['failure_key', 'string'],
  ['failed_at', 'date'],
  ['workflow_id', 'string'],
  ['workflow_name', 'string'],
  ['execution_id', 'string'],
  ['failed_node', 'string'],
  ['error_type', 'string'],
];

const notificationSchema = [
  ['alert_key', 'string'],
  ['schema_version', 'number'],
  ['severity', 'string'],
  ['created_at', 'date'],
  ['workflow_name', 'string'],
  ['execution_id', 'string'],
  ['failed_node', 'string'],
  ['error_type', 'string'],
  ['delivery_state', 'string'],
];

function resourceMapper(schema, value) {
  return {
    mappingMode: 'defineBelow',
    value,
    matchingColumns: [],
    schema: schema.map(([name, type]) => ({
      id: name,
      displayName: name,
      required: false,
      defaultMatch: false,
      display: true,
      type,
      readOnly: false,
      removed: false,
    })),
    attemptToConvertTypes: false,
    convertFieldsToString: false,
  };
}

export const dataTables = {
  ml_experiment_state: stateSchema.map(([name, type]) => ({ name, type })),
  ml_workflow_failures: failureSchema.map(([name, type]) => ({ name, type })),
  ml_failure_notifications: notificationSchema.map(([name, type]) => ({ name, type })),
};

const responseHeaders = {
  entries: [
    { name: 'Cache-Control', value: 'no-store' },
    { name: 'X-Content-Type-Options', value: 'nosniff' },
  ],
};

export const workflow = {
  name: 'ML Team - 1 Experiment Tracking (Inactive Draft)',
  nodes: [
    {
      id: 'a1000000-0000-4000-8000-000000000001',
      name: 'Authenticated Experiment Webhook',
      type: 'n8n-nodes-base.webhook',
      typeVersion: 2.1,
      position: [-960, 0],
      parameters: {
        multipleMethods: false,
        httpMethod: 'POST',
        path: 'ml-experiment-status-v1',
        authentication: 'headerAuth',
        responseMode: 'responseNode',
        options: {
          rawBody: false,
        },
      },
      credentials: {
        httpHeaderAuth: {
          id: 'REPLACE_WITH_WEBHOOK_HEADER_AUTH_CREDENTIAL_ID',
          name: 'ML Team Webhook Auth',
        },
      },
      notes: 'POST metadata-only JSON. Bind a Header Auth credential before activation. The secret is never stored in this workflow JSON.',
    },
    {
      id: 'a1000000-0000-4000-8000-000000000002',
      name: 'Validate and Normalize',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [-720, 0],
      parameters: {
        mode: 'runOnceForAllItems',
        language: 'javaScript',
        jsCode: validateAndNormalizeCode.trim(),
      },
      notes: 'Strict allowlist, 16 KiB cap, no binary, no raw business records, and safe metadata only.',
    },
    {
      id: 'a1000000-0000-4000-8000-000000000003',
      name: 'Payload Valid?',
      type: 'n8n-nodes-base.if',
      typeVersion: 2.3,
      position: [-480, 0],
      parameters: {
        conditions: {
          options: {
            version: 2,
            leftValue: '',
            caseSensitive: true,
            typeValidation: 'strict',
          },
          combinator: 'and',
          conditions: [
            {
              id: 'a1000000-0000-4000-8000-000000000031',
              leftValue: '={{ $json.valid }}',
              rightValue: true,
              operator: { type: 'boolean', operation: 'true', singleValue: true },
            },
          ],
        },
        options: {},
      },
    },
    {
      id: 'a1000000-0000-4000-8000-000000000004',
      name: 'Respond Validation Error',
      type: 'n8n-nodes-base.respondToWebhook',
      typeVersion: 1.5,
      position: [-220, 180],
      parameters: {
        respondWith: 'json',
        responseBody: '={{ JSON.stringify($json.response) }}',
        options: {
          responseCode: '={{ $json.http_status }}',
          responseHeaders,
        },
      },
    },
    {
      id: 'a1000000-0000-4000-8000-000000000005',
      name: 'Get Experiment State',
      type: 'n8n-nodes-base.dataTable',
      typeVersion: 1.1,
      position: [-220, -100],
      parameters: {
        resource: 'row',
        operation: 'get',
        dataTableId: {
          __rl: true,
          mode: 'name',
          value: 'ml_experiment_state',
          cachedResultName: 'ml_experiment_state',
        },
        matchType: 'allConditions',
        filters: {
          conditions: [
            {
              keyName: 'job_id',
              condition: 'eq',
              keyValue: '={{ $json.event.job_id }}',
            },
          ],
        },
        returnAll: false,
        limit: 1,
      },
      alwaysOutputData: true,
      notes: 'At most one compact state row is read for a job_id.',
    },
    {
      id: 'a1000000-0000-4000-8000-000000000006',
      name: 'Decide Event Handling',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [20, -100],
      parameters: {
        mode: 'runOnceForAllItems',
        language: 'javaScript',
        jsCode: decideEventHandlingCode.trim(),
      },
      notes: 'Idempotent outcomes: created, updated, duplicate, stale, or conflict.',
    },
    {
      id: 'a1000000-0000-4000-8000-000000000007',
      name: 'Should Persist?',
      type: 'n8n-nodes-base.if',
      typeVersion: 2.3,
      position: [260, -100],
      parameters: {
        conditions: {
          options: {
            version: 2,
            leftValue: '',
            caseSensitive: true,
            typeValidation: 'strict',
          },
          combinator: 'and',
          conditions: [
            {
              id: 'a1000000-0000-4000-8000-000000000032',
              leftValue: '={{ $json.action }}',
              rightValue: 'persist',
              operator: { type: 'string', operation: 'equals' },
            },
          ],
        },
        options: {},
      },
    },
    {
      id: 'a1000000-0000-4000-8000-000000000008',
      name: 'Persist Experiment State',
      type: 'n8n-nodes-base.dataTable',
      typeVersion: 1.1,
      position: [500, -180],
      parameters: {
        resource: 'row',
        operation: 'upsert',
        dataTableId: {
          __rl: true,
          mode: 'name',
          value: 'ml_experiment_state',
          cachedResultName: 'ml_experiment_state',
        },
        matchType: 'allConditions',
        filters: {
          conditions: [
            {
              keyName: 'job_id',
              condition: 'eq',
              keyValue: '={{ $json.row.job_id }}',
            },
          ],
        },
        columns: resourceMapper(stateSchema, {
          schema_version: '={{ $json.row.schema_version }}',
          job_id: '={{ $json.row.job_id }}',
          last_event_id: '={{ $json.row.event_id }}',
          last_event_version: '={{ $json.row.event_version }}',
          occurred_at: '={{ $json.row.occurred_at }}',
          received_at: '={{ $json.row.received_at }}',
          status: '={{ $json.row.status }}',
          runner_slot: '={{ $json.row.runner_slot }}',
          ram_gb: '={{ $json.row.ram_gb }}',
          experiment_name: '={{ $json.row.experiment_name }}',
          code_revision: '={{ $json.row.code_revision }}',
          config_hash: '={{ $json.row.config_hash }}',
          dataset_fingerprint: '={{ $json.row.dataset_fingerprint }}',
          f05: '={{ $json.row.f05 }}',
          precision: '={{ $json.row.precision }}',
          recall: '={{ $json.row.recall }}',
          candidate_count: '={{ $json.row.candidate_count }}',
          runtime_seconds: '={{ $json.row.runtime_seconds }}',
          peak_rss_mb: '={{ $json.row.peak_rss_mb }}',
          artifacts_json: '={{ $json.row.artifacts_json }}',
          event_fingerprint: '={{ $json.row.event_fingerprint }}',
          payload_bytes: '={{ $json.row.payload_bytes }}',
        }),
        options: {},
      },
      notes: 'Stores one current state row per job_id. No training data or raw records are stored.',
    },
    {
      id: 'a1000000-0000-4000-8000-000000000009',
      name: 'Build Accepted Response',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [740, -180],
      parameters: {
        mode: 'runOnceForAllItems',
        language: 'javaScript',
        jsCode: buildAcceptedResponseCode.trim(),
      },
    },
    {
      id: 'a1000000-0000-4000-8000-00000000000a',
      name: 'Respond Accepted',
      type: 'n8n-nodes-base.respondToWebhook',
      typeVersion: 1.5,
      position: [980, -180],
      parameters: {
        respondWith: 'json',
        responseBody: '={{ JSON.stringify($json.response) }}',
        options: {
          responseCode: '={{ $json.http_status }}',
          responseHeaders,
        },
      },
    },
    {
      id: 'a1000000-0000-4000-8000-00000000000b',
      name: 'Build Acknowledgement',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [500, 40],
      parameters: {
        mode: 'runOnceForAllItems',
        language: 'javaScript',
        jsCode: buildAcknowledgementCode.trim(),
      },
    },
    {
      id: 'a1000000-0000-4000-8000-00000000000c',
      name: 'Respond Duplicate Stale or Conflict',
      type: 'n8n-nodes-base.respondToWebhook',
      typeVersion: 1.5,
      position: [740, 40],
      parameters: {
        respondWith: 'json',
        responseBody: '={{ JSON.stringify($json.response) }}',
        options: {
          responseCode: '={{ $json.http_status }}',
          responseHeaders,
        },
      },
    },
    {
      id: 'a1000000-0000-4000-8000-00000000000d',
      name: 'Workflow Failure Trigger',
      type: 'n8n-nodes-base.errorTrigger',
      typeVersion: 1,
      position: [-960, 460],
      parameters: {},
      notes: 'Catches automatic execution failures. A same-workflow Error Trigger does not need to be activated separately.',
    },
    {
      id: 'a1000000-0000-4000-8000-00000000000e',
      name: 'Sanitize Failure',
      type: 'n8n-nodes-base.code',
      typeVersion: 2,
      position: [-720, 460],
      parameters: {
        mode: 'runOnceForAllItems',
        language: 'javaScript',
        jsCode: sanitizeFailureCode.trim(),
      },
      notes: 'Drops headers, credentials, input payloads, and free-form error messages.',
    },
    {
      id: 'a1000000-0000-4000-8000-00000000000f',
      name: 'Persist Failure Summary',
      type: 'n8n-nodes-base.dataTable',
      typeVersion: 1.1,
      position: [-480, 460],
      parameters: {
        resource: 'row',
        operation: 'upsert',
        dataTableId: {
          __rl: true,
          mode: 'name',
          value: 'ml_workflow_failures',
          cachedResultName: 'ml_workflow_failures',
        },
        matchType: 'allConditions',
        filters: {
          conditions: [
            {
              keyName: 'failure_key',
              condition: 'eq',
              keyValue: "={{ $('Sanitize Failure').first().json.failure.failure_key }}",
            },
          ],
        },
        columns: resourceMapper(failureSchema, {
          failure_key: "={{ $('Sanitize Failure').first().json.failure.failure_key }}",
          failed_at: "={{ $('Sanitize Failure').first().json.failure.failed_at }}",
          workflow_id: "={{ $('Sanitize Failure').first().json.failure.workflow_id }}",
          workflow_name: "={{ $('Sanitize Failure').first().json.failure.workflow_name }}",
          execution_id: "={{ $('Sanitize Failure').first().json.failure.execution_id }}",
          failed_node: "={{ $('Sanitize Failure').first().json.failure.failed_node }}",
          error_type: "={{ $('Sanitize Failure').first().json.failure.error_type }}",
        }),
        options: {},
      },
      onError: 'continueRegularOutput',
    },
    {
      id: 'a1000000-0000-4000-8000-000000000010',
      name: 'Create Failure Notification',
      type: 'n8n-nodes-base.dataTable',
      typeVersion: 1.1,
      position: [-220, 460],
      parameters: {
        resource: 'row',
        operation: 'upsert',
        dataTableId: {
          __rl: true,
          mode: 'name',
          value: 'ml_failure_notifications',
          cachedResultName: 'ml_failure_notifications',
        },
        matchType: 'allConditions',
        filters: {
          conditions: [
            {
              keyName: 'alert_key',
              condition: 'eq',
              keyValue: "={{ $('Sanitize Failure').first().json.notification.alert_key }}",
            },
          ],
        },
        columns: resourceMapper(notificationSchema, {
          alert_key: "={{ $('Sanitize Failure').first().json.notification.alert_key }}",
          schema_version: "={{ $('Sanitize Failure').first().json.notification.schema_version }}",
          severity: "={{ $('Sanitize Failure').first().json.notification.severity }}",
          created_at: "={{ $('Sanitize Failure').first().json.notification.created_at }}",
          workflow_name: "={{ $('Sanitize Failure').first().json.notification.workflow_name }}",
          execution_id: "={{ $('Sanitize Failure').first().json.notification.execution_id }}",
          failed_node: "={{ $('Sanitize Failure').first().json.notification.failed_node }}",
          error_type: "={{ $('Sanitize Failure').first().json.notification.error_type }}",
          delivery_state: "={{ $('Sanitize Failure').first().json.notification.delivery_state }}",
        }),
        options: {},
      },
      onError: 'continueRegularOutput',
      notes: 'Durable sanitized notification outbox. Team-visible without storing secrets or raw records.',
    },
    {
      id: 'a1000000-0000-4000-8000-000000000011',
      name: 'Optional External Alert - Configure and Enable',
      type: 'n8n-nodes-base.httpRequest',
      typeVersion: 4.4,
      position: [40, 460],
      disabled: true,
      parameters: {
        method: 'POST',
        url: 'https://replace-with-self-hosted-notifier.invalid/ml-failures',
        authentication: 'genericCredentialType',
        genericAuthType: 'httpHeaderAuth',
        sendHeaders: true,
        headerParameters: {
          parameters: [
            { name: 'Accept', value: 'application/json' },
          ],
        },
        sendBody: true,
        contentType: 'json',
        specifyBody: 'json',
        jsonBody: "={{ JSON.stringify($('Sanitize Failure').first().json.notification) }}",
        options: {
          timeout: 5000,
          redirect: { redirect: { followRedirects: false } },
          response: { response: { neverError: false, responseFormat: 'json' } },
        },
      },
      credentials: {
        httpHeaderAuth: {
          id: 'REPLACE_WITH_FAILURE_NOTIFIER_CREDENTIAL_ID',
          name: 'ML Failure Notifier Auth',
        },
      },
      retryOnFail: true,
      maxTries: 2,
      waitBetweenTries: 2000,
      notes: 'Disabled by design. Replace the .invalid URL, bind a Header Auth credential, and enable only after approval. This is notification delivery, not business identity lookup.',
    },
  ],
  connections: {
    'Authenticated Experiment Webhook': {
      main: [[{ node: 'Validate and Normalize', type: 'main', index: 0 }]],
    },
    'Validate and Normalize': {
      main: [[{ node: 'Payload Valid?', type: 'main', index: 0 }]],
    },
    'Payload Valid?': {
      main: [
        [{ node: 'Get Experiment State', type: 'main', index: 0 }],
        [{ node: 'Respond Validation Error', type: 'main', index: 0 }],
      ],
    },
    'Get Experiment State': {
      main: [[{ node: 'Decide Event Handling', type: 'main', index: 0 }]],
    },
    'Decide Event Handling': {
      main: [[{ node: 'Should Persist?', type: 'main', index: 0 }]],
    },
    'Should Persist?': {
      main: [
        [{ node: 'Persist Experiment State', type: 'main', index: 0 }],
        [{ node: 'Build Acknowledgement', type: 'main', index: 0 }],
      ],
    },
    'Persist Experiment State': {
      main: [[{ node: 'Build Accepted Response', type: 'main', index: 0 }]],
    },
    'Build Accepted Response': {
      main: [[{ node: 'Respond Accepted', type: 'main', index: 0 }]],
    },
    'Build Acknowledgement': {
      main: [[{ node: 'Respond Duplicate Stale or Conflict', type: 'main', index: 0 }]],
    },
    'Workflow Failure Trigger': {
      main: [[{ node: 'Sanitize Failure', type: 'main', index: 0 }]],
    },
    'Sanitize Failure': {
      main: [[{ node: 'Persist Failure Summary', type: 'main', index: 0 }]],
    },
    'Persist Failure Summary': {
      main: [[{ node: 'Create Failure Notification', type: 'main', index: 0 }]],
    },
    'Create Failure Notification': {
      main: [[{ node: 'Optional External Alert - Configure and Enable', type: 'main', index: 0 }]],
    },
  },
  settings: {
    executionOrder: 'v1',
    timezone: 'UTC',
    saveDataErrorExecution: 'none',
    saveDataSuccessExecution: 'none',
    saveManualExecutions: false,
    saveExecutionProgress: false,
    executionTimeout: 30,
  },
  pinData: {},
};

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  writeFileSync(
    new URL('./workflow_1_experiment_tracking.json', import.meta.url),
    `${JSON.stringify(workflow, null, 2)}\n`,
    'utf8',
  );
}
