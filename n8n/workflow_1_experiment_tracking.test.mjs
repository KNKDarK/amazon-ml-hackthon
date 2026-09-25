import assert from 'node:assert/strict';
import { workflow } from './workflow_1_experiment_tracking.mjs';

function node(name) {
  const found = workflow.nodes.find((candidate) => candidate.name === name);
  assert.ok(found, `Missing node: ${name}`);
  return found;
}

function validationItems(payload) {
  const code = node('Validate and Normalize').parameters.jsCode;
  const run = new Function('items', code);
  return run([
    {
      json: {
        headers: { authorization: 'Bearer synthetic-secret-must-not-propagate' },
        params: {},
        query: {},
        body: payload,
      },
    },
  ]);
}

function decisionItems(validated, existingRows) {
  const code = node('Decide Event Handling').parameters.jsCode;
  const validateOutput = [{ json: validated[0].json }];
  const getOutput = existingRows.length > 0 ? existingRows.map((row) => ({ json: row })) : [{}];
  const named = { 'Validate and Normalize': validateOutput };
  const dollar = (name) => ({ first: () => named[name][0] });
  const input = { first: () => getOutput[0] };
  const run = new Function('$', '$input', code);
  return run(dollar, input);
}

function stateFromRow(row) {
  return {
    schema_version: row.schema_version,
    job_id: row.job_id,
    last_event_id: row.event_id,
    last_event_version: row.event_version,
    occurred_at: row.occurred_at,
    received_at: row.received_at,
    status: row.status,
    runner_slot: row.runner_slot,
    ram_gb: row.ram_gb,
    experiment_name: row.experiment_name,
    code_revision: row.code_revision,
    config_hash: row.config_hash,
    dataset_fingerprint: row.dataset_fingerprint,
    f05: row.f05,
    precision: row.precision,
    recall: row.recall,
    candidate_count: row.candidate_count,
    runtime_seconds: row.runtime_seconds,
    peak_rss_mb: row.peak_rss_mb,
    artifacts_json: row.artifacts_json,
    event_fingerprint: row.event_fingerprint,
    payload_bytes: row.payload_bytes,
  };
}

function processEvent(payload, stateRows) {
  const validated = validationItems(payload);
  assert.equal(validated.length, 1);
  if (!validated[0].json.valid) return { validated, decision: null };

  const existing = stateRows.filter((row) => row.job_id === validated[0].json.event.job_id);
  const decision = decisionItems(validated, existing);
  assert.equal(decision.length, 1);
  const item = decision[0].json;

  if (item.action === 'persist') {
    const index = stateRows.findIndex((row) => row.job_id === item.row.job_id);
    const next = stateFromRow(item.row);
    if (index >= 0) stateRows[index] = next;
    else stateRows.push(next);
  }
  return { validated, decision };
}

const syntheticRunning = {
  schema_version: 1,
  event_id: 'evt-synthetic-job-001-v1',
  event_version: 1,
  job_id: 'job-synthetic-001',
  occurred_at: '2026-09-25T10:00:00Z',
  status: 'running',
  runner_slot: 'B',
  ram_gb: 16,
  experiment: {
    name: 'synthetic-logreg-v1',
    code_revision: 'abc1234',
    config_hash: 'a'.repeat(64),
    dataset_fingerprint: 'b'.repeat(64),
  },
  metrics: {
    candidate_count: 1200,
    runtime_seconds: 4.2,
    peak_rss_mb: 310,
  },
  artifacts: [
    {
      kind: 'report',
      ref: 'artifact:synthetic-report-001',
      sha256: 'c'.repeat(64),
      bytes: 2048,
    },
  ],
};

const stateRows = [];

const created = processEvent(syntheticRunning, stateRows);
assert.equal(created.decision[0].json.outcome, 'created');
assert.equal(created.decision[0].json.http_status, 202);
assert.equal(stateRows.length, 1);
assert.equal(stateRows[0].status, 'running');
assert.ok(created.decision[0].json.row.received_at);
assert.doesNotMatch(JSON.stringify(created), /synthetic-secret-must-not-propagate/);

const duplicate = processEvent(structuredClone(syntheticRunning), stateRows);
assert.equal(duplicate.decision[0].json.outcome, 'duplicate');
assert.equal(duplicate.decision[0].json.http_status, 200);
assert.equal(stateRows.length, 1);

const conflicting = processEvent({
  ...structuredClone(syntheticRunning),
  metrics: { ...syntheticRunning.metrics, runtime_seconds: 9.9 },
}, stateRows);
assert.equal(conflicting.decision[0].json.outcome, 'conflict');
assert.equal(conflicting.decision[0].json.http_status, 409);
assert.equal(stateRows.length, 1);

const completed = processEvent({
  ...structuredClone(syntheticRunning),
  event_id: 'evt-synthetic-job-001-v2',
  event_version: 2,
  occurred_at: '2026-09-25T10:05:00Z',
  status: 'completed',
  metrics: {
    f05: 0.8125,
    precision: 0.84,
    recall: 0.79,
    candidate_count: 1200,
    runtime_seconds: 4.2,
    peak_rss_mb: 310,
  },
}, stateRows);
assert.equal(completed.decision[0].json.outcome, 'updated');
assert.equal(completed.decision[0].json.http_status, 202);
assert.equal(stateRows.length, 1);
assert.equal(stateRows[0].last_event_version, 2);
assert.equal(stateRows[0].f05, 0.8125);

const stale = processEvent(syntheticRunning, stateRows);
assert.equal(stale.decision[0].json.outcome, 'stale');
assert.equal(stale.decision[0].json.http_status, 200);
assert.equal(stateRows[0].last_event_version, 2);

const oversized = validationItems({
  ...syntheticRunning,
  experiment: { ...syntheticRunning.experiment, name: 'x'.repeat(17 * 1024) },
});
assert.equal(oversized[0].json.http_status, 413);
assert.equal(oversized[0].json.response.reason, 'payload_too_large');

const rawBusinessData = validationItems({ ...syntheticRunning, records: [{ business_name: 'not accepted' }] });
assert.equal(rawBusinessData[0].json.http_status, 422);
assert.equal(rawBusinessData[0].json.response.reason, 'raw_business_data_forbidden');

const unexpectedField = validationItems({ ...syntheticRunning, notes: 'free text is not in schema' });
assert.equal(unexpectedField[0].json.http_status, 422);
assert.equal(unexpectedField[0].json.response.reason, 'unexpected_fields');

const wrongRam = validationItems({ ...syntheticRunning, runner_slot: 'A', ram_gb: 16 });
assert.equal(wrongRam[0].json.http_status, 422);
assert.equal(wrongRam[0].json.response.reason, 'runner_ram_mismatch');

const completedWithoutMetric = validationItems({ ...syntheticRunning, status: 'completed' });
assert.equal(completedWithoutMetric[0].json.http_status, 422);
assert.equal(completedWithoutMetric[0].json.response.reason, 'completed_metric_required');

const badArtifactRef = validationItems({
  ...syntheticRunning,
  artifacts: [{
    kind: 'report',
    ref: 'file:///home/user/private/report.json',
    sha256: 'd'.repeat(64),
    bytes: 100,
  }],
});
assert.equal(badArtifactRef[0].json.http_status, 422);
assert.equal(badArtifactRef[0].json.response.reason, 'invalid_artifact_metadata');

const failureCode = node('Sanitize Failure').parameters.jsCode;
const sanitizeFailure = new Function('$input', failureCode);
const sanitized = sanitizeFailure({
  first: () => ({
    json: {
      headers: { authorization: 'Bearer must-not-leak' },
      data: { records: [{ business_name: 'must-not-leak' }] },
      execution: {
        id: 'exec-synthetic-001',
        workflow: { id: 'wf-1', name: 'ML Team Experiment Tracking' },
      },
      lastNodeExecuted: 'Persist Experiment State',
      error: {
        name: 'NodeOperationError',
        message: 'Bearer must-not-leak raw business_name must-not-leak',
      },
    },
  }),
});
const sanitizedText = JSON.stringify(sanitized);
assert.match(sanitizedText, /exec-synthetic-001/);
assert.doesNotMatch(sanitizedText, /must-not-leak/);
assert.equal(sanitized[0].json.notification.delivery_state, 'pending');
assert.equal(Object.hasOwn(sanitized[0].json.notification, 'message'), false);

const mainWebhook = node('Authenticated Experiment Webhook');
assert.equal(mainWebhook.parameters.authentication, 'headerAuth');
assert.equal(mainWebhook.parameters.responseMode, 'responseNode');
assert.equal(workflow.settings.saveDataErrorExecution, 'none');
assert.equal(workflow.settings.saveDataSuccessExecution, 'none');
assert.equal(workflow.settings.executionTimeout, 30);
assert.equal(node('Optional External Alert - Configure and Enable').disabled, true);
assert.equal(workflow.nodes.filter((candidate) => candidate.type.includes('openAi') || candidate.type.includes('agent')).length, 0);
assert.equal(workflow.nodes.filter((candidate) => /leaderboard|training/i.test(candidate.name)).length, 0);

console.log('PASS workflow 1 synthetic tests');
console.log(`state_rows=${stateRows.length}`);
console.log('validated=created,duplicate,conflict,updated,stale');
console.log('rejected=oversized,raw_data,unknown_field,ram_mismatch,missing_metric,bad_artifact');
console.log('failure_sanitization=PASS');
