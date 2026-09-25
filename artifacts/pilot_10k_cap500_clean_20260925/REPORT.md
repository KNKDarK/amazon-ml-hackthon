# 10K Training S1 Pilot Report

## Scope

- Pilot queries: **10,000 uniformly sampled training S1 records**.
- Targets streamed: **10,320,219** training S2/S3 records.
- Test data processed: **0**.
- Primary score: macro F0.5 on the deterministic held-out pilot test split.
- No dense cross-source matrix was constructed.

## Blocking and candidates

- True-pair blocking recall: **93.8973%**.
- Candidate pairs: **2,386,317** (238.63/S1).
- Candidate reduction ratio: **99.997688%**.
- S1 rows with at least one candidate: **9,997**.
- Selected frequency cap: **500** target rows/block.
- Selected per-query posting cap: **500** raw postings/query.

## Matching model

- Model: compact CPU L2 logistic regression over 31 incremental name/address features.
- Validation macro F0.5: **79.6619%**.
- Held-out test macro F0.5: **79.7626%**.
- Policy: score threshold **0.99060**, at most **5** prediction(s)/S1.

## Resources

- Pilot wall time: **3954.2s**.
- Peak process RSS: **567.4 MiB**.
- Minimum system `MemAvailable`: **1592.0 MiB**.
- Pilot candidate DB: **257.7 MiB**.
- Pilot feature DB: **381.0 MiB**.

## Full-scale linear projection

- Candidate pairs: **413,439,920**.
- Naive scaled compute time: **183.84 h**.
- Working artifacts: **120.62 GiB**.
- Full inverted-index estimate: **7.90 GiB**.
- Projection assumes a **4 GiB** working-memory target and remains bounded batches; it is not authorization to run full scale.

See `pilot_report.json` for exact per-phase, per-scheme, and projection details.
