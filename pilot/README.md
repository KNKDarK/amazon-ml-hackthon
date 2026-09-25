# 10K Source-1 ER pilot

This directory contains the first-stage pilot requested for the business entity
resolution task. It is deliberately isolated from any full-test processing.

## Run

From the workspace root:

```bash
python3 pilot/run_pilot.py \
  --data-root dataset \
  --work-dir artifacts/pilot_10k \
  --sample-size 10000
```

The only third-party runtime dependency is NumPy. Python's standard-library
SQLite is used for disk-backed candidates, float32 feature batches, and scores.

## Safety properties

- No dense cross-source similarity matrix.
- No pandas/full-table load.
- Test files are not read.
- Each training target TSV is streamed once for block-frequency measurement
  and once for candidate persistence.
- S1 pilot records are sampled with a fixed-seed reservoir.
- Block frequency and per-query posting caps bound candidates.
- Pair features are computed in bounded batches and saved as float32 BLOBs.
- The classifier is a compact CPU logistic regression.
- Runtime batches halve when Linux `MemAvailable` drops below 2 GiB.
- The script stops after the 10K pilot and writes projections only.

## Main artifacts

- `REPORT.md`: concise measured results.
- `pilot_report.json`: complete machine-readable metrics/resources.
- `blocking_stats.json`: per-block recall/frequency-cap analysis.
- `candidate_stats.json`: candidate volume and blocking recall.
- `evaluation.json`: held-out macro F0.5 and decision policy.
- `model.json`: feature names, normalization, weights, and training metadata.
- `pilot_queries.tsv`, `pilot_labels.tsv`: fixed validation sample and labels.
- `pilot_candidates.sqlite`: disk-backed candidate pairs.
- `pilot_features.sqlite`: disk-backed pair-feature batches.
- `pilot_scores.sqlite`: batched model scores.
- `pilot_candidate_pairs.tsv`, `pilot_matching_results.tsv`: pilot predictions.

## Production output location

`stream_infer.py` writes the final `matching_results.tsv` and
`candidate_pairs.tsv` to `output/` by default (or to the directory supplied with
`--output-dir`). Resumable SQLite state, indexes, and logs remain in `--work-dir`;
the final TSVs are not limited to that work directory. The index writer defaults
to SQLite `FULL` durability; `--index-synchronous OFF` is intended only for
bounded local replays.
