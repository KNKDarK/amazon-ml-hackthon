# 10K Source-1 ER pilot

This directory contains the first-stage pilot requested for the business entity
resolution task. It is deliberately isolated from any full-test processing.

## Run

From the workspace root:

```text
python pilot/run_pilot.py \
  --data-root dataset \
  --work-dir artifacts/pilot_10k_cap500_clean \
  --sample-size 10000 \
  --selection-cap 500 \
  --max-projected-postings 5000000
```

`--selection-cap` predeclares the production 500/500 matched blocking caps. The
validation split is used to report their recall and select model policy; the
held-out pilot-test split remains report-only. The process lock prevents two
pilots from deleting or rebuilding the same work directory concurrently.

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
- Runtime batches halve when system available memory drops below 2 GiB (Linux `/proc`, macOS/BSD `sysconf`, or Windows Win32 memory status).
- The script stops after the 10K pilot and writes projections only.

## Preprocessing and data quality

`text_normalization.py` is the preprocessing / data-quality layer that runs before
blocking and matching. It is standard library only, streaming, deterministic and
idempotent, and it never decides that two records are the same entity.

```text
python -m pilot.text_normalization \
  --input dataset/train/train_source1.tsv dataset/train/train_source2.tsv \
  --output-dir artifacts/normalized \
  --qa-json artifacts/normalized/data_quality.json
```

It emits, per row, the untouched `*_raw` values plus `*_display` (case preserved),
`*_key` (folded comparison form), `*_signature` (order-insensitive, **blocking
only - never a merge rule**), `*_blocking_tokens`, extracted numbers/postal codes,
legal forms, trade name, web token, `*_is_missing` booleans and `*_flags`.
Missing values are never imputed. Every rule is documented on the constant that
implements it and covered by `tests/test_text_normalization.py`; the measured
motivation is in `DATASET_INVENTORY.md`.

## Main artifacts

- `REPORT.md`: concise measured results.
- `pilot_report.json`: complete machine-readable metrics/resources.
- `blocking_stats.json`: per-block recall/frequency-cap analysis.
- `candidate_stats.json`: candidate volume and blocking recall.
- `evaluation.json`: held-out macro F0.5 and decision policy.
- `model.json`: feature names, normalization, weights, and training metadata.
- `frozen_pilot_model.json`: tracked byte-identical copy of the successful frozen model used by production inference.
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
bounded local replays. Checkpoint identity includes SHA-256 hashes of all three
input files, the model, and shared feature code, plus the mode and blocking
profile. Full mode rejects a nonzero query stride and refuses to render final
TSVs unless every Source-1 test row is complete and every match is present in
the candidate set.
