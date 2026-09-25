# ML Challenge 2026: Business Entity Resolution Solution

- **Team Name:** KNK Dark
- **Team Members:** KNKDarK — integration, pipeline validation, and final submission
- **Submission Date:** 2026-09-25

## 1. Executive Summary

The solution is a disk-backed hybrid entity-resolution pipeline. Multiple
normalized name/address blocking keys generate a bounded candidate set, and a
compact 31-feature logistic classifier selects final Source-2/Source-3 matches.
The production path streams all records, uses SQLite only for bounded indexed
state, supports transactional resume, and runs with the same Python code on
Windows 11, macOS, and Linux.

The final model was rebuilt after two correctness controls were added: pilot and
production now use exactly the same canonical target representation, and the
500/500 blocking cap is predeclared rather than selected using pilot-test labels.
Only challenge-provided training data was used.

## 2. Data and Compliance

### 2.1 Data inventory

| Dataset | Source 1 | Source 2 | Source 3 |
|---|---:|---:|---:|
| Training | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

Training ground truth contains one row per training Source-1 entity. Test adds
France, which is absent from training. Country is handled as an open normalized
string; there is no fixed-country filter.

No external database, API, geocoder, business registry, web lookup, pretrained
model, or outside identity augmentation was used. The model was trained only on
the supplied labeled training TSVs and is distributed under the MIT License.

### 2.2 Text handling

All source and output files are streamed as UTF-8 TSVs with explicit tab
separation. The readers accept a UTF-8 BOM for cross-platform spreadsheet
exports, while all generated files use UTF-8 without a BOM and LF line endings.
Names, addresses, and country labels are processed as strings; commas are data,
not delimiters.

## 3. Normalization and Canonicalization

The normalizer performs Unicode NFKD case folding, Latin accent folding while
preserving non-Lingual marks, punctuation removal, and tokenization. It:

- maps common US/India address abbreviations;
- removes or normalizes common legal-form terms;
- creates order-insensitive name and address signatures;
- extracts significant numeric tokens and house/postal evidence;
- computes deterministic Soundex values for ASCII tokens and a stable fallback
  for other scripts.

Source-1 records retain their normalized token order for pair features. Target
records use the same canonical target fields in training and production. This
shared `canonical_target_fields` path prevents the training/inference feature
drift found in the earlier prototype.

## 4. Candidate Generation

### 4.1 Blocking keys

The inverted index uses eight complementary schemes:

1. exact normalized full name;
2. first eight characters of the normalized name;
3. significant name tokens;
4. name Soundex;
5. significant address numbers;
6. first house number plus longer numeric token;
7. canonical address tokens;
8. adjacent canonical address-token bigrams.

Country is embedded in keys as an open-value partition, preventing broad
cross-country blocks without excluding France.

### 4.2 Frequency control

A target-side SQLite index stores `(key, target_row_id)` postings and key
frequency counts. For every Source-1 query, posting lists are expanded from the
most selective key first. A key is rejected above the predeclared 500-target
frequency cap, and no query expands beyond 500 raw postings.

The persisted `candidates` table is the exact deduplicated union scored by the
classifier. Final accepted matches are required to be a subset of this set.

### 4.3 Clean pilot blocking result

| Metric | Validation | Held-out pilot test | All 10K descriptive |
|---|---:|---:|---:|
| True pairs | 5,093 | 5,255 | 34,591 |
| Candidate recall | **93.3634%** | **94.0247%** | **93.8973%** |
| Candidate pairs | — | — | **2,386,317** |
| Mean candidates / S1 | — | — | **238.63** |
| Reduction vs. all comparisons | — | — | **99.997688%** |

The 500/500 cap was fixed before the clean pilot. Validation labels were used to
report its recall; the held-out pilot-test split was not used to select the cap
or model policy.

## 5. Matching Model

### 5.1 Features

The classifier uses 31 float32 features:

- bias and same-country equality;
- exact, prefix, token Jaccard/Dice/containment, character 3/4-gram Dice,
  length ratio, and Soundex overlap for names;
- exact, prefix, token Jaccard/Dice/containment, character 3/4-gram Dice,
  length ratio, house-number equality, postal equality, and numeric overlap for
  addresses;
- name/address agreement summaries, both-exact interaction, missing-field
  indicators, and token-count difference.

### 5.2 Training

A fixed-seed reservoir sampled 10,000 Source-1 training rows. Stable hashing
split them into 7,017 training, 1,483 validation, and 1,500 held-out pilot-test
queries. Training used at most 30 deterministic hard negatives per positive
candidate. The compact L2 logistic model was implemented in NumPy with 12
mini-batch Adam epochs, learning rate 0.03, batch size 4,096, and class-
balanced loss.

The final weights contain 31 coefficients plus normalization statistics, far
below the challenge's 8-billion-parameter limit.

### 5.3 Decision policy

Threshold and top-K were selected only on validation macro F0.5. A coarse search
was followed by a 0.0001-resolution probability search in the high-precision
region. The selected policy is:

- **probability threshold:** 0.9906;
- **maximum predictions per Source-1:** 5;
- deterministic tie break: descending score, then ascending target ID.

A full 2,386,317-row audit comparing pilot scalar scoring with production batch
scoring found zero threshold-classification changes and zero top-5 set changes.
Maximum probability difference was `8.93e-08`.

## 6. Validation Results and Error Analysis

| Split | TP | FP | FN | Macro F0.5 | Micro precision | Micro recall |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 3,522 | 320 | 1,571 | **0.796619** | 0.916710 | 0.691537 |
| Held-out pilot test | 3,669 | 340 | 1,586 | **0.797626** | 0.915191 | 0.698192 |

Across all 10,000 queries, 2,111 true pairs are absent from candidate generation,
7,996 candidate pairs score below 0.9906, and 641 pass the threshold but fall
outside top 5. Threshold rejection is the dominant controlled error channel.
The sample contains 561 true singletons; 387 receive the correct empty result and
174 receive at least one false prediction.

The official unlabeled test set has not been used to fit, select, or report model
performance. Leaderboard results, if later available, are not training signals.

## 7. Production Architecture

### 7.1 Streaming and storage

`pilot/stream_infer.py` performs four bounded stages:

1. stream Source 2/3 and commit normalized targets/postings in batches;
2. build key frequencies and mark the index complete;
3. stream Source 1, retrieve at most 500 postings, compute features, score, and
   commit query/candidate/match rows transactionally;
4. assert complete Source-1 coverage and match/candidate consistency, then write
   both final TSVs through temporary files and atomic replacement.

No source table, feature matrix, or candidate list is accumulated globally in
Python. The design is intended for a 16 GB laptop and uses SQLite page caches
and commits to bound memory.

### 7.2 Restart safety

Checkpoint version 2 binds:

- logical input names, byte sizes, and SHA-256 content hashes;
- model SHA-256;
- production pipeline code SHA-256;
- shared feature code SHA-256;
- preflight/full mode and query limit;
- target sampling, query stride, block cap, and query posting cap.

Before resume, the program verifies required database tables and rejects missing
state. Full mode rejects a nonzero query stride and refuses final rendering
unless all 1,732,544 Source-1 rows are complete. Process locks use `fcntl` on
POSIX and `msvcrt` on Windows.

### 7.3 Resource controls

The clean pilot peaked at 567.4 MiB process RSS. The cap-500 production plan
uses approximately 29.23 GiB peak working storage and preserves a 20 GiB free
disk reserve. SQLite work state must reside on a local filesystem with reliable
locking; SMB, NFS, and actively synchronized cloud directories are unsupported.

## 8. Reproducibility

Supported runtime:

- Python 3.12 or newer;
- NumPy 2.5.3;
- Python standard-library SQLite and CSV.

Install:

```text
python -m pip install -r requirements_lock.txt
```

Run tests:

```text
python -m compileall -q pilot utils tests tools
python -m unittest discover -s tests -v
```

The test suite covers feature parity, validation-only cap selection, lock
contention, module/package imports, deterministic LF output, a tiny full
end-to-end run, unsafe-resume rejection, local validation, and ZIP assembly.
GitHub Actions runs the same package installation and tests on Ubuntu, Windows,
and macOS.

Full inference:

```text
python -u pilot/stream_infer.py \
  --data-root dataset/test \
  --work-dir artifacts/full_inference_cap500 \
  --output-dir output \
  --model pilot/frozen_pilot_model.json \
  --mode full --queries 0 --target-sample-rate 1 --query-stride 1 \
  --index-batch 5000 --index-synchronous FULL \
  --block-cap 500 --query-posting-cap 500
```

Validation:

```text
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test \
  --check-ids
```

## 9. Outputs and Packaging

`matching_results.tsv` contains exactly one row per test Source-1 entity and the
accepted Source-2/Source-3 IDs. `candidate_pairs.tsv` contains the exact
candidate set scored by the model. Both are tab-separated, UTF-8 without BOM,
and use LF line endings.

`tools/build_submission.py` validates both files, rejects warnings, excludes CI,
raw data, live SQLite state, logs, and caches, adds source/model licenses and a
SHA-256 manifest, and writes the required
`<team_name>_submission.zip` structure.

## 10. Limitations and Fairness

- Candidate frequency caps impose a hard recall ceiling.
- The classifier is precision-oriented and can miss noisy or partial links.
- Country and address patterns differ between India, the US, and unseen France;
  France remains open-valued but has no labeled training examples.
- The 10K pilot is much smaller than the full corpus, so production calibration
  should be monitored through preflight and validation artifacts, never through
  test labels.
- This model is intended for the supplied challenge data, not general-purpose
  identity verification.

## Appendix A: Code Layout

The final archive places all source under:

```text
code/business_entity_resolution/src/pilot/
code/business_entity_resolution/src/utils/
```

`pilot/stream_infer.py` is the production entry point;
`pilot/run_pilot.py` rebuilds the labeled pilot; `utils/validate_submission.py`
checks final files. The frozen MIT model is
`src/pilot/frozen_pilot_model.json`.

## Appendix B: Key Artifacts

- `artifacts/pilot_10k_cap500_clean_20260925/REPORT.md`
- `artifacts/pilot_10k_cap500_clean_20260925/evaluation.json`
- `artifacts/pilot_10k_cap500_clean_20260925/blocking_stats.json`
- `artifacts/pilot_10k_cap500_clean_20260925/candidate_stats.json`
- `artifacts/blocking_cap_validation_20260925/REPORT.md`
- `error_analysis.md`
- `MODEL_LICENSE.md`
