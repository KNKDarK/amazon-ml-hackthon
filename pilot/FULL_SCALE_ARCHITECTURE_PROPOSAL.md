# Proposed indexed blocking and bounded full-scale architecture

This is a proposal only. No full-test execution is authorized by this document.

## 1. Indexed blocking

Use an inverted index over normalized, complementary keys. Each posting is
stored on disk as `(key, source, entity_number/id, scheme)`; the pilot's exact
SQLite schema uses a `WITHOUT ROWID` primary key so deduplication does not
require a second candidate index.

### Name keys

1. Exact order-insensitive normalized name after legal-suffix removal.
2. First eight characters of that normalized name.
3. Each significant name token (country-partitioned).
4. Soundex for ASCII tokens; deterministic native-script prefix fallback for
   non-Latin scripts.

### Address keys

1. Each significant numeric token (ZIP/PIN/house-number evidence).
2. First house number plus each longer numeric token.
3. Each canonical address token after US/India abbreviation normalization.
4. Canonical adjacent address-token bigrams.

Country is a partition/equality key, not a closed-set model feature. This keeps
France records eligible while avoiding broad cross-country blocks.

### Frequency control

- Measure each key's posting count before query expansion.
- Drop blocks above an empirically selected frequency cap.
- Cap raw postings expanded per S1 query.
- Prefer the most selective retained keys; maximize labeled true-pair recall
  subject to a fixed projected candidate budget.
- Persist every union candidate, so `candidate_pairs.tsv` is exactly the set
  scored by the classifier.

The pilot chooses the cap from labeled 10K data. These caps must not be reused
blindly for test; test-frequency distributions and candidate volume should be
checked on a small approved test-side slice before a full run.

## 2. Disk-backed layout

For full scale, use country/scheme/key-range partitions rather than one
unbounded SQLite file. A practical layout is:

```text
work/
  index/<country>/<scheme>/<key-prefix>.sqlite
  candidates/<S1-shard>.sqlite
  features/<S1-shard>-00000.parquet
  scores/<S1-shard>.sqlite
  output/
```

SQLite is available in the standard library and is used successfully by the
pilot. DuckDB is optional; it would need to be installed/pinned and is not
required. SQLite is preferred here because setup is reproducible and each
posting lookup is bounded and indexed.

Important constraints:

- Never hold all S1 records, all target records, or all pair features at once.
- Do not use unrestricted `sklearn.cosine_similarity`.
- Use `IN` lists of at most a few hundred normalized keys per SQL query.
- Deduplicate in a disk-backed `(s1_id, target_id)` primary key.
- Use `journal_mode=OFF`, bounded cache, periodic commits, and explicit
  `MemAvailable` checks during index construction.
- Full-scale feature shards should be compressed (`float16`/Parquet or an
  equivalent compact binary) and may be deleted after scoring. The pilot's
  SQLite float32 BLOB layout is intentionally simple and auditable, not the
  final storage-efficiency recommendation.

## 3. Batch flow

1. Stream and normalize one source file at a time; write posting batches.
2. Build per-partition indexes and run `ANALYZE`/`PRAGMA optimize`.
3. Retrieve candidates for 2,000-5,000 S1 rows at a time, using bounded key
   lists and SQL result pages.
4. Write features for 5,000-20,000 pairs at a time, depending on live RAM.
5. If `MemAvailable < 2 GiB`, halve retrieval/feature batches, flush, and GC.
6. Train the compact CPU classifier on a representative, hard-negative sample;
   do not require all training candidates in RAM.
7. Score feature shards in order and write final IDs directly to the matching
   output stream.
8. Guarantee one output row for every test S1 ID, including empty singleton
   lists and all France entities.

## 4. Resource projection method

After the pilot, compute:

```text
query_scale  = 1,732,544 / 10,000
target_scale = (4,887,273 + 5,082,316) / (5,034,616 + 5,285,603)
work_scale   = query_scale * target_scale

projected_candidates = pilot_mean_candidates_per_S1 * 1,732,544
projected_index_postings = pilot_mean_keys_per_target * 9,969,589
```

Disk is estimated separately for index postings, compact candidate rows,
compressed feature shards, score rows, and text output. The linear runtime
estimate is a conservative lower-bound-style extrapolation, not a promise;
SQLite page cache behavior and target-key distribution can change it.

The measured values, pilot wall time, peak RSS, projected candidates, projected
disk, and whether the 52 GiB free-space margin is sufficient will be reported
in `artifacts/pilot_10k/REPORT.md` after the background run completes.
