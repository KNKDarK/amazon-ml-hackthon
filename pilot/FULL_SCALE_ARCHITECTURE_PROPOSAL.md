# Implemented full-scale indexed inference architecture

The production path is implemented in `pilot/stream_infer.py`. It is disk-backed,
restartable, and bounded in memory. It does not load a source table or construct a
dense cross-source matrix.

## 1. Indexed blocking

A SQLite inverted index stores normalized blocking keys and target row numbers.
The same canonicalization functions are used in the labeled pilot and test
inference.

### Name keys

1. Exact order-insensitive normalized name after legal-suffix removal.
2. First eight characters of that normalized name.
3. Each significant name token.
4. Soundex for ASCII tokens and a deterministic native-script fallback.

### Address keys

1. Significant numeric address tokens.
2. First house number plus longer numeric tokens.
3. Canonical address tokens after US/India abbreviation normalization.
4. Canonical adjacent address-token bigrams.

Country is an open string used in key partitioning and equality features. The
code does not restrict test records to the training countries, so France remains
eligible.

### Frequency control

- SQLite `keyfreq` stores posting-list sizes.
- Keys whose frequency exceeds `--block-cap` are excluded.
- Raw postings expanded for one S1 query are limited by
  `--query-posting-cap`.
- More selective keys are expanded first.
- The persisted candidate table is the exact union scored by the classifier.

The production model is rebuilt with a predeclared 500/500 cap. Its selection
uses validation labels only; the pilot-test split is report-only.

## 2. Disk-backed state

```text
work/
  checkpoint.json
  index.sqlite
  index.sqlite-wal
  index.sqlite-shm
  results.sqlite
  results.sqlite-wal
  results.sqlite-shm
  preflight_report.json
output/
  matching_results.tsv
  candidate_pairs.tsv
```

The index contains `targets`, `postings`, and `keyfreq` tables. The result
database contains `queries`, `candidates`, `pairs`, and `completed` tables.
Primary/unique constraints enforce one target per query and deterministic
pair uniqueness.

SQLite uses a bounded page cache, explicit busy timeout, full durability by
default, and a rollback-journal fallback when a local filesystem cannot provide
WAL semantics. Network and synchronized filesystems are not supported for live
work directories.

## 3. Bounded batch flow

1. Verify all input paths, model schema, work-directory state, and resource
   reserve.
2. Stream Source 2 and Source 3 once, normalizing target feature text and
   committing index batches transactionally.
3. Build target key frequencies after indexing.
4. Stream Source 1; retrieve a bounded posting set for each query.
5. Compute the 31 pair features only for that query's candidates.
6. Apply the frozen logistic model, validation-selected threshold, and top-K
   policy.
7. Commit completed query IDs in bounded transactions.
8. Assert full Source-1 coverage and match/candidate consistency.
9. Render both TSVs through temporary files and atomic replacement.

The process uses a small amount of memory relative to the data size. Candidate
IDs and features are never accumulated in a Python list across all queries.

## 4. Restart and integrity rules

Checkpoint identity includes:

- logical input names, byte sizes, and SHA-256 content hashes;
- model SHA-256;
- shared feature-code SHA-256;
- preflight/full mode and bounded-query setting;
- target sampling, query stride, block cap, and query posting cap.

Before trusting state, the program verifies the required database exists and
contains its required tables. A missing result database cannot be silently
skipped. A changed model, dataset, feature code, mode, or profile requires a new
work directory.

A query-free legacy index can be migrated only after its input size/mtime
fingerprint and profile are verified; stored target fields are canonicalized
before any query result is produced.

## 5. Output guarantees

- Exact UTF-8-without-BOM TSV headers.
- LF line endings on every platform.
- One row for every test Source-1 entity in full mode.
- Only known Source-2/Source-3 IDs.
- No duplicate IDs inside a list.
- Every accepted match exists in the persisted candidate set.
- Atomic final-file replacement.

The local validator performs the final coverage, duplicate, prefix, ID-existence,
and candidate-subset checks before packaging.

## 6. Resource guard

The measured 10K validation replay estimated approximately 29 GiB peak working
storage for cap 500. The production CLI defaults to preserving 20 GiB free and
stops before a batch or final output can cross that reserve. `--safety-free-gib
0` exists only for synthetic unit-test fixtures.
