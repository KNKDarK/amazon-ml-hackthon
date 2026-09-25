# Production blocking-cap validation — labeled 10K pilot replay

**Run date:** 2026-09-25  
**Scope:** candidate generation only; no test inference was started.

## Method

- Replayed the **1,483 `validation` S1 queries** from the frozen pilot (`pilot_queries.tsv`) against all **10,320,219 training S2/S3 target rows**.
- Labels contain **5,093 true links** across those queries.
- Used `pilot/stream_infer.py`, the production SQLite postings/key-frequency retrieval and candidate/scoring path, with `target_sample_rate=1`.
- For each requested value, both production knobs were set to that value: `--block-cap=C --query-posting-cap=C`.
- The replay index streamed the complete target corpus. To avoid persisting keys that cannot be reached by this fixed validation query set, it retained postings for all 17,462 validation-query keys; key frequencies and candidate sets for those queries are equivalent to the full production index. It retained 10,285,461 target rows and 71,755,119 postings (the omitted rows/keys are unreachable by these fixed queries). The production query/candidate code was unchanged.
- The frozen model was read-only: `artifacts/pilot_10k_recovery_manual_20250925/model.json`, SHA-256 `8d2dec027e7bbdc0aaba64351d7375492ed9b67f962824a186a6f2647adfd9b6`. No training or model write occurred.
- P95 below is the pilot-style linearly interpolated percentile, with zero-candidate queries included.

## Measured results

| Raw posting cap | True-link candidate recall | Hits / true links | Average candidates / S1 | P95 candidates / S1 | Total candidate pairs |
|---:|---:|---:|---:|---:|---:|
| **100** | **88.73%** | 4,519 / 5,093 | 40.89 | 84.0 | 60,638 |
| **500** | **93.36%** | 4,755 / 5,093 | 240.36 | 432.9 | 356,457 |
| **1,000** | **94.74%** | 4,825 / 5,093 | 502.31 | 894.9 | 744,930 |

The cap is charged against **raw postings before target-ID deduplication**, which is why the unique candidate averages are below 100/500/1,000. The measured query-plus-output phases were 12.3 s, 67.5 s, and 140.4 s for caps 100, 500, and 1,000 respectively (120.2, 22.0, and 10.6 validation queries/s).

### Cap-semantics note

The production CLI has separate block-frequency and query-posting knobs. The main table varies both together. If “posting cap” is intended to mean **only** `--query-posting-cap` while retaining the preflight default `--block-cap=100`, the measured validation sensitivity is:

| Fixed block cap | Query-posting cap | Recall | Average / S1 | P95 / S1 | Pairs |
|---:|---:|---:|---:|---:|---:|
| 100 | 500 | 90.24% | 72.97 | 189.9 | 108,218 |
| 100 | 1,000 | 90.24% | 72.98 | 189.9 | 108,234 |

Thus the large recall gains in the main 500/1,000 rows come from lifting the frequency cap as well as the query-posting cap. The recommendation and launch command below use the matched-cap interpretation (`500/500`).

## Full-test resource projection

These are planning estimates, not a full-test run. They scale the measured validation query workload by:

- S1: `1,732,544 / 1,483 = 1,168.27x`
- target corpus: `9,969,589 / 10,320,219 = 0.9660x`
- combined work scale: `1,128.58x`

Storage is a peak working estimate: the conservative production index basis (10.53 GiB), scaled result SQLite, and both temporary and final TSV copies. Runtime uses the previously measured full production-index basis (8,028 s) plus 1.25x-scaled query/output time. Test-corpus distribution and SQLite I/O can move these numbers materially.

| Cap | Projected full-test candidate pairs | Projected peak working storage | Linear runtime basis | Planning range |
|---:|---:|---:|---:|---:|
| **100** | 68,434,694 | **14.16 GiB** | **7.06 h** | **8–12 h** |
| **500** | 402,289,420 | **29.23 GiB** | **28.66 h** | **32–43 h** |
| **1,000** | 840,711,383 | **48.97 GiB** | **57.24 h** | **63–86 h** |

The current free space after cleanup is about 68.5 GiB. The 1,000-cap projection plus the production 20-GiB safety reserve is about 69 GiB, so it remains too close to (and slightly over) the available margin without a different storage layout or more free space. The 500-cap projection remains below that guard with roughly 19 GiB of headroom.

## Difference from the original pilot blocking logic

1. The original pilot selected a **250 block-frequency cap and 250 raw-posting query cap** under a 2,000,000 projected-posting budget; it did not select 100/500/1,000. Its full-10K candidate recall was 92.31%; restricted to this validation split it was **91.77%**, 171,488 pairs, 115.64 average candidates, and P95 216.9.
2. The original pilot optimized cap pairs using its sampled-query key index and a global projected-posting budget. Production applies fixed caps at retrieval time and has no analogous pilot cap-selection budget.
3. The original frequency pass counted keys present in the 10K query-key index; production builds target key frequencies from the target index. At `target_sample_rate=1`, frequencies for keys used by these validation queries are the same; the replay index is a storage optimization, not a blocking-rule change.
4. The original pilot persisted a block mask and used `(qrow,target)` deduplication. Production persists `(key,rid)` postings and unique `(qid,target)` candidate rows; it does not retain the block mask. Candidate recall is measured before model threshold/top-K filtering.
5. Production's fixed-cap replay is a set-level comparison against the validation queries, not a claim that the original pilot was rerun with those three cap pairs. Matching-file tie ordering can also differ at an equal-score boundary; that does not affect candidate recall.

## Recommendation

**Recommend cap 500 for both production knobs.** It raises validation candidate recall from 88.73% to **93.36%** (+4.63 percentage points) and exceeds the original pilot's validation-split blocking recall, while remaining below the storage guard. Cap 1,000 adds only **+1.37 points** over cap 500 but approximately doubles runtime and raises projected peak storage to about 49 GiB; it is not justified by the measured recall gain under the current resource guard. Cap 100 is faster, but its 4.63-point recall loss relative to cap 500 is too large to ignore.

## Exact Fish-compatible launch command (not run)

This launches the **test** full run with the recommended cap. It was deliberately not executed.

```fish
tmux new-session -d -s amazon-ml-inference-cap500 'fish -lc "cd /home/knk/ml/student_resource; and mkdir -p artifacts/full_inference_20260925_cap500 output; and /usr/bin/python3 -u pilot/stream_infer.py --data-root dataset/test --work-dir artifacts/full_inference_20260925_cap500 --output-dir output --model artifacts/pilot_10k_recovery_manual_20250925/model.json --mode full --queries 0 --target-sample-rate 1 --query-stride 1 --index-batch 5000 --index-synchronous FULL --block-cap 500 --query-posting-cap 500 >> artifacts/full_inference_20260925_cap500/run.log 2>&1"'
```

## Output-path confirmation

`pilot/stream_infer.py` now defaults `--output-dir` to `output` and writes both final TSVs atomically there. With the command above, the production files are exactly:

- `/home/knk/ml/student_resource/output/matching_results.tsv`
- `/home/knk/ml/student_resource/output/candidate_pairs.tsv`

The work directory contains the index, result database, checkpoint, and log; it is not the sole destination for final TSVs. The cap-replay TSVs are intentionally separate under `cap_100/output`, `cap_500/output`, and `cap_1000/output`. The root `output/` files do not exist yet because no full test run was started.
