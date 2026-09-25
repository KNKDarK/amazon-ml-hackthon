# Dataset and resource inventory

Measured on 2026-09-25 without loading full tables.

## Disk and RAM

- Workspace filesystem: `/dev/nvme0n1p2`, 234.46 GiB total, 180.06 GiB used,
  **51.44 GiB available** (78% used).
- Dataset logical bytes: **2,520,573,701 bytes (2.35 GiB)**.
- Physical RAM: 14 GiB; at pilot launch Linux reported about **8.0 GiB
  `MemAvailable`**.
- Swap: 14 GiB total, about 12 GiB free at launch.
- A full streaming profile used about 159 MiB peak RSS. The pipeline target
  remains 4 GiB working memory and must react if `MemAvailable < 2 GiB`.

| File | Logical bytes | Data rows |
|---|---:|---:|
| `train_source1.tsv` | 210,069,713 | 2,206,821 |
| `train_source2.tsv` | 489,301,488 | 5,034,616 |
| `train_source3.tsv` | 503,705,637 | 5,285,603 |
| `train_ground_truth.tsv` | 127,015,583 | 2,206,821 |
| `test_source1.tsv` | 175,022,086 | 1,732,544 |
| `test_source2.tsv` | 509,456,422 | 4,887,273 |
| `test_source3.tsv` | 506,002,772 | 5,082,316 |

Exact byte values may differ slightly from apparent `du` sizes because the table
uses logical file lengths.

## Schema

All source files are UTF-8 TSV with this exact header:

```text
entity_id	business_name	business_address	country
```

All four columns are represented as strings during streaming. IDs carry an
`S1-`, `S2-`, or `S3-` prefix. Commas are data, so every reader explicitly uses
tab separation and no quoting.

The training label file is:

```text
source1_entity_id	matched_entity_ids
```

`matched_entity_ids` is a comma-separated list and may be empty. Streaming
counts show one ground-truth row per training S1 record.

## Training profile

| File | US | India | Missing name | Missing address | Mean name chars | Mean address chars |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 1,323,633 | 883,188 | 0 | 0 | 24.0 | 52.1 |
| S2 | 3,016,817 | 2,017,799 | 0 | 168,967 | 25.1 | 46.2 |
| S3 | 3,170,056 | 2,115,547 | 0 | 175,916 | 25.2 | 46.7 |

Test additionally contains `France`, which is absent from training. The final
pipeline must not use a fixed `{US, India}` filter or categorical whitelist.
Country will be used as an open string and primarily as a partitioning/equality
feature.

A deterministic 10,000-row diagnostic sample had 34,684 true links (16,847 to
S2 and 17,837 to S3), 9,443 S1 entities with at least one match, and 557
singletons. The executable pilot uses its own fixed-seed uniform reservoir; its
exact sample counts are recorded in `artifacts/pilot_10k/pilot_report.json`.
