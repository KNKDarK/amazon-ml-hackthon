# 10K pilot error analysis

Only the existing labeled training pilot is analyzed. `validation` is the only split used to select policy; `test` is the deterministic held-out pilot split and is descriptive only. The official unlabeled test set was not scored.

## Pair level

| Split | Labeled true pairs | Candidate recall | TP | FP | FN | Macro F0.5 |
|---|---:|---:|---:|---:|---:|---:|
| validation | 5,093 | 91.7730% | 3,345 | 189 | 1,748 | 0.796989 |
| test | 5,255 | 92.3692% | 3,480 | 201 | 1,775 | 0.796470 |

For the baseline threshold 0.985 and top-K 5, all 11,906 misses decompose into 2,660 true pairs missing from blocking, 8,711 candidate true pairs scoring below threshold, and 535 above threshold but outside top-K. The top-K category is conditional on passing threshold; application keeps the first five eligible IDs by descending score and ID tie-break.

## Miss slices across all 10,000 queries

| Slice | True pairs | Missing candidate | Below threshold | Beyond top-K | Predicted |
|---|---:|---:|---:|---:|---:|
| source=S3 | 17,804 | 1,444 | 4,506 | 292 | 11,562 |
| source=S2 | 16,787 | 1,216 | 4,205 | 243 | 11,123 |
| country=us | 20,675 | 1,296 | 4,117 | 389 | 14,873 |
| country=india | 13,916 | 1,364 | 4,594 | 146 | 7,812 |
| split=test | 5,255 | 401 | 1,270 | 104 | 3,480 |
| split=train | 24,243 | 1,840 | 6,175 | 368 | 15,860 |
| split=validation | 5,093 | 419 | 1,266 | 63 | 3,345 |

This pilot sample has no empty business-name or address values among labeled positive source-1 queries, so those fields cannot explain positive-link misses here. Positive links necessarily come from non-singleton queries; singleton quality is therefore assessed through false-positive query outcomes rather than pair-level false negatives. The 1,500-query held-out pilot-test split has 78 labeled singletons, and 12 received at least one prediction (12 false-positive IDs); its remaining 66 singletons were correctly empty. Neither pilot query names nor addresses are missing (0/10,000 each). The stored detailed JSON has examples for qualitative review. The review shows the threshold rejection channel dominates; lowering it is not justified without a material validation-only macro F0.5 gain because precision is intentionally weighted.

## Controlled policy results already in the baseline

Validation selected threshold 0.985 / top-K 5 (macro F0.5 0.796989). At the same threshold, top-K ≥8 gives 0.796922; threshold 0.98 / top-K 5 gives 0.796781. These tiny differences do not justify replacing the frozen successful baseline. No test-split metric was used to choose a policy.
