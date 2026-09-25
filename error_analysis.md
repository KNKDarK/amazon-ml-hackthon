# Clean 10K cap-500 pilot error analysis

This report replaces the earlier cap-250 analysis. The earlier blocking-cap
selection included labels from the nominal held-out pilot-test split, and the
production target feature representation did not exactly match the pilot. Both
issues were corrected before this run. The current 500/500 cap was predeclared;
the validation split alone selected model policy, and pilot-test results are
report-only. The official unlabeled test set has not been scored.

## Split-level results

| Split | Queries | True pairs | Candidate recall | TP | FP | FN | Macro F0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| validation | 1,483 | 5,093 | 93.3634% | 3,522 | 320 | 1,571 | **0.796619** |
| pilot test | 1,500 | 5,255 | 94.0247% | 3,669 | 340 | 1,586 | **0.797626** |

The validation-selected policy is probability threshold **0.9906** with at most
**5** predictions per Source-1 entity. Validation micro precision/recall are
0.916710/0.691537. The held-out pilot-test micro precision/recall are
0.915191/0.698192.

## Miss decomposition

| Split | Missing from candidates | Candidate present but below threshold | Above threshold but outside top 5 |
|---|---:|---:|---:|
| validation | 338 | 1,160 | 73 |
| pilot test | 314 | 1,145 | 127 |
| all 10K descriptive | 2,111 | 7,976 | 641 |

Across all 10,000 queries, blocking retains 32,480 of 34,591 labeled true pairs
(**93.8973%**). Of the 10,728 missed true pairs, 2,111 are absent from blocking,
7,976 have a candidate score below 0.9906, and 641 pass the threshold but fall
outside the five-prediction cap. Threshold rejection is therefore the largest
controlled error channel; a lower threshold trades more false merges for recall
and did not improve validation macro F0.5.

## Singleton behavior

The 10K sample contains 561 true singleton queries. The final policy correctly
returns an empty list for 387 and falsely predicts at least one target for 174.
On the validation split, 65 of 90 singletons are correct; on pilot test, 57 of
78 are correct. This is a material precision risk because singleton false
positives receive zero entity-level F0.5 credit.

## Resource observations

- Candidate pairs: **2,386,317**, or 238.63 per Source-1 query.
- Candidate reduction versus all Source-1 × target comparisons: **99.997688%**.
- Peak process RSS: **567.4 MiB**.
- Pilot wall time: **3,954.2 seconds**.
- Feature and score database row counts both exactly match the candidate count.

No leaderboard result or unlabeled-test score was used to choose the threshold,
top-K cap, blocking cap, features, or model parameters.
