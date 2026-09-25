# Phase 1–5 audit and readiness report

> **Historical snapshot (2026-09-25):** This file preserves the original
> pre-validation audit and contains statements that are intentionally superseded
> (for example, the earlier lack of Git metadata and readiness assessment). The
> current implementation, portability checks, runbooks, and final-delivery status
> are tracked in the repository README and runbooks. Do not use the resource
> projections in this snapshot as final-run measurements.

Audit date: 2026-09-25 (Asia/Kolkata)  
Project: `student_resource`  
Baseline: `artifacts/pilot_10k_recovery_manual_20250925`

## Scope and status

This report covers input inventory, available competition instructions, the successful 10K pilot, validation/evaluation behavior, resource observations, error-analysis artifacts, and full-run readiness. The test set has no labels and was not used for fitting, threshold choice, or score evaluation. No dense all-pairs comparison or external identity lookup has been used. Full-scale test inference has not been started.

## Competition instructions and compliance

The project `README.md` states that each test S1 needs exactly one output row; IDs must be test S2/S3 IDs; an empty list represents no match; candidate rows must represent the final set scored by the classifier; predictions must be a subset of candidates. It documents macro F0.5, the local validator command, a required ZIP layout, and model constraints of MIT/Apache 2.0 license and no more than 8B parameters. Its Academic Integrity section prohibits external databases, APIs, geocoding and outside data augmentation for identity resolution. These instructions have been transcribed in the supplied competition README and match the public challenge overview [repository overview](https://github.com/Sugandh-vI/Amazon-ML-challenge) and the challenge listing [Amazon ML Challenge 2026](https://hiretoday.in/competitiondetails/40000186).

AI coding-assistant permission is **unresolved**. The public challenge listing permits Python, ML frameworks and data science tools, but does not explicitly say whether generative AI/coding agents may be used. The organizer-hosted rule document was not present in this project directory and no authoritative AI-assistant clause could be confirmed from accessible official material. Before representing the submission as compliant on this point, check the rules downloaded in the team portal or ask the organizer. This audit has made no external business-identity queries and has not sent challenge data to an external data-enrichment service.

## Dataset inventory

Verified by file line count (header included) and TSV header inspection. Counts match the requested inventory.

| File | Data rows | Schema |
|---|---:|---|
| `dataset/train/train_source1.tsv` | 2,206,821 | `entity_id, business_name, business_address, country` |
| `dataset/train/train_source2.tsv` | 5,034,616 | same |
| `dataset/train/train_source3.tsv` | 5,285,603 | same |
| `dataset/train/train_ground_truth.tsv` | 2,206,821 | `source1_entity_id, matched_entity_ids` |
| `dataset/test/test_source1.tsv` | 1,732,544 | same source schema |
| `dataset/test/test_source2.tsv` | 4,887,273 | same source schema |
| `dataset/test/test_source3.tsv` | 5,082,316 | same source schema |

All files use tab delimiters. Test includes France, absent from train; country handling in normalization is open-valued. SHA-256 hashes are recorded in `baseline_manifest.json` for all seven input files.

## Code and baseline review

The baseline is a deterministic 10K training-S1 pilot with a stable hash split into train/validation/pilot-test portions (7,017 / 1,483 / 1,500 queries). It streams training targets and labels, builds disk-backed candidates, computes 31 pair features, trains a compact NumPy logistic classifier and selects threshold/top-K using validation only. The frozen successful artifacts report 1,145,685 candidate rows and features, 92.3101% pilot blocking pair recall, validation macro F0.5 0.796989 at threshold 0.985 and top-K 5, and held-out pilot-test macro F0.5 0.796470. Pilot-test scores are reporting-only; no threshold or model choice was selected using those scores.

The pilot candidate DB has an integrity-check PASS in its recovery process and exact expected row count. The feature DB has 1,145,685 rows, all 31-float32 blobs of expected width, and a prefix check against candidate order; the report records no missing feature rows. Score DB row count is 1,145,685. Recovery exit code is 0. The source code and model use NumPy only beyond Python standard library. The full-scale proposal is explicitly proposal-only and must not be treated as an executable, tested pipeline.

`utils/validate_submission.py` is stdlib-only and supports full S1 coverage, row uniqueness, prefixes, intra-list duplicates, optional test-ID provenance (`--check-ids`) and prediction-subset-of-candidate warnings. No final output exists yet, so the official validator has not been run on a submission.

## Reproducibility and environment

The supplied project directory and its parent have no usable Git repository metadata (`git rev-parse HEAD` fails); therefore no source commit can be recorded. `baseline_manifest.json` hashes the key source files and artifacts instead. Environment inspected: Python 3.14.7, NumPy 2.5.3, 12 CPUs, 15.0 GiB physical RAM, 9.6 GiB MemAvailable at audit time, about 51.2 GiB free disk. The pilot requirements file pins `numpy==2.5.3`. No random seed parameter is used for inference; pilot query sampling is reservoir sampled and the per-record train/validation/test assignment uses stable hashing, but the manifest records the lack of a source commit and the need to preserve pilot artifacts as the reproducibility anchor.

## Error analysis (pilot only)

The frozen evaluation report records pilot-test 1,775 false negatives and 201 false positives (3,480 true positives), with micro precision 0.945395 and recall 0.662226. The candidate stage reports 92.3101% true-pair recall, so candidate omissions are the hard recall ceiling; the remaining missed true links occur after candidate generation due to model/policy. Validation policy comparisons are already available: at threshold 0.985, top-K 5 gives macro F0.5 0.796989; top-K 8 or more gives 0.796922. Threshold 0.98/top-K 5 gives 0.796781. These are near-ties on one small validation sample; keep the baseline policy unless later controlled validation-only tests show stable improvement. The 10K artifacts include row-level labels, query fields, candidate block masks, 31 features and candidate scores; categorical FN slices by target source, country, singleton, name/address presence and rejection/top-K should be generated from these records before selecting any changes. Do not infer or report those slices from the leaderboard/pilot-test labels.

## Full-scale projection and hold point

The existing linear projection estimates 198,494,967 test candidate pairs; roughly 7.9 GiB inverted-index storage; 61.93 GiB working artifacts with uncompressed pilot-style SQLite feature storage; and about 0.58 hours naive compute extrapolation. This projection omits test-corpus block-frequency validation, index-build overhead, I/O contention, checkpoint/restart testing and safety margin. At audit time only about 51.2 GiB free disk is available, less than the projected 61.93 GiB, so the proposed SQLite artifact layout is not safe to run unchanged. A compact/streaming implementation and empirically measured bounded test slice are required before an approval request. Current available memory is host-wide and may change; there is no verified persistent supervisor for a full run yet.

Phase 1 audit artifacts are written. Phase 2 error analysis is available in `error_analysis.md` and the row-level `artifacts/pilot_10k_recovery_manual_20250925/error_analysis.json`. Phase 3 comparison confirms retaining threshold 0.985/top-K 5; existing validation trials show no useful improvement. A bounded unlabeled smoke pass scored 6,240 pairs from 50 test S1 rows against prefixes of 1,000 rows per test target source in 1.65 seconds, with a compatible 31-feature schema and valid S2/S3 prefixes. This is a parser/model smoke check only: it does not represent full corpus blocking frequencies, exercise atomic resume, or validate the final exact-schema outputs. A scalable disk-backed inference implementation and production-equivalent bounded preflight are still required for Phases 4–5. No full-scale command has been run. Before any full test run, provide updated peak RAM/disk/runtime estimates, exact Fish-compatible supervisor command, and risks, then wait for explicit approval.
