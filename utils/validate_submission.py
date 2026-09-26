#!/usr/bin/env python3
"""
ML Challenge 2026 — Submission Validator

Run this BEFORE submitting. It is a local preflight checker for the published
challenge rules; the organizer's portal validator remains authoritative. It
reads only your output files and the test source files (to learn
which S1 entities are required and which S2/S3 IDs exist); it never needs the
ground truth and never computes your score.

It validates two files:

* ``matching_results.tsv`` (required) — your final matches, the file scored on the
  leaderboard.
* ``candidate_pairs.tsv`` (optional) — the candidate set from your blocking stage.
  When present, the validator also checks that your final matches are a subset of
  your candidates and *warns* (never fails) otherwise. When absent it is skipped
  with a warning; it is still expected in your final submission zip.

Stdlib only, Python 3.8+. Run from the ``student_resource/`` directory::

    python3 utils/validate_submission.py \
        --matching output/matching_results.tsv \
        --candidate output/candidate_pairs.tsv \
        --test-dir dataset/test

Exit code 0 means the files are safe to submit; 1 means fix the listed issues
(warnings never fail the run).

ID-existence check (off by default). By default the validator does NOT check that
every matched/candidate ID actually exists in the test set: that check loads all
Source-2/3 IDs into memory, which on the full test set costs a few GB. Candidate
pairs are streamed without retaining the full candidate mapping, keeping the check
bounded on a 16 GB machine. The default run stays fast and light and verifies every
other rule; it prints a warning noting the check was
skipped. Pass ``--check-ids`` to turn it on (it reads ``test_source2.tsv`` /
``test_source3.tsv`` from ``--test-dir``); unknown S2/S3 IDs are blocking errors.
If ``--check-ids`` runs out of memory, validate the matching file first and run
candidate provenance as a separate resource-aware check.
"""

import argparse
import heapq
import os
import sys
from pathlib import Path

DELIM = "\t"
MAX_EXAMPLES = 5  # how many offending IDs to show per issue
MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_ids(path):
    """Return the set of first-column entity IDs from a source TSV.

    The header row is skipped and blank lines are ignored.
    """
    with open(path, encoding="utf-8-sig", newline="") as f:
        next(f, None)  # skip header
        return {line.split(DELIM, 1)[0].strip() for line in f if line.strip()}


def examples(items):
    """Return a short sample without sorting a potentially huge error set."""
    shown = ", ".join(heapq.nsmallest(MAX_EXAMPLES, items))
    if len(items) > MAX_EXAMPLES:
        return f"{len(items)} total, e.g. {shown}, ..."
    return shown


def load_match_targets(test_dir, warnings):
    """Return the set of valid S2/S3 match IDs, or ``None`` if unavailable.

    Only called when ``--check-ids`` is on. When ``test_source2.tsv`` or
    ``test_source3.tsv`` is missing we cannot check that matched IDs exist, so we
    record a warning and return ``None`` to signal that the existence check should be
    skipped.
    """
    targets = set()
    for name in ("test_source2.tsv", "test_source3.tsv"):
        path = os.path.join(test_dir, name)
        if not os.path.isfile(path):
            warnings.append(
                f"{path} not found — skipping the (optional) check that matched "
                f"IDs exist in the test set. Every other rule is still checked. "
                f"This is the lighter-memory mode; provide test_source2/3.tsv to "
                f"enable the ID-existence check."
            )
            return None
        targets |= read_ids(path)
    return targets


def validate_id_list_file(
    path,
    expected_header,
    col_label,
    required,
    valid_ids,
    errors,
    *,
    retain_mapping=True,
    reference_mapping=None,
    subset_offenders=None,
):
    """Validate one results-style TSV (matching or candidate).

    ``retain_mapping`` is disabled for the potentially very large candidate file;
    candidate IDs are checked immediately against the much smaller accepted-match
    mapping to keep validation bounded on a 16 GB machine.
    """
    if not os.path.isfile(path):
        errors.append(f"File not found: {path}")
        return None

    name = os.path.basename(path)
    mapping = {} if retain_mapping else None
    seen, dup_rows, intra_dupes = set(), set(), set()
    self_matches, wrong_prefix, unknown = set(), set(), set()
    n_rows = empties = 0

    with open(path, encoding="utf-8-sig", newline="") as f:
        header = f.readline()
        if not header:
            errors.append(f"{name} is empty.")
            return None
        if DELIM not in header and "," in header:  # the #1 mistake: a CSV
            errors.append(
                f"{name}: header has no TAB but contains commas — the file looks "
                "COMMA-separated. Submissions must be TAB-separated (.tsv); "
                "write it with df.to_csv(sep='\\t', index=False)."
            )
            return None
        cols = header.rstrip("\r\n").split(DELIM)
        if cols != expected_header:
            errors.append(
                f"{name}: unexpected header {cols}. "
                f"Expected exactly {expected_header} (tab-separated)."
            )
            return None

        for line_num, line in enumerate(f, start=2):
            s1, tab, rest = line.partition(DELIM)
            if not tab:
                if s1.strip():
                    errors.append(
                        f"{name}: malformed row (no tab) at line {line_num}: "
                        f"{line.rstrip()!r}"
                    )
                continue

            n_rows += 1
            if s1 in seen:
                dup_rows.add(s1)
            seen.add(s1)

            ids = rest.rstrip("\r\n").split(",") if rest.strip() else []
            id_set = set(ids)
            if not id_set:
                # An empty candidate list used to skip the subset check, which
                # silently tolerated the worst case: matched IDs present with no
                # candidate to back them. Check it before bailing out.
                empties += 1
                if reference_mapping is not None:
                    if subset_offenders is None:
                        raise ValueError("subset_offenders is required with reference_mapping")
                    if reference_mapping.get(s1, set()):
                        subset_offenders.add(s1)
                continue
            if len(ids) != len(set(ids)):
                intra_dupes.add(s1)
            if retain_mapping:
                mapping[s1] = id_set
            if reference_mapping is not None and reference_mapping.get(s1, set()) - id_set:
                if subset_offenders is None:
                    raise ValueError("subset_offenders is required with reference_mapping")
                subset_offenders.add(s1)
            for mid in id_set:
                if mid.startswith("S1-"):
                    self_matches.add(mid)
                elif not mid.startswith(("S2-", "S3-")):
                    wrong_prefix.add(mid)
                elif valid_ids is not None and mid not in valid_ids:
                    unknown.add(mid)

    # Aggregate the per-category findings. Each entry is (offenders, message);
    # only non-empty categories become errors.
    findings = [
        (
            dup_rows,
            "{name}: duplicate source1_entity_id row(s): {ex}. "
            "Each S1 entity may appear on only one row.",
        ),
        (
            intra_dupes,
            "{name}: repeated ID inside a {col} list for: {ex}. "
            "No duplicate IDs are allowed within a list.",
        ),
        (
            self_matches,
            "{name}: {col} contains Source-1 IDs (self-matches): {ex}. "
            "Only S2-/S3- IDs are allowed.",
        ),
        (
            wrong_prefix,
            "{name}: {col} contains IDs without an S2-/S3- prefix: {ex}.",
        ),
        (
            unknown,
            "{name}: {col} references IDs not in the test "
            "Source-2/3 files: {ex}.",
        ),
        (
            required - seen,
            "{name}: required S1 entity(ies) missing: {ex}. "
            "Every entity in test_source1.tsv needs a row (empty = no match).",
        ),
        (
            seen - required,
            "{name}: row(s) using an S1 ID that is not in the test set: {ex}.",
        ),
    ]
    for offenders, message in findings:
        if offenders:
            errors.append(message.format(name=name, ex=examples(offenders), col=col_label))

    print(f"  {name}: {n_rows} rows ({empties} empty, {n_rows - empties} non-empty).")
    return mapping


def validate(matching_path, candidate_path, test_dir, check_ids=False):
    """Validate the submission output(s); return ``(errors, warnings)`` lists.

    ``check_ids`` (``--check-ids``) turns on the optional, memory-heavy check that
    every matched/candidate ID exists in the test Source-2/3 files. It is off by
    default so the common run stays fast and light.
    """
    errors, warnings = [], []

    source1 = os.path.join(test_dir, "test_source1.tsv")
    if not os.path.isfile(source1):
        errors.append(f"Test source1 file not found: {source1} (check --test-dir).")
        return errors, warnings
    required = read_ids(source1)
    print(f"  required S1 entities: {len(required)}")

    if check_ids:
        valid_ids = load_match_targets(test_dir, warnings)
        if valid_ids is not None:
            print(f"  valid S2/S3 match IDs: {len(valid_ids)}")
    else:
        valid_ids = None
        warnings.append(
            "ID-existence check is OFF (the default) — not checking that matched/"
            "candidate IDs exist in the test set. Every other rule is still checked. "
            "Re-run with --check-ids for final delivery (needs test_source2/3.tsv; "
            "uses more memory)."
        )

    matched = validate_id_list_file(
        matching_path, MATCHING_HEADER, "matched_entity_ids", required, valid_ids, errors
    )
    subset_offenders = set()

    # candidate_pairs.tsv is optional: if it's absent we skip its checks with a
    # warning (it's still expected in your final submission zip). A missing
    # candidate file never fails this run on its own.
    candidate = None
    if candidate_path and os.path.isfile(candidate_path):
        candidate = validate_id_list_file(
            candidate_path, CANDIDATE_HEADER, "candidate_entity_ids",
            required, valid_ids, errors,
            retain_mapping=False,
            reference_mapping=matched,
            subset_offenders=subset_offenders,
        )
    elif candidate_path:
        warnings.append(
            f"{candidate_path} not found — skipping candidate_pairs.tsv checks. "
            "It is optional here, but your final submission zip must include "
            "output/candidate_pairs.tsv."
        )

    # Soft check: your final matches should come from your blocking candidates.
    # A matched ID absent from candidate_pairs.tsv usually means a pipeline bug,
    # so we warn but never fail on it.
    if subset_offenders:
        warnings.append(
            f"{len(subset_offenders)} S1 entity(ies) have matched IDs not present in "
            f"candidate_pairs.tsv, e.g. {examples(subset_offenders)}. Final matches "
            "normally come from your blocking candidates — double-check these."
        )

    return errors, warnings


def read_predictions(path):
    """Return ``{s1_entity_id: [matched ids]}`` from a matching_results-style TSV.

    Same schema as the published train_ground_truth.tsv, so the same reader
    serves both the prediction file and the truth file.
    """
    preds = {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        header = f.readline().rstrip("\r\n").split(DELIM)
        if header[:1] != ["source1_entity_id"]:
            raise ValueError(f"{path}: unexpected header {header!r}")
        for lineno, line in enumerate(f, start=2):
            line = line.rstrip("\r\n")
            if not line:
                continue
            parts = line.split(DELIM)
            s1_id = parts[0].strip()
            rest = parts[1] if len(parts) > 1 else ""
            preds[s1_id] = [x for x in rest.split(",") if x]
    return preds


def score_against_truth(matching_path, truth_path):
    """Compute macro F0.5 and supporting counts against a labelled truth file.

    The metric mirrors ``pilot.er_common.macro_f05`` exactly, so a held-out
    training score and a training-time report are directly comparable:

    * per query, F0.5 = 1.25 * P * R / (0.25 * P + R), and 0.0 when there is
      no true match to retrieve (tp == 0);
    * a query whose truth is empty scores 1.0 if nothing was predicted and
      0.0 if anything was, so predicting a match for a singleton is penalised;
    * the headline number is the unweighted mean over queries (macro).

    Returns a dict, or raises ValueError when the truth file is unusable.
    """
    truth = read_predictions(truth_path)
    preds = read_predictions(matching_path)

    scores = []
    tp = fp = fn = 0
    missing_from_truth = 0
    for qid, actual_ids in truth.items():
        actual = set(actual_ids)
        predicted = set(preds.get(qid, ()))
        if qid not in preds:
            missing_from_truth += 1
        inter = len(actual & predicted)
        p_fp = len(predicted - actual)
        p_fn = len(actual - predicted)
        tp += inter
        fp += p_fp
        fn += p_fn
        if not actual:
            scores.append(1.0 if not predicted else 0.0)
            continue
        if inter == 0:
            scores.append(0.0)
        else:
            precision = inter / (inter + p_fp)
            recall = inter / (inter + p_fn)
            scores.append(
                1.25 * precision * recall / (0.25 * precision + recall)
                if precision + recall else 0.0
            )

    total_pred = sum(len(set(preds.get(q, ()))) for q in truth)
    return {
        "macro_f05": sum(scores) / len(scores) if scores else 0.0,
        "micro_precision": tp / total_pred if total_pred else 1.0,
        "micro_recall": tp / (tp + fn) if (tp + fn) else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "queries_scored": len(truth),
        "queries_with_truth": sum(1 for v in truth.values() if v),
        "queries_predicted_any": sum(1 for v in preds.values() if v),
        "predicted_ids_not_in_truth": sum(1 for q in preds if q not in truth),
        "truth_rows_absent_from_predictions": missing_from_truth,
    }


def report_score(matching_path, truth_path):
    """Print a score block, or explain why no score can be produced."""
    if not truth_path:
        print()
        print("Score: not requested. Pass --ground-truth <labelled file> to compute "
              "macro F0.5 (the test\n  set has no published ground truth, so a "
              "leaderboard score cannot be computed\n  locally without one).")
        return None
    print()
    print("Score")
    truth = Path(truth_path)
    if not truth.is_file():
        print(f"  NOT COMPUTED — no ground truth file at {truth}.")
        print(
            "  The competition does not release ground truth for the test set, so a\n"
            "  leaderboard score for output/matching_results.tsv cannot be computed\n"
            "  locally. Pass --ground-truth <labelled file> to score a held-out split;\n"
            "  the file uses the same two-column schema as train_ground_truth.tsv."
        )
        return None
    try:
        s = score_against_truth(matching_path, truth)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        print(f"  NOT COMPUTED — could not read the ground truth: {exc}")
        return None
    print(f"  ground truth : {truth}")
    print(f"  metric       : macro F0.5 (unweighted mean of per-query F0.5)")
    print()
    print(f"  MACRO F0.5        : {s['macro_f05']:.6f}")
    print(f"  micro precision   : {s['micro_precision']:.6f}")
    print(f"  micro recall      : {s['micro_recall']:.6f}")
    print(f"  TP / FP / FN      : {s['tp']:,} / {s['fp']:,} / {s['fn']:,}")
    print(f"  queries scored    : {s['queries_scored']:,} "
          f"({s['queries_with_truth']:,} with at least one true match)")
    print(f"  queries predicted : {s['queries_predicted_any']:,} non-empty")
    if s["predicted_ids_not_in_truth"] or s["truth_rows_absent_from_predictions"]:
        print(f"  NOTE: {s['predicted_ids_not_in_truth']:,} predicted S1 rows are "
              f"absent from the truth file and "
              f"{s['truth_rows_absent_from_predictions']:,} truth rows have no "
              f"prediction row; both are outside the scored set.")
    return s


def main():
    parser = argparse.ArgumentParser(
        description="Validate ML Challenge 2026 submission output files before submitting."
    )
    parser.add_argument(
        "--matching",
        "-m",
        default=PROJECT_ROOT / "output/matching_results.tsv",
        help="Path to matching_results.tsv (default: %(default)s)",
    )
    parser.add_argument(
        "--candidate",
        "-c",
        default=None,
        help="Path to candidate_pairs.tsv "
        "(default: output/candidate_pairs.tsv if it exists).",
    )
    parser.add_argument(
        "--test-dir",
        "-t",
        default=PROJECT_ROOT / "dataset/test",
        help="Folder with test_source1/2/3.tsv (default: %(default)s). "
        "test_source2/3.tsv are only read when --check-ids is given.",
    )
    parser.add_argument(
        "--check-ids",
        action="store_true",
        help="Also check that every matched/candidate ID exists in the test "
        "Source-2/3 files. Off by default (loads all S2/S3 IDs into memory — a few "
        "GB on the full test set). Candidate rows are streamed without retaining "
        "the full candidate mapping.",
    )
    parser.add_argument(
        "--ground-truth",
        "-g",
        default=None,
        help="Optional labelled file with the same two-column schema as "
        "train_ground_truth.tsv. When given, macro F0.5 and supporting counts are "
        "computed and printed. The test set has no published ground truth, so a "
        "leaderboard score cannot be produced locally without this.",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Skip the submission format checks and only compute the score against "
        "--ground-truth. Use this to score a held-out split, whose S1 IDs are "
        "training IDs and would otherwise fail the test-set format rules.",
    )
    args = parser.parse_args()

    if args.score_only:
        # Scoring a labelled split (typically held-out training data) is a
        # separate concern from submission format: those rows carry train S1
        # IDs, which the test-set format rules would rightly reject. Skip the
        # format gate and report the metric only.
        if not args.ground_truth:
            print("--score-only needs --ground-truth <labelled file>.")
            return 1
        scored = report_score(args.matching, args.ground_truth)
        return 0 if scored is not None else 1

    # candidate_pairs.tsv is optional; default to the conventional path and let
    # validate() skip (with a warning) if the file isn't there.
    candidate_path = args.candidate or PROJECT_ROOT / "output/candidate_pairs.tsv"

    print("ML Challenge 2026 — submission validator")
    print(f"  test dir: {args.test_dir}")
    try:
        errors, warnings = validate(
            args.matching, candidate_path, args.test_dir, check_ids=args.check_ids
        )
    except UnicodeDecodeError:
        print()
        print("FAIL — 1 issue(s) to fix before submitting:")
        print(
            f"  1. A file is not valid UTF-8 text (most likely {args.matching} or "
            f"{candidate_path}). Re-save it as a plain UTF-8, tab-separated .tsv — "
            "not cp1252/Latin-1, and not a compressed or binary file (.gz/.xlsx/"
            ".parquet) renamed to .tsv. In pandas: "
            "df.to_csv(path, sep='\\t', index=False, encoding='utf-8', "
            "lineterminator='\\n')."
        )
        return 1
    except OSError as exc:
        print()
        print("FAIL — 1 issue(s) to fix before submitting:")
        print(f"  1. Could not read a file: {exc}.")
        return 1

    print()
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        print(f"FAIL — {len(errors)} issue(s) to fix before submitting:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        return 1
    print("PASS — no blocking issues found. Safe to submit.")
    report_score(args.matching, args.ground_truth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
