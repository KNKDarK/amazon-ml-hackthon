# Production blocking-cap validation replay

This directory contains a replay-only experiment requested before full-test inference.

- The S1 input is the 1,483-query `validation` split from the frozen 10K pilot.
- The S2/S3 inputs are symlinks to the complete training target corpus (10,320,219 rows).
- `pilot/stream_infer.py` is run in `preflight` mode with `target_sample_rate=1`; no test file is read. The replay used the production query/candidate/scoring path with a replay-equivalent index: the builder streamed all 10,320,219 target rows and retained postings for every key used by the validation queries. SQLite index writes were set to `OFF` only for this local replay; the full-run command below leaves the production default at `FULL`.
- The requested cap is applied to both production knobs (`--block-cap` and `--query-posting-cap`) for each run.
- The frozen model is read-only; no training or model write occurs.
- Cap outputs are kept under `cap_*/output/`; they are not submission outputs.

The final summary is written to `cap_validation.json` and `REPORT.md` after all three replays complete. `targeted_index/targeted_index_meta.json` records the full-corpus scan and replay-index integrity counts; `build_targeted_index.py` and `prepare_targeted_replays.py` make the replay reproducible.
