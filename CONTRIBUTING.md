# Contributing

The repository is public. Anyone may clone or pull it and may propose changes
through a fork and pull request. Direct push access is limited to invited team
collaborators; do not publish credentials or ask GitHub to grant anonymous write
access.

## Workflow

1. Clone the repository and create a focused branch:
   `feature/<short-description>`, `fix/<short-description>`, or
   `docs/<short-description>`.
2. Make the smallest reviewable change. Do not commit challenge data, SQLite
   work files, logs, virtual environments, or final multi-gigabyte outputs.
3. Run the same checks used by CI:

   ```text
   python -m pip install -r requirements_lock.txt
   python -m compileall -q pilot utils tests tools
   python -m unittest discover -s tests -v
   python -m pip check
   ```

4. Push the branch to your fork or the team repository if you have collaborator
   access.
5. Open a pull request against `main` and wait for Ubuntu, Windows, and macOS CI
   jobs to pass.

## Safety and reproducibility

- Use UTF-8 without a BOM and LF line endings for source and TSV files.
- Do not use external business-identity data, APIs, geocoders, or web lookups.
- Use a new `--work-dir` whenever data, model, feature code, mode, or blocking
  profile changes.
- Never copy a live SQLite database without its `-wal` and `-shm` sidecars.
  Prefer restarting the same command against the same local work directory.
- Treat the local submission validator's warnings as blockers for final
  delivery.

## Reporting bugs

Open an issue with the operating system, Python version, command, expected
behavior, actual behavior, and a minimal synthetic example. Do not attach raw
challenge data.
