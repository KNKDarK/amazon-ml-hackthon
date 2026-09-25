# Model license and model card

## License

The model weights and normalization arrays in
`pilot/frozen_pilot_model.json` are part of this repository and are licensed
under the [MIT License](LICENSE).

## Model card

- **Task:** Cross-source business entity matching for the supplied ML Challenge
  2026 data.
- **Architecture:** Compact binary logistic regression implemented in NumPy.
- **Parameters:** 31 coefficients, plus the feature means and standard
  deviations stored in the same JSON artifact; far below the 8-billion-parameter
  limit.
- **Training data:** Only the challenge-provided labeled training TSVs and a
  deterministic 10,000-row Source-1 pilot sample.
- **External data:** None. No external business database, geocoder, identity
  API, web lookup, or pretrained model was used.
- **Inputs:** Name, address, and open-valued country strings from challenge
  records only.
- **Decision policy:** Validation-selected probability threshold `0.9906` and
  per-query top-K cap `5`, recorded alongside the weights.
- **Artifact SHA-256:** `cca0e008e75ed83a098ba1d7284ca1d27a47e190ad54cb7c45529d1b8525f573`.
- **Intended use:** Reproduce this challenge submission and related research on
  the supplied data. It is not a general-purpose identity-verification system.

The model is deterministic at inference apart from possible last-bit numerical
differences among supported NumPy/BLAS builds. The repository pins the audited
NumPy version.
