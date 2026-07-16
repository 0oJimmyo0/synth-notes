# Clinical Validation

This package supports review-calibrated evaluation and triage. It is not a
clinical-validity certifier: semantic factuality must be checked against the
source record.

- `ingest_manual_review_labels.py` summarizes completed detailed review sheets
  without exporting source-note text.
- `deterministic_safety_checks.py` detects objective issues such as missing
  substantive sections, unfinished output, extreme numeric values, repeated
  blocks, and PHI-like patterns.

Use these outputs to remove obvious failures and prioritize source-paired
verification for the source-grounded rescue pilot.
