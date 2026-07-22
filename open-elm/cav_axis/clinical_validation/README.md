# Clinical Validation

This package supports review-calibrated evaluation and triage. It is not a
clinical-validity certifier: semantic factuality must be checked against the
verified ledger.

## Phase 3a: local medication-judge feasibility

The initial task is deliberately narrow: reconcile compact verified medication
facts with the generated transition-note medication claims. It is evaluated as
a calibration/feasibility study, not used as an automatic gate.

- `build_medication_judge_dataset.py` creates restricted task JSONL files from
  reviewed notes. Cluster 25 is development-only; cluster 11 remains locked
  held-out evaluation data.
- `build_medication_judge_candidate_review_pack.py` samples additional existing
  cluster-25 candidates for blinded medication-focused development labels; it
  never changes generation or touches held-out cluster-11 data.
- `medication_judge_schema.json` defines the evidence-first output contract.
- `medication_judge_prompt_v1.txt` is the versioned prompt template. Calibrate
  candidate prompt versions only on cluster-25 development data, then freeze one.
- `validate_judge_json.py` checks schema conformance without exporting note or
  ledger text.
- `analyze_medication_judge.py` compares frozen judge decisions to human
  labels. It reports counts, sensitivity, specificity, and false rejection.
- `ingest_manual_review_labels.py` summarizes completed detailed review sheets
  without exporting source-note text.
- `deterministic_safety_checks.py` detects objective issues such as missing
  substantive sections, unfinished output, extreme numeric values, repeated
  blocks, and PHI-like patterns.

All task JSONL files contain compact verified ledger text and synthetic notes.
Keep them on approved project storage. Do not send them to external APIs. The
generation checkpoint must not be used as the primary judge; provision an
independent approved local instruction-following model before execution.
