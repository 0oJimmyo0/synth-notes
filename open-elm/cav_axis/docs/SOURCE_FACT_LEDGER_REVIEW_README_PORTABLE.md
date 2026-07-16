# Source Fact Ledger Manual Review — Portable Handoff

These files contain **no source-note text and no clinical fact values**. They can
be moved separately from the restricted MIMIC-IV files, subject to your project
governance rules.

## Included files

- `source_fact_ledger_review_patch_PORTABLE.csv` — one decision for every
  original ledger row plus source-offset instructions for verified additions.
- `apply_source_fact_ledger_review.py` — applies the patch on approved storage
  and creates the completed **restricted** ledger.
- `validate_source_fact_ledger_portable.py` — checks statuses, required-field
  coverage, uniqueness, and (when given the restricted source reference)
  source-span support without printing clinical text.
- `source_fact_ledger_review_qc_PORTABLE.json` — aggregate review/QC counts only.

## Apply on approved MIMIC-IV project storage

```bash
python apply_source_fact_ledger_review.py \
  --ledger source_fact_ledger_manual_verification.csv \
  --source-reference source_fact_ledger_source_reference_RESTRICTED.csv \
  --patch source_fact_ledger_review_patch_PORTABLE.csv \
  --output source_fact_ledger_manual_verification_COMPLETED_RESTRICTED.csv
```

The output contains source-derived clinical text and must not leave approved
project storage.

## Validate source support and required coverage

```bash
python validate_source_fact_ledger_portable.py \
  --ledger source_fact_ledger_manual_verification_COMPLETED_RESTRICTED.csv \
  --source-reference source_fact_ledger_source_reference_RESTRICTED.csv \
  --summary-json source_fact_ledger_validation_summary_PORTABLE.json
```

Then run the project validator requested in the handoff:

```bash
python validate_source_fact_ledger.py   source_fact_ledger_manual_verification_COMPLETED_RESTRICTED.csv
```

Adjust the final command if the project script uses named arguments. The project
validator itself was not among the uploaded files, so it was not executed here.

## Expected result

- 483 final rows across 45 cases
- 354 `verified`
- 82 `corrected`
- 47 `rejected`
- 0 `pending`
- Verified/corrected coverage for all 45 cases in each required field:
  `principal_diagnosis`, `hospital_course_events`, `discharge_medications`,
  `disposition`, `follow_up`, and `instructions`
