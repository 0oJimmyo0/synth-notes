# V2 Selective HIL Workload Protocol

V2 supersedes V1 for workload-effect estimates. V1 has an identity-resolution
defect: numeric `dataset_row_id` values overlap across original source splits,
while source extraction joined by that number alone. V1 clinical reviews remain
feasibility artifacts only and must not contribute timed arm comparisons.

## Design

- Select 80 fresh utility-train cases outcome-blind, one per patient.
- Use the compound identity `(source_split, dataset_row_id)` plus `subject_id`
  and `note_id` for every selection, ledger, pack, and audit join.
- Balance 40 full-manual and 40 automation-assisted cases, with 8 per arm in
  each source-note-length quintile.
- Exclude every V1 selected identity, every V1 actually displayed source
  identity, and every subject exposed in V1 before V2 selection.
- Both arms receive the same source/template material; assisted candidates are
  suggestions only and all remain `MANUAL_REVIEW`.

## Outcomes and safeguards

The primary workload outcome is active reviewer time per completed case.
Secondary outcomes are derived candidate handling counts, manual-route rate,
and independent blinded safety adjudication of finalized notes/contracts.
Timing starts at first case-material visibility. A safety escalation stops only
the current case unless the protocol lead suspends the study.

No development gold, held-out gold, outcomes, or case-specific tuning may be
opened during V2 review. The utility test split remains unopened.
