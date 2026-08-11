# Workload Contract Review Guideline V1

Use only the assigned arm directory. Do not open the other arm, development
gold, automation mismatch audits, or held-out-gold material.

## Files Per Arm

- `source_reference_RESTRICTED.csv`: the source note for the assigned cases.
- `final_contract_template_RESTRICTED.csv`: the final human-approved atomic
  contract record. This is the clinical work product.
- `workload_review_form.csv`: timestamps, reviewer actions, and safety endpoints.

The `automation_assisted` directory additionally contains
`selective_candidate_ledger_RESTRICTED.csv`. Its candidate states are aids, not
final decisions. The `full_manual` directory has no candidate ledger.

## Contract Construction

For every source parent, set `final_parent_resolution` to `atomic_created`,
`route_manual`, or `not_transition_relevant`.

- For `atomic_created`, create one row per independently actionable,
  source-supported obligation. Duplicate the parent row as needed.
- Use `required`, `optional`, `include`, `historical_context_only`, or `omitted`
  for `final_contract_status`.
- Required atoms must have a rendered section, nonblank generation value, and
  one contiguous source span.
- For `route_manual`, leave all final atom fields blank and explain the clipped,
  ambiguous, contradictory, or high-risk issue in `final_reviewer_note`.
- To add an obligation absent from the extracted parents, add a row with the
  case ID, source span, and source-supported atomic value. Do not infer details.

## Arm-Specific Workflow

`full_manual`: construct the contract from the source note and record total
active-review time.

`automation_assisted`: inspect source-linked candidates first. A direct
principal-diagnosis `AUTO_ACCEPT` may remain unchanged only if its source span
is correct. All medication, disposition, instruction, and follow-up content is
`MANUAL_REVIEW` and requires active reviewer adjudication.

## Time And Safety

Start timing when the source case opens and stop after the final contract-ready
or manual-route outcome is recorded. Record pauses longer than five minutes.
Complete all counts in the workload review form. If an unsupported high-risk
candidate would otherwise be accepted, stop the assisted arm and record the
event; do not repair it silently.
