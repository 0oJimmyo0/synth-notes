# Automation Task Reflection Review Guideline

This development-only review has two separate tasks. Do not open V1.3
predictions while completing the independent-contract task. All review files
contain restricted source-derived content and must remain on approved storage.

## A. V1.3 Mismatch Root-Cause Audit

Review `v1_3_raw_obligation_mismatch_audit_RESTRICTED.csv`. Each row is one
raw unmatched obligation from a safely automated development case.

Fill `primary_cause` with exactly one label:

- `true_parser_omission`: required gold obligation absent from prediction because the parser did not extract it.
- `true_unsupported_parser_addition`: prediction is not supported by its linked source span.
- `under_splitting`: one prediction combines independently actionable gold obligations.
- `over_splitting`: one gold obligation is divided without independent action predicates.
- `modifier_attachment_error`: source-supported modifier has the wrong action, target, site, timing, frequency, duration, or condition.
- `instruction_follow_up_classification_error`: supported obligation is routed to the wrong section.
- `clinically_equivalent_differently_atomized`: same clinically complete content but different safe atom boundaries.
- `lexical_canonicalization_mismatch`: same atom except formatting, punctuation, safe abbreviation, or explicit interval spelling.
- `ambiguous_source`: source does not establish a unique atomic representation.
- `incomplete_truncated_source_route_manual`: source is clipped, unresolved, or requires completion; it must route manually.
- `manual_contract_inconsistency_or_error`: manual contract is internally inconsistent or unsupported by its displayed source evidence.

Add a concise evidence-based `cause_reviewer_note`. Do not alter the displayed obligation, source identifiers, spans, parser rule, or case ID.

## B. Independent Human Contract Review

Review `human_agreement_source_review_pack_RESTRICTED.csv` without opening the
V1.3 prediction file, mismatch audit, or existing manual gold contract.

For each source-supported transition, populate only:

- `independent_contract_status`: `required`, `optional`, `include`, `historical_context_only`, or `omitted`.
- `independent_contract_section`: `principal_diagnosis`, `hospital_course_events`, `discharge_medications`, `disposition`, `instructions`, or `follow_up`.
- `independent_contract_generation_value`: atomic source-supported text; preserve `not specified` and do not reconstruct redactions.
- `independent_reviewer_note`: concise reason only when routing or atomization is non-obvious.

Use one atomic obligation per independently actionable transition. Keep provider, timing, and prerequisite test together when they form one linked follow-up plan. Route clipped, ambiguous, or discontinuous evidence as `omitted` with a note rather than completing it by inference.

## Completion Checks

- Every mismatch row has one valid `primary_cause` and nonblank reviewer note.
- Every independent-review row has a final status.
- No independent-review row uses V1.3 predictions or existing manual-gold content.
- No redacted value is reconstructed.
