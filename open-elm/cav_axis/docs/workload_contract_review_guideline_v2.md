# V2 Workload Contract Review Guideline

## Scope and blinding

Review only the assigned restricted source note, the unresolved parent template,
and, in the automation-assisted arm, the displayed candidate ledger. Do not open
development gold contracts, mismatch taxonomies, prior arm files, outcomes, or
held-out gold labels. Candidate suggestions are navigation aids only; none is
pre-approved.

Start the case timer when **any** source, template, or candidate material for
that case becomes visible. Record pauses and end time in the workload form.

## Atomic contract decision

For each source parent, choose exactly one parent resolution:

- `atomic_created`: create one or more independent, source-supported atoms.
- `not_transition_relevant`: retain no rendered atom.
- `route_manual`: do not create an atom when the transition is materially
  incomplete, contradictory, clipped, or needs inference.

Each created atom must contain one independently actionable clinical obligation.
Keep a medication action with its identity, dose, route, frequency, duration,
condition, and negative constraint when those modifiers apply to that action.
Split unrelated actions, such as a medication regimen and a follow-up visit.

For each atom, set both dimensions independently:

- `final_clinical_priority`: `required`, `optional`, or
  `historical_context_only`.
- `final_render_decision`: `render` or `do_not_render`.

`required` facts that define a safe discharge transition must render in an
allowed section: `principal_diagnosis`, `discharge_medications`, `disposition`,
`instructions`, or `follow_up`. Hospital-course facts may provide context but
must not be the sole location of an actionable discharge obligation.

## Evidence rule

Every created atom needs one contiguous primary source span that supports its
full clinical content. Optional corroborating spans may confirm the same atom;
they may not supply a missing dose, timing, condition, or antecedent. Do not
infer redacted values, reconcile contradictions, derive a duration from a
quantity, or complete a clipped clause.

A bad or clipped automation candidate does not itself justify `route_manual`.
Review the assigned source independently; route manually only if the source
evidence is itself insufficient for a safe atom.

## Safety escalation

If a case contains a materially unsafe unresolved transition, set
`current_case_safety_escalation_yes_no=yes`, stop work on that case, and record
the reason. This pauses **that case only**; it does not terminate the study arm
unless the protocol lead formally suspends it.

## Do not self-adjudicate endpoints

Do not assign final unsupported-claim, critical-omission, or safety-outcome
labels in this form. Those endpoints are determined later by an independent
blinded adjudicator using the finalized rendered contract/note and its sealed
reference.

## Required QC before marking a case complete

- Every created atom has an allowed section, nonblank generation value, and a
  valid contiguous primary span.
- Every `route_manual` parent has all final atom fields blank.
- Atom sequences are consecutive within a parent.
- No source identity differs from the case manifest's `(source_split,
  dataset_row_id, subject_id, note_id)` tuple.
- Reviewer counts are not entered manually; candidate/accept/edit/reject counts
  are derived from the finalized rows after the review.
