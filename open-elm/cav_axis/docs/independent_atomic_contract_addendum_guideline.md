# Independent Atomic Contract Addendum Guideline

Use this prediction-blind addendum only after completing the source-row routing
review. Do not open automation outputs, mismatch labels, or the original manual
gold contract while completing it.

For each required source parent row:

- Set `independent_parent_resolution` to `atomic_created` or `route_manual`.
- Create one row for each independently actionable, source-supported obligation.
- Duplicate the parent row when it supports multiple atoms.
- Set `independent_atom_sequence` consecutively within each parent fact.
- Set `independent_atomic_status` to `required` for retained atoms.
- Set `independent_atomic_section` to the section where the atom must render.
- Write the smallest clinically complete source-supported text in `independent_atomic_generation_value`.
- Record the exact contiguous source span used for that atom.

Keep provider, timing, and prerequisite test together when they form one linked
follow-up plan. Split only independent actions. Preserve `not specified`; never
derive a medication identity, dose, frequency, duration, timing, or condition
from surrounding context. If a parent contains clipped, discontinuous, or
ambiguous evidence, do not create an atom and explain why in the reviewer note.
Set the parent resolution to `route_manual`, leave every atomic field blank, and
write the reason in the reviewer note.
