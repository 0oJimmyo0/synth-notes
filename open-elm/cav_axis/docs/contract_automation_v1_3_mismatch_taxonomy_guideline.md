# V1.3 Instruction/Follow-Up Mismatch Taxonomy

Use this guide only for the development-only V1.2 mismatch audit. Review the
restricted gold and predicted values with their evidence metadata. Assign one
primary `mismatch_type` to every row; do not infer source content that is not
present in the evidence span.

Allowed primary labels:

- `under_splitting`: one predicted atom contains two or more independently actionable gold obligations.
- `over_splitting`: one gold obligation was divided into multiple predicted atoms without independent action predicates.
- `modifier_attachment`: a timing, site, laterality, frequency, duration, condition, or restriction is attached to the wrong action.
- `follow_up_decomposition`: provider, timing, reason, or prerequisite test was represented inconsistently within a follow-up relationship.
- `instruction_follow_up_section_confusion`: the same supported action was routed to the wrong rendered section.
- `conjunction_ambiguity`: a conjunction could reasonably bind either one compound action or two independent actions.
- `negation_constraint_attachment`: a hold, stop, avoid, or other negative constraint has the wrong target or scope.
- `truncated_or_unresolved_source_fragment`: the source cannot yield an exact atom without completing a fragment; V1.3 must route it to manual review.
- `evaluation_only_normalization_mismatch`: gold and predicted atoms are semantically equivalent after safe, frozen normalization, with no clinical distinction.
- `true_unsupported_extraction`: the prediction creates an action not supported by its own source span.

Review rules:

- Use `source_structure` to record sentence, semicolon, numbered list, conjunction, or fragment.
- Use `manual_contract_granularity` to state whether gold uses one linked atom, multiple independent atoms, or a non-actionable context item.
- Use `parser_behavior` to describe preserve, under-split, over-split, or wrong-section routing.
- Use `required_fix_class` only as `instruction_parser`, `follow_up_parser`, `canonicalizer`, or `manual_route`.
- If evidence comes from discontinuous spans, choose `manual_route`; V1.3 must not combine them.
