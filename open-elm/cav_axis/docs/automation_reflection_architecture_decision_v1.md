# Automation Reflection Architecture Decision V1

## Scope

Development-only evidence from the frozen 88-case automation split. The 38
`heldout_gold` cases remain unopened.

## Evidence

- V1.3 full-contract raw agreement: 59.0% recall and 57.4% precision.
- V1.3 safe-canonicalized agreement: 78.2% recall and 76.1% precision.
- Prediction-blind independent atomic review, 23 comparable non-excluded cases:
  56.6% raw recall / 59.0% raw precision and 68.1% safe-canonicalized recall /
  70.9% precision.
- The 362-row V1.3 root-cause audit found 139 normalization-only mismatches,
  109 split/granularity mismatches, and 67 follow-up or section-routing
  mismatches. Only 13 were true unsupported extractions.

## Decision

Do not continue Pass C deterministic full-contract parsing and do not use a
95% raw exact-atom gate for autonomous contract replication. The independent
review demonstrates that raw atomic representation is not unique enough for
that endpoint under the current guideline.

Adopt the selective, evidence-linked human-in-the-loop automation hypothesis:

- Automation may pre-populate only source-span-linked candidate obligations.
- Any unresolved, ambiguous, contradictory, or high-risk component routes to
  human review.
- Final contract correctness remains a human-approved endpoint.
- Full-case autonomous coverage is no longer the primary scalability metric.

## Next Evaluation Package

1. Define a frozen confidence policy for candidate obligation classes.
2. Measure candidate precision, manual-route sensitivity, and the fraction of
   source-supported obligations pre-populated.
3. Conduct a reviewer workload study: full manual construction versus
   automation-assisted review, using prediction-blind source packs and final
   human-approved contracts.
4. Retain raw and safe-canonicalized agreement as descriptive diagnostics, not
   as an autonomy-release gate.
5. Do not run `heldout_gold` until the human-in-the-loop protocol, endpoints,
   and safety audit are frozen.
