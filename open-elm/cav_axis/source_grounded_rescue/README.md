# Source-Grounded Rescue

This package evaluates local, source-fact-conditioned generation after the
embedding-to-note decoder failed the clinical source-faithfulness gate. It
separates fact extraction, human verification, generation, and evaluation so
that a generator failure is not confused with an incorrect source ledger.

## Completed smoke decision

The blinded four-case smoke compared raw ELM, raw-draft correction, and
fact-only generation under the untouched backbone and `checkpoint-8215`.

- `checkpoint-8215` fact-only: `4/4` rule-verified passes, with no unsupported
  major claims or critical omissions.
- untouched-backbone fact-only: `3/4` passes.
- raw ELM and both raw-draft correction conditions: `0/4` passes.

This is a method-selection signal, not a performance estimate: four cases have
wide uncertainty. The raw ELM draft is frozen as a baseline and is not used as
conditioning context in the replication.

## Next prospective replication

1. Freeze 30 previously unused cases before generation, stratified by source
   route, patient-disjoint status, clinical service/diagnosis, and ledger
   complexity. Do not select cases for apparent ease.
2. Convert verified source facts into reviewer-approved concise
   `generation_value`s. Never place source spans or full source-note text in a
   prompt.
3. Generate one deterministic `checkpoint-8215` fact-only note per case. No
   retries, manual prompt edits, candidate selection, or geometry optimization.
4. Retain the existing raw ELM note for each matched anchor as baseline.
5. Run an untouched-backbone fact-only control for a prespecified nested subset
   of 15 cases, to test whether the checkpoint contributes beyond the ledger.
6. Blind all conditions and use the predeclared factual pass rule before any
   re-embedding, geometry, privacy, or coverage analysis.

Proceed only if the 30-case `checkpoint-8215` fact-only arm has at least
`27/30` passes, no more than two unsupported-major-claim failures, no more than
two critical-omission failures, and no systematic medication, procedure, or
disposition failure. These are project feasibility thresholds, not clinical
deployment criteria.

Do not send source notes, ledgers, or outputs to a third-party API.
