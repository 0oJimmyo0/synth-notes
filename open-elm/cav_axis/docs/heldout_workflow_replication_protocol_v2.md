# Final Held-Out Replication Protocol v2

## Status of v1

Protocol v1 is closed as a 60-case feasibility and renderer-regression cohort.
Its source-contract review produced nine source-ready ledgers and seven
contract-eligible cases. The first renderer had one critical omission caused by
an unmapped required contract section. The section-bound renderer v2 repaired
that defect and passed 7/7 same-environment blinded source-contract reviews.
Because that result informed the renderer-section binding guard, v1 is not the
final confirmatory cohort.

## Final Replication Cohort

Freeze a new, subject-unique 400-anchor test cohort before inspecting source
text, ledger facts, generated notes, MedGemma output, or human-review labels.
Target 100 anchors in each frozen stratum:

- stable_sparse / patient_disjoint
- stable_sparse / patient_overlap
- stable_dense / patient_disjoint
- stable_dense / patient_overlap

Exclude all prior development anchors, all v1 screening anchors, and every
subject represented in those cohorts. Do not replace anchors after source,
contract, generation, MedGemma, or review outcomes are known. Report the full
attrition funnel and realized stratum counts. The 400-anchor size reflects the
observed v1 final contract-eligibility yield of 7/60 and targets approximately
40 final contract-eligible cases.

## Frozen Workflow

1. Use canonical-transition v2, the frozen repeated subject-grouped train
   references, and k=50 local support to create the cohort.
2. Perform source-only ledger review. Preserve uncertainty as `not specified`;
   do not reconstruct redacted medication components.
3. Compile status-labeled contracts and exclude unresolved material transition
   obligations. Every required or optional fact must map to a rendered canonical
   section: discharge diagnosis, hospital course, discharge medications,
   disposition, discharge instructions, or follow-up.
4. Generate four candidates per contract-eligible case with checkpoint_8215
   and the section-bound hybrid renderer v2.
5. Re-embed candidates using canonical section-balanced BGE embeddings.
6. Select one constraint-passing candidate per case by maximum minimum k=50
   support across the five frozen train reference splits, then mean support,
   then lowest candidate index. Do not use target-region membership, clinical
   review, or MedGemma outputs for selection.
7. Conduct blinded source-contract review of all selected notes.
8. Run local MedGemma contract alignment with three repeats and a 3072-token
   cap. Use it only for operational routing: invalid output, non-present
   alignment, or repeat inconsistency. It is not an automatic rejection gate.

## Endpoints and Boundaries

Primary endpoints are source-to-contract attrition, final contract eligibility,
generation completion, course-constraint completion, local-support selection,
and blinded strict contract-review pass rate. MedGemma endpoints are schema
validity, complete-repeat rate, repeat stability, and routing rate.

This remains a methodological source-grounded rendering study. Without a
separate blinded clinician or pharmacist reviewer, do not claim independent
clinical validation, autonomous safety, or MedGemma detection sensitivity or
specificity. Do not claim target-basin landing.
