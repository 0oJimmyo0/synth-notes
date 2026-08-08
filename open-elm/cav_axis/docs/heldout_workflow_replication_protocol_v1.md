# Held-Out Workflow Replication Protocol v1

## Decision

Development is frozen after the canonical-transition v2, k=50 local-support,
hybrid renderer v2.1, and MedGemma contract-alignment operational analyses.
The next experiment is one subject-disjoint held-out test replication. No
further tuning may use held-out generated notes, contract coverage, MedGemma
outputs, or review labels.

## Cohort

Select 60 test screening anchors, subject-unique and excluded from all prior
development anchor manifests. Target 15 notes in each frozen stratum:
stable_sparse/patient_disjoint, stable_sparse/patient_overlap,
stable_dense/patient_disjoint, and stable_dense/patient_overlap. After source
ledger and contract review, analyze every contract-eligible case (target at
least 40). If a stratum is unavailable or fewer than 40 cases remain, preserve
the shortfall and report the realized counts rather than replacing the design
post hoc.

## Frozen Workflow

1. Manually verify source fact ledgers and compile status-labeled contracts.
2. Exclude contracts with unresolved material discharge obligations.
3. Generate four candidates per remaining case using checkpoint_8215 and the
   hybrid renderer v2.1.
4. Score candidates against the frozen repeated subject-grouped train
   reference splits at k=50.
5. Select one candidate per case by maximum minimum support, then mean support,
   then lowest candidate index. Do not use target-basin membership, review
   labels, or MedGemma outputs for selection.
6. Audit deterministic contract coverage and course constraints.
7. Run local offline MedGemma contract alignment with three repeats. Route any
   invalid, non-present, or repeat-inconsistent alignment; do not use MedGemma
   as an automatic rejection gate.

## Endpoints

Primary operational endpoints are required-contract coverage, generation and
schema completion, conservative MedGemma routing rate, and candidate local
support. Secondary descriptive endpoints are source-ledger readiness,
patient-disjoint subgroup results, and course-constraint failures.

Clinical-usability adjudication remains exploratory unless performed by an
independent clinician or pharmacist who is blinded to condition and model
outputs. Current same-environment reviews must not be described as independent
clinical validation.

## Reporting Boundaries

The manuscript will describe the method as source-grounded contract rendering
with representation-stable local-support ranking. It will not claim target
basin landing, autonomous clinical safety, or validated MedGemma detection
accuracy. All data remain under the approved MIMIC access controls; code,
hashes, and derived non-text summaries will be prepared for reproducibility.
