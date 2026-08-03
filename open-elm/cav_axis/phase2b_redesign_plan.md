# Research Plan: Contract-First Sparse-Region Enrichment

## Current Direction

This document preserves the retired Phase 2b steering plan below for
provenance. It is superseded by the following active strategy.

### Primary Method

1. Identify a sparse real-note region using frozen BGE manifold clusters.
2. Select leakage-aware held-out anchors with adequate source-evidence
   eligibility.
3. Create a clinician-reviewed fact contract.
4. Deterministically render diagnosis, active discharge medications,
   disposition, and instructions from that contract.
5. Use ELM only for constrained hospital-course prose.
6. Re-embed all candidates with BGE and select outputs that land in the frozen
   target region while passing deterministic structural and contract checks.
7. Use blinded human review as the final clinical-quality endpoint.

### Evidence To Date

- Decoder round-trip audits ruled out inference-time embedding steering as the
  primary enrichment mechanism.
- Contract-first hybrid generation improved strict review quality in cluster-36
  development and achieved 10/12 strict passes in a fresh cluster-25
  contract-resolved pilot.
- These are feasibility findings. The next experiment must use fresh,
  anchor-disjoint scale anchors with no method changes.

### MedGemma Role

MedGemma-27B is locally provisioned, offline-only, and distinct from ELM. Its
current free-form medication-reconciliation judge is not safe for automatic
decisions: its finding bundles lacked grounding and fabricated v3.3/v3.4 tests
failed schema and unknown-component criteria.

The allowed redesign is a citation-bound evidence-alignment assistant:

- input: explicit active-discharge obligations, historical-context-only facts,
  and unknown components from the reviewed contract;
- output: one fixed label per obligation (`present_supported`, `missing`,
  `unsupported`, or `uncertain`) plus contract ID and note span;
- deterministic code, not the LLM, determines accept/review routing;
- `uncertain` and unavailable components always route to humans;
- no LLM editing until this assistant passes fabricated tests and an independent
  prospective clinical validation.

### Immediate Ordered Steps

1. Freeze MedGemma v3.2-v3.4 results as exploratory.
2. Build 30 fresh, anchor-disjoint cluster-25 anchors from the Tier-1 reserve,
   excluding all prior region-25 anchors.
3. Complete source-ledger and contract review before generation.
4. Run frozen hybrid v3 generation, BGE re-embedding, geometry selection,
   deterministic contract audit, and blinded full-note review.
5. Report exact counts, patient-disjoint results, selection yield, and review
   confidence intervals as feasibility evidence.
6. In parallel, build a fabricated contract benchmark for the redesigned LLM
   evidence-alignment task. Do not rerun it on clinical data until it meets its
   predeclared release criteria.

Status: the v2 fabricated alignment benchmark passed all six cases with valid,
stable, exact labels. The next LLM step is one locked 12-note calibration using
the frozen cluster-25 contract and pre-existing human labels. It cannot be
reported as independent clinical validation.

Update: the contract-matched blinded review is complete. Of 92 obligations,
91 were labeled `present_supported` and one was `unsupported`. MedGemma had
complete schema-valid three-repeat coverage for only 63 obligations. It agreed
on 62/63 covered items but missed the sole unsupported generic claim ("Resume
preadmission medications"), giving observed non-present sensitivity of 0/1.
The 98.4% covered agreement is therefore a class-imbalance result, not a
clinical-safety claim. Before any new clinical LLM evaluation, deterministic
code must route generic catch-all active-discharge medication claims for human
review and pass an expanded fabricated adversarial benchmark.

Implemented safeguard: the deterministic contract audit now routes generic
"resume/continue preadmission, home, or prior medications" claims from either
the discharge-medication or instruction section. It caught the exact locked
calibration `ledger_018` miss. This route is independent of MedGemma and is a
human-review safeguard, not automatic note acceptance or rejection.

### Manuscript Boundary

The main manuscript claim is a source-grounded, human-audited feasibility
framework for enriching sparse clinical-note regions. It must not claim clinical
deployment, autonomous medication reconciliation, privacy safety, or
cohort-wide benefit without a larger independent validation.

---

# Archived Phase 2b Steering Redesign Plan

## Objective

Build a second-generation steering method for sparse-region enrichment that is more target-specific than the current global linear axis bank.

The redesign goal is not to change the project scope. The scope remains:

1. identify sparse or low-density regions in the full real-note embedding manifold
2. generate synthetic notes that enrich those regions
3. require gains to beat matched vanilla and norm-matched random shift

Phase 2b changes only the steering mechanism, not the project objective.

## Current diagnosis

The completed first-generation pilots show:

- steering can improve some global held-out coverage metrics relative to matched vanilla
- steering does not robustly outperform norm-matched random shift
- target-cluster occupancy is often weak even when global coverage improves

Interpretation:

- current global metadata-linked axes move notes in embedding space
- but they do not yet reliably fill the intended sparse region after decoding
- therefore the current axis bank is better treated as an exploratory steering/probing result than as the final enrichment engine

## What we already have and should reuse

Do not rerun the following unless there is a specific bug or schema change:

- full real-manifold discovery outputs under `coverage/real_all_filtered_precompute_with_subgroups`
- vanilla baseline generation + audit + real-vs-synthetic coverage
- subgroup metadata build in `subgroup_metadata/subgroup_metadata_filtered.csv`
- filtered-aligned split manifest in `leakage_audit/split_manifest_note_level.csv`
- pilot-control summary in `cav_axis_inputs/pilot_control_summary.csv`
- target-cluster occupancy summary in `cav_axis_inputs/cav_stage_diagnosis_summary.csv`

Keep reusing the existing downstream stack:

- shifted dataset creation:
  - `build_shifted_embedding_dataset.py`
- random-shift control:
  - `build_random_shift_control_dataset.py`
- generation:
  - `generate_synthetic_notes.py`
  - `generate_synthetic_notes.slurm`
- re-embedding:
  - `reembed_generated_notes.py`
  - `reembed_generated_notes.slurm`
- audit:
  - `audit_vanilla_generation.py`
  - `audit_vanilla_generation.slurm`
- coverage:
  - `prepare_coverage_mapping.py`
  - `coverage_real_vs_synthetic.slurm`

This keeps Phase 2b as a method redesign, not an infrastructure rewrite.

## Generic algorithm

### Input regions

Use the full real-manifold coverage outputs to choose candidate sparse regions. Initial priority should remain the empirically motivated regions already studied, especially the ones that showed at least some signal:

- `cluster29`
- `cluster16`

Cluster 11 and 25 are still useful controls, but they should not be the first redesigned target.

### Input embeddings

Use:

- `encoded_training_filtered` to fit local directions
- `encoded_testing_filtered` for held-out steering evaluation

### Direction families

Implement these direction families in order, starting with the simplest:

1. `centroid_difference`
   - target cluster centroid minus source pool centroid
   - source pool can be:
     - global non-target rows
     - nearest confusing cluster
     - matched local neighborhood around each source anchor

2. `one_vs_rest_linear`
   - fit a linear classifier for target cluster vs non-target rows
   - use normalized weight vector as the steering direction

3. `local_neighbor_residual`
   - for each source anchor, compute a direction toward its nearest target-cluster neighborhood
   - optionally average within a small neighborhood to reduce noise

4. `hybrid_constrained`
   - start with one of the above directions
   - constrain the shift by norm and/or cosine-preservation thresholds
   - optionally reproject to a local real-manifold neighborhood

### Steering workflow

For each target region and direction family:

1. fit the direction on filtered training embeddings only
2. build a shifted held-out test dataset with the existing shifted-dataset builder
3. generate notes with the current ELM pipeline
4. compare against:
   - matched vanilla
   - norm-matched random shift

### Alpha policy

Keep the alpha grid narrow at first:

- `0.25`
- `0.5`
- `1.0`

For signed directions, test both positive and negative only when the direction semantics are ambiguous.

## Evaluation criteria

### Primary success criteria

A redesigned steering condition should be considered successful only if it beats both matched vanilla and norm-matched random shift on:

- patient-disjoint real-to-synthetic coverage
- at least one target-specific metric:
  - target-cluster occupancy
  - low-density-cluster coverage
  - target-region subgroup coverage

### Secondary criteria

It should also remain acceptable on:

- audit status: `PASS` or defensible `CAUTION`
- median source-to-generated cosine
- collapse rate
- privacy first-pass checks

### Failure criteria

Do not scale a redesigned direction if:

- it only beats vanilla but not random shift
- it improves broad coverage while failing target occupancy
- it requires very large alpha to show any effect
- it degrades patient-disjoint behavior materially

## Why this is still broad enough for the project

This redesign is not a one-off cluster rescue.

It remains broad because:

- the project objective is sparse-region enrichment, not global concept steering for its own sake
- the redesigned methods are generic recipes that can be applied to any identified sparse region
- the full-manifold discovery layer still defines the targets
- the same audit and control logic still governs every pilot

The method class changes from:

- `global metadata-linked concept axes`

to:

- `region-targeted steering directions for sparse-manifold enrichment`

That is fully aligned with the project’s stated motivation.

## Feasible implementation plan

### Step 1: freeze the current bank

Do not launch more pilots from the current global axis bank unless needed for a specific ablation.

### Step 2: add a small new builder

Implement a new script, preferably:

- `embedding_elm/open-elm/cav_axis/build_local_steering_directions.py`

Responsibilities:

- load embeddings and metadata/split manifest
- choose target cluster
- fit one of the direction families
- save:
  - direction vector(s)
  - fit metadata
  - cluster statistics
  - train/test row counts used

### Step 3: reuse the shifted dataset builder

Extend `build_shifted_embedding_dataset.py` minimally so it can accept:

- a saved direction vector path
- a direction mode label

instead of only axis-bank indices

This avoids duplicating the shifted-dataset pipeline.

### Step 4: start with one redesigned target

First redesigned target:

- `cluster29`

Reason:

- strongest first-generation signal
- best target occupancy signal among current CAV pilots
- clinically interpretable enough to serve as the first redesign test

### Step 5: keep the current evaluation stack unchanged

For the redesigned pilot, run exactly the current evaluation stack:

1. shifted generation
2. matched vanilla subset
3. norm-matched random shift
4. audit
5. held-out real-vs-synthetic coverage
6. target-cluster occupancy check

### Step 6: only then decide scale-up

Do not move to LLM judge/editor until a redesigned direction wins against both controls.

## Recommended immediate next implementation

1. implement `build_local_steering_directions.py`
2. extend `build_shifted_embedding_dataset.py` to accept saved direction vectors
3. run a first redesigned pilot on `cluster29`
