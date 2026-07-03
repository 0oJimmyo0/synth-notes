# Phase 2b Redesign Plan

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

