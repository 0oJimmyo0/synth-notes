# CAV Axis Bank

1. retrain ELM based on MIMIC III (1M-2M)
2. enforce hidden data pattern didn't handled by MIMIC data, alter the axis to enrich that semantic space
3. generate more representative data (if it can covert the MIMIC data blank space. identity set of concepts, cover all disease domain)
4. start with MIMIC, if we should enrich and expand the data

## Purpose

This folder is for building a more principled Concept Activation Vector (CAV) workflow for the ELM project.

Our ELM generates synthetic discharge-style notes from patient/note embeddings. The long-term goal is not just to generate more notes, but to generate a much richer synthetic corpus that can be used for continued pretraining before downstream clinical NLP tasks such as NER.

For this project, CAVs are not only an interpretability tool. We want to use them as controlled directions in embedding space so that one real embedding can be turned into several nearby but meaningfully different embeddings before ELM inference. This helps us:

- increase the number of usable input embeddings for generation
- diversify generated notes beyond decoder randomness alone
- steer generation along demographic or cohort-related factors that may reflect bias or confounding in the original data
- reduce over-concentration on the empirical distribution of the original cohort
- create a richer pretraining corpus for later model adaptation

In short, this folder is about moving from:

- one embedding -> one note with sampling noise

to:

- one embedding -> multiple controlled embedding variants -> multiple synthetic notes

## What We Had Before

The older CAV experiments in the repo used one concept at a time, mainly age and gender:

1. define a binary concept dataset
2. fit a linear classifier in embedding space
3. use the classifier coefficient vector as the CAV
4. shift an embedding with `e_tilde = e + alpha * u`
5. feed the shifted embedding into ELM and decode text

That was useful as a proof of concept, but it has some limitations:

- it is manual and concept-by-concept
- it does not give a principled answer for how many axes to keep
- it does not tell us whether the selected axes cover enough of the metadata-relevant semantic space
- it does not directly support a scalable bank of axes for large synthetic-note generation

## What We Have Done Here

This folder upgrades the earlier workflow into a small axis-bank pipeline:

1. join embeddings with structured factor labels
2. fit a multi-factor linear probe matrix
3. compress that coefficient matrix with SVD
4. obtain an orthogonal bank of axes
5. choose a recommended number of axes using held-out recovery rather than intuition alone
6. audit how embeddings move under each axis before using them for generation

This gives us a cleaner answer to:

- which axes should we use?
- how many axes should we keep?
- are these axes interpretable or confounded?
- do these axes support our goal of richer synthetic-note generation?

## Why This Fits Our Goal

Our synthetic notes are intended for large-scale pretraining, potentially at 200k to 2M notes, before downstream tasks like NER.

For that use case, the important thing is not just surface-form diversity. We want diversity in the embedding inputs themselves so the generated corpus covers a broader and better-controlled semantic region of the training space.

This axis-bank approach helps because:

- it creates structured variation in embedding space, not only variation from decoding temperature or seed
- it lets us target factors related to demographic balance, cohort bias, and confounding variables
- it lets us generate multiple variants per patient/note embedding while staying closer to the learned embedding manifold than arbitrary noise
- it provides an audit trail for why an axis was selected and whether it should be trusted

For pretraining, this is important because richer notes can expose the model to a wider range of entity mentions, phrasing, clinical framing, and subgroup-related language patterns. That should be more useful than simply oversampling the same original embedding distribution with different random seeds.

## Technical Design

### Inputs

The scripts assume:

- `sentence_embeddings.npy`: embedding matrix with shape `(n_notes, d)`
- `sentence_embeddings_metadata.csv`: row-aligned metadata from the embedding step
- `factors.csv`: structured cohort labels to join onto metadata

Typical join keys are:

- `subject_id,hadm_id`
- `note_id`

The factor table should have one row per join key.

### Main Scripts

- `fit_axis_bank.py`
- `audit_axis_bank.py`
- `common.py`

### Current Method

`fit_axis_bank.py` does the following:

1. load note embeddings and metadata
2. join them to a structured factor table
3. keep complete cases for the requested factor columns
4. encode factors into supervised targets
5. fit a regularized linear multi-output model
6. treat the full coefficient matrix as the metadata-predictive subspace
7. run SVD on that matrix
8. save the resulting orthogonal axes and summaries

The reason for using a linear model first is that linear probes give explicit directions in embedding space, which is exactly what we need for steering. The SVD step then turns many probe outputs into a smaller and cleaner axis bank.

### Axis Selection Logic

The current axis-selection logic in this folder is:

- fit the full multi-factor linear probe
- compute its coefficient matrix
- use SVD to find the dominant orthogonal directions
- evaluate how well the top `k` axes recover the held-out predictive performance of the full probe
- choose `recommended_k` as the smallest `k` that keeps enough predictive signal

So the number of axes is not chosen by guesswork. It is chosen by a held-out recovery criterion.

### Outputs

`fit_axis_bank.py` writes:

- `axis_bank.npz`
- `target_manifest.csv`
- `heldout_target_metrics_full.csv`
- `heldout_axis_count_curve.csv`
- `split_manifest.csv`
- `axis_bank_summary.json`

Important outputs:

- `axis_bank.npz`: the saved axis bank used later for steering
- `axis_bank_summary.json`: includes `recommended_k`
- `heldout_target_metrics_full.csv`: tells us which factors are actually predictable
- `heldout_axis_count_curve.csv`: tells us how many axes are enough

### Audit Logic

`audit_axis_bank.py` uses the saved bank to test how held-out embeddings move under different steering strengths `alpha`.

This is useful for checking:

- whether an axis has a monotonic effect
- whether it mainly controls the factors we expect
- whether it also shifts unrelated factors, which would suggest confounding

If an axis moves many unrelated targets at once, we should be cautious about using it for synthetic generation.

## How To Use It

### Step 1: Prepare a factor table

Create a `factors.csv` aligned to the embedding metadata, for example with columns like:

- `subject_id`
- `hadm_id`
- `age`
- `gender`
- `race`
- `insurance`
- `admission_type`
- `service`
- `icu_flag`
- `los_bin`
- `comorbidity_bin`

This table should contain factors that matter for:

- demographic balance
- cohort bias
- confounding structure
- note style variation
- clinical diversity relevant to pretraining

### Step 2: Fit the axis bank

Example:

```bash
module load miniconda
conda activate elm

python cav_axis/fit_axis_bank.py \
  --embeddings_path /path/to/sentence_embeddings.npy \
  --metadata_path /path/to/sentence_embeddings_metadata.csv \
  --factors_path /path/to/factors.csv \
  --join_cols subject_id,hadm_id \
  --factor_cols age,gender,race,insurance,admission_type,service \
  --continuous_factors age \
  --categorical_factors gender,race,insurance,admission_type,service \
  --group_col subject_id \
  --output_dir cav_axis/output/demo_run
```

After this, inspect:

- `axis_bank_summary.json`
- `heldout_target_metrics_full.csv`
- `heldout_axis_count_curve.csv`

### Step 3: Audit the axes

Example:

```bash
module load miniconda
conda activate elm

python cav_axis/audit_axis_bank.py \
  --bank_dir cav_axis/output/demo_run \
  --embeddings_path /path/to/sentence_embeddings.npy \
  --metadata_path /path/to/sentence_embeddings_metadata.csv \
  --factors_path /path/to/factors.csv \
  --join_cols subject_id,hadm_id
```

Then inspect:

- `steering_audit.csv`
- `steering_trends.csv`
- `steering_summary.json`

### Step 4: Use the bank during ELM inference

The intended inference workflow is:

1. load a base embedding `e`
2. load `axis_bank.npz`
3. choose one or more trusted axes from the bank
4. choose a steering magnitude `alpha`
5. construct a shifted embedding such as:

```text
e_tilde = normalize(e + alpha * axis_j)
```

or a small combination:

```text
e_tilde = normalize(e + alpha1 * axis_j + alpha2 * axis_k)
```

6. feed `e_tilde` to ELM instead of the original embedding
7. generate notes with several seeds / decoding settings

This part is conceptually ready, but it is not yet fully wired into the clinic-note generation script as a standard end-to-end path.

## Recommended Practical Strategy

For richer synthetic notes, a reasonable generation strategy is:

- keep some unshifted embeddings as anchors
- generate a few shifted variants per embedding
- use only the top trusted axes, not every saved axis
- use modest `alpha` values at first
- combine axis steering with multiple decoding seeds

That gives you both:

- semantic diversity from embedding shifts
- surface-form diversity from sampling

This is usually better than relying on decoding randomness alone.

## What Still Needs To Be Done

This folder provides the axis-learning and audit part, but several important tasks remain.

### 1. Build the real factor table for the clinic-note cohort

This is the main missing input. We still need a high-quality structured table of factors aligned to the embedding metadata.

### 2. Decide which factors belong in the bank

We should choose factors that matter both for fairness and for pretraining diversity. Demographics alone are probably not enough. We likely also want note-style and clinical-content factors.

### 3. Run the bank on the actual discharge-note embedding cohort

So far the code is ready, but the real experiment still needs to be run on your actual embedding files and metadata.

### 4. Interpret the learned axes carefully

Not every dominant axis should automatically be used. Some may be too confounded, too weak, or too hard to interpret.

### 5. Integrate axis steering into synthetic-note generation

We still need to connect `axis_bank.npz` directly to the clinic-note generation script so that shifted embeddings can be generated automatically during ELM inference.

### 6. Decide the generation policy

We still need concrete choices for:

- how many axes to use per embedding
- how large `alpha` should be
- whether to use single-axis or multi-axis perturbations
- how many seeds per shifted embedding
- what proportion of the final corpus should be shifted vs unshifted

### 7. Validate the generated corpus for pretraining

Before using the notes for pretraining, we should evaluate:

- factor distribution after generation
- lexical and semantic diversity
- clinical concept / entity coverage
- whether steering reduced or worsened dataset imbalance
- whether the synthetic corpus improves downstream pretraining utility

### 8. Measure utility on the downstream task

The final goal is not just a prettier synthetic corpus. The real test is whether pretraining on these richer synthetic notes improves downstream NER or related tasks.

## Current Status

This folder now gives us a principled starting point for:

- selecting axes from structured cohort factors
- deciding how many axes to keep
- auditing whether the axes are interpretable
- preparing controlled embedding perturbations for richer ELM generation

The main next step is operational: run this on the real clinic-note embedding cohort, inspect the learned bank, and then integrate trusted axes into the synthetic-note generation pipeline.
