# Project Memory

## Current Goal

Build a clinically grounded synthetic discharge-summary pipeline on the new MIMIC-IV note/HADM-aligned cohort, with:

- vanilla ELM generation on the held-out cohort,
- structured row-level manifest output,
- leakage-aware evaluation,
- later coverage analysis, CAV steering, and optional LLM editing.

## Core Project Framing

The current research plan positions the project as:

- real MIMIC-IV discharge summaries -> embeddings -> ELM decoding -> optional CAV steering -> optional constrained LLM editor -> evaluation
- main near-term focus: validate vanilla ELM generation on the full ~300k-note cohort
- target framing: clinically grounded generation + validation protocol, not just a model demo

Important conceptual points from the plan:

- coverage means occupancy of the empirical real-note embedding manifold, not all of 1024-d space
- CAV is meant to be structured, metadata-linked steering, not random noise
- downstream NER is later, not the immediate endpoint
- Yale or another external reference cohort is proposed for external coverage comparison

## Data / Cohort State

New main cohort path:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task`

Key counts:

- note/HADM cohort size: ~331,793 rows
- filtered train: 265,434
- filtered dev: 33,179
- filtered test: 33,180

Embedding metadata path:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings_metadata.csv`

Important note:

- current embeddings are reusable and do not need regeneration just because the split is note-level
- if we want a patient-level split later, we can usually reuse the same embeddings and rebuild datasets/splits

## Training State

ELM training on the new full cohort is complete.

Checkpoint:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/elm_training_outputs/filtered_training/checkpoint-8215`

Training log confirming success:

- `embedding_elm/open-elm/log/train_clinic_notes_1546027.out`

## Generation State

The checked-in generation launcher originally pointed to an older scratch pipeline and old checkpoint (`checkpoint-1746`). We updated it to use the new shared path and new checkpoint.

Updated launcher:

- `embedding_elm/open-elm/generate_synthetic_notes.slurm`

Important launcher fix:

- use canonical `open-elm` script directory, not `SLURM_SUBMIT_DIR`
- this fixed backbone path resolution for `initial_elm_model`

Backbone path:

- `/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/initial_elm_model`

Old running job:

- job `1989100` was launched before the manifest-aware generator was in place
- it is an old-format run and will only produce plain-text output, not the new Phase 1 manifest

Output text path used for vanilla baseline:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/synthetic_notes/synthetic_notes_test_vanilla_seed42.txt`

Conclusion:

- to satisfy the new Phase 1 manifest requirement, vanilla generation should be rerun with the updated generator

## Manifest Work

Manifest implementation is done in code.

Main script:

- `embedding_elm/open-elm/generate_synthetic_notes.py`

What it now supports:

- plain-text note output preserved
- incremental JSONL manifest output during generation
- CLI args:
  - `--manifest_output`
  - `--generation_condition`
  - `--split`
  - `--split_manifest_path`
  - `--append_manifest`

Manifest design decisions:

- join by row alignment, not by plain text note numbering
- use row-level split manifest for provenance and leakage flags
- include stable provenance fields like generation id, source ids, checkpoint/backbone paths, seed, decoding params, config snapshot, package versions, script path, git commit if available
- include immediate quality flags:
  - word count
  - char count
  - success
  - empty output
  - too short
  - repetition/collapse
- include forward-compatible nullable fields for future CAV/editor conditions

Manifest output default pattern:

- `<output_stem>_manifest.jsonl`

Validation built into generator:

- manifest row count must equal generated note count
- no duplicate generation ids
- row order must match generation order
- decoding params must match run config

## Coverage Mapping Prep

Phase 1 coverage mapping has now been upgraded from a guard-only stub to a full layered prep/evaluation script:

- `embedding_elm/open-elm/cav_axis/prepare_coverage_mapping.py`

Current design choices:

- the primary quantitative analysis space is the full normalized 1024-d BGE embedding space
- PCA / UMAP are visualization only and should not be used for scientific coverage claims
- `real_only_precompute` remains available now for held-out test-only real-note manifold preparation
- `real_all_filtered_precompute` now supports manifold discovery across the entire filtered real cohort (train/dev/test)
- `real_vs_synthetic` is still guard-blocked unless:
  - vanilla audit status is `PASS` or `CAUTION`
  - synthetic manifest exists
  - synthetic embeddings exist
  - synthetic manifest row count matches synthetic embedding row count
  - synthetic manifest row order and `dataset_row_id` exactly match filtered `encoded_testing_filtered`
  - leakage flags are present

What `real_vs_synthetic` now computes when guards pass:

- real vs synthetic cluster occupancy over real-fitted clusters
- low-density real-cluster coverage
- nearest-real-neighbor distance summaries
- real-to-synthetic coverage / recall style metrics
- synthetic-to-real precision / on-manifoldness style metrics
- synthetic density proxy
- approximate MMD and energy distance on sampled embeddings
- leakage-aware summaries for:
  - full test
  - patient-disjoint
  - patient-overlap
- subgroup summaries if metadata are available or joined in

Main outputs now targeted:

- `coverage_real_vs_synthetic.json`
- `coverage_real_vs_synthetic.md`
- `cluster_occupancy_real_vs_synthetic.csv`
- `low_density_cluster_coverage.csv`
- `nearest_real_distance_summary.csv`
- `coverage_full_vs_patient_disjoint.csv`
- optional `subgroup_coverage_summary.csv`
- optional `coverage_umap_real_vs_synthetic.png`

Important caveat:

- current filtered-aligned split manifest still does not include age / insurance / LOS / ICU style subgroup metadata, so subgroup claims remain blocked unless an extra metadata table is joined with `--extra_metadata_path`

Clarification:

- `real_all_filtered_precompute` is for exploratory discovery and CAV-target planning across all filtered real embeddings
- `real_only_precompute` plus `real_vs_synthetic` remain the official held-out evaluation path

Launcher additions:

- `embedding_elm/open-elm/cav_axis/coverage_real_all_filtered.slurm`
  - whole-filtered-real manifold discovery across filtered train/dev/test
- `embedding_elm/open-elm/cav_axis/coverage_real_vs_synthetic.slurm`
  - held-out real-vs-synthetic coverage comparison after vanilla audit outputs exist

Standalone synthetic re-embedding fix:

- `embedding_elm/open-elm/cav_axis/reembed_generated_notes.py`
- `embedding_elm/open-elm/cav_axis/reembed_generated_notes.slurm`

Purpose:

- generate `generated_note_embeddings_bge_large.npy` directly from the vanilla generation manifest
- avoid rerunning the full audit when only the synthetic embedding matrix is missing
- unblock `real_vs_synthetic` coverage mapping

## Vanilla Audit Runtime Update

The first manual run of:

- `embedding_elm/open-elm/cav_axis/audit_vanilla_generation.py`

showed that re-embedding generated notes on the login node was slow enough to warrant a GPU-backed Slurm path.

Audit script improvements now in place:

- added `--embedding_device {auto,cpu,cuda}`
- added `--embedding_batch_size`
- audit summary now records requested vs resolved embedding device and embedding batch size

## Subgroup Metadata Build

Subgroup metadata discovery and build are now in place for Phase 1 coverage interpretation.

Source tables confirmed under MIMIC-IV:

- `.../hosp/patients.csv`
  - `subject_id`, `gender`, `anchor_age`, `anchor_year`
- `.../hosp/admissions.csv`
  - `subject_id`, `hadm_id`, `admittime`, `dischtime`, `admission_type`, `insurance`, `race`
- `.../hosp/services.csv`
  - `subject_id`, `hadm_id`, `transfertime`, `curr_service`
- `.../icu/icustays.csv`
  - `subject_id`, `hadm_id`, `stay_id`, `los`

New builder script:

- `embedding_elm/open-elm/cav_axis/build_subgroup_metadata.py`

Outputs written to:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/subgroup_metadata/subgroup_metadata_full.csv`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/subgroup_metadata/subgroup_metadata_filtered.csv`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/subgroup_metadata/subgroup_metadata_summary.json`

Canonical subgroup fields now emitted for downstream coverage mapping:

- `age_bin`
- `sex_gender`
- `race_ethnicity`
- `insurance`
- `admission_type`
- `service`
- `los_bin`
- `icu_flag`

Important fix:

- the builder originally wrote `insurance_group`, `admission_type_group`, and `service_group` only
- it now also writes canonical coverage-compatible fields `insurance`, `admission_type`, and `service`
- this avoids silent empty subgroup summaries in `prepare_coverage_mapping.py`

Current summary snapshot:

- full rows: `331,793`
- filtered rows: `328,585`
- missingness is very low for all major subgroup fields
- filtered cohort examples:
  - age bins: mostly `40-64`, `65-79`, `80+`
  - services dominated by `medicine` then `surgery`
  - ICU flag true for about `64.7k` filtered rows

Important interpretation decision:

- candidate CAV target regions should be defined from the whole filtered real embedding manifold, not from vanilla synthetic gaps alone
- vanilla held-out real-vs-synthetic coverage should be used as a validation/prioritization signal, not as the primary discovery space
- practical next step: rerun `real_all_filtered_precompute` with `--extra_metadata_path subgroup_metadata_filtered.csv` so sparse/dense clusters can be interpreted by subgroup enrichment

New launcher:

- `embedding_elm/open-elm/cav_axis/audit_vanilla_generation.slurm`

Intent:

- run the BGE re-embedding step on GPU for the manifest-backed vanilla audit
- avoid long login-node runs
- keep output paths aligned with:
  - `synthetic_notes_test_vanilla_seed42_manifest.jsonl`
  - `encoded_testing_filtered`
  - `split_manifest_note_level.csv`
  - `generation_audit/vanilla_test_seed42`

## Leakage Audit

Leakage audit script added:

- `embedding_elm/preprocessing_mimiciv/audit_note_hadm_split_leakage.py`

Outputs:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level.csv`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_overlap_summary.csv`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_leakage_audit_summary.json`

Leakage results:

- split is note-level, not patient-level
- train/dev/test are fully note-disjoint
- train/dev/test are fully HADM-disjoint
- train/dev/test are not patient-disjoint

Important numbers:

- train vs dev subject overlap: 17,835
- train vs test subject overlap: 17,892
- patient-disjoint dev rows relative to train: 9,883 / 33,179 (29.79%)
- patient-disjoint test rows relative to train: 9,949 / 33,180 (29.98%)

Interpretation:

- held-out rows are all new notes and new admissions
- only ~30% of dev/test rows are from patients never seen in training
- this means future reporting should stratify by `patient_disjoint_from_train`

Important technical correction discovered later:

- the old leakage manifest matched the pre-filter split, not the actual filtered HF datasets used by ELM training/generation
- split happens first in `prep_hf_dataset/post_emb_dataprep.py`
- long-sequence filtering happens later in `open-elm/filter_long_sequences.py`
- therefore `encoded_*_filtered` can drift from the original split manifest unless leakage audit is regenerated against the filtered datasets

Current full vs filtered counts:

- train: `265,434 -> 262,895` (`-2,539`)
- dev: `33,179 -> 32,847` (`-332`)
- test: `33,180 -> 32,843` (`-337`)

This explains why manifest-enabled vanilla generation failed when it tried to join:

- `encoded_testing_filtered` now has `32,843` rows
- old `split_manifest_note_level.csv` still had `33,180` test rows

New leakage-audit design:

- keep a stable whole-cohort source manifest for all embedding rows and split assignments
- regenerate a filtered-aligned split manifest that matches `encoded_training_filtered`, `encoded_dev_filtered`, and `encoded_testing_filtered`
- use the filtered-aligned manifest as the canonical join target for vanilla generation and Phase 1 audit
- keep generation manifests separate per run / condition; do not overwrite source provenance
- refreshed script now aims to emit:
  - `split_manifest_note_level.csv` for filtered-aligned downstream joins
  - `split_manifest_note_level_full.csv` for whole-cohort provenance
  - `split_manifest_removed_by_filter.csv` for dropped rows
  - filtered and full overlap summaries

Implication for future CAV/editor work:

- source manifest should stay fixed
- each generation run should write its own generation manifest keyed by `source_row_id` / `embedding_row_id`
- CAV-specific fields (axis id, alpha, normalization flag) belong in the generation manifest, not the source manifest

What this means for the manifest:

- every generated row should carry:
  - `patient_disjoint_from_train`
  - `hadm_disjoint_from_train`
  - `note_disjoint_from_train`
  - overlap flags as available

## Split Logic

Current split is:

- note-level random split
- not patient-level

So if a stricter patient-level evaluation or retraining setup is desired later:

- reuse current embeddings
- create a patient-level split manifest
- rebuild train/dev/test datasets from that split
- optionally retrain ELM on that patient-level split

## Research Plan Status

The updated `research_plan.tex` is stronger than the original version.

Good additions:

- clearer novelty positioning
- stronger embedding-space coverage story
- Yale/external reference cohort idea
- optional constrained LLM editor
- stronger baseline set:
  - real held-out
  - vanilla ELM
  - GPT-only
  - ELM + editor
  - random shift
  - CAV-steered

Important caution:

- the plan is broader now, so Phase 1 should still stay focused on:
  - vanilla generation
  - manifest
  - leakage-aware audit
  - basic faithfulness / coverage infrastructure

## Recommended Next Steps

Immediate:

1. rerun vanilla generation with the manifest-aware script
2. verify manifest row counts and output integrity
3. run vanilla audit stratified by leakage flags, especially patient-disjoint vs patient-overlap

After that:

4. implement structured vanilla generation audit outputs
5. build coverage infrastructure:
   - real-note clustering
   - density / low-coverage region detection
   - subgroup coverage summaries
6. prepare `factors.csv` for axis-bank work
7. fit and audit CAV axis bank

## Exact Rerun Command

To rerun manifest-enabled vanilla generation:

```bash
sbatch \
  --export=ALL,\
OUTPUT_FILE=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/synthetic_notes/synthetic_notes_test_vanilla_seed42.txt,\
MANIFEST_FILE=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/synthetic_notes/synthetic_notes_test_vanilla_seed42_manifest.jsonl,\
SPLIT_NAME=test,\
GENERATION_CONDITION=vanilla,\
SPLIT_MANIFEST_PATH=/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level.csv,\
SEED=42 \
/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/generate_synthetic_notes.slurm
```

If the old non-manifest job is still running and should be stopped first:

```bash
scancel 1989100
```

## Git / Repo Notes

Important repo location:

- the Git repo is at `/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm`
- the parent `/files` directory is not the Git repo root

Local Git config set in the `embedding_elm` repo:

- `user.name = Mingyang Jiang`
- `user.email = mingyang.jiang@vanderbilt.edu`

This was done to fix the VS Code sidebar Git warning for that repo.

## Meeting-Ready Summary

Useful concise summary for future meetings:

- full-cohort ELM training on the new ~332k note/HADM-aligned MIMIC-IV cohort is done
- leakage audit is implemented and quantified
- current split is note- and admission-disjoint but not patient-disjoint
- manifest-aware baseline generation code is ready
- current old-format generation run should be rerun to produce the official Phase 1 JSONL manifest

## Vanilla Audit Pipeline

Phase 1 audit script added:

- `embedding_elm/open-elm/cav_axis/audit_vanilla_generation.py`

Purpose:

- audit manifest-driven vanilla generation only
- do not start CAV, LLM editor, or downstream NER here

Inputs:

- `--manifest_path`
- `--dataset_path`
- optional `--split_manifest_path`
- `--output_dir`
- optional `--embedding_model_name`
- optional `--sample_size_for_manual_review`

Main audit stages:

1. manifest integrity checks
2. basic quality audit
3. faithfulness audit by re-embedding generated notes with BGE
4. lightweight privacy / memorization screen
5. concise PASS / CAUTION / FAIL summary

Expected outputs:

- `generation_audit_baseline.json`
- `generation_audit_baseline.md`
- `vanilla_quality_table.csv`
- `vanilla_faithfulness_table.csv`
- `patient_disjoint_vs_full_metrics.csv`
- `manual_review_sample.csv`

What the script checks:

- row count alignment between manifest and `encoded_testing_filtered`
- unique `generation_id`
- required non-null generation/provenance fields where available
- `split == test`
- `generation_condition == vanilla`
- leakage flags if split manifest is provided
- decoding parameter consistency warnings
- no accidental source-note text columns in the manifest

What the script computes:

- empty / too-short / repetition-collpase rates
- empty / too-short / repetition-collapse rates
- word / char count summaries
- rough section-header sanity check
- source-vs-generated embedding cosine
- source self-retrieval top-1 / top-5 / top-10 recovery
- leakage-stratified metrics for:
  - full test
  - patient-disjoint test
  - patient-overlap test
- exact duplicate generated notes
- nearest-train embedding screen
- exact duplicate vs train text if train text is accessible
- simple PHI-like regex flags

Validation status:

- script compiled successfully with `python -m py_compile`

Practical note:

- this audit requires the manifest-enabled vanilla generation rerun
- the currently running old-format job `1989100` does not produce the required JSONL manifest

## Research Plan Update

`research_plan.tex` was updated to reflect:

- two-layer manifest architecture
- filtered-aligned leakage manifest as canonical downstream artifact
- current filtered cohort counts (`262,895 / 32,847 / 32,843`)
- vanilla rerun blocked on refreshed filtered-aligned manifest

## June 10 launcher changes

Generation code:

- `open-elm/generate_synthetic_notes.py` now supports shard slicing with:
  - `--start_index`
  - `--end_index`
- shard runs preserve global `generation_index`, note numbering, and `dataset_row_id`

Generation launcher:

- `open-elm/generate_synthetic_notes.slurm` was rewritten to:
  - request up to 48h wall time
  - request 2 H100 GPUs by default
  - use 8 CPUs / 96G RAM
  - run 1 or 2 shard workers in parallel (`NUM_SHARDS`, default `2`)
  - merge shard text outputs back into one ordered note file
  - merge shard manifests back into one ordered JSONL manifest
  - validate merged note count and manifest row count against dataset length

Operational note:

- a 2-GPU request alone would not speed generation unless the script itself shards work
- the new launcher implements that sharded workflow explicitly

## Phase 1 Prep Folder

Prepared under:

- `embedding_elm/open-elm/cav_axis/phase1_prep/`

Files added:

- `run_audit_vanilla_generation.sh`
  - exact post-generation audit command for the official vanilla manifest
- `manual_review_rubric_50.csv`
  - 50-row blank rubric with fields for readability, discharge-summary structure, repetition/collapse, section layout, hallucination, PHI-like leakage, source-conditioned feel, and pass/fail
- `manifest_integrity_template.csv`
- `quality_metrics_template.csv`
- `faithfulness_metrics_template.csv`
- `full_vs_patient_disjoint_template.csv`
- `privacy_memorization_warnings_template.csv`
- `coverage_mapping_config.yaml`
  - real held-out coverage prep only; blocked until synthetic notes are re-embedded and vanilla audit is PASS/CAUTION
- `factors.csv`
  - candidate future CAV factor spec, not a fitted axis-bank input yet

Additional Phase 1 prep script:

- `open-elm/cav_axis/prepare_coverage_mapping.py`
  - supports:
    - `--mode real_only_precompute`
    - `--mode real_vs_synthetic`
  - `real_vs_synthetic` is guard-blocked unless vanilla manifest, vanilla audit, and synthetic embeddings all exist and agree

## Incomplete Vanilla Output Quick Read

The old timeout-truncated vanilla file still provides a useful qualitative preview:

- path: `.../synthetic_notes_test_vanilla_seed42.txt`
- note headers present: `22,592`
- first note clearly has discharge-summary style fields such as:
  - `Admission Date`
  - `Discharge Date`
  - `Service`
  - `Allergies`
  - `Chief Complaint`
  - `Major Surgical or Invasive Procedure`

Interpretation:

- the model is producing free-text synthetic clinical notes in discharge-summary-like format
- but the incomplete run is not an official baseline because it timed out and has no valid manifest

## Coverage Prep Status

Completed real-only precompute under:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/coverage/real_only_precompute`

Outputs created:

- `real_cluster_assignments.csv`
- `real_cluster_summary.csv`
- `real_group_cluster_summary.csv`
- `real_subgroup_cluster_summary.csv`
- `coverage_real_only_precompute.json`
- `coverage_real_only_precompute.md`

Real-only precompute summary:

- held-out real rows: `32,843`
- clusters: `50`
- row-count alignment: `true`
- warning: subgroup metadata fields like age/insurance/LOS/ICU are not present yet in the filtered split manifest, so subgroup cluster summary is currently empty

Current official vanilla job:

- Slurm job `1996960`
- state at last check: `PENDING`
- submitted with refreshed filtered-aligned split manifest and 2-GPU sharded launcher
