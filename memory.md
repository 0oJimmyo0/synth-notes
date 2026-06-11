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

- `embedding_elm/open-elm/audit_vanilla_generation.py`

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
