# Project Memory

## Current Stage: Closed-Loop Output-Space Enrichment Validation

Input-space CAV/local transport and decoder-adaptation pilots are frozen as negative/partial findings: pre-decode geometry could improve, but the ELM decode--BGE re-embed map did not preserve sparse-basin identity reliably enough for input-space steering to be the Phase 2 mechanism.

The active Phase 2 method is therefore closed-loop output-space enrichment: generate vanilla ELM candidates from real held-out anchors, re-embed every candidate with BGE, and retain only candidates that satisfy target-region, faithfulness, quality, privacy-risk, and diversity gates. This controls the final generated-note embedding instead of assuming an edited input embedding survives decoding.

Frozen validation run:

- run directory: `closed_loop_output_enrichment/cluster29_basin_v1_test8_2h100_strict_quality_max8192_default_rerun`
- 256 target-basin held-out anchors; 8 candidates per anchor; 2,048 candidates total,
- 106 strict accepted notes (`5.18%` acceptance), compared with 14 (`0.68%`) before the `8192`-token default-prompt update,
- accepted notes: mean source cosine `0.8231`, structure-pass rate `1.00`, clinical-sanity-pass rate `1.00`, accepted EOS rate `0.9906`,
- accepted notes land across the pooled local basin `9/17/29/45`; exact cluster 29 remains secondary rather than the sole success metric.

Interpretation and current guardrail:

- this is evidence that output-space selection can enrich a pre-defined local sparse basin,
- it is not yet evidence that the notes are clinically reliable synthetic discharge summaries,
- clinical review found narrative stitching, diagnosis/procedure mismatch, medication/temporal inconsistencies, and generic boilerplate in some accepted notes,
- do not scale to `N=16` or begin LLM editing/downstream NER until fair same-anchor vanilla controls and structured source-faithfulness review are complete.

Current validation tooling:

- `cav_axis/prepare_closed_loop_validation_pack.py` freezes accepted review sheets, blinded accepted/near-miss/vanilla sets, transition tables, and gate routes,
- `cav_axis/build_matched_vanilla_8192_control.py` builds a one-draw, unselected, default-prompt `8192`-token vanilla control from the same 106 accepted anchors and then creates an anchor-level paired analysis,
- generated anchor manifest: `.../cluster29_basin_v1_test8_2h100_strict_quality_max8192_default_rerun/matched_vanilla_8192_control/matched_vanilla_8192_anchor_manifest.csv`.

Immediate next step:

1. run the matched one-draw vanilla control with the frozen `8192`-token/default decoding policy;
2. run paired accepted-vs-vanilla analysis and blinded clinical/source-faithfulness review;
3. decide whether the next method is a clinical reranker/quality model or a constrained repair/editor, based on explicit error categories rather than more candidate scale.

## Current Goal

Build a clinically grounded synthetic discharge-summary pipeline on the new MIMIC-IV note/HADM-aligned cohort, with:

- vanilla ELM generation on the held-out cohort,
- structured row-level manifest output,
- leakage-aware evaluation,
- later coverage analysis, CAV steering, and optional LLM editing.

## Latest Phase 2b Finding

The current cluster-29 steering work has now produced a much sharper diagnostic result:

- on easy anchors and pooled-basin evaluation, shifted generations often stay in the local `29/9/17/45` neighborhood,
- but on a true hard-anchor set from the full held-out pool, the current linear local basin-margin steering family still yields `target_cluster_win_rate = 0.0`,
- even the best hard-anchor sweep setting (`alpha=0.30`, `margin_gamma=0.20`) only improves the mean target-vs-competitor margin from about `-0.087` to about `-0.031`; it does not flip the decision boundary,
- the dominant competitors remain clusters `9` and `45`, with `17` and sometimes `7` as nearby local alternatives.

This changes the interpretation of the task:

- the general basin-level framing remains scientifically useful,
- but the current \textbf{linear additive basin-margin steering family should be frozen as a negative/partial finding},
- the next redesign should be judged by whether it can actually beat the local competitor clusters on hard anchors before we spend more time decoding notes.

Current working hypothesis:

- `29/9/17/45` may form a clinically meaningful local basin,
- `7` is geometrically close but may be clinically different enough that it should be evaluated separately rather than automatically pooled.

Current next-step logic:

- keep basin-level evaluation as the regional success lens,
- use exact target-cluster win rate and target-vs-best-competitor margin as the pre-decode gate,
- redesign the operator around explicit competitor-aware boundary crossing instead of another broad alpha sweep,
- only scale a method after it shows a genuine hard-anchor win before decode and then survives decode -> re-embed.

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

## Latest Cluster-29 Steering Readout

Recent upper-bound and basin-margin pilots clarified the decoder behavior:

- \textbf{Upper-bound decode of true cluster-29 embeddings}
  - quality/audit acceptable,
  - still produced `0` exact cluster-29 occupancy after decode -> re-embed,
  - landed mostly in nearby clusters such as `17`, `9`, `7`, and `45`.

- \textbf{Centroid similarity analysis}
  - cluster `29` is extremely close in embedding geometry to `17`, `45`, `9`, and `7`,
  - so the failure is a local basin competition problem, not a random off-manifold jump.

- \textbf{Local basin-margin pilot}
  - new script added:
    - `embedding_elm/open-elm/cav_axis/build_local_basin_margin_dataset.py`
  - winning predecode condition:
    - `alpha=0.15`
    - `margin_gamma=0.05`
  - this improved local target pull predecode while preserving source cosine.

- \textbf{Post-decode basin-margin result}
  - held-out audit: `PASS`
  - coverage improved modestly on global/full metrics and low-density coverage,
  - but exact cluster-29 occupancy still remained `0`.

- \textbf{Official-basin comparison for basin-margin pilot}
  - best-cluster counts among `29/9/17/45/7`:
    - `9: 86`
    - `7: 63`
    - `45: 46`
    - `17: 39`
    - `29: 22`
  - target-29 win rate: `0.0859`
  - mean target-29 margin vs best competitor: negative

Conclusion:

- the basin-margin redesign is scientifically more meaningful than the earlier global-axis pilots,
- but it is still not enough to claim exact target-cluster enrichment,
- the project should now evaluate basin-level enrichment as the main Phase 2 target.

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

## 2026-06-23 Status Update

- `embedding_elm/research_plan.tex` has been updated to reflect the current official state:
  - vanilla held-out generation is complete
  - vanilla audit is `PASS`
  - whole-filtered-real coverage discovery is complete
  - held-out real-vs-vanilla coverage mapping is complete
  - the first `cluster25` CAV pilot should be treated as a pilot/control lesson, not the scale-up target
  - the next CAV retargeting candidates are clusters `16`, `29`, and `36`

- Current vanilla baseline evidence:
  - `32843 / 32843` manifest-backed held-out notes generated
  - median cosine `0.7892` overall
  - median cosine `0.7864` on patient-disjoint held-out rows
  - collapse rate `9.23%`
  - exact duplicate generated notes: `0`
  - exact duplicates vs train text: `0`
  - vanilla audit decision: ready for coverage mapping

- Current whole-real and held-out coverage evidence:
  - whole filtered real manifold precompute completed on `328585` rows across train/dev/test
  - vanilla held-out coverage:
    - full-test real-to-synthetic coverage `0.6283`
    - patient-disjoint real-to-synthetic coverage `0.6085`
    - low-density cluster coverage `10/10` on full-test and patient-disjoint subsets

- Current CAV decision rule:
  - do not scale a steering condition unless it beats both:
    - count-matched vanilla
    - norm-matched random shift
  - the next official steering step should be one carefully chosen pilot from the retargeted `16/29/36` bank, then compared again against matched controls

## 2026-06-25 CAV Diagnosis Update

- We completed the main first-generation cluster/axis steering pilots with matched vanilla and norm-matched random-shift controls:
  - `cluster11` with `axis1` negative steering
  - `cluster16` with `axis15`
  - `cluster25` with `axis15`
  - `cluster29` with `axis11`

- Current diagnosis:
  - several CAV pilots beat matched vanilla on some global held-out coverage metrics
  - but none of the completed pilots robustly beat norm-matched random shift strongly enough to justify scale-up from the current axis bank
  - target-cluster occupancy is often weak even when global coverage improves

- Practical interpretation:
  - the current global linear CAV bank produces broad manifold movement more reliably than clean target-cluster filling after decoding
  - therefore, this bank should now be treated as an exploratory steering/probing stage, not yet as the final synthetic-note enrichment engine

- Cluster-level summary:
  - `cluster11`: steering improved some coverage metrics vs matched vanilla, but vanilla remained best on target-cluster occupancy
  - `cluster16`: steering beat matched vanilla on held-out coverage, but not random shift on several key coverage/low-density metrics; target cluster occupancy remained effectively zero
  - `cluster25`: steering did not beat the strongest vanilla control and did not justify further scaling
  - `cluster29`: strongest current steering signal; beats matched vanilla and modestly improves target occupancy, but still does not separate cleanly enough from random shift on broader coverage

- Decision carried forward:
  - pause additional pilots from the current global axis bank
  - do not move the project’s main focus to LLM judge/editor yet
  - begin a Phase 2b redesign around more local and target-specific steering directions

- Phase 2b redesign options to prioritize:
  - cluster-local centroid directions
  - one-vs-rest cluster discriminative directions
  - local neighborhood residual directions
  - hybrid constrained steering with manifold-aware projection or norm/cosine preservation
  - stronger count-matched multi-seed vanilla controls for subset pilots

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

## CAV Prep Update

The CAV-axis folder now has explicit Phase 2 prep utilities for candidate clusters `11`, `20`, and `25`.

New scripts:

- `embedding_elm/open-elm/cav_axis/build_cav_factor_table.py`
- `embedding_elm/open-elm/cav_axis/rank_candidate_cav_clusters.py`
- `embedding_elm/open-elm/cav_axis/fit_axis_bank.slurm`
- `embedding_elm/open-elm/cav_axis/audit_axis_bank.slurm`

New outputs created:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/cav_factor_table_clusters_11_20_25.csv`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/cav_factor_table_clusters_11_20_25_summary.json`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/candidate_cluster_enrichment_11_20_25.csv`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/candidate_cluster_enrichment_11_20_25.md`

Important CAV findings:

- cluster `11` is strongly `medicine`, older-skewed, and Medicare-leaning
- cluster `20` is strongly `scheduled + surgery + ICU`
- cluster `25` is strongly `orthopedics`, younger-skewed, short-LOS, and mostly non-ICU

Important bug fix:

- `common.py` now preserves `embedding_row_id` when the factor table already contains it
- this fixed a real `fit_axis_bank.py` failure discovered during smoke testing

Smoke-test status:

- tiny sampled `fit_axis_bank.py` run completed after the `embedding_row_id` fix
- sampled `audit_axis_bank.py` previously failed because of join-type mismatch between `split_manifest.csv` and normalized string join keys
- `audit_axis_bank.py` now normalizes join-column dtypes from `split_manifest.csv` before merging

Operational note:

- factor/enrichment prep and axis-bank fitting are CPU jobs; no GPU is needed
- if using an interactive allocation, the recommended request remains:
  - `salloc --partition=devel --cpus-per-task=8 --mem=48G --time=06:00:00`

## Dedicated Cluster-20 Path

After targeted audit of the mixed `11/20/25` bank:

- `axis 15` behaved cleanly and bidirectionally for clusters `11` and `25`
- cluster `20` did move under steering, but its signal was spread across multiple axes (`0`, `1`, `8`, `11`) and strongly entangled with surgery / ICU / admission-type factors

Conclusion:

- do not broaden to many more clusters yet
- instead, create a dedicated cluster-20 bank to test whether `20` becomes cleaner when modeled alone

New wrappers added:

- `embedding_elm/open-elm/cav_axis/fit_axis_bank_cluster20.slurm`
- `embedding_elm/open-elm/cav_axis/audit_axis_bank_cluster20.slurm`

Other wrapper updates:

- `fit_axis_bank.slurm` now defaults to CPU `devel` and prints the exact command
- `audit_axis_bank.slurm` now supports env overrides for:
  - `TOP_TARGETS`
  - `OUTPUT_NAME`
  - safe `ALPHAS`

## CAV Readiness For Clusters 11 20 25

We moved from generic CAV scaffolding to a concrete Phase 2 prep path for candidate clusters `11`, `20`, and `25`.

New utilities added:

- `embedding_elm/open-elm/cav_axis/build_cav_factor_table.py`
  - builds a CAV-ready factor table from subgroup metadata plus whole-real cluster assignments
  - outputs binary columns:
    - `cluster_target_11`
    - `cluster_target_20`
    - `cluster_target_25`
  - preserves `split` and `patient_disjoint_from_train`
- `embedding_elm/open-elm/cav_axis/rank_candidate_cav_clusters.py`
  - formalizes subgroup enrichment using Fisher exact tests and BH-adjusted q-values
  - this addresses the earlier caution that raw `fraction_within_subgroup` values are not enough for claims
- `embedding_elm/open-elm/cav_axis/fit_axis_bank.slurm`
- `embedding_elm/open-elm/cav_axis/audit_axis_bank.slurm`

Real-data outputs created:

- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/cav_factor_table_clusters_11_20_25.csv`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/cav_factor_table_clusters_11_20_25_summary.json`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/candidate_cluster_enrichment_11_20_25.csv`
- `/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/candidate_cluster_enrichment_11_20_25.md`

Current cluster interpretations from the formal enrichment report:

- `cluster 11`
  - enriched for `service=medicine`
  - enriched for `age_bin=80+`
  - enriched for `insurance=medicare`
- `cluster 20`
  - strongly enriched for `admission_type=scheduled`
  - strongly enriched for `service=surgery`
  - strongly enriched for `icu_flag=True`
- `cluster 25`
  - strongly enriched for `service=orthopedics`
  - enriched for younger `age_bin=18-39`
  - enriched for short LOS and non-ICU stays

Important bug fixes discovered by smoke testing:

- `common.py`
  - `load_and_merge_tables()` now preserves canonical `embedding_row_id` even when the factor table also carries that column
  - `save_json()` now handles NumPy scalar types
- `audit_axis_bank.py`
  - normalizes join-column dtypes from `split_manifest.csv`
  - coalesces suffixed `split` columns after merge
- `build_cav_factor_table.py`
  - now coalesces `split` and `patient_disjoint_from_train` cleanly from merged inputs

Smoke-test status:

- tiny sampled fit completed successfully with:
  - `fit_axis_bank.py`
  - `audit_axis_bank.py`
- smoke output dir:
  - `/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/smoke_sample_tiny/output`

Interpretation:

- the current CAV code is now operational enough for a first real run on clusters `11/20/25`
- for paper claims, use the formal enrichment report as justification for target selection rather than only the subgroup cluster-summary proportions

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

## June 17 CAV-to-Generation Wiring

We now have the generic bridge from audited CAV axes to the existing ELM generation pipeline.

New script:

- `embedding_elm/open-elm/cav_axis/build_shifted_embedding_dataset.py`

Purpose:

- load a source HF dataset such as `encoded_testing_filtered`
- join filtered-aligned split manifest metadata and optional factor-table metadata
- select source rows with a generic pandas query such as `cluster_target_25 == 1`
- apply one or more saved axis directions from `axis_bank.npz` with one or more `alpha` values
- optionally L2-normalize after steering
- save a new HF dataset that still looks like the normal ELM input dataset:
  - `input_ids`
  - `domain_embeddings`
- also store steering provenance directly on each dataset row:
  - source row ids
  - leakage flags
  - `axis_id`
  - `axis_label`
  - `alpha`
  - `normalized_after_steering`
  - `random_shift_norm`
  - `post_edit_source_cosine`
  - `source_dataset_path`
  - `source_split`
  - `selection_query`
  - `steering_run_metadata_path`

Important implementation note:

- for the real filtered cohort, `dataset_row_id` is only unique within split
- the builder therefore prefers `split + dataset_row_id` when bridging into the filtered split manifest
- after that it prefers stronger global keys such as `source_row_id` or `embedding_row_id` for later joins

Generator-side wiring:

- `embedding_elm/open-elm/generate_synthetic_notes.py` now reads steering-related metadata directly from HF dataset rows
- those fields are propagated into the normal JSONL generation manifest
- `embedding_elm/open-elm/generate_synthetic_notes.slurm` now supports two modes:
  - vanilla mode with external `SPLIT_MANIFEST_PATH`
  - shifted-dataset mode with `SPLIT_MANIFEST_PATH=''`, relying on dataset-embedded metadata

Smoke-test result:

- a 2-row smoke build on `encoded_testing_filtered` with:
  - axis `15`
  - alpha `0.5`
  - `source_split=test`
  - `selection_query='cluster_target_25 == 1'`
  completed successfully
- output dataset rows preserved note ids and leakage metadata and carried:
  - `axis_id = 15`
  - `axis_label = axis_15__cluster_target_25::1.0`
  - `normalized_after_steering = True`

Practical next-step implication:

- the first real CAV-steered ELM pilot should now be:
  - build a shifted dataset from held-out `cluster_target_25 == 1` anchors
  - start with `axis 15`
  - use modest positive alphas such as `0.5` and `1.0`
  - reuse the existing `generate_synthetic_notes.py` / `generate_synthetic_notes.slurm` path instead of introducing a separate CAV-specific generator

## June 18 Shifted-Generation Launch Fix

First shifted-generation submission:

- Slurm job `2014782`
- state: `FAILED`
- elapsed: `00:00:01`
- log files:
  - `embedding_elm/open-elm/log/generate_synthetic_notes_2014782.out`
  - `embedding_elm/open-elm/log/generate_synthetic_notes_2014782.err`
  were both zero bytes
- no shifted synthetic-note outputs were created under the target `synthetic_notes` directory

Root cause found in launcher configuration:

- for shifted-dataset generation we tried to disable the vanilla split manifest with:
  - `SPLIT_MANIFEST_PATH=''`
- but `generate_synthetic_notes.slurm` used Bash `${VAR:-default}` expansion, which treats an empty string as missing and silently restores the vanilla default manifest path
- this is wrong for shifted datasets because:
  - shifted dataset rows: `660`
  - vanilla filtered test manifest rows: `32,843`

Launcher fix now applied:

- `embedding_elm/open-elm/generate_synthetic_notes.slurm`
  - changed:
    - from `${SPLIT_MANIFEST_PATH:-default}`
    - to `${SPLIT_MANIFEST_PATH-default}`
- result:
  - unset variable still uses the vanilla default
  - explicitly empty string now truly disables split-manifest injection and lets shifted datasets rely on their embedded metadata

Second shifted-generation submission:

- Slurm job `2014940`
- same symptom as `2014782`
  - `FAILED`
  - `00:00:01`
  - empty `.out` and `.err`
  - still no shifted synthetic-note outputs

Operational logging fix now applied:

- `embedding_elm/open-elm/generate_synthetic_notes.slurm`
  - no longer uses:
    - `#SBATCH --output=/dev/null`
    - `#SBATCH --error=/dev/null`
    - shell `exec > >(tee ...)`
- now uses native Slurm log targets:
  - `embedding_elm/open-elm/log/generate_synthetic_notes_%j.out`
  - `embedding_elm/open-elm/log/generate_synthetic_notes_%j.err`
- also adds a shell `trap ... ERR` line so batch-script startup failures report the failing line number directly

Interpretation:

- the previous empty logs were a launcher-observability problem on top of the real failure
- after this change, the next rerun should expose the actual startup error instead of failing silently

Third-launch diagnosis:

- the revealed startup error was:
  - `ERROR: Checkpoint not found at /gpfs/.../1_taskelm_training_outputs/.../checkpoint-8215`
- root cause:
  - the launcher still depended on `DATAHOME` ending with `/`
  - user export used `DATAHOME=.../1_task` without trailing slash
  - default checkpoint path concatenation collapsed:
    - `1_task` + `elm_training_outputs`

Path-hardening fix now applied:

- `embedding_elm/open-elm/generate_synthetic_notes.slurm`
  - `DATAHOME` default no longer includes a trailing slash
  - launcher normalizes both:
    - `BASE_DIR="${BASE_DIR%/}"`
    - `DATAHOME="${DATAHOME%/}"`
  - all derived defaults now join with explicit `/`, including:
    - `CHECKPOINT_PATH`
    - `OUTPUT_DIR`
    - `DATASET_PATH`
    - `SPLIT_MANIFEST_PATH`

Operational lesson:

- for future important runs, pass `CHECKPOINT_PATH` explicitly in the submission command as well, even though the launcher default is now safe

## June 18 Shifted-Pilot Evaluation Generalization

After the first successful shifted generation pilot (`2014977`), the evaluation stack was generalized so the 660-row shifted subset can be audited and compared without pretending it is a full vanilla test-set decode.

Audit updates:

- `embedding_elm/open-elm/cav_axis/audit_vanilla_generation.py`
  - now accepts:
    - `--expected_generation_condition`
    - `--allow_subset_dataset_rows`
  - subset mode:
    - treats manifest-vs-dataset row-count mismatch as a warning instead of automatic failure
    - checks `dataset_row_id` is monotonic and unique rather than requiring `0..N-1`
    - validates leakage flags by joining on `dataset_row_id`
    - reuses source embeddings by selecting the referenced dataset rows instead of assuming a full row-aligned decode

Coverage updates:

- `embedding_elm/open-elm/cav_axis/prepare_coverage_mapping.py`
  - now accepts:
    - `--expected_generation_condition`
    - `--allow_subset_synthetic_manifest`
  - subset mode:
    - allows synthetic manifests whose row count is smaller than the full held-out test set
    - requires shifted `dataset_row_id` values to be a subset of the filtered test manifest
    - no longer hardcodes `generation_condition == vanilla` when explicitly configured otherwise

Wrapper updates:

- `embedding_elm/open-elm/cav_axis/audit_vanilla_generation.slurm`
  - adds env vars:
    - `EXPECTED_GENERATION_CONDITION`
    - `ALLOW_SUBSET_DATASET_ROWS`
- `embedding_elm/open-elm/cav_axis/coverage_real_vs_synthetic.slurm`
  - adds env vars:
    - `EXPECTED_GENERATION_CONDITION`
    - `ALLOW_SUBSET_SYNTHETIC_MANIFEST`

Practical use:

- shifted pilot outputs can now be evaluated in their own directories under:
  - `generation_audit/cav_axis15_cluster25_alpha0p5_1p0_norm`
  - `coverage/real_vs_synthetic_cav_axis15_cluster25_alpha0p5_1p0_norm`
- note: the audit script still writes legacy filenames such as `vanilla_quality_table.csv`, but they are safe because they live inside the shifted pilot’s dedicated output directory

Follow-up fixes after first shifted audit run:

- multi-alpha subset manifests legitimately repeat `dataset_row_id`
- `audit_vanilla_generation.py` was relaxed further so duplicate `dataset_row_id` in subset mode is now a warning rather than a hard failure
- top-k self-retrieval rows now merge back by manifest row index instead of `dataset_row_id`, which avoids dropping faithfulness fields for subset pilots
- `prepare_coverage_mapping.py` already allows repeated subset rows as long as the unique `dataset_row_id` set is contained within the filtered real test split

Matched vanilla baseline helper:

- new script:
  - `embedding_elm/open-elm/cav_axis/build_matched_manifest_subset.py`
- purpose:
  - take a reference manifest such as the shifted pilot
  - extract the unique `dataset_row_id` source set
  - filter a candidate manifest such as the full vanilla manifest to the same source rows
  - optionally filter an aligned candidate embedding matrix in the same order
- intended use:
  - build an apples-to-apples vanilla baseline for coverage comparison against the shifted pilot on the exact same source anchors

## June 19 Shifted-vs-Matched-Vanilla Results

Corrected subset-aware evaluation runs are now complete for the first steering pilot:

- shifted pilot:
  - condition: `cav_axis15_cluster25_alpha0p5_1p0_norm`
  - source anchors: `330`
  - generated notes: `660` (`alpha=0.5` and `alpha=1.0` for each anchor)
- matched vanilla baseline:
  - condition: `vanilla_matched_axis15_cluster25_seed42`
  - source anchors: `330`
  - generated notes: `330`

Current output directories:

- shifted audit:
  - `.../generation_audit/cav_axis15_cluster25_alpha0p5_1p0_norm`
- shifted coverage:
  - `.../coverage/real_vs_synthetic_cav_axis15_cluster25_alpha0p5_1p0_norm`
- matched vanilla audit:
  - `.../generation_audit/vanilla_matched_to_axis15_cluster25`
- matched vanilla coverage:
  - `.../coverage/real_vs_synthetic_vanilla_matched_to_axis15_cluster25`

Audit headline:

- both shifted and matched-vanilla runs landed at `CAUTION`, not `FAIL`
- main caution reason in both cases is repetition/collapse slightly above the desired threshold
- no empty-output or too-short-output problem was observed
- privacy first-pass remained acceptable:
  - exact duplicate generated notes: `0`
  - exact duplicates vs train text: `0`
  - PHI-like flagged notes: `2`

Shifted pilot metrics:

- full set:
  - collapse rate: `11.21%`
  - mean source cosine: `0.7581`
  - median source cosine: `0.7591`
- patient-disjoint:
  - collapse rate: `12.16%`
  - mean source cosine: `0.7509`
- patient-overlap:
  - collapse rate: `10.44%`
  - mean source cosine: `0.7639`

Matched vanilla metrics:

- full set:
  - collapse rate: `10.30%`
  - mean source cosine: `0.7597`
  - median source cosine: `0.7603`
- patient-disjoint:
  - collapse rate: `12.84%`
  - mean source cosine: `0.7544`
- patient-overlap:
  - collapse rate: `8.24%`
  - mean source cosine: `0.7640`

Coverage comparison that matters:

- shifted pilot improved target-slice coverage relative to matched vanilla
- full-test real-to-synthetic coverage:
  - shifted: `0.09795`
  - matched vanilla: `0.05471`
- patient-disjoint real-to-synthetic coverage:
  - shifted: `0.12403`
  - matched vanilla: `0.07838`
- low-density-cluster coverage:
  - shifted full test: `8/10`
  - matched vanilla full test: `5/10`
  - shifted patient-disjoint: `7/10`
  - matched vanilla patient-disjoint: `1/10`

Interpretation carried forward:

- this first pilot does not yet justify broad claims, but it is a useful positive signal that steering can move synthetic notes into under-covered regions better than a matched vanilla baseline
- the best immediate next step is not full generation expansion yet; it is to replicate this pilot across more carefully chosen cluster-axis pairs and keep the comparison anchored to matched vanilla subsets
- for this pilot, `alpha=1.0` looked preferable to `alpha=0.5` because:
  - collapse was lower (`9.09%` vs `13.33%`)
  - mean cosine was essentially unchanged

## June 19 Shifted-Dataset Join Fix

While preparing the next replication pilot (`axis 15`, `cluster 11`), `build_shifted_embedding_dataset.py` failed when joining `encoded_testing_filtered` to `split_manifest_note_level.csv`:

- error:
  - `Split manifest has 98537 duplicate rows for join keys ['dataset_row_id']`

Cause:

- `encoded_testing_filtered` has only `input_ids` and `domain_embeddings`; it does not carry row metadata columns
- therefore the builder initially joins through the filtered split manifest
- `dataset_row_id` is only unique within each split, not across the full train/dev/test manifest

Fix applied:

- `embedding_elm/open-elm/cav_axis/build_shifted_embedding_dataset.py`
  - `merge_optional_metadata()` now accepts `source_split`
  - if the available join key is only `dataset_row_id`, the split manifest is first filtered to the requested split
  - if `--source_split` is omitted in this ambiguous case, the script now raises a clearer error instructing the caller to pass it

Operational rule:

- for future shifted held-out pilots built from `encoded_testing_filtered`, pass:
  - `--source_split test`

Smoke test:

- a 3-row smoke build for `axis 15` / `cluster 11` / `alpha 1.0` succeeded under:
  - `.../cav_shifted_datasets/_smoke_axis15_cluster11_alpha1p0_norm`

## June 19 Cluster-11 Replication Generation

The next controlled replication pilot has now completed generation:

- condition:
  - `cav_axis15_cluster11_alpha1p0_norm`
- shifted dataset:
  - `.../cav_shifted_datasets/test_axis15_cluster11_alpha1p0_norm`
- synthetic output:
  - `.../synthetic_notes/cav_axis15_cluster11_alpha1p0_norm`

Generation integrity:

- merged manifest rows: `350`
- merged note headers: `350`
- unique source dataset rows: `350`
- axis ids present: `15`
- alphas present: `1.0`
- patient-disjoint rows in this pilot: `150`

Interpretability caution to carry forward:

- the manifest still records:
  - `axis_label = axis_15__cluster_target_25::1.0`
- so axis 15 is still best interpreted as a learned mixed direction whose top loading is cluster 25, even when we apply it to a cluster-11 source subset
- therefore the next essential step is evaluation, not more blind generation:
  - re-embed cluster-11 shifted notes
  - audit quality/faithfulness/privacy
  - build matched vanilla subset for the same 350 anchors
  - compare shifted vs matched-vanilla coverage

## June 19 Audit Fast-Path

Repeated audit reruns were slow because `audit_vanilla_generation.py` always re-embedded generated notes, even when a matching `generated_note_embeddings_bge_large.npy` already existed.

Patch applied:

- `embedding_elm/open-elm/cav_axis/audit_vanilla_generation.py`
  - new optional CLI arg:
    - `--precomputed_generated_embeddings_path`
  - when provided:
    - loads the `.npy` embedding matrix directly
    - validates row count against manifest rows
    - L2-normalizes defensively
    - skips the expensive `SentenceTransformer.encode(...)` step
- `embedding_elm/open-elm/cav_axis/audit_vanilla_generation.slurm`
  - new env var:
    - `PRECOMPUTED_GENERATED_EMBEDDINGS_PATH`

Operational rule:

- for rerun audits, especially full vanilla, prefer:
  - re-embed once with `reembed_generated_notes.py`
  - audit with `PRECOMPUTED_GENERATED_EMBEDDINGS_PATH=...`

Reasoning reminder:

- audit uses `dataset_path=encoded_testing_filtered` because it is a held-out source-faithfulness evaluation
- full real embeddings are used for whole-manifold coverage discovery, not as the direct row-aligned faithfulness reference for held-out generation

## June 20 Cluster-11 Matched Comparison Outcome

The full `cluster11` comparison is now complete:

- shifted condition:
  - `cav_axis15_cluster11_alpha1p0_norm`
- matched vanilla condition:
  - `vanilla_matched_axis15_cluster11_seed42`

Audit result:

- both runs passed audit
- shifted cluster11:
  - status: `PASS`
  - median cosine: `0.7808`
  - collapse rate: `8.86%`
  - PHI-like flagged notes: `2`
- matched vanilla cluster11:
  - status: `PASS`
  - median cosine: `0.7860`
  - collapse rate: `10.00%`
  - PHI-like flagged notes: `0`

Coverage result:

- unlike the earlier `cluster25` pilot, `axis15 -> cluster11` did **not** beat matched vanilla on coverage
- full-test real-to-synthetic coverage:
  - shifted: `0.04260`
  - matched vanilla: `0.05118`
- patient-disjoint coverage:
  - shifted: `0.06474`
  - matched vanilla: `0.07353`
- low-density-cluster coverage:
  - shifted full test: `5/10`
  - matched vanilla full test: `7/10`
  - shifted patient-disjoint: `1/10`
  - matched vanilla patient-disjoint: `5/10`

Interpretation carried forward:

- this is an important negative control / failed replication for broadening `axis15`
- the current evidence now says:
  - `axis15 -> cluster25` is promising
  - `axis15 -> cluster11` is not justified as a general enrichment recipe
- therefore the next step should not be broad scaling of `axis15` across many clusters
- instead, Phase 2 should pivot to one of:
  - test a more cluster11-aligned axis from the same bank
  - refine axis construction / targeting before more generation
  - keep `cluster25` as the positive pilot while using `cluster11` as evidence that axis-target matching matters

Operational lesson:

- the new 2-GPU sharded audit wrapper works structurally, but repeated audit runtime is still heavily influenced by train-neighbor privacy checks and text-overlap screening, not just re-embedding

## June 20 Cluster-11 Candidate-Axis Audit

To follow up the failed `axis15 -> cluster11` matched comparison, candidate axes `1,2,12,15` were audited using:

- `steering_audit_cluster11_candidate_axes.csv`
- `steering_trends.csv`
- `steering_summary.json`

Key interpretation:

- `axis15` does move `cluster_target_11::1.0` upward, but it also strongly loads on `cluster_target_25::1.0`, `service::gynecology`, and other mixed signals
- this supports the earlier conclusion that `axis15` is too entangled to treat as a general cluster11 enrichment direction

Most plausible cluster11-aligned direction from the audit:

- `axis1`, using **negative** alpha

Reason:

- cluster11 enrichment profile from earlier analysis was:
  - medicine-heavy
  - medicare-heavy
  - older-age skew
- `axis1` with negative alpha strongly increases:
  - `service::medicine`
  - `insurance::medicare`
  - `age_bin::80+`
  - `age_bin::65-79`
- and strongly decreases:
  - `service::surgery`
  - `insurance::private`
  - younger-age bins

Other candidate readings:

- `axis2` positive alpha increases `service::medicine`, but also strongly increases `icu_flag::True` and long LOS, so it may be pulling toward a more critical-care / acuity direction rather than the broader cluster11 signature
- `axis12` positive alpha increases `age_bin::65-79` and some urgent/observation patterns, but its age pattern is mixed and less directly aligned to the original cluster11 enrichment than `axis1`

Decision carried forward:

- do not keep expanding `axis15` to more cluster11-like subsets
- the next cluster11-specific pilot, if run, should test:
  - `axis1` with a negative alpha grid first
  - likely `alpha in {-0.5, -1.0}`

## June 20 Axis1-Cluster11 Pilot Outcome

The refined cluster11 pilot using `axis1` with negative steering is now complete:

- condition:
  - `cav_axis1_cluster11_alpha0p5_1p0_neg_norm`
- source anchors:
  - `350`
- generated notes:
  - `700`
  - `350` at `alpha=-0.5`
  - `350` at `alpha=-1.0`

Audit result:

- shifted run:
  - status: `CAUTION`
  - main reason: collapse rate `11.00%`
  - median cosine: `0.7849`
  - mean cosine: `0.7854`
  - PHI-like flagged notes: `3`
- matched vanilla:
  - status: `PASS`
  - collapse rate: `10.00%`
  - median cosine: `0.7860`
  - PHI-like flagged notes: `0`

Coverage result:

- unlike `axis15 -> cluster11`, the `axis1 -> cluster11` pilot **does beat** matched vanilla on the main coverage metrics
- full-test real-to-synthetic coverage:
  - shifted: `0.09159`
  - matched vanilla: `0.05118`
- patient-disjoint coverage:
  - shifted: `0.12857`
  - matched vanilla: `0.07353`
- synthetic-to-real precision:
  - shifted: `0.48714`
  - matched vanilla: `0.43714`
- low-density-cluster coverage:
  - full test: both `7/10`
  - patient-disjoint: both `5/10`
  - patient-overlap: shifted `5/10`, vanilla `6/10`

Interpretation carried forward:

- this rescues the earlier failed `cluster11` experiment
- the current evidence supports a stronger Phase 2 claim:
  - axis-target matching matters
  - a more phenotype-aligned axis can improve coverage where an entangled axis fails
- but quality is not yet strictly better than vanilla, so this should still be framed as:
  - promising coverage gain with acceptable but not clearly superior generation quality

Operational next step:

- before broad scaling, add a stronger control or selection step:
  - choose a preferred alpha (`-0.5` is slightly cleaner on collapse; `-1.0` is slightly higher on cosine)
  - and/or compare against a norm-matched random-shift control for the same source anchors

## June 20 Norm-Matched Random-Shift Control Builder

To support the planned control for `cluster11 + axis1`, a dedicated builder was added:

- `embedding_elm/open-elm/cav_axis/build_random_shift_control_dataset.py`

Purpose:

- take a reference shifted HF dataset
- reuse the same source anchors and the same per-row `random_shift_norm`
- replace the learned CAV direction with a random unit vector
- preserve the same downstream dataset structure for `generate_synthetic_notes.py`

Current intended use:

- reference dataset:
  - `cav_shifted_datasets/test_axis1_cluster11_alpha0p5_1p0_neg_norm`
- control framing:
  - norm-matched random perturbation on the exact same `350` anchors and `700` shifted rows

Provenance behavior:

- output rows keep:
  - source IDs
  - alpha
  - leakage flags
  - shift norm
- control rows set:
  - `axis_id = None`
  - `axis_label = random_shift_control`

## June 26 Phase 2b geometry conclusion: freeze current additive local steering as a partial finding

We ran the new pre-decode geometry audit for `cluster_target_29` using:

- local centroid direction
- local one-vs-rest linear direction
- alpha sweeps
- note-level comparison against matched vanilla and random shift

Main technical conclusion:

- `cluster_target_29` is highly separable in the real embedding space
  - eval AUROC about `0.999`
  - eval AP about `0.979`
- so the target region itself is not the bottleneck

But current additive local steering is not yet a reliable target-enrichment mechanism:

- on non-target anchors, small alpha can weakly improve pre-decode target-neighborhood entry
  - best regime is in the low-alpha range around `0.25`
- larger alpha overshoots and moves embeddings away from the true local target neighborhood
- classifier score saturates early and is misleading on its own
  - notes can look more target-like to the classifier while still being farther from the empirical cluster neighborhood
- after ELM decoding, the weak pre-decode gain is mostly washed out
  - matched vanilla remains as good or better on note-level target entry than current local centroid steering

Project interpretation carried forward:

- freeze current additive local steering as a `negative/partial` Phase 2b finding
- do not scale more note generation from the current centroid/linear additive setup yet
- next principled step is:
  - low-alpha-only geometry sweep
  - choose the best pre-decode entry regime
  - only then run one more tightly scoped generation pilot

## June 29 projected-boundary cluster29 conclusion: freeze current projection family as a negative finding

We completed a stricter Phase 2b redesign test for cluster 29:

- method:
  - source-conditioned projected boundary steering
  - boundary anchors selected from non-target rows near cluster 29
  - alphas `0.10, 0.15, 0.20, 0.25`
- run:
  - `projected_cluster29_boundary0p90_alpha0p10_0p25_norm_test`
  - `1024` generated notes total
  - `256` notes per alpha

Main result:

- generation quality was acceptable
  - audit `PASS`
  - empty rate `0`
  - aggregate collapse about `9.18%`
  - aggregate median cosine about `0.803`
- but target enrichment failed after decoding
  - aggregate cluster29 full-test synthetic fraction `0.000977`
  - aggregate cluster29 patient-disjoint synthetic fraction `0`
  - aggregate full coverage about `0.1157`
  - aggregate patient-disjoint coverage about `0.1159`
- per-alpha coverage confirmed the failure
  - `0p1`, `0p2`, `0p25`: zero cluster29 occupancy
  - `0p15`: tiny full-test cluster29 signal only, still zero patient-disjoint occupancy
  - all four alphas had low full/patient-disjoint coverage (`~0.03-0.04`) when evaluated as separate subsets

Interpretation carried forward:

- this is a negative method finding, not a pipeline failure
- the current evaluation stack is doing its job:
  - embedding edits can look plausible pre-decode
  - generated notes can still pass quality audit
  - yet decoded notes may fail to land in the intended sparse region
- therefore, the next redesign should focus on methods that are better aligned with the actual ELM interface:
  - a single `1024`-d embedding passed through a small adapter
  - decoder compatibility depends on staying close to the empirical real embedding manifold

Phase 2b redesign principles now preferred:

- stop broad generation from the current additive/projection families
- keep the best local-additive cluster29 run only as the current internal steering baseline
- prioritize source-conditioned, on-manifold operators such as:
  - local transport / barycentric target mixtures
  - decode-reembed correction
  - lightweight projector or adapter-only refinement if inference-only steering keeps failing

## July 2 hard-anchor basin-margin conclusion: current linear local steering family failed its strongest test

We completed the hard-anchor competitor-margin test on a \textbf{true hard-anchor set} drawn from the full held-out test pool rather than the earlier easy 64-anchor subset.

Setup:

- target cluster: `29`
- main local competitors: `9` and `45`, with `17` and `7` treated as nearby basin context
- source baseline, linear basin-margin steering, and norm-matched random local control were all scored on the same hard anchors
- the key gate metric was \textbf{target-cluster margin vs best competitor}, not only pooled-basin entry

Main result:

- source baseline on hard anchors:
  - mean target margin about `-0.087`
  - target cluster win rate `0.0`
- best linear basin-margin hard-anchor sweep condition:
  - `alpha=0.30`
  - `margin_gamma=0.20`
  - mean source cosine about `0.9767`
  - mean target margin improved to about `-0.0308`
  - but target cluster win rate remained `0.0`
- all top hard-anchor sweep settings kept `target_cluster_win_rate = 0.0`

Interpretation carried forward:

- this is \textbf{not} a pipeline failure
- this is \textbf{not} evidence that the local basin framing is wrong
- it \textbf{is} evidence that the current linear additive local basin-margin operator is too weak for real boundary crossing on hard anchors
- therefore Phase 2b should stop broad note generation from this steering family

Operational conclusion:

- freeze the current linear local basin-margin family as a `negative/partial` finding
- keep the generic basin-margin audit framework because it scales and remains useful
- redesign the steering operator around \textbf{competitor-aware hard-boundary crossing} and \textbf{decoder-compatible local manifold moves}

Ranked redesign options now preferred:

1. \textbf{Competitor-aware local transport / barycentric crossing}
   - Best fit to the failure mode.
   - Reason: the problem is specifically losing to nearby competitor clusters `9` and `45`, not random drift. A source-conditioned transport operator that moves toward target exemplars while explicitly pushing away from the current local winner is the most direct response.

2. \textbf{Local basin-margin framework with explicit competitor penalties}
   - Strong fit and easy to reuse with current tooling.
   - Reason: we already have a generic pre-decode margin gate. The next version should optimize target29 minus best(`9`,`45`,`17`) more directly instead of only nudging toward the pooled basin.

3. \textbf{Decode-reembed correction on top of a local operator}
   - Good fit to the decoder mismatch we keep seeing.
   - Reason: pre-decode geometry can improve while post-decode occupancy still fails. A one-step correction loop targets exactly that gap.

4. \textbf{Lightweight projector / adapter calibration}
   - Plausible but heavier and should come after a cleaner inference-time operator.
   - Reason: if all inference-only local operators keep failing, then the adapter likely needs to be taught how to decode locally shifted embeddings more faithfully.

5. \textbf{Another broad global-axis or plain additive sweep}
   - Lowest priority.
   - Reason: the hard-anchor result already shows the bottleneck is local competitor crossing, not lack of additional global alpha search.

## June 29 implementation update: barycentric transport builder added

New script:

- `embedding_elm/open-elm/cav_axis/build_barycentric_transport_dataset.py`

Purpose:

- build a source-conditioned, on-manifold steering dataset for ELM
- for each source anchor:
  - find a local target neighborhood
  - compute a weighted barycentric target mixture
  - mix source and local target barycenter with alpha in `[0,1]`
- preserve the same HF dataset structure and manifest-style provenance used by the current generation pipeline

Current design choices:

- local target selection via cosine kNN in the real embedding space
- neighbor weighting modes:
  - `softmax_cosine`
  - `inverse_distance`
  - `uniform`
- row-level provenance includes:
  - `axis_label = barycentric_transport__<target_column>`
  - `alpha`
  - nearest target information
  - local target mixture similarity
  - neighbor weighting metadata

Smoke-test status:

- a tiny test run on cluster29 boundary anchors completed successfully
- output dataset and run metadata were written correctly

Interpretation:

- this is the next principal Phase 2b method family to test
- it is more compatible with the ELM interface than broad additive or projection-only steering because it keeps shifted embeddings closer to real observed embeddings
