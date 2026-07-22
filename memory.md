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

## Definitive Validation Update: Equal-Compute Selector Analysis

The completed manifests were reanalyzed without new generation using explicit gate routes and a random-one-of-eight equal-compute baseline.

- same-anchor paired comparison (106 accepted anchors): exact pooled-basin landing is `32.1%` for selected notes versus `15.1%` for one-draw matched vanilla; paired risk difference `+17.0%` (bootstrap 95% CI `+5.7%` to `+28.3%`, exact McNemar `p=0.0079`),
- patient-disjoint paired subset: `42.9%` selected versus `0%` matched vanilla (`n=21`), so this signal is not confined to patient-overlap rows, though the subset is small,
- all-anchor operational yield (256 anchors): `34` exact-plus-centroid accepted notes (`13.3%`), `72` centroid-only accepted notes (`28.1%`), and `150` anchors with no final accepted note (`58.6%`),
- raw random one-of-eight sampling has higher exact pooled landing (`21.5%`) than final selected output (`13.3%`), but it has very low clinical-rule pass (`7.3%`) and usable-exact pooled yield (`1.75%`, simulation 95% interval `0.39%`--`3.52%`),
- therefore the defensible selector endpoint is \textbf{usable exact pooled-basin enrichment}: selected `13.3%` versus random one-of-eight `1.75%`; raw exact landing alone must be reported as a secondary diagnostic because it favors low-quality candidates.

Current outputs:

- `cav_axis/analyze_closed_loop_validation.py` writes paired, equal-compute, and all-anchor yield reports,
- `cav_axis/backfill_closed_loop_gate_routes.py` writes non-destructive corrected legacy manifests with `exact_pooled_cluster_pass`, `centroid_proximity_pass`, and `gate_route`,
- `cav_axis/build_source_faithfulness_review.py` produced a protected 68-case source review pack: all 34 exact pooled-basin accepted notes plus 34 centroid-only notes, each with a blinded matched vanilla comparator.

Privacy-screen implementation update:

- `cav_axis/closed_loop_train_text_privacy_screen.py` now treats `--max_train_texts 0` as uncapped and supports a BGE semantic top-$k$ train-neighbor shortlist before lexical 10-gram comparison,
- lexical 10-grams are now cached only for retrieved train neighbors rather than materialized for the entire training corpus,
- `cav_axis/closed_loop_train_text_privacy_screen.slurm` is the canonical CPU launcher (`devel`, 8 CPUs, 48G, 6h) and uses unbuffered progress logs.

Next required gate remains human validation and full-train privacy screening. Do not scale candidate count or start multi-region generation until exact-basin selected notes are clinically non-inferior to fair vanilla and source faithfulness is acceptable.

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

## July 15 closed-loop clinical validation: freeze 106-note run as geometry feasibility, fail clinical-faithfulness gate

The completed matched-vanilla and privacy screens support the automated/embedding part of the closed-loop result:

- 106 closed-loop selected notes and 106 fair same-anchor 8192-token vanilla controls were screened against all 262,895 filtered training notes.
- exact train-note duplicates: `0` in both cohorts.
- lexical similarity >= `0.8`: `0` in both cohorts.
- accepted notes had better automated target alignment, source cosine, structure, and clinical-rule scores than matched vanilla.
- the existing ``any shared 10-gram'' privacy flag was positive for every row in both cohorts and is too sensitive for interpretation because it captures routine discharge-summary boilerplate. Future privacy reporting must use an overlap ratio or material-copy threshold.

The row-level human review overturns the interpretation that automatic acceptance alone establishes usable source-conditioned notes:

- blinded clinical-quality review: `205/206` notes had missing or non-substantive sections; follow-up, disposition, diagnoses, instructions, medication reconciliation, physiologic values, and narrative coherence frequently failed.
- source-paired review of 68 cases and two synthetic variants per case: principal diagnosis was not preserved in `67/68` (A) and `68/68` (B); unsupported major claims and critical source omissions occurred in all variants; source-anchor loss occurred in `67/68` (A) and `68/68` (B).
- exact pooled-basin versus centroid-only route did not rescue source faithfulness.

Decision:

- freeze the 106-note closed-loop run as a positive \textbf{output-space geometry-selection} finding but a negative \textbf{clinical source-faithfulness} finding.
- do not scale candidate count, claim clinically meaningful enrichment, start source-free LLM editing, or run downstream NER from this condition.
- do not add more inference-time steering families.

Next required feasibility block:

1. retain the fair matched-vanilla, privacy, and review artifacts as the formal baseline;
2. refine privacy reporting to distinguish boilerplate overlap from material copying;
3. construct a small, approved \textbf{three-arm source-fact-conditioned} pilot: raw ELM, source-fact-conditioned correction, and source-fact-only generation from the same verified ledger. This directly tests whether the ELM draft is a helpful scaffold or harmful narrative context;
4. use source-paired factual preservation (diagnosis, procedures, complications, medication changes, disposition, unsupported claims, omissions) as the primary gate before any multi-region or cohort-scale enrichment work.

## July 15 source-grounded rescue preparation

Completed manual review files are now stored in `open-elm/cav_axis/docs` and were ingested into label-only derived outputs. The review-calibrated deterministic triage is supporting infrastructure only: it flags obvious defects but cannot certify semantic factuality.

- A reproducible rescue-pilot anchor selector was added at `open-elm/cav_axis/source_grounded_rescue/build_rescue_pilot_anchor_manifest.py`.
- It selected `45` distinct held-out cases from the completed source-paired review: `24` exact-pooled, `21` centroid-only, and `10` patient-disjoint. All had source-anchor loss in manual review, so the pilot tests the real failure mode rather than easy cases.
- `build_source_fact_ledger.py` wrote a provisional ledger for all 45 cases: `406` source-cited facts. All are `pending` verification.
- Automatic source-section extraction covered admission reason, sex, follow-up, and procedures in all cases, but found principal diagnosis in only `31/45` and hospital course in `33/45`; reviewers must verify, correct, and add required facts before any local generation/correction experiment.

### Completed ledger review and validation

- The completed portable review patch in `open-elm/cav_axis/docs/source_fact_ledger_review_patch_PORTABLE.csv` was applied only on approved storage with the new `apply_source_fact_ledger_review.py` utility.
- The completed restricted ledger has `483` rows: `354` verified, `82` corrected, `47` rejected, and no pending rows. It includes `77` manually reviewed source-supported additions.
- `validate_source_fact_ledger.py` reports all `45/45` cases ready for generation, complete required-field coverage, and `436/436` verified-or-corrected facts supported by the restricted source note after whitespace-normalized span checking.
- A single empty original extracted value was safely recovered from its original source offsets during patch application; this was an extraction artifact, not a reviewer disagreement.

Current blocker before the three-arm pilot:

- choose an approved local model/checkpoint and local inference interface for fact-conditioned correction (arm B) and fact-only generation (arm C). Do not send source notes, ledgers, or outputs to a third-party API.

### July 15 generation-ledger completion and next smoke test

- The reviewer-completed prompt-safe generation ledger is stored in `open-elm/cav_axis/docs/generation_ledger_smoke_review_completed_RESTRICTED.csv`.
- `build_generation_ledger.py` produced `generation_ledgers.jsonl` for four source-paired pilot cases: `38` reviewed fact values, with no source-note spans or source text in the serialized generation input.
- All four ledger anchors join uniquely to the frozen `106`-note accepted raw-ELM manifest. The ledger records only `fact_id`, `field`, and approved `generation_value`.
- The local ordinary-text generation path passed a deterministic smoke test for both the untouched initial ELM backbone and PEFT `checkpoint-8215`: exact expected phrase, EOS termination, no prompt echo, and no output-cap hit.
- The next required experiment is a local four-case, two-arm, two-model-condition smoke test, not cohort scale-up:
  - raw-ELM-draft correction using verified facts as the sole factual authority;
  - fact-only generation using the same verified ledger;
  - each under untouched backbone and `checkpoint-8215`.
- The smoke manifest must retain model condition, arm, compact-ledger hash, raw candidate id, token count, EOS/cap flags, and generated text. Its primary evaluation is blinded source-paired factual preservation; basin geometry is secondary until factual preservation is acceptable.

### July 15 four-case rescue smoke completed

- Local Slurm job `2121068` completed the planned four-case rescue smoke: `16/16` B/C outputs across the untouched backbone and `checkpoint-8215`.
- All rescue outputs were non-empty, EOS-terminated, below the `3072` token cap, uniquely identified, and carried both ledger-hash and correction-draft provenance.
- `build_source_grounded_review_pack.py` created a blinded ledger-grounded review set with `20` notes: raw ELM baseline plus four B/C model-condition outputs for each of four cases.
- Review file: `.../four_case_text_smoke_v1/blinded_review_pack/source_grounded_blinded_review.csv`; keep `source_grounded_blinded_key.csv` separate until review labels are final.
- Next decision gate: compare factual support, unsupported major claims, critical omissions, and clinical usability across conditions. Do not expand the 45-case pilot or resume enrichment scale-up until this review identifies a condition that improves over raw ELM.

### July 16 four-case blinded rescue result

- The completed review was ingested and unblinded with `analyze_source_grounded_review.py`. All `20` rows satisfied the predeclared pass-rule validation.
- Overall: `7/20` passes, `12/20` unsupported-major-claim rate, and `13/20` critical-omission rate.
- Condition result (four cases each):
  - `checkpoint_8215__fact_only`: `4/4` passes, zero unsupported-major claims, zero critical omissions, mean factual-faithfulness `4.75/5`.
  - `untouched_backbone__fact_only`: `3/4` passes, zero unsupported-major claims, one critical omission, mean factual-faithfulness `3.75/5`.
  - both correction arms and raw ELM baseline: `0/4` passes, unsupported-major-claim rate `1.0`, critical-omission rate `1.0`.
- Interpretation: the untrusted raw ELM draft is harmful context in this setting. The viable candidate is fact-only generation using reviewed ledger values with `checkpoint-8215`.
- Next experiment: a bounded, pre-specified 20--30 case replication of `checkpoint_8215__fact_only`, with the same ledger verification and blinded factual review. Keep raw ELM as a baseline for each case; drop correction arms unless a separate draft-grounding mechanism is developed.

### July 16 replication-design correction

- The 4/4 result is a condition-specific method-selection signal, not proof of near-perfect performance; its exact 95\% confidence interval is wide (approximately 40\%--100\%). The pooled 7/20 result is not an estimate for the selected condition because it combines distinct arms.
- Freeze a prospective 30-case replication from the 41 unused verified-ledger cases. Selection must be stratified across exact-pooled/centroid-only routes, patient-disjoint/overlap status, clinical domains, and ledger complexity; it must not select apparent easy cases.
- Primary condition: one deterministic `checkpoint-8215` fact-only output from a reviewer-approved compact ledger, without raw draft, retries, manual prompt edits, candidate selection, or geometry optimization.
- Main baseline: existing matched raw ELM note. Secondary prespecified control: untouched-backbone fact-only on a nested 15-case subset, testing whether checkpoint training adds value beyond the verified ledger.
- Predeclared feasibility thresholds for the primary 30 cases: proceed at >=27/30 passes, <=2 unsupported-major-claim failures, <=2 critical-omission failures, no systematic medication/procedure/disposition failure, and acceptable patient-disjoint performance. A 24--26/30 result triggers revision; <24/30 stops whole-note fact-only scale-up.
- The current full 45-case source ledger is verified but only the smoke four have reviewer-approved concise `generation_value`s. The remaining replication cases must complete this compact-ledger review before generation; raw source spans must never enter prompts.
- A prospective 30-case candidate manifest and restricted concise-ledger review template were prepared at `.../source_grounded_rescue/fact_only_replication_30_v1/`. It excludes all four smoke anchors and contains 15 exact-pooled plus 15 centroid-only cases, seven patient-disjoint cases, and 292 usable fact rows requiring reviewer-approved concise values.
- The available unused target-pool cases are intrinsically service-skewed (37 medicine, 4 surgery); the 30-case candidate set includes three surgery cases. Report this as target-region composition rather than implying a service-balanced cohort.

### July 16 30-case ledger readiness completed

- The completed restricted review is in `open-elm/cav_axis/docs/fact_only_replication_30_review_completed_RESTRICTED.csv`: 30 cases, 292 reviewed rows, 212 verified, 57 corrected, and 23 omitted.
- `build_generation_ledger.py` validated and serialized the prompt-safe primary ledger: 30 unique cases/anchors and 269 non-empty reviewed facts. It contains no source spans, phone-number patterns, exact numeric dates, or blank usable fact values.
- No generation has started. The only valid next generation condition is one deterministic `checkpoint-8215` fact-only output for each of these 30 frozen ledgers.
- The untouched-backbone fact-only nested control was frozen pre-generation at `.../fact_only_replication_30_v1/nested_backbone_control_15_v1/`: 15 cases, 8 exact-pooled, 7 centroid-only, and 4 patient-disjoint, selected by fixed hash seed `20260716`.
- `subset_generation_ledger.py` created the prompt-safe 15-case control ledger (134 facts) from the frozen primary ledger. `source_grounded_fact_only.slurm` is the generic one-A40 deterministic launcher for each separately named condition. Both launcher and ledger subset passed Python/shell preflight; generation remains unstarted at this checkpoint.

### July 16 prospective generation completed; review gate pending

- The deterministic primary and control jobs completed cleanly: 30 `checkpoint-8215` fact-only outputs and 15 untouched-backbone fact-only outputs, all non-empty, EOS-terminated, under the 3072-token cap, and uniquely identified.
- A blinded restricted review pack is ready at `.../fact_only_replication_30_v1/blinded_review_pack/`: 75 outputs (30 matched raw ELM baselines, 30 primary checkpoint fact-only, 15 nested untouched-backbone fact-only). The condition key must remain unopened until labels are final.
- Current phase: prospective clinical-factual validation. No re-embedding, geometry analysis, coverage claim, or scale-up is allowed until the primary 30-case blinded review meets its predeclared feasibility thresholds.
- Defensible project mechanism: real-embedding manifold analysis identifies under-covered anchor regions; verified compact facts supply factual control; blinded review establishes validity; only passing outputs are re-embedded to evaluate output-space enrichment. This is distinct from, and more defensible than, a claim of reliable direct embedding-to-note inversion.

### Terminology correction: Phase 2 is not a single failure

- Input-space CAV/local transport and decoder adaptation are negative/partial findings: they do not reliably control the final decoded embedding.
- Closed-loop output-space selection is a positive geometry-feasibility finding: some generated notes did land in the pooled sparse basin and improved basin landing versus fair one-draw matched vanilla.
- The 106-note closed-loop condition failed the separate clinical source-faithfulness gate. The current source-grounded replication is intended to repair this clinical-validity bottleneck, after which clinically passing notes can be re-embedded to test whether geometry/enrichment is retained.

### July 16 30-case blinded factual replication result

- The 75-note blinded review unblinded cleanly, with all reviewer pass/fail values matching the predeclared rule. The second reviewer independently completed 24 overlapping blinded outputs with 100% exact agreement across recorded labels; retain both files for audit rather than treating this small agreement subset as a population estimate.
- Primary `checkpoint_8215_fact_only`: `30/30` passes, zero unsupported-major-claim and critical-omission failures, mean factual-faithfulness `4.97/5`; all `7/7` patient-disjoint cases passed.
- Matched raw ELM baseline: `0/30` passes, unsupported-major-claim and critical-omission rates both `1.0`.
- Nested untouched-backbone fact-only control: `15/15` passes. Therefore the current evidence supports the reviewed fact ledger plus fact-only generation mechanism; it does not establish that the ELM checkpoint is superior to the untouched backbone for this text-conditioned task.
- The primary factual gate manifest was frozen with all 30 passing outputs. The next approved action is BGE re-embedding of this factual-gated cohort, followed by a pilot geometry comparison against matched raw ELM and real source embeddings. Do not make cohort-scale enrichment claims from 30 cases.
- The first factual-gated re-embedding submission failed before inference because the generic utility required `generation_id`, while source-grounded manifests use stable `rescue_id`. `reembed_generated_notes.py` now accepts either identifier, records the identifier column in metadata, and preflight confirms all 30 factual-gated rows have unique rescue IDs, nonempty text, and dataset row IDs. Rerun is required; no BGE matrix was written by the failed job.
- The rerun completed successfully (`2124949`): the factual-gated checkpoint fact-only matrix is `30 x 1024`, row-aligned to the 30 factual-gate-passing manifest rows, finite, and L2-normalized (mean norm `1.0`). BGE model is `BAAI/bge-large-en-v1.5` on CUDA. It is now valid for the bounded post-review geometry comparison; it is not yet a cohort-scale coverage result.

## July 16 post-review paired geometry audit: clinical fidelity and basin retention remain separate gates

- The 30 factual-gated `checkpoint-8215` fact-only notes were compared with the same anchors' frozen closed-loop-selected raw ELM outputs against the fixed Phase 1 real-test clusters and pooled basin `9/17/29/45`.
- The fact-only condition is clinically successful but has weaker geometry: mean source-output cosine `0.7723`, pooled-basin retention `30.0%` overall and `14.3%` among seven patient-disjoint rows, and mean target-centroid distance change `+0.1288`.
- The frozen raw comparator has stronger geometry (`50.0%` pooled-basin retention; `100%` among the seven patient-disjoint rows), but it is the earlier closed-loop-selected raw output rather than a one-draw vanilla baseline and failed factual review. It must be called a selected raw-ELM comparator, not a vanilla comparator.
- Therefore do not scale the current deterministic fact-only procedure as sparse-region enrichment yet. The next bounded bridge is four sampled fact-only candidates per frozen anchor, BGE re-embedding of every candidate, transparent selection by final pooled-basin landing, and a new blinded factual review of selected notes. The objective is dual success: source-grounded factual validity plus final-output basin retention.

## July 16 fact-only geometry bridge: dual-gate feasibility success, expanded confirmation next

- On a frozen eight-anchor bridge pilot with four sampled `checkpoint-8215` fact-only candidates per anchor, all 32 candidates were non-empty, unique, and EOS-terminated without cap hits.
- Candidate pooled-basin landing was `16/32` (50.0%). Final-output selection retained one note per anchor and increased pooled-basin landing to `5/8` (62.5%), with three patient-disjoint selected cases.
- Blinded ledger-grounded review of all eight selected notes passed: `8/8` rule-verified passes, zero unsupported-major claims, zero critical omissions, and mean factual faithfulness `4.75/5`. This is the first dual-gate feasibility result: fact-ledger-conditioned candidate generation plus final-output selection can produce clinically factual notes while achieving pooled-basin landing in this small pilot.
- The result remains an eight-case feasibility signal, not a cohort-scale result. The next confirmation is the existing 30 reviewed-anchor cohort with four candidates per anchor, followed by BGE selection and blinded review of all 30 selected outputs. The 32-candidate bridge required `8m15s` on one A40, so the 120-candidate confirmation is operationally suitable for the six-hour A40 limit.

## July 16 30-anchor fact-only geometry bridge: geometry replication, factual review pending

- The frozen 30-anchor cohort produced 120 sampled fact-only candidates (four per anchor). Candidate pooled-basin landing was `42/120` (35.0%); final-output selection increased this to `16/30` (53.3%), exactly eight selected target-basin outputs in each `exact_pooled` and `centroid_only` stratum.
- All 30 selected notes are non-empty, EOS-terminated, and below the generation cap. The selected cohort includes all seven patient-disjoint anchors, of which two land in the pooled basin.
- The blinded 30-note ledger-grounded factual review pack is prepared at `.../fact_only_geometry_bridge_30x4_v1/blinded_review_pack/`. This is now the decisive confirmation gate. Do not make multi-region or cohort-scale claims until completed labels establish factual validity of the selected notes.

## July 16 30-anchor geometry bridge: expanded dual-gate confirmation

- The annotated blinded review completed with `28/30` rule-verified factual passes (93.3%), two unsupported-major-claim failures (6.7%), one critical-omission failure (3.3%), and mean factual faithfulness `4.63/5`.
- Joining blinded labels to the protected geometry key gives `15/30` clinically passing, pooled-basin-selected notes (50.0%). Of seven patient-disjoint anchors, all seven pass factual review and two are dual-gate successes. The target-cluster distribution among target selected notes is `9: 5/5 factual passes`, `17: 3/4`, `29: 5/5`, and `45: 2/2`.
- The two failures are localized medication-reconciliation errors: contradictory continue/discontinue status and duplicated/unsupported malformed respiratory medication. The frozen result is retained unchanged; future source-grounded prompts explicitly require one ledger-only, non-contradictory medication list. This is a prompt safety refinement, not retrospective repair.
- The project has now passed a 30-anchor dual-gate confirmation for the `9/17/29/45` basin. Next: objectively choose and preregister a second clinically coherent under-covered basin, then repeat the same 30-anchor, four-candidate, select--review protocol. Full-cohort scale-up and LLM editing remain deferred until multi-region replication.

## July 17 second-region eligibility screen: cluster 20 is not generation eligible at the planned sample size

- A deterministic 30-anchor cluster-20 cohort (eight patient-disjoint) was frozen and its 268 provisional facts reviewed. The review has 183 corrected, 41 verified, and 44 omitted facts.
- Follow-up is legitimately optional when unsupported because the fact-only prompt omits absent details. With follow-up excluded from the required contract, only `13/30` anchors retain all core verified fields: principal diagnosis, hospital course, discharge medications, disposition, and instructions.
- Do not generate from the remaining 17 anchors and do not weaken the core-fact gate. This is a source-grounding eligibility failure, not an ELM failure. It establishes that target-region selection must include factual-ledger completeness before a region enters generation.
- Retarget the second-region replication to cluster 36, a clinically distinct surgery-enriched region in the existing enrichment analysis. Build and review its provisional ledger before candidate generation.

## July 17 cluster 36 extraction audit: six-field source grounding is not currently viable for the sampled cohort

- The 30-anchor cluster-36 review has 37 verified, 171 corrected, and 86 omitted provisional facts. Reviewers correctly omitted placeholder follow-up and incomplete instruction fragments rather than guessing values.
- Restricted extraction audit confirms this is not merely an alias failure: follow-up was found in all 30 source notes but was placeholder-only in all 30; instructions were found in all 30 but fragment-like in 28. Only two sampled anchors retained all six likely core fields after review.
- Do not generate cluster-36 fact-only notes with the current six-field ledger contract and do not simply mark instructions optional. The next technical redesign is a source-eligibility pre-screen across a full target region, selecting only anchors with non-placeholder, complete source evidence before manual fact review. This source-availability gate must be added before future target-region generation.

## July 17 bounded cluster-36 evidence recovery audit: worth trying, insufficient for complete-note replication

- The mentor-proposed full-note recovery audit was implemented with restricted provenance-aware outputs, plus `admissions.discharge_location` as a structured disposition candidate. It recovered substantive follow-up evidence in `13/30` cases versus the original placeholder-only extraction, but substantive instruction evidence in only `3/30`; `admissions` supplied a disposition candidate in `28/30` cases.
- This demonstrates that high-recall full-note recovery is useful as an eligibility screen, but structured augmentation cannot safely replace unsupported personalized instructions. Diagnoses, procedures, and medication orders remain corroboration-only, not automatic ledger facts.
- Cluster 36 remains ineligible for a 30-case complete-discharge replication. Preserve it for manifold/eligibility analysis and potentially a separately labeled partial-document task; move to another target region for the primary complete-note track. Apply the recovery audit before further manual review to avoid repeated low-yield annotation.

## July 17 reusable source-eligibility pre-screen added before the next region

- Added `open-elm/cav_axis/source_grounded_rescue/screen_region_source_eligibility.py`. It evaluates all held-out rows in a fixed target cluster or basin without exporting source-note text, assigns Tier 1 complete-note review candidate, Tier 2 partial-document review candidate, or Tier 3 insufficient-evidence status, and records field-level evidence routes plus patient-disjoint provenance.
- Tier 1 is still only a manual-review candidate: a source ledger must be reviewed before generation. The screen uses full-note heading recovery for follow-up/instructions and `admissions.discharge_location` only as a disposition candidate; it never automatically converts diagnoses, procedures, or medication orders into discharge facts.
- A three-row cluster-36 smoke test yielded only Tier 3 rows, consistent with the completed recovery audit. Next action: run the screen across clinically distinct candidate sparse regions, then select the next 30-anchor replication only from a region with an adequate Tier 1 reserve.

## July 17 calibrated eligibility screen and frozen cluster-16 review cohort

- The initial generic text-length/punctuation rule was too strict for short diagnoses and list-form medications/instructions. `screen_region_source_eligibility.py` is now field-aware and remains a high-recall triage screen, not a fact validator.
- Across held-out clusters 11/16/25, the recalibrated screen found Tier 1 complete-note review candidates: cluster 11 `19` (8 patient-disjoint), cluster 16 `52` (25 patient-disjoint), and cluster 25 `66` (6 patient-disjoint). Cluster 16 has the strongest patient-disjoint reserve and `39.4%` Tier 1 rate, making it the next clinically distinct primary replication candidate.
- `region16_fact_only_replication_30_v1/region_anchor_manifest.csv` is frozen with deterministic seed `20260718`: 30 cluster-16 Tier 1 review candidates, including 14 patient-disjoint rows. This is not yet source-ledger verified and no generation may start until its ledger review clears the existing factual contract.

## July 17 cluster-16 recovery consistency check

- The first standard extraction audit reported all 30 cluster-16 follow-up sections as placeholders. This was an extraction-order artifact: a note can contain an early placeholder heading and a later substantive heading. It was not evidence that the Tier 1 screen was wrong.
- `recover_anchor_evidence.py`, `screen_region_source_eligibility.py`, and `build_source_fact_ledger.py` now select the strongest matching heading instead of the first one. The corrected restricted recovery audit finds substantive follow-up and instructions in all 30 frozen cluster-16 anchors, with a structured disposition candidate in all 30.
- Rebuild the cluster-16 ledger into a versioned recovery-aware folder and manually verify it before any generation. The recovery result establishes review eligibility only; it does not automatically validate any extracted fact.

## July 17 cluster-16 reviewed ledger: complete-note contract rejected; transition-note track viable

- The authoritative biomedical review (`docs/bariatric_fact_ledger_review_completed_RESTRICTED.csv`) covered 291 facts across 30 frozen cluster-16 cases: 90 verified, 155 corrected, 46 omitted, and zero pending. Only `ledger_024` and `ledger_027` retain all six core fields.
- All other 28 cases are missing only a reliable follow-up fact; the broad heading-recovery match was medication-history or unrelated text. Therefore the broad automated recovery result must not be interpreted as validated follow-up evidence, and the six-field complete-discharge track is blocked for this region.
- All 30 cases retain the five source-supported transition fields: principal diagnosis, hospital course, discharge medications, disposition, and instructions. The project may evaluate these as a separately named ``source-supported discharge-transition note'' task, with follow-up optional and omitted when unsupported. This is a task definition change, not a weakening of the complete-note claim.
- `validate_source_fact_ledger.py --optional_fields follow_up` confirms 30/30 ready for this new track, and `build_generation_ledger.py --optional_fields follow_up` serialized 245 prompt-safe facts across 30 cases. No transition-note generation has started yet.
- The fact-only launcher now supports `--document_type discharge_transition_note` without requiring a raw-ELM manifest when running only the fact-only arm. The prompt-safe cluster-16 ledger was rebuilt with 30 unique `anchor_id`s and frozen patient-disjoint provenance at `.../region16_fact_only_replication_30_v1/discharge_transition_note_v1/prompt_safe_ledger_with_provenance_v2/`.
- The first bounded transition-note pilot is complete: eight frozen anchors (four patient-disjoint), four sampled checkpoint-8215 fact-only candidates per anchor, 32/32 nonempty EOS-terminated outputs, and zero cap hits. BGE re-embedding and cluster-16 final-output selection gave 25/32 candidate landing (`78.1%`) and 7/8 selected landing (`87.5%`). An eight-note blinded review pack is ready; follow-up is marked optional and must be `not_applicable` only if absent from both ledger and note. No scale-up before this factual review clears.

## July 17 transition-note confirmation: prompt-only follow-up guard rejected

- First transition-note pilot blinded review: 6/8 pass, with two unsupported ``Follow-up: None'' claims and no critical omissions. A fresh eight-anchor confirmation was frozen rather than rerunning failed anchors.
- The phrase-heavy follow-up guard made performance worse: 3/8 pass, four unsupported follow-up claims, one critical disposition omission, and 4/8 selected cluster-16 landing. Do not scale either prompt condition and do not add stronger wording around prohibited phrases; it induced paraphrased unsupported assertions.
- New rule: factual output eligibility precedes geometry selection. When a ledger lacks follow-up, a deterministic output filter rejects any candidate that mentions follow-up; geometry ranks only eligible candidates. This retrospective filter is a diagnostic on the reviewed pilot, not confirmation evidence. A future independent pilot must use the minimal omission prompt plus this preselection filter.
- The filtered diagnostic rejected 7/32 guard-pilot candidates while retaining at least one eligible candidate for every anchor. It remains post hoc. A third independent pilot is frozen at `.../pilot_8x4_dual_gate_confirm_v3/`: eight unused anchors, three patient-disjoint, seed `20260720`, excluding both prior pilots. Predeclared readiness for a 30-anchor cluster-16 transition-note replication: at least 5/8 selected eligible outputs land in cluster 16, at least 7/8 blinded factual passes, zero unsupported follow-up claims, and zero critical omissions. Otherwise freeze cluster 16 as a bounded negative/partial result and do not scale it.

## July 17 cluster-16 prospective dual-gate confirmation: promising feasibility, no scale-up

- The preselected factual filter rejected 13/32 candidates for unsupported follow-up mentions. Seven of eight anchors retained an eligible candidate; final selection landed in cluster 16 for 5/8 frozen anchors. Blinded review of the seven selected notes gave 6/7 factual passes, zero unsupported-major-claim failures, and one critical omission (the ledger-supported home disposition was omitted).
- All five target-cluster-16 outputs passed review; one of two out-of-basin outputs passed. This is compatible with good clinical quality within the target region, but is too small for an association claim and does not make geometry a clinical-quality proxy.
- The strict readiness rule was not met: factual pass count is 6/7 rather than at least 7/8, one critical omission occurred, and one anchor had no eligible candidate. Freeze cluster 16 as a bounded prospective dual-gate feasibility/partial result; do not scale it to 30 anchors.
- For the five-field transition-note contract, the calibrated screen gives high-recall candidate reserves: cluster 11 `87` (32 patient-disjoint), cluster 16 `106` (59 patient-disjoint), cluster 25 `317` (46 patient-disjoint). Cluster 25 is the next evidence-based replication target, pending frozen-anchor ledger review; do not reuse cluster-16 pilot cases for another confirmation.

## July 21 cluster-25 replication: enforce patient-disjoint representation before ledger review

- The first deterministic cluster-25 transition-note sample (30 anchors, seed `20260721`) contained only four patient-disjoint anchors because proportional sampling reflected the full eligible pool. This is reproducible but is not the official replication cohort: the eligible cluster-25 reserve contains 317 rows, including 46 patient-disjoint rows.
- `build_region_anchor_manifest.py` now accepts `--min_patient_disjoint`. Freeze a new versioned cluster-25 manifest with at least eight patient-disjoint anchors before building the restricted source ledger. This is a pilot representation minimum, not a claim that the selected cohort reflects the natural patient-disjoint prevalence.
- The next sequence is restricted ledger construction, evidence-recovery audit, manual fact verification, prompt-safe ledger validation, four-candidate fact-only generation, final-output geometry selection, and blinded clinical-factual review. Do not generate from the four-patient-disjoint manifest.

## July 21 cluster-25 v2 restricted extraction: ready for manual fact review, not generation

- The official `region25_transition_note_replication_30_v2_pd8` manifest is frozen: 30 cluster-25 anchors, eight patient-disjoint, deterministic seed `20260721`.
- The restricted provisional ledger contains 287 facts. All 30 cases have hospital-course and instruction candidates; follow-up was placeholder-only in the standard extraction, while high-recall recovery found substantive follow-up candidates in only 7/30. Follow-up therefore remains optional and must be omitted unless manual review confirms a supported value.
- Disposition is present in 29/30 standard extractions and has a structured admissions candidate in 30/30, but the short source phrases are only candidates. Medication, instruction, diagnosis, and disposition values require manual verification before serialization into a prompt-safe ledger.
- This is a source-evidence eligibility result, not a clinical-quality or geometry result. The correct next action is manual restricted-ledger review, followed by structural validation with optional follow-up; do not launch generation yet.

## July 21 cluster-25 v2 reviewed-ledger result: bounded dual-gate pilot is ready

- Manual review of all 287 provisional facts retained 77 verified and 164 corrected values, omitting 46 unsafe or unsupported values. The validator confirms 21/30 cases retain all five required transition fields when follow-up is optional; nine cases are blocked for missing instructions (four), medication plus instruction/diagnosis support (four), or disposition (one).
- Exact string-span matching covers only 7/241 retained values because reviewers normalized concise generation values. This automated rate is a provenance limitation, not a replacement for the completed manual source-evidence review; retain both measures in reporting.
- `build_generation_ledger.py` now accepts `--case_readiness_path`, preventing blocked cases from entering a prompt-safe ledger. `build_generation_ledger_pilot_subset.py` now accepts `--min_patient_disjoint`.
- A prompt-safe 21-case ledger (179 facts) and frozen cluster-25 pilot are prepared at `.../region25_transition_note_replication_30_v2_pd8/discharge_transition_note_v1/pilot_8x4_dual_gate_v1/`: eight cases, four patient-disjoint, seed `20260722`, and 69 facts. Next action is four-candidate checkpoint-8215 fact-only generation, then final-output BGE selection with the absent-follow-up factual filter and blinded clinical-factual review. This remains a bounded second-region feasibility replication, not cohort-scale generation.

## July 18 cluster-25 pilot launch: shared PI filesystem full, rerun outputs moved to project storage

- The first cluster-25 generation launch stopped after eight outputs from two anchors with `OSError: [Errno 122] Disk quota exceeded`; the expected generated BGE `.npy` file is therefore absent and geometry selection correctly cannot start. The partial `17K` manifest is preserved as a failed-run diagnostic and must not be appended to or analyzed.
- The cause is shared infrastructure, not this project: `/gpfs/radev/pi/xu_hua` is `20T/20T` full, while `/gpfs/radev/project/xu_hua/mj756` has ample capacity. Keep MIMIC inputs read-only in the PI path, but write new cluster-25 generation, re-embedding, selection, review artifacts, and Slurm logs under `/gpfs/radev/project/xu_hua/mj756/synthnote/`.
- `source_grounded_fact_only.slurm` now writes Slurm stdout/stderr to `/gpfs/radev/project/xu_hua/mj756/synthnote/log/source_grounded_rescue/`. Rerun the frozen 8x4 pilot into a new `generation_project_v2` directory rather than appending to the partial PI-path output.

## July 18 storage triage: retire legacy pickle workflow without touching the active cohort

- Read-only audit found the active note/HADM pipeline is `data_note_hadm_all` (19G) plus `pickle_ds_note_hadm_all` (7.1G); both remain required for checkpoint-8215 provenance, leakage alignment, source-ledger construction, privacy screening, and the cluster-25 replication.
- The obsolete pre-note/HADM MIMIC-IV workflow is `pickle_ds` (121G, 94,460 legacy patient-object files), plus `data/clinic_notes` (4G), legacy embedding outputs (about 1.7G), and `output/core.csv` (513M). No active workspace, home/project code, or Slurm job references the exact legacy pickle path; the one old launcher points to a nonexistent different path. Retiring this workflow does not affect the current research plan, but removes the ability to rerun obsolete preliminary experiments without an archive.
- Added `preprocessing_mimiciv/transfer_legacy_pickle_to_scratch.slurm`: a four-worker, resumable copy-and-verify launcher that never deletes the source. It logs to scratch and has separate `copy` and checksum `verify` modes. Do not delete the PI source until copy verification is explicitly reviewed.
- The copy stage completed cleanly through four rsync workers: source and scratch destination each contain 94,460 files and report 121G. The original interactive allocation was released after the transfer. A four-worker checksum verification remains required; only an empty `checksum_differences.txt`, matching count/byte summary, and clean worker stderr files permit a later explicit source-removal decision.
- Scratch verification completed successfully: 94,460 files and 128,133,819,135 bytes matched exactly, checksum differences were zero, and all worker stderr files were empty. The explicitly approved legacy PI source `mimiciv/3.1/pickle_ds` was removed. `/gpfs/radev/pi/xu_hua` now reports 121G available; active `pickle_ds_note_hadm_all` remains present at 7.1G and is unchanged.

## July 19 cluster-25 clean candidate generation complete

- The quota-failed partial generation remains preserved at `.../pilot_8x4_dual_gate_v1/generation/` and is not used. The clean official rerun at `.../generation_clean_rerun_v2/` completed with checkpoint-8215: 32/32 unique fact-only candidates across eight frozen anchors (four per anchor), zero empty outputs, zero token-cap hits, and 32/32 EOS-terminated outputs.
- Its run summary records the intended `discharge_transition_note` task, 3072 maximum new tokens, sampled decoding, and four candidates per case. `run_source_grounded_rescue.py` now also records `document_type` per candidate for future runs; the current immutable manifest is linked to its run-level summary sidecar.
- Next step: BGE re-embed the 32 clean candidates, apply the prospective absent-follow-up factual filter and final-output cluster-25 geometry selection, then prepare a blinded factual review pack. No scale-up or cohort-level claim before that review.

## July 19 cluster-25 clean candidate re-embedding complete

- Re-embedding job `2135857` completed successfully. The clean cluster-25 manifest has 32 rows and its BGE matrix is finite, shape `(32, 1024)`, and exactly row-aligned. Metadata records BAAI/bge-large-en-v1.5 on CUDA, batch size 128, and the generation/re-embedding paths.
- Next action is CPU-side prospective selection: reject outputs mentioning follow-up when their ledger has no verified follow-up fact, then rank the remaining candidates within each frozen anchor by final BGE cluster-25 membership, basin margin, and target cosine. Build a blinded review pack from the selected eligible subset only.

## July 19 cluster-25 prospective selection complete: blinded review is the next gate

- The prospective absent-follow-up filter retained 21/32 candidates and rejected 11/32. Six of eight frozen anchors retained at least one eligible candidate; both patient-disjoint and patient-overlap subsets contribute 3/4 selected target-cluster outputs. Two anchors were correctly withheld: one patient-overlap anchor had no target candidate, and one patient-disjoint anchor had target candidates but all four were rejected for unsupported follow-up mentions.
- All six selected eligible outputs land in target cluster 25. Thus the honest geometry result is 6/8 frozen-anchor target coverage (75%), not 6/6. This supports output-space feasibility only; it is not a clinical-quality result and must not trigger scale-up.
- A six-row restricted blinded pack is ready at `.../pilot_8x4_dual_gate_v1/blinded_review_clean_rerun_v2/geometry_selected_fact_only_blinded_review.csv`, with a separate key. Next action: blinded ledger-grounded clinical-factual review using the established required-field rule, then post-review analysis and patient-disjoint reporting.

## July 19 cluster-25 prospective dual-gate review: geometry passed; disposition completeness blocks scale-up

- The clean cluster-25 pilot selected one eligible output for six of eight frozen anchors; all six landed in target cluster 25. The honest geometry endpoint is therefore 6/8 (75\%) frozen-anchor target coverage, with three of four patient-disjoint anchors represented in the selected set.
- Finalized blinded ledger-grounded review of the six selected notes gave 4/6 factual passes (66.7\%), zero unsupported-major claims, and two critical omissions. Both failures omitted the ledger-supported disposition (``Home'' or ``Home With Service''); no failure was attributable to target-basin landing, and the six rows are too small and selected to test a geometry--quality association.
- Candidate-level audit found a viable target-landing, follow-up-eligible alternative with a disposition heading for each failed anchor. The immediate limitation is therefore the selector's lack of prospective required-field structural eligibility, not evidence that the fact-only model cannot produce a disposition for those anchors.
- `select_fact_only_geometry_candidates.py` now supports a generic `--enforce_required_field_headings` gate. It rejects outputs missing headings for ledger-required principal diagnosis, hospital course, discharge medications, disposition, or instructions; it retains the separate absent-follow-up omission rule and records field-specific rejection counts. This structural gate does not establish factuality and cannot be applied retrospectively as new evidence.
- Freeze cluster 25 as a bounded second-region partial result. Before any new generation, audit yield under the new gate on the immutable pilot. A future independent confirmation must use the new predeclared gate and retain explicit patient-disjoint limitations; do not scale, use LLM editing, or make cohort-level enrichment claims yet.

## July 19 cluster-25 prospective completeness-screen diagnostic: confirmation design updated

- The immutable 32-candidate pilot was reprocessed only as a diagnostic with the new required-field-heading screen. It retained 16/32 structurally eligible candidates, six of eight anchors with an eligible candidate, and 6/8 selected target-cluster-25 outputs. The screen identified missing disposition headings in six candidates, while each previously failed selected anchor retained an alternative target-landing candidate with a disposition heading.
- This supports testing the structural screen prospectively; it does not alter the frozen 4/6 blinded result. The next cluster-25 cohort must be independently frozen from the broader high-recall eligibility reserve, excluding all 30 rows in `region25_transition_note_replication_30_v2_pd8` and enforcing a patient-disjoint minimum before manual ledger review.
- `build_region_anchor_manifest.py` now supports `--exclude_manifest_path`, records excluded IDs in its summary, and can therefore enforce non-overlap with any prior frozen cohort.
- The independent confirmation manifest is frozen at `.../region25_transition_note_confirmation_12_v3_pd4/`: 12 unseen cluster-25 anchors, four patient-disjoint, seed `20260724`, excluding all 30 anchors from the original cluster-25 cohort. It is ready only for restricted provisional-ledger construction and manual fact verification; no generation has started.

## July 21 cluster-25 independent confirmation: source ledger cleared, pilot frozen

- Manual source-evidence review covered 113 facts across 12 unseen anchors: 33 verified, 62 corrected, and 18 omitted. Placeholder-only follow-up was retained only when substantively supported (`ledger_010`, `ledger_012`); it remains optional in the transition-note task.
- The validator confirms 10/12 cases are ready for generation with all five required fields. `ledger_008` is blocked for unsupported disposition; `ledger_009` is blocked for missing discharge medications and instructions. Both were excluded rather than reconstructed.
- The prompt-safe ledger at `.../prompt_safe_ledger_ready10_v1/` contains 10 cases and 82 compact facts, no source-note spans, and the required transition fields with optional follow-up. The fixed prospective generation cohort at `.../discharge_transition_note_v1/pilot_10x4_required_fields_v1/` contains those ten cases, including three patient-disjoint cases, seed `20260724`.
- Next action: generate exactly four checkpoint-8215 fact-only candidates per frozen case, BGE re-embed, then select only candidates passing both unsupported-follow-up and required-field-heading screens before blinded review. Do not add blocked cases or alter this frozen cohort.

## July 21 cluster-25 independent confirmation: prospective geometry/completeness gate

- The fixed 10-case cohort generated 40 checkpoint-8215 fact-only candidates. After the predeclared absent-follow-up and required-field-heading gates, 22/40 candidates remained eligible and 7/10 anchors retained a selected output. Five selected outputs land in target cluster 25, two selected outputs are outside, and three anchors have no structurally eligible candidate. The honest target geometry yield is therefore 5/10 frozen anchors, below the predeclared scale-up threshold.
- A blinded seven-note review pack was created from all selected eligible outputs, not only target-landing notes: `.../pilot_10x4_required_fields_v1/blinded_review_required_fields_v1/geometry_selected_fact_only_blinded_review.csv`. The key is separate and must remain unopened until all labels are final. This review tests whether the prospective required-field screen improves factual completeness despite insufficient geometry yield; it cannot justify scaling by itself.

## July 21 cluster-25 independent confirmation: blinded clinical gate clears controlled scale-up

- Blinded review of all seven selected eligible outputs completed before unblinding: 6/7 passed (85.7\%), zero unsupported-major claims, and one critical omission. The failure (`ledger_003`) omitted a source-supported 12-day antibiotic plan and home drain-management services. Both patient-disjoint selected cases passed and both landed in cluster 25.
- Deterministic post-review analysis at `.../blinded_review_required_fields_v1/post_review_analysis/` reproduces the result. Target outputs passed 4/5 and non-target outputs passed 2/2; the sample is too small to infer a geometry--quality association.
- The required-field-heading screen prevents missing sections but does not guarantee within-section fact coverage. This remaining failure is an acceptance/yield issue, not a reason to treat omitted critical transition details as acceptable.
- The result clears only a controlled moderate-scale next step: increase the source-grounded eligible anchor pool and candidate count with the same predeclared gates, then use blinded stratified quality-control review. It does not justify full-cohort production, a claim of real-note equivalence, or automatic clinical deployment.

## July 21 cluster-25 moderate-scale cohort: source-evidence gate cleared

- A new independent 30-anchor cluster-25 cohort (`seed 20260725`) was frozen after excluding all 42 anchors used in the two prior cluster-25 cohorts. It contains eight patient-disjoint anchors.
- Manual review of 284 provisional facts retained 84 verified and 154 corrected values, omitting 46. The validator cleared 24/30 cases with all five required transition fields, including six patient-disjoint cases; six cases remain blocked rather than imputed.
- The prompt-safe ledger (`prompt_safe_ledger_ready24_v1`) contains 24 cases and 197 compact facts. The fixed moderate-scale generation cohort (`pilot_24x4_v1`) contains exactly those 24 cases, six patient-disjoint, with four sampled candidates per case (96 total). Next action is checkpoint-8215 fact-only generation, final-output BGE re-embedding, the same predeclared factual/structural eligibility and cluster-25 geometry selection, then blinded stratified review. This remains moderate-scale validation, not cohort-wide production.

## July 21 cluster-25 moderate-scale generation complete: re-embedding and selection pending

- Checkpoint-8215 generated 96/96 unique, non-empty fact-only candidates across 24 frozen anchors. Ninety-four outputs ended with EOS; two reached the 3072-token cap and must not be eligible for final selection.
- `select_fact_only_geometry_candidates.py` now supports `--reject_cap_hits`, records `hit_max_new_tokens` as a specific rejection reason, and includes its count in the selection summary. Re-embed all 96 candidates first, then apply cap, unsupported-follow-up, and required-field-heading screens before final cluster-25 selection.

## July 21 cluster-25 moderate-scale selection: blinded 20-note quality gate pending

- Of 96 re-embedded candidates, 62 passed the cap, absent-follow-up, and required-field-heading screens. Twenty of 24 frozen anchors retained an eligible selected output; 12 selected outputs land in cluster 25. The honest final target yield is 12/24 frozen anchors (50\%), while 16/24 had at least one target-landing candidate before factual/structural selection.
- Heading failures are dominated by disposition (13 candidates), followed by instructions (7); two cap-hit candidates were deterministically rejected. These are acceptance/yield accounting results, not silent exclusions.
- A blinded 20-note pack contains every selected eligible output: 12 target-cluster and 8 non-target, with four frozen anchors unselected. The review key remains separate. Next action is blinded ledger-grounded review of all 20 notes; no further generation or threshold change before that gate.

## July 21 cluster-25 moderate-scale clinical gate: passed; second-region replication begins

- Deterministic post-review analysis of all 20 selected outputs confirms 19/20 passes (95\%), zero unsupported-major claims, one critical omission, mean factual faithfulness 4.55/5, and mean clinical consistency 4.45/5. All 12 target-cluster-25 outputs passed; the target/non-target quality difference is not interpretable at this sample size. All five patient-disjoint selected outputs passed.
- The one failure was a non-target output that omitted two ledger-supported discharge medications (nadolol and furosemide). This preserves the key limitation: final-output screening yields a high-quality subset but is not a perfect medication-reconciliation guarantee.
- Cluster 25 therefore clears moderate-scale Phase 2b validation in one region: 24 frozen anchors, 12/24 final target outputs, and 19/20 selected-output clinical passes. It still does not establish multi-region generalization, full-cohort production, clinical deployment, or a causal relation between geometry and quality.
- A second-region cluster-11 cohort is frozen at `.../region11_transition_note_replication_19_v1_pd8/`: all 19 available Tier-1 candidates, including eight patient-disjoint cases, seed `20260726`. Next action is restricted ledger construction and manual source-evidence verification under the unchanged five-field transition-note contract before any cluster-11 generation.

## July 21 cluster-11 second-region replication: source-evidence gate cleared

- Manual review of 188 provisional facts retained 62 verified and 96 corrected values, omitting 30. Seventeen of 19 cluster-11 cases retain all five required transition fields, including seven patient-disjoint cases. The two blocked cases lack source-supported disposition (`ledger_003`) or instructions (`ledger_010`) and remain excluded.
- The cluster-11 prompt-safe ledger contains 17 cases and 144 compact facts. The fixed independent replication cohort is `.../region11_transition_note_replication_19_v1_pd8/discharge_transition_note_v1/pilot_17x4_v1/`: 17 anchors, seven patient-disjoint, four sampled candidates per anchor (68 total), seed `20260726`.
- Next action: checkpoint-8215 fact-only generation, BGE re-embedding, and the unchanged cap/follow-up/required-heading selection protocol targeting cluster 11. This tests whether the cluster-25 moderate-scale result generalizes across a distinct sparse real-note region.

## July 21 cluster-11 second-region generation and geometry selection: clinical gate pending

- The 17-anchor cohort generated 68 unique non-empty candidates; three cap-hit candidates are rejected. Under the unchanged cap/follow-up/required-heading protocol, 40/68 candidates are eligible and 15/17 anchors retain a selected output.
- Eleven of the 15 selected outputs land in cluster 11, for an honest target geometry yield of 11/17 frozen anchors (64.7\%). Thirteen anchors had at least one target-landing candidate before selection. Disposition-heading failures are the largest structural rejection source (17 candidates).
- A blinded 15-note review pack has been created with every selected eligible output: 11 target-cluster and four non-target outputs; two frozen anchors are unselected. Next action is blinded ledger-grounded review of all 15 notes before unblinding or any cross-region scale-up claim.

## July 21 cluster-11 blinded review: geometry generalizes, raw clinical quality does not

- Deterministic unblinded analysis of all 15 selected cluster-11 outputs confirms 12/15 factual passes (80\%), two unsupported-major-claim failures, and one critical medication-reconciliation omission. Mean factual faithfulness was 4.27/5 and mean clinical consistency 4.20/5; four of five patient-disjoint selected notes passed.
- The two unsupported failures added unverified or internally contradictory discharge medications. The omission failure dropped three ledger-supported discharge medications. These are medication-content failures occurring despite required medication headings, so section-presence screening is insufficient.
- Cross-region conclusion: the output-space geometry workflow generalized from cluster 25 to cluster 11 (11/17 final target outputs), but the raw fact-only generation quality did not generalize uniformly. Cluster 25 remains a positive one-region moderate-scale result; the combined evidence does not support cross-region or cohort-wide raw generation claims.
- Do not scale raw selected notes further. Next technical block is a source-ledger-to-output medication-reconciliation diagnostic on the completed cluster-25 and cluster-11 reviews. It must be evaluated as a detector against existing blinded human labels before becoming a prospective rejection gate. Only if deterministic coverage is inadequate should the project move to an approved local LLM-as-judge; an LLM editor must not be used to conceal these errors.

## July 21 medication-reconciliation diagnostic: deterministic lexical gate rejected

- Added `source_grounded_rescue/audit_medication_reconciliation.py`, a conservative lexical source-ledger-to-output diagnostic. It audits the discharge-medication section against verified ledger medication terms and evaluates flags against completed human labels; it is explicitly not an automatic clinical verifier.
- On the 35 completed selected-output reviews from cluster 25 and cluster 11, the lexical detector flagged 10 outputs but detected only 1/2 manual medication-addition failures (sensitivity 50\%) while flagging 27.3\% of manual medication passes. It also caught the cluster-25 medication-omission case through missing terms. The detector is therefore unsuitable as a prospective rejection gate.
- Next Phase 3a step: evaluate an approved local ledger-grounded LLM-as-judge, using cluster-25 reviews as development data and the completed cluster-11 review as held-out evaluation. The judge must produce field-level support, medication omission, extra-medication, contradiction, and final accept/reject labels. It must demonstrate detection of all held-out critical/unsupported failures with a tolerable false-rejection burden before any prospective use. Do not implement an editor until this judge is validated.

New package boundary:

- `open-elm/cav_axis/clinical_validation/`: secure manual-label ingestion and deterministic triage.
- `open-elm/cav_axis/source_grounded_rescue/`: source-fact ledger schema, builder, validator, and local-only correction prompt.
- No local editor model is configured yet; do not send source notes or ledgers to a third-party API.

## July 21 Phase 3a refinement: judge feasibility and calibration before any editor

- Mentor review agrees with the Phase 3a pivot but correctly tightens the claim: the current cluster-25 development set (20 selected notes, one failure) and locked cluster-11 evaluation set (15 selected notes, three severe medication failures) are adequate for a smoke-test/method-selection study, not for validation of an automatic clinical gate. Even detecting all three held-out failures would have wide uncertainty.
- Phase 3a is therefore defined as a local, ledger-grounded **medication-reconciliation judge feasibility and calibration study**. It is evidence-first: compact verified medication facts plus the generated note must yield explicit alignment, missing-medication, unsupported-addition, and contradiction evidence before a deterministic rejection recommendation. Invalid JSON or material uncertainty routes to review, never automatic pass.
- Cluster 25 is development-only for prompt/model/schema selection. Cluster 11 remains locked for a single frozen held-out evaluation. The next prospective test must be a fresh third region, with human review of selected outputs, judge-rejected outputs, and a random sample of judge-passing nonselected candidates. An editor remains prohibited until the judge shows prospective value.
- Added `open-elm/cav_axis/clinical_validation/medication_judge_schema.json`, `build_medication_judge_dataset.py`, `run_local_medication_judge.py`, `validate_judge_json.py`, and `analyze_medication_judge.py`. They are local-only and model-agnostic; `run_local_medication_judge.py` requires an approved independent local instruction-following model and will not download or call an external API.
- Built restricted task sets under `$DATAHOME/clinical_validation/medication_judge_v1/`: 20 labelled cluster-25 development tasks and 15 unlabelled locked cluster-11 evaluation tasks. These contain compact verified ledger text and synthetic notes and must remain on approved project storage.
- A blinded 40-candidate cluster-25 medication-review pack was also created at `$DATAHOME/clinical_validation/medication_judge_v1/development_candidate_review_pack/`. It excludes all 20 already selected/reviewed moderate-scale outputs and samples deterministically from the remaining 76 existing candidates. Human medication labels are required before any prompt/model calibration.

## July 22 candidate-level medication calibration review complete

- Blinded review of all 40 additional cluster-25 candidates is complete and unblinded. Two severe medication errors were identified: one lactulose omission and one action contradiction that instructed continuation of a ledger-stopped hepatitis-C treatment. The remaining 38 candidates had no medication error under the focused taxonomy.
- Both errors were target-basin candidates but were already structurally ineligible. This is compatible with the current deterministic eligibility screen reducing risk, but does not validate it as a medication-safety gate: the sample is deliberately stratified, repeated anchors are present, and the exploratory eligible/ineligible comparison is underpowered.
- The 40 new cases exclude the 20 previous selected cluster-25 outputs. They have been combined with those 20 into a 60-task restricted cluster-25 development set at `$DATAHOME/clinical_validation/medication_judge_v1/development_combined60/`. It is for prompt/schema/model calibration only. Cluster 11 remains frozen as the 15-note held-out evaluation set.
- Next decision point: provision an approved independent local instruction-following judge, calibrate only on the cluster-25 development data, freeze all settings, then run the cluster-11 set once with three deterministic repeats. A judge result is feasibility evidence only; a fresh third-region prospective study is required before prospective automatic rejection.
