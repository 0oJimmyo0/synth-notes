#!/usr/bin/env bash
set -euo pipefail

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

python /gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/audit_vanilla_generation.py \
  --manifest_path /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/synthetic_notes/synthetic_notes_test_vanilla_seed42_manifest.jsonl \
  --dataset_path /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/encoded_testing_filtered \
  --split_manifest_path /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level.csv \
  --output_dir /gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/generation_audit/vanilla_test_seed42 \
  --embedding_model_name BAAI/bge-large-en-v1.5 \
  --sample_size_for_manual_review 50
