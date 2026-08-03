#!/bin/bash
#SBATCH --job-name=hybrid_geometry
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/source_grounded_rescue/logs/hybrid_geometry_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/source_grounded_rescue/logs/hybrid_geometry_%j.err
#SBATCH --partition=devel
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00

# Reconstructing frozen centroids reads all train/dev/test embeddings, so run it
# off the login node and retain the full log used to establish final geometry.
set -euo pipefail

OPEN_ELM_DIR=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm
DATAHOME=${DATAHOME:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task}
SCRIPT_PATH=$OPEN_ELM_DIR/cav_axis/source_grounded_rescue/select_fact_only_geometry_candidates.py

: "${CANDIDATE_MANIFEST_PATH:?Set CANDIDATE_MANIFEST_PATH}"
: "${CANDIDATE_EMBEDDINGS_PATH:?Set CANDIDATE_EMBEDDINGS_PATH}"
: "${GENERATION_LEDGER_PATH:?Set GENERATION_LEDGER_PATH}"
: "${CONTRACT_NOTE_COVERAGE_CSV:?Set CONTRACT_NOTE_COVERAGE_CSV}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

REAL_DATASET_PATH=${REAL_DATASET_PATH:-$DATAHOME/encoded_testing_filtered}
REAL_CLUSTER_ASSIGNMENTS_PATH=${REAL_CLUSTER_ASSIGNMENTS_PATH:-$DATAHOME/coverage/real_all_filtered_precompute_with_subgroups/real_all_filtered_cluster_assignments.csv}
TRAIN_DATASET_PATH=${TRAIN_DATASET_PATH:-$DATAHOME/encoded_training_filtered}
DEV_DATASET_PATH=${DEV_DATASET_PATH:-$DATAHOME/encoded_dev_filtered}
TEST_DATASET_PATH=${TEST_DATASET_PATH:-$DATAHOME/encoded_testing_filtered}
TARGET_CLUSTER_IDS=${TARGET_CLUSTER_IDS:-36}
SOURCE_SPLIT=${SOURCE_SPLIT:-test}

mkdir -p "$OUTPUT_DIR" "$OPEN_ELM_DIR/cav_axis/source_grounded_rescue/logs"
module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm
cd "$OPEN_ELM_DIR"

python -u "$SCRIPT_PATH" \
  --candidate_manifest_path "$CANDIDATE_MANIFEST_PATH" \
  --candidate_embeddings_path "$CANDIDATE_EMBEDDINGS_PATH" \
  --real_dataset_path "$REAL_DATASET_PATH" \
  --real_cluster_assignments_path "$REAL_CLUSTER_ASSIGNMENTS_PATH" \
  --frozen_centroid_dataset_path "train=$TRAIN_DATASET_PATH" \
  --frozen_centroid_dataset_path "dev=$DEV_DATASET_PATH" \
  --frozen_centroid_dataset_path "test=$TEST_DATASET_PATH" \
  --source_split "$SOURCE_SPLIT" \
  --target_cluster_ids "$TARGET_CLUSTER_IDS" \
  --generation_ledger_path "$GENERATION_LEDGER_PATH" \
  --contract_note_coverage_csv "$CONTRACT_NOTE_COVERAGE_CSV" \
  --enforce_absent_followup_heading \
  --enforce_required_field_headings \
  --reject_cap_hits \
  --reject_course_constraint_failures \
  --output_dir "$OUTPUT_DIR"
