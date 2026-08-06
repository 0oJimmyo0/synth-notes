#!/bin/bash
#SBATCH --job-name=generated_local_support
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/generated_local_support_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/generated_local_support_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu_devel

set -euo pipefail

OPEN_ELM_DIR=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm
SCRIPT=$OPEN_ELM_DIR/cav_axis/representation_audit/score_generated_canonical_local_support.py

: "${TRAIN_EMBEDDINGS_PATH:?Set TRAIN_EMBEDDINGS_PATH}"
: "${TRAIN_SUBJECT_METADATA_PATH:?Set TRAIN_SUBJECT_METADATA_PATH}"
: "${CANDIDATE_EMBEDDINGS_PATH:?Set CANDIDATE_EMBEDDINGS_PATH}"
: "${CANDIDATE_EMBEDDING_METADATA_PATH:?Set CANDIDATE_EMBEDDING_METADATA_PATH}"
: "${CANDIDATE_SUBJECT_MANIFEST_PATH:?Set CANDIDATE_SUBJECT_MANIFEST_PATH}"
: "${REFERENCE_SPLIT_DIR:?Set REFERENCE_SPLIT_DIR containing frozen split CSVs}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT}"
SEEDS=${SEEDS:-20260811,20260812,20260813,20260814,20260815}
K=${K:-50}

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm
export TOKENIZERS_PARALLELISM=false

IFS=',' read -r -a seed_array <<< "$SEEDS"
for seed in "${seed_array[@]}"; do
  python "$SCRIPT" \
    --train_embeddings_path "$TRAIN_EMBEDDINGS_PATH" \
    --train_subject_metadata_path "$TRAIN_SUBJECT_METADATA_PATH" \
    --candidate_embeddings_path "$CANDIDATE_EMBEDDINGS_PATH" \
    --candidate_embedding_metadata_path "$CANDIDATE_EMBEDDING_METADATA_PATH" \
    --candidate_subject_manifest_path "$CANDIDATE_SUBJECT_MANIFEST_PATH" \
    --reference_split_path "$REFERENCE_SPLIT_DIR/train_reference_split_seed_${seed}.csv" \
    --split_seed "$seed" \
    --k "$K" \
    --output_dir "$OUTPUT_ROOT/seed_${seed}" \
    --device cuda
done
