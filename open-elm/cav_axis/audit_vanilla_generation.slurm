#!/bin/bash
#SBATCH --job-name=elm_audit_vanilla
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu_devel

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis"
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/audit_vanilla_generation_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/audit_vanilla_generation_${SLURM_JOB_ID}.err" >&2)

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date)"
echo "Working Directory: ${SCRIPT_DIR}"
echo "=========================================="
echo ""

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

cd "${SCRIPT_DIR}"

MANIFEST_PATH="${MANIFEST_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/synthetic_notes/synthetic_notes_test_vanilla_seed42_manifest.jsonl}"
DATASET_PATH="${DATASET_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/encoded_testing_filtered}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/generation_audit/vanilla_test_seed42}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-BAAI/bge-large-en-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-256}"
MANUAL_REVIEW_SAMPLE_SIZE="${MANUAL_REVIEW_SAMPLE_SIZE:-50}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Audit Configuration"
echo "=========================================="
echo "Manifest path: ${MANIFEST_PATH}"
echo "Dataset path: ${DATASET_PATH}"
echo "Split manifest path: ${SPLIT_MANIFEST_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Embedding model: ${EMBEDDING_MODEL_NAME}"
echo "Embedding device: ${EMBEDDING_DEVICE}"
echo "Embedding batch size: ${EMBEDDING_BATCH_SIZE}"
echo "Manual review sample size: ${MANUAL_REVIEW_SAMPLE_SIZE}"
echo "=========================================="
echo ""

python "${SCRIPT_DIR}/audit_vanilla_generation.py" \
  --manifest_path "${MANIFEST_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --split_manifest_path "${SPLIT_MANIFEST_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --embedding_model_name "${EMBEDDING_MODEL_NAME}" \
  --embedding_device "${EMBEDDING_DEVICE}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --sample_size_for_manual_review "${MANUAL_REVIEW_SAMPLE_SIZE}"

echo ""
echo "=========================================="
echo "Audit finished at $(date)"
echo "=========================================="
