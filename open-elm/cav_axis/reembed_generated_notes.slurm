#!/bin/bash
#SBATCH --job-name=elm_reembed_syn
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/reembed_generated_notes_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/reembed_generated_notes_%j.err
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu_devel

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis"
LOG_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/reembed_generated_notes_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/reembed_generated_notes_${SLURM_JOB_ID}.err" >&2)

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
OUTPUT_DIR="${OUTPUT_DIR:-}"
if [ -n "${OUTPUT_DIR}" ]; then
  OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/generated_note_embeddings_bge_large.npy}"
  METADATA_OUTPUT_PATH="${METADATA_OUTPUT_PATH:-${OUTPUT_DIR}/generated_note_embeddings_bge_large.npy.meta.json}"
else
  OUTPUT_PATH="${OUTPUT_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/generation_audit/vanilla_test_seed42/generated_note_embeddings_bge_large.npy}"
  METADATA_OUTPUT_PATH="${METADATA_OUTPUT_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/generation_audit/vanilla_test_seed42/generated_note_embeddings_bge_large.npy.meta.json}"
fi
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-BAAI/bge-large-en-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-256}"
TEXT_COLUMN="${TEXT_COLUMN:-generated_text}"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "=========================================="
echo "Re-embedding Configuration"
echo "=========================================="
echo "Manifest path: ${MANIFEST_PATH}"
echo "Output path: ${OUTPUT_PATH}"
echo "Metadata output path: ${METADATA_OUTPUT_PATH}"
echo "Embedding model: ${EMBEDDING_MODEL_NAME}"
echo "Embedding device: ${EMBEDDING_DEVICE}"
echo "Embedding batch size: ${EMBEDDING_BATCH_SIZE}"
echo "Text column: ${TEXT_COLUMN}"
echo "=========================================="
echo ""

python "${SCRIPT_DIR}/reembed_generated_notes.py" \
  --manifest_path "${MANIFEST_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --metadata_output_path "${METADATA_OUTPUT_PATH}" \
  --embedding_model_name "${EMBEDDING_MODEL_NAME}" \
  --embedding_device "${EMBEDDING_DEVICE}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE}" \
  --text_column "${TEXT_COLUMN}"

echo ""
echo "=========================================="
echo "Re-embedding finished at $(date)"
echo "=========================================="
