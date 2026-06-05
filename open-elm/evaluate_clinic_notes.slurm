#!/bin/bash
#SBATCH --job-name=elm_semantic_verify
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:l40s:1
#SBATCH --partition=gpu

# Semantic consistency verification: compare input embeddings with re-encoded
# generated text (cosine similarity) to ensure the ELM works as expected.

if [ -n "$SLURM_SUBMIT_DIR" ]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/evaluate_clinic_notes_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/evaluate_clinic_notes_${SLURM_JOB_ID}.err" >&2)

cd "${SCRIPT_DIR}"

echo "=========================================="
echo "Semantic Consistency Verification (ELM)"
echo "Job ID: $SLURM_JOB_ID | $(date)"
echo "=========================================="

BASE_DIR="/gpfs/radev/scratch/xu_hua/shared/data/synthnote/mimiciv/3.1"
DATAHOME="${BASE_DIR}/data/clinic_notes/1_task"
CHECKPOINT_PATH="${BASE_DIR}/data/clinic_notes/1_task/elm_training_outputs/filtered_training"
BACKBONE_MODEL="${SCRIPT_DIR}/initial_elm_model"
TEST_DATA="${DATAHOME}/encoded_testing_filtered"

# Use latest checkpoint if CHECKPOINT_PATH is the training dir
if [ -d "${CHECKPOINT_PATH}" ] && [ ! -f "${CHECKPOINT_PATH}/config.json" ]; then
    LATEST=$(ls -d "${CHECKPOINT_PATH}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "${LATEST}" ]; then
        CHECKPOINT_PATH="${LATEST}"
        echo "Using checkpoint: ${CHECKPOINT_PATH}"
    fi
fi

if [ ! -d "${CHECKPOINT_PATH}" ]; then
    echo "ERROR: Checkpoint not found at ${CHECKPOINT_PATH}"
    exit 1
fi
if [ ! -d "${BACKBONE_MODEL}" ]; then
    echo "ERROR: Backbone model not found at ${BACKBONE_MODEL}"
    exit 1
fi
if [ ! -d "${TEST_DATA}" ]; then
    echo "ERROR: Test dataset not found at ${TEST_DATA}"
    exit 1
fi

module load miniconda
CONDA_BASE=$(conda info --base 2>/dev/null)
[ -n "${CONDA_BASE}" ] && source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm || true

python "${SCRIPT_DIR}/evaluate_clinic_notes.py" \
    --backbone_model_path "${BACKBONE_MODEL}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --test_data_path "${TEST_DATA}" \
    --embedding_model_path "BAAI/bge-large-en-v1.5" \
    --batch_size 8 \
    --max_samples 5000

echo ""
echo "Done. Logs: ${LOG_DIR}/evaluate_clinic_notes_${SLURM_JOB_ID}.out"
