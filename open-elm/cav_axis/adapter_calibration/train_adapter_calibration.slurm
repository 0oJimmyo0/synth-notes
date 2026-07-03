#!/bin/bash
#SBATCH --job-name=adapter_calib
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/adapter_calibration/adapter_calibration_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/adapter_calibration/adapter_calibration_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu_devel

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/adapter_calibration"
OPEN_ELM_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm"
LOG_DIR="${OPEN_ELM_DIR}/log/adapter_calibration"
mkdir -p "${LOG_DIR}"
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO} with exit code ${status}" >&2; exit ${status}' ERR

cd "${OPEN_ELM_DIR}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

BASE_DIR="${BASE_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1}"
DATAHOME="${DATAHOME:-${BASE_DIR}/data_note_hadm_all/clinic_notes/1_task}"
DATAHOME="${DATAHOME%/}"

CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DATAHOME}/elm_training_outputs/filtered_training/checkpoint-8215}"
BACKBONE_MODEL_PATH="${BACKBONE_MODEL_PATH:-${OPEN_ELM_DIR}/initial_elm_model}"
TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-${DATAHOME}/cav_shifted_datasets/adapter_calibration/train_cluster29_adapter_calibration}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-${DATAHOME}/cav_shifted_datasets/adapter_calibration/dev_cluster29_adapter_calibration}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATAHOME}/elm_training_outputs/adapter_calibration/cluster29_pilot}"

BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
MAX_STEPS="${MAX_STEPS:-1500}"
SAVE_STEPS="${SAVE_STEPS:-250}"
EVAL_STEPS="${EVAL_STEPS:-250}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
ANCHOR_LOSS_WEIGHT="${ANCHOR_LOSS_WEIGHT:-0.05}"
SEED="${SEED:-42}"
VALIDATION_PROBE_SIZE="${VALIDATION_PROBE_SIZE:-64}"
VALIDATION_PROBE_BATCH_SIZE="${VALIDATION_PROBE_BATCH_SIZE:-4}"
VALIDATION_PROBE_MAX_NEW_TOKENS="${VALIDATION_PROBE_MAX_NEW_TOKENS:-1024}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-BAAI/bge-large-en-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"

for required_dir in "${CHECKPOINT_PATH}" "${BACKBONE_MODEL_PATH}" "${TRAIN_DATASET_PATH}" "${EVAL_DATASET_PATH}"; do
    if [ ! -d "${required_dir}" ]; then
        echo "ERROR: Required directory not found: ${required_dir}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_DIR}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export DS_SKIP_CUDA_CHECK=1
export ACCELERATE_USE_DEEPSPEED=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    IFS=',' read -r -a GPU_LIST <<< "${SLURM_JOB_GPUS}"
    GPUS_PER_NODE="${#GPU_LIST[@]}"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    GPUS_PER_NODE="${#GPU_LIST[@]}"
else
    GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29501}"

echo "GPU Information:"
nvidia-smi
echo ""

echo "=========================================="
echo "Adapter Calibration Configuration"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Backbone: ${BACKBONE_MODEL_PATH}"
echo "Train dataset: ${TRAIN_DATASET_PATH}"
echo "Eval dataset: ${EVAL_DATASET_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accum: ${GRAD_ACCUM}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Max steps: ${MAX_STEPS}"
echo "Anchor weight: ${ANCHOR_LOSS_WEIGHT}"
echo "Seed: ${SEED}"
echo "GPUs per node: ${GPUS_PER_NODE}"
echo "Master addr: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"
echo "=========================================="
echo ""

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${GPUS_PER_NODE}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/train_adapter_calibration.py" \
    --train_dataset_path "${TRAIN_DATASET_PATH}" \
    --eval_dataset_path "${EVAL_DATASET_PATH}" \
    --checkpoint_path "${CHECKPOINT_PATH}" \
    --backbone_model_path "${BACKBONE_MODEL_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate "${LEARNING_RATE}" \
    --max_steps "${MAX_STEPS}" \
    --save_steps "${SAVE_STEPS}" \
    --eval_steps "${EVAL_STEPS}" \
    --logging_steps "${LOGGING_STEPS}" \
    --max_seq_length "${MAX_SEQ_LENGTH}" \
    --anchor_loss_weight "${ANCHOR_LOSS_WEIGHT}" \
    --seed "${SEED}" \
    --validation_probe_size "${VALIDATION_PROBE_SIZE}" \
    --validation_probe_batch_size "${VALIDATION_PROBE_BATCH_SIZE}" \
    --validation_probe_max_new_tokens "${VALIDATION_PROBE_MAX_NEW_TOKENS}" \
    --embedding_model_name "${EMBEDDING_MODEL_NAME}" \
    --embedding_device "${EMBEDDING_DEVICE}"

echo ""
echo "=========================================="
echo "Job Completed Successfully"
echo "End Time: $(date)"
echo "=========================================="
