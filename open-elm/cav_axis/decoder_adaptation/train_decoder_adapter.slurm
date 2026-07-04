#!/bin/bash
#SBATCH --job-name=decoder_adapt
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/decoder_adaptation/decoder_adaptation_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/decoder_adaptation/decoder_adaptation_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu_devel

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/decoder_adaptation"
OPEN_ELM_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm"
LOG_DIR="${OPEN_ELM_DIR}/log/decoder_adaptation"
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

ADAPT_DATAHOME="${ADAPT_DATAHOME:-${DATAHOME}/decoder_adaptation/cluster29_basin_balanced_v1/}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DATAHOME}/elm_training_outputs/filtered_training/checkpoint-8215}"
OUTPUT_DIR="${OUTPUT_DIR:-${DATAHOME}/elm_training_outputs/decoder_adaptation/cluster29_basin_balanced_v1_adapter}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
EVAL_STEPS="${EVAL_STEPS:-100}"
SAVE_STEPS="${SAVE_STEPS:-0.25}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [ ! -d "${ADAPT_DATAHOME}/encoded_training" ] && [ -d "${ADAPT_DATAHOME}/encoded_train" ]; then
    ln -sfn "${ADAPT_DATAHOME}/encoded_train" "${ADAPT_DATAHOME}/encoded_training"
fi

for required_dir in \
    "${CHECKPOINT_PATH}" \
    "${ADAPT_DATAHOME}/encoded_training" \
    "${ADAPT_DATAHOME}/encoded_dev"; do
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

echo "GPU Information:"
nvidia-smi
echo ""

echo "=========================================="
echo "Decoder Adaptation Configuration"
echo "=========================================="
echo "Adapt datahome: ${ADAPT_DATAHOME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Grad accum: ${GRAD_ACCUM}"
echo "Learning rate: ${LEARNING_RATE}"
echo "Num train epochs: ${NUM_TRAIN_EPOCHS}"
echo "Eval steps: ${EVAL_STEPS}"
echo "Save steps: ${SAVE_STEPS}"
echo "Max seq length: ${MAX_SEQ_LENGTH}"
echo "Resume from checkpoint: ${RESUME_FROM_CHECKPOINT:-<none>}"
echo "=========================================="
echo ""

if [ -n "${SLURM_JOB_GPUS:-}" ]; then
    IFS=',' read -r -a GPU_LIST <<< "${SLURM_JOB_GPUS}"
    GPUS_PER_NODE="${#GPU_LIST[@]}"
elif [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
    GPUS_PER_NODE="${#GPU_LIST[@]}"
else
    GPUS_PER_NODE=1
fi

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29511}"

echo "Detected GPUs per node: ${GPUS_PER_NODE}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "MASTER_PORT: ${MASTER_PORT}"
echo ""

CMD=(
  torchrun
  --standalone
  --nnodes=1
  --nproc_per_node="${GPUS_PER_NODE}"
  --master_addr="${MASTER_ADDR}"
  --master_port="${MASTER_PORT}"
  "${OPEN_ELM_DIR}/p1_train.py"
  --datahome "${ADAPT_DATAHOME}/"
  --output_dir "${OUTPUT_DIR}"
  --batch_size "${BATCH_SIZE}"
  --gradient_accumulation_steps "${GRAD_ACCUM}"
  --learning_rate "${LEARNING_RATE}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --eval_steps "${EVAL_STEPS}"
  --save_steps "${SAVE_STEPS}"
  --checkpoint_path "${CHECKPOINT_PATH}"
  --max_seq_length "${MAX_SEQ_LENGTH}"
  --seed 42
)

if [ -n "${RESUME_FROM_CHECKPOINT}" ]; then
  CMD+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
fi

"${CMD[@]}"

echo ""
echo "=========================================="
echo "Job Completed Successfully"
echo "End Time: $(date)"
echo "=========================================="
