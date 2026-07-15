#!/bin/bash
#SBATCH --job-name=closed_loop_privacy
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/closed_loop_privacy/closed_loop_privacy_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/closed_loop_privacy/closed_loop_privacy_%j.err
#SBATCH --partition=devel
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

set -euo pipefail

OPEN_ELM_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm"
SCRIPT_PATH="${OPEN_ELM_DIR}/cav_axis/closed_loop_train_text_privacy_screen.py"
LOG_DIR="${OPEN_ELM_DIR}/log/closed_loop_privacy"
mkdir -p "${LOG_DIR}"
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO} with exit code ${status}" >&2; exit ${status}' ERR

required_vars=(MANIFEST_PATH SPLIT_MANIFEST_PATH DATASET_PATH TRAIN_DATASET_PATH OUTPUT_DIR)
for var_name in "${required_vars[@]}"; do
    if [ -z "${!var_name:-}" ]; then
        echo "ERROR: Missing required environment variable: ${var_name}" >&2
        exit 1
    fi
done
for path_var in MANIFEST_PATH SPLIT_MANIFEST_PATH DATASET_PATH TRAIN_DATASET_PATH; do
    if [ ! -e "${!path_var}" ]; then
        echo "ERROR: Required path does not exist: ${path_var}=${!path_var}" >&2
        exit 1
    fi
done

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "${OUTPUT_DIR}"
cd "${OPEN_ELM_DIR}"

echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start Time: $(date)"
echo "Manifest: ${MANIFEST_PATH}"
echo "Train dataset: ${TRAIN_DATASET_PATH}"
echo "Max train texts: ${MAX_TRAIN_TEXTS:-0}"
echo "Semantic shortlist: ${SEMANTIC_TOP_K:-20}"
echo "=========================================="

python -u "${SCRIPT_PATH}" \
  --manifest_path "${MANIFEST_PATH}" \
  --split_manifest_path "${SPLIT_MANIFEST_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --train_dataset_path "${TRAIN_DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_train_texts "${MAX_TRAIN_TEXTS:-0}" \
  --top_k_lexical_checks "${TOP_K_LEXICAL_CHECKS:-0}" \
  --semantic_top_k "${SEMANTIC_TOP_K:-20}" \
  --embedding_model_name "${EMBEDDING_MODEL_NAME:-BAAI/bge-large-en-v1.5}" \
  --embedding_device "${EMBEDDING_DEVICE:-cpu}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE:-128}"

echo "Completed successfully at: $(date)"
