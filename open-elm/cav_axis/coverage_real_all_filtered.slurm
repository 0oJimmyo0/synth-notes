#!/bin/bash
#SBATCH --job-name=elm_cov_all_real
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis"
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/coverage_real_all_filtered_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/coverage_real_all_filtered_${SLURM_JOB_ID}.err" >&2)

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

TRAIN_DATASET_PATH="${TRAIN_DATASET_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/encoded_training_filtered}"
DEV_DATASET_PATH="${DEV_DATASET_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/encoded_dev_filtered}"
TEST_DATASET_PATH="${TEST_DATASET_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/encoded_testing_filtered}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/coverage/real_all_filtered_precompute}"
EXTRA_METADATA_PATH="${EXTRA_METADATA_PATH:-}"
N_CLUSTERS="${N_CLUSTERS:-50}"
RANDOM_SEED="${RANDOM_SEED:-42}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Coverage Configuration"
echo "=========================================="
echo "Train dataset: ${TRAIN_DATASET_PATH}"
echo "Dev dataset: ${DEV_DATASET_PATH}"
echo "Test dataset: ${TEST_DATASET_PATH}"
echo "Split manifest: ${SPLIT_MANIFEST_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Extra metadata: ${EXTRA_METADATA_PATH:-<none>}"
echo "n_clusters: ${N_CLUSTERS}"
echo "random_seed: ${RANDOM_SEED}"
echo "=========================================="
echo ""

CMD=(
  python "${SCRIPT_DIR}/prepare_coverage_mapping.py"
  --mode real_all_filtered_precompute
  --train_dataset_path "${TRAIN_DATASET_PATH}"
  --dev_dataset_path "${DEV_DATASET_PATH}"
  --real_dataset_path "${TEST_DATASET_PATH}"
  --split_manifest_path "${SPLIT_MANIFEST_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --n_clusters "${N_CLUSTERS}"
  --random_seed "${RANDOM_SEED}"
)

if [ -n "${EXTRA_METADATA_PATH}" ]; then
  CMD+=(--extra_metadata_path "${EXTRA_METADATA_PATH}")
fi

"${CMD[@]}"

echo ""
echo "=========================================="
echo "Coverage run finished at $(date)"
echo "=========================================="
