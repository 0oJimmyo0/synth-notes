#!/bin/bash
#SBATCH --job-name=elm_cov_real_vs_syn
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu_devel

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis"
LOG_DIR="${SCRIPT_DIR}/log"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/coverage_real_vs_synthetic_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/coverage_real_vs_synthetic_${SLURM_JOB_ID}.err" >&2)

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

REAL_DATASET_PATH="${REAL_DATASET_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/encoded_testing_filtered}"
SPLIT_MANIFEST_PATH="${SPLIT_MANIFEST_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/leakage_audit/split_manifest_note_level.csv}"
SYNTHETIC_MANIFEST_PATH="${SYNTHETIC_MANIFEST_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/synthetic_notes/synthetic_notes_test_vanilla_seed42_manifest.jsonl}"
SYNTHETIC_EMBEDDINGS_PATH="${SYNTHETIC_EMBEDDINGS_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/generation_audit/vanilla_test_seed42/generated_note_embeddings_bge_large.npy}"
AUDIT_JSON_PATH="${AUDIT_JSON_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/generation_audit/vanilla_test_seed42/generation_audit_baseline.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/coverage/real_vs_synthetic_vanilla_seed42}"
EXTRA_METADATA_PATH="${EXTRA_METADATA_PATH:-}"
N_CLUSTERS="${N_CLUSTERS:-50}"
RANDOM_SEED="${RANDOM_SEED:-42}"
PRDC_K="${PRDC_K:-5}"
LOW_DENSITY_QUANTILE="${LOW_DENSITY_QUANTILE:-0.2}"
DISTRIBUTION_SAMPLE_SIZE="${DISTRIBUTION_SAMPLE_SIZE:-2000}"
MAX_POINTS_FOR_PLOT="${MAX_POINTS_FOR_PLOT:-3000}"

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Coverage Configuration"
echo "=========================================="
echo "Real dataset: ${REAL_DATASET_PATH}"
echo "Split manifest: ${SPLIT_MANIFEST_PATH}"
echo "Synthetic manifest: ${SYNTHETIC_MANIFEST_PATH}"
echo "Synthetic embeddings: ${SYNTHETIC_EMBEDDINGS_PATH}"
echo "Audit JSON: ${AUDIT_JSON_PATH}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Extra metadata: ${EXTRA_METADATA_PATH:-<none>}"
echo "n_clusters: ${N_CLUSTERS}"
echo "random_seed: ${RANDOM_SEED}"
echo "prdc_k: ${PRDC_K}"
echo "low_density_quantile: ${LOW_DENSITY_QUANTILE}"
echo "distribution_sample_size: ${DISTRIBUTION_SAMPLE_SIZE}"
echo "max_points_for_plot: ${MAX_POINTS_FOR_PLOT}"
echo "=========================================="
echo ""

CMD=(
  python "${SCRIPT_DIR}/prepare_coverage_mapping.py"
  --mode real_vs_synthetic
  --real_dataset_path "${REAL_DATASET_PATH}"
  --split_manifest_path "${SPLIT_MANIFEST_PATH}"
  --synthetic_manifest_path "${SYNTHETIC_MANIFEST_PATH}"
  --synthetic_embeddings_path "${SYNTHETIC_EMBEDDINGS_PATH}"
  --audit_json_path "${AUDIT_JSON_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --n_clusters "${N_CLUSTERS}"
  --random_seed "${RANDOM_SEED}"
  --prdc_k "${PRDC_K}"
  --low_density_quantile "${LOW_DENSITY_QUANTILE}"
  --distribution_sample_size "${DISTRIBUTION_SAMPLE_SIZE}"
  --max_points_for_plot "${MAX_POINTS_FOR_PLOT}"
)

if [ -n "${EXTRA_METADATA_PATH}" ]; then
  CMD+=(--extra_metadata_path "${EXTRA_METADATA_PATH}")
fi

"${CMD[@]}"

echo ""
echo "=========================================="
echo "Coverage run finished at $(date)"
echo "=========================================="
