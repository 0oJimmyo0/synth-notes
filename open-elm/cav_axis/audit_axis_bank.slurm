#!/bin/bash
#SBATCH --job-name=elm_audit_axis_bank
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
exec > >(tee "${LOG_DIR}/audit_axis_bank_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/audit_axis_bank_${SLURM_JOB_ID}.err" >&2)

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

BANK_DIR="${BANK_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_outputs/axis_bank_clusters_11_20_25}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings.npy}"
METADATA_PATH="${METADATA_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings_metadata.csv}"
FACTORS_PATH="${FACTORS_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/cav_factor_table_clusters_11_20_25.csv}"
JOIN_COLS="${JOIN_COLS:-subject_id,hadm_id,note_id}"
ALPHAS="${ALPHAS:-0.5,1,2}"
AXIS_INDICES="${AXIS_INDICES:-}"
TOP_TARGETS="${TOP_TARGETS:-5}"
OUTPUT_NAME="${OUTPUT_NAME:-steering_audit.csv}"

CMD=(
  python "${SCRIPT_DIR}/audit_axis_bank.py"
  --bank_dir "${BANK_DIR}"
  --embeddings_path "${EMBEDDINGS_PATH}"
  --metadata_path "${METADATA_PATH}"
  --factors_path "${FACTORS_PATH}"
  --join_cols "${JOIN_COLS}"
  "--alphas=${ALPHAS}"
  --top_targets "${TOP_TARGETS}"
  --output_name "${OUTPUT_NAME}"
)

if [ -n "${AXIS_INDICES}" ]; then
  CMD+=(--axis_indices "${AXIS_INDICES}")
fi

printf 'Running command: '
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
