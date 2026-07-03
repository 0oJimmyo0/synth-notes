#!/bin/bash
#SBATCH --job-name=elm_fit_axis_bank
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
exec > >(tee "${LOG_DIR}/fit_axis_bank_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/fit_axis_bank_${SLURM_JOB_ID}.err" >&2)

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings.npy}"
METADATA_PATH="${METADATA_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/embeddings-BAAI-bge-large-en-v1.5/sentence_embeddings_metadata.csv}"
FACTORS_PATH="${FACTORS_PATH:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_inputs/cav_factor_table_clusters_11_20_25.csv}"
JOIN_COLS="${JOIN_COLS:-subject_id,hadm_id,note_id}"
FACTOR_COLS="${FACTOR_COLS:-age_bin,sex_gender,race_ethnicity,insurance,admission_type,service,los_bin,icu_flag,cluster_target_11,cluster_target_20,cluster_target_25}"
CONTINUOUS_FACTORS="${CONTINUOUS_FACTORS:-}"
CATEGORICAL_FACTORS="${CATEGORICAL_FACTORS:-age_bin,sex_gender,race_ethnicity,insurance,admission_type,service,los_bin,icu_flag,cluster_target_11,cluster_target_20,cluster_target_25}"
GROUP_COL="${GROUP_COL:-subject_id}"
OUTPUT_DIR="${OUTPUT_DIR:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task/cav_axis_outputs/axis_bank_clusters_11_20_25}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python "${SCRIPT_DIR}/fit_axis_bank.py"
  --embeddings_path "${EMBEDDINGS_PATH}"
  --metadata_path "${METADATA_PATH}"
  --factors_path "${FACTORS_PATH}"
  --join_cols "${JOIN_COLS}"
  --factor_cols "${FACTOR_COLS}"
  --continuous_factors "${CONTINUOUS_FACTORS}"
  --categorical_factors "${CATEGORICAL_FACTORS}"
  --group_col "${GROUP_COL}"
  --output_dir "${OUTPUT_DIR}"
)

printf 'Running command: '
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
