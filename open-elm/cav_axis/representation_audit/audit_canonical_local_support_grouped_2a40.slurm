#!/bin/bash
#SBATCH --job-name=canonical_support
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/canonical_support_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/canonical_support_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu_devel

set -euo pipefail
SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis"
: "${TRAIN_EMBEDDINGS_PATH:?Set TRAIN_EMBEDDINGS_PATH.}"
: "${TRAIN_SUBJECT_METADATA_PATH:?Set TRAIN_SUBJECT_METADATA_PATH.}"
: "${DEV_EMBEDDINGS_PATH:?Set DEV_EMBEDDINGS_PATH.}"
: "${DEV_SUBJECT_METADATA_PATH:?Set DEV_SUBJECT_METADATA_PATH.}"
: "${REFERENCE_SPLIT_PATH:?Set REFERENCE_SPLIT_PATH.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR.}"
: "${SPLIT_SEED:?Set SPLIT_SEED.}"
QUERY_SPLIT=${QUERY_SPLIT:-dev}
if [[ "$QUERY_SPLIT" != "dev" && "$QUERY_SPLIT" != "test" ]]; then
  echo "ERROR: QUERY_SPLIT must be dev or test." >&2
  exit 1
fi

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

srun --ntasks=2 --ntasks-per-node=2 --gres=gpu:1 bash -lc '
  python "'"${SCRIPT_DIR}"'"/representation_audit/audit_canonical_local_support_grouped.py \
    --train_embeddings_path "'"${TRAIN_EMBEDDINGS_PATH}"'" \
    --train_subject_metadata_path "'"${TRAIN_SUBJECT_METADATA_PATH}"'" \
    --dev_embeddings_path "'"${DEV_EMBEDDINGS_PATH}"'" \
    --dev_subject_metadata_path "'"${DEV_SUBJECT_METADATA_PATH}"'" \
    --reference_split_path "'"${REFERENCE_SPLIT_PATH}"'" \
    --output_dir "'"${OUTPUT_DIR}"'" --split_seed "'"${SPLIT_SEED}"'" \
    --query_split "'"${QUERY_SPLIT}"'" \
    --k_grid "'"${K_GRID:-10,25,50,100}"'" \
    --query_batch_size "'"${QUERY_BATCH_SIZE:-128}"'" \
    --reference_batch_size "'"${REFERENCE_BATCH_SIZE:-8192}"'" \
    --device cuda --shard_count 2 --shard_index "${SLURM_PROCID}"
'

python "${SCRIPT_DIR}/representation_audit/audit_canonical_local_support_grouped.py" \
  --train_embeddings_path "${TRAIN_EMBEDDINGS_PATH}" \
  --train_subject_metadata_path "${TRAIN_SUBJECT_METADATA_PATH}" \
  --dev_embeddings_path "${DEV_EMBEDDINGS_PATH}" \
  --dev_subject_metadata_path "${DEV_SUBJECT_METADATA_PATH}" \
  --reference_split_path "${REFERENCE_SPLIT_PATH}" \
  --output_dir "${OUTPUT_DIR}" --split_seed "${SPLIT_SEED}" \
  --query_split "${QUERY_SPLIT}" \
  --k_grid "${K_GRID:-10,25,50,100}" --shard_count 2 --merge_only
