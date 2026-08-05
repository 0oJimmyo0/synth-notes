#!/bin/bash
#SBATCH --job-name=canonical_density
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/canonical_density_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/canonical_density_%j.err
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
: "${DEV_EMBEDDINGS_PATH:?Set DEV_EMBEDDINGS_PATH.}"
: "${DEV_METADATA_PATH:?Set DEV_METADATA_PATH.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR.}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

srun --ntasks=2 --ntasks-per-node=2 --gres=gpu:1 bash -lc '
  python "'"${SCRIPT_DIR}"'"/representation_audit/audit_canonical_local_density.py \
    --train_embeddings_path "'"${TRAIN_EMBEDDINGS_PATH}"'" \
    --dev_embeddings_path "'"${DEV_EMBEDDINGS_PATH}"'" \
    --dev_metadata_path "'"${DEV_METADATA_PATH}"'" \
    --output_dir "'"${OUTPUT_DIR}"'" \
    --k_grid "'"${K_GRID:-10,25,50}"'" \
    --query_batch_size "'"${QUERY_BATCH_SIZE:-128}"'" \
    --reference_batch_size "'"${REFERENCE_BATCH_SIZE:-8192}"'" \
    --device cuda --shard_count 2 --shard_index "${SLURM_PROCID}"
'

python "${SCRIPT_DIR}/representation_audit/audit_canonical_local_density.py" \
  --train_embeddings_path "${TRAIN_EMBEDDINGS_PATH}" \
  --dev_embeddings_path "${DEV_EMBEDDINGS_PATH}" \
  --dev_metadata_path "${DEV_METADATA_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --k_grid "${K_GRID:-10,25,50}" \
  --shard_count 2 --merge_only
