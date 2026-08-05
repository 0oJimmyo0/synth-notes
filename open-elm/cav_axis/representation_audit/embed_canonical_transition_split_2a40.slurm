#!/bin/bash
#SBATCH --job-name=canonical_bge_2a40
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/canonical_bge_2a40_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/canonical_bge_2a40_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=2
# Two workers x two CPUs stays within the four-CPU per-user QOS limit.
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:2
#SBATCH --partition=gpu_devel

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/canonical_bge_2a40_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/canonical_bge_2a40_${SLURM_JOB_ID}.err" >&2)

: "${MANIFEST_PATH:?Set MANIFEST_PATH to a canonical transition manifest.}"
: "${SPEC_PATH:?Set SPEC_PATH to the frozen canonical representation spec.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for merged embeddings and metadata.}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

SHARD_ROOT="${OUTPUT_DIR}/shards"
mkdir -p "${SHARD_ROOT}"
echo "Starting two-A40 canonical BGE embedding at $(date)"

srun --ntasks=2 --ntasks-per-node=2 --gres=gpu:1 bash -lc '
  python "'"${SCRIPT_DIR}"'"/representation_audit/embed_canonical_transition_split.py \
    --manifest_path "'"${MANIFEST_PATH}"'" \
    --spec_path "'"${SPEC_PATH}"'" \
    --output_dir "'"${SHARD_ROOT}"'"/shard_$(printf "%02d" "${SLURM_PROCID}") \
    --device cuda \
    --batch_size "'"${EMBEDDING_BATCH_SIZE:-128}"'" \
    --document_batch_size "'"${DOCUMENT_BATCH_SIZE:-32}"'" \
    --chunk_tokens "'"${CHUNK_TOKENS:-448}"'" \
    --chunk_overlap "'"${CHUNK_OVERLAP:-64}"'" \
    --shard_count 2 \
    --shard_index "${SLURM_PROCID}"
'

EXPECTED_ROWS=$(wc -l < "${MANIFEST_PATH}")
python "${SCRIPT_DIR}/representation_audit/merge_canonical_embedding_shards.py" \
  --shard_root "${SHARD_ROOT}" \
  --n_shards 2 \
  --expected_rows "${EXPECTED_ROWS}" \
  --output_dir "${OUTPUT_DIR}"
echo "Finished two-A40 canonical BGE embedding at $(date)"
