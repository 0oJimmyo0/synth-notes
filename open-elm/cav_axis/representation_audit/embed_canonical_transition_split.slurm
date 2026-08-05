#!/bin/bash
#SBATCH --job-name=canonical_bge
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/canonical_bge_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/logs/canonical_bge_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:a40:1
#SBATCH --partition=gpu_devel

set -euo pipefail

SCRIPT_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"
exec > >(tee "${LOG_DIR}/canonical_bge_${SLURM_JOB_ID}.out")
exec 2> >(tee "${LOG_DIR}/canonical_bge_${SLURM_JOB_ID}.err" >&2)

: "${MANIFEST_PATH:?Set MANIFEST_PATH to a canonical transition manifest.}"
: "${SPEC_PATH:?Set SPEC_PATH to the frozen canonical representation spec.}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR for embeddings and text-free metadata.}"

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

echo "Starting canonical BGE embedding at $(date)"
python "${SCRIPT_DIR}/representation_audit/embed_canonical_transition_split.py" \
  --manifest_path "${MANIFEST_PATH}" \
  --spec_path "${SPEC_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --device "${EMBEDDING_DEVICE:-cuda}" \
  --batch_size "${EMBEDDING_BATCH_SIZE:-128}" \
  --document_batch_size "${DOCUMENT_BATCH_SIZE:-32}" \
  --chunk_tokens "${CHUNK_TOKENS:-448}" \
  --chunk_overlap "${CHUNK_OVERLAP:-64}"
echo "Finished canonical BGE embedding at $(date)"
