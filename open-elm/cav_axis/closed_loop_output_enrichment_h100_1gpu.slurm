#!/bin/bash
#SBATCH --job-name=closed_loop_enrich_1g
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/closed_loop_output_enrichment/closed_loop_output_enrichment_1g_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/closed_loop_output_enrichment/closed_loop_output_enrichment_1g_%j.err
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:h100:1
#SBATCH --partition=gpu

set -euo pipefail

OPEN_ELM_DIR="/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm"
SCRIPT_PATH="${OPEN_ELM_DIR}/cav_axis/closed_loop_output_enrichment.py"
LOG_DIR="${OPEN_ELM_DIR}/log/closed_loop_output_enrichment"
mkdir -p "${LOG_DIR}"
trap 'status=$?; echo "[ERROR] ${BASH_SOURCE[0]} failed at line ${LINENO} with exit code ${status}" >&2; exit ${status}' ERR

required_vars=(
  ANCHOR_MANIFEST_PATH
  DATASET_PATH
  CHECKPOINT_PATH
  BACKBONE_PATH
  CLUSTER_ASSIGNMENTS_PATH
  TARGET_CLUSTER_IDS
  OUTPUT_DIR
)
for var_name in "${required_vars[@]}"; do
  if [ -z "${!var_name:-}" ]; then
    echo "ERROR: Missing required environment variable: ${var_name}"
    exit 1
  fi
done

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm

cd "${OPEN_ELM_DIR}"

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="
echo ""

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_PATH}" \
  --anchor_manifest_path "${ANCHOR_MANIFEST_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --backbone_path "${BACKBONE_PATH}" \
  --cluster_assignments_path "${CLUSTER_ASSIGNMENTS_PATH}" \
  --target_cluster_ids "${TARGET_CLUSTER_IDS}" \
  --output_dir "${OUTPUT_DIR}" \
  --embedding_model_name "${EMBEDDING_MODEL_NAME:-BAAI/bge-large-en-v1.5}" \
  --split_manifest_path "${SPLIT_MANIFEST_PATH:-}" \
  --source_split "${SOURCE_SPLIT:-test}" \
  --n_candidates_per_anchor "${N_CANDIDATES_PER_ANCHOR:-8}" \
  --accepted_per_anchor "${ACCEPTED_PER_ANCHOR:-1}" \
  --seeds "${SEEDS:-42,43,44,45,46,47,48,49}" \
  --temperature_values "${TEMPERATURE_VALUES:-0.8}" \
  --top_p_values "${TOP_P_VALUES:-0.9}" \
  --top_k_values "${TOP_K_VALUES:-50}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --max_new_tokens "${MAX_NEW_TOKENS:-2048}" \
  --min_new_tokens "${MIN_NEW_TOKENS:-0}" \
  --repetition_penalty "${REPETITION_PENALTY:-1.2}" \
  --clinic_note_prompt_mode "${CLINIC_NOTE_PROMPT_MODE:-default}" \
  --device "${DEVICE:-cuda}" \
  --embedding_device "${EMBEDDING_DEVICE:-cuda}" \
  --embedding_batch_size "${EMBEDDING_BATCH_SIZE:-256}" \
  --min_word_count "${MIN_WORD_COUNT:-100}" \
  --min_source_cosine "${MIN_SOURCE_COSINE:-0.78}" \
  --target_centroid_threshold_quantile "${TARGET_CENTROID_THRESHOLD_QUANTILE:-0.90}" \
  --candidate_near_duplicate_cosine "${CANDIDATE_NEAR_DUPLICATE_COSINE:-0.995}" \
  --anchor_diversity_cosine "${ANCHOR_DIVERSITY_COSINE:-0.992}" \
  --manual_review_sample_size "${MANUAL_REVIEW_SAMPLE_SIZE:-50}" \
  --max_train_texts_for_overlap "${MAX_TRAIN_TEXTS_FOR_OVERLAP:-50000}" \
  --selection_strategy "${SELECTION_STRATEGY:-best_target_score}" \
  --shard_id 0 \
  --num_shards 1 \
  ${TARGET_BASIN_PATH:+--target_basin_path "${TARGET_BASIN_PATH}"} \
  ${TARGET_CENTROID_PATH:+--target_centroid_path "${TARGET_CENTROID_PATH}"} \
  ${TRAIN_TEXT_PATH:+--train_text_path "${TRAIN_TEXT_PATH}"} \
  ${PICKLE_DIR:+--pickle_dir "${PICKLE_DIR}"} \
  ${JOIN_COLS:+--join_cols "${JOIN_COLS}"}

echo ""
echo "=========================================="
echo "Job Completed Successfully"
echo "End Time: $(date)"
echo "=========================================="
