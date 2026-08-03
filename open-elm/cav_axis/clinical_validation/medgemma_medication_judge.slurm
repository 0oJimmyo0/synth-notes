#!/bin/bash
#SBATCH --job-name=medgemma_med_judge
#SBATCH --partition=gpu_devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/clinical_validation/logs/medgemma_med_judge_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/cav_axis/clinical_validation/logs/medgemma_med_judge_%j.err

set -euo pipefail
umask 007

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
# Batch jobs inherit the submit shell's Conda variables. Do not unset them:
# doing so leaves Conda unable to deactivate the inherited environment.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  conda deactivate || true
fi
conda activate medgemma_judge

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

REPO_ROOT=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm
MODEL_PATH=${MODEL_PATH:?Set the local MedGemma snapshot path.}
TASK_PATH=${TASK_PATH:?Set the restricted task JSONL path on approved project storage.}
OUTPUT_PATH=${OUTPUT_PATH:?Set an output JSONL path on approved project storage.}
# v3.3 explicitly separates active discharge therapy, historical inpatient
# treatment, and evidence components that are genuinely unspecified.
PROMPT_PATH=${PROMPT_PATH:-$REPO_ROOT/open-elm/cav_axis/clinical_validation/medication_judge_prompt_v3_3_discharge_state.txt}
SCHEMA_PATH=${SCHEMA_PATH:-$REPO_ROOT/open-elm/cav_axis/clinical_validation/medication_judge_compact_schema_v3.json}
REPEATS=${REPEATS:-3}
# The compact contract is bounded to 12 findings. Use 1536 tokens to leave
# headroom for evidence citations; cap hits remain review-only outcomes.
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1536}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.95}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_GENERATION_SECONDS=${MAX_GENERATION_SECONDS:-0}
PROGRESS_EVERY=${PROGRESS_EVERY:-10}
MEDICATION_EVIDENCE_FIELDS=${MEDICATION_EVIDENCE_FIELDS:-discharge_medications,instructions}

nvidia-smi
# The runner uses device_map=auto, which shards MedGemma-27B BF16 across the
# two allocated A40s. Require two 44 GB-or-larger GPUs before loading weights.
mapfile -t GPU_MEMORY_MIBS < <(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')
GPU_COUNT=${#GPU_MEMORY_MIBS[@]}
GPU_TOTAL_MIB=0
for memory_mib in "${GPU_MEMORY_MIBS[@]}"; do
  if [[ ! "$memory_mib" =~ ^[0-9]+$ ]]; then
    echo "ERROR: Could not read GPU memory capacity." >&2
    exit 1
  fi
  GPU_TOTAL_MIB=$((GPU_TOTAL_MIB + memory_mib))
done
if (( GPU_COUNT < 2 )) || (( GPU_MEMORY_MIBS[0] < 44000 )) || (( GPU_MEMORY_MIBS[1] < 44000 )) || (( GPU_TOTAL_MIB < 88000 )); then
  echo "ERROR: MedGemma-27B BF16 requires two 44 GB-or-larger GPUs; detected ${GPU_COUNT} GPUs totaling ${GPU_TOTAL_MIB} MiB." >&2
  exit 1
fi
python "$REPO_ROOT/open-elm/cav_axis/clinical_validation/run_local_medication_judge.py" \
  --task_path "$TASK_PATH" \
  --model_path "$MODEL_PATH" \
  --output_path "$OUTPUT_PATH" \
  --prompt_path "$PROMPT_PATH" \
  --schema_path "$SCHEMA_PATH" \
  --repeats "$REPEATS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --use_chat_template \
  --system_prompt "You are an evidence-first medication reconciliation reviewer. Return only the requested JSON object. Do not include analysis, markdown, or text outside the JSON object." \
  --bf16 \
  --do_sample \
  --temperature "$TEMPERATURE" \
  --top_p "$TOP_P" \
  --batch_size "$BATCH_SIZE" \
  --max_generation_seconds "$MAX_GENERATION_SECONDS" \
  --progress_every "$PROGRESS_EVERY" \
  --medication_evidence_fields "$MEDICATION_EVIDENCE_FIELDS" \
  --medication_only_evidence
