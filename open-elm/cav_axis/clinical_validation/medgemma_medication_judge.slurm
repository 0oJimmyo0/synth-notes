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
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
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
PROMPT_PATH=${PROMPT_PATH:-$REPO_ROOT/open-elm/cav_axis/clinical_validation/medication_judge_prompt_v2_compact.txt}
SCHEMA_PATH=${SCHEMA_PATH:-$REPO_ROOT/open-elm/cav_axis/clinical_validation/medication_judge_compact_schema_v2.json}
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
