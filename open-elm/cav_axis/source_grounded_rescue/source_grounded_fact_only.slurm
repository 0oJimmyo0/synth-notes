#!/bin/bash
#SBATCH --job-name=source_fact_only
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/source_grounded_rescue/fact_only_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/source_grounded_rescue/fact_only_%j.err
#SBATCH --partition=gpu_devel
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

OPEN_ELM_DIR=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm
SCRIPT_PATH=$OPEN_ELM_DIR/cav_axis/source_grounded_rescue/run_source_grounded_rescue.py
DATAHOME=${DATAHOME:-/gpfs/radev/pi/xu_hua/shared/datasets/synthnote/mimiciv/3.1/data_note_hadm_all/clinic_notes/1_task}
RUN=${RUN:-$DATAHOME/closed_loop_output_enrichment/cluster29_basin_v1_test8_2h100_strict_quality_max8192_default_rerun}
BACKBONE_PATH=${BACKBONE_PATH:-$OPEN_ELM_DIR/initial_elm_model}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$DATAHOME/elm_training_outputs/filtered_training/checkpoint-8215}
: "${GENERATION_LEDGER_PATH:?Set GENERATION_LEDGER_PATH to a prompt-safe JSONL ledger}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${MODEL_CONDITION:?Set MODEL_CONDITION to checkpoint_8215 or untouched_backbone}"
RAW_ELM_MANIFEST_PATH=${RAW_ELM_MANIFEST_PATH:-}
DOCUMENT_TYPE=${DOCUMENT_TYPE:-complete_discharge_summary}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}
OUTPUT_STEM=${OUTPUT_STEM:-source_grounded_fact_only}
N_CANDIDATES_PER_CASE=${N_CANDIDATES_PER_CASE:-1}
SEED=${SEED:-42}
DO_SAMPLE=${DO_SAMPLE:-0}
TEMPERATURE=${TEMPERATURE:-0.8}
TOP_P=${TOP_P:-0.9}

if [[ "$MODEL_CONDITION" != "checkpoint_8215" && "$MODEL_CONDITION" != "untouched_backbone" ]]; then
  echo "ERROR: MODEL_CONDITION must be checkpoint_8215 or untouched_backbone" >&2
  exit 1
fi
for required in "$SCRIPT_PATH" "$BACKBONE_PATH" "$GENERATION_LEDGER_PATH"; do
  [[ -e "$required" ]] || { echo "ERROR: required path does not exist: $required" >&2; exit 1; }
done
if [[ "${ARMS:-fact_only}" == *correction* ]]; then
  [[ -n "$RAW_ELM_MANIFEST_PATH" && -e "$RAW_ELM_MANIFEST_PATH" ]] || { echo "ERROR: correction arm requires RAW_ELM_MANIFEST_PATH" >&2; exit 1; }
fi
if [[ "$MODEL_CONDITION" == "checkpoint_8215" ]]; then
  [[ -d "$CHECKPOINT_PATH" ]] || { echo "ERROR: checkpoint directory does not exist: $CHECKPOINT_PATH" >&2; exit 1; }
fi

mkdir -p "$OUTPUT_DIR" "$OPEN_ELM_DIR/log/source_grounded_rescue"
module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID"
echo "Model condition: $MODEL_CONDITION"
echo "Ledger: $GENERATION_LEDGER_PATH"
echo "Output: $OUTPUT_DIR"
echo "Candidates per case: $N_CANDIDATES_PER_CASE; sample: $DO_SAMPLE; seed: $SEED"
nvidia-smi
cd "$OPEN_ELM_DIR"

cmd=(python -u "$SCRIPT_PATH"
  --generation_ledger_path "$GENERATION_LEDGER_PATH"
  --backbone_path "$BACKBONE_PATH"
  --model_condition "$MODEL_CONDITION"
  --arms "${ARMS:-fact_only}"
  --document_type "$DOCUMENT_TYPE"
  --output_dir "$OUTPUT_DIR"
  --output_stem "$OUTPUT_STEM"
  --device cuda
  --max_new_tokens "$MAX_NEW_TOKENS"
  --n_candidates_per_case "$N_CANDIDATES_PER_CASE"
  --seed "$SEED"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P")
if [[ "$DO_SAMPLE" == "1" ]]; then
  cmd+=(--do_sample)
fi
if [[ -n "$RAW_ELM_MANIFEST_PATH" ]]; then
  cmd+=(--raw_elm_manifest_path "$RAW_ELM_MANIFEST_PATH")
fi
if [[ "$MODEL_CONDITION" == "checkpoint_8215" ]]; then
  cmd+=(--checkpoint_path "$CHECKPOINT_PATH")
fi
"${cmd[@]}"
