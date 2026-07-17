#!/bin/bash
#SBATCH --job-name=source_rescue_smoke
#SBATCH --output=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/source_grounded_rescue/rescue_smoke_%j.out
#SBATCH --error=/gpfs/radev/pi/xu_hua/shared/synthnote/physionet.org/files/embedding_elm/open-elm/log/source_grounded_rescue/rescue_smoke_%j.err
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
CONTROL_DIR=${CONTROL_DIR:-$RUN/matched_vanilla_8192_control}
BACKBONE_PATH=${BACKBONE_PATH:-$OPEN_ELM_DIR/initial_elm_model}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-$DATAHOME/elm_training_outputs/filtered_training/checkpoint-8215}
GENERATION_LEDGER_PATH=${GENERATION_LEDGER_PATH:-$CONTROL_DIR/source_grounded_rescue/generation_ledger_smoke_v1/prompt_safe_ledger/generation_ledgers.jsonl}
RAW_ELM_MANIFEST_PATH=${RAW_ELM_MANIFEST_PATH:-$RUN/closed_loop_accepted_manifest.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$CONTROL_DIR/source_grounded_rescue/four_case_text_smoke_v1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-3072}

mkdir -p "$OUTPUT_DIR" "$OPEN_ELM_DIR/log/source_grounded_rescue"
for required in "$SCRIPT_PATH" "$BACKBONE_PATH" "$CHECKPOINT_PATH" "$GENERATION_LEDGER_PATH" "$RAW_ELM_MANIFEST_PATH"; do
  if [ ! -e "$required" ]; then
    echo "ERROR: required path does not exist: $required" >&2
    exit 1
  fi
done

module load miniconda
CONDA_BASE=$(conda info --base)
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate elm
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Job ID: $SLURM_JOB_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Generation ledger: $GENERATION_LEDGER_PATH"
nvidia-smi
cd "$OPEN_ELM_DIR"

python -u "$SCRIPT_PATH" \
  --generation_ledger_path "$GENERATION_LEDGER_PATH" \
  --raw_elm_manifest_path "$RAW_ELM_MANIFEST_PATH" \
  --backbone_path "$BACKBONE_PATH" \
  --model_condition untouched_backbone \
  --output_dir "$OUTPUT_DIR" \
  --output_stem source_grounded_rescue_smoke \
  --device cuda \
  --max_new_tokens "$MAX_NEW_TOKENS"

python -u "$SCRIPT_PATH" \
  --generation_ledger_path "$GENERATION_LEDGER_PATH" \
  --raw_elm_manifest_path "$RAW_ELM_MANIFEST_PATH" \
  --backbone_path "$BACKBONE_PATH" \
  --model_condition checkpoint_8215 \
  --checkpoint_path "$CHECKPOINT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --output_stem source_grounded_rescue_smoke \
  --device cuda \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --append
